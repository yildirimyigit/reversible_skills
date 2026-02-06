#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete


def _as_f32(x):
    return np.asarray(x, dtype=np.float32)


def get_top_level_models(pyrep):
    """
    First-generation objects under 'world', filtered to models.
    This typically includes robot, task props, cameras (if modeled), etc.
    """
    roots = pyrep.get_objects_in_tree(root_object=None, first_generation_only=True)
    models = [o for o in roots if o.is_model()]
    models.sort(key=lambda o: o.get_name())
    return models


def encode_model_trees_to_blob(model_tree_list):
    """
    model_tree_list: List[(name: str, tree_bytes: bytes)]
    Returns:
      names: (N,) <U
      blob_u8: (sum_len,) uint8
      offsets: (N+1,) int64  where tree_i = blob[offsets[i]:offsets[i+1]]
    """
    names = [name for name, _ in model_tree_list]
    trees = [tree for _, tree in model_tree_list]

    lengths = np.array([len(t) for t in trees], dtype=np.int64)
    offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)])

    blob = b"".join(trees)
    blob_u8 = np.frombuffer(blob, dtype=np.uint8)

    names_arr = np.array(names, dtype="<U256")
    return names_arr, blob_u8, offsets


def get_sim_and_control_dt(env):
    """
    Best-effort extraction of sim timestep and a plausible control_dt.
    Returns (sim_dt, physics_steps_per_action, control_dt). NaN if unavailable.
    """
    sim_dt = np.nan
    steps = np.nan
    control_dt = np.nan

    pyrep = getattr(env, "_pyrep", None)
    if pyrep is not None and hasattr(pyrep, "get_simulation_timestep"):
        try:
            sim_dt = float(pyrep.get_simulation_timestep())
        except Exception:
            pass

    scene = getattr(env, "_scene", None)
    if scene is not None:
        for attr in ("_physics_steps_per_control_step", "_steps_per_action", "_physics_steps_per_action"):
            if hasattr(scene, attr):
                try:
                    steps = float(getattr(scene, attr))
                    break
                except Exception:
                    pass

    if np.isfinite(sim_dt) and np.isfinite(steps):
        control_dt = sim_dt * steps

    return sim_dt, steps, control_dt


def pack_obs(obs, record_front=True, record_wrist=True, record_overhead=False):
    """
    Minimal fields for reversibility judgment + Gemini frames.
    """
    out = {}
    out["joint_positions"] = _as_f32(obs.joint_positions)      # (7,)
    out["joint_velocities"] = _as_f32(obs.joint_velocities)    # (7,)

    out["gripper_open"] = _as_f32([obs.gripper_open])          # (1,)
    if obs.gripper_pose is not None:
        out["gripper_pose"] = _as_f32(obs.gripper_pose)        # (7,)

    if record_front and getattr(obs, "front_rgb", None) is not None:
        out["front_rgb"] = obs.front_rgb.astype(np.uint8)      # (H,W,3)
    if record_wrist and getattr(obs, "wrist_rgb", None) is not None:
        out["wrist_rgb"] = obs.wrist_rgb.astype(np.uint8)      # (H,W,3)
    if record_overhead and getattr(obs, "overhead_rgb", None) is not None:
        out["overhead_rgb"] = obs.overhead_rgb.astype(np.uint8)  # (H,W,3)

    return out


def stack_trajectory(frames):
    keys = sorted({k for f in frames for k in f.keys()})
    traj = {}
    for k in keys:
        if any(k not in f for f in frames):
            continue
        traj[k] = np.stack([f[k] for f in frames], axis=0)
    return traj


def derive_actions(traj, gripper_threshold=0.03):
    """
    Derive action proxies consistent with:
      MoveArmThenGripper(JointVelocity(), Discrete())

    Arm action: observed joint_velocities as proxy for commanded joint velocity.
    Gripper action: binarize gripper_open into {0,1}.
    """
    T = traj["joint_velocities"].shape[0]
    action_arm = traj["joint_velocities"].astype(np.float32)                 # (T,7)

    go = traj["gripper_open"].reshape(T)                                     # (T,)
    action_gripper = (go > gripper_threshold).astype(np.float32).reshape(T, 1)

    traj["action_arm"] = action_arm
    traj["action_gripper"] = action_gripper
    traj["action"] = np.concatenate([action_arm, action_gripper], axis=1)    # (T,8)

    traj["action_is_derived"] = np.array([1], dtype=np.int32)
    traj["action_arm_source"] = np.array(["joint_velocities"], dtype="<U32")
    traj["action_gripper_source"] = np.array(["threshold(gripper_open)"], dtype="<U64")
    traj["gripper_threshold"] = np.array([gripper_threshold], dtype=np.float32)


def select_keyframes(gripper_open_T1, T, max_k=12, event_window=3, gripper_threshold=0.03):
    """
    Pick indices:
      - include start & end
      - include indices where gripper state flips (open<->close), plus +/- event_window
      - fill remaining with uniform samples
    """
    go = gripper_open_T1.reshape(T)
    bin_state = (go > gripper_threshold).astype(np.int32)
    flips = np.where(bin_state[1:] != bin_state[:-1])[0] + 1

    idx = {0, T - 1}
    for f in flips:
        for d in range(-event_window, event_window + 1):
            j = f + d
            if 0 <= j < T:
                idx.add(int(j))

    idx = sorted(idx)

    # Fill with uniform samples if short
    if len(idx) < max_k:
        uni = np.linspace(0, T - 1, num=max_k, dtype=np.int32).tolist()
        for u in uni:
            if len(idx) >= max_k:
                break
            if u not in idx:
                idx.append(int(u))
        idx = sorted(set(idx))

    # Subsample if too many, keep endpoints
    if len(idx) > max_k:
        keep = [idx[0]]
        middle = idx[1:-1]
        take = np.linspace(0, len(middle) - 1, num=max_k - 2, dtype=np.int32)
        keep += [middle[i] for i in take]
        keep += [idx[-1]]
        idx = keep

    return np.array(idx, dtype=np.int32)


def load_core_conditions(task_name: str, config_dir: str = "/workspace/config"):
    base = os.path.join(config_dir, task_name)
    pre_path = os.path.join(base, "pre.txt")
    post_path = os.path.join(base, "post.txt")
    with open(pre_path, "r", encoding="utf-8") as f:
        pre = f.read().strip()
    with open(post_path, "r", encoding="utf-8") as f:
        post = f.read().strip()
    return pre, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="/workspace/data/rlbench_demos")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--task", type=str, default="StackBlocks",
                    help="RLBench task class name, e.g. StackBlocks, TakeUsbOutOfComputer, PutRubbishInBin, etc.")
    ap.add_argument("--variation", type=int, default=0)

    # Headless UX: default headless=True, allow --no-headless to show GUI
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.set_defaults(headless=True)

    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--no_wrist", action="store_true")
    ap.add_argument("--overhead", action="store_true", help="Record overhead_rgb too (often helpful)")

    ap.add_argument("--keyframes", type=int, default=12)
    ap.add_argument("--event_window", type=int, default=3)
    ap.add_argument("--gripper_threshold", type=float, default=0.03)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve task class dynamically
    if not hasattr(rlbench_tasks, args.task):
        raise ValueError(f"Unknown RLBench task '{args.task}'. "
                         f"Check rlbench.tasks for available task class names.")
    task_cls = getattr(rlbench_tasks, args.task)

    obs_config = ObservationConfig()
    obs_config.set_all(False)

    # Robot state
    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = True

    # Cameras: RGB only
    obs_config.front_camera.set_all(False)
    obs_config.front_camera.rgb = True
    obs_config.front_camera.image_size = (args.img, args.img)

    obs_config.wrist_camera.set_all(False)
    obs_config.wrist_camera.rgb = True
    obs_config.wrist_camera.image_size = (args.img, args.img)

    obs_config.overhead_camera.set_all(False)
    obs_config.overhead_camera.rgb = bool(args.overhead)
    obs_config.overhead_camera.image_size = (args.img, args.img)

    action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
    env = Environment(action_mode, obs_config=obs_config, headless=args.headless)
    env.launch()

    try:
        task = env.get_task(task_cls)
        task.set_variation(args.variation)

        sim_dt, steps_per_action, control_dt = get_sim_and_control_dt(env)

        # ------------------------------------------------------------
        # Hook scene.get_demo to snapshot CORE POST right after demo runs
        # ------------------------------------------------------------
        pyrep = getattr(env, "_pyrep", None)
        scene = getattr(task, "_scene", None) or getattr(env, "_scene", None)
        if pyrep is None or scene is None or not hasattr(scene, "get_demo"):
            raise RuntimeError("Could not access pyrep/scene.get_demo for snapshotting. "
                               "Check RLBench version/attributes.")

        snapshots = []  # list of per-demo: List[(name, tree_bytes)]
        orig_get_demo = scene.get_demo

        def get_demo_with_snapshot(*a, **kw):
            demo = orig_get_demo(*a, **kw)  # executes planner demo
            pyrep.step()

            models = get_top_level_models(pyrep)
            trees = [(m.get_name(), m.get_configuration_tree()) for m in models]
            snapshots.append(trees)
            return demo

        scene.get_demo = get_demo_with_snapshot
        try:
            demos = task.get_demos(amount=args.n, live_demos=True)
        finally:
            scene.get_demo = orig_get_demo

        if len(snapshots) != len(demos):
            print(f"[warn] snapshots={len(snapshots)} demos={len(demos)}. "
                  f"Will align by min length.")

        for i, demo in enumerate(demos):
            frames = []
            for obs in demo:
                frames.append(pack_obs(
                    obs,
                    record_front=True,
                    record_wrist=(not args.no_wrist),
                    record_overhead=bool(args.overhead),
                ))

            traj = stack_trajectory(frames)

            # Derive action proxies
            derive_actions(traj, gripper_threshold=args.gripper_threshold)

            # Keyframes for Gemini querying
            T = int(traj["joint_positions"].shape[0])
            k_idx = select_keyframes(
                traj["gripper_open"], T,
                max_k=args.keyframes,
                event_window=args.event_window,
                gripper_threshold=args.gripper_threshold
            )
            traj["keyframe_indices"] = k_idx

            # Attach CORE POST snapshot (model trees)
            if i < len(snapshots):
                model_trees = snapshots[i]
                names_arr, blob_u8, offsets = encode_model_trees_to_blob(model_trees)
                traj["post_state_model_names"] = names_arr
                traj["post_state_model_tree_blob"] = blob_u8
                traj["post_state_model_tree_offsets"] = offsets
                traj["post_state_snapshot_kind"] = np.array(["top_level_model_trees"], dtype="<U64")
                traj["post_state_model_count"] = np.array([len(names_arr)], dtype=np.int32)
            else:
                traj["post_state_snapshot_kind"] = np.array(["missing"], dtype="<U64")
                traj["post_state_model_count"] = np.array([0], dtype=np.int32)

            # Demo-level metadata
            traj["task"] = np.array([args.task], dtype="<U64")
            traj["variation"] = np.array([args.variation], dtype=np.int32)
            traj["demo_index"] = np.array([i], dtype=np.int32)
            traj["timestamp"] = np.array([time.time()], dtype=np.float64)
            traj["image_size"] = np.array([[args.img, args.img]], dtype=np.int32)

            traj["action_mode"] = np.array(["MoveArmThenGripper"], dtype="<U64")
            traj["arm_action_mode"] = np.array(["JointVelocity"], dtype="<U64")
            traj["gripper_action_mode"] = np.array(["Discrete"], dtype="<U64")

            traj["sim_dt"] = np.array([sim_dt], dtype=np.float64)
            traj["physics_steps_per_action"] = np.array([steps_per_action], dtype=np.float64)
            traj["control_dt"] = np.array([control_dt], dtype=np.float64)

            pre_core, post_core = load_core_conditions(args.task, config_dir="/workspace/config")
            traj["preconditions_core"] = np.array([pre_core], dtype="<U4096")
            traj["postconditions_core"] = np.array([post_core], dtype="<U4096")

            out_path = os.path.join(
                args.out_dir,
                f"{args.task}_var{args.variation:02d}_demo{i:04d}.npz"
            )
            np.savez_compressed(out_path, **traj)

            print(
                f"Saved {out_path}  T={T}  keyframes={len(k_idx)}  actions=DERIVED  "
                f"control_dt={control_dt}  post_state_models={int(traj['post_state_model_count'][0])}"
            )

    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
