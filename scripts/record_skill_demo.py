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


def get_sim_and_control_dt(env):
    """
    Best-effort extraction of simulator timestep and control step duration.
    Returns (sim_dt, physics_steps_per_action, control_dt). NaN if unavailable.
    """
    sim_dt = np.nan
    steps = np.nan
    control_dt = np.nan

    scene = getattr(env, "_scene", None)
    if scene is None:
        return sim_dt, steps, control_dt

    pyrep = getattr(scene, "_pyrep", None)
    if pyrep is not None and hasattr(pyrep, "get_simulation_timestep"):
        try:
            sim_dt = float(pyrep.get_simulation_timestep())
        except Exception:
            pass

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


def get_top_level_models(pyrep):
    """
    First-generation objects under 'world', keep only models (task props typically are models).
    Deterministic order by name.
    """
    roots = pyrep.get_objects_in_tree(root_object=None, first_generation_only=True)
    models = [o for o in roots if o.is_model()]
    models.sort(key=lambda o: o.get_name())
    return models


def pack_obs(obs, record_front=True, record_wrist=True, record_overhead=False):
    """
    Minimal fields for reversibility judgment + Gemini frames + RL warm-start.
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
        out["overhead_rgb"] = obs.overhead_rgb.astype(np.uint8)

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


def load_core_conditions(task_name: str, config_dir: str):
    base = os.path.join(config_dir, task_name)
    pre_path = os.path.join(base, "pre.txt")
    post_path = os.path.join(base, "post.txt")
    pre = ""
    post = ""
    if os.path.isfile(pre_path):
        with open(pre_path, "r", encoding="utf-8") as f:
            pre = f.read().strip()
    if os.path.isfile(post_path):
        with open(post_path, "r", encoding="utf-8") as f:
            post = f.read().strip()
    return pre, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="/workspace/data/rlbench_demos")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--task", type=str, default="StackBlocks",
                    help="RLBench task class name, e.g. StackBlocks, CloseDrawer, PutRubbishInBin, etc.")
    ap.add_argument("--variation", type=int, default=0)

    # Headless UX: default headless=True, allow --no-headless to show GUI
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.set_defaults(headless=True)

    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--no_wrist", action="store_true")
    ap.add_argument("--overhead", action="store_true", help="Record overhead_rgb too")

    ap.add_argument("--keyframes", type=int, default=12)
    ap.add_argument("--event_window", type=int, default=3)
    ap.add_argument("--gripper_threshold", type=float, default=0.03)

    ap.add_argument("--config_dir", type=str, default="/workspace/config",
                    help="Contains config/<TaskName>/pre.txt and post.txt")

    # Snapshot saving (for curriculum resets)
    ap.add_argument("--save_snapshots", action="store_true", default=True)
    ap.add_argument("--no_snapshots", dest="save_snapshots", action="store_false")
    ap.add_argument("--snapshot_settle_steps", type=int, default=0,
                    help="Extra pyrep.step() calls after demo step capture. 0 is safest (do not advance sim).")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve task class dynamically
    if not hasattr(rlbench_tasks, args.task):
        raise ValueError(f"Unknown RLBench task '{args.task}'. Check rlbench.tasks for available class names.")
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

        # Load CORE PRE/POST from config files (recommended)
        pre_core, post_core = load_core_conditions(args.task, config_dir=args.config_dir)

        # Record demos one-by-one so we can attach per-demo snapshot hooks cleanly
        for i in range(args.n):
            # --- snapshot hook state ---
            all_step_trees = []          # list[ list[tree] ] per forward_t
            model_names = None           # list[str]
            snapshot_failed = False

            scene = getattr(task, "_scene", None)
            if scene is None:
                raise RuntimeError("Could not access task._scene (RLBench internals differ).")

            pyrep = getattr(scene, "_pyrep", None)
            if pyrep is None:
                # fallback: some builds store it on env
                pyrep = getattr(env, "_pyrep", None)
            if args.save_snapshots and pyrep is None:
                raise RuntimeError("Snapshots requested but could not access PyRep instance to capture trees.")

            # Patch RLBench internal step recorder to capture config trees per step
            orig_demo_record_step = getattr(scene, "_demo_record_step", None)

            def _patched_demo_record_step(*a, **kw):
                nonlocal model_names, snapshot_failed
                out = orig_demo_record_step(*a, **kw)

                if args.save_snapshots and not snapshot_failed:
                    try:
                        models = get_top_level_models(pyrep)
                        if model_names is None:
                            model_names = [m.get_name() for m in models]
                        trees = [m.get_configuration_tree() for m in models]
                        all_step_trees.append(trees)

                        # Do NOT advance sim unless user explicitly wants it (default is 0).
                        for _ in range(int(args.snapshot_settle_steps)):
                            pyrep.step()
                    except Exception:
                        snapshot_failed = True
                        # still keep alignment length by appending None
                        all_step_trees.append(None)

                return out

            if args.save_snapshots:
                if orig_demo_record_step is None:
                    raise RuntimeError("This RLBench build does not expose scene._demo_record_step; cannot capture snapshots.")
                scene._demo_record_step = _patched_demo_record_step

            try:
                demos = task.get_demos(amount=1, live_demos=True)
            finally:
                # restore hook no matter what
                if args.save_snapshots and orig_demo_record_step is not None:
                    scene._demo_record_step = orig_demo_record_step

            if len(demos) == 0:
                raise RuntimeError("No demo returned.")
            demo = demos[0]

            # Pack observations
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

            # Keyframes
            T = int(traj["joint_positions"].shape[0])
            k_idx = select_keyframes(
                traj["gripper_open"], T,
                max_k=args.keyframes,
                event_window=args.event_window,
                gripper_threshold=args.gripper_threshold
            )
            traj["keyframe_indices"] = k_idx

            # --- Snapshots: prune to keyframes + post ---
            if args.save_snapshots:
                # Align lengths if the hook captured slightly different count
                if len(all_step_trees) == 0:
                    raise RuntimeError("Snapshots requested but none captured (hook did not run).")

                if len(all_step_trees) != T:
                    m = min(len(all_step_trees), T)
                    all_step_trees = all_step_trees[:m]
                    for key in ("joint_positions", "joint_velocities", "gripper_open", "gripper_pose",
                                "front_rgb", "wrist_rgb", "overhead_rgb",
                                "action_arm", "action_gripper", "action"):
                        if key in traj and traj[key].shape[0] >= m:
                            traj[key] = traj[key][:m]
                    T = m
                    traj["keyframe_indices"] = np.array([t for t in k_idx.tolist() if t < T], dtype=np.int32)
                    k_idx = traj["keyframe_indices"]

                # If model_names never set, still store empty
                if model_names is None:
                    model_names = []

                # Save only keyframe snapshots (object array)
                keyframe_trees = [all_step_trees[int(t)] for t in k_idx.tolist()]
                post_trees = all_step_trees[T - 1]

                traj["snapshot_model_names"] = np.array(model_names, dtype="<U256")
                traj["snapshot_keyframe_trees"] = np.array(keyframe_trees, dtype=object)   # length K
                traj["snapshot_post_trees"] = np.array([post_trees], dtype=object)        # length 1
                traj["snapshot_captured"] = np.array([1], dtype=np.int32)
                traj["snapshot_failed"] = np.array([1 if snapshot_failed else 0], dtype=np.int32)
                traj["snapshot_source"] = np.array(["scene._demo_record_step"], dtype="<U64")
            else:
                traj["snapshot_captured"] = np.array([0], dtype=np.int32)

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

            traj["preconditions_core"] = np.array([pre_core], dtype="<U4096")
            traj["postconditions_core"] = np.array([post_core], dtype="<U4096")

            out_path = os.path.join(
                args.out_dir,
                f"{args.task}_var{args.variation:02d}_demo{i:04d}.npz"
            )
            np.savez_compressed(out_path, **traj)

            snap = int(traj["snapshot_captured"][0])
            print(
                f"Saved {out_path}  T={T}  keyframes={len(k_idx)}  actions=DERIVED  "
                f"control_dt={control_dt}  snapshots={snap}"
            )

    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
