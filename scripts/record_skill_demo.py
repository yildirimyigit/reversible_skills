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

# PyRep backend (for cffi access if needed)
try:
    from pyrep.backend import sim as sim_backend
except Exception:
    sim_backend = None


def _as_f32(x):
    return np.asarray(x, dtype=np.float32)


def get_sim_and_control_dt(env, task):
    """
    Best-effort extraction of simulator timestep and control step duration.
    Returns (sim_dt, physics_steps_per_action, control_dt). NaN if unavailable.
    """
    sim_dt = np.nan
    steps = np.nan
    control_dt = np.nan

    scene = getattr(task, "_scene", None)
    if scene is None:
        scene = getattr(env, "_scene", None)
    if scene is None:
        return sim_dt, steps, control_dt

    pyrep = getattr(scene, "_pyrep", None)
    if pyrep is None:
        pyrep = getattr(env, "_pyrep", None)

    if pyrep is not None and hasattr(pyrep, "get_simulation_timestep"):
        try:
            sim_dt = float(pyrep.get_simulation_timestep())
        except Exception:
            pass

    # RLBench internal names vary a bit across versions
    for attr in (
        "_physics_steps_per_control_step",
        "_steps_per_action",
        "_physics_steps_per_action",
        "physics_steps_per_action",
    ):
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


def _tree_cdata_to_bytes(tree_cdata, max_bytes=50_000_000):
    """
    Convert a CData pointer returned by CoppeliaSim/PyRep into full bytes.

    Strategy:
      1) If PyRep backend exposes a buffer size function, use it.
      2) Otherwise, try a common CoppeliaSim pattern: first 4 bytes encode total size.
    """
    if sim_backend is None:
        raise RuntimeError("pyrep.backend.sim is not available; cannot convert CData config tree.")

    ffi = getattr(sim_backend, "ffi", None)
    lib = getattr(sim_backend, "lib", None)
    if ffi is None:
        raise RuntimeError("sim_backend.ffi not found; cannot convert CData config tree.")

    # 1) Try explicit buffer size getters if present
    if lib is not None:
        for fn_name in ("simGetBufferSize", "simGetStringSize"):
            fn = getattr(lib, fn_name, None)
            if fn is not None:
                try:
                    n = int(fn(tree_cdata))
                    if 8 <= n <= max_bytes:
                        b = bytes(ffi.buffer(tree_cdata, n))
                        # release if available
                        rel = getattr(lib, "simReleaseBuffer", None)
                        if rel is not None:
                            try:
                                rel(tree_cdata)
                            except Exception:
                                pass
                        return b
                except Exception:
                    pass

    # 2) Heuristic: first 4 bytes store length (little endian)
    try:
        hdr = bytes(ffi.buffer(tree_cdata, 4))
        n0 = int.from_bytes(hdr, byteorder="little", signed=False)

        # Some APIs store either total size or payload size. Try both n0 and n0+4.
        for n in (n0, n0 + 4):
            if 8 <= n <= max_bytes:
                b = bytes(ffi.buffer(tree_cdata, n))
                # release if available
                if lib is not None:
                    rel = getattr(lib, "simReleaseBuffer", None)
                    if rel is not None:
                        try:
                            rel(tree_cdata)
                        except Exception:
                            pass
                return b
    except Exception:
        pass

    # Final fallback (will often be wrong, but better than silently saving 1 byte)
    raise RuntimeError("Could not infer configuration tree buffer size; refusing to save 1-byte snapshot.")


def get_configuration_tree_bytes(model):
    """
    Robustly obtain configuration tree as Python bytes.
    Handles:
      - bytes/bytearray
      - numpy uint8 arrays
      - cffi CData pointers (binary buffer)
    """
    tree = model.get_configuration_tree()

    if isinstance(tree, (bytes, bytearray)):
        return bytes(tree)

    if isinstance(tree, np.ndarray):
        # some builds return np.uint8 arrays
        if tree.dtype == np.uint8:
            return tree.tobytes()
        # object arrays not expected here
        raise TypeError(f"Unexpected ndarray dtype for config tree: {tree.dtype}")

    # cffi pointer
    return _tree_cdata_to_bytes(tree)


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


def select_keyframes(
    gripper_open_T1: np.ndarray,
    T: int,
    max_k: int = 12,
    event_window: int = 1,
    gripper_threshold: float = 0.03,
) -> np.ndarray:
    """
    Select EXACTLY max_k keyframes.

    Must-include (priority 1):
      - t = 0
      - t = T-1
      - every t where (gripper_open > threshold) changes (flip indices)

    If must-include count k < max_k:
      - optionally add context frames within +/- event_window around flips (priority 2),
        WITHOUT ever removing must-include frames
      - fill remaining slots using evenly spaced targets over [0, T-1] excluding already selected (priority 3)

    If must-include count k > max_k:
      - impossible to include all flips; we keep endpoints and uniformly subsample flip indices.
    """
    if max_k <= 0:
        return np.zeros((0,), dtype=np.int32)

    # If very short trajectory, pad by repeating last index.
    if T <= max_k:
        base = list(range(T))
        while len(base) < max_k:
            base.append(T - 1)
        return np.array(base, dtype=np.int32)

    go = gripper_open_T1.reshape(T)
    state = (go > gripper_threshold).astype(np.int32)
    flips = (np.where(state[1:] != state[:-1])[0] + 1).astype(np.int32)

    must = set([0, T - 1])
    must.update(int(f) for f in flips.tolist())
    must_list = sorted(must)

    # Case A: too many must-include frames (cannot satisfy requirement fully)
    if len(must_list) > max_k:
        if max_k == 1:
            return np.array([0], dtype=np.int32)

        locked = [0, T - 1]
        middle = [x for x in must_list if x not in locked]
        n_mid = max_k - 2

        if n_mid <= 0:
            return np.array(locked[:max_k], dtype=np.int32)

        take = np.linspace(0, len(middle) - 1, num=n_mid, dtype=np.int32)
        picked = [locked[0]] + [middle[i] for i in take.tolist()] + [locked[1]]
        picked = sorted(set(picked))

        # If duplicates collapsed (rare), fill sequentially
        t = 0
        while len(picked) < max_k and t < T:
            if t not in picked:
                picked.append(t)
            t += 1
        picked = sorted(picked)[:max_k]
        return np.array(picked, dtype=np.int32)

    # Case B: we can include all must frames
    selected = set(must_list)

    # Priority 2: optional context around flips, but only if budget allows
    if event_window > 0 and len(selected) < max_k and len(flips) > 0:
        context = set()
        for f in flips.tolist():
            for d in range(-event_window, event_window + 1):
                t = int(f + d)
                if 0 <= t < T and t not in selected:
                    context.add(t)

        # Add closer-to-flip context first
        flip_list = flips.tolist()

        def dist_to_nearest_flip(t: int) -> int:
            return min(abs(t - ff) for ff in flip_list)

        context_sorted = sorted(context, key=lambda t: (dist_to_nearest_flip(t), t))
        for t in context_sorted:
            if len(selected) >= max_k:
                break
            selected.add(int(t))

    # Priority 3: fill remaining using evenly spaced targets across the whole horizon
    if len(selected) < max_k:
        targets = np.linspace(0, T - 1, num=max_k, dtype=np.float32)

        for tf in targets:
            if len(selected) >= max_k:
                break

            t0 = int(np.round(tf))

            if t0 not in selected:
                selected.add(t0)
                continue

            # Find nearest free index around t0
            r = 1
            while (t0 - r) >= 0 or (t0 + r) < T:
                a = t0 - r
                b = t0 + r
                if a >= 0 and a not in selected:
                    selected.add(a)
                    break
                if b < T and b not in selected:
                    selected.add(b)
                    break
                r += 1

        # Final fallback: sequential fill (guarantees termination)
        if len(selected) < max_k:
            for t in range(T):
                if len(selected) >= max_k:
                    break
                if t not in selected:
                    selected.add(t)

    out = np.array(sorted(selected), dtype=np.int32)

    # Ensure exact length (rare edge case: trim deterministically if something went wrong)
    if out.shape[0] > max_k:
        locked = [0, T - 1]
        middle = [x for x in out.tolist() if x not in locked]
        n_mid = max_k - 2

        if n_mid <= 0:
            out = np.array(locked[:max_k], dtype=np.int32)
        else:
            take = np.linspace(0, len(middle) - 1, num=n_mid, dtype=np.int32)
            out = np.array([locked[0]] + [middle[i] for i in take.tolist()] + [locked[1]], dtype=np.int32)
            out = np.unique(out)

    # If uniqueness collapsed, fill to length
    out_list = out.tolist()
    t = 0
    while len(out_list) < max_k:
        if t not in out_list:
            out_list.append(t)
        t += 1
    out = np.array(sorted(out_list)[:max_k], dtype=np.int32)

    print(f"Flips: {flips.tolist()}, must-include frames: {must_list}")
    print(f"Selected keyframes (total {len(out)}/{T}): {out.tolist()}")

    assert out.shape[0] == max_k, f"Expected {max_k} keyframes, got {out.shape[0]}"
    return out



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
    ap.add_argument("--task", type=str, default="StackBlocks")
    ap.add_argument("--variation", type=int, default=0)

    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.set_defaults(headless=True)

    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--no_wrist", action="store_true")
    ap.add_argument("--overhead", action="store_true")

    ap.add_argument("--keyframes", type=int, default=12)
    ap.add_argument("--event_window", type=int, default=1)
    ap.add_argument("--gripper_threshold", type=float, default=0.03)

    ap.add_argument("--config_dir", type=str, default="/workspace/config")

    # Snapshots (curriculum resets)
    ap.add_argument("--save_snapshots", action="store_true", default=True)
    ap.add_argument("--no_snapshots", dest="save_snapshots", action="store_false")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not hasattr(rlbench_tasks, args.task):
        raise ValueError(f"Unknown RLBench task '{args.task}'.")
    task_cls = getattr(rlbench_tasks, args.task)

    obs_config = ObservationConfig()
    obs_config.set_all(False)

    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = True

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

        sim_dt, steps_per_action, control_dt = get_sim_and_control_dt(env, task)

        pre_core, post_core = load_core_conditions(args.task, config_dir=args.config_dir)

        for i in range(args.n):
            scene = getattr(task, "_scene", None)
            if scene is None:
                raise RuntimeError("Could not access task._scene; RLBench internals differ.")
            pyrep = getattr(scene, "_pyrep", None)
            if pyrep is None:
                pyrep = getattr(env, "_pyrep", None)

            # Snapshot storage per forward_t:
            all_step_rows = []   # list of rows; each row is list[bytes] len M
            model_names = None
            snapshot_failed = False

            orig_demo_record_step = getattr(scene, "_demo_record_step", None)

            def _patched_demo_record_step(*a, **kw):
                nonlocal model_names, snapshot_failed
                out = orig_demo_record_step(*a, **kw)

                if args.save_snapshots and not snapshot_failed:
                    try:
                        models = get_top_level_models(pyrep)
                        if model_names is None:
                            model_names = [m.get_name() for m in models]

                        row = [get_configuration_tree_bytes(m) for m in models]
                        all_step_rows.append(row)
                    except Exception as e:
                        snapshot_failed = True
                        # maintain alignment with a placeholder row
                        if model_names is None:
                            try:
                                models = get_top_level_models(pyrep)
                                model_names = [m.get_name() for m in models]
                            except Exception:
                                model_names = []
                        all_step_rows.append([b"" for _ in model_names])

                return out

            if args.save_snapshots:
                if orig_demo_record_step is None:
                    raise RuntimeError("scene._demo_record_step not found; cannot capture snapshots in this build.")
                scene._demo_record_step = _patched_demo_record_step

            try:
                demos = task.get_demos(amount=1, live_demos=True)
            finally:
                if args.save_snapshots and orig_demo_record_step is not None:
                    scene._demo_record_step = orig_demo_record_step

            if len(demos) == 0:
                raise RuntimeError("No demo returned.")
            demo = demos[0]

            frames = []
            for obs in demo:
                frames.append(pack_obs(
                    obs,
                    record_front=True,
                    record_wrist=(not args.no_wrist),
                    record_overhead=bool(args.overhead),
                ))

            traj = stack_trajectory(frames)
            derive_actions(traj, gripper_threshold=args.gripper_threshold)

            T = int(traj["joint_positions"].shape[0])

            # Keyframes
            k_idx = select_keyframes(
                traj["gripper_open"], T,
                max_k=args.keyframes,
                event_window=args.event_window,
                gripper_threshold=args.gripper_threshold,
            )
            traj["keyframe_indices"] = k_idx

            # Snapshots: downselect to keyframes + post
            if args.save_snapshots:
                snap_T = len(all_step_rows)
                if snap_T == 0:
                    raise RuntimeError("Snapshots requested but none captured.")

                # Capture a snapshot of the *true* final simulator state after demo ends
                models_now = get_top_level_models(pyrep)
                if model_names is None:
                    model_names = [m.get_name() for m in models_now]
                final_row = [get_configuration_tree_bytes(m) for m in models_now]

                # Ensure the last stored row corresponds to the true final state
                all_step_rows[-1] = final_row

                # Now match lengths (pad or truncate)
                snap_T = len(all_step_rows)
                if snap_T < T:
                    all_step_rows.extend([final_row.copy() for _ in range(T - snap_T)])
                elif snap_T > T:
                    all_step_rows = all_step_rows[:T]

                traj["keyframe_indices"] = k_idx

                if model_names is None:
                    model_names = []

                M = len(model_names)
                K = int(k_idx.shape[0])

                # Build (K, M) object array of bytes
                kf_mat = np.empty((K, M), dtype=object)
                for r, t in enumerate(k_idx.tolist()):
                    row = all_step_rows[int(t)]
                    # hard guard: ensure bytes and not 1-byte accidents
                    for c in range(M):
                        b = row[c]
                        if not isinstance(b, (bytes, bytearray)):
                            raise TypeError(f"Snapshot cell not bytes at t={t}, model={c}: {type(b)}")
                        if len(b) <= 1:
                            raise RuntimeError(f"Snapshot too small at t={t}, model={c}: len={len(b)}")
                    kf_mat[r, :] = row

                post_row = all_step_rows[T - 1]
                post_arr = np.array(post_row, dtype=object)
                for c, b in enumerate(post_arr.tolist()):
                    if not isinstance(b, (bytes, bytearray)):
                        raise TypeError(f"Post snapshot cell not bytes at model={c}: {type(b)}")
                    if len(b) <= 1:
                        raise RuntimeError(f"Post snapshot too small at model={c}: len={len(b)}")

                traj["snapshot_storage"] = np.array(["bytes_v1"], dtype="<U16")
                traj["snapshot_model_names"] = np.array(model_names, dtype="<U256")
                traj["snapshot_keyframe_trees"] = kf_mat
                traj["snapshot_post_trees"] = post_arr
                traj["snapshot_captured"] = np.array([1], dtype=np.int32)
                traj["snapshot_failed"] = np.array([1 if snapshot_failed else 0], dtype=np.int32)
                traj["snapshot_source"] = np.array(["scene._demo_record_step"], dtype="<U64")
            else:
                traj["snapshot_storage"] = np.array(["none"], dtype="<U16")
                traj["snapshot_captured"] = np.array([0], dtype=np.int32)
                traj["snapshot_failed"] = np.array([0], dtype=np.int32)

            # Metadata
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

            out_path = os.path.join(args.out_dir, f"{args.task}_var{args.variation:02d}_demo{i:04d}.npz")
            np.savez_compressed(out_path, **traj)

            print(
                f"Saved {out_path}  T={T}  keyframes={int(traj['keyframe_indices'].shape[0])}  actions=DERIVED  "
                f"control_dt={float(traj['control_dt'][0])}  snapshots={int(traj['snapshot_captured'][0])}  "
                f"snapshot_failed={int(traj['snapshot_failed'][0])}"
            )

    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
