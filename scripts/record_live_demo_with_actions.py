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

# PyRep backend (optional, only for snapshot bytes conversion)
try:
    from pyrep.backend import sim as sim_backend
except Exception:
    sim_backend = None


def _as_f32(x):
    return np.asarray(x, dtype=np.float32)


# -----------------------------
# dt info (best effort)
# -----------------------------
def get_sim_and_control_dt(env, task):
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


# -----------------------------
# Snapshots
# -----------------------------
def get_top_level_models(pyrep):
    roots = pyrep.get_objects_in_tree(root_object=None, first_generation_only=True)
    models = [o for o in roots if o.is_model()]
    models.sort(key=lambda o: o.get_name())
    return models


def _tree_cdata_to_bytes(tree_cdata, max_bytes=50_000_000):
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
        for n in (n0, n0 + 4):
            if 8 <= n <= max_bytes:
                b = bytes(ffi.buffer(tree_cdata, n))
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

    raise RuntimeError("Could not infer configuration tree buffer size.")


def get_configuration_tree_bytes(model):
    tree = model.get_configuration_tree()
    if isinstance(tree, (bytes, bytearray)):
        return bytes(tree)
    if isinstance(tree, np.ndarray):
        if tree.dtype == np.uint8:
            return tree.tobytes()
        raise TypeError(f"Unexpected ndarray dtype for config tree: {tree.dtype}")
    return _tree_cdata_to_bytes(tree)


# -----------------------------
# Obs packing
# -----------------------------
def pack_obs(obs, record_front=True, record_wrist=True, record_overhead=False, record_lowdim=False):
    out = {}
    out["joint_positions"] = _as_f32(obs.joint_positions)  # (7,)

    if getattr(obs, "joint_velocities", None) is not None:
        out["joint_velocities"] = _as_f32(obs.joint_velocities)  # (7,)

    # gripper_open in RLBench is usually bool/float
    go = float(obs.gripper_open) if not isinstance(obs.gripper_open, (bool, np.bool_)) else (1.0 if obs.gripper_open else 0.0)
    out["gripper_open"] = _as_f32([go])  # (1,)

    if getattr(obs, "gripper_pose", None) is not None:
        out["gripper_pose"] = _as_f32(obs.gripper_pose)  # (7,)

    # record misc joint_poses if present (likely 7 joints x 7D pose each)
    jp = None
    try:
        jp = obs.misc.get("joint_poses", None)
    except Exception:
        jp = None
    if jp is not None:
        arr = np.asarray(jp, dtype=np.float32)
        if arr.shape == (7, 7):
            out["joint_link_poses"] = arr  # (7,7)

    if record_lowdim and getattr(obs, "task_low_dim_state", None) is not None:
        out["task_low_dim_state"] = _as_f32(obs.task_low_dim_state)

    if record_front and getattr(obs, "front_rgb", None) is not None:
        out["front_rgb"] = obs.front_rgb.astype(np.uint8)
    if record_wrist and getattr(obs, "wrist_rgb", None) is not None:
        out["wrist_rgb"] = obs.wrist_rgb.astype(np.uint8)
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


# -----------------------------
# Action synthesis
# -----------------------------
def synthesize_actions(traj, *, gripper_threshold=0.03, invert_gripper=False, assume_control_dt=0.05):
    """
    Produces:
      - action_qpos: (T-1, 7+1) = q_target(next) + gripper_cmd(next)
      - action_vel_fd: (T-1, 7+1) finite difference using dt_used + gripper_cmd(next)
      - action_vel_obs: (T-1, 7+1) from obs.joint_velocities (if available) + gripper_cmd(next)
    Also sets:
      - action: alias to action_qpos  (dt-free)
    """
    q = traj["joint_positions"].astype(np.float32)  # (T,7)
    T = int(q.shape[0])
    if T < 2:
        raise RuntimeError("Demo too short (need at least 2 frames).")

    go = traj["gripper_open"].reshape(T).astype(np.float32)
    g_open = (go > float(gripper_threshold)).astype(np.float32)  # 1=open, 0=closed
    if invert_gripper:
        g_open = 1.0 - g_open

    # command applied to reach state t+1:
    g_cmd = g_open[1:]  # (T-1,)

    # q-target actions
    q_tgt = q[1:]  # (T-1,7)
    action_qpos = np.concatenate([q_tgt, g_cmd[:, None]], axis=1).astype(np.float32)

    # dt for vel_fd
    dt_used = float(traj.get("control_dt", np.array([np.nan], dtype=np.float64))[0])
    dt_source = "control_dt"
    if not np.isfinite(dt_used) or dt_used <= 0:
        dt_used = float(assume_control_dt)
        dt_source = "assumed"

    v_fd = (q[1:] - q[:-1]) / float(dt_used)
    action_vel_fd = np.concatenate([v_fd.astype(np.float32), g_cmd[:, None]], axis=1).astype(np.float32)

    action_vel_obs = None
    if "joint_velocities" in traj:
        v_obs = traj["joint_velocities"].astype(np.float32)
        if v_obs.shape[0] == T:
            v_obs = v_obs[:-1]  # align to (T-1,7)
        if v_obs.shape[0] == T - 1:
            action_vel_obs = np.concatenate([v_obs, g_cmd[:, None]], axis=1).astype(np.float32)

    traj["action_qpos"] = action_qpos
    traj["action_vel_fd"] = action_vel_fd
    if action_vel_obs is not None:
        traj["action_vel_obs"] = action_vel_obs

    # Default action alias: dt-free, replayable
    traj["action"] = action_qpos

    traj["gripper_threshold"] = np.array([float(gripper_threshold)], dtype=np.float32)
    traj["invert_gripper"] = np.array([1 if invert_gripper else 0], dtype=np.int32)
    traj["dt_used_for_vel_fd"] = np.array([dt_used], dtype=np.float64)
    traj["dt_source_for_vel_fd"] = np.array([dt_source], dtype="<U16")


# -----------------------------
# Keyframes (simple, deterministic)
# -----------------------------
def select_keyframes_simple(gripper_open_T1: np.ndarray, T: int, max_k: int = 12, gripper_threshold: float = 0.03):
    """
    Always include:
      - t=0
      - t=T-1
      - all gripper flip indices
    Then fill remaining with uniform spacing.
    If too many flips, subsample flips uniformly.
    """
    if max_k <= 0:
        return np.zeros((0,), dtype=np.int32)

    if T <= max_k:
        base = list(range(T))
        while len(base) < max_k:
            base.append(T - 1)
        return np.array(base, dtype=np.int32)

    go = gripper_open_T1.reshape(T)
    state = (go > gripper_threshold).astype(np.int32)
    flips = (np.where(state[1:] != state[:-1])[0] + 1).astype(np.int32)

    must = [0, T - 1] + flips.tolist()
    must = sorted(set(must))

    if len(must) > max_k:
        locked = [0, T - 1]
        middle = [x for x in must if x not in locked]
        n_mid = max_k - 2
        take = np.linspace(0, len(middle) - 1, num=max(0, n_mid), dtype=np.int32) if n_mid > 0 else np.array([], dtype=np.int32)
        picked = [locked[0]] + ([middle[i] for i in take.tolist()] if n_mid > 0 else []) + [locked[1]]
        return np.array(sorted(set(picked))[:max_k], dtype=np.int32)

    selected = set(must)
    if len(selected) < max_k:
        targets = np.linspace(0, T - 1, num=max_k, dtype=np.float32)
        for tf in targets:
            if len(selected) >= max_k:
                break
            t0 = int(np.round(tf))
            if t0 not in selected:
                selected.add(t0)

    out = np.array(sorted(selected), dtype=np.int32)
    if out.shape[0] > max_k:
        out = out[:max_k]
    while out.shape[0] < max_k:
        # pad deterministically
        out = np.concatenate([out, np.array([T - 1], dtype=np.int32)], axis=0)
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", type=str, default="/workspace/data/rlbench_demos")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--variation", type=int, default=0)

    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.set_defaults(headless=True)

    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--no_wrist", action="store_true")
    ap.add_argument("--overhead", action="store_true")
    ap.add_argument("--record_lowdim", action="store_true")

    ap.add_argument("--keyframes", type=int, default=12)
    ap.add_argument("--gripper_threshold", type=float, default=0.03)
    ap.add_argument("--invert_gripper", action="store_true")

    ap.add_argument("--assume_control_dt", type=float, default=0.05, help="Used only if control_dt is NaN.")

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
    obs_config.task_low_dim_state = bool(args.record_lowdim)

    obs_config.front_camera.set_all(False)
    obs_config.front_camera.rgb = True
    obs_config.front_camera.image_size = (args.img, args.img)

    obs_config.wrist_camera.set_all(False)
    obs_config.wrist_camera.rgb = True
    obs_config.wrist_camera.image_size = (args.img, args.img)

    obs_config.overhead_camera.set_all(False)
    obs_config.overhead_camera.rgb = bool(args.overhead)
    obs_config.overhead_camera.image_size = (args.img, args.img)

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=args.headless,
    )
    env.launch()

    try:
        task = env.get_task(task_cls)
        task.set_variation(args.variation)

        sim_dt, steps_per_action, control_dt = get_sim_and_control_dt(env, task)

        for di in range(int(args.n)):
            scene = getattr(task, "_scene", None)
            if scene is None:
                raise RuntimeError("Could not access task._scene; RLBench internals differ.")
            pyrep = getattr(scene, "_pyrep", None)
            if pyrep is None:
                pyrep = getattr(env, "_pyrep", None)

            # Snapshot storage aligned per demo step
            all_step_rows = []
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
                    except Exception:
                        snapshot_failed = True
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
                    record_lowdim=bool(args.record_lowdim),
                ))

            traj = stack_trajectory(frames)

            # Attach dt metadata early so synthesize_actions can use it
            traj["sim_dt"] = np.array([sim_dt], dtype=np.float64)
            traj["physics_steps_per_action"] = np.array([steps_per_action], dtype=np.float64)
            traj["control_dt"] = np.array([control_dt], dtype=np.float64)

            # Actions
            synthesize_actions(
                traj,
                gripper_threshold=float(args.gripper_threshold),
                invert_gripper=bool(args.invert_gripper),
                assume_control_dt=float(args.assume_control_dt),
            )

            T = int(traj["joint_positions"].shape[0])

            # Keyframes
            k_idx = select_keyframes_simple(traj["gripper_open"], T, max_k=int(args.keyframes), gripper_threshold=float(args.gripper_threshold))
            traj["keyframe_indices"] = k_idx

            # Snapshots (downselect to keyframes + post)
            if args.save_snapshots:
                if len(all_step_rows) == 0:
                    raise RuntimeError("Snapshots requested but none captured.")

                models_now = get_top_level_models(pyrep)
                if model_names is None:
                    model_names = [m.get_name() for m in models_now]
                final_row = [get_configuration_tree_bytes(m) for m in models_now]

                # force last row to true final state
                all_step_rows[-1] = final_row

                # match length to T (pad/truncate)
                snap_T = len(all_step_rows)
                if snap_T < T:
                    all_step_rows.extend([final_row.copy() for _ in range(T - snap_T)])
                elif snap_T > T:
                    all_step_rows = all_step_rows[:T]

                M = len(model_names)
                K = int(k_idx.shape[0])

                kf_mat = np.empty((K, M), dtype=object)
                for r, t in enumerate(k_idx.tolist()):
                    row = all_step_rows[int(t)]
                    for c in range(M):
                        b = row[c]
                        if not isinstance(b, (bytes, bytearray)) or len(b) <= 1:
                            raise RuntimeError(f"Bad snapshot bytes at t={t}, model={c}, type={type(b)}, len={len(b) if isinstance(b,(bytes,bytearray)) else 'NA'}")
                    kf_mat[r, :] = row

                post_row = all_step_rows[T - 1]
                post_arr = np.array(post_row, dtype=object)
                for c, b in enumerate(post_arr.tolist()):
                    if not isinstance(b, (bytes, bytearray)) or len(b) <= 1:
                        raise RuntimeError(f"Bad post snapshot bytes at model={c}, type={type(b)}, len={len(b) if isinstance(b,(bytes,bytearray)) else 'NA'}")

                traj["snapshot_storage"] = np.array(["bytes_v1"], dtype="<U16")
                traj["snapshot_model_names"] = np.array(model_names, dtype="<U256")
                traj["snapshot_keyframe_trees"] = kf_mat
                traj["snapshot_post_trees"] = post_arr
                traj["snapshot_captured"] = np.array([1], dtype=np.int32)
                traj["snapshot_failed"] = np.array([1 if snapshot_failed else 0], dtype=np.int32)
            else:
                traj["snapshot_storage"] = np.array(["none"], dtype="<U16")
                traj["snapshot_captured"] = np.array([0], dtype=np.int32)
                traj["snapshot_failed"] = np.array([0], dtype=np.int32)

            # Metadata
            traj["task"] = np.array([args.task], dtype="<U64")
            traj["variation"] = np.array([args.variation], dtype=np.int32)
            traj["demo_index"] = np.array([di], dtype=np.int32)
            traj["timestamp"] = np.array([time.time()], dtype=np.float64)
            traj["image_size"] = np.array([[args.img, args.img]], dtype=np.int32)

            traj["action_mode"] = np.array(["MoveArmThenGripper"], dtype="<U64")
            traj["arm_action_mode"] = np.array(["JointVelocity"], dtype="<U64")
            traj["gripper_action_mode"] = np.array(["Discrete"], dtype="<U64")

            # Explicitly document meanings
            traj["action_meaning_action_qpos"] = np.array(["[q_target_next(7), gripper_cmd_next(1)]"], dtype="<U96")
            traj["action_meaning_action_vel_fd"] = np.array(["[(q_next-q_now)/dt_used(7), gripper_cmd_next(1)]"], dtype="<U96")
            traj["action_meaning_action_vel_obs"] = np.array(["[obs_joint_vel(7), gripper_cmd_next(1)]"], dtype="<U96")

            out_path = os.path.join(args.out_dir, f"{args.task}_var{args.variation:02d}_demo{di:04d}.npz")
            np.savez_compressed(out_path, **traj)

            print(f"[saved] {out_path}")
            print(f"  T={T}  keyframes={int(traj['keyframe_indices'].shape[0])}  snapshots={int(traj['snapshot_captured'][0])}")
            print(f"  control_dt={float(traj['control_dt'][0])}  dt_used_for_vel_fd={float(traj['dt_used_for_vel_fd'][0])} (source={traj['dt_source_for_vel_fd'][0]})")
            print(f"  actions: action(alias)=action_qpos, plus action_vel_fd, action_vel_obs(if available)")

    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
