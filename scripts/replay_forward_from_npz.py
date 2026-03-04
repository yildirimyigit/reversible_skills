#!/usr/bin/env python3
"""
replay_forward_from_npz.py

Replays a recorded RLBench demo saved by record_live_demo_with_actions.py.

This version is compatible with BOTH snapshot formats:

  - bytes_v1 (old): snapshot_keyframe_trees recorded for top-level MODELS only
  - bytes_v2_roots (new): snapshot_keyframe_trees recorded for ALL first-generation ROOT objects

Important: In your PyRep build, root objects generally do NOT expose set_configuration_tree(),
so snapshot restore must be done via env._pyrep.set_configuration_tree(tree_bytes) in the
original captured order.

Replay:
- dt-free joint servo to recorded q[t+1]
- gripper command from action_qpos[t,7] (open=1 close=0), with optional invert_gripper
"""

from __future__ import annotations

import os
import time
import argparse
import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete


def as_f32(x):
    return np.asarray(x, dtype=np.float32)


def make_obs_config_no_cameras():
    oc = ObservationConfig()
    try:
        oc.set_all(False)
    except Exception:
        pass

    for attr in ["joint_positions", "gripper_open", "gripper_pose"]:
        if hasattr(oc, attr):
            setattr(oc, attr, True)

    for cam_attr in [
        "front_camera", "wrist_camera", "overhead_camera",
        "left_shoulder_camera", "right_shoulder_camera",
    ]:
        cam = getattr(oc, cam_attr, None)
        if cam is not None:
            try:
                cam.set_all(False)
            except Exception:
                for f in ["rgb", "depth", "point_cloud", "mask"]:
                    if hasattr(cam, f):
                        setattr(cam, f, False)
    return oc


def task_success(task) -> bool:
    try:
        s = task._task.success()
        if isinstance(s, (tuple, list)):
            return bool(s[0])
        return bool(s)
    except Exception:
        return False


def get_pyrep(env, task):
    pr = getattr(env, "_pyrep", None)
    if pr is not None:
        return pr
    scene = getattr(task, "_scene", None)
    if scene is not None:
        pr = getattr(scene, "_pyrep", None)
        if pr is not None:
            return pr
    raise RuntimeError("Could not access PyRep instance (env._pyrep / task._scene._pyrep).")


def settle(task, g_cmd: float, steps: int):
    a = np.zeros((8,), dtype=np.float32)
    a[7] = 1.0 if g_cmd > 0.5 else 0.0
    obs = None
    for _ in range(int(steps)):
        obs, _, _ = task.step(a)
    return obs


def set_robot_state(scene, q: np.ndarray, g: float):
    """Force arm joints + targets to prevent drift after restore."""
    arm = scene.robot.arm
    q = as_f32(q).reshape(-1)
    q_list = q.tolist()

    try:
        arm.set_joint_positions(q_list, disable_dynamics=True)
    except TypeError:
        try:
            arm.set_joint_positions(q_list)
        except Exception:
            for i, j in enumerate(arm.joints):
                try:
                    j.set_joint_position(float(q[i]))
                except Exception:
                    pass

    try:
        arm.set_joint_target_positions(q_list)
    except Exception:
        pass

    try:
        arm.set_joint_target_velocities([0.0] * len(q_list))
    except Exception:
        pass

    try:
        if g > 0.5:
            scene.robot.gripper.open()
        else:
            scene.robot.gripper.close()
    except Exception:
        pass


def _read_snapshot_format(npz) -> str:
    if "snapshot_storage" not in npz.files:
        return "none"
    return str(npz["snapshot_storage"][0])


def load_initial_snapshot_trees(npz):
    """
    Returns:
      trees_for_kf0: list[bytes] to pass to pyrep.set_configuration_tree in order
      names:         list[str] of corresponding roots/models (for debug only)
      fmt:           snapshot_storage string
    """
    fmt = _read_snapshot_format(npz)

    snap_ok = (
        fmt != "none"
        and int(npz.get("snapshot_captured", np.array([0], dtype=np.int32))[0]) == 1
        and int(npz.get("snapshot_failed", np.array([0], dtype=np.int32))[0]) == 0
        and ("snapshot_keyframe_trees" in npz.files)
    )
    if not snap_ok:
        return None, None, fmt

    trees = np.asarray(npz["snapshot_keyframe_trees"], dtype=object)
    if trees.ndim != 2:
        raise RuntimeError(f"snapshot_keyframe_trees must be 2D, got {trees.shape}")

    # Names: v2 uses snapshot_root_names; v1 used snapshot_model_names
    if "snapshot_root_names" in npz.files:
        names = [str(x) for x in np.asarray(npz["snapshot_root_names"]).tolist()]
    elif "snapshot_model_names" in npz.files:
        names = [str(x) for x in np.asarray(npz["snapshot_model_names"]).tolist()]
    else:
        names = [f"idx{i}" for i in range(int(trees.shape[1]))]

    # Determine expected K from keyframe_indices if present
    K = None
    if "keyframe_indices" in npz.files:
        K = int(np.asarray(npz["keyframe_indices"]).ravel().shape[0])

    R = int(len(names))

    # Your recorders store (K, R). But handle accidental transpose robustly.
    if K is not None:
        if trees.shape == (K, R):
            trees_kR = trees
        elif trees.shape == (R, K):
            trees_kR = trees.T
        else:
            # if square ambiguous, prefer stored layout as-is
            trees_kR = trees
    else:
        trees_kR = trees

    # Keyframe row 0 should correspond to the snapshot at keyframe_indices[0].
    trees_for_kf0 = list(trees_kR[0, :].tolist())

    # Sanity check bytes
    for i, b in enumerate(trees_for_kf0):
        if not isinstance(b, (bytes, bytearray)) or len(b) < 8:
            raise RuntimeError(f"Bad snapshot bytes at col={i} name={names[i]} type={type(b)} len={len(b) if isinstance(b,(bytes,bytearray)) else 'NA'}")

    if len(trees_for_kf0) != R:
        raise RuntimeError(f"Snapshot tree count mismatch: got {len(trees_for_kf0)}, expected {R}")

    return trees_for_kf0, names, fmt


def restore_snapshot_via_pyrep(task, env, trees_for_kf0, q0, g0, settle_steps=10, verbose=False, names=None, fmt=""):
    """
    Restore by applying configuration trees via env._pyrep.set_configuration_tree(tree_bytes) in order.
    This works even when object wrappers don't expose set_configuration_tree().
    """
    pr = get_pyrep(env, task)

    # Apply all trees
    for idx, tree in enumerate(list(trees_for_kf0)):
        pr.set_configuration_tree(tree)

    scene = getattr(task, "_scene", None)
    if scene is None:
        raise RuntimeError("task._scene not available; cannot set robot state after restore.")

    set_robot_state(scene, q0, g0)
    task.step(np.zeros((8,), dtype=np.float32))
    settle(task, g0, settle_steps)

    if verbose:
        n = len(trees_for_kf0)
        print(f"[restore-debug] fmt={fmt} applied_trees={n} first_name={names[0] if names else 'NA'} last_name={names[-1] if names else 'NA'}")


def servo_to_q(task, q_des, g_cmd, kp=6.0, vmax=1.5, tol=0.02, max_steps=25):
    q_des = as_f32(q_des).ravel()
    tol = float(tol)

    for _ in range(int(max_steps)):
        obs = task.get_observation()
        q_now = as_f32(obs.joint_positions).ravel()
        err = q_des - q_now
        if np.max(np.abs(err)) <= tol:
            return True, obs

        v = np.clip(float(kp) * err, -float(vmax), float(vmax)).astype(np.float32)
        a = np.concatenate([v, np.array([1.0 if g_cmd > 0.5 else 0.0], dtype=np.float32)], axis=0)
        obs, _, _ = task.step(a)

    try:
        obs = task.get_observation()
    except Exception:
        obs = None
    return False, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--headless", action="store_true", default=False)
    ap.add_argument("--repeats", type=int, default=1)

    ap.add_argument("--kp", type=float, default=6.0)
    ap.add_argument("--vmax", type=float, default=1.5)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--max_steps_per_frame", type=int, default=25)

    ap.add_argument("--start_settle_steps", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=0.0)

    ap.add_argument("--no_snapshot_restore", action="store_true")
    ap.add_argument("--disable_rendering", action="store_true", default=False)
    ap.add_argument("--verbose_restore", action="store_true", default=False)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    task_name = str(d["task"][0])
    variation = int(d["variation"][0])

    q_ref = as_f32(d["joint_positions"]).reshape(-1, 7)
    a_qpos = as_f32(d["action_qpos"]).reshape(-1, 8)
    T = int(q_ref.shape[0])

    invert_gripper = bool(int(d.get("invert_gripper", np.array([0], dtype=np.int32))[0]))

    def g_from_action(x):
        g = float(x)
        if invert_gripper:
            g = 1.0 - g
        return 1.0 if g >= 0.5 else 0.0

    if not hasattr(rlbench_tasks, task_name):
        raise ValueError(f"Unknown RLBench task '{task_name}' in file {args.npz}")

    # Load snapshot trees (kf0)
    trees_for_kf0, snap_names, snap_fmt = load_initial_snapshot_trees(d)
    has_snap = trees_for_kf0 is not None and (not args.no_snapshot_restore)

    obs_config = make_obs_config_no_cameras()
    env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=bool(args.headless),
    )
    env.launch()

    try:
        if args.disable_rendering:
            try:
                env._pyrep.set_rendering(False)
            except Exception:
                pass

        task_cls = getattr(rlbench_tasks, task_name)
        task = env.get_task(task_cls)
        task.set_variation(variation)

        # One reset to load scene/scripts
        task.reset()

        for r in range(int(args.repeats)):
            print(f"\n[replay] {os.path.basename(args.npz)}  repeat {r+1}/{args.repeats}")

            if has_snap:
                g0 = g_from_action(a_qpos[0, 7]) if a_qpos.shape[0] > 0 else 1.0
                restore_snapshot_via_pyrep(
                    task, env,
                    trees_for_kf0=trees_for_kf0,
                    q0=q_ref[0],
                    g0=g0,
                    settle_steps=args.start_settle_steps,
                    verbose=args.verbose_restore,
                    names=snap_names,
                    fmt=snap_fmt,
                )
                print(f"[restore] snapshot applied via pyrep.set_configuration_tree ({snap_fmt})")
            else:
                task.reset()
                print("[restore] using task.reset() only")

            ok_all = True
            for t in range(T - 1):
                q_des = q_ref[t + 1]
                g_cmd = g_from_action(a_qpos[t, 7])

                ok, _ = servo_to_q(
                    task,
                    q_des=q_des,
                    g_cmd=g_cmd,
                    kp=args.kp,
                    vmax=args.vmax,
                    tol=args.tol,
                    max_steps=args.max_steps_per_frame,
                )
                if not ok:
                    ok_all = False
                    print(f"[warn] servo timeout at t={t}/{T-2}")
                    break

                if args.sleep > 0:
                    time.sleep(float(args.sleep))

            succ = task_success(task)
            obs = task.get_observation()
            q_end = as_f32(obs.joint_positions).ravel()
            print(f"[done] servo_ok={ok_all}  success={succ}  max|q-q_ref_end|={float(np.max(np.abs(q_end - q_ref[-1]))):.4f}")

            if not args.headless:
                time.sleep(0.5)

    finally:
        try:
            if args.disable_rendering:
                env._pyrep.set_rendering(True)
        except Exception:
            pass
        env.shutdown()


if __name__ == "__main__":
    main()