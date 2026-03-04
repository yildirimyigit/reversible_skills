#!/usr/bin/env python3
"""
replay_forward_from_npz.py

Replay a recorded RLBench demo saved by record_live_demo_with_actions.py

Fixes PlugChargerInPowerSupply snapshot restore inconsistency by restoring snapshots
BY MODEL NAME (Model(name).set_configuration_tree(bytes)), which is robust when RLBench
recreates objects/handles on reset. We skip names that are not models (e.g., DefaultLights).

Replay:
- dt-free joint servo to recorded q[t+1]
- gripper command from action_qpos[t,7] (open=1 close=0), with optional invert_gripper
- GUI by default (headless=False)
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

# PyRep Model wrapper (name -> current handle)
try:
    from pyrep.objects.model import Model
except Exception:
    Model = None


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


def normalize_snapshot_trees(npz):
    """
    Returns:
      model_names: (M,) list[str]
      trees_km:    (K, M) object array of bytes  (exactly as stored by your recorder)
      kf_idx:      (K,) keyframe indices
    """
    if "snapshot_keyframe_trees" not in npz.files:
        raise RuntimeError("No snapshot_keyframe_trees in npz.")
    if "snapshot_model_names" not in npz.files:
        raise RuntimeError("No snapshot_model_names in npz.")
    if "keyframe_indices" not in npz.files:
        raise RuntimeError("No keyframe_indices in npz.")

    model_names = [str(x) for x in np.asarray(npz["snapshot_model_names"]).tolist()]
    trees = np.asarray(npz["snapshot_keyframe_trees"], dtype=object)
    kf_idx = np.asarray(npz["keyframe_indices"]).astype(np.int64).ravel()

    if trees.ndim != 2:
        raise RuntimeError(f"snapshot_keyframe_trees must be 2D, got {trees.shape}")

    K = int(kf_idx.shape[0])
    M = int(len(model_names))

    # Your recorder writes kf_mat[r, :] = row, so trees is (K, M).
    # Enforce that, but handle accidental transposes safely.
    if trees.shape == (K, M):
        trees_km = trees
    elif trees.shape == (M, K):
        trees_km = trees.T
    else:
        # If square, prefer recorder convention (K,M) where K=len(keyframe_indices)
        if trees.shape[0] == trees.shape[1] == K:
            trees_km = trees
        else:
            raise RuntimeError(f"Unexpected snapshot shape {trees.shape}; expected (K,M)=({K},{M}) or (M,K)=({M},{K})")

    return model_names, trees_km, kf_idx


def restore_snapshot_by_model_names(task, env, model_names, trees_for_one_kf, q0, g0,
                                   settle_steps=10, verbose=False):
    """
    Name-based snapshot restore WITHOUT pyrep.objects.model.Model.

    Strategy:
      1) Use env._pyrep.get_objects_in_tree(..., first_generation_only=True)
      2) Filter objects that are models and have set_configuration_tree()
      3) Map name -> model_object
      4) For each (name, tree_bytes): if model exists and supports set_configuration_tree, apply it
         otherwise skip (e.g. DefaultLights)
      5) Force robot q/g, then settle

    This fixes PlugCharger cases where handles change across resets by resolving models by name
    in the current scene.
    """
    pr = get_pyrep(env, task)

    # Build a name->model map from CURRENT scene
    roots = pr.get_objects_in_tree(root_object=None, first_generation_only=True)
    models = []
    for o in roots:
        try:
            if o.is_model():
                models.append(o)
        except Exception:
            pass

    name_to_model = {}
    for m in models:
        try:
            n = m.get_name()
        except Exception:
            continue
        # only keep models that can actually accept a config tree
        if hasattr(m, "set_configuration_tree"):
            name_to_model[n] = m

    restored = 0
    skipped = 0
    failed = 0

    for name, tree in zip(list(model_names), list(trees_for_one_kf)):
        name = str(name)
        if not isinstance(tree, (bytes, bytearray)) or len(tree) < 8:
            failed += 1
            if verbose:
                print(f"[restore] bad bytes: {name} len={len(tree) if isinstance(tree,(bytes,bytearray)) else 'NA'}")
            continue

        m = name_to_model.get(name, None)
        if m is None:
            skipped += 1
            if verbose:
                print(f"[restore] skip (not a restorable model in this build): {name}")
            continue

        try:
            m.set_configuration_tree(tree)
            restored += 1
        except Exception:
            failed += 1
            if verbose:
                print(f"[restore] failed set_configuration_tree: {name}")

    if verbose:
        print(f"[restore] restored={restored} skipped={skipped} failed={failed}")

    scene = getattr(task, "_scene", None)
    if scene is None:
        raise RuntimeError("task._scene not available; cannot set robot state after restore.")

    set_robot_state(scene, q0, g0)

    # Commit + settle
    task.step(np.zeros((8,), dtype=np.float32))
    settle(task, g0, settle_steps)


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

    has_snap = (
        ("snapshot_storage" in d.files) and (str(d["snapshot_storage"][0]) == "bytes_v1")
        and int(d.get("snapshot_captured", np.array([0]))[0]) == 1
        and int(d.get("snapshot_failed", np.array([0]))[0]) == 0
        and ("snapshot_keyframe_trees" in d.files)
        and ("snapshot_model_names" in d.files)
        and ("keyframe_indices" in d.files)
    )

    if not hasattr(rlbench_tasks, task_name):
        raise ValueError(f"Unknown RLBench task '{task_name}' in file {args.npz}")

    # Snapshot decoded (K,M)
    model_names = None
    trees_km = None
    if has_snap:
        model_names, trees_km, kf_idx = normalize_snapshot_trees(d)

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

        # One reset to load scene / scripts
        task.reset()

        for r in range(int(args.repeats)):
            print(f"\n[replay] {os.path.basename(args.npz)}  repeat {r+1}/{args.repeats}")

            if (not args.no_snapshot_restore) and has_snap:
                g0 = g_from_action(a_qpos[0, 7]) if a_qpos.shape[0] > 0 else 1.0

                # Keyframe row 0 corresponds to t=0 in your recorder (since keyframe_indices is sorted and includes 0).
                trees_for_kf0 = trees_km[0, :]  # shape (M,)

                restore_snapshot_by_model_names(
                    task, env,
                    model_names=model_names,
                    trees_for_one_kf=trees_for_kf0,
                    q0=q_ref[0],
                    g0=g0,
                    settle_steps=args.start_settle_steps,
                    verbose=args.verbose_restore,
                )
                print("[restore] snapshot applied via name->model.set_configuration_tree")
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