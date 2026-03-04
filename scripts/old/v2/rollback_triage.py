#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

from z_extractor import load_zspec, ZExtractor, z_distance


def as_f32(x):
    return np.asarray(x, dtype=np.float32)


# ---------------------------
# Keyframe snapshot utilities
# ---------------------------

def find_keyframe_indices(npz) -> np.ndarray:
    preferred = [
        "keyframe_indices",
        "snapshot_keyframe_idxs",
        "snapshot_keyframe_idx",
        "keyframe_idxs",
        "kf_indices",
        "kf_idxs",
    ]
    for k in preferred:
        if k in npz.files:
            arr = np.asarray(npz[k]).astype(np.int64).ravel()
            if arr.size >= 1:
                return arr

    for k in npz.files:
        lk = k.lower()
        if "keyframe" in lk and "idx" in lk:
            arr = np.asarray(npz[k]).astype(np.int64).ravel()
            if arr.size >= 1:
                return arr
    return np.array([], dtype=np.int64)


def get_snapshot_trees_and_kf(npz) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns standardized:
      trees: (num_models, num_keyframes) object array of bytes
      kf:    (num_keyframes,) aligned with trees columns

    Handles both layouts:
      (num_models, num_keyframes) OR (num_keyframes, num_models)
    """
    trees = None
    for k in ["snapshot_keyframe_trees", "keyframe_trees", "snapshot_trees"]:
        if k in npz.files:
            trees = np.asarray(npz[k], dtype=object)
            break
    if trees is None:
        raise RuntimeError("No keyframe snapshot trees found in demo npz.")
    if trees.ndim != 2:
        raise RuntimeError(f"Expected 2D snapshot trees, got shape {trees.shape}")

    n_models = None
    if "snapshot_model_names" in npz.files:
        n_models = int(np.asarray(npz["snapshot_model_names"]).shape[0])

    kf = find_keyframe_indices(npz)

    # orient by n_models when possible
    if n_models is not None:
        if trees.shape[0] == n_models:
            pass
        elif trees.shape[1] == n_models:
            trees = trees.T

    # align kf length to columns
    if kf.size == 0:
        kf = np.arange(trees.shape[1], dtype=np.int64)
    else:
        if kf.size == trees.shape[0] and kf.size != trees.shape[1]:
            trees = trees.T
        if kf.size != trees.shape[1]:
            # best effort fallback
            if kf.size > trees.shape[1]:
                kf = kf[:trees.shape[1]]
            else:
                kf = np.arange(trees.shape[1], dtype=np.int64)

    if kf.size != trees.shape[1]:
        raise RuntimeError(f"Cannot align keyframes: kf.size={kf.size}, trees.shape={trees.shape}")

    return trees, kf.astype(np.int64)


# ---------------------------
# Speed: disable cameras
# ---------------------------

def make_obs_config_no_cameras():
    oc = ObservationConfig()
    try:
        oc.set_all(False)
    except Exception:
        pass

    for attr in ["joint_positions", "gripper_open", "gripper_pose"]:
        if hasattr(oc, attr):
            setattr(oc, attr, True)

    for cam_attr in ["front_camera", "wrist_camera", "overhead_camera",
                     "left_shoulder_camera", "right_shoulder_camera"]:
        cam = getattr(oc, cam_attr, None)
        if cam is not None:
            try:
                cam.set_all(False)
            except Exception:
                for f in ["rgb", "depth", "point_cloud", "mask"]:
                    if hasattr(cam, f):
                        setattr(cam, f, False)
    return oc


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


# ---------------------------
# Restore + controller helpers
# ---------------------------

def settle(task, g_cmd: float, steps: int) -> Optional[object]:
    action = np.zeros((8,), dtype=np.float32)
    action[-1] = 1.0 if g_cmd > 0.5 else 0.0
    obs = None
    for _ in range(int(steps)):
        obs, _, _ = task.step(action)
    return obs


def set_robot_state(scene, q: np.ndarray, g: float):
    """Force actual joints + targets to prevent drift after restore."""
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


def restore_snapshot(task, env, trees_1d, q: np.ndarray, g: float,
                     settle_steps: int = 10):
    pr = get_pyrep(env, task)
    task.reset()

    for tree in trees_1d:
        pr.set_configuration_tree(tree)

    scene = getattr(task, "_scene", None)
    if scene is None:
        raise RuntimeError("task._scene not available; cannot set robot state.")

    set_robot_state(scene, q, g)

    # step once to commit state
    obs, _, _ = task.step(np.zeros((8,), dtype=np.float32))
    obs = settle(task, g, settle_steps) or obs
    return obs


def step_towards_q(task, obs, q_des: np.ndarray, g_cmd: float, kp: float, vmax: float):
    q_des = as_f32(q_des).ravel()
    q_now = as_f32(obs.joint_positions).ravel()
    vel = np.clip(kp * (q_des - q_now), -vmax, vmax).astype(np.float32)
    action = np.concatenate([vel, np.asarray([1.0 if g_cmd > 0.5 else 0.0], dtype=np.float32)], axis=0)
    obs, _, _ = task.step(action)
    return obs


def follow_ref(task, obs, q_ref, g_ref, start_idx: int, end_idx: int,
               kp: float, vmax: float, substeps: int):
    """
    Follow q_ref from start_idx -> end_idx inclusive (forward or backward).
    """
    steps_total = 0
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    step = 1 if end_idx >= start_idx else -1

    for t in range(start_idx + step, end_idx + step, step):
        q_des = q_ref[t]
        g_cmd = float(g_ref[t])
        for _ in range(int(substeps)):
            obs = step_towards_q(task, obs, q_des, g_cmd, kp, vmax)
            steps_total += 1
    return obs, steps_total


# ---------------------------
# Hybrid triage: full reverse first, then segments
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_prep_npz", required=True)
    ap.add_argument("--zspec_json", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--headless", action="store_true", default=False)

    ap.add_argument("--disable_rendering", action="store_true", default=True)

    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--vmax", type=float, default=0.6)
    ap.add_argument("--substeps_per_ref", type=int, default=4)

    ap.add_argument("--start_settle_steps", type=int, default=10)
    ap.add_argument("--end_settle_steps", type=int, default=10)

    ap.add_argument("--z_tol", type=float, default=0.10)
    ap.add_argument("--w_quat", type=float, default=0.2)
    ap.add_argument("--w_grip", type=float, default=0.05)

    # Only skip segments if full reversal success_rate >= this value
    ap.add_argument("--full_ok_rate", type=float, default=1.0)

    args = ap.parse_args()

    d = np.load(args.demo_prep_npz, allow_pickle=True)
    task_name = str(d["task"][0])
    variation = int(d["variation"][0])

    q_ref = as_f32(d["joint_positions"]).reshape(-1, 7)
    g_ref = as_f32(d["gripper_open"]).reshape(-1)
    T = q_ref.shape[0]

    keyframe_trees_2d, kf = get_snapshot_trees_and_kf(d)

    # Start snapshot = earliest recorded keyframe index (aligned to trees columns)
    j0 = int(np.argmin(kf))
    start_idx = int(kf[j0])
    start_trees = keyframe_trees_2d[:, j0]

    # Keyframes for segments: sorted unique within [start_idx, T-1]
    kf_sorted = sorted({int(x) for x in kf.tolist() if start_idx <= int(x) <= (T - 1)})
    if len(kf_sorted) < 2:
        # fallback: make 10 points
        kf_sorted = sorted(set(np.round(np.linspace(start_idx, T - 1, 10)).astype(int).tolist()))

    if args.out_json is None:
        base, _ = os.path.splitext(args.demo_prep_npz)
        args.out_json = base + "_triage_hybrid.json"

    zspec = load_zspec(args.zspec_json)
    zext = ZExtractor(zspec)

    action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
    obs_config = make_obs_config_no_cameras()
    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=args.headless)
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

        # -------------------------
        # Phase A: full reversal test
        # -------------------------
        full_trials = []
        for rep in range(int(args.repeats)):
            # restore start
            obs = restore_snapshot(
                task, env, start_trees,
                q=q_ref[start_idx], g=float(g_ref[start_idx]),
                settle_steps=args.start_settle_steps
            )

            z_start = zext.extract(obs, task)

            # forward to end
            obs, _ = follow_ref(task, obs, q_ref, g_ref, start_idx=start_idx, end_idx=T - 1,
                                kp=args.kp, vmax=args.vmax, substeps=args.substeps_per_ref)
            settle(task, float(g_ref[-1]), args.end_settle_steps)

            # reverse end -> start (same episode)
            obs_back, steps_total = follow_ref(task, obs, q_ref, g_ref, start_idx=T - 1, end_idx=start_idx,
                                               kp=args.kp, vmax=args.vmax, substeps=args.substeps_per_ref)

            z_back = zext.extract(obs_back, task)
            dz = z_distance(z_back, z_start, zspec.k_shapes, zspec.k_joints,
                            w_quat=float(args.w_quat), w_grip=float(args.w_grip))
            # print(z_start)
            # print(z_back)
            print(dz)
            print()
            success = bool(dz <= float(args.z_tol))

            full_trials.append({
                "rep": rep,
                "steps_total": int(steps_total),
                "z_dist": float(dz),
                "success": bool(success),
            })

        full_success_rate = float(np.mean([1.0 if t["success"] else 0.0 for t in full_trials]))
        print(f"[full] success_rate={full_success_rate:.2f} | z_dist_mean={np.mean([t['z_dist'] for t in full_trials]):.4f}")

        segment_results = []
        suggested_split = None

        # -------------------------
        # Phase B: only if full reversal is insufficient, do segment checks
        # -------------------------
        if full_success_rate < float(args.full_ok_rate):
            # For each adjacent segment (k_prev -> k_curr), test rollback (k_curr -> k_prev)
            for seg_i in range(1, len(kf_sorted)):
                k_prev = int(kf_sorted[seg_i - 1])
                k_curr = int(kf_sorted[seg_i])

                trials = []
                for rep in range(int(args.repeats)):
                    # restore start
                    obs = restore_snapshot(
                        task, env, start_trees,
                        q=q_ref[start_idx], g=float(g_ref[start_idx]),
                        settle_steps=args.start_settle_steps
                    )

                    # forward to k_prev, capture z_prev
                    obs, _ = follow_ref(task, obs, q_ref, g_ref, start_idx=start_idx, end_idx=k_prev,
                                        kp=args.kp, vmax=args.vmax, substeps=args.substeps_per_ref)
                    z_prev = zext.extract(obs, task)

                    # forward k_prev -> k_curr
                    obs, _ = follow_ref(task, obs, q_ref, g_ref, start_idx=k_prev, end_idx=k_curr,
                                        kp=args.kp, vmax=args.vmax, substeps=args.substeps_per_ref)

                    # reverse k_curr -> k_prev (local rollback)
                    obs_back, steps_total = follow_ref(task, obs, q_ref, g_ref, start_idx=k_curr, end_idx=k_prev,
                                                       kp=args.kp, vmax=args.vmax, substeps=args.substeps_per_ref)

                    z_back = zext.extract(obs_back, task)
                    dz = z_distance(z_back, z_prev, zspec.k_shapes, zspec.k_joints,
                                    w_quat=float(args.w_quat), w_grip=float(args.w_grip))
                    success = bool(dz <= float(args.z_tol))

                    trials.append({
                        "rep": rep,
                        "steps_total": int(steps_total),
                        "z_dist": float(dz),
                        "success": bool(success),
                    })

                succ_rate = float(np.mean([1.0 if t["success"] else 0.0 for t in trials]))
                segment_results.append({
                    "k_prev": k_prev,
                    "k_curr": k_curr,
                    "success_rate": succ_rate,
                    "trials": trials,
                })

                print(f"[seg] {k_prev:4d}->{k_curr:4d} rollback | success_rate={succ_rate:.2f} | "
                      f"z_dist_mean={np.mean([t['z_dist'] for t in trials]):.4f}")

            # Suggest split at the k_prev of the first failing segment
            for r in segment_results:
                if float(r["success_rate"]) < 0.5:
                    suggested_split = int(r["k_prev"])
                    break

        out = {
            "schema": "plan_a.rollback_triage.hybrid.v1",
            "task": task_name,
            "variation": variation,
            "demo_npz": str(args.demo_prep_npz),
            "zspec_json": str(args.zspec_json),
            "T": int(T),
            "start_idx": int(start_idx),
            "keyframes_raw": [int(x) for x in kf.tolist()],
            "keyframes_sorted": [int(x) for x in kf_sorted],
            "params": {
                "repeats": int(args.repeats),
                "kp": float(args.kp),
                "vmax": float(args.vmax),
                "substeps_per_ref": int(args.substeps_per_ref),
                "start_settle_steps": int(args.start_settle_steps),
                "end_settle_steps": int(args.end_settle_steps),
                "z_tol": float(args.z_tol),
                "w_quat": float(args.w_quat),
                "w_grip": float(args.w_grip),
                "full_ok_rate": float(args.full_ok_rate),
                "disable_rendering": bool(args.disable_rendering),
                "k_shapes": int(zspec.k_shapes),
                "k_joints": int(zspec.k_joints),
            },
            "full_reversal": {
                "success_rate": float(full_success_rate),
                "trials": full_trials,
            },
            "segments_ran": bool(full_success_rate < float(args.full_ok_rate)),
            "segment_results": segment_results,
            "suggested_split_idx": suggested_split,
        }

        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[ok] wrote {args.out_json}")

    finally:
        try:
            if args.disable_rendering:
                env._pyrep.set_rendering(True)
        except Exception:
            pass
        env.shutdown()


if __name__ == "__main__":
    main()