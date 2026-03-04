#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Tuple, Dict, Any

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


def set_rendering(pr, enabled: bool):
    try:
        if pr is None:
            return
        if hasattr(pr, "set_rendering"):
            pr.set_rendering(enabled)
        elif hasattr(pr, "set_rendering_enabled"):
            pr.set_rendering_enabled(enabled)
    except Exception:
        pass


# ---------------------------
# Keyframe snapshot utilities (robust for KxR roots)
# ---------------------------

def find_keyframe_indices(npz) -> np.ndarray:
    preferred = [
        "keyframe_indices",
        "snapshot_keyframe_indices",
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


def get_keyframe_rows(npz) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      trees_kR: (K, R) object array of bytes (K keyframes, R roots/models)
      kf_idx:   (K,) keyframe indices (timesteps)
      root_names: list[str] length R

    Works for both snapshot formats:
      - bytes_v1 (models): snapshot_model_names, trees often (K, R) or (R, K)
      - bytes_v2_roots: snapshot_root_names, trees (K, R)
    """
    if "snapshot_keyframe_trees" not in npz.files:
        raise RuntimeError("snapshot_keyframe_trees missing in npz.")

    trees = np.asarray(npz["snapshot_keyframe_trees"], dtype=object)
    if trees.ndim != 2:
        raise RuntimeError(f"snapshot_keyframe_trees must be 2D, got {trees.shape}")

    if "snapshot_root_names" in npz.files:
        root_names = [str(x) for x in np.asarray(npz["snapshot_root_names"]).tolist()]
    elif "snapshot_model_names" in npz.files:
        root_names = [str(x) for x in np.asarray(npz["snapshot_model_names"]).tolist()]
    else:
        root_names = [f"idx{i}" for i in range(int(trees.shape[1]))]

    R = int(len(root_names))

    kf = find_keyframe_indices(npz)
    K = int(kf.size) if kf.size > 0 else None

    # Orient to (K, R) when possible
    if K is not None:
        if trees.shape == (K, R):
            trees_kR = trees
        elif trees.shape == (R, K):
            trees_kR = trees.T
        else:
            # best effort: prefer dimension that matches R as columns
            trees_kR = trees if trees.shape[1] == R else trees.T
    else:
        # assume rows are keyframes
        trees_kR = trees if trees.shape[1] == R else trees.T
        kf = np.arange(trees_kR.shape[0], dtype=np.int64)

    if trees_kR.shape[1] != R:
        raise RuntimeError(f"Cannot align snapshot trees: trees_kR.shape={trees_kR.shape}, R={R}")

    if kf.size != trees_kR.shape[0]:
        # align lengths (best effort)
        kf = np.arange(trees_kR.shape[0], dtype=np.int64)

    return trees_kR, kf.astype(np.int64), root_names


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

    for cam_attr in [
        "front_camera", "wrist_camera", "overhead_camera",
        "left_shoulder_camera", "right_shoulder_camera"
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
# Restore + controller helpers (keep the known-good behavior)
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


def restore_snapshot(task, env, trees_1d: List[bytes], q: np.ndarray, g: float,
                     settle_steps: int = 10):
    """
    Keep the behavior that you reported as reliable:
      - task.reset() per restore
      - apply configuration trees
      - force robot q and gripper state
      - settle

    IMPORTANT CHANGE vs your old code:
      - trees_1d must be the correct list of bytes for the current snapshot format.
        For v2 roots stored as (K,R), you must pass a list from trees_kR[row, :].
    """
    pr = get_pyrep(env, task)

    # Reset first (this was the stable behavior in your working script)
    task.reset()

    # Apply trees in recorded order
    for tree in list(trees_1d):
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
    The exact dt-free "substeps per reference" servo from your previously working script.
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
# Main (full trial first, segments only if needed)
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_prep_npz", required=True)
    ap.add_argument("--zspec_json", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--headless", action="store_true", default=False)

    ap.add_argument("--disable_rendering", action="store_true", default=True)

    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--kp", type=float, default=4.0)
    ap.add_argument("--vmax", type=float, default=0.5)
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
    T = int(q_ref.shape[0])

    # Load keyframe trees robustly (K,R)
    trees_kR, kf, root_names = get_keyframe_rows(d)

    # Start snapshot = earliest keyframe row
    j0 = int(np.argmin(kf))
    start_idx = int(kf[j0])
    start_trees = list(trees_kR[j0, :].tolist())  # list[bytes] length R

    # Segment keyframes (indices in trajectory)
    kf_sorted = sorted({int(x) for x in kf.tolist() if start_idx <= int(x) <= (T - 1)})
    if len(kf_sorted) < 2:
        kf_sorted = sorted(set(np.round(np.linspace(start_idx, T - 1, 10)).astype(int).tolist()))

    if args.out_json is None:
        base, _ = os.path.splitext(args.demo_prep_npz)
        args.out_json = base + "_triage_hybrid.json"

    zspec = load_zspec(args.zspec_json)
    zext = ZExtractor(zspec)

    env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=make_obs_config_no_cameras(),
        headless=bool(args.headless),
    )
    env.launch()

    try:
        if args.disable_rendering:
            set_rendering(getattr(env, "_pyrep", None), False)

        task_cls = getattr(rlbench_tasks, task_name)
        task = env.get_task(task_cls)
        task.set_variation(variation)

        # -------------------------
        # Phase A: full forward/backward loop first
        # -------------------------
        full_trials: List[Dict[str, Any]] = []
        for rep in range(int(args.repeats)):
            # restore start snapshot (correct trees list for v2)
            obs = restore_snapshot(
                task, env, start_trees,
                q=q_ref[start_idx], g=float(g_ref[start_idx]),
                settle_steps=int(args.start_settle_steps),
            )
            z_start = zext.extract(obs, task)

            # forward to end
            obs, fwd_steps = follow_ref(
                task, obs, q_ref, g_ref,
                start_idx=start_idx, end_idx=T - 1,
                kp=float(args.kp), vmax=float(args.vmax),
                substeps=int(args.substeps_per_ref),
            )
            settle(task, float(g_ref[-1]), int(args.end_settle_steps))

            # reverse end -> start (same episode)
            obs_back, bwd_steps = follow_ref(
                task, obs, q_ref, g_ref,
                start_idx=T - 1, end_idx=start_idx,
                kp=float(args.kp), vmax=float(args.vmax),
                substeps=int(args.substeps_per_ref),
            )

            z_back = zext.extract(obs_back, task)
            dz = z_distance(
                z_back, z_start,
                zspec.k_shapes, zspec.k_joints,
                w_quat=float(args.w_quat), w_grip=float(args.w_grip),
            )
            success = bool(dz <= float(args.z_tol))

            full_trials.append({
                "rep": int(rep),
                "steps_forward": int(fwd_steps),
                "steps_backward": int(bwd_steps),
                "z_dist": float(dz),
                "success": bool(success),
            })

            print(f"[full][{rep}] dz={dz:.4f} success={success}")

        full_success_rate = float(np.mean([1.0 if t["success"] else 0.0 for t in full_trials]))
        print(f"[full] success_rate={full_success_rate:.2f} | z_dist_mean={np.mean([t['z_dist'] for t in full_trials]):.4f}")

        # -------------------------
        # Phase B: only if full reversal insufficient, test segments
        # -------------------------
        segment_results: List[Dict[str, Any]] = []
        suggested_split: Optional[int] = None

        if full_success_rate < float(args.full_ok_rate):
            for seg_i in range(1, len(kf_sorted)):
                k_prev = int(kf_sorted[seg_i - 1])
                k_curr = int(kf_sorted[seg_i])

                trials = []
                for rep in range(int(args.repeats)):
                    # restore start
                    obs = restore_snapshot(
                        task, env, start_trees,
                        q=q_ref[start_idx], g=float(g_ref[start_idx]),
                        settle_steps=int(args.start_settle_steps),
                    )

                    # forward to k_prev
                    obs, s1 = follow_ref(
                        task, obs, q_ref, g_ref,
                        start_idx=start_idx, end_idx=k_prev,
                        kp=float(args.kp), vmax=float(args.vmax),
                        substeps=int(args.substeps_per_ref),
                    )
                    z_prev = zext.extract(obs, task)

                    # forward k_prev -> k_curr
                    obs, s2 = follow_ref(
                        task, obs, q_ref, g_ref,
                        start_idx=k_prev, end_idx=k_curr,
                        kp=float(args.kp), vmax=float(args.vmax),
                        substeps=int(args.substeps_per_ref),
                    )

                    # reverse k_curr -> k_prev
                    obs_back, s3 = follow_ref(
                        task, obs, q_ref, g_ref,
                        start_idx=k_curr, end_idx=k_prev,
                        kp=float(args.kp), vmax=float(args.vmax),
                        substeps=int(args.substeps_per_ref),
                    )

                    z_back = zext.extract(obs_back, task)
                    dz = z_distance(
                        z_back, z_prev,
                        zspec.k_shapes, zspec.k_joints,
                        w_quat=float(args.w_quat), w_grip=float(args.w_grip),
                    )
                    succ = bool(dz <= float(args.z_tol))

                    trials.append({
                        "rep": int(rep),
                        "steps_total": int(s1 + s2 + s3),
                        "z_dist": float(dz),
                        "success": bool(succ),
                    })

                succ_rate = float(np.mean([1.0 if t["success"] else 0.0 for t in trials]))
                segment_results.append({
                    "k_prev": int(k_prev),
                    "k_curr": int(k_curr),
                    "success_rate": float(succ_rate),
                    "trials": trials,
                })

                print(f"[seg] {k_prev:4d}->{k_curr:4d} rollback | success_rate={succ_rate:.2f} | "
                      f"z_dist_mean={np.mean([t['z_dist'] for t in trials]):.4f}")

            for r in segment_results:
                if float(r["success_rate"]) < 0.5:
                    suggested_split = int(r["k_prev"])
                    break

        out = {
            "schema": "plan_a.rollback_triage.hybrid.v2",
            "task": task_name,
            "variation": variation,
            "demo_npz": str(args.demo_prep_npz),
            "zspec_json": str(args.zspec_json),
            "T": int(T),
            "start_idx": int(start_idx),
            "keyframes_raw": [int(x) for x in kf.tolist()],
            "keyframes_sorted": [int(x) for x in kf_sorted],
            "snapshot_roots_count": int(len(root_names)),
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
                set_rendering(getattr(env, "_pyrep", None), True)
        except Exception:
            pass
        env.shutdown()


if __name__ == "__main__":
    main()