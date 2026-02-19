#!/usr/bin/env python3
"""
rollback_triage.py

PLAN-A Step 2/3 component:
- For each segment between consecutive keyframes:
    restore snapshot at kf_curr
    attempt rollback to kf_prev using reverse "delta-q servo" controller
    measure success rate + distances in compact state space z

Outputs a JSON file following your plan_a.rollback_triage.v1 schema.

Key design choice (your option 2):
 - Replay FORWARD using recorded actions (including the correct Discrete gripper command).
 - Rollback uses joint-position servo in reverse time (no inverse actions):
       v = clip(kp * (q_desired - q_now), -vmax, vmax)

This avoids needing control_dt (often nan) and is robust across builds.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Any, Tuple

import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from state_utils import select_active_lowdim_indices, compact_state_from_npz, compact_state_from_obs, l2_dist

from snapshot_utils import (
    load_demo_npz,
    get_pyrep,
    get_observation_strict,
    restore_keyframe,
    estimate_snapshot_offset,
)


def _stats(xs: List[float]) -> Dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "dist_mean": float(np.mean(arr)),
        "dist_median": float(np.median(arr)),
        "dist_p90": float(np.percentile(arr, 90)),
        "dist_min": float(np.min(arr)),
        "dist_max": float(np.max(arr)),
    }


def compress_keyframes_by_gap(orig_kf_ts: list[int], min_gap: int) -> list[int]:
    """
    Returns a list of ORIGINAL keyframe indices to keep.
    orig_kf_ts is the raw timestep list aligned with snapshot_keyframe_trees rows.
    """
    K = len(orig_kf_ts)
    if K == 0:
        return []
    keep = [0]
    last_t = int(orig_kf_ts[0])
    for i in range(1, K - 1):
        t = int(orig_kf_ts[i])
        if t - last_t >= int(min_gap):
            keep.append(i)
            last_t = t
    if keep[-1] != K - 1:
        keep.append(K - 1)
    return keep


def _select_active_from_arrays(
    low_prev: np.ndarray,
    low_curr: np.ndarray,
    k: int = 16,
    delta_min: float = 1e-3,
    quat_tol: float = 0.20,
) -> np.ndarray:
    """
    Same idea as select_active_lowdim_indices, but operates on two vectors
    (the reached lowdim at t_prev and t_curr in the *replay*).
    delta_min is your “floor”: dims/quat-blocks below it are ignored.
    """
    from state_utils import _is_unit_quat, _quat_diff_sign_invariant

    a = np.asarray(low_prev, dtype=np.float32).ravel()
    b = np.asarray(low_curr, dtype=np.float32).ravel()
    n = int(min(a.size, b.size))
    if n <= 0:
        return np.zeros((0,), dtype=np.int32)

    feats = []
    i = 0
    while i < n:
        if i + 4 <= n and _is_unit_quat(a[i:i+4], tol=quat_tol) and _is_unit_quat(b[i:i+4], tol=quat_tol):
            s = _quat_diff_sign_invariant(a[i:i+4], b[i:i+4])
            feats.append((float(s), [i, i+1, i+2, i+3]))
            i += 4
        else:
            s = float(abs(b[i] - a[i]))
            feats.append((s, [i]))
            i += 1

    feats.sort(key=lambda x: -x[0])

    picked = []
    for s, idxs in feats:
        if s < float(delta_min):
            break
        picked.extend(idxs)
        if len(picked) >= 4 * int(k):
            break

    picked = sorted(set(picked))
    if len(picked) > int(k):
        picked = picked[: int(k)]
    return np.asarray(picked, dtype=np.int32)


def _servo_track_to_q(task, q_des, g_cmd, kp, vmax, tol, per_waypoint_max_steps, action_noise_std=0.0):
    q_des = np.asarray(q_des, dtype=np.float32).ravel()
    for _ in range(int(per_waypoint_max_steps)):
        obs = get_observation_strict(task)
        q_now = np.asarray(obs.joint_positions, dtype=np.float32).ravel()
        err = q_des - q_now
        if float(np.linalg.norm(err)) < float(tol):
            break
        v = kp * err
        if action_noise_std > 0:
            v = v + np.random.normal(0.0, action_noise_std, size=v.shape).astype(np.float32)
        v = np.clip(v, -vmax, vmax).astype(np.float32)
        action = np.concatenate([v, np.array([g_cmd], dtype=np.float32)], axis=0)
        task.step(action)


def step_towards_q(task, q_des, g_cmd, kp: float, vmax: float):
    """
    One RLBench control step in JointVelocity mode that moves joints toward q_des.
    """
    obs = get_observation_strict(task)
    q_now = np.asarray(obs.joint_positions, dtype=np.float32).ravel()
    q_des = np.asarray(q_des, dtype=np.float32).ravel()

    err = q_des - q_now
    v = np.clip(kp * err, -vmax, vmax).astype(np.float32)
    action = np.concatenate([v, np.array([float(g_cmd)], dtype=np.float32)], axis=0)
    ret = task.step(action)
    return ret


def replay_forward_to_t_closed_loop(
    task,
    data,
    t_target: int,
    t_prev: int,
    kp: float,
    vmax: float,
    q_tol: float = 0.02,            # radians, tune
    max_micro_steps_per_t: int = 8, # tune (insertion may need >1)
):
    """
    Drive from current state to q_all[1], q_all[2], ... q_all[t_target] using
    delta-q velocity control (closed loop).
    Returns reached lowdim vectors at t_prev and t_target as encountered.
    """
    q_all = np.asarray(data["joint_positions"], dtype=np.float32)
    actions = np.asarray(data["action"], dtype=np.float32)

    low_prev = None
    low_curr = None

    # We assume current state is already restored to kf0.
    for t in range(1, int(t_target) + 1):
        q_des = q_all[t].ravel()
        g_cmd = float(actions[t-1, -1])  # action applied to enter state t

        # micro-steps until close enough or we give up
        for _ in range(int(max_micro_steps_per_t)):
            obs = get_observation_strict(task)
            q_now = np.asarray(obs.joint_positions, dtype=np.float32).ravel()
            if float(np.linalg.norm(q_des - q_now)) <= float(q_tol):
                break
            step_towards_q(task, q_des, g_cmd, kp=kp, vmax=vmax)

        # capture lowdim at the *moment we consider ourselves at timestep t*
        obs_t = get_observation_strict(task)
        if int(t) == int(t_prev) and getattr(obs_t, "task_low_dim_state", None) is not None:
            low_prev = np.asarray(obs_t.task_low_dim_state, dtype=np.float32).ravel().copy()
        if int(t) == int(t_target) and getattr(obs_t, "task_low_dim_state", None) is not None:
            low_curr = np.asarray(obs_t.task_low_dim_state, dtype=np.float32).ravel().copy()

    return low_prev, low_curr


def _controlled_settle(task, n_arm: int, g_cmd: float, steps: int):
    """
    Replace raw pyrep.step() settling with controlled settling:
    apply zero joint-velocities while holding the current discrete gripper command.
    """
    if int(steps) <= 0:
        return
    zero_v = np.zeros((int(n_arm),), dtype=np.float32)
    action = np.concatenate([zero_v, np.array([float(g_cmd)], dtype=np.float32)], axis=0)
    for _ in range(int(steps)):
        ret = task.step(action)
        # If the episode terminates during settling, stop settling.
        if isinstance(ret, (tuple, list)) and len(ret) >= 3 and bool(ret[2]):
            break


def rollback_once_deltaq_servo(
    env,
    task,
    data,
    t_curr: int,
    t_prev: int,
    settle_steps: int,
    kp: float,
    vmax: float,
    action_noise_std: float,
    tol: float = 0.01,
    per_waypoint_max_steps: int = 30,
    waypoint_stride: int = 1,
    robot_weight: float = 0.05,
    quat_tol: float = 0.20,
    active_k: int = 16,
    delta_min: float = 1e-3,
) -> float:
    """
    CLOSED-LOOP RECONSTRUCTION → ROLLBACK scoring.

    Key change vs your current version:
    - Forward reconstruction is NOT open-loop replay of recorded joint velocities.
      Instead, it tracks the recorded joint POSITIONS (q_all[t]) using delta-q velocity servo,
      while holding the recorded discrete gripper command.
    - This makes forward reconstruction robust to dt/physics-step differences and contact chaos.

    Pipeline:
      1) Restore only initial snapshot (kf=0) with NO settling (important for insertion tasks).
      2) Forward-track to t_curr using q_all (closed-loop), caching lowdim at t_prev and t_curr.
         Retry a few times if we fail to reach t_curr within joint/gripper thresholds.
      3) Rollback from reached t_curr to t_prev using your existing reverse servo.
      4) Score on lowdim(active dims from reached states) + small robot regularizer.
    """
    # ----------------------------
    # 0) Basic checks / helpers
    # ----------------------------
    q_all = np.asarray(data["joint_positions"], dtype=np.float32)
    T = int(q_all.shape[0])
    n_arm = int(q_all.shape[1])

    if int(t_prev) < 0 or int(t_curr) < 0 or int(t_prev) >= T or int(t_curr) >= T:
        return float("inf")
    if int(t_curr) <= int(t_prev):
        return float("inf")

    # Require recorded actions for discrete gripper cmd.
    if "action" not in data.files:
        raise RuntimeError(
            "Demo NPZ missing 'action'. Re-record demos with actions saved "
            "(arm joint velocities + Discrete gripper cmd)."
        )
    actions = np.asarray(data["action"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < (n_arm + 1):
        raise RuntimeError(f"Unexpected action shape {actions.shape}; expected (T, {n_arm+1}) at least.")

    def _gcmd_for_state(t: int) -> float:
        """
        Discrete gripper command to HOLD while servoing toward state at time t.
        Approx: command applied to ENTER state t, i.e., action[t-1][-1].
        """
        t = int(t)
        if actions.shape[0] <= 0:
            return 0.0
        idx = 0 if t <= 0 else (t - 1)
        idx = int(np.clip(idx, 0, actions.shape[0] - 1))
        return float(actions[idx, -1])

    def _servo_to_q_report(
        q_des: np.ndarray,
        g_cmd: float,
        kp_: float,
        vmax_: float,
        tol_: float,
        max_steps: int,
        noise_std: float = 0.0,
    ) -> bool:
        """
        Like _servo_track_to_q, but returns whether we reached tol within max_steps,
        and bails out if the episode terminates.
        """
        q_des = np.asarray(q_des, dtype=np.float32).ravel()
        for _ in range(int(max_steps)):
            obs = get_observation_strict(task)
            q_now = np.asarray(obs.joint_positions, dtype=np.float32).ravel()
            err = q_des - q_now
            if float(np.linalg.norm(err)) <= float(tol_):
                return True

            v = kp_ * err
            if float(noise_std) > 0.0:
                v = v + np.random.normal(0.0, float(noise_std), size=v.shape).astype(np.float32)
            v = np.clip(v, -float(vmax_), float(vmax_)).astype(np.float32)

            action = np.concatenate([v, np.array([float(g_cmd)], dtype=np.float32)], axis=0)
            ret = task.step(action)

            # terminated?
            if isinstance(ret, (tuple, list)) and len(ret) >= 3 and bool(ret[2]):
                return False

        # One last check
        obs = get_observation_strict(task)
        q_now = np.asarray(obs.joint_positions, dtype=np.float32).ravel()
        return float(np.linalg.norm(q_des - q_now)) <= float(tol_)

    # Forward reconstruction tuning (separate from rollback tuning)
    # Important: keep this moderate to avoid jamming during insertion.
    fwd_kp = float(min(max(kp, 2.0), 6.0))
    fwd_vmax = float(min(max(vmax, 0.2), 0.8))
    fwd_q_tol = float(max(0.02, 2.0 * float(tol)))  # looser than rollback tol usually helps
    fwd_micro_steps = int(max(2, min(6, per_waypoint_max_steps // 5)))  # small number of extra steps per demo t

    # Forward acceptance thresholds (RMSE-like)
    fwd_q_rmse_thresh = 0.03
    fwd_gp_rmse_thresh = 0.03

    from state_utils import lowdim_subset_distance  # used later for scoring

    # ----------------------------
    # 1) Forward reconstruction (closed-loop to q_all)
    # ----------------------------
    max_forward_attempts = 5
    low_prev_reached = None
    low_curr_reached = None

    for _attempt in range(int(max_forward_attempts)):
        # Restore initial snapshot ONLY.
        # Crucial: for contact-rich tasks (USB insertion), settling right after restore can introduce drift.
        _ = restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)

        low_prev_reached = None
        low_curr_reached = None

        # Cache lowdim at t=0 if needed
        obs0 = get_observation_strict(task)
        if int(t_prev) == 0 and getattr(obs0, "task_low_dim_state", None) is not None:
            low_prev_reached = np.asarray(obs0.task_low_dim_state, dtype=np.float32).ravel().copy()
        if int(t_curr) == 0 and getattr(obs0, "task_low_dim_state", None) is not None:
            low_curr_reached = np.asarray(obs0.task_low_dim_state, dtype=np.float32).ravel().copy()

        terminated = False

        # Drive through demo timesteps 1..t_curr by tracking q_all[t]
        for t in range(1, int(t_curr) + 1):
            q_des = q_all[t].ravel()
            g_cmd = _gcmd_for_state(t)

            # Allow a few micro steps to get close to q_des (robustness to dt/physics-step mismatch).
            ok = _servo_to_q_report(
                q_des=q_des,
                g_cmd=g_cmd,
                kp_=fwd_kp,
                vmax_=fwd_vmax,
                tol_=fwd_q_tol,
                max_steps=fwd_micro_steps,
                noise_std=0.0,  # do NOT inject noise during forward reconstruction
            )
            if not ok:
                terminated = True
                break

            obs_t = get_observation_strict(task)
            if int(t) == int(t_prev) and getattr(obs_t, "task_low_dim_state", None) is not None:
                low_prev_reached = np.asarray(obs_t.task_low_dim_state, dtype=np.float32).ravel().copy()
            if int(t) == int(t_curr) and getattr(obs_t, "task_low_dim_state", None) is not None:
                low_curr_reached = np.asarray(obs_t.task_low_dim_state, dtype=np.float32).ravel().copy()

        if terminated:
            continue

        # Forward acceptance check:
        # Use joint + gripper_pose closeness (more reliable than full task_low_dim_state in contact tasks).
        obs_c = get_observation_strict(task)

        q_now = np.asarray(obs_c.joint_positions, dtype=np.float32).ravel()
        q_ref = np.asarray(q_all[int(t_curr)], dtype=np.float32).ravel()
        m = int(min(q_now.size, q_ref.size))
        q_rmse = float(np.linalg.norm(q_now[:m] - q_ref[:m]) / np.sqrt(max(1, m)))

        gp_rmse = 0.0
        if getattr(obs_c, "gripper_pose", None) is not None and "gripper_pose" in data.files:
            gp_now = np.asarray(obs_c.gripper_pose, dtype=np.float32).ravel()
            gp_ref = np.asarray(data["gripper_pose"][int(t_curr)], dtype=np.float32).ravel()
            mg = int(min(gp_now.size, gp_ref.size))
            if mg > 0:
                gp_rmse = float(np.linalg.norm(gp_now[:mg] - gp_ref[:mg]) / np.sqrt(mg))

        if (q_rmse <= float(fwd_q_rmse_thresh)) and (gp_rmse <= float(fwd_gp_rmse_thresh)):
            break
        else:
            continue
    else:
        # Could not reliably reconstruct t_curr.
        return float("inf")

    # Ensure lowdim vectors exist (fallback)
    if low_prev_reached is None:
        obs = get_observation_strict(task)
        low_prev_reached = np.asarray(getattr(obs, "task_low_dim_state", []), dtype=np.float32).ravel()
    if low_curr_reached is None:
        obs = get_observation_strict(task)
        low_curr_reached = np.asarray(getattr(obs, "task_low_dim_state", []), dtype=np.float32).ravel()

    # ----------------------------
    # 2) Active dims selection (based on REACHED states)
    # ----------------------------
    active_idxs = _select_active_from_arrays(
        low_prev_reached, low_curr_reached,
        k=int(active_k),
        delta_min=float(delta_min),
        quat_tol=float(quat_tol),
    )

    # ----------------------------
    # 3) Rollback from reached t_curr to t_prev (your original logic)
    # ----------------------------
    for t_des in range(int(t_curr) - 1, int(t_prev) - 1, -int(max(1, waypoint_stride))):
        q_des = q_all[t_des].reshape(-1)
        g_cmd = _gcmd_for_state(int(t_des))

        _servo_track_to_q(
            task, q_des, g_cmd,
            kp=kp, vmax=vmax, tol=tol,
            per_waypoint_max_steps=per_waypoint_max_steps,
            action_noise_std=action_noise_std,
        )

    if int(settle_steps) > 0:
        g_cmd_prev = _gcmd_for_state(int(t_prev))
        _controlled_settle(task, n_arm=q_all.shape[1], g_cmd=g_cmd_prev, steps=int(settle_steps))

    # ----------------------------
    # 4) Distance: lowdim(active dims) + weak robot stabilizer
    # ----------------------------
    obs_f = get_observation_strict(task)

    d_task = 0.0
    if getattr(obs_f, "task_low_dim_state", None) is not None and low_prev_reached.size > 0:
        low_now = np.asarray(obs_f.task_low_dim_state, dtype=np.float32).ravel()
        d_task = lowdim_subset_distance(low_now, low_prev_reached, active_idxs, quat_tol=quat_tol, normalize=True)

    d_robot = 0.0
    if robot_weight > 0:
        if getattr(obs_f, "gripper_pose", None) is not None and "gripper_pose" in data.files:
            gp_ref = np.asarray(data["gripper_pose"][int(t_prev)], dtype=np.float32).ravel()
            gp_now = np.asarray(obs_f.gripper_pose, dtype=np.float32).ravel()
            m = int(min(gp_ref.size, gp_now.size))
            if m > 0:
                d_robot += float(np.linalg.norm(gp_now[:m] - gp_ref[:m]) / np.sqrt(m))

        if getattr(obs_f, "joint_positions", None) is not None and "joint_positions" in data.files:
            q_ref = np.asarray(q_all[int(t_prev)], dtype=np.float32).ravel()
            q_now = np.asarray(obs_f.joint_positions, dtype=np.float32).ravel()
            m = int(min(q_ref.size, q_now.size))
            if m > 0:
                d_robot += float(np.linalg.norm(q_now[:m] - q_ref[:m]) / np.sqrt(m))

        d_robot *= 0.5

    return float(d_task + float(robot_weight) * float(d_robot))


def choose_split_first_irreversible(segments: List[Dict[str, Any]], gate_success: float) -> Dict[str, Any]:
    """
    Return chosen_split dict (matches schema).
    If all segments pass, return mode=all_reversible and seg_index=-1.
    """
    for seg in segments:
        if float(seg["success_rate"]) < float(gate_success):
            return {
                "mode": "first_irreversible",
                "seg_index": int(seg["seg_index"]),
                "kf_prev": int(seg["kf_prev"]),
                "kf_curr": int(seg["kf_curr"]),
                "t_prev": int(seg["t_prev"]),
                "t_curr": int(seg["t_curr"]),
            }
    # all reversible
    last = segments[-1]
    return {
        "mode": "all_reversible",
        "seg_index": -1,
        "kf_prev": int(last["kf_prev"]),
        "kf_curr": int(last["kf_curr"]),
        "t_prev": int(last["t_prev"]),
        "t_curr": int(last["t_curr"]),
    }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--out_json", required=True)

    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--settle_steps", type=int, default=2)
    ap.add_argument("--success_thresh", type=float, default=1e-2)

    ap.add_argument("--kp", type=float, default=6.0)
    ap.add_argument("--vmax", type=float, default=1.0)

    ap.add_argument("--action_noise_std", type=float, default=0.0)
    ap.add_argument("--state_noise_std", type=float, default=0.0)  # placeholder (not applied yet)

    ap.add_argument("--gate_success", type=float, default=0.8)

    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--min_kf_gap", type=int, default=5)
    args = ap.parse_args()

    data = load_demo_npz(args.demo_npz)
    q_all = data["joint_positions"]
    T = int(q_all.shape[0])

    # Require actions to avoid any gripper inference and to be robust for contact-rich tasks.
    if "action" not in data.files:
        raise RuntimeError(
            "Demo NPZ missing 'action'. Re-record demos with actions saved "
            "(arm joint velocities + Discrete gripper cmd)."
        )

    # Minimal obs config for state + servo
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_pose = True
    obs_config.task_low_dim_state = True

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=args.headless,
    )
    env.launch()

    try:
        if not hasattr(rlbench_tasks, args.task):
            raise ValueError(f"Unknown RLBench task '{args.task}'.")
        task_cls = getattr(rlbench_tasks, args.task)
        task = env.get_task(task_cls)
        task.set_variation(args.variation)

        # Anchor to the demo's initial world (non-negotiable)
        restore_keyframe(env, task, data, kf_index=0, settle_steps=int(args.settle_steps), snap_offset=0)
        snap_off = estimate_snapshot_offset(env, task, data, kf_sample_count=5, settle_steps=0)

         # Estimate snapshot offset (your build currently returns 0, but keep it)
        snap_off = estimate_snapshot_offset(env, task, data, kf_sample_count=5, settle_steps=0)

        # IMPORTANT:
        # - snapshot_keyframe_trees rows are aligned with the ORIGINAL keyframe_indices in the NPZ.
        # - If we compress keyframes, we MUST keep ORIGINAL keyframe ROW INDICES, not just timesteps.
        kf_ts_raw = data["keyframe_indices"].astype(int).tolist()  # length K, aligned with snapshot rows
        K_raw = len(kf_ts_raw)
        if K_raw < 2:
            raise RuntimeError("Not enough keyframes in demo (need at least 2).")

        def compress_keyframes_by_gap(kf_ts_raw_list: list[int], min_gap: int) -> list[int]:
            """
            Return ORIGINAL keyframe row indices to keep.
            Keeps first and last.
            Only keeps intermediate keyframes if they are at least min_gap after the last kept.
            Special case: if the last keyframe is too close to the previous kept keyframe,
            replace the previous kept keyframe with the last one (avoids a tiny final segment).
            """
            K = len(kf_ts_raw_list)
            if K == 0:
                return []
            if K == 1:
                return [0]

            keep = [0]
            last_t = int(kf_ts_raw_list[0])

            for i in range(1, K - 1):
                t = int(kf_ts_raw_list[i])
                if t - last_t >= int(min_gap):
                    keep.append(i)
                    last_t = t

            last_idx = K - 1
            if keep[-1] != last_idx:
                last_timestep = int(kf_ts_raw_list[last_idx])
                prev_kept_timestep = int(kf_ts_raw_list[keep[-1]])

                if len(keep) >= 2 and (last_timestep - prev_kept_timestep) < int(min_gap):
                    # Replace the previous kept keyframe with the last keyframe
                    keep[-1] = last_idx
                else:
                    keep.append(last_idx)

            # Ensure first is still first (safety)
            if keep[0] != 0:
                keep = [0] + [k for k in keep if k != 0]

            # Ensure strictly increasing indices
            keep = sorted(set(keep))
            return keep


        keep_kf = compress_keyframes_by_gap(kf_ts_raw, args.min_kf_gap)  # ORIGINAL kf row indices

        # Adjusted keyframe timesteps (used only for horizon / target selection)
        kf_ts = [int(np.clip(kf_ts_raw[i] + snap_off, 0, T - 1)) for i in keep_kf]

        # JSON keyframes (compressed indices, but store original row idx too for debugging)
        keyframes_json = []
        for j, orig_i in enumerate(keep_kf):
            keyframes_json.append(
                {"kf_index": int(j), "t": int(kf_ts[j]), "orig_kf_index": int(orig_i)}
            )

        # ---------------------------
        # SEGMENTS
        # ---------------------------
        segments = []
        for j in range(1, len(keep_kf)):
            print(f"[triage] Evaluating segment {j}/{len(keep_kf)-1})")
            # compressed indices for reporting
            kf_prev = j - 1
            kf_curr = j

            # ORIGINAL keyframe indices for snapshot restore
            orig_kf_prev = int(keep_kf[kf_prev])
            orig_kf_curr = int(keep_kf[kf_curr])

            # adjusted timesteps for target selection / horizon
            t_prev = int(kf_ts[kf_prev])
            t_curr = int(kf_ts[kf_curr])

            if t_curr <= t_prev:
                continue

            horizon = int(t_curr - t_prev)

            final_dists = []
            successes = 0

            for _ in range(int(args.n_rollouts)):
                d = rollback_once_deltaq_servo(
                    env=env,
                    task=task,
                    data=data,
                    t_curr=t_curr,
                    t_prev=t_prev,
                    settle_steps=int(args.settle_steps),
                    kp=float(args.kp),
                    vmax=float(args.vmax),
                    action_noise_std=float(args.action_noise_std),
                )
                final_dists.append(float(d))
                if d <= float(args.success_thresh):
                    successes += 1

            succ_rate = successes / float(args.n_rollouts)
            st = _stats(final_dists)

            seg = {
                "seg_index": int(len(segments)),
                "kf_prev": int(kf_prev),
                "kf_curr": int(kf_curr),
                "t_prev": int(t_prev),
                "t_curr": int(t_curr),
                "horizon": int(horizon),
                "n_rollouts": int(args.n_rollouts),
                "success_rate": float(succ_rate),
                **st,
                "final_dists": final_dists,
                # debugging: show which snapshot rows were actually used
                "orig_kf_prev": int(orig_kf_prev),
                "orig_kf_curr": int(orig_kf_curr),
            }
            segments.append(seg)

        if not segments:
            raise RuntimeError("No valid segments produced (check keyframes / offsets).")

        chosen_split = choose_split_first_irreversible(segments, gate_success=float(args.gate_success))

        demo_id = os.path.splitext(os.path.basename(args.demo_npz))[0]
        out = {
            "schema_version": "plan_a.rollback_triage.v1",
            "task": args.task,
            "variation": int(args.variation),
            "demo_npz": args.demo_npz,
            "demo_id": demo_id,
            "T": int(T),
            "config": {
                "n_rollouts": int(args.n_rollouts),
                "settle_steps": int(args.settle_steps),
                "success_thresh": float(args.success_thresh),
                "action_noise_std": float(args.action_noise_std),
                "state_noise_std": float(args.state_noise_std),
                "min_kf_gap": int(args.min_kf_gap),
            },
            "keyframes": keyframes_json,
            "segments": segments,
            "chosen_split": chosen_split,
            "snapshot_time_offset_est": int(snap_off),
            "min_kf_gap": int(args.min_kf_gap),
        }

        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        print(f"[triage] wrote {args.out_json}")
        print(f"[triage] snap_offset_est={snap_off}  chosen_split={chosen_split}")

    finally:
        env.shutdown()



if __name__ == "__main__":
    main()
