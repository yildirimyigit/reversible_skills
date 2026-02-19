#!/usr/bin/env python3
"""rollback_triage_updated.py

PLAN-A Step 2/3 component.

Behavior (what you asked for):
  1) First run a FULL forward+backward probe (same episode):
       - restore kf0
       - run recorded forward actions up to t_end
       - run reverse playback by servoing to recorded JOINT POSITIONS in reverse time
     If this returns close to the initial state (success rate >= gate_success),
     we declare the whole demo reversible and SKIP segmentation.

  2) Only if the full probe fails, run the usual segment-wise rollback triage.

Why this helps your InsertUsbInComputer case:
  - Segment-wise rollback is a harsher setting (it needs accurate forward reconstruction to an
    intermediate contact state, then reverse only that small window). USB insertion/removal is
    contact-rich and sensitive to tiny state/history differences.
  - The full probe mimics what you demonstrated manually: one continuous forward execution, then
    reverse playback in the same episode.

Notes:
  - Forward probe uses recorded ACTIONS (open-loop) because that preserves the original contact
    trajectory better than reconstructing purely from q targets.
  - Reverse probe uses JOINT-POSITION servo (like your example script), because it is robust to
    dt/physics-step mismatch.

"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Any

import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

from snapshot_utils import (
    load_demo_npz,
    get_observation_strict,
    restore_keyframe,
    estimate_snapshot_offset,
)


# Optional: enforce CoppeliaSim dt
try:
    from pyrep.backend import sim as sim_backend
except Exception:
    sim_backend = None

def _get_sim_dt():
    if sim_backend is None:
        return None
    try:
        return float(sim_backend.simGetFloatParameter(sim_backend.sim_floatparam_simulation_time_step))
    except Exception:
        return None

def _set_sim_dt(dt: float):
    if sim_backend is None:
        return False
    try:
        sim_backend.simSetFloatParameter(sim_backend.sim_floatparam_simulation_time_step, float(dt))
        return True
    except Exception:
        return False


# -----------------------------
# Small utilities
# -----------------------------

def _stats(xs: List[float]) -> Dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "dist_mean": float("nan"),
            "dist_median": float("nan"),
            "dist_p90": float("nan"),
            "dist_min": float("nan"),
            "dist_max": float("nan"),
        }
    return {
        "dist_mean": float(np.mean(arr)),
        "dist_median": float(np.median(arr)),
        "dist_p90": float(np.percentile(arr, 90)),
        "dist_min": float(np.min(arr)),
        "dist_max": float(np.max(arr)),
    }


def compress_keyframes_by_gap(kf_ts_raw: list[int], min_gap: int) -> list[int]:
    """Return ORIGINAL keyframe row indices to keep."""
    K = len(kf_ts_raw)
    if K == 0:
        return []
    if K == 1:
        return [0]

    keep = [0]
    last_t = int(kf_ts_raw[0])

    for i in range(1, K - 1):
        t = int(kf_ts_raw[i])
        if t - last_t >= int(min_gap):
            keep.append(i)
            last_t = t

    last_idx = K - 1
    if keep[-1] != last_idx:
        last_timestep = int(kf_ts_raw[last_idx])
        prev_kept_timestep = int(kf_ts_raw[keep[-1]])

        if len(keep) >= 2 and (last_timestep - prev_kept_timestep) < int(min_gap):
            keep[-1] = last_idx
        else:
            keep.append(last_idx)

    return sorted(set(keep))


def _controlled_settle(task, n_arm: int, g_cmd: float, steps: int):
    """Hold zero arm velocity + constant discrete gripper command for `steps`."""
    if int(steps) <= 0:
        return
    zero_v = np.zeros((int(n_arm),), dtype=np.float32)
    action = np.concatenate([zero_v, np.array([float(g_cmd)], dtype=np.float32)], axis=0)
    for _ in range(int(steps)):
        ret = task.step(action)
        if isinstance(ret, (tuple, list)) and len(ret) >= 3 and bool(ret[2]):
            break


def servo_forward_to_t(
    task,
    data,
    t_end: int,
    kp: float,
    vmax: float,
    tol: float,
    per_waypoint_max_steps: int,
    stride: int = 1,
):
    q_all = np.asarray(data["joint_positions"], dtype=np.float32)
    actions = np.asarray(data["action"], dtype=np.float32)
    n_arm = int(q_all.shape[1])

    t_end = int(np.clip(int(t_end), 1, min(q_all.shape[0] - 1, actions.shape[0])))
    stride = int(max(1, stride))

    for t in range(1, t_end + 1, stride):
        q_des = q_all[t].ravel()
        g_cmd = float(actions[t - 1, -1])  # command applied to enter state t
        _servo_track_to_q(
            task,
            q_des=q_des,
            g_cmd=g_cmd,
            kp=float(kp),
            vmax=float(vmax),
            tol=float(tol),
            per_waypoint_max_steps=int(per_waypoint_max_steps),
            action_noise_std=0.0,
        )


def forward_construct_to_t_end(
    env,
    task,
    data,
    t_end: int,
    *,
    max_attempts: int = 6,
    dilation_schedule=(1, 2, 4, 6, 8, 10),   # D
    accept_lowdim_thresh: float = 0.03,      # tune per task, start 0.02 to 0.05
    quat_tol: float = 0.20,
    allow_large_error_correction: bool = False,
    corr_err_gate: float = 0.25,             # radians L2 joint error gate to trigger correction
    corr_steps: int = 10,
    corr_kp: float = 4.0,
    corr_vmax: float = 0.4,
    corr_tol: float = 0.03,
) -> bool:
    """
    Deterministic forward constructor for contact tasks.

    Uses time dilation that preserves the demo:
      for each recorded action a:
        apply a/D for D control steps
    This keeps net integrated velocity approximately identical to the recorded demo,
    but slower, so contacts can settle.

    Acceptance is based on task_low_dim_state match to the recorded t_end, not joints.
    """
    from state_utils import lowdim_subset_distance

    q_all = np.asarray(data["joint_positions"], dtype=np.float32)
    actions = np.asarray(data["action"], dtype=np.float32)
    n_arm = int(q_all.shape[1])

    t_end = int(np.clip(int(t_end), 1, min(q_all.shape[0] - 1, actions.shape[0])))

    if "task_low_dim_state" not in data.files:
        raise RuntimeError("NPZ missing task_low_dim_state. Needed for forward acceptance in contact tasks.")

    low_ref_full = np.asarray(data["task_low_dim_state"][t_end], dtype=np.float32).ravel()
    if low_ref_full.size == 0 or not np.all(np.isfinite(low_ref_full)):
        return False

    def lowdim_dist_to_ref() -> float:
        obs = get_observation_strict(task)
        low_now = np.asarray(getattr(obs, "task_low_dim_state", []), dtype=np.float32).ravel()
        if low_now.size == 0:
            return float("inf")
        if not np.all(np.isfinite(low_now)):
            return float("inf")
        m = int(min(low_now.size, low_ref_full.size))
        idxs = np.arange(m, dtype=np.int32)
        return float(lowdim_subset_distance(low_now[:m], low_ref_full[:m], idxs, quat_tol=float(quat_tol), normalize=True))

    def maybe_correct_toward_q(i_next: int, g_cmd: float):
        if not allow_large_error_correction:
            return
        obs = get_observation_strict(task)
        q_now = np.asarray(obs.joint_positions, dtype=np.float32).ravel()
        q_tgt = np.asarray(q_all[i_next], dtype=np.float32).ravel()
        err = float(np.linalg.norm(q_tgt - q_now))
        if err < float(corr_err_gate):
            return
        _servo_track_to_q(
            task,
            q_des=q_tgt,
            g_cmd=g_cmd,
            kp=float(corr_kp),
            vmax=float(corr_vmax),
            tol=float(corr_tol),
            per_waypoint_max_steps=int(corr_steps),
            action_noise_std=0.0,
        )

    for attempt in range(int(max_attempts)):
        D = int(dilation_schedule[min(attempt, len(dilation_schedule) - 1)])
        D = int(max(1, D))

        restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)

        done_early = False

        for i in range(int(t_end)):
            a = np.asarray(actions[i], dtype=np.float32).copy()
            g_cmd = float(a[-1])

            # scale by 1/D so that repeating D times preserves the integrated motion
            a[:n_arm] = a[:n_arm] / float(D)

            for _ in range(D):
                ret = task.step(a)
                done = bool(ret[2]) if isinstance(ret, (tuple, list)) and len(ret) >= 3 else False
                if done:
                    done_early = True
                    break
            if done_early:
                break

            # Optional and off by default for USB insertion
            maybe_correct_toward_q(i_next=i + 1, g_cmd=g_cmd)

        d = lowdim_dist_to_ref()
        if d <= float(accept_lowdim_thresh):
            return True

    return False


def _servo_track_to_q(
    task,
    q_des: np.ndarray,
    g_cmd: float,
    kp: float,
    vmax: float,
    tol: float,
    per_waypoint_max_steps: int,
    action_noise_std: float = 0.0,
):
    """Velocity servo toward q_des, with optional noise."""
    q_des = np.asarray(q_des, dtype=np.float32).ravel()
    for _ in range(int(per_waypoint_max_steps)):
        obs = get_observation_strict(task)
        q_now = np.asarray(obs.joint_positions, dtype=np.float32).ravel()
        err = q_des - q_now
        if float(np.linalg.norm(err)) <= float(tol):
            break

        v = float(kp) * err
        if float(action_noise_std) > 0.0:
            v = v + np.random.normal(0.0, float(action_noise_std), size=v.shape).astype(np.float32)
        v = np.clip(v, -float(vmax), float(vmax)).astype(np.float32)

        action = np.concatenate([v, np.array([float(g_cmd)], dtype=np.float32)], axis=0)
        task.step(action)


def find_terminal_t(task, env, data, max_t=None) -> int:
    """Return earliest terminal state index (i+1 when action i ends episode)."""
    q_all = np.asarray(data["joint_positions"], np.float32)
    actions = np.asarray(data["action"], np.float32)
    T = int(min(q_all.shape[0] - 1, actions.shape[0]))
    if max_t is not None:
        T = min(T, int(max_t))

    restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)

    for i in range(T):
        ret = task.step(actions[i])
        done = bool(ret[2]) if isinstance(ret, (tuple, list)) and len(ret) >= 3 else False
        if done:
            return int(i + 1)
    return int(T)


# -----------------------------
# Full forward+reverse probe (your requested behavior)
# -----------------------------

def full_reverse_probe_forward_actions_then_reverse_servo(
    env,
    task,
    data,
    t_end: int,
    kp: float,
    vmax: float,
    tol: float,
    per_waypoint_max_steps: int,
    stride: int,
    settle_steps: int,
    quat_tol: float = 0.20,
) -> float:
    """Run forward actions to t_end, then reverse via q-waypoint servo. Return dist to initial."""

    from state_utils import lowdim_subset_distance

    q_all = np.asarray(data["joint_positions"], dtype=np.float32)
    actions = np.asarray(data["action"], dtype=np.float32)
    n_arm = int(q_all.shape[1])

    # Clamp to available arrays
    t_end = int(np.clip(int(t_end), 1, min(q_all.shape[0] - 1, actions.shape[0])))
    stride = int(max(1, stride))

    def _gcmd_for_state(t: int) -> float:
        # command applied to enter state t; fall back to 0
        if actions.shape[0] <= 0:
            return 0.0
        t = int(t)
        idx = 0 if t <= 0 else (t - 1)
        idx = int(np.clip(idx, 0, actions.shape[0] - 1))
        return float(actions[idx, -1])

    # --- 1) Restore initial snapshot (NO settling for contact tasks) ---
    restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)
    obs0 = get_observation_strict(task)
    low0 = np.asarray(getattr(obs0, "task_low_dim_state", []), dtype=np.float32).ravel().copy()

    # --- 2) Forward: must reach REAL success ---
    ok_fwd, t_reached = forward_replay_until_success(
        env=env,
        task=task,
        data=data,
        t_end=int(t_end),          # this t_end MUST be RAW (see section A)
        max_attempts=8,
        dt_target=0.05,
        settle_on_grip_steps=12,
        resync_every=0,
    )

    if not ok_fwd:
        return float("inf")  # do NOT reverse if forward never succeeded

    t_end_use = int(t_reached)

    # --- 3) Reverse: servo to recorded q targets in reverse order ---
    indices = list(range(0, int(t_end_use) + 1, int(stride)))
    if indices[-1] != int(t_end):
        indices.append(int(t_end))

    # We are already at (approximately) state t_end, so skip the last index.
    for t in reversed(indices[:-1]):
        q_des = q_all[int(t)].ravel()
        g_cmd = _gcmd_for_state(int(t))
        _servo_track_to_q(
            task,
            q_des=q_des,
            g_cmd=g_cmd,
            kp=float(kp),
            vmax=float(vmax),
            tol=float(tol),
            per_waypoint_max_steps=int(per_waypoint_max_steps),
            action_noise_std=0.0,
        )

    if int(settle_steps) > 0:
        _controlled_settle(task, n_arm=n_arm, g_cmd=_gcmd_for_state(0), steps=int(settle_steps))

    # --- 4) Score vs initial lowdim (all dims) ---
    obsf = get_observation_strict(task)
    lowf = np.asarray(getattr(obsf, "task_low_dim_state", []), dtype=np.float32).ravel()

    if low0.size == 0 or lowf.size == 0:
        return float("inf")

    # Avoid numpy warnings / NaN distances if sim returns NaNs
    if not np.all(np.isfinite(low0)) or not np.all(np.isfinite(lowf)):
        return float("inf")

    m = int(min(low0.size, lowf.size))
    idxs_all = np.arange(m, dtype=np.int32)
    return float(lowdim_subset_distance(lowf[:m], low0[:m], idxs_all, quat_tol=float(quat_tol), normalize=True))


# -----------------------------
# Segment rollback (same as before)
# -----------------------------

def _select_active_from_arrays(
    low_prev: np.ndarray,
    low_curr: np.ndarray,
    k: int = 16,
    delta_min: float = 1e-3,
    quat_tol: float = 0.20,
) -> np.ndarray:
    from state_utils import _is_unit_quat, _quat_diff_sign_invariant

    a = np.asarray(low_prev, dtype=np.float32).ravel()
    b = np.asarray(low_curr, dtype=np.float32).ravel()
    n = int(min(a.size, b.size))
    if n <= 0:
        return np.zeros((0,), dtype=np.int32)

    feats = []
    i = 0
    while i < n:
        if i + 4 <= n and _is_unit_quat(a[i:i + 4], tol=quat_tol) and _is_unit_quat(b[i:i + 4], tol=quat_tol):
            s = _quat_diff_sign_invariant(a[i:i + 4], b[i:i + 4])
            feats.append((float(s), [i, i + 1, i + 2, i + 3]))
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
    """Your current segment rollback (closed-loop forward reconstruction, reverse servo)."""

    from state_utils import lowdim_subset_distance

    q_all = np.asarray(data["joint_positions"], dtype=np.float32)
    T = int(q_all.shape[0])
    n_arm = int(q_all.shape[1])

    if int(t_prev) < 0 or int(t_curr) < 0 or int(t_prev) >= T or int(t_curr) >= T:
        return float("inf")
    if int(t_curr) <= int(t_prev):
        return float("inf")

    if "action" not in data.files:
        raise RuntimeError(
            "Demo NPZ missing 'action'. Re-record demos with actions saved (arm joint velocities + Discrete gripper cmd)."
        )
    actions = np.asarray(data["action"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < (n_arm + 1):
        raise RuntimeError(f"Unexpected action shape {actions.shape}; expected (T, {n_arm + 1}) at least.")

    def _gcmd_for_state(t: int) -> float:
        t = int(t)
        if actions.shape[0] <= 0:
            return 0.0
        idx = 0 if t <= 0 else (t - 1)
        idx = int(np.clip(idx, 0, actions.shape[0] - 1))
        return float(actions[idx, -1])

    def _servo_to_q_report(q_des: np.ndarray, g_cmd: float, kp_: float, vmax_: float, tol_: float, max_steps: int) -> bool:
        q_des = np.asarray(q_des, dtype=np.float32).ravel()
        for _ in range(int(max_steps)):
            obs = get_observation_strict(task)
            q_now = np.asarray(obs.joint_positions, dtype=np.float32).ravel()
            err = q_des - q_now
            if float(np.linalg.norm(err)) <= float(tol_):
                return True
            v = np.clip(float(kp_) * err, -float(vmax_), float(vmax_)).astype(np.float32)
            a = np.concatenate([v, np.array([float(g_cmd)], dtype=np.float32)], axis=0)
            ret = task.step(a)
            done = bool(ret[2]) if isinstance(ret, (tuple, list)) and len(ret) >= 3 else False
            if done:
                break
        # final check
        obs = get_observation_strict(task)
        q_now = np.asarray(obs.joint_positions, dtype=np.float32).ravel()
        return float(np.linalg.norm(q_des - q_now)) <= float(tol_)

    # Forward reconstruction tuning
    fwd_kp = float(min(max(kp, 2.0), 6.0))
    fwd_vmax = float(min(max(vmax, 0.2), 0.8))
    fwd_q_tol = float(max(0.02, 2.0 * float(tol)))
    fwd_micro_steps = int(max(2, min(6, per_waypoint_max_steps // 5)))

    max_forward_attempts = 5
    low_prev_reached = None
    low_curr_reached = None

    for _attempt in range(int(max_forward_attempts)):
        restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)

        low_prev_reached = None
        low_curr_reached = None

        obs0 = get_observation_strict(task)
        if int(t_prev) == 0 and getattr(obs0, "task_low_dim_state", None) is not None:
            low_prev_reached = np.asarray(obs0.task_low_dim_state, dtype=np.float32).ravel().copy()
        if int(t_curr) == 0 and getattr(obs0, "task_low_dim_state", None) is not None:
            low_curr_reached = np.asarray(obs0.task_low_dim_state, dtype=np.float32).ravel().copy()

        failed = False
        for t in range(1, int(t_curr) + 1):
            q_des = q_all[t].ravel()
            g_cmd = _gcmd_for_state(t)
            if not _servo_to_q_report(q_des, g_cmd, fwd_kp, fwd_vmax, fwd_q_tol, fwd_micro_steps):
                failed = True
                break

            obs_t = get_observation_strict(task)
            if int(t) == int(t_prev) and getattr(obs_t, "task_low_dim_state", None) is not None:
                low_prev_reached = np.asarray(obs_t.task_low_dim_state, dtype=np.float32).ravel().copy()
            if int(t) == int(t_curr) and getattr(obs_t, "task_low_dim_state", None) is not None:
                low_curr_reached = np.asarray(obs_t.task_low_dim_state, dtype=np.float32).ravel().copy()

        if failed:
            continue
        break
    else:
        return float("inf")

    if low_prev_reached is None:
        obs = get_observation_strict(task)
        low_prev_reached = np.asarray(getattr(obs, "task_low_dim_state", []), dtype=np.float32).ravel()
    if low_curr_reached is None:
        obs = get_observation_strict(task)
        low_curr_reached = np.asarray(getattr(obs, "task_low_dim_state", []), dtype=np.float32).ravel()

    active_idxs = _select_active_from_arrays(
        low_prev_reached,
        low_curr_reached,
        k=int(active_k),
        delta_min=float(delta_min),
        quat_tol=float(quat_tol),
    )

    for t_des in range(int(t_curr) - 1, int(t_prev) - 1, -int(max(1, waypoint_stride))):
        q_des = q_all[int(t_des)].reshape(-1)
        g_cmd = _gcmd_for_state(int(t_des))
        _servo_track_to_q(
            task,
            q_des=q_des,
            g_cmd=g_cmd,
            kp=float(kp),
            vmax=float(vmax),
            tol=float(tol),
            per_waypoint_max_steps=int(per_waypoint_max_steps),
            action_noise_std=float(action_noise_std),
        )

    if int(settle_steps) > 0:
        _controlled_settle(task, n_arm=n_arm, g_cmd=_gcmd_for_state(int(t_prev)), steps=int(settle_steps))

    obs_f = get_observation_strict(task)

    d_task = 0.0
    if getattr(obs_f, "task_low_dim_state", None) is not None and low_prev_reached.size > 0:
        low_now = np.asarray(obs_f.task_low_dim_state, dtype=np.float32).ravel()
        d_task = lowdim_subset_distance(low_now, low_prev_reached, active_idxs, quat_tol=float(quat_tol), normalize=True)

    d_robot = 0.0
    if float(robot_weight) > 0:
        if getattr(obs_f, "joint_positions", None) is not None and "joint_positions" in data.files:
            q_ref = np.asarray(q_all[int(t_prev)], dtype=np.float32).ravel()
            q_now = np.asarray(obs_f.joint_positions, dtype=np.float32).ravel()
            m = int(min(q_ref.size, q_now.size))
            if m > 0:
                d_robot += float(np.linalg.norm(q_now[:m] - q_ref[:m]) / np.sqrt(m))

        if getattr(obs_f, "gripper_pose", None) is not None and "gripper_pose" in data.files:
            gp_ref = np.asarray(data["gripper_pose"][int(t_prev)], dtype=np.float32).ravel()
            gp_now = np.asarray(obs_f.gripper_pose, dtype=np.float32).ravel()
            mg = int(min(gp_ref.size, gp_now.size))
            if mg > 0:
                d_robot += float(np.linalg.norm(gp_now[:mg] - gp_ref[:mg]) / np.sqrt(mg))

        d_robot *= 0.5

    return float(d_task + float(robot_weight) * float(d_robot))


def choose_split_first_irreversible(segments: List[Dict[str, Any]], gate_success: float) -> Dict[str, Any]:
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

    last = segments[-1]
    return {
        "mode": "all_reversible",
        "seg_index": -1,
        "kf_prev": int(last["kf_prev"]),
        "kf_curr": int(last["kf_curr"]),
        "t_prev": int(last["t_prev"]),
        "t_curr": int(last["t_curr"]),
    }



def _task_success(task) -> bool:
    # RLBench tasks usually expose this
    try:
        s = task._task.success()
        if isinstance(s, (tuple, list)):
            return bool(s[0])
        return bool(s)
    except Exception:
        return False


def forward_replay_until_success(
    env,
    task,
    data,
    t_end: int,
    *,
    max_attempts: int = 8,
    dt_target: float = 0.05,
    settle_on_grip_steps: int = 12,
    resync_every: int = 0,          # 0 disables
    resync_kp: float = 2.5,
    resync_vmax: float = 0.25,
    resync_tol: float = 0.01,
    resync_steps: int = 120,
) -> tuple[bool, int]:
    """
    Tries to replay the recorded actions as faithfully as possible.
    Returns (success, t_reached). Success means the TASK succeeded (USB inserted), not "close enough".
    """

    q_all = np.asarray(data["joint_positions"], dtype=np.float32)
    actions = np.asarray(data["action"], dtype=np.float32)
    n_arm = int(q_all.shape[1])

    t_end = int(np.clip(int(t_end), 1, min(q_all.shape[0] - 1, actions.shape[0])))

    # If sim dt is different, scale velocities so displacement per step matches the recording dt.
    dt_now = _get_sim_dt()
    vel_scale = 1.0
    if dt_now is not None and np.isfinite(dt_now) and dt_now > 0:
        vel_scale = float(dt_target) / float(dt_now)
        # If dt_now=0.2 and dt_target=0.05 => vel_scale=0.25 (slows to match)
        # If dt_now already 0.05 => vel_scale=1.0

    # Identify gripper command transitions (important moments)
    g = actions[:, -1].copy()
    trans = set(np.where(g[1:] != g[:-1])[0].tolist())  # transition at index i means action i causes change

    for attempt in range(int(max_attempts)):
        restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)

        last_g = float(actions[0, -1])
        t_reached = 0

        for i in range(int(t_end)):
            a = np.asarray(actions[i], dtype=np.float32).copy()
            a[:n_arm] *= float(vel_scale)  # compensate dt mismatch if needed

            ret = task.step(a)
            t_reached = i + 1

            done = bool(ret[2]) if isinstance(ret, (tuple, list)) and len(ret) >= 3 else False
            if done or _task_success(task):
                return True, int(t_reached)

            # If gripper command changed at this step, let the grasp "seat"
            gi = float(a[-1])
            if i in trans or gi != last_g:
                _controlled_settle(task, n_arm=n_arm, g_cmd=gi, steps=int(settle_on_grip_steps))
            last_g = gi

            # Optional: light resync to recorded joint state to prevent drift (does NOT rewrite the whole trajectory)
            if int(resync_every) > 0 and (t_reached % int(resync_every) == 0):
                _servo_track_to_q(
                    task,
                    q_des=q_all[t_reached].ravel(),
                    g_cmd=last_g,
                    kp=float(resync_kp),
                    vmax=float(resync_vmax),
                    tol=float(resync_tol),
                    per_waypoint_max_steps=int(resync_steps),
                    action_noise_std=0.0,
                )

        # attempt ended without success, try again
    return False, int(t_end)



# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--out_json", required=True)

    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--settle_steps", type=int, default=0)
    ap.add_argument("--success_thresh", type=float, default=1e-2)

    ap.add_argument("--kp", type=float, default=4.0)
    ap.add_argument("--vmax", type=float, default=0.4)

    ap.add_argument("--action_noise_std", type=float, default=0.0)
    ap.add_argument("--state_noise_std", type=float, default=0.0)

    ap.add_argument("--gate_success", type=float, default=0.8)

    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--min_kf_gap", type=int, default=5)

    # Full-probe controls
    ap.add_argument("--no_full_reverse_probe", action="store_true", help="Skip full probe and always run segmentation.")
    ap.add_argument("--probe_rollouts", type=int, default=3)
    ap.add_argument("--probe_stride", type=int, default=1)
    ap.add_argument("--probe_waypoint_max_steps", type=int, default=60)
    ap.add_argument("--probe_tol", type=float, default=0.02)
    ap.add_argument("--probe_success_thresh", type=float, default=None)

    args = ap.parse_args()

    data = load_demo_npz(args.demo_npz)

    if "action" not in data.files:
        raise RuntimeError(
            "Demo NPZ missing 'action'. Re-record demos with actions saved (arm joint velocities + Discrete gripper cmd)."
        )

    q_all = np.asarray(data["joint_positions"], dtype=np.float32)
    T = int(q_all.shape[0])

    # Minimal obs config
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

        DT_TARGET = 0.05  # typical RLBench control step. If you recorded at a different dt, set it to that.
        if _set_sim_dt(DT_TARGET):
            dt_now = _get_sim_dt()
            print(f"[triage] sim_dt set to {dt_now}")
        else:
            dt_now = _get_sim_dt()
            print(f"[triage] sim_dt is {dt_now} (could not set)")

        task.set_variation(args.variation)

        # Anchor once (also initializes task)
        restore_keyframe(env, task, data, kf_index=0, settle_steps=int(args.settle_steps), snap_offset=0)

        snap_off = int(estimate_snapshot_offset(env, task, data, kf_sample_count=5, settle_steps=0))

        # terminal timestep (Fix 1)
        t_term = int(find_terminal_t(task=task, env=env, data=data))
        restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)

        # keyframes
        kf_ts_raw = data["keyframe_indices"].astype(int).tolist()
        if len(kf_ts_raw) < 2:
            raise RuntimeError("Not enough keyframes in demo (need at least 2).")
        
        kf_ts_raw = data["keyframe_indices"].astype(int).tolist()
        keep_kf = compress_keyframes_by_gap(kf_ts_raw, args.min_kf_gap)

        # IMPORTANT: clip using RAW timesteps, not (raw + snap_off)
        keep_kf = [idx for idx in keep_kf if int(kf_ts_raw[idx]) <= int(t_term)]
        if len(keep_kf) < 2:
            raise RuntimeError("After clipping to terminal timestep, fewer than 2 keyframes remain.")

        # For snapshot-based segmentation you can still compute the adjusted times:
        kf_ts = [int(np.clip(int(kf_ts_raw[idx]) + int(snap_off), 0, T - 1)) for idx in keep_kf]

        # Full-probe forward replay must use RAW time
        t_end_probe = int(min(int(t_term), int(kf_ts_raw[keep_kf[-1]])))

        keep_kf = compress_keyframes_by_gap(kf_ts_raw, args.min_kf_gap)
        kf_ts = [int(np.clip(int(kf_ts_raw[i]) + snap_off, 0, T - 1)) for i in keep_kf]

        # clip to terminal
        clipped_keep_kf = []
        clipped_kf_ts = []
        for orig_i, t_adj in zip(keep_kf, kf_ts):
            if int(t_adj) <= int(t_term):
                clipped_keep_kf.append(int(orig_i))
                clipped_kf_ts.append(int(t_adj))
        if len(clipped_keep_kf) < 2:
            raise RuntimeError(
                f"After clipping to terminal timestep t_term={t_term}, fewer than 2 keyframes remain. "
                f"(kept_before={len(keep_kf)}, kept_after={len(clipped_keep_kf)})"
            )
        keep_kf = clipped_keep_kf
        kf_ts = clipped_kf_ts

        print(f"[triage] terminal_t={t_term}  keyframes_kept={len(keep_kf)} (after clipping)")

        keyframes_json = [
            {"kf_index": int(j), "t": int(kf_ts[j]), "orig_kf_index": int(orig_i)}
            for j, orig_i in enumerate(keep_kf)
        ]

        # full probe end time: last kept keyframe time
        t_end_probe = int(kf_ts[-1])
        probe_thresh = float(args.success_thresh) if args.probe_success_thresh is None else float(args.probe_success_thresh)

        full_probe = None

        # ---------------------------
        # (A) FULL PROBE FIRST
        # ---------------------------
        if not args.no_full_reverse_probe:
            probe_dists = []
            probe_succ = 0
            probe_invalid = 0

            for _ in range(int(args.probe_rollouts)):
                d = full_reverse_probe_forward_actions_then_reverse_servo(
                    env=env,
                    task=task,
                    data=data,
                    t_end=int(t_end_probe),
                    kp=float(args.kp),
                    vmax=float(args.vmax),
                    tol=float(args.probe_tol),
                    per_waypoint_max_steps=int(args.probe_waypoint_max_steps),
                    stride=int(args.probe_stride),
                    settle_steps=0,
                    quat_tol=0.20,
                )
                probe_dists.append(float(d))
                if not np.isfinite(d):
                    probe_invalid += 1
                    continue
                if float(d) <= float(probe_thresh):
                    probe_succ += 1

            if probe_invalid == int(args.probe_rollouts):
                chosen_split = {
                    "mode": "probe_failed_forward",
                    "seg_index": -1,
                    "kf_prev": 0,
                    "kf_curr": int(len(keep_kf) - 1),
                    "t_prev": int(kf_ts[0]),
                    "t_curr": int(t_end_probe),
                }
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
                        "terminal_t": int(t_term),
                        "full_reverse_probe_first": True,
                    },
                    "keyframes": keyframes_json,
                    "segments": [],
                    "chosen_split": chosen_split,
                    "snapshot_time_offset_est": int(snap_off),
                    "min_kf_gap": int(args.min_kf_gap),
                    "full_reverse_probe": full_probe,
                }

                os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
                with open(args.out_json, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)

                print(f"[triage] full probe FAIL (forward never succeeded in {args.probe_rollouts} rollouts) -> wrote {args.out_json}")
                return


            probe_rate = probe_succ / float(max(1, int(args.probe_rollouts)))
            full_probe = {
                "t_end_probe": int(t_end_probe),
                "probe_rollouts": int(args.probe_rollouts),
                "probe_stride": int(args.probe_stride),
                "probe_waypoint_max_steps": int(args.probe_waypoint_max_steps),
                "probe_tol": float(args.probe_tol),
                "probe_success_thresh": float(probe_thresh),
                "probe_success_rate": float(probe_rate),
                "probe_dists": probe_dists,
                "probe_invalid": int(probe_invalid),
            }

            if float(probe_rate) >= float(args.gate_success):
                chosen_split = {
                    "mode": "all_reversible_full_reverse_probe",
                    "seg_index": -1,
                    "kf_prev": 0,
                    "kf_curr": int(len(keep_kf) - 1),
                    "t_prev": int(kf_ts[0]),
                    "t_curr": int(t_end_probe),
                }

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
                        "terminal_t": int(t_term),
                        "full_reverse_probe_first": True,
                    },
                    "keyframes": keyframes_json,
                    "segments": [],
                    "chosen_split": chosen_split,
                    "snapshot_time_offset_est": int(snap_off),
                    "min_kf_gap": int(args.min_kf_gap),
                    "full_reverse_probe": full_probe,
                }

                os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
                with open(args.out_json, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)

                print(f"[triage] full probe PASS (rate={probe_rate:.2f}) -> wrote {args.out_json}")
                return

        # ---------------------------
        # (B) SEGMENT TRIAGE ONLY IF PROBE FAILED
        # ---------------------------
        segments: List[Dict[str, Any]] = []

        for j in range(1, len(keep_kf)):
            print(f"[triage] Evaluating segment {j}/{len(keep_kf) - 1}")

            kf_prev = j - 1
            kf_curr = j

            orig_kf_prev = int(keep_kf[kf_prev])
            orig_kf_curr = int(keep_kf[kf_curr])

            t_prev = int(kf_ts[kf_prev])
            t_curr = int(kf_ts[kf_curr])
            if t_curr <= t_prev:
                continue

            horizon = int(t_curr - t_prev)

            final_dists = []
            successes = 0
            invalid = 0

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

                if not np.isfinite(d):
                    invalid += 1
                    continue

                if float(d) <= float(args.success_thresh):
                    successes += 1

            succ_rate = successes / float(args.n_rollouts) if int(args.n_rollouts) > 0 else 0.0
            st = _stats(final_dists)

            segments.append(
                {
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
                    "orig_kf_prev": int(orig_kf_prev),
                    "orig_kf_curr": int(orig_kf_curr),
                    "n_invalid": int(invalid),
                }
            )

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
                "terminal_t": int(t_term),
                "full_reverse_probe_first": (not args.no_full_reverse_probe),
            },
            "keyframes": keyframes_json,
            "segments": segments,
            "chosen_split": chosen_split,
            "snapshot_time_offset_est": int(snap_off),
            "min_kf_gap": int(args.min_kf_gap),
            "full_reverse_probe": full_probe,
        }

        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        print(f"[triage] wrote {args.out_json}")
        print(f"[triage] snap_offset_est={snap_off}  terminal_t={t_term}  chosen_split={chosen_split}")

    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
