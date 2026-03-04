#!/usr/bin/env python3
"""
rollback_triage_clean.py

Key rule: FORWARD replay must be a faithful reproduction of the demo.
So forward = recorded actions, no scaling, no extra settle, no resync.

Full probe:
  restore kf0 -> forward replay until success (or t_end) -> reverse servo by recorded joint positions
  if forward never succeeds, we return inf (and do NOT reverse).

Segment triage:
  restore kf0 -> forward replay to t_curr (must reach it) -> reverse servo to t_prev -> score
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

from snapshot_utils import load_demo_npz, get_observation_strict, restore_keyframe
from state_utils import lowdim_subset_distance


def _task_success(task) -> bool:
    try:
        s = task._task.success()
        if isinstance(s, (tuple, list)):
            return bool(s[0])
        return bool(s)
    except Exception:
        return False


def _stats(xs: List[float]) -> Dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(dist_mean=float("nan"), dist_median=float("nan"), dist_p90=float("nan"),
                    dist_min=float("nan"), dist_max=float("nan"))
    return dict(dist_mean=float(np.mean(arr)), dist_median=float(np.median(arr)),
                dist_p90=float(np.percentile(arr, 90)), dist_min=float(np.min(arr)), dist_max=float(np.max(arr)))


def compress_keyframes_by_gap(kf_ts_raw: list[int], min_gap: int) -> list[int]:
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
    if keep[-1] != (K - 1):
        keep.append(K - 1)
    return sorted(set(keep))


def _servo_track_to_q(task, q_des: np.ndarray, g_cmd: float, kp: float, vmax: float, tol: float, max_steps: int):
    q_des = np.asarray(q_des, dtype=np.float32).ravel()
    for _ in range(int(max_steps)):
        obs = get_observation_strict(task)
        q_now = np.asarray(obs.joint_positions, dtype=np.float32).ravel()
        err = q_des - q_now
        if float(np.linalg.norm(err)) <= float(tol):
            break
        v = np.clip(float(kp) * err, -float(vmax), float(vmax)).astype(np.float32)
        action = np.concatenate([v, np.array([float(g_cmd)], dtype=np.float32)], axis=0)
        task.step(action)


def _gcmd_for_state(actions: np.ndarray, t: int) -> float:
    if actions.shape[0] <= 0:
        return 0.0
    t = int(t)
    idx = 0 if t <= 0 else (t - 1)
    idx = int(np.clip(idx, 0, actions.shape[0] - 1))
    return float(actions[idx, -1])


def forward_replay_plain(env, task, data, t_max: int) -> Tuple[bool, int]:
    """Faithful forward replay. Returns (success, t_reached)."""
    actions = np.asarray(data["action"], dtype=np.float32)
    t_max = int(np.clip(int(t_max), 1, actions.shape[0]))
    restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)

    for i in range(t_max):
        ret = task.step(actions[i])
        done = bool(ret[2]) if isinstance(ret, (tuple, list)) and len(ret) >= 3 else False
        if done or _task_success(task):
            return True, int(i + 1)
    return False, int(t_max)


def find_terminal_t(env, task, data) -> int:
    """Earliest timestep where the episode terminates during plain replay; else full length."""
    actions = np.asarray(data["action"], dtype=np.float32)
    restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)
    for i in range(actions.shape[0]):
        ret = task.step(actions[i])
        done = bool(ret[2]) if isinstance(ret, (tuple, list)) and len(ret) >= 3 else False
        if done or _task_success(task):
            return int(i + 1)
    return int(actions.shape[0])


def full_probe(env, task, data, t_end: int, kp: float, vmax: float, tol: float, max_steps: int, stride: int) -> float:
    """Forward must succeed; then reverse-servo; score lowdim distance to initial."""
    q_all = np.asarray(data["joint_positions"], dtype=np.float32)
    actions = np.asarray(data["action"], dtype=np.float32)

    restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)
    low0 = np.asarray(get_observation_strict(task).task_low_dim_state, dtype=np.float32).ravel().copy()

    ok, t_reached = forward_replay_plain(env, task, data, t_max=t_end)
    if not ok:
        return float("inf")

    t_use = int(min(t_reached, q_all.shape[0] - 1))
    stride = int(max(1, stride))

    idxs = list(range(0, t_use + 1, stride))
    if idxs[-1] != t_use:
        idxs.append(t_use)

    for t in reversed(idxs[:-1]):
        _servo_track_to_q(
            task,
            q_des=q_all[int(t)].ravel(),
            g_cmd=_gcmd_for_state(actions, int(t)),
            kp=float(kp),
            vmax=float(vmax),
            tol=float(tol),
            max_steps=int(max_steps),
        )

    lowf = np.asarray(get_observation_strict(task).task_low_dim_state, dtype=np.float32).ravel()
    m = int(min(low0.size, lowf.size))
    if m == 0:
        return float("inf")
    idxs_all = np.arange(m, dtype=np.int32)
    return float(lowdim_subset_distance(lowf[:m], low0[:m], idxs_all, quat_tol=0.20, normalize=True))


def rollback_once_segment(env, task, data, t_prev: int, t_curr: int, kp: float, vmax: float, tol: float, max_steps: int) -> float:
    """Forward replay to t_curr, then reverse-servo back to t_prev, score lowdim distance."""
    q_all = np.asarray(data["joint_positions"], dtype=np.float32)
    actions = np.asarray(data["action"], dtype=np.float32)

    # forward replay must reach t_curr (no success requirement here)
    restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)

    low_prev = None
    for i in range(int(t_curr)):
        ret = task.step(actions[i])
        if (i + 1) == int(t_prev):
            low_prev = np.asarray(get_observation_strict(task).task_low_dim_state, dtype=np.float32).ravel().copy()
        done = bool(ret[2]) if isinstance(ret, (tuple, list)) and len(ret) >= 3 else False
        if done:
            return float("inf")

    if low_prev is None:
        return float("inf")

    # reverse back to t_prev
    for t in range(int(t_curr) - 1, int(t_prev) - 1, -1):
        _servo_track_to_q(
            task,
            q_des=q_all[int(t)].ravel(),
            g_cmd=_gcmd_for_state(actions, int(t)),
            kp=float(kp),
            vmax=float(vmax),
            tol=float(tol),
            max_steps=int(max_steps),
        )

    low_now = np.asarray(get_observation_strict(task).task_low_dim_state, dtype=np.float32).ravel()
    m = int(min(low_now.size, low_prev.size))
    if m == 0:
        return float("inf")
    idxs_all = np.arange(m, dtype=np.int32)
    return float(lowdim_subset_distance(low_now[:m], low_prev[:m], idxs_all, quat_tol=0.20, normalize=True))


def choose_split_first_irreversible(segments: List[Dict[str, Any]], gate_success: float) -> Dict[str, Any]:
    for seg in segments:
        if float(seg["success_rate"]) < float(gate_success):
            return dict(mode="first_irreversible", **{k: seg[k] for k in ("seg_index","kf_prev","kf_curr","t_prev","t_curr")})
    last = segments[-1]
    return dict(mode="all_reversible", **{k: last[k] for k in ("seg_index","kf_prev","kf_curr","t_prev","t_curr")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--out_json", required=True)

    ap.add_argument("--min_kf_gap", type=int, default=5)

    ap.add_argument("--probe_rollouts", type=int, default=3)
    ap.add_argument("--probe_stride", type=int, default=1)
    ap.add_argument("--probe_waypoint_max_steps", type=int, default=80)
    ap.add_argument("--probe_tol", type=float, default=0.02)
    ap.add_argument("--probe_success_thresh", type=float, default=1e-2)
    ap.add_argument("--gate_success", type=float, default=0.8)

    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--success_thresh", type=float, default=1e-2)

    ap.add_argument("--kp", type=float, default=4.0)
    ap.add_argument("--vmax", type=float, default=0.4)

    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    data = load_demo_npz(args.demo_npz)
    if "action" not in data.files:
        raise RuntimeError("NPZ missing 'action'.")

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
        task_cls = getattr(rlbench_tasks, args.task)
        task = env.get_task(task_cls)
        task.set_variation(args.variation)

        t_term = find_terminal_t(env, task, data)

        kf_ts_raw = data["keyframe_indices"].astype(int).tolist()
        keep_rows = compress_keyframes_by_gap(kf_ts_raw, args.min_kf_gap)
        kept_ts = [int(kf_ts_raw[r]) for r in keep_rows if int(kf_ts_raw[r]) <= int(t_term)]
        if len(kept_ts) < 2:
            raise RuntimeError("Not enough keyframes after clipping to terminal.")

        t_end_probe = int(kept_ts[-1])

        # FULL PROBE
        probe_dists = []
        probe_succ = 0
        probe_invalid = 0
        for _ in range(int(args.probe_rollouts)):
            d = full_probe(
                env, task, data, t_end=t_end_probe,
                kp=args.kp, vmax=args.vmax,
                tol=args.probe_tol,
                max_steps=args.probe_waypoint_max_steps,
                stride=args.probe_stride,
            )
            probe_dists.append(float(d))
            if not np.isfinite(d):
                probe_invalid += 1
            elif float(d) <= float(args.probe_success_thresh):
                probe_succ += 1

        probe_rate = probe_succ / float(max(1, int(args.probe_rollouts)))

        out: Dict[str, Any] = dict(
            schema_version="plan_a.rollback_triage.clean.v1",
            task=args.task,
            variation=int(args.variation),
            demo_npz=args.demo_npz,
            terminal_t=int(t_term),
            keyframes=[dict(k=i, t=int(t)) for i, t in enumerate(kept_ts)],
            full_reverse_probe=dict(
                t_end_probe=int(t_end_probe),
                probe_rollouts=int(args.probe_rollouts),
                probe_success_rate=float(probe_rate),
                probe_dists=probe_dists,
                probe_invalid=int(probe_invalid),
                probe_success_thresh=float(args.probe_success_thresh),
            ),
        )

        if float(probe_rate) >= float(args.gate_success):
            out["chosen_split"] = dict(mode="all_reversible_full_probe", t_prev=int(kept_ts[0]), t_curr=int(t_end_probe))
            out["segments"] = []
        else:
            # SEGMENT TRIAGE (simple)
            segments: List[Dict[str, Any]] = []
            for j in range(1, len(kept_ts)):
                t_prev = int(kept_ts[j - 1])
                t_curr = int(kept_ts[j])
                final_dists = []
                succ = 0
                inv = 0
                for _ in range(int(args.n_rollouts)):
                    d = rollback_once_segment(
                        env, task, data,
                        t_prev=t_prev, t_curr=t_curr,
                        kp=args.kp, vmax=args.vmax,
                        tol=0.02,
                        max_steps=60,
                    )
                    final_dists.append(float(d))
                    if not np.isfinite(d):
                        inv += 1
                    elif float(d) <= float(args.success_thresh):
                        succ += 1
                segments.append(dict(
                    seg_index=len(segments),
                    kf_prev=j - 1,
                    kf_curr=j,
                    t_prev=t_prev,
                    t_curr=t_curr,
                    horizon=int(t_curr - t_prev),
                    n_rollouts=int(args.n_rollouts),
                    success_rate=float(succ / float(max(1, int(args.n_rollouts)))),
                    n_invalid=int(inv),
                    final_dists=final_dists,
                    **_stats(final_dists),
                ))
            out["segments"] = segments
            out["chosen_split"] = choose_split_first_irreversible(segments, gate_success=float(args.gate_success))

        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[triage] wrote {args.out_json}")

    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
