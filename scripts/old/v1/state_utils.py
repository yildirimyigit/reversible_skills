#!/usr/bin/env python3
"""
state_utils.py

Compact state z for PLAN-A rollback triage.

z = [joint_positions(7),
     gripper_open(1),
     gripper_pose(7) if present else nothing,
     task_low_dim_state(*) if present else nothing]

Key fix:
- l2_dist is quaternion-aware: treats q and -q as identical rotations by flipping
  antipodal quaternions in b before computing distance.
"""

from __future__ import annotations

import numpy as np


def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).ravel()


def compact_state_from_obs(obs) -> np.ndarray:
    parts = []

    if getattr(obs, "joint_positions", None) is not None:
        parts.append(_f32(obs.joint_positions))

    if getattr(obs, "gripper_open", None) is not None:
        parts.append(_f32([float(obs.gripper_open)]))

    gp = getattr(obs, "gripper_pose", None)
    if gp is not None:
        parts.append(_f32(gp))

    low = getattr(obs, "task_low_dim_state", None)
    if low is not None:
        parts.append(_f32(low))

    if not parts:
        raise RuntimeError("compact_state_from_obs: no components found in obs.")
    return np.concatenate(parts, axis=0)


def compact_state_from_npz(data, t: int) -> np.ndarray:
    parts = []

    if "joint_positions" in data.files:
        parts.append(_f32(data["joint_positions"][t]))

    if "gripper_open" in data.files:
        go = float(np.asarray(data["gripper_open"][t]).reshape(-1)[0])
        parts.append(_f32([go]))

    if "gripper_pose" in data.files:
        parts.append(_f32(data["gripper_pose"][t]))

    if "task_low_dim_state" in data.files:
        parts.append(_f32(data["task_low_dim_state"][t]))

    if not parts:
        raise RuntimeError("compact_state_from_npz: no components found in npz.")
    return np.concatenate(parts, axis=0)


def _find_quaternion_groups(a: np.ndarray, b: np.ndarray,
                            norm_tol: float = 0.15,
                            max_abs: float = 1.25) -> list[tuple[int, int]]:
    """
    Heuristically find quaternion blocks in concatenated vectors.

    We look for 4D chunks where:
      - both have norm ~ 1
      - values are within a sane range (to avoid treating joints as quats)

    Returns list of (start, length=4) non-overlapping groups.
    """
    n = min(a.size, b.size)
    groups: list[tuple[int, int]] = []
    i = 0
    while i + 4 <= n:
        xa = a[i:i+4]
        xb = b[i:i+4]

        if np.all(np.isfinite(xa)) and np.all(np.isfinite(xb)):
            na = float(np.linalg.norm(xa))
            nb = float(np.linalg.norm(xb))

            if (abs(na - 1.0) <= norm_tol and abs(nb - 1.0) <= norm_tol
                and float(np.max(np.abs(xa))) <= max_abs
                and float(np.max(np.abs(xb))) <= max_abs):
                groups.append((i, 4))
                i += 4
                continue

        i += 1

    return groups


def _align_antipodal_quaternions(a: np.ndarray, b: np.ndarray,
                                groups: list[tuple[int, int]]) -> np.ndarray:
    """
    For each quaternion block, flip sign of b-block if dot(a,b) < 0.
    """
    b2 = b.copy()
    for s, L in groups:
        qa = a[s:s+L]
        qb = b2[s:s+L]
        if float(np.dot(qa, qb)) < 0.0:
            b2[s:s+L] = -qb
    return b2


def l2_dist(a: np.ndarray, b: np.ndarray, quat_aware: bool = True) -> float:
    """
    Quaternion-aware L2 distance on compact state vectors.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")

    a = a[:n]
    b = b[:n]

    if quat_aware:
        groups = _find_quaternion_groups(a, b)
        if groups:
            b = _align_antipodal_quaternions(a, b, groups)

    return float(np.linalg.norm(a - b))


def _is_unit_quat(q4: np.ndarray, tol: float = 0.20) -> bool:
    q4 = np.asarray(q4, dtype=np.float32).reshape(-1)
    if q4.size != 4:
        return False
    n = float(np.linalg.norm(q4))
    return abs(n - 1.0) <= tol


def _quat_diff_sign_invariant(q1: np.ndarray, q2: np.ndarray) -> float:
    """Sign-invariant quaternion diff magnitude: min(||q1-q2||, ||q1+q2||)."""
    q1 = np.asarray(q1, dtype=np.float32).reshape(4)
    q2 = np.asarray(q2, dtype=np.float32).reshape(4)
    return float(min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2)))


def select_active_lowdim_indices(
    data,
    t_prev: int,
    t_curr: int,
    k: int = 16,
    eps: float = 1e-4,
    delta_min_abs: float = 5e-3,
    rel_frac: float = 0.10,
    min_dims: int = 1,
    quat_tol: float = 0.20,
) -> np.ndarray:
    """
    Pick indices in task_low_dim_state that meaningfully change from t_prev->t_curr.

    Thresholding:
      keep feature if s >= thr
      thr = max(eps, delta_min_abs, rel_frac * max_s)

    Quaternion-aware:
      treat unit-quat 4D blocks as a single feature; if picked, include all 4 indices.

    Returns:
      1D int array of indices (may be empty).
    """
    if "task_low_dim_state" not in data.files:
        return np.zeros((0,), dtype=np.int32)

    a = np.asarray(data["task_low_dim_state"][t_prev], dtype=np.float32).ravel()
    b = np.asarray(data["task_low_dim_state"][t_curr], dtype=np.float32).ravel()
    n = int(min(a.size, b.size))
    if n <= 0:
        return np.zeros((0,), dtype=np.int32)

    # Build features: each entry is (score, idx_list)
    feats: list[tuple[float, list[int]]] = []
    i = 0
    while i < n:
        if i + 4 <= n and _is_unit_quat(a[i:i+4], tol=quat_tol) and _is_unit_quat(b[i:i+4], tol=quat_tol):
            s = _quat_diff_sign_invariant(a[i:i+4], b[i:i+4])
            feats.append((float(s), [i, i+1, i+2, i+3]))
            i += 4
        else:
            s = float(abs(b[i] - a[i]))
            feats.append((float(s), [i]))
            i += 1

    if not feats:
        return np.zeros((0,), dtype=np.int32)

    max_s = float(max(s for s, _ in feats))
    # If nothing really changes, return empty (segment is essentially static in lowdim)
    if not np.isfinite(max_s) or max_s <= float(eps):
        return np.zeros((0,), dtype=np.int32)

    thr = max(float(eps), float(delta_min_abs), float(rel_frac) * max_s)

    # Sort by score descending
    feats.sort(key=lambda x: -x[0])

    picked: list[int] = []
    used = set()

    # Pick features while preserving quaternion blocks and keeping <= k dims
    for s, idxs in feats:
        if s < thr:
            break

        # skip if any idx already used
        if any(int(ii) in used for ii in idxs):
            continue

        # would this exceed k dims?
        if len(picked) + len(idxs) > int(k):
            continue

        picked.extend(idxs)
        for ii in idxs:
            used.add(int(ii))

        if len(picked) >= int(k):
            break

    picked = sorted(picked)
    if len(picked) < int(min_dims):
        return np.zeros((0,), dtype=np.int32)

    return np.asarray(picked, dtype=np.int32)


def lowdim_subset_distance(
    low_now: np.ndarray,
    low_ref: np.ndarray,
    idxs: np.ndarray,
    quat_tol: float = 0.20,
    normalize: bool = True,
) -> float:
    """
    Distance on selected indices only.
    Quaternion-aware if a full 4D block is present in idxs and looks like a unit quat in ref.
    """
    idxs = np.asarray(idxs, dtype=np.int32).ravel()
    if idxs.size == 0:
        return 0.0

    x = np.asarray(low_now, dtype=np.float32).ravel()
    y = np.asarray(low_ref, dtype=np.float32).ravel()
    n = int(min(x.size, y.size))
    idxs = idxs[idxs < n]
    if idxs.size == 0:
        return 0.0

    idx_set = set(int(i) for i in idxs.tolist())
    used = set()
    err2 = 0.0

    # Find quaternion blocks fully covered by idxs
    for i in sorted(idx_set):
        if i in used:
            continue
        block = [i, i+1, i+2, i+3]
        if all(j in idx_set for j in block) and (i + 4) <= n and _is_unit_quat(y[i:i+4], tol=quat_tol):
            d = _quat_diff_sign_invariant(x[i:i+4], y[i:i+4])
            err2 += d * d
            used.update(block)

    # Remaining scalar dims
    for i in idxs.tolist():
        i = int(i)
        if i in used:
            continue
        d = float(x[i] - y[i])
        err2 += d * d

    dist = float(np.sqrt(err2))
    if normalize:
        dist /= float(np.sqrt(max(1, idxs.size)))
    return dist
