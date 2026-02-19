#!/usr/bin/env python3
"""
snapshot_utils.py

Utilities for:
- loading PLAN-A recorded npz (with snapshots)
- restoring keyframe/post snapshots into an RLBench env
- estimating snapshot<->timestep alignment offset (common off-by-one)

Assumptions:
- npz contains:
    snapshot_model_names          (M,)
    snapshot_keyframe_trees       (K,M) object bytes
    snapshot_post_trees           (M,) object bytes
    keyframe_indices              (K,)
    joint_positions               (T,7)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# IO
# -----------------------------
def load_demo_npz(path: str) -> np.lib.npyio.NpzFile:
    data = np.load(path, allow_pickle=True)
    required = ["snapshot_model_names", "snapshot_keyframe_trees", "snapshot_post_trees", "keyframe_indices", "joint_positions"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(f"Missing keys in {path}: {missing}")
    return data


# -----------------------------
# PyRep / Observation helpers
# -----------------------------
def get_pyrep(env, task):
    # Prefer scene._pyrep if exposed; else env._pyrep
    scene = getattr(task, "_scene", None)
    if scene is not None and hasattr(scene, "_pyrep"):
        pyrep = getattr(scene, "_pyrep")
        if pyrep is not None:
            return pyrep
    pyrep = getattr(env, "_pyrep", None)
    if pyrep is None:
        raise RuntimeError("Could not find PyRep handle (scene._pyrep or env._pyrep).")
    return pyrep


def get_observation_strict(task):
    """
    Prefer scene-level getter (reflects simulator state right after set_configuration_tree).
    """
    scene = getattr(task, "_scene", None)
    if scene is not None:
        if hasattr(scene, "_get_observation"):
            return scene._get_observation()
        if hasattr(scene, "get_observation"):
            return scene.get_observation()
    if hasattr(task, "get_observation"):
        return task.get_observation()
    raise RuntimeError("No observation getter found (scene/task).")


def get_top_level_models(pyrep):
    roots = pyrep.get_objects_in_tree(root_object=None, first_generation_only=True)
    models = [o for o in roots if o.is_model()]
    models.sort(key=lambda o: o.get_name())
    return models


def _restore_with_models_by_name(pyrep, model_names: List[str], row_bytes: List[bytes]) -> None:
    """
    Apply a snapshot row.

    IMPORTANT:
    Your build seems happiest with pyrep.set_configuration_tree(bytes).
    We still keep a fallback to model.set_configuration_tree(bytes).
    """
    models = get_top_level_models(pyrep)
    name2m = {m.get_name(): m for m in models}

    missing = [n for n in model_names if n not in name2m]
    if missing:
        raise RuntimeError("Missing models in current scene (first 20): " + ", ".join(missing[:20]))

    for name, b in zip(model_names, row_bytes):
        if not isinstance(b, (bytes, bytearray)) or len(b) <= 1:
            raise RuntimeError(f"Bad snapshot bytes for model '{name}' (type={type(b)}, len={len(b) if hasattr(b,'__len__') else 'NA'})")

        try:
            pyrep.set_configuration_tree(b)
        except Exception:
            name2m[name].set_configuration_tree(b)


def restore_row(pyrep, model_names: List[str], row_bytes: List[bytes], settle_steps: int = 0) -> None:
    _restore_with_models_by_name(pyrep, model_names, row_bytes)
    for _ in range(int(settle_steps)):
        pyrep.step()


# -----------------------------
# Alignment
# -----------------------------
def best_matching_t_for_joints(q_now: np.ndarray, q_all: np.ndarray) -> int:
    """
    q_now: (7,)
    q_all: (T,7)
    """
    q_now = np.asarray(q_now, dtype=np.float64).reshape(1, -1)
    q_all = np.asarray(q_all, dtype=np.float64)
    errs = np.linalg.norm(q_all - q_now, axis=1)
    return int(np.argmin(errs))


def estimate_snapshot_offset(
    env,
    task,
    data: np.lib.npyio.NpzFile,
    kf_sample_count: int = 5,
    settle_steps: int = 0,
) -> int:
    """
    Estimate a single global offset 'off' such that:
      snapshot at keyframe t  corresponds to recorded timestep (t + off) approximately.

    We compute offsets on a few keyframes (excluding the final keyframe if it is T-1),
    then take the mode (most common).
    """
    pyrep = get_pyrep(env, task)

    model_names = data["snapshot_model_names"].tolist()
    kf_ts = data["keyframe_indices"].astype(int).tolist()
    kf_trees = data["snapshot_keyframe_trees"]
    q_all = data["joint_positions"]

    T = int(q_all.shape[0])

    # Exclude final keyframe if it’s exactly T-1 (often special-cased by recorder)
    candidates = [(i, int(t)) for i, t in enumerate(kf_ts)]
    if candidates and candidates[-1][1] == (T - 1):
        candidates = candidates[:-1]

    if not candidates:
        return 0

    # Pick evenly spaced samples among candidates
    S = min(kf_sample_count, len(candidates))
    idxs = np.linspace(0, len(candidates) - 1, num=S, dtype=int).tolist()

    offsets = []
    for j in idxs:
        kf_index, t_req = candidates[j]
        row = kf_trees[kf_index, :].tolist()

        _ = task.reset()
        for _ in range(3):
            pyrep.step()

        restore_row(pyrep, model_names, row, settle_steps=settle_steps)
        obs = get_observation_strict(task)

        t_best = best_matching_t_for_joints(obs.joint_positions, q_all)
        offsets.append(int(t_best - t_req))

    # mode (most common). break ties by choosing the median-ish.
    vals, counts = np.unique(np.array(offsets, dtype=int), return_counts=True)
    best = int(vals[np.argmax(counts)])
    return best


# -----------------------------
# High-level restore
# -----------------------------
def restore_keyframe(
    env,
    task,
    data: np.lib.npyio.NpzFile,
    kf_index: int,
    settle_steps: int = 0,
    snap_offset: int = 0,
) -> int:
    """
    Restore the snapshot for keyframe index `kf_index`.

    Returns the *effective* recorded timestep associated with this restored state:
      t_eff = clamp(keyframe_indices[kf_index] + snap_offset, 0, T-1)
    """
    pyrep = get_pyrep(env, task)

    model_names = data["snapshot_model_names"].tolist()
    kf_ts = data["keyframe_indices"].astype(int)
    kf_trees = data["snapshot_keyframe_trees"]
    q_all = data["joint_positions"]
    T = int(q_all.shape[0])

    if kf_index < 0 or kf_index >= int(kf_ts.shape[0]):
        raise ValueError(f"kf_index out of range: {kf_index} (K={int(kf_ts.shape[0])})")

    t_req = int(kf_ts[kf_index])
    t_eff = int(np.clip(t_req + int(snap_offset), 0, T - 1))

    row = kf_trees[kf_index, :].tolist()

    _ = task.reset()
    for _ in range(5):
        pyrep.step()

    restore_row(pyrep, model_names, row, settle_steps=settle_steps)
    return t_eff


def restore_post(
    env,
    task,
    data: np.lib.npyio.NpzFile,
    settle_steps: int = 0,
) -> int:
    """
    Restore post snapshot. Returns t_post = T-1.
    """
    pyrep = get_pyrep(env, task)

    model_names = data["snapshot_model_names"].tolist()
    post_row = data["snapshot_post_trees"].tolist()

    q_all = data["joint_positions"]
    T = int(q_all.shape[0])
    t_post = T - 1

    _ = task.reset()
    for _ in range(5):
        pyrep.step()

    restore_row(pyrep, model_names, post_row, settle_steps=settle_steps)
    return t_post
