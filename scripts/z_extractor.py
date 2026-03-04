#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from pyrep.objects.object import Object
from pyrep.objects.joint import Joint
from pyrep.backend import sim as sim_backend


@dataclass
class ZSpec:
    task: str
    variation: int
    shapes: List[str]
    joints: List[str]

    @property
    def k_shapes(self) -> int:
        return len(self.shapes)

    @property
    def k_joints(self) -> int:
        return len(self.joints)

    @property
    def z_dim(self) -> int:
        return 8 + 7 * self.k_shapes + 1 * self.k_joints


def load_zspec(path: str) -> ZSpec:
    with open(path, "r") as f:
        d = json.load(f)
    return ZSpec(
        task=str(d.get("task", "")),
        variation=int(d.get("variation", 0)),
        shapes=[str(x) for x in d.get("shapes", [])],
        joints=[str(x) for x in d.get("joints", [])],
    )


def _quat_dist(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = np.asarray(q1, dtype=np.float64).ravel()
    q2 = np.asarray(q2, dtype=np.float64).ravel()
    if q1.size != 4 or q2.size != 4:
        return 0.0
    d = float(abs(np.dot(q1, q2)))
    d = max(0.0, min(1.0, d))
    return 1.0 - d


def pose_dist(p1: np.ndarray, p2: np.ndarray, w_quat: float = 0.2) -> float:
    p1 = np.asarray(p1, dtype=np.float64).ravel()
    p2 = np.asarray(p2, dtype=np.float64).ravel()
    if p1.size < 7 or p2.size < 7:
        return 0.0
    dp = float(np.linalg.norm(p1[:3] - p2[:3]))
    dq = _quat_dist(p1[3:7], p2[3:7])
    return dp + w_quat * dq


class ZExtractor:
    """
    Extract z(s) from an RLBench task state using a ZSpec.

    z = [gripper_pose(7), gripper_open(1), shape_pose(7)*K, joint_pos(1)*J]

    IMPORTANT:
    - Objects in RLBench can be recreated/reset. So we lazily (re)resolve names.
    - We NEVER cache missing objects as None forever.
    """

    def __init__(self, zspec: ZSpec):
        self.zspec = zspec
        self._shape_objs: Dict[str, Optional[Object]] = {}
        self._joint_objs: Dict[str, Optional[Joint]] = {}

    def _resolve_shape(self, name: str) -> Optional[Object]:
        obj = self._shape_objs.get(name, None)
        if obj is not None:
            return obj
        try:
            obj = Object.get_object(name)
            self._shape_objs[name] = obj
            return obj
        except Exception:
            # do not lock in failure
            self._shape_objs[name] = None
            return None

    def _resolve_joint(self, name: str) -> Optional[Joint]:
        j = self._joint_objs.get(name, None)
        if j is not None:
            return j
        try:
            j = Joint.get_object(name)
            self._joint_objs[name] = j
            return j
        except Exception:
            self._joint_objs[name] = None
            return None

    def extract(self, obs, task=None, retry_missing: bool = True) -> np.ndarray:
        # gripper pose
        gpose = None
        if hasattr(obs, "gripper_pose") and obs.gripper_pose is not None:
            gpose = np.asarray(obs.gripper_pose, dtype=np.float32).ravel()

        if gpose is None or gpose.size < 7:
            # fallback to robot tip pose
            if task is None:
                raise RuntimeError("obs.gripper_pose missing and no task provided for fallback.")
            scene = getattr(task, "_scene", None)
            if scene is None:
                raise RuntimeError("Cannot access task._scene for gripper pose fallback.")
            tip = scene.robot.arm.get_tip()
            gpose = np.asarray(tip.get_pose(), dtype=np.float32).ravel()

        # gripper open
        gopen = 0.0
        if hasattr(obs, "gripper_open") and obs.gripper_open is not None:
            go = np.asarray(obs.gripper_open, dtype=np.float32).ravel()
            if go.size > 0:
                gopen = float(go[0])

        parts: List[np.ndarray] = []
        parts.append(gpose[:7].astype(np.float32))
        parts.append(np.asarray([gopen], dtype=np.float32))

        # shapes
        for name in self.zspec.shapes:
            obj = self._resolve_shape(name)
            if obj is None and retry_missing:
                # try again once (objects often appear after first sim step)
                obj = self._resolve_shape(name)

            if obj is None:
                parts.append(np.zeros((7,), dtype=np.float32))
                continue

            try:
                parts.append(np.asarray(obj.get_pose(), dtype=np.float32).ravel()[:7])
            except Exception:
                # if the cached handle became invalid, clear it and output zeros
                self._shape_objs[name] = None
                parts.append(np.zeros((7,), dtype=np.float32))

        # joints
        for name in self.zspec.joints:
            j = self._resolve_joint(name)
            if j is None and retry_missing:
                j = self._resolve_joint(name)

            if j is None:
                parts.append(np.zeros((1,), dtype=np.float32))
                continue

            try:
                parts.append(np.asarray([float(sim_backend.simGetJointPosition(j.get_handle()))], dtype=np.float32))
            except Exception:
                self._joint_objs[name] = None
                parts.append(np.zeros((1,), dtype=np.float32))

        z = np.concatenate(parts, axis=0)
        return z


def z_distance(z1: np.ndarray, z2: np.ndarray, k_shapes: int, k_joints: int,
              w_quat: float = 0.2, w_grip: float = 0.05) -> float:
    """
    Pose-aware distance for z vectors.
    Structure:
      gripper_pose(7), gripper_open(1), shape_pose(7)*K, joint_pos(1)*J
    """
    z1 = np.asarray(z1, dtype=np.float64).ravel()
    z2 = np.asarray(z2, dtype=np.float64).ravel()

    idx = 0
    d = 0.0

    d += pose_dist(z1[idx:idx+7], z2[idx:idx+7], w_quat=w_quat)
    idx += 7

    d += w_grip * float(abs(z1[idx] - z2[idx]))
    idx += 1

    for _ in range(int(k_shapes)):
        d += pose_dist(z1[idx:idx+7], z2[idx:idx+7], w_quat=w_quat)
        idx += 7

    for _ in range(int(k_joints)):
        d += float(abs(z1[idx] - z2[idx]))
        idx += 1

    return float(d)