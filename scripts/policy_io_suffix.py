#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass(frozen=True)
class SuffixPolicyObsSpec:
    use_q: bool = True
    use_g: bool = True
    use_goal_q: bool = True
    use_goal_g: bool = True
    use_z: bool = False
    use_goal_z: bool = False

    def obs_dim(self, q_dim: int, z_dim: int = 0, g_dim: int = 1) -> int:
        d = 0
        if self.use_q:
            d += q_dim
        if self.use_g:
            d += g_dim
        if self.use_goal_q:
            d += q_dim
        if self.use_goal_g:
            d += g_dim
        if self.use_z:
            d += z_dim
        if self.use_goal_z:
            d += z_dim
        return d


def build_suffix_policy_obs(
    *,
    q_t: np.ndarray,
    g_t: np.ndarray,
    q_goal: np.ndarray,
    g_goal: np.ndarray,
    z_t: Optional[np.ndarray],
    z_goal: Optional[np.ndarray],
    spec: SuffixPolicyObsSpec,
) -> np.ndarray:
    parts = []

    q_t = np.asarray(q_t, dtype=np.float32).reshape(-1)
    g_t = np.asarray(g_t, dtype=np.float32).reshape(-1)
    q_goal = np.asarray(q_goal, dtype=np.float32).reshape(-1)
    g_goal = np.asarray(g_goal, dtype=np.float32).reshape(-1)

    if spec.use_q:
        parts.append(q_t)
    if spec.use_g:
        parts.append(g_t)
    if spec.use_goal_q:
        parts.append(q_goal)
    if spec.use_goal_g:
        parts.append(g_goal)

    if spec.use_z:
        if z_t is None:
            raise ValueError("spec.use_z=True but z_t is None")
        parts.append(np.asarray(z_t, dtype=np.float32).reshape(-1))

    if spec.use_goal_z:
        if z_goal is None:
            raise ValueError("spec.use_goal_z=True but z_goal is None")
        parts.append(np.asarray(z_goal, dtype=np.float32).reshape(-1))

    if not parts:
        raise ValueError("Empty observation spec")

    obs = np.concatenate(parts, axis=0).astype(np.float32)
    return obs


def build_reverse_action_target(
    *,
    q_now: np.ndarray,
    q_prev: np.ndarray,
    g_prev: np.ndarray,
    arm_k: float = 4.0,
    include_gripper_action: bool = True,
) -> np.ndarray:
    arm = np.clip(
        arm_k * (np.asarray(q_prev, dtype=np.float32).reshape(-1) -
                 np.asarray(q_now, dtype=np.float32).reshape(-1)),
        -1.0, 1.0
    ).astype(np.float32)

    if not include_gripper_action:
        return arm

    g_scalar = float(np.asarray(g_prev, dtype=np.float32).reshape(-1)[0])
    g_cmd = np.array([2.0 * g_scalar - 1.0], dtype=np.float32)
    g_cmd = np.clip(g_cmd, -1.0, 1.0)
    return np.concatenate([arm, g_cmd], axis=0)


DEFAULT_SUFFIX_OBS_SPEC = SuffixPolicyObsSpec(
    use_q=True,
    use_g=True,
    use_goal_q=True,
    use_goal_g=True,   # keep this if you believe goal gripper belongs in policy input
    use_z=False,
    use_goal_z=False,
)