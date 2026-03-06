#!/usr/bin/env python3
"""
PLAN-A Step 4: RL wrapper (suffix learning)

Environment:
  - Reset-to-final: restore final snapshot (from _prep.npz)
  - Reset-to-boundary: restore keyframe snapshot at recommended split

Goal:
  - z_goal is z(s) at boundary snapshot
  - Reward drives z(s) toward z_goal

This file intentionally avoids any predicate logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

from z_extractor import load_zspec, ZExtractor, z_distance as z_distance_pose
from rollback_triage import get_pyrep, set_robot_state, settle

def restore_snapshot_no_open_commit(task, env, trees_1d, q: np.ndarray, g: float, settle_steps: int = 10):
    """
    Same as rollback_triage.restore_snapshot, but the first commit step uses the correct gripper command.
    This prevents a one-step 'open' glitch that can change contacts and z(s).
    """
    pr = get_pyrep(env, task)

    task.reset()

    for tree in list(trees_1d):
        pr.set_configuration_tree(tree)

    scene = getattr(task, "_scene", None)
    if scene is None:
        raise RuntimeError("task._scene not available; cannot set robot state.")

    set_robot_state(scene, q, g)

    # commit step WITH consistent gripper command
    a = np.zeros((8,), dtype=np.float32)
    a[-1] = 1.0 if float(g) > 0.5 else 0.0
    obs, _, _ = task.step(a)

    obs = settle(task, float(g), int(settle_steps)) or obs
    return obs

# =========================
# Adapters you already have
# =========================

def restore_snapshot_trees(task, env, trees_1d, q: np.ndarray, g: float, settle_steps: int = 10):
    return restore_snapshot_no_open_commit(
        task, env,
        trees_1d=list(trees_1d),
        q=np.asarray(q, dtype=np.float32),
        g=float(g),
        settle_steps=int(settle_steps),
    )

def compute_z(task, obs, zext: ZExtractor) -> np.ndarray:
    """
    Compute z(s) using your ZExtractor.
    """
    return zext.extract(obs, task=task)


def z_distance(z: np.ndarray, z_goal: np.ndarray, zext: ZExtractor,
              w_quat: float = 0.2, w_grip: float = 0.05) -> float:
    """
    Pose-aware distance using your existing metric and the zspec structure.
    """
    spec = zext.zspec
    return z_distance_pose(
        z, z_goal,
        k_shapes=spec.k_shapes,
        k_joints=spec.k_joints,
        w_quat=float(w_quat),
        w_grip=float(w_grip),
    )


# =========================
# Data passed from Step 3
# =========================

@dataclass
class DemoSnapshots:
    final_snapshot_trees_1d: List[bytes]          # length R
    keyframe_trees_kR: np.ndarray                 # shape (K, R), dtype object(bytes)
    keyframe_indices: np.ndarray                  # shape (K,), timestep index for each keyframe row

    q_ref: np.ndarray                             # (T,7)
    g_ref: np.ndarray                             # (T,)


@dataclass
class SplitSpec:
    # this is a TIMESTEP index (as in consensus_splits.json)
    split_t: int


# =========================
# Environment
# =========================

class ReverseSuffixEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        task_name: str,
        variation: int,
        demo_snaps: DemoSnapshots,
        split_spec: SplitSpec,
        obs_dim: int,
        action_arm_dim: int = 7,
        max_steps: int = 200,
        goal_tol: float = 0.02,
        goal_hold_steps: int = 5,
        reward_mode: str = "shaped",   # "dense" or "shaped"
        include_z_in_obs: bool = False,
        z_dim: Optional[int] = None,
        reset_mode: str = "final",     # "final" or "boundary"
        zspec_json_path: str = "data/demos/BlockPyramid_var00_demo0000_prep_zspec.json",
        settle_steps: int = 10,
        joint_vel_clip: float = 1.0,
        seed: int = 0,
        render: bool = False,
    ):
        super().__init__()

        assert reset_mode in ("final", "boundary")
        assert reward_mode in ("dense", "shaped")

        self.task_name = task_name
        self.variation = int(variation)

        self.demo_snaps = demo_snaps
        self.split_spec = split_spec

        self.max_steps = int(max_steps)
        self.goal_tol = float(goal_tol)
        self.goal_hold_steps = int(goal_hold_steps)
        self.reward_mode = reward_mode
        self.include_z_in_obs = bool(include_z_in_obs)
        self.reset_mode = reset_mode
        self.settle_steps = int(settle_steps)
        self.joint_vel_clip = float(joint_vel_clip)
        self.render = bool(render)

        self._rng = np.random.default_rng(seed)

        # RLBench env setup
        self._env = None
        self._task = None

        # z goal
        self._z_goal = None
        self._z_dim = z_dim

        # episode state
        self._t = 0
        self._hold = 0
        self._d_prev = None

        # Observation space
        base_dim = int(obs_dim)
        if self.include_z_in_obs:
            if self._z_dim is None:
                raise ValueError("z_dim must be provided if include_z_in_obs=True")
            base_dim += int(self._z_dim)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_dim,), dtype=np.float32
        )

        # Action space: arm joint velocities + gripper discrete command encoded as float in [-1, 1]
        # We will map last component to {0, 1} inside step.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(int(action_arm_dim) + 1,), dtype=np.float32
        )

        # Lazily init RLBench to allow VecEnv fork safety
        self._init_rlbench()

        self._zspec = load_zspec(zspec_json_path)
        self._zext = ZExtractor(self._zspec)

        self._last_obs = None
        self._boundary_k_row = None
        self._boundary_t = None
        self._split_t = None


    def _init_rlbench(self):
        if self._env is not None:
            return

        headless = (not self.render)

        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(False)

        action_mode = MoveArmThenGripper(
            arm_action_mode=JointVelocity(),
            gripper_action_mode=Discrete()
        )

        self._env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=headless
        )
        self._env.launch()

        task_cls = getattr(rlbench_tasks, self.task_name)
        self._task = self._env.get_task(task_cls)
        self._task.set_variation(self.variation)

    def close(self):
        if self._env is not None:
            try:
                self._env.shutdown()
            except Exception:
                pass
        self._env = None
        self._task = None

    def _settle(self, n: int):
        # Let physics settle with no-op actions.
        for _ in range(int(n)):
            # 0 joint velocity, open gripper command 0 (or neutral)
            a = np.zeros((8,), dtype=np.float32)
            a[-1] = 1.0  # TODO: why keep gripper closed? What about gripper open settling?
            self._task.step(a)

    def _restore_final(self):
        t = int(self.demo_snaps.q_ref.shape[0] - 1)
        self._last_obs = restore_snapshot_trees(
            self._task, self._env,
            trees_1d=self.demo_snaps.final_snapshot_trees_1d,
            q=self.demo_snaps.q_ref[t],
            g=float(self.demo_snaps.g_ref[t]),
            settle_steps=self.settle_steps,
        )

    def _restore_boundary(self):
        split_t = int(self.split_spec.split_t)

        kf = np.asarray(self.demo_snaps.keyframe_indices, dtype=np.int64).ravel()
        if kf.size == 0:
            raise RuntimeError("keyframe_indices is empty; cannot map split_t to a keyframe.")

        # nearest keyframe in this demo
        k_row = int(np.argmin(np.abs(kf - split_t)))
        t = int(kf[k_row])

        trees_1d = list(self.demo_snaps.keyframe_trees_kR[k_row, :].tolist())

        self._last_obs = restore_snapshot_trees(
            self._task, self._env,
            trees_1d=trees_1d,
            q=self.demo_snaps.q_ref[t],
            g=float(self.demo_snaps.g_ref[t]),
            settle_steps=self.settle_steps,
        )

        # optional: keep for logging/debug
        self._boundary_k_row = k_row
        self._boundary_t = t
        self._split_t = split_t


    def _set_goal_from_boundary(self):
        # restore boundary, set goal z from that exact restored state
        self._restore_boundary()
        if self._last_obs is None:
            raise RuntimeError("restore_boundary did not produce an observation.")
        z = compute_z(self._task, self._last_obs, self._zext)
        self._z_goal = np.asarray(z, dtype=np.float32).copy()

    def _make_obs(self) -> Tuple[np.ndarray, float]:
        if self._z_goal is None:
            raise RuntimeError("z_goal is None. Did reset() call _set_goal_from_boundary()?")
        obs = self._last_obs
        if obs is None:
            # fallback (should not happen if restore_* set _last_obs)
            obs, _, _ = self._task.step(np.zeros((8,), dtype=np.float32))

        # low-dim extraction (your inline version)
        parts = []
        if hasattr(obs, "joint_positions") and obs.joint_positions is not None:
            parts.append(np.asarray(obs.joint_positions, dtype=np.float32).ravel())
        if hasattr(obs, "joint_velocities") and obs.joint_velocities is not None:
            parts.append(np.asarray(obs.joint_velocities, dtype=np.float32).ravel())
        if hasattr(obs, "gripper_open") and obs.gripper_open is not None:
            parts.append(np.asarray([obs.gripper_open], dtype=np.float32))
        x = np.concatenate(parts, axis=0).astype(np.float32)

        z = compute_z(self._task, obs, self._zext)
        d = z_distance(z, self._z_goal, self._zext)

        if self.include_z_in_obs:
            x = np.concatenate([x, np.asarray(z, dtype=np.float32).ravel()], axis=0)

        return x, float(d)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._t = 0
        self._hold = 0
        self._d_prev = None

        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        if self.reset_mode == "final":
            # 1) define goal for *this* episode from boundary restore
            self._set_goal_from_boundary()

            # 2) now restore the actual episode start (final snapshot)
            self._restore_final()

        else:
            # boundary reset mode means: start at boundary
            # define goal from the same exact state, so d=0 by construction
            self._set_goal_from_boundary()

        x, d = self._make_obs()
        self._d_prev = d

        info = {
            "d": d,
            "goal_tol": self.goal_tol,
            "split_t": int(self._split_t) if self._split_t is not None else None,
            "boundary_t": int(self._boundary_t) if self._boundary_t is not None else None,
            "boundary_k_row": int(self._boundary_k_row) if self._boundary_k_row is not None else None,
            "reset_mode": self.reset_mode,
        }
        return x, info

    def step(self, action: np.ndarray):
        self._t += 1

        a = np.asarray(action, dtype=np.float32).ravel()
        if a.shape[0] != 8:
            raise ValueError(f"Expected action dim 8 (7 arm + 1 grip), got {a.shape[0]}")

        arm = np.clip(a[:7], -1.0, 1.0) * self.joint_vel_clip

        # Map last scalar to discrete {0,1}
        # Convention: < 0 => open(0), >= 0 => close(1)
        grip = 1.0 if a[7] >= 0.0 else 0.0

        rlbench_action = np.concatenate([arm, np.asarray([grip], dtype=np.float32)], axis=0)

        self._last_obs, _, _ = self._task.step(rlbench_action)
        x, d = self._make_obs()

        # Termination logic
        if d <= self.goal_tol:
            self._hold += 1
        else:
            self._hold = 0

        terminated = (self._hold >= self.goal_hold_steps)
        truncated = (self._t >= self.max_steps)

        # Reward
        if self.reward_mode == "dense":
            reward = -d
        else:
            # shaped: positive when you reduce distance
            reward = (self._d_prev - d) if self._d_prev is not None else -d

        self._d_prev = d

        info = {
            "t": self._t,
            "d": d,
            "hold": self._hold,
            "success": bool(terminated),
        }

        return x, float(reward), bool(terminated), bool(truncated), info