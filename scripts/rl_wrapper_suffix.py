#!/usr/bin/env python3
"""
PLAN-A Step 4: RL wrapper (suffix learning)

Environment:
  - Reset-to-final: restore final snapshot (from _prep.npz)
  - Reset-to-boundary: restore keyframe snapshot at recommended split

Goal:
  - z_goal is z(s) at boundary snapshot
  - Reward drives z(s) toward z_goal

Policy observation:
  - Built ONLY through policy_io_suffix.py so BC and RL share one contract
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

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

from policy_io_suffix import (
    DEFAULT_SUFFIX_OBS_SPEC,
    SuffixPolicyObsSpec,
    build_suffix_policy_obs,
)


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

    a = np.zeros((8,), dtype=np.float32)
    a[-1] = 1.0 if float(g) > 0.5 else 0.0
    obs, _, _ = task.step(a)

    obs = settle(task, float(g), int(settle_steps)) or obs
    return obs


def restore_snapshot_trees(task, env, trees_1d, q: np.ndarray, g: float, settle_steps: int = 10):
    return restore_snapshot_no_open_commit(
        task,
        env,
        trees_1d=list(trees_1d),
        q=np.asarray(q, dtype=np.float32),
        g=float(g),
        settle_steps=int(settle_steps),
    )


def compute_z(task, obs, zext: ZExtractor) -> np.ndarray:
    return zext.extract(obs, task=task)


def z_distance(
    z: np.ndarray,
    z_goal: np.ndarray,
    zext: ZExtractor,
    w_quat: float = 0.2,
    w_grip: float = 0.05,
) -> float:
    spec = zext.zspec
    return z_distance_pose(
        z,
        z_goal,
        k_shapes=spec.k_shapes,
        k_joints=spec.k_joints,
        w_quat=float(w_quat),
        w_grip=float(w_grip),
    )


@dataclass
class DemoSnapshots:
    final_snapshot_trees_1d: List[bytes]
    final_gripper_open: float
    final_settle_steps: int

    keyframe_trees_kR: np.ndarray
    keyframe_indices: np.ndarray

    q_ref: np.ndarray
    g_ref: np.ndarray


@dataclass
class SplitSpec:
    split_t: int


class ReverseSuffixEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        task_name: str,
        variation: int,
        demo_snaps: DemoSnapshots,
        split_spec: SplitSpec,
        obs_spec: Optional[SuffixPolicyObsSpec] = None,
        obs_dim: Optional[int] = None,            # kept only for backward compatibility
        action_arm_dim: int = 7,
        max_steps: int = 200,
        goal_tol: float = 0.02,
        goal_hold_steps: int = 5,
        reward_mode: str = "shaped",
        include_z_in_obs: bool = False,           # kept only for backward compatibility
        z_dim: Optional[int] = None,
        reset_mode: str = "final",
        zspec_json_path: str = "data/demos/BlockPyramid_var00_demo0000_prep_zspec.json",
        settle_steps: int = 20,
        joint_vel_clip: float = 1.0,
        seed: int = 0,
        render: bool = False,
        time_penalty: float = 0.01,
        action_penalty: float = 0.0005,
        grip_toggle_penalty: float = 0.001,
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
        self.reset_mode = reset_mode
        self.settle_steps = int(settle_steps)
        self.joint_vel_clip = float(joint_vel_clip)
        self.render_enabled = bool(render)

        self._rng = np.random.default_rng(seed)

        self.time_penalty = float(time_penalty)
        self.action_penalty = float(action_penalty)
        self.grip_toggle_penalty = float(grip_toggle_penalty)
        self._prev_grip_cmd = None

        self._env = None
        self._task = None

        self._z_goal = None
        self._z_dim = z_dim

        self._q_goal = None
        self._g_goal = None

        self._t = 0
        self._hold = 0
        self._d_prev = None

        self._last_obs = None
        self._boundary_k_row = None
        self._boundary_t = None
        self._split_t = None

        self.obs_spec = obs_spec if obs_spec is not None else DEFAULT_SUFFIX_OBS_SPEC

        if include_z_in_obs and not (self.obs_spec.use_z or self.obs_spec.use_goal_z):
            raise ValueError(
                "include_z_in_obs=True but obs_spec does not include z. "
                "Use obs_spec to control policy observation contents."
            )

        if (self.obs_spec.use_z or self.obs_spec.use_goal_z) and self._z_dim is None:
            raise ValueError(
                "obs_spec requests z in the policy observation, but z_dim is None."
            )

        self._q_dim = 7
        self._g_dim = 1
        self.obs_dim = self.obs_spec.obs_dim(
            q_dim=self._q_dim,
            z_dim=int(self._z_dim or 0),
            g_dim=self._g_dim,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(int(action_arm_dim) + 1,),
            dtype=np.float32,
        )

        self._init_rlbench()

        self._zspec = load_zspec(zspec_json_path)
        self._zext = ZExtractor(self._zspec)

        if obs_dim is not None and int(obs_dim) != int(self.obs_dim):
            print(
                f"[warn] ReverseSuffixEnv received obs_dim={obs_dim}, "
                f"but shared obs_spec implies obs_dim={self.obs_dim}. "
                f"Ignoring passed obs_dim."
            )

    def _init_rlbench(self):
        if self._env is not None:
            return

        headless = not self.render_enabled

        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(False)

        action_mode = MoveArmThenGripper(
            arm_action_mode=JointVelocity(),
            gripper_action_mode=Discrete(),
        )

        self._env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=headless,
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

    def render(self):
        return None

    def _extract_qg_from_obs(self, obs) -> Tuple[np.ndarray, np.ndarray]:
        if not hasattr(obs, "joint_positions") or obs.joint_positions is None:
            raise RuntimeError("Observation does not contain joint_positions.")
        q = np.asarray(obs.joint_positions, dtype=np.float32).ravel()
        if q.shape[0] != 7:
            raise RuntimeError(f"Expected 7 joint positions, got shape {q.shape}")

        if not hasattr(obs, "gripper_open") or obs.gripper_open is None:
            raise RuntimeError("Observation does not contain gripper_open.")
        g = np.asarray([obs.gripper_open], dtype=np.float32).ravel()

        return q, g

    def _restore_final(self):
        t = int(self.demo_snaps.q_ref.shape[0] - 1)
        self._last_obs = restore_snapshot_trees(
            self._task,
            self._env,
            trees_1d=self.demo_snaps.final_snapshot_trees_1d,
            q=self.demo_snaps.q_ref[t],
            g=float(self.demo_snaps.final_gripper_open),
            settle_steps=int(self.demo_snaps.final_settle_steps),
        )

    def _restore_boundary(self):
        split_t = int(self.split_spec.split_t)

        kf = np.asarray(self.demo_snaps.keyframe_indices, dtype=np.int64).ravel()
        if kf.size == 0:
            raise RuntimeError("keyframe_indices is empty; cannot map split_t to a keyframe.")

        k_row = int(np.argmin(np.abs(kf - split_t)))
        t = int(kf[k_row])

        trees_1d = list(self.demo_snaps.keyframe_trees_kR[k_row, :].tolist())

        self._last_obs = restore_snapshot_trees(
            self._task,
            self._env,
            trees_1d=trees_1d,
            q=self.demo_snaps.q_ref[t],
            g=float(self.demo_snaps.g_ref[t]),
            settle_steps=self.settle_steps,
        )

        self._boundary_k_row = k_row
        self._boundary_t = t
        self._split_t = split_t

    def _set_goal_from_boundary(self):
        self._restore_boundary()
        if self._last_obs is None:
            raise RuntimeError("restore_boundary did not produce an observation.")

        self._z_goal = np.asarray(
            compute_z(self._task, self._last_obs, self._zext),
            dtype=np.float32,
        ).copy()

        q_goal, g_goal = self._extract_qg_from_obs(self._last_obs)
        self._q_goal = q_goal.copy()
        self._g_goal = g_goal.copy()

    def _make_obs(self) -> Tuple[np.ndarray, float]:
        if self._z_goal is None:
            raise RuntimeError("z_goal is None. Did reset() call _set_goal_from_boundary()?")

        if self._q_goal is None or self._g_goal is None:
            raise RuntimeError("Goal q/g are None. Did reset() call _set_goal_from_boundary()?")

        obs = self._last_obs
        if obs is None:
            obs, _, _ = self._task.step(np.zeros((8,), dtype=np.float32))

        q_t, g_t = self._extract_qg_from_obs(obs)

        z_t = None
        z_goal = None
        if self.obs_spec.use_z or self.obs_spec.use_goal_z:
            z_t = np.asarray(compute_z(self._task, obs, self._zext), dtype=np.float32).ravel()
            z_goal = np.asarray(self._z_goal, dtype=np.float32).ravel()

        x = build_suffix_policy_obs(
            q_t=q_t,
            g_t=g_t,
            q_goal=self._q_goal,
            g_goal=self._g_goal,
            z_t=z_t,
            z_goal=z_goal,
            spec=self.obs_spec,
        ).astype(np.float32)

        z_now = np.asarray(compute_z(self._task, obs, self._zext), dtype=np.float32)
        d = z_distance(z_now, self._z_goal, self._zext)

        return x, float(d)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._t = 0
        self._hold = 0
        self._d_prev = None

        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        if self.reset_mode == "final":
            self._set_goal_from_boundary()
            self._restore_final()
        else:
            self._set_goal_from_boundary()

        x, d = self._make_obs()
        self._d_prev = d
        self._prev_grip_cmd = 1.0

        info = {
            "d": d,
            "goal_tol": self.goal_tol,
            "split_t": int(self._split_t) if self._split_t is not None else None,
            "boundary_t": int(self._boundary_t) if self._boundary_t is not None else None,
            "boundary_k_row": int(self._boundary_k_row) if self._boundary_k_row is not None else None,
            "reset_mode": self.reset_mode,
            "obs_dim": int(self.obs_dim),
        }
        return x, info

    def step(self, action: np.ndarray):
        self._t += 1

        a = np.asarray(action, dtype=np.float32).ravel()
        if a.shape[0] != 8:
            raise ValueError(f"Expected action dim 8 (7 arm + 1 grip), got {a.shape[0]}")

        arm = np.clip(a[:7], -1.0, 1.0) * self.joint_vel_clip

        prev_grip = self._prev_grip_cmd
        if prev_grip is None:
            prev_grip = 1.0

        if a[7] > 0.5:
            grip = 1.0
        elif a[7] < -0.5:
            grip = 0.0
        else:
            grip = prev_grip

        rlbench_action = np.concatenate([arm, np.asarray([grip], dtype=np.float32)], axis=0)

        self._last_obs, _, _ = self._task.step(rlbench_action)
        x, d = self._make_obs()

        if d <= self.goal_tol:
            self._hold += 1
        else:
            self._hold = 0

        terminated = self._hold >= self.goal_hold_steps
        truncated = self._t >= self.max_steps

        if self.reward_mode == "dense":
            reward = -d
        else:
            reward = (self._d_prev - d) if self._d_prev is not None else -d

        arm_cost = float(np.mean(np.square(arm)))
        reward -= self.time_penalty
        reward -= self.action_penalty * arm_cost

        if grip != prev_grip:
            reward -= self.grip_toggle_penalty

        self._prev_grip_cmd = grip
        self._d_prev = d

        info = {
            "t": self._t,
            "d": d,
            "hold": self._hold,
            "success": bool(terminated),
            "obs_dim": int(self.obs_dim),
        }

        return x, float(reward), bool(terminated), bool(truncated), info