#!/usr/bin/env python3
import os
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

from pyrep.objects.joint import Joint


# ----------------------------
# Predicate parsing (minimal)
# ----------------------------
@dataclass
class Atom:
    neg: bool
    pred: str
    arg: str  # single-arg for now: open(x)

def parse_atoms(set_str: str) -> List[Atom]:
    """
    Parse strings like:
      "{open(drawer), empty(gripper)}"
      "{not(open(drawer))}"
    We only *evaluate* open(x)/not(open(x)) for now.
    """
    s = (set_str or "").strip()
    if not s:
        return []
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[Atom] = []
    for p in parts:
        neg = False
        if p.startswith("not(") and p.endswith(")"):
            neg = True
            p = p[4:-1].strip()

        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\(([^()]+)\)$", p)
        if not m:
            continue
        pred = m.group(1).strip()
        arg = m.group(2).strip()
        out.append(Atom(neg=neg, pred=pred, arg=arg))
    return out


# ----------------------------
# Robust PyRep access
# ----------------------------
def get_pyrep(env: Environment, task_env) -> object:
    """
    RLBench usually exposes pyrep as env._pyrep (after launch).
    Scene is under task_env._scene, which may have _pyrep.
    """
    pyrep = getattr(env, "_pyrep", None)
    if pyrep is not None:
        return pyrep

    scene = getattr(task_env, "_scene", None)
    pyrep = getattr(scene, "_pyrep", None) if scene is not None else None
    if pyrep is not None:
        return pyrep

    # Some builds keep it on env._scene (rare), try anyway:
    scene2 = getattr(env, "_scene", None)
    pyrep = getattr(scene2, "_pyrep", None) if scene2 is not None else None
    if pyrep is not None:
        return pyrep

    raise RuntimeError("Could not access PyRep instance (needed for snapshot restore).")


# ----------------------------
# Open-map handling
# ----------------------------
@dataclass
class OpenSpec:
    joint_name: str
    threshold: float
    direction: str  # ">" or "<"
    joint: Joint

def load_open_map(pyrep, open_map_path: Optional[str]) -> Dict[str, OpenSpec]:
    if not open_map_path:
        return {}
    with open(open_map_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    out: Dict[str, OpenSpec] = {}
    for sym, spec in cfg.items():
        jname = spec["joint"]
        thr = float(spec.get("threshold", 0.0))
        direc = str(spec.get("direction", ">")).strip()
        # Joint can be constructed by name
        j = Joint(jname)
        out[sym] = OpenSpec(joint_name=jname, threshold=thr, direction=direc, joint=j)
    return out

def is_open(open_spec: OpenSpec) -> Tuple[bool, float]:
    pos = float(open_spec.joint.get_joint_position())
    if open_spec.direction == ">":
        return (pos > open_spec.threshold), pos
    else:
        return (pos < open_spec.threshold), pos


# ----------------------------
# Env
# ----------------------------
class ReverseSkillEnv(gym.Env):
    """
    Reset uses curriculum snapshots saved in demo_npz:
      - keyframe snapshots (K,M) in snapshot_keyframe_trees
      - post snapshot (M,) in snapshot_post_trees

    Goal: reach CORE PRE (forward) starting from CORE POST or intermediate snapshots (reverse skill).
    We currently evaluate only open(x)/not(open(x)) atoms for termination/reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        demo_npz_path: str,
        task_name: str,
        variation: int = 0,
        headless: bool = True,
        max_episode_steps: int = 150,
        settle_steps_after_restore: int = 10,
        open_map_path: Optional[str] = None,
    ):
        super().__init__()
        self.demo_npz_path = demo_npz_path
        self.task_name = task_name
        self.variation = int(variation)
        self.headless = bool(headless)
        self.max_episode_steps = int(max_episode_steps)
        self.settle_steps_after_restore = int(settle_steps_after_restore)

        self.curriculum_level = 1.0  # 0=easiest(start near pre), 1=hardest(start from post)
        self._episode_step = 0

        # ---- load demo npz (snapshots + predicates) ----
        self.demo = np.load(self.demo_npz_path, allow_pickle=True)

        self.keyframe_indices = self.demo["keyframe_indices"].astype(np.int32)
        self.K = int(self.keyframe_indices.shape[0])

        self.snapshot_storage = str(self.demo["snapshot_storage"][0]) if "snapshot_storage" in self.demo else "unknown"
        if self.snapshot_storage not in ("bytes_v1", "bytes"):
            # still try to proceed, but expect bytes-like cells
            pass

        self.snapshot_keyframe_trees = self.demo["snapshot_keyframe_trees"]  # (K,M) object bytes
        self.snapshot_post_trees = self.demo["snapshot_post_trees"]          # (M,) object bytes

        self.M = int(self.snapshot_post_trees.shape[0])

        self.pre_core = str(self.demo["preconditions_core"][0]) if "preconditions_core" in self.demo else ""
        self.post_core = str(self.demo["postconditions_core"][0]) if "postconditions_core" in self.demo else ""

        self.target_atoms = [a for a in parse_atoms(self.pre_core) if a.pred == "open"]

        # ---- build RLBench env ----
        obs_config = ObservationConfig()
        obs_config.set_all(False)
        obs_config.joint_positions = True
        obs_config.joint_velocities = True
        obs_config.gripper_open = True
        obs_config.gripper_pose = True

        action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
        self._rlbench_env = Environment(action_mode, obs_config=obs_config, headless=self.headless)
        self._rlbench_env.launch()

        if not hasattr(rlbench_tasks, self.task_name):
            raise ValueError(f"Unknown RLBench task '{self.task_name}'")
        task_cls = getattr(rlbench_tasks, self.task_name)

        self._task = self._rlbench_env.get_task(task_cls)
        self._task.set_variation(self.variation)

        self._pyrep = get_pyrep(self._rlbench_env, self._task)

        # open(x) evaluator
        self.open_map = load_open_map(self._pyrep, open_map_path)

        # ---- gym spaces ----
        # obs = [q(7), dq(7), gripper_open(1), gripper_pose(7), open_features(len(open_map))*2]
        # open_features: for each symbol -> [is_open, joint_pos]
        self._open_syms = sorted(list(self.open_map.keys()))
        obs_dim = 7 + 7 + 1 + 7 + 2 * len(self._open_syms)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # action: 7 joint velocities + 1 gripper (sign -> discrete)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    # ----------------------------
    # Curriculum API (called via env_method)
    # ----------------------------
    def set_curriculum(self, level: float):
        self.curriculum_level = float(np.clip(level, 0.0, 1.0))

    # ----------------------------
    # Snapshot restore
    # ----------------------------
    def _restore_snapshot_bytes(self, trees_1d: np.ndarray):
        """
        trees_1d: shape (M,) object array of bytes-like
        """
        # Reset first to ensure scene objects exist and physics is clean
        self._task.reset()
        self._pyrep.step()

        for cell in trees_1d.tolist():
            # cell may be bytes or np.bytes_
            b = bytes(cell)
            self._pyrep.set_configuration_tree(b)

        for _ in range(self.settle_steps_after_restore):
            self._pyrep.step()

    def _choose_reset_source(self) -> Tuple[str, Optional[int], int]:
        """
        Returns (label, keyframe_row or None, forward_t)
        label in {"post", "keyframe"}
        """
        # target forward_t = curriculum_level * (T-1), but we only have keyframes.
        # Use post snapshot for near-1, otherwise nearest keyframe.
        if self.curriculum_level >= 0.999:
            return "post", None, int(self.keyframe_indices[-1])

        target_t = int(round(self.curriculum_level * float(max(1, int(self.keyframe_indices[-1])))))
        diffs = np.abs(self.keyframe_indices.astype(np.int32) - target_t)
        row = int(np.argmin(diffs))
        return "keyframe", row, int(self.keyframe_indices[row])

    # ----------------------------
    # Obs packing
    # ----------------------------
    def _obs_to_vec(self, obs) -> np.ndarray:
        q = np.asarray(obs.joint_positions, dtype=np.float32).reshape(7)
        dq = np.asarray(obs.joint_velocities, dtype=np.float32).reshape(7)
        go = np.asarray([float(obs.gripper_open)], dtype=np.float32)  # 1
        gp = np.asarray(obs.gripper_pose, dtype=np.float32).reshape(7) if obs.gripper_pose is not None else np.zeros((7,), np.float32)

        feats = []
        for sym in self._open_syms:
            spec = self.open_map[sym]
            opened, pos = is_open(spec)
            feats.append(1.0 if opened else 0.0)
            feats.append(float(pos))
        feats = np.asarray(feats, dtype=np.float32)

        return np.concatenate([q, dq, go, gp, feats], axis=0).astype(np.float32)

    # ----------------------------
    # Goal evaluation + reward
    # ----------------------------
    def _atoms_satisfied(self) -> Tuple[bool, Dict[str, bool]]:
        """
        Evaluate target_atoms (open/not(open)) only.
        """
        sat: Dict[str, bool] = {}
        ok_all = True
        for a in self.target_atoms:
            if a.arg not in self.open_map:
                # cannot evaluate -> treat as not satisfied
                sat[f"{'not(' if a.neg else ''}open({a.arg}){')' if a.neg else ''}"] = False
                ok_all = False
                continue
            opened, _ = is_open(self.open_map[a.arg])
            want = (not opened) if a.neg else opened
            sat[f"{'not(' if a.neg else ''}open({a.arg}){')' if a.neg else ''}"] = bool(want)
            ok_all = ok_all and bool(want)
        return ok_all, sat

    def _reward(self) -> float:
        # small dense reward based on open predicate satisfaction
        ok_all, sat = self._atoms_satisfied()
        r = 0.0
        for k, v in sat.items():
            r += 1.0 if v else -0.2
        if ok_all and len(sat) > 0:
            r += 5.0
        return float(r)

    # ----------------------------
    # Gym API
    # ----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._episode_step = 0

        label, row, ft = self._choose_reset_source()
        if label == "post":
            self._restore_snapshot_bytes(self.snapshot_post_trees)
            reset_tag = "post"
        else:
            trees = self.snapshot_keyframe_trees[row]  # shape (M,)
            self._restore_snapshot_bytes(trees)
            reset_tag = f"keyframe[{row}]"

        # Get an observation from the task without changing state
        # safest: a zero-action step is not ideal; instead use scene getter if present.
        if hasattr(self._task, "get_observation"):
            obs = self._task.get_observation()
        else:
            scene = getattr(self._task, "_scene", None)
            if scene is None or not hasattr(scene, "get_observation"):
                # last resort: do a single no-op step
                noop = np.zeros((8,), dtype=np.float32)
                obs, _, _ = self._task.step(noop)
            else:
                obs = scene.get_observation()

        obs_vec = self._obs_to_vec(obs)
        info = {
            "reset_snapshot": reset_tag,
            "reset_forward_t": int(ft),
            "curriculum_level": float(self.curriculum_level),
        }

        # success, sat = self._atoms_satisfied()
        # opened, q = is_open(self.open_map["drawer"])  # or the correct arg name

        # print("reset_forward_t =", info["reset_forward_t"])
        # print("[reset] snapshot=", info.get("reset_snapshot"),
        #     "q=", q, "opened=", opened, "success=", success, "sat=", sat)

        return obs_vec, info

    def step(self, action: np.ndarray):
        self._episode_step += 1

        a = np.asarray(action, dtype=np.float32).reshape(8)
        u = np.clip(a[:7], -1.0, 1.0)
        g = 1.0 if float(a[7]) > 0.0 else 0.0  # 1=open, 0=closed
        rl_action = np.concatenate([u, np.asarray([g], dtype=np.float32)], axis=0)

        obs, _, rlbench_done = self._task.step(rl_action)

        obs_vec = self._obs_to_vec(obs)
        reward = self._reward()

        success, sat = self._atoms_satisfied()
        terminated = bool(success)
        truncated = bool(self._episode_step >= self.max_episode_steps)

        info = {
            "rlbench_done": bool(rlbench_done),
            "atoms": sat,
        }
        return obs_vec, float(reward), terminated, truncated, info

    def close(self):
        try:
            self._rlbench_env.shutdown()
        except Exception:
            pass
