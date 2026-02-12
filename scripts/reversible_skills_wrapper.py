#!/usr/bin/env python3
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

from pyrep.objects.shape import Shape
from pyrep.objects.object import Object


def _mat34_to_44(m34) -> np.ndarray:
    m = np.asarray(m34, dtype=np.float32).reshape(3, 4)
    M = np.eye(4, dtype=np.float32)
    M[:3, :4] = m
    return M

def world_aabb(obj) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (min_xyz, max_xyz) in world frame.
    Assumes obj.get_bounding_box() returns [minx, maxx, miny, maxy, minz, maxz] in local frame,
    and obj.get_matrix() returns a 3x4 transform in row-major.
    """
    bb = obj.get_bounding_box()
    minx, maxx, miny, maxy, minz, maxz = [float(x) for x in bb]

    corners = np.array(
        [[x, y, z, 1.0]
         for x in (minx, maxx)
         for y in (miny, maxy)
         for z in (minz, maxz)],
        dtype=np.float32
    )  # (8,4)

    M = _mat34_to_44(obj.get_matrix())
    wc = (M @ corners.T).T[:, :3]  # (8,3)

    return wc.min(axis=0), wc.max(axis=0)

def aabb_center(min_xyz: np.ndarray, max_xyz: np.ndarray) -> np.ndarray:
    return 0.5 * (min_xyz + max_xyz)

class SpatialResolver:
    def __init__(self, spatial_map_path: str):
        with open(spatial_map_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

    def get_obj(self, sym: str):
        spec = self.cfg.get(sym, {})
        name = spec.get("object", sym)

        # robust: try Shape first, then generic Object
        try:
            return Shape(name)
        except Exception:
            try:
                return Object.get_object(name)
            except Exception:
                raise KeyError(f"Could not resolve symbol '{sym}' to object '{name}'")

    def on_params(self, sym: str) -> dict:
        return (self.cfg.get(sym, {}).get("on", {}) or {})

    def in_params(self, sym: str) -> dict:
        return (self.cfg.get(sym, {}).get("in", {}) or {})


# ----------------------------
# Predicate parsing (multi-arg)
# ----------------------------
@dataclass
class Atom:
    neg: bool
    pred: str
    args: List[str]

def _split_top_level(s: str, sep: str = ",") -> List[str]:
    out = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == sep and depth == 0:
            out.append(s[start:i].strip())
            start = i + 1
    tail = s[start:].strip()
    if tail:
        out.append(tail)
    return [x for x in out if x]

def parse_atoms(set_str: str) -> List[Atom]:
    s = (set_str or "").strip()
    if not s:
        return []
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    parts = _split_top_level(s, ",")
    out: List[Atom] = []
    for p in parts:
        neg = False
        p = p.strip()
        if p.startswith("not(") and p.endswith(")"):
            neg = True
            p = p[4:-1].strip()

        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)$", p)
        if not m:
            continue
        pred = m.group(1).strip()
        args_str = m.group(2).strip()
        args = [a.strip() for a in _split_top_level(args_str, ",")]
        out.append(Atom(neg=neg, pred=pred, args=args))
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


def eval_on(obj, support, z_tol=0.02, xy_margin=0.02) -> bool:
    omin, omax = world_aabb(obj)
    smin, smax = world_aabb(support)

    obj_bottom_z = float(omin[2])
    sup_top_z = float(smax[2])
    c = aabb_center(omin, omax)

    z_ok = abs(obj_bottom_z - sup_top_z) <= float(z_tol)

    x_ok = (float(smin[0]) - xy_margin) <= float(c[0]) <= (float(smax[0]) + xy_margin)
    y_ok = (float(smin[1]) - xy_margin) <= float(c[1]) <= (float(smax[1]) + xy_margin)
    return bool(z_ok and x_ok and y_ok)

def eval_in(obj, container, mode="center_in_aabb",
            shrink_xy=0.02, shrink_z_top=0.01, shrink_z_bottom=0.0) -> bool:
    omin, omax = world_aabb(obj)
    cmin, cmax = world_aabb(container)

    cmin = cmin.copy()
    cmax = cmax.copy()
    cmin[0] += shrink_xy
    cmin[1] += shrink_xy
    cmin[2] += shrink_z_bottom
    cmax[0] -= shrink_xy
    cmax[1] -= shrink_xy
    cmax[2] -= shrink_z_top

    if mode == "center_in_aabb":
        p = aabb_center(omin, omax)
        return bool(np.all(p >= cmin) and np.all(p <= cmax))

    # stricter: require ALL object AABB corners inside container AABB
    if mode == "aabb_contained":
        return bool(np.all(omin >= cmin) and np.all(omax <= cmax))

    raise ValueError(f"Unknown in-mode: {mode}")


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
        spatial_map_path: Optional[str] = None,
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

        # self.target_atoms = [a for a in parse_atoms(self.pre_core) if a.pred == "open"]

        # ---- build RLBench env ----
        obs_config = ObservationConfig()
        obs_config.set_all(False)
        obs_config.joint_positions = True
        obs_config.joint_velocities = True
        obs_config.gripper_open = True
        obs_config.gripper_pose = True
        obs_config.gripper_touch_forces = True

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

        self.spatial = SpatialResolver(spatial_map_path) if spatial_map_path else None
        self.target_atoms = parse_atoms(self.pre_core)
        self.shaping_atoms = [a for a in self.target_atoms if a.pred not in ("gripper_open", "empty", "held")]
        self._prev_shape_potential = 0.0

        self.empty_touch_thr = 0.1

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
        if self.K > 1:
            row = max(1, row)
        return "keyframe", row, int(self.keyframe_indices[row])
    
    def _is_gripper_open(self, obs, thr=0.5) -> bool:
        if hasattr(obs, "gripper_open_amount") and obs.gripper_open_amount is not None:
            return float(obs.gripper_open_amount) > thr
        return bool(getattr(obs, "gripper_open", False))

    # ----------------------------
    # Obs packing
    # ----------------------------
    def _obs_to_vec(self, obs) -> np.ndarray:
        q = np.asarray(obs.joint_positions, dtype=np.float32).reshape(7)
        dq = np.asarray(obs.joint_velocities, dtype=np.float32).reshape(7)
        go = np.asarray([1.0 if self._is_gripper_open(obs) else 0.0], dtype=np.float32)
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
    def _atoms_satisfied(self, obs=None, atoms=None) -> Tuple[bool, Dict[str, bool]]:
        atoms = self.target_atoms if atoms is None else atoms
        sat: Dict[str, bool] = {}
        ok_all = True

        for a in atoms:
            key = f"{'not(' if a.neg else ''}{a.pred}({', '.join(a.args)}){')' if a.neg else ''}"
            val = False

            try:
                if a.pred == "open":
                    sym = a.args[0]
                    if sym in self.open_map:
                        opened, _ = is_open(self.open_map[sym])
                        val = (not opened) if a.neg else opened
                    else:
                        val = False

                elif a.pred in ("on", "in"):
                    if self.spatial is None:
                        val = False
                    else:
                        if a.pred == "on":
                            obj_sym, sup_sym = a.args[0], a.args[1]
                            obj = self.spatial.get_obj(obj_sym)
                            sup = self.spatial.get_obj(sup_sym)
                            p = self.spatial.on_params(sup_sym)
                            mode = p.get("mode", "aabb_top")
                            if mode == "aabb_top":
                                raw = eval_on(obj, sup, z_tol=p.get("z_tol", 0.02), xy_margin=p.get("xy_margin", 0.02))
                            else:
                                raise ValueError(f"Unknown on-mode: {mode}")
                            val = (not raw) if a.neg else raw

                        else:  #a.pred == "in":
                            obj_sym, cont_sym = a.args[0], a.args[1]
                            obj = self.spatial.get_obj(obj_sym)
                            cont = self.spatial.get_obj(cont_sym)
                            p = self.spatial.in_params(cont_sym)
                            raw = eval_in(
                                obj, cont,
                                mode=p.get("mode", "center_in_aabb"),
                                shrink_xy=p.get("shrink_xy", 0.02),
                                shrink_z_top=p.get("shrink_z_top", 0.01),
                                shrink_z_bottom=p.get("shrink_z_bottom", 0.0),
                            )
                            val = (not raw) if a.neg else raw

                elif a.pred == "gripper_open":
                    # if you use 0-arg predicate. needs obs
                    if obs is None:
                        val = False
                    else:
                        raw = self._is_gripper_open(obs)
                        val = (not raw) if a.neg else raw

                elif a.pred == "held":
                    # simple generic heuristic: close to gripper + gripper closed
                    if obs is None or self.spatial is None:
                        val = False
                    else:
                        obj_sym = a.args[0]
                        obj = self.spatial.get_obj(obj_sym)
                        obj_p = np.array(obj.get_position(), dtype=np.float32)
                        grip_p = np.array(obs.gripper_pose[:3], dtype=np.float32)
                        dist = float(np.linalg.norm(obj_p - grip_p))
                        raw = (dist < 0.05) and (not self._is_gripper_open(obs))
                        val = (not raw) if a.neg else raw

                elif a.pred == "empty":
                    if obs is None:
                        val = False
                    else:
                        tf = getattr(obs, "gripper_touch_forces", None)
                        if tf is None:
                            raw = True  # conservative fallback; or use a distance-to-object heuristic
                        else:
                            raw = float(np.linalg.norm(np.asarray(tf))) < self.empty_touch_thr
                        val = (not raw) if a.neg else raw
                else:
                    val = False

            except Exception:
                val = False

            sat[key] = bool(val)
            ok_all = ok_all and bool(val)

        return ok_all, sat

    
    def _reward_from_sat(self, ok_all: bool, sat: Dict[str, bool]) -> float:
        r = -0.01
        for _, v in sat.items():
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

        # Get an observation without changing state
        if hasattr(self._task, "get_observation"):
            obs = self._task.get_observation()
        else:
            scene = getattr(self._task, "_scene", None)
            if scene is None or not hasattr(scene, "get_observation"):
                noop = np.zeros((8,), dtype=np.float32)
                obs, _, _ = self._task.step(noop)
            else:
                obs = scene.get_observation()

        # FULL termination check (all preconditions)
        success_all, sat_all = self._atoms_satisfied(obs=obs, atoms=self.target_atoms)

        # SHAPING potential init (exclude gripper-related atoms)
        success_shape, sat_shape = self._atoms_satisfied(obs=obs, atoms=self.shaping_atoms)
        self._prev_shape_potential = float(sum(sat_shape.values())) / float(max(1, len(sat_shape)))

        obs_vec = self._obs_to_vec(obs)
        info = {
            "reset_snapshot": reset_tag,
            "reset_forward_t": int(ft),
            "curriculum_level": float(self.curriculum_level),

            # Debug / bookkeeping:
            "reset_success_all": bool(success_all),
            "reset_atoms_all": sat_all,
            "reset_success_shape": bool(success_shape),
            "reset_atoms_shape": sat_shape,
            "shape_potential": float(self._prev_shape_potential),
        }
        return obs_vec, info


    def step(self, action: np.ndarray):
        self._episode_step += 1

        a = np.asarray(action, dtype=np.float32).reshape(8)
        u = np.clip(a[:7], -1.0, 1.0)
        g = 1.0 if float(a[7]) > 0.0 else 0.0  # 1=open, 0=closed
        rl_action = np.concatenate([u, np.asarray([g], dtype=np.float32)], axis=0)

        obs, _, rlbench_done = self._task.step(rl_action)
        obs_vec = self._obs_to_vec(obs)

        # FULL termination check (all preconditions)
        success_all, sat_all = self._atoms_satisfied(obs=obs, atoms=self.target_atoms)

        # SHAPING progress (exclude gripper-related atoms)
        success_shape, sat_shape = self._atoms_satisfied(obs=obs, atoms=self.shaping_atoms)
        shape_potential = float(sum(sat_shape.values())) / float(max(1, len(sat_shape)))

        # Potential-difference shaping: reward only for *progress*
        reward = (shape_potential - float(self._prev_shape_potential)) - 0.01
        self._prev_shape_potential = shape_potential

        # Optional: small bonus when the shaping subset is fully satisfied
        if success_shape and len(sat_shape) > 0:
            reward += 1.0

        terminated = bool(success_all)
        if terminated:
            reward += 5.0

        truncated = bool(self._episode_step >= self.max_episode_steps)

        info = {
            "rlbench_done": bool(rlbench_done),

            # Debug:
            "atoms_all": sat_all,
            "atoms_shape": sat_shape,
            "shape_potential": float(shape_potential),
        }
        return obs_vec, float(reward), terminated, truncated, info


    def close(self):
        try:
            self._rlbench_env.shutdown()
        except Exception:
            pass
