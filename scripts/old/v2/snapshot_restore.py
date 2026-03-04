#!/usr/bin/env python3
"""
snapshot_restore.py

PLAN-A Step 3: rollback triage evaluator with robust keyframe restoration.

Key fix (generic):
- Some RLBench tasks "hold" objects via parent changes / constraints / scripted motion.
- CoppeliaSim configuration trees do NOT reliably capture parent relations.
- So we CALIBRATE per-keyframe object state by replaying the forward demo once:
    parent handle + world pose for tracked objects at each keyframe.
- After each snapshot restore, we force-apply the calibrated state.

This preserves PLAN-A: all information comes from the forward demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import json
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

from pyrep.objects.object import Object


# =============================================================================
# Component 1: Env builder
# =============================================================================

def make_obs_config(
    img_size: int = 128,
    include_lowdim: bool = True,
    include_images: bool = False,
) -> ObservationConfig:
    oc = ObservationConfig()
    oc.set_all(False)

    oc.joint_positions = True
    oc.joint_velocities = True
    oc.gripper_open = True
    oc.gripper_pose = True
    oc.task_low_dim_state = bool(include_lowdim)

    if include_images:
        oc.front_camera.set_all(False)
        oc.front_camera.rgb = True
        oc.front_camera.image_size = (img_size, img_size)

        oc.wrist_camera.set_all(False)
        oc.wrist_camera.rgb = True
        oc.wrist_camera.image_size = (img_size, img_size)

        oc.overhead_camera.set_all(False)
        oc.overhead_camera.rgb = True
        oc.overhead_camera.image_size = (img_size, img_size)

    return oc


@dataclass
class EnvBundle:
    env: Environment
    task: Any
    pyrep: Any


def build_env_and_task(
    task_name: str,
    variation: int = 0,
    *,
    headless: bool = True,
    include_lowdim: bool = True,
    include_images: bool = False,
    img_size: int = 128,
) -> EnvBundle:
    if not hasattr(rlbench_tasks, task_name):
        raise ValueError(f"Unknown RLBench task '{task_name}'.")

    task_cls = getattr(rlbench_tasks, task_name)

    obs_config = make_obs_config(
        img_size=img_size,
        include_lowdim=include_lowdim,
        include_images=include_images,
    )

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=headless,
    )
    env.launch()

    task = env.get_task(task_cls)
    task.set_variation(int(variation))

    scene = getattr(task, "_scene", None) or getattr(env, "_scene", None)
    if scene is None:
        env.shutdown()
        raise RuntimeError("Could not access task._scene or env._scene (RLBench internals differ).")

    pyrep = getattr(scene, "_pyrep", None) or getattr(env, "_pyrep", None)
    if pyrep is None:
        env.shutdown()
        raise RuntimeError("Could not access pyrep handle from task/env.")

    return EnvBundle(env=env, task=task, pyrep=pyrep)


# =============================================================================
# Component 2: Snapshot restore (bytes_v1)
# =============================================================================

def get_top_level_models(pyrep) -> List[Any]:
    roots = pyrep.get_objects_in_tree(root_object=None, first_generation_only=True)
    models = [o for o in roots if o.is_model()]
    models.sort(key=lambda o: o.get_name())
    return models


def restore_snapshot(
    pyrep,
    snapshot_model_names: List[str],
    snapshot_trees: List[bytes],
    *,
    strict: bool = True,
) -> None:
    if len(snapshot_model_names) != len(snapshot_trees):
        raise ValueError(
            f"snapshot_model_names and snapshot_trees length mismatch: "
            f"{len(snapshot_model_names)} vs {len(snapshot_trees)}"
        )

    live_models = get_top_level_models(pyrep)
    live_by_name: Dict[str, Any] = {}
    for m in live_models:
        n = m.get_name()
        if n in live_by_name:
            raise RuntimeError(f"Duplicate top-level model name in scene: '{n}'")
        live_by_name[n] = m

    missing = [n for n in snapshot_model_names if n not in live_by_name]
    if missing:
        msg = (
            "Snapshot restore failed: some snapshot models are not present in current scene.\n"
            f"Missing ({len(missing)}): {missing}\n"
            f"Live top-level models ({len(live_models)}): {[m.get_name() for m in live_models]}\n"
        )
        if strict:
            raise RuntimeError(msg)
        else:
            print("[warn]", msg)

    for name, tree in zip(snapshot_model_names, snapshot_trees):
        if name not in live_by_name:
            continue
        if not isinstance(tree, (bytes, bytearray)):
            raise TypeError(f"Tree for model '{name}' is not bytes: {type(tree)}")

        if hasattr(pyrep, "set_configuration_tree"):
            pyrep.set_configuration_tree(bytes(tree))
        elif hasattr(live_by_name[name], "set_configuration_tree"):
            live_by_name[name].set_configuration_tree(bytes(tree))
        else:
            raise RuntimeError(
                "No set_configuration_tree found on either pyrep or model object. "
                "PyRep API mismatch."
            )


def reset_dynamics_all(pyrep):
    if hasattr(pyrep, "reset_dynamic_objects"):
        try:
            pyrep.reset_dynamic_objects()
            return
        except Exception:
            pass

    try:
        objs = pyrep.get_objects_in_tree(root_object=None, first_generation_only=False)
        did_any = False
        for o in objs:
            if hasattr(o, "reset_dynamic_object"):
                try:
                    o.reset_dynamic_object()
                    did_any = True
                except Exception:
                    pass
        if did_any:
            return
    except Exception:
        pass

    try:
        from pyrep.backend import sim as sim_backend
        lib = getattr(sim_backend, "lib", None)
        if lib is None:
            return

        fn = None
        for name in ("simResetDynamicObject", "simResetDynamicObject_internal"):
            fn = getattr(lib, name, None)
            if fn is not None:
                break
        if fn is None:
            return

        objs = pyrep.get_objects_in_tree(root_object=None, first_generation_only=False)
        for o in objs:
            try:
                h = o.get_handle()
                fn(int(h))
            except Exception:
                pass
    except Exception:
        pass


# =============================================================================
# Observations + actions
# =============================================================================

def get_observation(task):
    if hasattr(task, "get_observation"):
        return task.get_observation()

    scene = getattr(task, "_scene", None)
    if scene is not None and hasattr(scene, "get_observation"):
        return scene.get_observation()

    out = task.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[1]
    return out


def _step_task(task, action: np.ndarray):
    out = task.step(action)
    if isinstance(out, tuple):
        return out[0]
    return out


def settle_with_zero_action(bundle, steps: int, gripper_cmd: float):
    action = np.concatenate(
        [np.zeros((7,), dtype=np.float32), np.array([gripper_cmd], dtype=np.float32)],
        axis=0
    ).astype(np.float32)
    for _ in range(int(steps)):
        _step_task(bundle.task, action)


# =============================================================================
# ZSpec (Option C)
# =============================================================================

@dataclass
class ZSpec:
    object_names: List[str]
    object_handles: List[int]
    joint_names: List[str]
    joint_handles: List[int]
    root_models_used: List[str]


def _pose_delta(p0: np.ndarray, p1: np.ndarray) -> float:
    p0 = np.asarray(p0, dtype=np.float32).reshape(7)
    p1 = np.asarray(p1, dtype=np.float32).reshape(7)
    return float(np.linalg.norm(p1[:3] - p0[:3]))


def _collect_scene_objects(pyrep, ignore_model_names: set) -> Dict[str, Any]:
    try:
        from pyrep.objects.shape import Shape  # type: ignore
        have_shape = True
    except Exception:
        Shape = None
        have_shape = False

    def _ignore_name(n: str) -> bool:
        s = n.lower()
        if "waypoint" in s:
            return True
        if s.startswith("dummy") or "dummy" in s:
            return True
        if "camera" in s or "light" in s:
            return True
        return False

    objs: Dict[str, Any] = {}
    for m in get_top_level_models(pyrep):
        mn = m.get_name()
        if mn in ignore_model_names:
            continue

        try:
            subtree = pyrep.get_objects_in_tree(root_object=m, first_generation_only=False)
        except Exception:
            continue

        for o in subtree:
            try:
                n = o.get_name()
            except Exception:
                continue
            if n in objs:
                continue
            if _ignore_name(n):
                continue

            if have_shape:
                try:
                    _ = Shape(o.get_handle())
                except Exception:
                    continue

            if hasattr(o, "get_pose"):
                try:
                    _ = o.get_pose()
                    objs[n] = o
                except Exception:
                    pass
    return objs


def _collect_scene_joints(pyrep, ignore_model_names: set) -> Dict[str, Any]:
    joints = {}
    for m in get_top_level_models(pyrep):
        mn = m.get_name()
        if mn in ignore_model_names:
            continue
        try:
            subtree = pyrep.get_objects_in_tree(root_object=m, first_generation_only=False)
        except Exception:
            continue

        for o in subtree:
            try:
                n = o.get_name()
            except Exception:
                continue
            if n in joints:
                continue
            if hasattr(o, "get_joint_position"):
                try:
                    _ = o.get_joint_position()
                    joints[n] = o
                except Exception:
                    pass
    return joints


def build_zspec_from_demo(
    bundle,
    snapshot_model_names: List[str],
    snapshot_keyframe_trees: np.ndarray,
    *,
    kf_start: int = 0,
    kf_end: Optional[int] = None,
    top_k_objects: int = 12,
    top_k_joints: int = 4,
    min_pose_delta: float = 1e-3,
    min_joint_delta: float = 1e-4,
) -> ZSpec:
    K = int(snapshot_keyframe_trees.shape[0])
    if kf_end is None:
        kf_end = K - 1
    if not (0 <= kf_start < K and 0 <= kf_end < K):
        raise ValueError(f"kf_start/kf_end out of range (K={K}): {kf_start}, {kf_end}")

    ignore = {"DefaultLights", "ResizableFloor_5_25", "XYZCameraProxy", "Panda"}
    roots_used = [m.get_name() for m in get_top_level_models(bundle.pyrep) if m.get_name() not in ignore]

    cand_objs = _collect_scene_objects(bundle.pyrep, ignore_model_names=ignore)
    cand_joints = _collect_scene_joints(bundle.pyrep, ignore_model_names=ignore)

    trees0 = [bytes(x) for x in snapshot_keyframe_trees[int(kf_start)].tolist()]
    restore_snapshot(bundle.pyrep, snapshot_model_names, trees0, strict=True)
    reset_dynamics_all(bundle.pyrep)
    restore_snapshot(bundle.pyrep, snapshot_model_names, trees0, strict=True)

    pose0 = {}
    for name, o in cand_objs.items():
        try:
            pose0[name] = np.asarray(o.get_pose(), dtype=np.float32).reshape(7)
        except Exception:
            pass
    joint0 = {}
    for name, j in cand_joints.items():
        try:
            joint0[name] = float(j.get_joint_position())
        except Exception:
            pass

    trees1 = [bytes(x) for x in snapshot_keyframe_trees[int(kf_end)].tolist()]
    restore_snapshot(bundle.pyrep, snapshot_model_names, trees1, strict=True)
    reset_dynamics_all(bundle.pyrep)
    restore_snapshot(bundle.pyrep, snapshot_model_names, trees1, strict=True)

    pose1 = {}
    for name, o in cand_objs.items():
        try:
            pose1[name] = np.asarray(o.get_pose(), dtype=np.float32).reshape(7)
        except Exception:
            pass
    joint1 = {}
    for name, j in cand_joints.items():
        try:
            joint1[name] = float(j.get_joint_position())
        except Exception:
            pass

    obj_scores = []
    for name in pose0.keys():
        if name in pose1:
            d = _pose_delta(pose0[name], pose1[name])
            if d >= float(min_pose_delta):
                obj_scores.append((d, name))
    obj_scores.sort(reverse=True)
    chosen_obj_names = [name for _, name in obj_scores[:int(top_k_objects)]]

    joint_scores = []
    for name in joint0.keys():
        if name in joint1:
            d = abs(float(joint1[name]) - float(joint0[name]))
            if d >= float(min_joint_delta):
                joint_scores.append((d, name))
    joint_scores.sort(reverse=True)
    chosen_joint_names = [name for _, name in joint_scores[:int(top_k_joints)]]

    chosen_obj_pairs = [(n, int(cand_objs[n].get_handle())) for n in chosen_obj_names if n in cand_objs]
    chosen_joint_pairs = [(n, int(cand_joints[n].get_handle())) for n in chosen_joint_names if n in cand_joints]

    obj_names = [n for (n, _) in chosen_obj_pairs]
    obj_handles = [h for (_, h) in chosen_obj_pairs]
    joint_names = [n for (n, _) in chosen_joint_pairs]
    joint_handles = [h for (_, h) in chosen_joint_pairs]

    print(f"[zspec] roots_used={roots_used}")
    print(f"[zspec] chosen_objects({len(obj_names)}): {obj_names}")
    print(f"[zspec] chosen_joints({len(joint_names)}): {joint_names}")

    return ZSpec(
        object_names=obj_names,
        object_handles=obj_handles,
        joint_names=joint_names,
        joint_handles=joint_handles,
        root_models_used=roots_used,
    )


def _as1d_f32(x):
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float32)
    return a.reshape(-1)


def z_from_zspec(bundle, zspec: ZSpec) -> np.ndarray:
    obs = get_observation(bundle.task)
    gp = _as1d_f32(getattr(obs, "gripper_pose", None))
    go = _as1d_f32(getattr(obs, "gripper_open", None))
    if gp is None or go is None:
        raise RuntimeError("z_from_zspec: missing gripper_pose or gripper_open")

    parts = [gp, go]

    for h in zspec.object_handles:
        try:
            o = Object(int(h))
            if hasattr(o, "get_position"):
                p = np.asarray(o.get_position(), dtype=np.float32).reshape(3)
            else:
                p = np.asarray(o.get_pose(), dtype=np.float32).reshape(7)[:3]
            parts.append(p)
        except Exception:
            parts.append(np.zeros((3,), dtype=np.float32))

    for h in zspec.joint_handles:
        try:
            j = Object(int(h))
            if hasattr(j, "get_joint_position"):
                parts.append(np.asarray([float(j.get_joint_position())], dtype=np.float32))
            else:
                parts.append(np.zeros((1,), dtype=np.float32))
        except Exception:
            parts.append(np.zeros((1,), dtype=np.float32))

    return np.concatenate(parts, axis=0).astype(np.float32)


def dz_rms_ignore_gripper_open(z_end: np.ndarray, z_target: np.ndarray) -> float:
    ze = np.asarray(z_end, dtype=np.float32).reshape(-1).copy()
    zt = np.asarray(z_target, dtype=np.float32).reshape(-1)

    if ze.shape != zt.shape:
        raise ValueError(f"shape mismatch {ze.shape} vs {zt.shape}")

    if ze.size >= 7:
        qe, qt = ze[3:7], zt[3:7]
        if float(np.dot(qe, qt)) < 0.0:
            ze[3:7] = -qe

    diff = ze - zt

    if diff.size > 7:
        diff = np.concatenate([diff[:7], diff[8:]], axis=0)

    return float(np.linalg.norm(diff) / np.sqrt(diff.size)) if diff.size else 0.0


# =============================================================================
# Forward calibration: record parent+pose of tracked objects at each keyframe
# =============================================================================

def demo_gripper_binary(traj: dict) -> np.ndarray:
    go = traj["gripper_open"].reshape(-1).astype(np.float32)
    thr = float(traj.get("gripper_threshold", np.array([0.03], dtype=np.float32))[0])
    inv = int(traj.get("invert_gripper", np.array([0], dtype=np.int32))[0]) == 1
    g = (go > thr).astype(np.float32)
    if inv:
        g = 1.0 - g
    return g  # 1=open, 0=closed


def servo_to_q_once(bundle, q_des: np.ndarray, g_cmd: float, *, kp: float, vmax: float, q_tol_inf: float, max_inner_steps: int):
    q_des = np.asarray(q_des, dtype=np.float32).reshape(7)
    for _ in range(int(max_inner_steps)):
        obs = get_observation(bundle.task)
        q_now = np.asarray(obs.joint_positions, dtype=np.float32).reshape(7)
        err = q_des - q_now
        if float(np.max(np.abs(err))) <= float(q_tol_inf):
            break
        v = np.clip(float(kp) * err, -float(vmax), float(vmax)).astype(np.float32)
        act = np.concatenate([v, np.asarray([float(g_cmd)], dtype=np.float32)], axis=0).astype(np.float32)
        _step_task(bundle.task, act)


def calibrate_keyframe_object_state(
    bundle,
    traj: Dict[str, Any],
    *,
    snapshot_model_names: List[str],
    snapshot_keyframe_trees: np.ndarray,
    keyframe_indices: np.ndarray,
    zspec: ZSpec,
    kp: float,
    vmax: float,
    q_tol_inf: float,
    max_inner_steps: int,
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """
    Returns:
      calib[kf_row][obj_handle] = {"parent": parent_handle_or_-1, "pose": pose7_float32_list}
    """
    q_demo = traj["joint_positions"].astype(np.float32)
    T = int(q_demo.shape[0])
    g_demo = demo_gripper_binary(traj)

    # map timestep -> kf_row
    t_to_kf: Dict[int, int] = {}
    for kf_row, t in enumerate(keyframe_indices.tolist()):
        t_to_kf[int(t)] = int(kf_row)

    # Start from keyframe 0 snapshot (usually pre-grasp, so safe)
    trees0 = [bytes(x) for x in snapshot_keyframe_trees[0].tolist()]
    restore_snapshot(bundle.pyrep, snapshot_model_names, trees0, strict=True)
    reset_dynamics_all(bundle.pyrep)
    restore_snapshot(bundle.pyrep, snapshot_model_names, trees0, strict=True)
    settle_with_zero_action(bundle, steps=1, gripper_cmd=float(g_demo[int(keyframe_indices[0])]))

    calib: Dict[int, Dict[int, Dict[str, Any]]] = {}

    # Prefer recorded actions if present and compatible
    use_actions = False
    if "actions" in traj:
        a = np.asarray(traj["actions"])
        if a.ndim == 2 and a.shape[0] == T and a.shape[1] >= 8:
            use_actions = True

    max_t = int(max(keyframe_indices))
    max_t = min(max_t, T - 1)

    for t in range(0, max_t + 1):
        if t > 0:
            if use_actions:
                act = np.asarray(traj["actions"][t], dtype=np.float32).reshape(-1)
                act = act[:8].astype(np.float32)
                _step_task(bundle.task, act)
            else:
                servo_to_q_once(
                    bundle,
                    q_demo[t],
                    float(g_demo[t]),
                    kp=kp,
                    vmax=vmax,
                    q_tol_inf=q_tol_inf,
                    max_inner_steps=max_inner_steps,
                )

        if t in t_to_kf:
            kf_row = t_to_kf[t]
            calib[kf_row] = {}
            for h in zspec.object_handles:
                hh = int(h)
                try:
                    o = Object(hh)
                    p = o.get_parent()
                    ph = int(p.get_handle()) if p is not None else -1
                    pose = np.asarray(o.get_pose(), dtype=np.float32).reshape(7)
                    calib[kf_row][hh] = {"parent": ph, "pose": pose.tolist()}
                except Exception:
                    calib[kf_row][hh] = {"parent": -1, "pose": [0.0] * 7

                                     }

    # Print quick summary for main object if any
    if zspec.object_handles:
        main_h = int(zspec.object_handles[0])
        parents = []
        for k in sorted(calib.keys()):
            parents.append(calib[k][main_h]["parent"])
        uniq = sorted(set(parents))
        print(f"[calib] tracked_handle={main_h} unique_parents={uniq}")

    return calib


def apply_calibrated_object_state(
    calib_row: Dict[int, Dict[str, Any]],
):
    """
    Apply parent then pose for each object handle in a calibrated row.
    """
    # parent first
    for h, st in calib_row.items():
        try:
            o = Object(int(h))
            ph = int(st.get("parent", -1))
            if ph == -1:
                o.set_parent(None, keep_in_place=True)
            else:
                o.set_parent(Object(ph), keep_in_place=True)
        except Exception:
            pass

    # pose second (absolute/world pose)
    for h, st in calib_row.items():
        try:
            o = Object(int(h))
            pose = np.asarray(st.get("pose", [0] * 7), dtype=np.float32).reshape(7)
            o.set_pose(pose.tolist())
        except Exception:
            pass


def restore_keyframe(
    bundle,
    *,
    snapshot_model_names: List[str],
    trees: List[bytes],
    g_cmd: float,
    calib_row: Optional[Dict[int, Dict[str, Any]]] = None,
    settle_steps: int = 1,
):
    restore_snapshot(bundle.pyrep, snapshot_model_names, trees, strict=True)

    if calib_row is not None:
        apply_calibrated_object_state(calib_row)

    reset_dynamics_all(bundle.pyrep)
    restore_snapshot(bundle.pyrep, snapshot_model_names, trees, strict=True)

    if calib_row is not None:
        apply_calibrated_object_state(calib_row)

    if settle_steps > 0:
        settle_with_zero_action(bundle, steps=int(settle_steps), gripper_cmd=float(g_cmd))


# =============================================================================
# Reverse servo rollout (unchanged)
# =============================================================================

def servo_reverse_rollout(
    bundle,
    traj: dict,
    t_from: int,
    t_to: int,
    *,
    kp: float = 8.0,
    vmax: float = 1.5,
    q_tol_inf: float = 0.01,
    max_total_steps: int = 2000,
    max_inner_steps: int = 4,
    action_noise_std: float = 0.0,
) -> dict:
    q_demo = traj["joint_positions"].astype(np.float32)  # (T,7)
    T = int(q_demo.shape[0])
    if not (0 <= t_to < T and 0 <= t_from < T):
        raise ValueError(f"t_from/t_to out of range (T={T}): from={t_from}, to={t_to}")
    if t_to > t_from:
        raise ValueError(f"Need t_to <= t_from for reverse rollout. Got from={t_from}, to={t_to}")

    g_demo = demo_gripper_binary(traj)

    steps_used = 0
    obs = get_observation(bundle.task)
    q_now = np.asarray(getattr(obs, "joint_positions", None), dtype=np.float32).reshape(-1)
    if q_now.size < 7:
        raise RuntimeError("Observation missing joint_positions (check obs_config).")

    for t_target in range(int(t_from) - 1, int(t_to) - 1, -1):
        q_des = q_demo[t_target].reshape(7)
        g_cmd = float(g_demo[t_target])

        for _ in range(int(max_inner_steps)):
            obs = get_observation(bundle.task)
            q_now = np.asarray(obs.joint_positions, dtype=np.float32).reshape(7)

            err = q_des - q_now
            err_inf = float(np.max(np.abs(err)))
            if err_inf <= float(q_tol_inf):
                break

            v_cmd = float(kp) * err
            if float(action_noise_std) > 0.0:
                v_cmd = v_cmd + np.random.normal(0.0, float(action_noise_std), size=err.shape).astype(np.float32)
            v = np.clip(v_cmd, -float(vmax), float(vmax)).astype(np.float32)

            action = np.concatenate([v, np.array([g_cmd], dtype=np.float32)], axis=0).astype(np.float32)
            _ = _step_task(bundle.task, action)

            steps_used += 1
            if steps_used >= int(max_total_steps):
                obs = get_observation(bundle.task)
                q_now = np.asarray(obs.joint_positions, dtype=np.float32).reshape(7)
                q_err_inf = float(np.max(np.abs(q_demo[t_to].reshape(7) - q_now)))
                return {
                    "obs_end": obs,
                    "steps_used": steps_used,
                    "t_reached": t_target,
                    "q_err_inf": q_err_inf,
                    "hit_max_steps": True,
                }

    obs_end = get_observation(bundle.task)
    q_end = np.asarray(obs_end.joint_positions, dtype=np.float32).reshape(7)
    q_err_inf = float(np.max(np.abs(q_demo[t_to].reshape(7) - q_end)))
    return {
        "obs_end": obs_end,
        "steps_used": steps_used,
        "t_reached": int(t_to),
        "q_err_inf": q_err_inf,
        "hit_max_steps": False,
    }


# =============================================================================
# Rollback triage
# =============================================================================

def _stats(x: List[float]) -> Dict[str, float]:
    a = np.asarray(x, dtype=np.float32)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "p90": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def eval_one_segment(
    bundle,
    traj: Dict[str, Any],
    snapshot_model_names: List[str],
    snapshot_keyframe_trees: np.ndarray,
    keyframe_indices: np.ndarray,
    *,
    kf_prev_row: int,
    kf_curr_row: int,
    n_trials: int,
    tau_success: float,
    z_success_eps: float,
    kp: float,
    vmax: float,
    q_tol_inf: float,
    max_total_steps: int,
    max_inner_steps: int,
    action_noise_std: float,
    zspec: ZSpec,
    calib: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    t_prev = int(keyframe_indices[kf_prev_row])
    t_curr = int(keyframe_indices[kf_curr_row])

    dz_list: List[float] = []
    qerr_list: List[float] = []
    steps_list: List[float] = []
    hit_max_steps_count = 0
    success_count = 0

    trees_prev = [bytes(x) for x in snapshot_keyframe_trees[kf_prev_row].tolist()]
    trees_curr = [bytes(x) for x in snapshot_keyframe_trees[kf_curr_row].tolist()]

    g_demo = demo_gripper_binary(traj)

    calib_prev = calib.get(kf_prev_row) if calib is not None else None
    calib_curr = calib.get(kf_curr_row) if calib is not None else None

    for _ in range(int(n_trials)):
        # target
        restore_keyframe(
            bundle,
            snapshot_model_names=snapshot_model_names,
            trees=trees_prev,
            g_cmd=float(g_demo[t_prev]),
            calib_row=calib_prev,
            settle_steps=1,
        )
        z_target = z_from_zspec(bundle, zspec)

        # start
        restore_keyframe(
            bundle,
            snapshot_model_names=snapshot_model_names,
            trees=trees_curr,
            g_cmd=float(g_demo[t_curr]),
            calib_row=calib_curr,
            settle_steps=1,
        )

        out = servo_reverse_rollout(
            bundle,
            traj,
            t_from=t_curr,
            t_to=t_prev,
            kp=kp,
            vmax=vmax,
            q_tol_inf=q_tol_inf,
            max_total_steps=max_total_steps,
            max_inner_steps=max_inner_steps,
            action_noise_std=action_noise_std,
        )

        settle_with_zero_action(bundle, steps=1, gripper_cmd=1.0)
        z_end = z_from_zspec(bundle, zspec)

        dz = dz_rms_ignore_gripper_open(z_end, z_target)

        dz_list.append(float(dz))
        qerr_list.append(float(out["q_err_inf"]))
        steps_list.append(float(out["steps_used"]))
        if out["hit_max_steps"]:
            hit_max_steps_count += 1

        if (dz <= float(z_success_eps)) and (not out["hit_max_steps"]):
            success_count += 1

    success_rate = float(success_count) / float(max(1, int(n_trials)))
    segment_ok = success_rate >= float(tau_success)

    return {
        "kf_prev_row": int(kf_prev_row),
        "kf_curr_row": int(kf_curr_row),
        "t_prev": int(t_prev),
        "t_curr": int(t_curr),
        "success_rate": float(success_rate),
        "segment_ok": bool(segment_ok),
        "dz_l2": _stats(dz_list),
        "q_err_inf": _stats(qerr_list),
        "steps_used": _stats(steps_list),
        "hit_max_steps": int(hit_max_steps_count),
    }


def rollback_triage(
    bundle,
    demo_npz_path: str,
    *,
    n_trials: int,
    tau_success: float,
    z_success_eps: float,
    kp: float,
    vmax: float,
    q_tol_inf: float,
    max_total_steps: int,
    max_inner_steps: int,
    action_noise_std: float,
    out_json: Optional[str],
    calibrate_keyframes: bool,
) -> Dict[str, Any]:
    d = np.load(demo_npz_path, allow_pickle=True)
    traj = {k: d[k] for k in d.files}

    task_name = str(d["task"][0])
    variation = int(d["variation"][0])

    storage = str(d.get("snapshot_storage", np.array(["none"]))[0])
    captured = int(d.get("snapshot_captured", np.array([0], dtype=np.int32))[0])
    if storage != "bytes_v1" or captured != 1:
        raise RuntimeError(f"Demo has no usable snapshots (storage={storage}, captured={captured}).")

    snapshot_model_names = [str(x) for x in d["snapshot_model_names"].tolist()]
    snapshot_keyframe_trees = d["snapshot_keyframe_trees"]
    keyframe_indices = d["keyframe_indices"].astype(np.int32).reshape(-1)

    K = int(snapshot_keyframe_trees.shape[0])
    if keyframe_indices.size != K:
        raise RuntimeError(f"Keyframe mismatch: keyframe_indices={keyframe_indices.size} vs snapshot_keyframe_trees K={K}")

    zspec = build_zspec_from_demo(
        bundle,
        snapshot_model_names,
        snapshot_keyframe_trees,
        kf_start=0,
        kf_end=K - 1,
        top_k_objects=12,
        top_k_joints=4,
    )

    calib = None
    if calibrate_keyframes:
        calib = calibrate_keyframe_object_state(
            bundle,
            traj,
            snapshot_model_names=snapshot_model_names,
            snapshot_keyframe_trees=snapshot_keyframe_trees,
            keyframe_indices=keyframe_indices,
            zspec=zspec,
            kp=kp,
            vmax=vmax,
            q_tol_inf=q_tol_inf,
            max_inner_steps=max_inner_steps,
        )
        print(f"[calib] recorded keyframes: {sorted(calib.keys())}")

    segments: List[Dict[str, Any]] = []
    first_fail: Optional[Dict[str, Any]] = None

    for kf_curr_row in range(1, K):
        kf_prev_row = kf_curr_row - 1

        seg = eval_one_segment(
            bundle,
            traj,
            snapshot_model_names,
            snapshot_keyframe_trees,
            keyframe_indices,
            kf_prev_row=kf_prev_row,
            kf_curr_row=kf_curr_row,
            n_trials=n_trials,
            tau_success=tau_success,
            z_success_eps=z_success_eps,
            kp=kp,
            vmax=vmax,
            q_tol_inf=q_tol_inf,
            max_total_steps=max_total_steps,
            max_inner_steps=max_inner_steps,
            action_noise_std=action_noise_std,
            zspec=zspec,
            calib=calib,
        )
        segments.append(seg)

        print(
            f"[triage] seg {kf_prev_row}<-{kf_curr_row} "
            f"(t {seg['t_prev']}<-{seg['t_curr']}): "
            f"sr={seg['success_rate']:.2f} dz_mean={seg['dz_l2']['mean']:.4g} ok={seg['segment_ok']}"
        )

        if first_fail is None and (not seg["segment_ok"]):
            first_fail = seg

    directly_reversible = first_fail is None
    split = None
    if not directly_reversible:
        split = {"kf_row": int(first_fail["kf_prev_row"]), "timestep": int(first_fail["t_prev"])}

    report = {
        "schema": "plan_a.rollback_triage.v1",
        "task": task_name,
        "variation": int(variation),
        "demo_npz": str(demo_npz_path),
        "keyframe_indices": keyframe_indices.tolist(),
        "directly_reversible": bool(directly_reversible),
        "first_failure_segment": first_fail,
        "split": split,
        "segments": segments,
        "params": {
            "n_trials": int(n_trials),
            "tau_success": float(tau_success),
            "z_success_eps": float(z_success_eps),
            "kp": float(kp),
            "vmax": float(vmax),
            "q_tol_inf": float(q_tol_inf),
            "max_total_steps": int(max_total_steps),
            "max_inner_steps": int(max_inner_steps),
            "action_noise_std": float(action_noise_std),
            "calibrate_keyframes": bool(calibrate_keyframes),
        },
    }

    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True) if os.path.dirname(out_json) else None
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[triage] wrote {out_json}")

    return report


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.set_defaults(headless=True)

    ap.add_argument("--rollback_eval", action="store_true")
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--tau_success", type=float, default=0.8)
    ap.add_argument("--z_success_eps", type=float, default=0.1)

    ap.add_argument("--kp", type=float, default=6.0)
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--q_tol_inf", type=float, default=0.01)
    ap.add_argument("--max_total_steps", type=int, default=2000)
    ap.add_argument("--max_inner_steps", type=int, default=4)
    ap.add_argument("--action_noise_std", type=float, default=0.0)

    ap.add_argument("--out_json", type=str, default=None)

    ap.add_argument("--no_calib", action="store_true", help="Disable forward keyframe calibration (not recommended for umbrella-like tasks).")

    args = ap.parse_args()

    d = np.load(args.demo_npz, allow_pickle=True)
    task_name = str(d["task"][0])
    variation = int(d["variation"][0])

    storage = str(d.get("snapshot_storage", np.array(["none"]))[0])
    captured = int(d.get("snapshot_captured", np.array([0], dtype=np.int32))[0])
    if storage != "bytes_v1" or captured != 1:
        raise RuntimeError(f"Demo has no usable snapshots (storage={storage}, captured={captured}).")

    names = [str(x) for x in d["snapshot_model_names"].tolist()]
    print(f"[demo] task={task_name} variation={variation}")
    print(f"[demo] snapshot models ({len(names)}): {names}")

    bundle = build_env_and_task(
        task_name,
        variation,
        headless=args.headless,
        include_lowdim=True,
        include_images=False,
    )

    try:
        bundle.task.reset()

        live = [m.get_name() for m in get_top_level_models(bundle.pyrep)]
        print(f"[live] top-level models ({len(live)}): {live}")

        if args.rollback_eval:
            report = rollback_triage(
                bundle,
                args.demo_npz,
                n_trials=args.n_trials,
                tau_success=args.tau_success,
                z_success_eps=args.z_success_eps,
                kp=args.kp,
                vmax=args.vmax,
                q_tol_inf=args.q_tol_inf,
                max_total_steps=args.max_total_steps,
                max_inner_steps=args.max_inner_steps,
                action_noise_std=args.action_noise_std,
                out_json=args.out_json,
                calibrate_keyframes=(not args.no_calib),
            )
            if not report["directly_reversible"]:
                ff = report["first_failure_segment"]
                print(f"[triage] FIRST FAIL: seg {ff['kf_prev_row']}<-{ff['kf_curr_row']}")
                print(f"[triage] SPLIT: kf_row={report['split']['kf_row']} timestep={report['split']['timestep']}")
                raise SystemExit(4)
            else:
                print("[triage] demo is directly reversible under this controller")
                return

        print("Nothing to do. Use --rollback_eval.")
        return

    finally:
        bundle.env.shutdown()


if __name__ == "__main__":
    main()