#!/usr/bin/env python3
"""
PLAN-A Step 3 building blocks:
  Component 1: RLBench env + task builder (evaluation-time)
  Component 2: Snapshot restore from demo_npz (bytes_v1)

Matches record_live_demo_with_actions.py snapshot conventions:
  - snapshot targets are top-level MODELS sorted by name
  - snapshot_model_names align to that sorted order
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

# -------------------------
# Component 1: Env builder
# -------------------------
def make_obs_config(
    img_size: int = 128,
    include_lowdim: bool = True,
    include_images: bool = False,
) -> ObservationConfig:
    """
    Evaluation-time config.

    We usually want task_low_dim_state=True for rollback scoring z(s),
    even if you did not record it during demo collection.
    """
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

    # Best-effort access to pyrep, aligned with your recorder
    scene = getattr(task, "_scene", None) or getattr(env, "_scene", None)
    if scene is None:
        env.shutdown()
        raise RuntimeError("Could not access task._scene or env._scene (RLBench internals differ).")

    pyrep = getattr(scene, "_pyrep", None) or getattr(env, "_pyrep", None)
    if pyrep is None:
        env.shutdown()
        raise RuntimeError("Could not access pyrep handle from task/env.")

    return EnvBundle(env=env, task=task, pyrep=pyrep)


# ------------------------------
# Component 2: Snapshot restore
# ------------------------------
def get_top_level_models(pyrep) -> List[Any]:
    """
    Must match recorder ordering:
      roots = pyrep.get_objects_in_tree(root_object=None, first_generation_only=True)
      models = [o for o in roots if o.is_model()]
      sort by name
    """
    roots = pyrep.get_objects_in_tree(root_object=None, first_generation_only=True)
    models = [o for o in roots if o.is_model()]
    models.sort(key=lambda o: o.get_name())
    return models

def restore_snapshot(
    pyrep,
    snapshot_model_names: List[str],
    snapshot_trees: List[bytes],
    *,
    settle_steps: int = 0,
    strict: bool = True,
) -> None:
    """
    Apply configuration trees to top-level models by name.

    snapshot_model_names: list of model names saved in demo_npz
    snapshot_trees: list of bytes, same length/order as names
    """
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
            # Rare, but better to fail loudly
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

    # Apply trees (PyRep API: restore via pyrep.set_configuration_tree)
    for name, tree in zip(snapshot_model_names, snapshot_trees):
        if name not in live_by_name:
            continue
        if not isinstance(tree, (bytes, bytearray)):
            raise TypeError(f"Tree for model '{name}' is not bytes: {type(tree)}")

        if hasattr(pyrep, "set_configuration_tree"):
            pyrep.set_configuration_tree(bytes(tree))
        elif hasattr(live_by_name[name], "set_configuration_tree"):
            # fallback for rare builds that implement it on Object
            live_by_name[name].set_configuration_tree(bytes(tree))
        else:
            raise RuntimeError(
                "No set_configuration_tree found on either pyrep or model object. "
                "PyRep API mismatch."
            )

    return
    # # Let physics settle
    # for _ in range(int(settle_steps)):
    #     if hasattr(pyrep, "step"):
    #         pyrep.step()
    #     else:
    #         break

def reset_dynamics_all(pyrep):
    """
    Best-effort reset of dynamics for determinism.
    Tries several APIs depending on PyRep build.
    """
    # 1) Some PyRep versions expose this
    if hasattr(pyrep, "reset_dynamic_objects"):
        try:
            pyrep.reset_dynamic_objects()
            return
        except Exception:
            pass

    # 2) Per-object reset (if available)
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

    # 3) CoppeliaSim backend fallback (if present)
    try:
        from pyrep.backend import sim as sim_backend
        lib = getattr(sim_backend, "lib", None)
        if lib is None:
            return

        # Try common symbol names
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


def settle_with_zero_action(bundle, steps: int, gripper_cmd: float):
    """
    Steps the sim through RLBench's control path (task.step),
    ensuring joint velocities are explicitly set to zero.
    """
    action = np.concatenate(
        [np.zeros((7,), dtype=np.float32), np.array([gripper_cmd], dtype=np.float32)],
        axis=0
    ).astype(np.float32)

    for _ in range(int(steps)):
        _step_task(bundle.task, action)


def restore_snapshot_deterministic(
    bundle,
    snapshot_model_names,
    snapshot_trees,
    *,
    settle_steps: int = 0,
    reset_dynamics: bool = True,
    reapply_after_reset: bool = True,
):
    """
    Deterministic restore:
      - apply trees (no stepping)
      - reset dynamics (velocities/impulses)
      - reapply trees
      - flush controllers by stepping with zero actions

    IMPORTANT: do not call pyrep.step() for settling here.
    """
    # Apply trees only (your existing function)
    restore_snapshot(
        bundle.pyrep,
        snapshot_model_names,
        snapshot_trees,
        settle_steps=0,   # <- do NOT step here
        strict=True,
    )

    if reset_dynamics:
        reset_dynamics_all(bundle.pyrep)

    if reapply_after_reset:
        restore_snapshot(
            bundle.pyrep,
            snapshot_model_names,
            snapshot_trees,
            settle_steps=0,
            strict=True,
        )

    # Determine a "hold" gripper command that matches restored state
    obs = get_observation(bundle.task)
    go = float(getattr(obs, "gripper_open", 1.0))
    gripper_cmd = 1.0 if go >= 0.5 else 0.0

    # Flush any stale velocity commands and allow gentle stabilization
    if settle_steps > 0:
        settle_with_zero_action(bundle, steps=settle_steps, gripper_cmd=gripper_cmd)

    return obs


# ------------------------------
# Component 3: Compact state + Determinism
# ------------------------------

def get_observation(task):
    """
    Robust observation getter across RLBench versions.

    Preference order:
      1) task.get_observation()
      2) task._scene.get_observation()
      3) fallback: task.reset() (returns (descriptions, obs) or obs)
    """
    if hasattr(task, "get_observation"):
        return task.get_observation()

    scene = getattr(task, "_scene", None)
    if scene is not None and hasattr(scene, "get_observation"):
        return scene.get_observation()

    # Fallback: reset gives an observation too (not ideal, but keeps script usable)
    out = task.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[1]
    return out


def _as1d_f32(x):
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float32)
    return a.reshape(-1)


def z_from_obs(obs, include_joint_positions: bool = False) -> np.ndarray:
    """
    Compact state z(s) for rollback scoring.

    Default:
      z = [gripper_pose(7), gripper_open(1), task_low_dim_state(L)]

    Optionally add joint_positions(7) for extra arm precision.
    """
    parts = []

    gp = _as1d_f32(getattr(obs, "gripper_pose", None))
    if gp is not None:
        parts.append(gp)

    go = _as1d_f32(getattr(obs, "gripper_open", None))
    if go is not None:
        parts.append(go)

    if include_joint_positions:
        q = _as1d_f32(getattr(obs, "joint_positions", None))
        if q is not None:
            parts.append(q)

    tld = _as1d_f32(getattr(obs, "task_low_dim_state", None))
    if tld is not None:
        parts.append(tld)

    if not parts:
        raise RuntimeError("z_from_obs(): no fields found in observation (check obs_config).")

    return np.concatenate(parts, axis=0).astype(np.float32)


def dz_robust(z_end: np.ndarray, z_target: np.ndarray, *, binary_tol: float = 1e-3) -> float:
    """
    Robust z-distance for rollback scoring.

    Fixes two common failure modes:
      1) Mixed continuous + binary features: a single binary flip contributes 1.0 to L2.
         We exclude dimensions that look binary (near {0,1}) in BOTH vectors.
      2) Quaternion sign ambiguity for gripper_pose quaternion: q and -q represent same orientation.
         We align quaternion signs before differencing.

    Returns:
      L2 norm over (heuristically) continuous dimensions only.
    """
    ze = np.asarray(z_end, dtype=np.float32).reshape(-1).copy()
    zt = np.asarray(z_target, dtype=np.float32).reshape(-1)

    if ze.shape != zt.shape:
        raise ValueError(f"dz_robust: shape mismatch {ze.shape} vs {zt.shape}")

    # Quaternion sign invariance for gripper_pose quaternion (dims 3:7 inside gripper_pose(7))
    if ze.size >= 7:
        qe = ze[3:7]
        qt = zt[3:7]
        if float(np.dot(qe, qt)) < 0.0:
            ze[3:7] = -qe

    # Heuristic binary mask: dim is "binary" if BOTH vectors are near 0 or 1 at that dim
    close0_t = np.abs(zt - 0.0) <= float(binary_tol)
    close1_t = np.abs(zt - 1.0) <= float(binary_tol)
    isbin_t = np.logical_or(close0_t, close1_t)

    close0_e = np.abs(ze - 0.0) <= float(binary_tol)
    close1_e = np.abs(ze - 1.0) <= float(binary_tol)
    isbin_e = np.logical_or(close0_e, close1_e)

    isbin = np.logical_and(isbin_t, isbin_e)

    # Never treat gripper_pose dims as binary (pose coords can coincidentally be near 0/1)
    if ze.size >= 7:
        isbin[:7] = False
    # Always treat gripper_open as binary if present (index 7 in your z layout)
    if ze.size >= 8:
        isbin[7] = True

    cont = ~isbin
    diff = (ze - zt)[cont]
    return float(np.linalg.norm(diff)) if diff.size else 0.0


def determinism_check(
    bundle,
    snapshot_model_names,
    snapshot_trees,
    *,
    settle_steps: int = 10,
    include_joint_positions_in_z: bool = False,
    eps_linf: float = 1e-4,
    eps_l2: float = 1e-4,
) -> bool:
    """
    Restore the SAME snapshot twice and compare z vectors.

    Returns True if differences are within eps thresholds.
    """
    # restore #1
    restore_snapshot(
        bundle.pyrep,
        snapshot_model_names,
        snapshot_trees,
        settle_steps=settle_steps,
        strict=True,
    )
    obs1 = get_observation(bundle.task)
    z1 = z_from_obs(obs1, include_joint_positions=include_joint_positions_in_z)

    # restore #2
    restore_snapshot(
        bundle.pyrep,
        snapshot_model_names,
        snapshot_trees,
        settle_steps=settle_steps,
        strict=True,
    )
    obs2 = get_observation(bundle.task)
    z2 = z_from_obs(obs2, include_joint_positions=include_joint_positions_in_z)

    if z1.shape != z2.shape:
        raise RuntimeError(f"z shape mismatch after two restores: {z1.shape} vs {z2.shape}")

    diff = z2 - z1
    linf = float(np.max(np.abs(diff)))
    l2 = float(np.linalg.norm(diff))

    # Basic diagnostics
    print(f"[z] dim={z1.size}  linf={linf:.6g}  l2={l2:.6g}")
    ok = (linf <= eps_linf) and (l2 <= eps_l2)
    print(f"[z] determinism_ok={ok} (eps_linf={eps_linf}, eps_l2={eps_l2})")

    # Optional: print worst index if not ok
    if not ok:
        worst_i = int(np.argmax(np.abs(diff)))
        print(f"[z] worst_idx={worst_i}  z1={float(z1[worst_i]):.6g}  z2={float(z2[worst_i]):.6g}  |diff|={float(abs(diff[worst_i])):.6g}")

    return ok


# ------------------------------
# Component 4: Reverse delta-q velocity servo rollout
# ------------------------------

def demo_gripper_binary(traj: dict) -> np.ndarray:
    """
    Returns g_open[T] as float32 in {0,1}, matching your recorder logic. :contentReference[oaicite:2]{index=2}
    """
    go = traj["gripper_open"].reshape(-1).astype(np.float32)
    thr = float(traj.get("gripper_threshold", np.array([0.03], dtype=np.float32))[0])
    inv = int(traj.get("invert_gripper", np.array([0], dtype=np.int32))[0]) == 1
    g = (go > thr).astype(np.float32)
    if inv:
        g = 1.0 - g
    return g  # (T,)


def _step_task(task, action: np.ndarray):
    """
    RLBench task.step return shape differs across versions.
    Common: obs, reward, terminate.
    """
    out = task.step(action)
    if isinstance(out, tuple):
        # (obs, reward, terminate) or (obs, reward) etc.
        return out[0]
    return out


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
    settle_steps: int = 0,
    action_noise_std: float = 0.0,
) -> dict:
    """
    Reverse servo from timestep t_from down to timestep t_to (inclusive target).

    Assumes the simulator is already at (or near) demo state at t_from,
    typically by restoring the snapshot for the keyframe that equals t_from.

    Returns a dict with:
      obs_end, steps_used, t_reached, q_err_inf
    """
    q_demo = traj["joint_positions"].astype(np.float32)  # (T,7)
    T = int(q_demo.shape[0])
    if not (0 <= t_to < T and 0 <= t_from < T):
        raise ValueError(f"t_from/t_to out of range (T={T}): from={t_from}, to={t_to}")
    if t_to > t_from:
        raise ValueError(f"Need t_to <= t_from for reverse rollout. Got from={t_from}, to={t_to}")

    g_demo = demo_gripper_binary(traj)  # (T,)

    # optional settle after snapshot restore (usually 0 here, snapshot restore already settles)
    for _ in range(int(settle_steps)):
        if hasattr(bundle.pyrep, "step"):
            bundle.pyrep.step()

    steps_used = 0
    obs = get_observation(bundle.task)
    q_now = np.asarray(getattr(obs, "joint_positions", None), dtype=np.float32).reshape(-1)
    if q_now.size < 7:
        raise RuntimeError("Observation missing joint_positions (check obs_config).")

    # walk backwards: want to reach t_to
    for t_target in range(int(t_from) - 1, int(t_to) - 1, -1):
        q_des = q_demo[t_target].reshape(7)
        g_cmd = float(g_demo[t_target])  # command gripper to match target state

        for _ in range(int(max_inner_steps)):
            # refresh current
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
            obs = _step_task(bundle.task, action)

            steps_used += 1
            if steps_used >= int(max_total_steps):
                # return early
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


# ------------------------------
# Component 5: Rollback evaluator
# ------------------------------

def _stats(x: List[float]) -> Dict[str, float]:
    a = np.asarray(x, dtype=np.float32)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"), "p90": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def precompute_z_targets(
    bundle,
    snapshot_model_names: List[str],
    snapshot_keyframe_trees: np.ndarray,  # (K,M) object
    *,
    settle_steps_snapshot: int = 10,
    include_joint_positions_in_z: bool = False,
) -> List[np.ndarray]:
    """
    Precompute z(s) for each keyframe snapshot once.
    """
    K = int(snapshot_keyframe_trees.shape[0])
    z_targets: List[np.ndarray] = []

    for k in range(K):
        trees = [bytes(x) for x in snapshot_keyframe_trees[k].tolist()]
        restore_snapshot(bundle.pyrep, snapshot_model_names, trees, settle_steps=settle_steps_snapshot, strict=True)
        obs = get_observation(bundle.task)
        z = z_from_obs(obs, include_joint_positions=include_joint_positions_in_z)
        z_targets.append(z)

    return z_targets


def apply_start_perturbation(
    bundle,
    *,
    steps: int = 0,
    std: float = 0.0,
    vmax: float = 1.5,
):
    """
    Simple perturbation: a few random velocity actions.
    Does not require direct joint setting.
    """
    if steps <= 0 or std <= 0.0:
        return
    for _ in range(int(steps)):
        v = np.random.normal(0.0, float(std), size=(7,)).astype(np.float32)
        v = np.clip(v, -float(vmax), float(vmax)).astype(np.float32)
        # keep gripper command neutral (open)
        action = np.concatenate([v, np.array([1.0], dtype=np.float32)], axis=0).astype(np.float32)
        _ = _step_task(bundle.task, action)


def eval_one_segment(
    bundle,
    traj: Dict[str, Any],
    snapshot_model_names: List[str],
    snapshot_keyframe_trees: np.ndarray,
    keyframe_indices: np.ndarray,
    *,
    kf_prev_row: int,
    kf_curr_row: int,
    n_trials: int = 5,
    tau_success: float = 0.8,
    z_success_eps: float = 0.1,
    settle_steps_snapshot: int = 10,
    start_noise_steps: int = 0,
    start_noise_std: float = 0.0,
    action_noise_std: float = 0.0,
    kp: float = 8.0,
    vmax: float = 1.5,
    q_tol_inf: float = 0.01,
    max_total_steps: int = 2000,
    max_inner_steps: int = 4,
    zspec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate rollback from kf_curr_row -> kf_prev_row.
    """
    t_prev = int(keyframe_indices[kf_prev_row])
    t_curr = int(keyframe_indices[kf_curr_row])

    dz_list: List[float] = []
    qerr_list: List[float] = []
    steps_list: List[float] = []
    hit_max_steps_count = 0
    success_count = 0

    trees_prev = [bytes(x) for x in snapshot_keyframe_trees[kf_prev_row].tolist()]
    trees_curr = [bytes(x) for x in snapshot_keyframe_trees[kf_curr_row].tolist()]

    for trial in range(int(n_trials)):
        # 1) restore target snapshot and read z_target
        obs_target = restore_snapshot_deterministic(bundle, snapshot_model_names, trees_prev, settle_steps=2)
        z_target = z_from_zspec(bundle, zspec)

        # 2) restore start snapshot and rollout
        restore_snapshot_deterministic(bundle, snapshot_model_names, trees_curr, settle_steps=2)

        apply_start_perturbation(bundle, steps=start_noise_steps, std=start_noise_std, vmax=vmax)

        out = servo_reverse_rollout(
            bundle, traj,
            t_from=t_curr, t_to=t_prev,
            kp=kp, vmax=vmax, q_tol_inf=q_tol_inf,
            max_total_steps=max_total_steps, max_inner_steps=max_inner_steps,
            action_noise_std=action_noise_std,
        )

        # 3) optional end settle (recommended)
        settle_with_zero_action(bundle, steps=10, gripper_cmd=1.0)

        obs_end = get_observation(bundle.task)
        z_end = z_from_zspec(bundle, zspec)

        dz = dz_zspec_rms(z_end, z_target, zspec)

        dz_list.append(dz)
        qerr_list.append(float(out["q_err_inf"]))
        steps_list.append(float(out["steps_used"]))
        if out["hit_max_steps"]:
            hit_max_steps_count += 1

        ok = (dz <= float(z_success_eps)) and (not out["hit_max_steps"])
        if ok:
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
        "params": {
            "n_trials": int(n_trials),
            "tau_success": float(tau_success),
            "z_success_eps": float(z_success_eps),
            "settle_steps_snapshot": int(settle_steps_snapshot),
            "start_noise_steps": int(start_noise_steps),
            "start_noise_std": float(start_noise_std),
            "action_noise_std": float(action_noise_std),
            "kp": float(kp),
            "vmax": float(vmax),
            "q_tol_inf": float(q_tol_inf),
            "max_total_steps": int(max_total_steps),
            "max_inner_steps": int(max_inner_steps),
        },
    }


def rollback_triage(
    bundle,
    demo_npz_path: str,
    *,
    n_trials: int = 5,
    tau_success: float = 0.8,
    z_success_eps: float = 0.02,
    settle_steps_snapshot: int = 10,
    start_noise_steps: int = 0,
    start_noise_std: float = 0.0,
    action_noise_std: float = 0.0,
    kp: float = 8.0,
    vmax: float = 1.5,
    q_tol_inf: float = 0.01,
    max_total_steps: int = 2000,
    max_inner_steps: int = 4,
    out_json: Optional[str] = None,
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
    snapshot_keyframe_trees = d["snapshot_keyframe_trees"]  # (K,M) object
    keyframe_indices = d["keyframe_indices"].astype(np.int32).reshape(-1)

    K = int(snapshot_keyframe_trees.shape[0])

    zspec = build_zspec_from_demo(
        bundle,
        snapshot_model_names,
        snapshot_keyframe_trees,
        kf_start=0,
        kf_end=K-1,
        top_k_objects=12,
        top_k_joints=4,
    )
    if keyframe_indices.size != K:
        raise RuntimeError(f"Keyframe mismatch: keyframe_indices={keyframe_indices.size} vs snapshot_keyframe_trees K={K}")

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
            settle_steps_snapshot=settle_steps_snapshot,
            start_noise_steps=start_noise_steps,
            start_noise_std=start_noise_std,
            action_noise_std=action_noise_std,
            kp=kp,
            vmax=vmax,
            q_tol_inf=q_tol_inf,
            max_total_steps=max_total_steps,
            max_inner_steps=max_inner_steps,
            zspec=zspec,
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
        # split is the earlier keyframe of the first failing segment
        split = {
            "kf_row": int(first_fail["kf_prev_row"]),
            "timestep": int(first_fail["t_prev"]),
        }

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
            "settle_steps_snapshot": int(settle_steps_snapshot),
            "start_noise_steps": int(start_noise_steps),
            "start_noise_std": float(start_noise_std),
            "action_noise_std": float(action_noise_std),
            "kp": float(kp),
            "vmax": float(vmax),
            "q_tol_inf": float(q_tol_inf),
            "max_total_steps": int(max_total_steps),
            "max_inner_steps": int(max_inner_steps),
        },
    }

    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True) if os.path.dirname(out_json) else None
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[triage] wrote {out_json}")

    return report


# -------------------------
# Minimal segment test on z-distance
# -------------------------

def segment_servo_test(
    bundle,
    demo_npz_path: str,
    keyframe_row_from: int,
    keyframe_row_to: int,
    *,
    settle_steps_snapshot: int = 10,
    kp: float = 8.0,
    vmax: float = 1.5,
    q_tol_inf: float = 0.01,
    max_total_steps: int = 2000,
    max_inner_steps: int = 4,
    z_eps: float = 0.05,
) -> bool:
    d = np.load(demo_npz_path, allow_pickle=True)

    traj = {k: d[k] for k in d.files}

    k_idx = d["keyframe_indices"].astype(np.int32).reshape(-1)
    kf_trees = d["snapshot_keyframe_trees"]  # (K,M) object
    names = [str(x) for x in d["snapshot_model_names"].tolist()]

    t_from = int(k_idx[int(keyframe_row_from)])
    t_to = int(k_idx[int(keyframe_row_to)])

    # target z (restore target snapshot, read z)
    trees_to = [bytes(x) for x in kf_trees[int(keyframe_row_to)].tolist()]
    restore_snapshot(bundle.pyrep, names, trees_to, settle_steps=settle_steps_snapshot, strict=True)
    obs_target = get_observation(bundle.task)
    z_target = z_from_obs(obs_target, include_joint_positions=False)

    # start (restore from snapshot)
    trees_from = [bytes(x) for x in kf_trees[int(keyframe_row_from)].tolist()]
    restore_snapshot(bundle.pyrep, names, trees_from, settle_steps=settle_steps_snapshot, strict=True)

    # rollout
    out = servo_reverse_rollout(
        bundle,
        traj,
        t_from=t_from,
        t_to=t_to,
        kp=kp,
        vmax=vmax,
        q_tol_inf=q_tol_inf,
        max_total_steps=max_total_steps,
        max_inner_steps=max_inner_steps,
        settle_steps=0,
    )
    obs_end = out["obs_end"]
    z_end = z_from_obs(obs_end, include_joint_positions=False)

    dz = dz_robust(z_end, z_target)

    print(f"[servo] kf_from_row={keyframe_row_from} (t={t_from})  -> kf_to_row={keyframe_row_to} (t={t_to})")
    print(f"[servo] steps_used={out['steps_used']}  hit_max_steps={out['hit_max_steps']}  q_err_inf={out['q_err_inf']:.6g}")
    print(f"[servo] dz_robust={dz:.6g}  z_eps={z_eps}")

    ok = (dz <= float(z_eps)) and (not out["hit_max_steps"])
    print(f"[servo] success={ok}")
    return ok


# -------------------------
# Minimal CLI smoke test
# -------------------------

def _load_demo_snapshot(npz_path: str, keyframe_row: int = 0) -> Tuple[str, int, List[str], List[bytes]]:
    d = np.load(npz_path, allow_pickle=True)

    task_name = str(d["task"][0])
    variation = int(d["variation"][0])

    storage = str(d.get("snapshot_storage", np.array(["none"]))[0])
    captured = int(d.get("snapshot_captured", np.array([0], dtype=np.int32))[0])
    if storage != "bytes_v1" or captured != 1:
        raise RuntimeError(f"Demo has no usable snapshots (storage={storage}, captured={captured}).")

    names = [str(x) for x in d["snapshot_model_names"].tolist()]
    kf_trees = d["snapshot_keyframe_trees"]  # object array (K,M)
    if keyframe_row < 0 or keyframe_row >= kf_trees.shape[0]:
        raise IndexError(f"keyframe_row out of range: {keyframe_row} (K={kf_trees.shape[0]})")

    trees_row = [bytes(x) for x in kf_trees[keyframe_row].tolist()]
    return task_name, variation, names, trees_row


# -------------------------
# CLI restore smoke test
# -------------------------

def pose_signature_in_model(
    pyrep,
    root_model_name: str = "block_pyramid",
    name_filter_substr: str = "block",
):
    # restrict scanning to a single top-level model to avoid LIGHT warnings
    root = None
    for m in get_top_level_models(pyrep):
        if m.get_name() == root_model_name:
            root = m
            break

    objs = pyrep.get_objects_in_tree(root_object=root, first_generation_only=False) if root is not None \
           else pyrep.get_objects_in_tree(root_object=None, first_generation_only=False)

    kept = []
    poses = []
    for o in objs:
        try:
            n = o.get_name()
        except Exception:
            continue
        if name_filter_substr and (name_filter_substr not in n):
            continue
        if hasattr(o, "get_pose"):
            try:
                p = np.asarray(o.get_pose(), dtype=np.float32).reshape(7)
                kept.append(n)
                poses.append(p)
            except Exception:
                pass

    order = np.argsort(np.array(kept, dtype=object))
    kept = [kept[i] for i in order.tolist()]
    poses = [poses[i] for i in order.tolist()]
    sig = np.concatenate(poses, axis=0) if poses else np.zeros((0,), dtype=np.float32)
    return sig, kept


def restore_stress_test(
    bundle,
    snapshot_model_names,
    snapshot_keyframe_trees,
    kf_row: int,
    *,
    n: int = 30,
    post_steps: int = 0,          # steps AFTER restore, using zero-action stepping
    name_filter_substr: str = "block",
    root_model_name: str = "block_pyramid",
    tol_restore_linf: float = 1e-6,   # restore-only should be extremely tight
    tol_post_linf: float = 1e-3,      # after steps you may allow small drift
):
    K = int(snapshot_keyframe_trees.shape[0])
    if kf_row < 0 or kf_row >= K:
        raise IndexError(f"kf_row out of range: {kf_row} (K={K})")

    trees = [bytes(x) for x in snapshot_keyframe_trees[int(kf_row)].tolist()]

    ref_restore = None
    ref_post = None
    max_restore_linf = 0.0
    max_post_linf = 0.0

    for i in range(int(n)):
        # restore deterministically BUT DO NOT step here
        restore_snapshot_deterministic(
            bundle,
            snapshot_model_names,
            trees,
            settle_steps=0,                 # <-- critical: no stepping inside restore
            reset_dynamics=True,
            reapply_after_reset=True,
        )

        sig_restore, names = pose_signature_in_model(
            bundle.pyrep,
            root_model_name=root_model_name,
            name_filter_substr=name_filter_substr,
        )

        if ref_restore is None:
            ref_restore = sig_restore.copy()
        else:
            diff = sig_restore - ref_restore
            linf = float(np.max(np.abs(diff))) if diff.size else 0.0
            max_restore_linf = max(max_restore_linf, linf)

        # optional post-step drift test (physics)
        if post_steps > 0:
            settle_with_zero_action(bundle, steps=post_steps, gripper_cmd=1.0)
            sig_post, _ = pose_signature_in_model(
                bundle.pyrep,
                root_model_name=root_model_name,
                name_filter_substr=name_filter_substr,
            )
            if ref_post is None:
                ref_post = sig_post.copy()
            else:
                diff2 = sig_post - ref_post
                linf2 = float(np.max(np.abs(diff2))) if diff2.size else 0.0
                max_post_linf = max(max_post_linf, linf2)

    print(f"[stress] kf_row={kf_row} n={n} filter='{name_filter_substr}' root='{root_model_name}' "
          f"restore_max_linf={max_restore_linf:.6g} post_steps={post_steps} post_max_linf={max_post_linf:.6g}")

    ok_restore = max_restore_linf <= float(tol_restore_linf)
    ok_post = (post_steps == 0) or (max_post_linf <= float(tol_post_linf))
    print(f"[stress] ok_restore={ok_restore} (tol={tol_restore_linf}) ok_post={ok_post} (tol={tol_post_linf})")

    return ok_restore and ok_post


# -------------------------
# z from demo 
# -------------------------

from dataclasses import dataclass

@dataclass
class ZSpec:
    """
    Defines which entities we include in z(s).
    We store object names and joint names; we also cache a name->object handle map for speed.
    """
    object_names: List[str]
    joint_names: List[str]
    root_models_used: List[str]


def _quat_align_in_place(vec_end: np.ndarray, vec_tgt: np.ndarray, start: int):
    """
    Align quaternion sign for a pose block [x,y,z,qx,qy,qz,qw] starting at index start.
    """
    qe = vec_end[start+3:start+7]
    qt = vec_tgt[start+3:start+7]
    if float(np.dot(qe, qt)) < 0.0:
        vec_end[start+3:start+7] = -qe


def _pose_delta(p0: np.ndarray, p1: np.ndarray) -> float:
    """
    Pose change magnitude between two poses [x,y,z,qx,qy,qz,qw].
    Uses position delta + orientation delta (via aligned quaternion dot).
    """
    p0 = np.asarray(p0, dtype=np.float32).reshape(7)
    p1 = np.asarray(p1, dtype=np.float32).reshape(7)
    dp = float(np.linalg.norm(p1[:3] - p0[:3]))

    q0 = p0[3:7]
    q1 = p1[3:7]
    # sign-invariant dot
    dot = float(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0))
    # orientation distance proxy in [0,1]
    dq = float(1.0 - dot)
    return dp + dq


def _collect_scene_objects(pyrep, ignore_model_names: set) -> Dict[str, Any]:
    """
    Collect candidate objects from all top-level models excluding ignored ones.
    Returns dict name->object for things that support get_pose().
    """
    objs = {}
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
            if hasattr(o, "get_pose"):
                try:
                    _ = o.get_pose()  # probe
                    objs[n] = o
                except Exception:
                    pass
    return objs


def _collect_scene_joints(pyrep, ignore_model_names: set) -> Dict[str, Any]:
    """
    Collect candidate joints from all top-level models excluding ignored ones.
    Returns dict name->jointlike object that supports get_joint_position().
    """
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
                    _ = o.get_joint_position()  # probe
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
    """
    Build ZSpec automatically by finding which objects/joints change most between
    keyframe kf_start and keyframe kf_end.
    """
    K = int(snapshot_keyframe_trees.shape[0])
    if kf_end is None:
        kf_end = K - 1
    if not (0 <= kf_start < K and 0 <= kf_end < K):
        raise ValueError(f"kf_start/kf_end out of range (K={K}): {kf_start}, {kf_end}")

    # Ignore obvious non-task models
    ignore = {
        "DefaultLights", "ResizableFloor_5_25", "XYZCameraProxy",
        "Panda",  # robot
    }
    # Keep only models not ignored (for reporting)
    roots_used = [m.get_name() for m in get_top_level_models(bundle.pyrep) if m.get_name() not in ignore]

    # Collect candidates once (names->objects)
    cand_objs = _collect_scene_objects(bundle.pyrep, ignore_model_names=ignore)
    cand_joints = _collect_scene_joints(bundle.pyrep, ignore_model_names=ignore)

    # Restore start snapshot and record poses
    trees0 = [bytes(x) for x in snapshot_keyframe_trees[int(kf_start)].tolist()]
    restore_snapshot_deterministic(bundle, snapshot_model_names, trees0, settle_steps=0)
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

    # Restore end snapshot and record poses
    trees1 = [bytes(x) for x in snapshot_keyframe_trees[int(kf_end)].tolist()]
    restore_snapshot_deterministic(bundle, snapshot_model_names, trees1, settle_steps=0)
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

    # Score deltas and pick top-K
    obj_scores = []
    for name in pose0.keys():
        if name in pose1:
            d = _pose_delta(pose0[name], pose1[name])
            if d >= float(min_pose_delta):
                obj_scores.append((d, name))
    obj_scores.sort(reverse=True)
    chosen_objs = [name for _, name in obj_scores[:int(top_k_objects)]]

    joint_scores = []
    for name in joint0.keys():
        if name in joint1:
            d = abs(float(joint1[name]) - float(joint0[name]))
            if d >= float(min_joint_delta):
                joint_scores.append((d, name))
    joint_scores.sort(reverse=True)
    chosen_joints = [name for _, name in joint_scores[:int(top_k_joints)]]

    print(f"[zspec] roots_used={roots_used}")
    print(f"[zspec] chosen_objects({len(chosen_objs)}): {chosen_objs}")
    print(f"[zspec] chosen_joints({len(chosen_joints)}): {chosen_joints}")

    return ZSpec(object_names=chosen_objs, joint_names=chosen_joints, root_models_used=roots_used)

def z_from_zspec(bundle, zspec: ZSpec) -> np.ndarray:
    """
    z = [gripper_pose(7), gripper_open(1), obj1_pose(7), ..., joint_positions...]
    """
    obs = get_observation(bundle.task)

    parts = []
    gp = _as1d_f32(getattr(obs, "gripper_pose", None))
    go = _as1d_f32(getattr(obs, "gripper_open", None))

    if gp is None or go is None:
        raise RuntimeError("z_from_zspec: missing gripper_pose or gripper_open in observation.")

    parts.append(gp)     # 7
    parts.append(go)     # 1

    # Build a name->object cache once per call (cheap enough) or cache globally if you want
    all_objs = {}
    try:
        objs = bundle.pyrep.get_objects_in_tree(root_object=None, first_generation_only=False)
        for o in objs:
            try:
                all_objs[o.get_name()] = o
            except Exception:
                pass
    except Exception:
        pass

    for name in zspec.object_names:
        o = all_objs.get(name, None)
        if o is None or not hasattr(o, "get_pose"):
            # Fill with zeros if object not found to keep dimensionality fixed
            parts.append(np.zeros((7,), dtype=np.float32))
            continue
        try:
            parts.append(np.asarray(o.get_pose(), dtype=np.float32).reshape(7))
        except Exception:
            parts.append(np.zeros((7,), dtype=np.float32))

    for name in zspec.joint_names:
        j = all_objs.get(name, None)
        if j is None or not hasattr(j, "get_joint_position"):
            parts.append(np.zeros((1,), dtype=np.float32))
            continue
        try:
            parts.append(np.asarray([float(j.get_joint_position())], dtype=np.float32))
        except Exception:
            parts.append(np.zeros((1,), dtype=np.float32))

    return np.concatenate(parts, axis=0).astype(np.float32)

def dz_zspec_rms(z_end: np.ndarray, z_target: np.ndarray, zspec: ZSpec) -> float:
    ze = np.asarray(z_end, dtype=np.float32).reshape(-1).copy()
    zt = np.asarray(z_target, dtype=np.float32).reshape(-1)

    if ze.shape != zt.shape:
        raise ValueError(f"dz_zspec_rms: shape mismatch {ze.shape} vs {zt.shape}")

    # Align quaternions for gripper pose and each object pose
    # layout:
    # 0..6 gripper_pose, 7 gripper_open
    # then N objects each 7
    # then joints (scalars)
    _quat_align_in_place(ze, zt, 0)

    base = 8
    for i in range(len(zspec.object_names)):
        _quat_align_in_place(ze, zt, base + 7 * i)

    diff = ze - zt
    # RMS, not L2
    return float(np.linalg.norm(diff) / np.sqrt(diff.size)) if diff.size else 0.0



def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--keyframe_row", type=int, default=0)
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.set_defaults(headless=True)
    ap.add_argument("--settle_steps", type=int, default=10)
    ap.add_argument("--z_check", action="store_true", help="Restore snapshot twice and compare z(s).")
    ap.add_argument("--z_eps_linf", type=float, default=1e-4)
    ap.add_argument("--z_eps_l2", type=float, default=1e-4)
    ap.add_argument("--z_include_joint_positions", action="store_true")
    ap.add_argument("--restore_stress", action="store_true")
    ap.add_argument("--stress_n", type=int, default=50)
    ap.add_argument("--stress_filter", type=str, default="block")
    ap.add_argument("--servo_segment", action="store_true", help="Run reverse servo between two keyframe rows and report dz.")
    ap.add_argument("--kf_from", type=int, default=1)
    ap.add_argument("--kf_to", type=int, default=0)
    ap.add_argument("--kp", type=float, default=8.0)
    ap.add_argument("--vmax", type=float, default=1.5)
    ap.add_argument("--q_tol_inf", type=float, default=0.01)
    ap.add_argument("--max_total_steps", type=int, default=2000)
    ap.add_argument("--max_inner_steps", type=int, default=4)
    ap.add_argument("--z_eps", type=float, default=0.05)
    ap.add_argument("--rollback_eval", action="store_true", help="Evaluate all adjacent keyframe rollback segments (Component 5).")
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--tau_success", type=float, default=0.8)
    ap.add_argument("--z_success_eps", type=float, default=0.05)
    ap.add_argument("--start_noise_steps", type=int, default=0)
    ap.add_argument("--start_noise_std", type=float, default=0.0)
    ap.add_argument("--action_noise_std", type=float, default=0.0)
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--stress_post_steps", type=int, default=0)
    ap.add_argument("--stress_tol_restore_linf", type=float, default=1e-6)
    ap.add_argument("--stress_tol_post_linf", type=float, default=1e-3)
    ap.add_argument("--stress_root_model", type=str, default="block_pyramid")

    args = ap.parse_args()

    d = np.load(args.demo_npz, allow_pickle=True)
    task_name = str(d["task"][0])
    variation = int(d["variation"][0])

    storage = str(d.get("snapshot_storage", np.array(["none"]))[0])
    captured = int(d.get("snapshot_captured", np.array([0], dtype=np.int32))[0])
    if storage != "bytes_v1" or captured != 1:
        raise RuntimeError(f"Demo has no usable snapshots (storage={storage}, captured={captured}).")

    names = [str(x) for x in d["snapshot_model_names"].tolist()]
    kf_trees = d["snapshot_keyframe_trees"]  # (K,M) object

    K = int(kf_trees.shape[0])
    if args.keyframe_row < 0 or args.keyframe_row >= K:
        raise IndexError(f"--keyframe_row out of range: {args.keyframe_row} (K={K})")

    print(f"[demo] task={task_name} variation={variation} keyframe_row={args.keyframe_row}")
    print(f"[demo] snapshot models ({len(names)}): {names}")
    
    print(f"[demo] task={task_name} variation={variation} keyframe_row={args.keyframe_row}")
    print(f"[demo] snapshot models ({len(names)}): {names}")

    bundle = build_env_and_task(
        task_name,
        variation,
        headless=args.headless,
        include_lowdim=True,
        include_images=False,
    )

    try:
        # Ensure scene objects exist
        bundle.task.reset()

        live = [m.get_name() for m in get_top_level_models(bundle.pyrep)]
        print(f"[live] top-level models ({len(live)}): {live}")

        if args.restore_stress:
            ok = restore_stress_test(
                bundle,
                names,
                kf_trees,
                kf_row=args.keyframe_row,
                n=args.stress_n,
                post_steps=args.stress_post_steps,
                name_filter_substr=args.stress_filter,
                root_model_name=args.stress_root_model,
                tol_restore_linf=args.stress_tol_restore_linf,
                tol_post_linf=args.stress_tol_post_linf,
            )
            if not ok:
                raise SystemExit(1)
            print("[ok] restore stress test passed")
            return

        if args.rollback_eval:
            report = rollback_triage(
                bundle,
                args.demo_npz,
                n_trials=args.n_trials,
                tau_success=args.tau_success,
                z_success_eps=args.z_success_eps,
                settle_steps_snapshot=args.settle_steps,
                start_noise_steps=args.start_noise_steps,
                start_noise_std=args.start_noise_std,
                action_noise_std=args.action_noise_std,
                kp=args.kp,
                vmax=args.vmax,
                q_tol_inf=args.q_tol_inf,
                max_total_steps=args.max_total_steps,
                max_inner_steps=args.max_inner_steps,
                out_json=args.out_json,
            )
            if not report["directly_reversible"]:
                print(f"[triage] FIRST FAIL: seg {report['first_failure_segment']['kf_prev_row']}<-{report['first_failure_segment']['kf_curr_row']}")
                print(f"[triage] SPLIT: kf_row={report['split']['kf_row']} timestep={report['split']['timestep']}")
                raise SystemExit(4)
            else:
                print("[triage] demo is directly reversible under this controller")
                return

        if args.servo_segment:
            ok = segment_servo_test(
                bundle,
                args.demo_npz,
                keyframe_row_from=args.kf_from,
                keyframe_row_to=args.kf_to,
                settle_steps_snapshot=args.settle_steps,
                kp=args.kp,
                vmax=args.vmax,
                q_tol_inf=args.q_tol_inf,
                max_total_steps=args.max_total_steps,
                max_inner_steps=args.max_inner_steps,
                z_eps=args.z_eps,
            )
            if not ok:
                raise SystemExit(3)
            print("[ok] servo segment succeeded")
            return

        restore_snapshot(
            bundle.pyrep,
            names,
            kf_trees,
            settle_steps=args.settle_steps,
            strict=True,
        )
        if args.z_check:
            ok = determinism_check(
                bundle,
                names,
                kf_trees,
                settle_steps=args.settle_steps,
                include_joint_positions_in_z=args.z_include_joint_positions,
                eps_linf=args.z_eps_linf,
                eps_l2=args.z_eps_l2,
            )
            if not ok:
                # exit nonzero so you can catch this in scripts/CI
                raise SystemExit(2)
        else:
            restore_snapshot(
                bundle.pyrep,
                names,
                kf_trees,
                settle_steps=args.settle_steps,
                strict=True,
            )
            print("[ok] snapshot restored")

    finally:
        bundle.env.shutdown()


if __name__ == "__main__":
    main()
