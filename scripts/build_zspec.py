#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os
import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

from pyrep.objects.object import Object
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint


# IMPORTANT: keep these lowercase because we compare against name.lower()
EXCLUDE_NAME_SUBSTR = [
    "waypoint", "anchor", "target", "success", "fail",
    "defaultlights", "resizablefloor", "xyzcamera", "camera",
    "cam_", "vision_sensor",
    "panda", "franka", "collision", "dummy", "plane", "boundary",
]
EXCLUDE_IF_CONTAINS = ["distractor"]  # optional, but good default for BlockPyramid


def _get_keyframe_trees_and_kf(npz) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      trees2d: (num_models, num_keyframes) object array of bytes
      kf:      (num_keyframes,) keyframe indices aligned with trees columns
    Handles both layouts: (num_models,num_keyframes) or (num_keyframes,num_models).
    """
    trees_key = None
    for k in ["snapshot_keyframe_trees", "keyframe_trees", "snapshot_trees"]:
        if k in npz.files:
            trees_key = k
            break
    if trees_key is None:
        raise RuntimeError("No snapshot_keyframe_trees found in npz.")

    trees = np.asarray(npz[trees_key], dtype=object)
    if trees.ndim != 2:
        raise RuntimeError(f"Expected 2D keyframe trees, got shape {trees.shape}")

    if "snapshot_model_names" not in npz.files:
        raise RuntimeError("snapshot_model_names missing; cannot align keyframe trees.")
    n_models = int(np.asarray(npz["snapshot_model_names"]).shape[0])

    # Orient trees as (num_models, num_keyframes)
    if trees.shape[0] == n_models:
        trees2d = trees
    elif trees.shape[1] == n_models:
        trees2d = trees.T
    else:
        raise RuntimeError(f"Cannot orient keyframe trees with n_models={n_models}, trees.shape={trees.shape}")

    # Keyframe indices (prefer snapshot_keyframe_indices, fallback to keyframe_indices)
    kf_key = None
    for k in ["snapshot_keyframe_indices", "keyframe_indices", "keyframe_idxs", "keyframe_indices_raw"]:
        if k in npz.files:
            kf_key = k
            break
    if kf_key is None:
        # if missing, assume sequential
        kf = np.arange(trees2d.shape[1], dtype=np.int64)
    else:
        kf = np.asarray(npz[kf_key], dtype=np.int64).ravel()

    # Align length to columns
    if kf.size != trees2d.shape[1]:
        if kf.size > trees2d.shape[1]:
            kf = kf[:trees2d.shape[1]]
        else:
            kf = np.arange(trees2d.shape[1], dtype=np.int64)

    if kf.size != trees2d.shape[1]:
        raise RuntimeError(f"Cannot align keyframes: kf.size={kf.size}, trees2d.shape={trees2d.shape}")

    return trees2d, kf


def _restore_trees(task, env, trees_1d, g_cmd: float, settle_steps: int):
    """Reset task, apply configuration trees, then settle with stable gripper command."""
    pr = _get_pyrep(env, task)
    task.reset()
    for tree in trees_1d:
        pr.set_configuration_tree(tree)
    _settle(task, g_cmd, settle_steps)


def _pick_first_key(npz, keys):
    for k in keys:
        if k in npz.files:
            return k
    return None


def _quat_dist(q1, q2) -> float:
    q1 = np.asarray(q1, dtype=np.float64).ravel()
    q2 = np.asarray(q2, dtype=np.float64).ravel()
    if q1.size != 4 or q2.size != 4:
        return 0.0
    d = float(abs(np.dot(q1, q2)))
    d = max(0.0, min(1.0, d))
    return 1.0 - d


def _pose_score(p0, p1) -> float:
    p0 = np.asarray(p0, dtype=np.float64).ravel()
    p1 = np.asarray(p1, dtype=np.float64).ravel()
    if p0.size < 7 or p1.size < 7:
        return 0.0
    dp = np.linalg.norm(p1[:3] - p0[:3])
    dq = _quat_dist(p0[3:7], p1[3:7])
    return float(dp + 0.2 * dq)


def _get_pyrep(env, task):
    pr = getattr(env, "_pyrep", None)
    if pr is not None:
        return pr
    scene = getattr(task, "_scene", None)
    if scene is not None:
        pr = getattr(scene, "_pyrep", None)
        if pr is not None:
            return pr
    raise RuntimeError("Could not access PyRep instance (env._pyrep / task._scene._pyrep).")


def _set_rendering(pr, enabled: bool):
    # PyRep API differs across versions
    try:
        if hasattr(pr, "set_rendering"):
            pr.set_rendering(enabled)
        elif hasattr(pr, "set_rendering_enabled"):
            pr.set_rendering_enabled(enabled)
    except Exception:
        pass


def _settle(task, g_cmd_last: float, steps: int):
    action = np.zeros((8,), dtype=np.float32)
    action[-1] = 1.0 if g_cmd_last > 0.5 else 0.0
    obs = None
    for _ in range(int(steps)):
        obs, _, _ = task.step(action)
    return obs


def _keep_name(name: str) -> bool:
    lname = str(name).lower()
    if any(sub in lname for sub in EXCLUDE_NAME_SUBSTR):
        return False
    if any(sub in lname for sub in EXCLUDE_IF_CONTAINS):
        return False
    return True


def _list_shapes_and_joints(root_name: str):
    root_name = str(root_name)
    if not _keep_name(root_name):
        return [], []

    try:
        root = Object.get_object(root_name)
    except Exception:
        return [], []

    if hasattr(root, "get_objects_in_tree"):
        try:
            objs = root.get_objects_in_tree(exclude_base=False, first_generation_only=False)
        except TypeError:
            objs = root.get_objects_in_tree()
    else:
        objs = [root]

    shapes, joints = [], []
    seen = set()

    for o in objs:
        try:
            name = o.get_name()
        except Exception:
            continue
        if name in seen:
            continue
        seen.add(name)

        if not _keep_name(name):
            continue

        if isinstance(o, Dummy):
            continue

        if isinstance(o, Shape):
            shapes.append(name)
        elif isinstance(o, Joint):
            joints.append(name)

    return shapes, joints


def _joint_score(j0, j1) -> float:
    try:
        return float(abs(float(j1) - float(j0)))
    except Exception:
        return 0.0


def _get_pose_by_name(name: str):
    try:
        o = Object.get_object(name)
        return np.asarray(o.get_pose(), dtype=np.float32)  # (x,y,z,qx,qy,qz,qw)
    except Exception:
        return None


def make_obs_config_no_cameras():
    oc = ObservationConfig()
    try:
        oc.set_all(False)
    except Exception:
        pass

    # we don't actually need these obs fields for scoring, but keeping them is harmless
    for attr in ["joint_positions", "gripper_open", "gripper_pose"]:
        if hasattr(oc, attr):
            setattr(oc, attr, True)

    for cam_attr in ["front_camera", "wrist_camera", "overhead_camera",
                     "left_shoulder_camera", "right_shoulder_camera"]:
        cam = getattr(oc, cam_attr, None)
        if cam is not None:
            try:
                cam.set_all(False)
            except Exception:
                for f in ["rgb", "depth", "point_cloud", "mask"]:
                    if hasattr(cam, f):
                        setattr(cam, f, False)
    return oc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_prep_npz", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--K_joints", type=int, default=4)
    ap.add_argument("--headless", action="store_true", default=False)
    ap.add_argument("--settle_steps", type=int, default=10)
    ap.add_argument("--disable_rendering", action="store_true", default=True)
    args = ap.parse_args()

    d = np.load(args.demo_prep_npz, allow_pickle=True)

    task_name = str(d["task"][0])
    variation = int(d["variation"][0])

    # Prefer the filtered, aligned final snapshot roots from prep_final_snapshot.py
    if "final_snapshot_model_names" in d.files and "final_snapshot_trees" in d.files:
        model_names = d["final_snapshot_model_names"]
        final_trees = d["final_snapshot_trees"]
    else:
        # fallback for older prep files
        model_names = d["snapshot_model_names"]
        final_trees_key = _pick_first_key(d, ["final_snapshot_trees", "final_trees"])
        if final_trees_key is None:
            raise RuntimeError("Could not find final snapshot trees in NPZ.")
        final_trees = d[final_trees_key]

    g_last_key = _pick_first_key(d, ["final_gripper_open", "end_gripper_open"])
    if g_last_key is not None:
        g_last = float(np.asarray(d[g_last_key], dtype=np.float32).ravel()[0])
    else:
        # fallback: last recorded gripper value
        if "gripper_open" in d.files:
            g_last = float(np.asarray(d["gripper_open"], dtype=np.float32).ravel()[-1])
        else:
            g_last = 1.0

    if args.out_json is None:
        base, _ = os.path.splitext(args.demo_prep_npz)
        args.out_json = base + "_zspec.json"

    action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
    obs_config = make_obs_config_no_cameras()
    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=args.headless)
    env.launch()

    try:
        pr = getattr(env, "_pyrep", None)
        if pr is not None and args.disable_rendering:
            _set_rendering(pr, False)

        task_cls = getattr(rlbench_tasks, task_name)
        task = env.get_task(task_cls)
        task.set_variation(variation)

        # Candidate objects: union over all snapshot roots (final roots preferred)
        shape_set, joint_set = set(), set()
        for rn in model_names:
            s_list, j_list = _list_shapes_and_joints(str(rn))
            shape_set.update(s_list)
            joint_set.update(j_list)

        shape_names = sorted(shape_set)
        joint_names = sorted(joint_set)

        # -------------------------
        # Baseline = earliest keyframe snapshot (demo-consistent start state)
        # -------------------------
        kf_trees2d, kf = _get_keyframe_trees_and_kf(d)

        j0 = int(np.argmin(kf))                 # earliest keyframe column
        start_trees = kf_trees2d[:, j0]         # (num_models,) bytes
        start_idx = int(kf[j0])                 # timestep index in the recorded trajectory

        # Use recorded gripper open at that timestep (fallback to first sample / open)
        g_ref = np.asarray(d["gripper_open"], dtype=np.float32).ravel() if "gripper_open" in d.files else None
        if g_ref is not None and g_ref.size > 0:
            start_idx_clamped = max(0, min(start_idx, int(g_ref.size - 1)))
            g_start = float(g_ref[start_idx_clamped])
        else:
            g_start = 1.0

        # Restore earliest keyframe snapshot and read "init" state from it
        _restore_trees(task, env, start_trees, g_cmd=g_start, settle_steps=args.settle_steps)

        init_shape_pose = {n: _get_pose_by_name(n) for n in shape_names}
        init_joint_pos = {}
        for jn in joint_names:
            try:
                init_joint_pos[jn] = float(Joint.get_object(jn).get_joint_position())
            except Exception:
                pass

        # Restore final settled snapshot
        pr_task = _get_pyrep(env, task)
        task.reset()
        for tree in final_trees:
            pr_task.set_configuration_tree(tree)
        _settle(task, g_last, args.settle_steps)

        final_shape_pose = {n: _get_pose_by_name(n) for n in shape_names}
        final_joint_pos = {}
        for jn in joint_names:
            try:
                final_joint_pos[jn] = float(Joint.get_object(jn).get_joint_position())
            except Exception:
                pass

        # Score shapes (motion between reset and final snapshot)
        shape_scored = []
        for n in shape_names:
            p0, p1 = init_shape_pose.get(n), final_shape_pose.get(n)
            if p0 is None or p1 is None:
                continue
            shape_scored.append((_pose_score(p0, p1), n))
        shape_scored.sort(reverse=True, key=lambda x: x[0])

        # Score joints
        joint_scored = []
        for jn in joint_names:
            j0, j1 = init_joint_pos.get(jn), final_joint_pos.get(jn)
            if j0 is None or j1 is None:
                continue
            joint_scored.append((_joint_score(j0, j1), jn))
        joint_scored.sort(reverse=True, key=lambda x: x[0])

        K_shapes = int(args.K)
        K_joints = int(args.K_joints)

        chosen_shapes = [name for score, name in shape_scored[:K_shapes] if score > 1e-6]

        chosen_joints = []
        for score, name in joint_scored:
            if score > 1e-4:
                chosen_joints.append(name)
            if len(chosen_joints) >= K_joints:
                break

        spec = {
            "task": task_name,
            "variation": variation,
            "snapshot_roots": [str(x) for x in model_names],
            "K_shapes": int(K_shapes),
            "K_joints": int(K_joints),
            "shapes": chosen_shapes,
            "joints": chosen_joints,
            "z_dim": int(8 + 7 * len(chosen_shapes) + 1 * len(chosen_joints)),
            "notes": "z=[gripper_pose(7),gripper_open(1),shape_pose(7)*Ks,joint_pos(1)*Kj]; candidates from final_snapshot_model_names when available"
        }

        with open(args.out_json, "w") as f:
            json.dump(spec, f, indent=2)

        print(f"[ok] wrote {args.out_json}")
        print(f"[info] roots={len(model_names)} | candidate shapes={len(shape_names)} joints={len(joint_names)}")
        print("[zspec] shape scores (top 20):")
        for score, name in shape_scored[:min(len(shape_scored), 20)]:
            mark = "*" if name in chosen_shapes else " "
            print(f"  {mark} {score: .4f}  {name}")

        print("[zspec] joint scores (top 20):")
        for score, name in joint_scored[:min(len(joint_scored), 20)]:
            mark = "*" if name in chosen_joints else " "
            print(f"  {mark} {score: .4f}  {name}")

    finally:
        try:
            pr = getattr(env, "_pyrep", None)
            if pr is not None and args.disable_rendering:
                _set_rendering(pr, True)
        except Exception:
            pass
        env.shutdown()


if __name__ == "__main__":
    main()