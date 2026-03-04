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
    "panda", "franka", "collision", "dummy", "plane", "boundary",
]
EXCLUDE_IF_CONTAINS = ["distractor"]  # optional, but good default for BlockPyramid


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


def _settle(task, g_cmd_last: float, steps: int):
    action = np.zeros((8,), dtype=np.float32)
    action[-1] = 1.0 if g_cmd_last > 0.5 else 0.0
    obs = None
    for _ in range(int(steps)):
        obs, _, _ = task.step(action)
    return obs


def _pick_root_model_name(model_names):
    # Choose first model name that doesn't look like robot/camera/light/waypoint/etc.
    for n in model_names:
        s = str(n)
        ls = s.lower()
        if any(sub in ls for sub in EXCLUDE_NAME_SUBSTR):
            continue
        if any(sub in ls for sub in EXCLUDE_IF_CONTAINS):
            continue
        return s
    return str(model_names[-1])


def _list_shapes_and_joints(root_name: str):
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

        lname = name.lower()
        if any(sub in lname for sub in EXCLUDE_NAME_SUBSTR):
            continue
        if any(sub in lname for sub in EXCLUDE_IF_CONTAINS):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_prep_npz", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--headless", action="store_true", default=False)
    ap.add_argument("--settle_steps", type=int, default=10)
    args = ap.parse_args()

    d = np.load(args.demo_prep_npz, allow_pickle=True)

    task_name = str(d["task"][0])
    variation = int(d["variation"][0])

    model_names = d["snapshot_model_names"]
    root_model = _pick_root_model_name(model_names)

    final_trees = d["final_snapshot_trees"]
    g_last = float(np.asarray(d["final_gripper_open"], dtype=np.float32).ravel()[0])

    if args.out_json is None:
        base, _ = os.path.splitext(args.demo_prep_npz)
        args.out_json = base + "_zspec.json"

    action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
    obs_config = ObservationConfig()
    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=args.headless)
    env.launch()

    try:
        task_cls = getattr(rlbench_tasks, task_name)
        task = env.get_task(task_cls)
        task.set_variation(variation)

        shape_names, joint_names = _list_shapes_and_joints(root_model)

        # candidates: union over all snapshot models (more robust than picking one root)
        shape_set, joint_set = set(), set()
        for rn in model_names:
            s_list, j_list = _list_shapes_and_joints(str(rn))
            shape_set.update(s_list)
            joint_set.update(j_list)

        shape_names = sorted(shape_set)
        joint_names = sorted(joint_set)

        task.reset()

        init_shape_pose = {n: _get_pose_by_name(n) for n in shape_names}
        init_joint_pos = {}
        for jn in joint_names:
            try:
                init_joint_pos[jn] = float(Joint.get_object(jn).get_joint_position())
            except Exception:
                pass

        # restore final settled snapshot
        pr = _get_pyrep(env, task)
        task.reset()
        for tree in final_trees:
            pr.set_configuration_tree(tree)
        _settle(task, g_last, args.settle_steps)

        final_shape_pose = {n: _get_pose_by_name(n) for n in shape_names}
        final_joint_pos = {}
        for jn in joint_names:
            try:
                final_joint_pos[jn] = float(Joint.get_object(jn).get_joint_position())
            except Exception:
                pass

        # score shapes
        shape_scored = []
        for n in shape_names:
            p0, p1 = init_shape_pose.get(n), final_shape_pose.get(n)
            if p0 is None or p1 is None:
                continue
            shape_scored.append((_pose_score(p0, p1), n))
        shape_scored.sort(reverse=True, key=lambda x: x[0])

        # score joints
        joint_scored = []
        for jn in joint_names:
            j0, j1 = init_joint_pos.get(jn), final_joint_pos.get(jn)
            if j0 is None or j1 is None:
                continue
            joint_scored.append((_joint_score(j0, j1), jn))
        joint_scored.sort(reverse=True, key=lambda x: x[0])

        K_shapes = int(args.K)
        K_joints = 4

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
            "notes": "z=[gripper_pose(7),gripper_open(1),shape_pose(7)*Ks,joint_pos(1)*Kj], shapes are non-Dummy movers, joints are movers"
        }

        with open(args.out_json, "w") as f:
            json.dump(spec, f, indent=2)

        print(f"[ok] wrote {args.out_json}")
        print(f"[info] candidates from {len(model_names)} snapshot roots | shapes={len(shape_names)} joints={len(joint_names)}")
        print("[zspec] shape scores (top 20):")
        for score, name in shape_scored[:min(len(shape_scored), 20)]:
            mark = "*" if name in chosen_shapes else " "
            print(f"  {mark} {score: .4f}  {name}")

        print("[zspec] joint scores (top 20):")
        for score, name in joint_scored[:min(len(joint_scored), 20)]:
            mark = "*" if name in chosen_joints else " "
            print(f"  {mark} {score: .4f}  {name}")

    finally:
        env.shutdown()


if __name__ == "__main__":
    main()