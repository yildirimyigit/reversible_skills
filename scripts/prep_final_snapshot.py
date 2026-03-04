#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

from pyrep.objects.object import Object
from pyrep.backend import sim as sim_backend


EXCLUDE_NAME_SUBSTR = [
    "waypoint", "anchor", "target", "success", "fail",
    "defaultlights", "resizablefloor", "xyzcamera", "camera",
    "cam_", "vision_sensor",  # important for your error
    "panda", "franka", "collision", "dummy", "plane", "boundary",
]

def _keep_name(name: str) -> bool:
    ln = str(name).lower()
    return not any(sub in ln for sub in EXCLUDE_NAME_SUBSTR)


def _tree_ptr_to_bytes(ptr):
    # ptr is a cffi char* returned by simGetConfigurationTree
    # First 4 bytes store total length (little-endian uint32), like your existing demos.
    n = int.from_bytes(bytes(sim_backend.ffi.buffer(ptr, 4)), byteorder="little", signed=False)
    if n <= 0 or n > 50_000_000:
        raise RuntimeError(f"Unreasonable configuration tree size: {n}")
    b = bytes(sim_backend.ffi.buffer(ptr, n))
    sim_backend.simReleaseBuffer(ptr)
    return b

def capture_trees(model_names):
    trees = []
    kept_names = []
    for name in model_names:
        name = str(name)
        if not _keep_name(name):
            continue
        try:
            obj = Object.get_object(name)
        except Exception:
            # name not in scene (e.g., cam_front) -> skip
            continue
        try:
            ptr = sim_backend.simGetConfigurationTree(obj.get_handle())
            trees.append(_tree_ptr_to_bytes(ptr))
            kept_names.append(name)
        except Exception:
            # if tree capture fails for any reason, skip
            continue
    return np.asarray(kept_names, dtype=object), np.asarray(trees, dtype=object)

def as_f32(x):
    return np.asarray(x, dtype=np.float32)


def replay_forward_qservo(task, q_ref, g_ref, kp=6.0, vmax=1.0):
    """
    One RLBench step per recorded timestep.
    Uses a proportional joint-velocity controller towards q_ref[t+1].
    """
    q_ref = as_f32(q_ref).reshape(-1, 7)
    g_ref = as_f32(g_ref).reshape(-1)

    desc, obs = task.reset()
    T = q_ref.shape[0]
    for t in range(T - 1):
        q_now = as_f32(obs.joint_positions).reshape(-1)
        q_des = q_ref[t + 1]
        vel = np.clip(kp * (q_des - q_now), -vmax, vmax)

        g_cmd = 1.0 if float(g_ref[t + 1]) > 0.5 else 0.0
        action = np.concatenate([vel, np.array([g_cmd], dtype=np.float32)], axis=0)

        obs, _, _ = task.step(action)

    return obs  # last obs


def settle(task, g_cmd_last: float, settle_steps: int):
    """
    Zero arm velocity, keep gripper command stable.
    Discrete gripper only actuates if commanded state differs, so repeated commands are safe. :contentReference[oaicite:1]{index=1}
    """
    action = np.zeros((8,), dtype=np.float32)
    action[-1] = 1.0 if g_cmd_last > 0.5 else 0.0
    obs = None
    for _ in range(int(settle_steps)):
        obs, _, _ = task.step(action)
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--out_npz", default=None)
    ap.add_argument("--headless", action="store_true", default=False)

    ap.add_argument("--kp", type=float, default=6.0)
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--settle_steps", type=int, default=30)

    ap.add_argument("--validate_restore", action="store_true", default=False)
    args = ap.parse_args()

    data = np.load(args.demo_npz, allow_pickle=True)

    task_name = str(data["task"][0])
    variation = int(data["variation"][0])

    q_ref = data["joint_positions"]          # (T,7)
    g_ref = data["gripper_open"]             # (T,1) or (T,)
    model_names = data["snapshot_model_names"]

    if args.out_npz is None:
        base, ext = os.path.splitext(args.demo_npz)
        args.out_npz = base + "_prep" + ext

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(),
        gripper_action_mode=Discrete()
    )
    obs_config = ObservationConfig()  # keep minimal; we only need joint + gripper, which are in low-dim obs.
    try:
        obs_config.set_all(False)
    except Exception:
        pass
    for attr in ["joint_positions", "gripper_open", "gripper_pose"]:
        if hasattr(obs_config, attr):
            setattr(obs_config, attr, True)

    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=args.headless)
    env.launch()
    try:
        pr = getattr(env, "_pyrep", None)
        if pr is not None:
            if hasattr(pr, "set_rendering"):
                pr.set_rendering(False)
            elif hasattr(pr, "set_rendering_enabled"):
                pr.set_rendering_enabled(False)
    except Exception:
        pass

    try:
        task_cls = getattr(rlbench_tasks, task_name)
        task = env.get_task(task_cls)
        task.set_variation(variation)  # fixed variation :contentReference[oaicite:2]{index=2}

        # Replay forward and settle
        obs_last = replay_forward_qservo(task, q_ref, g_ref, kp=args.kp, vmax=args.vmax)
        g_last = float(as_f32(g_ref).reshape(-1)[-1])
        obs_last = settle(task, g_last, args.settle_steps)

        # Capture final snapshot trees
        final_model_names, final_trees = capture_trees(model_names)

        # Optional sanity check: reset and restore final snapshot, compare joint positions roughly
        if args.validate_restore:
            task.reset()
            # Get the underlying PyRep instance (works for RLBench Environment)
            pr = getattr(env, "_pyrep", None)
            if pr is None:
                # fallback: task scene usually has it
                pr = getattr(getattr(task, "_scene", None), "_pyrep", None)
            if pr is None:
                raise RuntimeError("Could not access PyRep instance to restore configuration trees.")

            for tree in final_trees:
                pr.set_configuration_tree(tree)

            # Important corner case:
            # If the final gripper state is "closed", Discrete will NOT call grasp() unless it detects a change.
            # So we do a tiny fix-up to re-grasp any graspables if needed.
            if g_last < 0.5:
                scene = getattr(task, "_scene", None)
                inner_task = getattr(task, "_task", None)
                if scene is not None and inner_task is not None and hasattr(inner_task, "get_graspable_objects"):
                    graspables = inner_task.get_graspable_objects()
                    gripper = scene.robot.gripper
                    for gobj in graspables:
                        try:
                            # distance between gripper and object
                            d = gripper.check_distance(gobj)
                        except Exception:
                            d = 1e9
                        if d < 0.03:
                            gripper.grasp(gobj)
                            break

            settle(task, g_last, 10)
            settle(task, g_last, 10)
            obs2, _, _ = task.step(np.zeros((8,), dtype=np.float32))
            err = np.linalg.norm(as_f32(obs2.joint_positions) - as_f32(obs_last.joint_positions))
            print(f"[validate] ||q_restored - q_final|| = {err:.4f}")

        # Write updated NPZ
        out = {k: data[k] for k in data.files}
        out["final_snapshot_model_names"] = np.asarray(final_model_names, dtype=object)
        out["final_snapshot_trees"] = final_trees
        out["final_settle_steps"] = np.asarray([args.settle_steps], dtype=np.int32)
        out["final_gripper_open"] = np.asarray([g_last], dtype=np.float32)

        np.savez_compressed(args.out_npz, **out)
        print(f"[ok] wrote {args.out_npz}")

    finally:
        try:
            pr = getattr(env, "_pyrep", None)
            if pr is not None:
                if hasattr(pr, "set_rendering"):
                    pr.set_rendering(True)
                elif hasattr(pr, "set_rendering_enabled"):
                    pr.set_rendering_enabled(True)
        except Exception:
            pass
        env.shutdown()


if __name__ == "__main__":
    main()