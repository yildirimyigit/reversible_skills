#!/usr/bin/env python3
"""
test_utils.py

Smoke test for:
- snapshot_utils.py (restore keyframe/post)
- state_utils.py (compact state + distances)

Prints BOTH:
  (a) component-wise errors (joint_positions / gripper_pose / task_low_dim_state)
  (b) compact-state z error

Usage:
  python3 /workspace/scripts/test_utils.py \
    --demo_npz /workspace/data/demos/CloseDrawer_var00_demo0000.npz \
    --task CloseDrawer --variation 0 --kf_index 10 --settle_steps 0

Add --headless to run headless.
"""

import argparse
import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

from snapshot_utils import load_demo_npz, restore_keyframe, restore_post, get_pyrep
from state_utils import compact_state_from_npz, compact_state_from_obs, l2_dist


def l2(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n == 0:
        return float("nan")
    return float(np.linalg.norm(x[:n] - y[:n]))


def get_observation_strict(env, task):
    """
    Prefer scene-level observation getters (more likely to reflect simulator state after set_configuration_tree).
    Fall back to task.get_observation only if needed.
    """
    scene = getattr(task, "_scene", None)
    if scene is not None:
        if hasattr(scene, "_get_observation"):
            return scene._get_observation()
        if hasattr(scene, "get_observation"):
            return scene.get_observation()

    if hasattr(task, "get_observation"):
        return task.get_observation()

    raise RuntimeError("Could not obtain observation (no scene/task observation getter found).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--kf_index", type=int, default=0)
    ap.add_argument("--settle_steps", type=int, default=0)
    ap.add_argument("--headless", action="store_true", help="Run headless (no GUI).")
    args = ap.parse_args()

    data = load_demo_npz(args.demo_npz)

    # Minimal obs config needed for your compact state + debug prints
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_pose = True
    obs_config.gripper_open = True
    obs_config.task_low_dim_state = True

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=args.headless,
    )
    env.launch()

    try:
        if not hasattr(rlbench_tasks, args.task):
            raise ValueError(f"Unknown RLBench task '{args.task}'.")
        task_cls = getattr(rlbench_tasks, args.task)
        task = env.get_task(task_cls)
        task.set_variation(args.variation)

        pyrep = get_pyrep(env, task)

        # ---- Keyframe restore test ----
        t_kf = restore_keyframe(env, task, data, args.kf_index, settle_steps=args.settle_steps)
        obs = get_observation_strict(env, task)

        # Recorded references at timestep t_kf
        q_rec = data["joint_positions"][t_kf]
        g_rec = float(np.asarray(data["gripper_open"][t_kf]).reshape(-1)[0])

        pose_rec = data["gripper_pose"][t_kf] if "gripper_pose" in data.files else None
        low_rec = data["task_low_dim_state"][t_kf] if "task_low_dim_state" in data.files else None

        # Now values
        q_now = obs.joint_positions
        g_now = float(obs.gripper_open)
        pose_now = getattr(obs, "gripper_pose", None)
        low_now = getattr(obs, "task_low_dim_state", None)

        print(f"[keyframe] kf_index={args.kf_index}  timestep t={t_kf}")
        print(f"  joint_positions L2: {l2(q_now, q_rec):.9f}")
        print(f"  gripper_open abs:   {abs(g_now - g_rec):.9f}")
        if pose_rec is not None and pose_now is not None:
            print(f"  gripper_pose L2:    {l2(pose_now, pose_rec):.9f}")
        else:
            print("  gripper_pose:       (missing)")
        if low_rec is not None and low_now is not None:
            print(f"  task_low_dim L2:    {l2(low_now, low_rec):.9f}")
        else:
            print("  task_low_dim_state: (missing)")

        z_rec = compact_state_from_npz(data, t_kf)
        z_now = compact_state_from_obs(obs)
        print(f"  z L2 error:         {l2_dist(z_now, z_rec):.9f}")
        print(f"  z dims now/rec:     {z_now.size}/{z_rec.size}")

        # ---- Post restore test ----
        t_post = restore_post(env, task, data, settle_steps=args.settle_steps)
        obs2 = get_observation_strict(env, task)

        q_rec2 = data["joint_positions"][t_post]
        g_rec2 = float(np.asarray(data["gripper_open"][t_post]).reshape(-1)[0])

        pose_rec2 = data["gripper_pose"][t_post] if "gripper_pose" in data.files else None
        low_rec2 = data["task_low_dim_state"][t_post] if "task_low_dim_state" in data.files else None

        print(f"\n[post] timestep t={t_post}")
        print(f"  joint_positions L2: {l2(obs2.joint_positions, q_rec2):.9f}")
        print(f"  gripper_open abs:   {abs(float(obs2.gripper_open) - g_rec2):.9f}")
        if pose_rec2 is not None and getattr(obs2, "gripper_pose", None) is not None:
            print(f"  gripper_pose L2:    {l2(obs2.gripper_pose, pose_rec2):.9f}")
        else:
            print("  gripper_pose:       (missing)")
        if low_rec2 is not None and getattr(obs2, "task_low_dim_state", None) is not None:
            print(f"  task_low_dim L2:    {l2(obs2.task_low_dim_state, low_rec2):.9f}")
        else:
            print("  task_low_dim_state: (missing)")

        z_rec2 = compact_state_from_npz(data, t_post)
        z_now2 = compact_state_from_obs(obs2)
        print(f"  z L2 error:         {l2_dist(z_now2, z_rec2):.9f}")
        print(f"  z dims now/rec:     {z_now2.size}/{z_rec2.size}")

    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
