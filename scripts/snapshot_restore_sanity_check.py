#!/usr/bin/env python3
import argparse
import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete


def get_top_level_models(pyrep):
    roots = pyrep.get_objects_in_tree(root_object=None, first_generation_only=True)
    models = [o for o in roots if o.is_model()]
    models.sort(key=lambda o: o.get_name())
    return models


def restore_row(pyrep, model_names, row_bytes, settle_steps=10):
    """
    Restore a configuration snapshot row (list of bytes) into the current scene.
    """
    models = get_top_level_models(pyrep)
    name2m = {m.get_name(): m for m in models}

    missing = [n for n in model_names if n not in name2m]
    if missing:
        raise RuntimeError(f"Missing models in current scene (first 10): {missing[:10]}")

    for name, b in zip(model_names, row_bytes):
        if not isinstance(b, (bytes, bytearray)):
            raise TypeError(f"Snapshot for model {name} is not bytes: {type(b)}")
        if len(b) <= 1:
            raise RuntimeError(f"Snapshot for model {name} too small: len={len(b)}")
        name2m[name].set_configuration_tree(b)

    for _ in range(int(settle_steps)):
        pyrep.step()


def l2(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n == 0:
        return np.nan
    return float(np.linalg.norm(x[:n] - y[:n]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True, help="Path to a recorded .npz demo")
    ap.add_argument("--task", required=True, help="RLBench task class name, e.g. CloseDrawer")
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--kf_index", type=int, default=0, help="Index into keyframe_indices")
    ap.add_argument("--settle_steps", type=int, default=10)
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.set_defaults(headless=True)
    args = ap.parse_args()

    data = np.load(args.demo_npz, allow_pickle=True)

    if "snapshot_keyframe_trees" not in data.files:
        raise RuntimeError("This demo file does not contain snapshot_keyframe_trees (did you record with snapshots?)")

    model_names = data["snapshot_model_names"].tolist()
    kf_idx = data["keyframe_indices"].astype(int).tolist()
    trees = data["snapshot_keyframe_trees"]  # shape (K, M), dtype=object

    if args.kf_index < 0 or args.kf_index >= len(kf_idx):
        raise ValueError(f"--kf_index out of range: {args.kf_index} (K={len(kf_idx)})")

    t = int(kf_idx[args.kf_index])
    row = trees[args.kf_index, :].tolist()

    # Keep obs minimal; we compare restored joints / gripper vs recorded.
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = True
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

        # Create a valid scene instance, then restore into it.
        _ = task.reset()

        scene = task._scene
        pyrep = scene._pyrep

        restore_row(pyrep, model_names, row, settle_steps=args.settle_steps)

        # After restore, read a fresh observation
        obs = task.get_observation()

        # Recorded references at timestep t
        q_rec = data["joint_positions"][t]
        g_rec = float(data["gripper_open"][t].reshape(-1)[0])
        pose_rec = data["gripper_pose"][t] if "gripper_pose" in data.files else None
        low_rec = data["task_low_dim_state"][t] if "task_low_dim_state" in data.files else None

        # Current after restore
        q_now = obs.joint_positions
        g_now = float(obs.gripper_open)
        pose_now = obs.gripper_pose if getattr(obs, "gripper_pose", None) is not None else None
        low_now = obs.task_low_dim_state if getattr(obs, "task_low_dim_state", None) is not None else None

        print(f"Restored keyframe index={args.kf_index}  timestep t={t}")
        print(f"  joint_positions L2 error: {l2(q_now, q_rec):.6f}")
        print(f"  gripper_open abs error:   {abs(g_now - g_rec):.6f}")

        if pose_rec is not None and pose_now is not None:
            print(f"  gripper_pose L2 error:    {l2(pose_now, pose_rec):.6f}")
        else:
            print("  gripper_pose: (missing in record or in observation)")

        if low_rec is not None and low_now is not None:
            print(f"  task_low_dim_state L2 error: {l2(low_now, low_rec):.6f}")
        else:
            print("  task_low_dim_state: (missing in record or in observation)")

        # Optional: also test restoring final snapshot (post)
        if "snapshot_post_trees" in data.files:
            print("\nTesting restore of final (post) snapshot...")
            post_row = data["snapshot_post_trees"].tolist()
            restore_row(pyrep, model_names, post_row, settle_steps=args.settle_steps)
            obs2 = task.get_observation()
            t2 = int(data["joint_positions"].shape[0] - 1)
            q_rec2 = data["joint_positions"][t2]
            g_rec2 = float(data["gripper_open"][t2].reshape(-1)[0])
            print(f"  final joint_positions L2 error: {l2(obs2.joint_positions, q_rec2):.6f}")
            print(f"  final gripper_open abs error:   {abs(float(obs2.gripper_open) - g_rec2):.6f}")

    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
