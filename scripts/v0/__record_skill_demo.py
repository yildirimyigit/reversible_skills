#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np

from rlbench.environment import Environment
from rlbench.tasks import StackBlocks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete


def pack_obs(obs):
    """Pick and convert fields you care about into numpy arrays."""
    out = {}

    # Joint states
    out["joint_positions"] = np.asarray(obs.joint_positions, dtype=np.float32)
    out["joint_velocities"] = np.asarray(obs.joint_velocities, dtype=np.float32)
    if obs.joint_forces is not None:
        out["joint_forces"] = np.asarray(obs.joint_forces, dtype=np.float32)

    # Gripper and ee state
    out["gripper_open"] = np.asarray([obs.gripper_open], dtype=np.float32)
    if obs.gripper_pose is not None:
        out["gripper_pose"] = np.asarray(obs.gripper_pose, dtype=np.float32)
    if obs.gripper_joint_positions is not None:
        out["gripper_joint_positions"] = np.asarray(obs.gripper_joint_positions, dtype=np.float32)

    # Cameras (if enabled in obs_config)
    if obs.front_rgb is not None:
        out["front_rgb"] = obs.front_rgb.astype(np.uint8)
    if obs.front_depth is not None:
        out["front_depth"] = obs.front_depth.astype(np.float32)
    if obs.front_mask is not None:
        out["front_mask"] = obs.front_mask.astype(np.uint8)

    if obs.wrist_rgb is not None:
        out["wrist_rgb"] = obs.wrist_rgb.astype(np.uint8)
    if obs.wrist_depth is not None:
        out["wrist_depth"] = obs.wrist_depth.astype(np.float32)
    if obs.wrist_mask is not None:
        out["wrist_mask"] = obs.wrist_mask.astype(np.uint8)

    return out


def stack_trajectory(frames):
    """frames: list[dict]. Return dict[key] -> np.ndarray stacked over time."""
    keys = sorted({k for f in frames for k in f.keys()})
    traj = {}
    for k in keys:
        vals = [f[k] for f in frames if k in f]
        # If a key is missing in some frames (unlikely), skip it.
        if len(vals) != len(frames):
            continue
        traj[k] = np.stack(vals, axis=0)
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="/workspace/data/stackblocks_demos")
    ap.add_argument("--n", type=int, default=1, help="number of demos to record")
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--img", type=int, default=128, help="camera image size (square)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Observation config
    obs_config = ObservationConfig()
    obs_config.set_all(False)

    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.joint_forces = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = True
    obs_config.gripper_joint_positions = True

    obs_config.front_camera.set_all(True)
    obs_config.wrist_camera.set_all(True)
    obs_config.front_camera.image_size = (args.img, args.img)
    obs_config.wrist_camera.image_size = (args.img, args.img)

    action_mode = MoveArmThenGripper(JointVelocity(), Discrete())

    env = Environment(action_mode, obs_config=obs_config, headless=args.headless)
    env.launch()

    try:
        task = env.get_task(StackBlocks)
        task.set_variation(args.variation)

        # Motion planned (expert) demos generated on the fly
        demos = task.get_demos(amount=args.n, live_demos=True)

        for i, demo in enumerate(demos):
            frames = []
            for item in demo:
                # Some forks wrap obs, but upstream commonly gives Observation directly.
                obs = getattr(item, "observation", item)
                frames.append(pack_obs(obs))

            traj = stack_trajectory(frames)
            traj["task"] = np.array(["StackBlocks"])
            traj["variation"] = np.array([args.variation], dtype=np.int32)
            traj["demo_index"] = np.array([i], dtype=np.int32)
            traj["timestamp"] = np.array([time.time()], dtype=np.float64)

            out_path = os.path.join(args.out_dir, f"StackBlocks_var{args.variation:02d}_demo{i:04d}.npz")
            np.savez_compressed(out_path, **traj)
            print(f"Saved {out_path}  T={traj['joint_positions'].shape[0]}")

    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
