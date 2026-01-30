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


def _as_f32(x):
    return np.asarray(x, dtype=np.float32)


def pack_obs(obs, record_front=True, record_wrist=True):
    """Minimal fields for reverse-feasibility + Gemini frames."""
    out = {}
    out["joint_positions"] = _as_f32(obs.joint_positions)      # (7,)
    out["joint_velocities"] = _as_f32(obs.joint_velocities)    # (7,)

    out["gripper_open"] = _as_f32([obs.gripper_open])          # (1,)
    if obs.gripper_pose is not None:
        out["gripper_pose"] = _as_f32(obs.gripper_pose)        # (7,)

    if record_front and obs.front_rgb is not None:
        out["front_rgb"] = obs.front_rgb.astype(np.uint8)      # (H,W,3)
    if record_wrist and obs.wrist_rgb is not None:
        out["wrist_rgb"] = obs.wrist_rgb.astype(np.uint8)      # (H,W,3)
    return out


def stack_trajectory(frames):
    keys = sorted({k for f in frames for k in f.keys()})
    traj = {}
    for k in keys:
        if any(k not in f for f in frames):
            continue
        traj[k] = np.stack([f[k] for f in frames], axis=0)
    return traj


def derive_actions(traj, gripper_threshold=0.03):
    """
    Derive action proxies consistent with:
      MoveArmThenGripper(JointVelocity(), Discrete())

    Arm action: use observed joint_velocities as a proxy for commanded joint velocity.
    Gripper action: binarize gripper_open into {0,1}.

    Saves:
      action_arm:     (T,7)
      action_gripper: (T,1)
      action:         (T,8) concatenated
    """
    T = traj["joint_velocities"].shape[0]

    action_arm = traj["joint_velocities"].astype(np.float32)                 # (T,7)

    # gripper_open is float; convert to discrete open/close command
    go = traj["gripper_open"].reshape(T)                                     # (T,)
    action_gripper = (go > gripper_threshold).astype(np.float32).reshape(T, 1)

    action = np.concatenate([action_arm, action_gripper], axis=1)            # (T,8)

    traj["action_arm"] = action_arm
    traj["action_gripper"] = action_gripper
    traj["action"] = action

    # Helpful metadata
    traj["action_is_derived"] = np.array([1], dtype=np.int32)
    traj["action_arm_source"] = np.array(["joint_velocities"], dtype="<U32")
    traj["action_gripper_source"] = np.array(["threshold(gripper_open)"], dtype="<U64")
    traj["gripper_threshold"] = np.array([gripper_threshold], dtype=np.float32)


def select_keyframes(gripper_open_T1, T, max_k=12, event_window=3):
    """
    Pick Gemini-friendly indices:
      - always include start & end
      - include indices where gripper state flips (open<->close)
      - add +/- event_window around flips
      - fill remaining with uniform samples
    """
    go = gripper_open_T1.reshape(T)  # (T,)
    bin_state = (go > 0.03).astype(np.int32)
    flips = np.where(bin_state[1:] != bin_state[:-1])[0] + 1  # indices of flips

    idx = set()
    idx.add(0)
    idx.add(T - 1)

    for f in flips:
        for d in range(-event_window, event_window + 1):
            j = f + d
            if 0 <= j < T:
                idx.add(j)

    # Fill with uniform samples
    idx = sorted(idx)
    if len(idx) < max_k:
        needed = max_k - len(idx)
        uni = np.linspace(0, T - 1, num=max_k, dtype=np.int32).tolist()
        for u in uni:
            if len(idx) >= max_k:
                break
            if u not in idx:
                idx.append(int(u))
        idx = sorted(set(idx))

    # If too many, subsample but keep start/end
    if len(idx) > max_k:
        keep = [idx[0]]
        middle = idx[1:-1]
        take = np.linspace(0, len(middle) - 1, num=max_k - 2, dtype=np.int32)
        keep += [middle[i] for i in take]
        keep += [idx[-1]]
        idx = keep

    return np.array(idx, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="/workspace/data/stackblocks_demos")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--variation", type=int, default=0)

    # Fix headless UX: default headless=True, allow --no-headless
    ap.add_argument("--headless", action="store_true", default=True)

    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--no_wrist", action="store_true")
    ap.add_argument("--keyframes", type=int, default=12, help="frames saved for Gemini queries")
    ap.add_argument("--event_window", type=int, default=3, help="+/- steps around gripper flips")
    ap.add_argument("--gripper_threshold", type=float, default=0.03)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    obs_config = ObservationConfig()
    obs_config.set_all(False)

    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = True

    obs_config.front_camera.set_all(False)
    obs_config.front_camera.rgb = True
    obs_config.front_camera.image_size = (args.img, args.img)

    obs_config.wrist_camera.set_all(False)
    obs_config.wrist_camera.rgb = True
    obs_config.wrist_camera.image_size = (args.img, args.img)

    action_mode = MoveArmThenGripper(JointVelocity(), Discrete())

    env = Environment(action_mode, obs_config=obs_config, headless=args.headless)
    env.launch()

    try:
        task = env.get_task(StackBlocks)
        task.set_variation(args.variation)

        demos = task.get_demos(amount=args.n, live_demos=True)

        for i, demo in enumerate(demos):
            frames = []
            for obs in demo:
                frames.append(pack_obs(
                    obs,
                    record_front=True,
                    record_wrist=(not args.no_wrist),
                ))

            traj = stack_trajectory(frames)

            # Derive action proxies (since this RLBench build does not expose actions)
            derive_actions(traj, gripper_threshold=args.gripper_threshold)

            # Gemini keyframes
            T = traj["joint_positions"].shape[0]
            k_idx = select_keyframes(
                traj["gripper_open"], T,
                max_k=args.keyframes,
                event_window=args.event_window
            )
            traj["keyframe_indices"] = k_idx

            if "front_rgb" in traj:
                traj["front_rgb_keyframes"] = traj["front_rgb"][k_idx]
            if (not args.no_wrist) and ("wrist_rgb" in traj):
                traj["wrist_rgb_keyframes"] = traj["wrist_rgb"][k_idx]

            # Demo-level metadata
            traj["task"] = np.array(["StackBlocks"])
            traj["variation"] = np.array([args.variation], dtype=np.int32)
            traj["demo_index"] = np.array([i], dtype=np.int32)
            traj["timestamp"] = np.array([time.time()], dtype=np.float64)
            traj["image_size"] = np.array([[args.img, args.img]], dtype=np.int32)

            traj["action_mode"] = np.array(["MoveArmThenGripper"])
            traj["arm_action_mode"] = np.array(["JointVelocity"])
            traj["gripper_action_mode"] = np.array(["Discrete"])

            out_path = os.path.join(
                args.out_dir,
                f"StackBlocks_var{args.variation:02d}_demo{i:04d}.npz"
            )
            np.savez_compressed(out_path, **traj)

            print(f"Saved {out_path}  T={T}  keyframes={len(k_idx)}  actions=DERIVED")

    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
