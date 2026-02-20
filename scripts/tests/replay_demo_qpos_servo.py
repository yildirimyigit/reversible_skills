#!/usr/bin/env python3
import argparse
import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

from snapshot_utils import load_demo_npz, restore_keyframe, get_observation_strict


def task_success(task) -> bool:
    try:
        s = task._task.success()
        if isinstance(s, (tuple, list)):
            return bool(s[0])
        return bool(s)
    except Exception:
        return False


def servo_to_q(task, q_des, g_cmd, kp, vmax, tol, max_steps):
    """
    Velocity-servo toward q_des while holding discrete gripper command g_cmd.
    Returns True if task succeeds during servo, else False.
    """
    q_des = np.asarray(q_des, dtype=np.float32).ravel()
    for _ in range(int(max_steps)):
        obs = get_observation_strict(task)
        q_now = np.asarray(obs.joint_positions, dtype=np.float32).ravel()
        err = q_des - q_now
        if float(np.linalg.norm(err)) <= float(tol):
            return task_success(task)

        v = np.clip(float(kp) * err, -float(vmax), float(vmax)).astype(np.float32)
        a = np.concatenate([v, np.array([float(g_cmd)], dtype=np.float32)], axis=0)

        ret = task.step(a)
        done = bool(ret[2]) if isinstance(ret, (tuple, list)) and len(ret) >= 3 else False
        if done or task_success(task):
            return True

    return task_success(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--headless", action="store_true")

    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--kp", type=float, default=4.0)
    ap.add_argument("--vmax", type=float, default=0.4)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--max_steps", type=int, default=80)

    ap.add_argument("--invert_gripper", action="store_true",
                    help="Flip 0/1 mapping if your Discrete mode uses opposite convention.")

    ap.add_argument("--no_snapshot_start", action="store_true",
                    help="Do not restore demo snapshot at start; use task.reset() instead (not recommended for contact tasks).")
    args = ap.parse_args()

    data = load_demo_npz(args.demo_npz)

    if "action_qpos" not in data.files:
        raise RuntimeError("NPZ missing action_qpos. This script replays action_qpos exactly.")

    action_qpos = np.asarray(data["action_qpos"], dtype=np.float32)
    if action_qpos.ndim != 2 or action_qpos.shape[1] < 8:
        raise RuntimeError(f"Unexpected action_qpos shape {action_qpos.shape}; expected (T-1, 8).")

    Tm1 = int(action_qpos.shape[0])  # actions for steps 0..T-2 (targets q_{t+1})
    stride = int(max(1, args.stride))

    # Minimal obs config (keep task_low_dim_state available for some tasks; overhead camera optional)
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
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

        # Critical for contact tasks: start from the exact recorded initial simulator state
        if not args.no_snapshot_start:
            try:
                restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to restore snapshot initial state. "
                    f"If your NPZ truly has no snapshots, rerun with --no_snapshot_start. Error: {e}"
                )
        else:
            task.reset()

        # Replay actions: action_qpos[i] targets state (i+1)
        # Each row: [q_target(7), g_cmd(1)]
        for i in range(0, Tm1, stride):
            q_des = action_qpos[i, :7]
            g_cmd = float(action_qpos[i, 7])
            if args.invert_gripper:
                g_cmd = 1.0 - g_cmd

            ok = servo_to_q(task, q_des, g_cmd, args.kp, args.vmax, args.tol, args.max_steps)
            if ok:
                print(f"[replay_qpos_servo] SUCCESS at action i={i}/{Tm1}")
                return

        print(f"[replay_qpos_servo] FAILED (no success in {Tm1} action waypoints)")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
