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
    q_des = np.asarray(q_des, dtype=np.float32).ravel()
    for _ in range(int(max_steps)):
        obs = get_observation_strict(task)
        q_now = np.asarray(obs.joint_positions, dtype=np.float32).ravel()
        err = q_des - q_now
        if float(np.linalg.norm(err)) <= float(tol):
            return
        v = np.clip(float(kp) * err, -float(vmax), float(vmax)).astype(np.float32)
        a = np.concatenate([v, np.array([float(g_cmd)], dtype=np.float32)], axis=0)
        task.step(a)


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
    args = ap.parse_args()

    data = load_demo_npz(args.demo_npz)
    q_all = np.asarray(data["joint_positions"], dtype=np.float32)
    g_open = np.asarray(data["gripper_open"], dtype=np.float32).ravel()
    T = int(q_all.shape[0])

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    obs_config.task_low_dim_state = True
    obs_config.overhead_camera.set_all(True)

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=args.headless,
    )
    env.launch()

    try:
        task_cls = getattr(rlbench_tasks, args.task)
        task = env.get_task(task_cls)
        task.set_variation(args.variation)

        # Restore demo initial snapshot (your snapshot_utils)
        restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)

        for t in range(1, T, int(max(1, args.stride))):
            q_des = q_all[t]
            g_des = float(1.0 if g_open[t] > 0.5 else 0.0)
            if args.invert_gripper:
                g_des = 1.0 - g_des

            servo_to_q(task, q_des, g_des, args.kp, args.vmax, args.tol, args.max_steps)

            if task_success(task):
                print(f"[replay_qservo] SUCCESS at t={t}/{T}")
                return

        print(f"[replay_qservo] FAILED (no success in {T} waypoints)")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
