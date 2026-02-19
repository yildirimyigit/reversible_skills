#!/usr/bin/env python3
import argparse
import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

from snapshot_utils import load_demo_npz, restore_keyframe


def _task_success(task) -> bool:
    try:
        s = task._task.success()
        if isinstance(s, (tuple, list)):
            return bool(s[0])
        return bool(s)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    data = load_demo_npz(args.demo_npz)
    actions = np.asarray(data["action"], dtype=np.float32)

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_pose = True
    obs_config.task_low_dim_state = True

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

        # Critical: restore exactly the recorded initial snapshot.
        restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)

        done_at = None
        for i in range(actions.shape[0]):
            ret = task.step(actions[i])
            done = bool(ret[2]) if isinstance(ret, (tuple, list)) and len(ret) >= 3 else False
            if done or _task_success(task):
                done_at = i + 1
                break

        print(f"[plain replay] success={_task_success(task)} done_at={done_at} / T={actions.shape[0]}")

    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
