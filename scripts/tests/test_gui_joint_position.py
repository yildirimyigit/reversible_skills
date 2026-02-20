import argparse
from rlbench.environment import Environment
from rlbench import tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="StackBlocks")
    args = ap.parse_args()

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = True

    obs_config.front_camera.set_all(False)
    obs_config.front_camera.rgb = True
    obs_config.front_camera.image_size = (128, 128)

    obs_config.wrist_camera.set_all(False)
    obs_config.wrist_camera.rgb = True
    obs_config.wrist_camera.image_size = (128, 128)

    obs_config.front_camera.set_all(True)
    obs_config.record_gripper_closing = True
    obs_config.task_low_dim_state = True

    env = Environment(
        MoveArmThenGripper(JointPosition(), Discrete()),
        obs_config=obs_config,
        headless=False,   # show GUI
    )
    env.launch()

    task = env.get_task(getattr(tasks, args.task))
    task.set_variation(0)

    input("Generating demo. Press Enter to start...")
    demo = task.get_demos(amount=1, live_demos=True)[0]  # runs the demo in the sim
    for t in [0, 1, len(demo)//2, len(demo)-1]:
        jp = demo[t].misc.get("joint_poses", None)
        arr = None if jp is None else np.asarray(jp)
        print("t", t,
            "joint_positions shape", np.asarray(demo[t].joint_positions).shape,
            "joint_poses type", type(jp),
            "joint_poses shape", None if arr is None else arr.shape)

    input("Demo finished. Press Enter to close...")
    env.shutdown()


if __name__ == "__main__":
    main()