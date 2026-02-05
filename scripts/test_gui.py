import argparse
import time
from rlbench.environment import Environment
from rlbench import tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="StackBlocks")
    args = ap.parse_args()

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.front_camera.set_all(True)

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=False,   # show GUI
    )
    env.launch()

    task = env.get_task(getattr(tasks, args.task))
    task.set_variation(0)

    input("Generating demo. Press Enter to start...")
    _ = task.get_demos(amount=1, live_demos=True)  # runs the demo in the sim

    input("Demo finished. Press Enter to close...")
    env.shutdown()


if __name__ == "__main__":
    main()