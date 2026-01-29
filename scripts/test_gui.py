import time
from rlbench.environment import Environment
from rlbench.tasks import TakeUsbOutOfComputer
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.front_camera.set_all(True)

env = Environment(
    MoveArmThenGripper(JointVelocity(), Discrete()),
    obs_config=obs_config,
    headless=False,   # show GUI
)
env.launch()

task = env.get_task(TakeUsbOutOfComputer)
task.set_variation(0)

print("Generating + running 1 live expert demo...")
_ = task.get_demos(amount=1, live_demos=True)  # runs the demo in the sim

input("Demo finished. Press Enter to close...")
env.shutdown()
