import numpy as np
from rlbench.environment import Environment
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.front_camera.set_all(True)  # enable RGB/depth/mask as available

action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
env = Environment(action_mode, obs_config=obs_config, headless=True)
env.launch()

task = env.get_task(ReachTarget)
descriptions, obs = task.reset()

print("front_rgb:", None if obs.front_rgb is None else obs.front_rgb.shape)
print("front_depth:", None if obs.front_depth is None else obs.front_depth.shape)

action = np.random.normal(size=env.action_shape)
obs, reward, terminate = task.step(action)

env.shutdown()
print("OK")
