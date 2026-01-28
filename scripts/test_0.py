from rlbench.environment import Environment
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import ReachTarget

import numpy as np


# DATASET = 'path_to_rlbench_data'


# class Agent(object):

#     def __init__(self, action_shape):
#         self.action_shape = action_shape

#     def ingest(self, demos):
#         pass

#     def act(self, obs):
#         arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
#         gripper = [1.0]  # Always open
#         return np.concatenate([arm, gripper], axis=-1)


action_mode = MoveArmThenGripper(
  arm_action_mode=JointVelocity(),
  gripper_action_mode=Discrete()
)


env = Environment(action_mode)
env.launch()

task = env.get_task(ReachTarget)
descriptions, obs = task.reset()
obs, reward, terminate = task.step(np.random.normal(size=env.action_shape))

# demos = task.get_demos(2)

# agent = Agent(env.action_shape)
# agent.ingest(demos)

# training_steps = 100
# episode_length = 100
# obs = None

# for i in range(training_steps):
#     if i % episode_length == 0:
#         descriptions, obs = task.reset()
#     action = agent.act(obs)
#     obs, reward, terminate = task.step(action)
# env.shutdown()