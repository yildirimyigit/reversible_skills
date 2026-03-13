import numpy as np
from pyrep.objects.joint import Joint

from policy_io_suffix import DEFAULT_SUFFIX_OBS_SPEC
from rl_wrapper_suffix import ReverseSuffixEnv, SplitSpec
from train_suffix_sac import load_demo_snaps, load_split_idx
import os
from stable_baselines3 import SAC

def drawer_state():
    return np.array([
        Joint("drawer_joint_bottom").get_joint_position(),
        Joint("drawer_joint_middle").get_joint_position(),
        Joint("drawer_joint_top").get_joint_position(),
    ], dtype=np.float32)

split_idx = load_split_idx("../data/consensus_splits.json", "CloseDrawer", 0)
env = ReverseSuffixEnv(
    task_name="CloseDrawer",
    variation=0,
    demo_snaps=load_demo_snaps("../data/demos/CloseDrawer_var00_demo0000_prep.npz"),
    split_spec=SplitSpec(split_t=split_idx),
    obs_spec=DEFAULT_SUFFIX_OBS_SPEC,
    max_steps=20,
    goal_tol=0.1,
    goal_hold_steps=5,
    reset_mode="final",
    zspec_json_path="../data/demos/CloseDrawer_var00_demo0000_prep_zspec.json",
    render=True,
)

# xs = []
# for i in range(20):
#     obs, info = env.reset()
#     qd = drawer_state()
#     xs.append(qd.copy())
#     print(i, qd, "d=", info["d"])

# xs = np.stack(xs, axis=0)
# print("mean:", xs.mean(axis=0))
# print("std :", xs.std(axis=0))
# env.close()


model_path = os.path.join("../runs/sac_suffix_closedrawer_var00_residual/sac_suffix.zip")
norm_path = os.path.join("../runs/sac_suffix_closedrawer_var00_residual/vecnormalize.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Missing model: {model_path}")
if not os.path.exists(norm_path):
    raise FileNotFoundError(f"Missing VecNormalize stats: {norm_path}")



def drawer_state():
    return np.array([
        Joint("drawer_joint_bottom").get_joint_position(),
        Joint("drawer_joint_middle").get_joint_position(),
        Joint("drawer_joint_top").get_joint_position(),
    ], dtype=np.float32)

obs, info = env.reset()
print("RESET     ", drawer_state(), "d=", info["d"])
input("Look at the scene now, then press Enter...")

obs2, r2, term2, trunc2, info2 = env.step(np.zeros(8, dtype=np.float32))
print("AFTER NOOP", drawer_state(), "d=", info2["d"])
input("Look again, then press Enter...")

model = SAC.load(model_path, env=env)

action, _ = model.predict(obs, deterministic=True)
print("ACTION    ", action)

obs3, r3, term3, trunc3, info3 = env.step(action)
print("AFTER ACT ", drawer_state(), "d=", info3["d"])
input("Look again, then press Enter...")