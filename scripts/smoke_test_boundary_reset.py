from policy_io_suffix import DEFAULT_SUFFIX_OBS_SPEC
from rl_wrapper_suffix import ReverseSuffixEnv, SplitSpec
from train_suffix_sac import load_demo_snaps, load_split_idx
import numpy as np

split_idx = load_split_idx('../data/consensus_splits.json', "CloseDrawer", 0)
split_spec = SplitSpec(split_t=split_idx)


env = ReverseSuffixEnv(
    task_name="CloseDrawer",
    variation=0,
    demo_snaps=load_demo_snaps('../data/demos/CloseDrawer_var00_demo0000_prep.npz'),
    split_spec=split_spec,
    obs_spec=DEFAULT_SUFFIX_OBS_SPEC,
    max_steps=20,
    goal_tol=0.1,
    goal_hold_steps=5,
    reset_mode="boundary",
    zspec_json_path='../data/demos/CloseDrawer_var00_demo0000_prep_zspec.json',
)

obs, info = env.reset()
print("reset d:", info["d"])

for i in range(6):
    obs, reward, terminated, truncated, info = env.step(np.zeros(8, dtype=np.float32))
    print(i, info["d"], info["hold"], terminated, truncated)
