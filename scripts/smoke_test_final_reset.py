from policy_io_suffix import DEFAULT_SUFFIX_OBS_SPEC, build_reverse_action_target
from rl_wrapper_suffix import ReverseSuffixEnv, SplitSpec
from train_suffix_sac import load_demo_snaps, load_split_idx
import numpy as np

task = "CloseDrawer"
variation = 0
prep_npz = "../data/demos/CloseDrawer_var00_demo0000_prep.npz"
zspec_json = "../data/demos/CloseDrawer_var00_demo0000_prep_zspec.json"
consensus_json = "../data/consensus_splits.json"

demo_snaps = load_demo_snaps(prep_npz)
split_idx = load_split_idx(consensus_json, task, variation)
split_spec = SplitSpec(split_t=split_idx)

env = ReverseSuffixEnv(
    task_name=task,
    variation=variation,
    demo_snaps=demo_snaps,
    split_spec=split_spec,
    obs_spec=DEFAULT_SUFFIX_OBS_SPEC,
    max_steps=300,
    goal_tol=0.1,
    goal_hold_steps=5,
    reset_mode="final",
    zspec_json_path=zspec_json,
)

obs, info = env.reset()
print("reset_mode=final")
print("initial d:", info["d"])
print("boundary_t:", info["boundary_t"], "split_t:", info["split_t"])

q_ref = demo_snaps.q_ref
g_ref = demo_snaps.g_ref
boundary_t = int(info["boundary_t"])
T = q_ref.shape[0]

best_d = info["d"]

for t in range(T - 1, boundary_t, -1):
    act = build_reverse_action_target(
        q_now=q_ref[t],
        q_prev=q_ref[t - 1],
        g_prev=np.array([g_ref[t - 1]], dtype=np.float32),
        arm_k=4.0,
        include_gripper_action=True,
    )

    obs, reward, terminated, truncated, step_info = env.step(act)
    best_d = min(best_d, step_info["d"])

    print(
        f"ref_t={t:03d} -> {t-1:03d} | "
        f"d={step_info['d']:.6f} | hold={step_info['hold']} | "
        f"term={terminated} | trunc={truncated}"
    )

    if terminated or truncated:
        break

print("best_d:", best_d)
env.close()