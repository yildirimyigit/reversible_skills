import numpy as np

from rl_wrapper_suffix import ReverseSuffixEnv, DemoSnapshots, SplitSpec
from rollback_triage import get_keyframe_rows  # this exists in rollback_triage.py
import numpy as np

def load_demo_snaps(prep_npz_path: str) -> DemoSnapshots:
    d = np.load(prep_npz_path, allow_pickle=True)

    # q_ref / g_ref
    q_ref = np.asarray(d["joint_positions"], dtype=np.float32).reshape(-1, 7)
    g_ref = np.asarray(d["gripper_open"], dtype=np.float32).reshape(-1)

    # keyframe trees (K,R) + kf indices
    trees_kR, kf, root_names = get_keyframe_rows(d)

    # final snapshot trees: prefer final_snapshot_trees if present; else fall back
    if "final_snapshot_trees" in d.files:
        final_1d = list(np.asarray(d["final_snapshot_trees"], dtype=object).tolist())
    else:
        # If your prep has different key, adapt here
        raise RuntimeError("final_snapshot_trees missing in prep npz.")

    return DemoSnapshots(
        final_snapshot_trees_1d=final_1d,
        keyframe_trees_kR=trees_kR,
        keyframe_indices=kf,
        q_ref=q_ref,
        g_ref=g_ref,
    )

prep_npz = "data/demos/BlockPyramid_var00_demo0000_prep.npz"
zspec_json = "data/demos/BlockPyramid_var00_demo0000_prep_zspec.json"
split_idx = 3  # example keyframe-row index; replace with consensus recommended_split_idx

demo_snaps = load_demo_snaps(prep_npz)
split_spec = SplitSpec(recommended_split_idx=split_idx)

# NOTE: obs_dim is whatever your _extract_lowdim returns.
# Quick hack for now: instantiate once, call reset, and print obs shape.
env_final = ReverseSuffixEnv(
    task_name="BlockPyramid",
    variation=0,
    demo_snaps=demo_snaps,
    split_spec=split_spec,
    obs_dim=15,                # temporary; if mismatch, fix after first run
    max_steps=50,
    reset_mode="final",
    zspec_json_path=zspec_json,
    seed=0,
)

obs, info = env_final.reset()
print("[final reset] obs.shape =", obs.shape, "d =", info["d"])

env_final.reset_mode = "boundary"
obs2, info2 = env_final.reset()
print("[boundary reset] obs.shape =", obs2.shape, "d =", info2["d"])

env_final.close()
