#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from rl_wrapper_suffix import ReverseSuffixEnv, DemoSnapshots, SplitSpec
from rollback_triage import get_keyframe_rows
from policy_io_suffix import DEFAULT_SUFFIX_OBS_SPEC


def load_demo_snaps(prep_npz_path: str) -> DemoSnapshots:
    d = np.load(prep_npz_path, allow_pickle=True)

    # recorded refs
    q_ref = np.asarray(d["joint_positions"], dtype=np.float32).reshape(-1, 7)
    g_ref = np.asarray(d["gripper_open"], dtype=np.float32).reshape(-1)

    # keyframes
    trees_kR, kf, _root_names = get_keyframe_rows(d)

    # final snapshot trees + final gripper metadata
    if "snapshot_post_trees" in d.files:
        final_1d = list(np.asarray(d["snapshot_post_trees"], dtype=object).tolist())
        final_g = (
            float(np.asarray(d["final_gripper_open"], dtype=np.float32).ravel()[0])
            if "final_gripper_open" in d.files
            else float(g_ref[-1])
        )
        final_settle = (
            int(np.asarray(d["final_settle_steps"], dtype=np.int32).ravel()[0])
            if "final_settle_steps" in d.files
            else 10
        )
    else:
        if "final_snapshot_trees" not in d.files:
            raise RuntimeError(
                "Neither snapshot_post_trees nor final_snapshot_trees found in prep npz."
            )
        final_1d = list(np.asarray(d["final_snapshot_trees"], dtype=object).tolist())
        final_g = float(g_ref[-1])
        final_settle = 10

    return DemoSnapshots(
        final_snapshot_trees_1d=final_1d,
        final_gripper_open=float(final_g),
        final_settle_steps=int(final_settle),
        keyframe_trees_kR=trees_kR,
        keyframe_indices=np.asarray(kf, dtype=np.int64),
        q_ref=q_ref,
        g_ref=g_ref,
    )


def load_split_idx(consensus_json: str, task: str, variation: int) -> int:
    with open(consensus_json, "r") as f:
        data = json.load(f)

    groups = data.get("groups", {})
    group_key = f"{task}__var{int(variation):02d}"
    if group_key not in groups:
        raise KeyError(
            f"{group_key} not found in consensus file. "
            f"Available: {list(groups.keys())[:8]}..."
        )

    g = groups[group_key]
    rec = g.get("recommended_split_idx", None)

    if rec is None:
        raise ValueError(
            f"{group_key} has recommended_split_idx=null "
            f"(accepted={g.get('accepted')}, support={g.get('support')}). "
            "Choose a task/variation with an accepted split "
            "(e.g., CloseDrawer var00 or PutItemInDrawer var00)."
        )
    return int(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)

    ap.add_argument("--prep_npz", required=True)
    ap.add_argument("--zspec_json", required=True)
    ap.add_argument("--consensus_json", required=True)

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--timesteps", type=int, default=300_000)

    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--goal_tol", type=float, default=0.05)
    ap.add_argument("--goal_hold", type=int, default=5)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    demo_snaps = load_demo_snaps(args.prep_npz)
    split_idx = load_split_idx(args.consensus_json, args.task, args.variation)
    split_spec = SplitSpec(split_t=split_idx)

    def make_env():
        env = ReverseSuffixEnv(
            task_name=args.task,
            variation=args.variation,
            demo_snaps=demo_snaps,
            split_spec=split_spec,
            obs_spec=DEFAULT_SUFFIX_OBS_SPEC,
            max_steps=args.max_steps,
            goal_tol=args.goal_tol,
            goal_hold_steps=args.goal_hold,
            reward_mode="shaped",
            reset_mode="final",
            zspec_json_path=args.zspec_json,
            seed=args.seed,
            render=args.render,
        )
        return Monitor(env)

    # Probe once so you can verify the shared observation contract
    probe_env = make_env()
    print(f"[info] split_idx={split_idx}")
    print(f"[info] obs_dim={probe_env.observation_space.shape[0]}")
    print(f"[info] act_dim={probe_env.action_space.shape[0]}")
    probe_env.close()

    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = SAC(
        "MlpPolicy",
        venv,
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(args.outdir, "tb"),
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
    )

    model.learn(total_timesteps=int(args.timesteps))

    model.save(os.path.join(args.outdir, "sac_suffix.zip"))
    venv.save(os.path.join(args.outdir, "vecnormalize.pkl"))
    print("Saved model + VecNormalize to", args.outdir)


if __name__ == "__main__":
    main()