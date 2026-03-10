#!/usr/bin/env python3
import argparse
import os
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_wrapper_suffix import ReverseSuffixEnv, SplitSpec
from train_suffix_sac import load_demo_snaps, load_split_idx
from policy_io_suffix import DEFAULT_SUFFIX_OBS_SPEC


def make_env(
    task,
    variation,
    prep_npz,
    zspec_json,
    consensus_json,
    max_steps,
    goal_tol,
    goal_hold,
    seed,
    render,
):
    demo_snaps = load_demo_snaps(prep_npz)
    split_t = load_split_idx(consensus_json, task, variation)
    split_spec = SplitSpec(split_t=split_t)

    env = ReverseSuffixEnv(
        task_name=task,
        variation=variation,
        demo_snaps=demo_snaps,
        split_spec=split_spec,
        obs_spec=DEFAULT_SUFFIX_OBS_SPEC,
        max_steps=max_steps,
        goal_tol=goal_tol,
        goal_hold_steps=goal_hold,
        reward_mode="shaped",
        reset_mode="final",
        zspec_json_path=zspec_json,
        seed=seed,
        render=render,
    )
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--prep_npz", required=True)
    ap.add_argument("--zspec_json", required=True)
    ap.add_argument("--consensus_json", required=True)

    ap.add_argument(
        "--model_dir",
        required=True,
        help="run directory containing sac_suffix.zip and vecnormalize.pkl",
    )
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--goal_tol", type=float, default=0.05)
    ap.add_argument("--goal_hold", type=int, default=5)

    ap.add_argument("--render", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model_path = os.path.join(args.model_dir, "sac_suffix.zip")
    norm_path = os.path.join(args.model_dir, "vecnormalize.pkl")

    def _make():
        return make_env(
            task=args.task,
            variation=args.variation,
            prep_npz=args.prep_npz,
            zspec_json=args.zspec_json,
            consensus_json=args.consensus_json,
            max_steps=args.max_steps,
            goal_tol=args.goal_tol,
            goal_hold=args.goal_hold,
            seed=args.seed,
            render=args.render,
        )

    # Probe once to print env dimensions under the shared obs contract
    probe_env = _make()
    print(f"[info] obs_dim={probe_env.observation_space.shape[0]}")
    print(f"[info] act_dim={probe_env.action_space.shape[0]}")
    probe_env.close()

    # Build VecNormalize exactly like training
    venv = DummyVecEnv([_make])
    venv = VecNormalize.load(norm_path, venv)
    venv.training = False
    venv.norm_reward = False

    model = SAC.load(model_path, env=venv)

    successes = 0
    final_ds = []
    lengths = []

    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        ep_steps = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done_arr, info = venv.step(action)

            done = bool(done_arr[0])
            ep_steps += 1
            last_info = info[0]

            if last_info.get("success", False):
                done = True

        d = float(last_info.get("d", np.nan))
        s = bool(last_info.get("success", False))
        successes += int(s)
        final_ds.append(d)
        lengths.append(ep_steps)

        print(f"[ep {ep:03d}] success={s} steps={ep_steps} final_d={d:.6f}")

    print("\n=== Summary ===")
    print(f"success_rate = {successes}/{args.episodes} = {successes / args.episodes:.3f}")
    print(
        f"final_d: mean={np.mean(final_ds):.6f} "
        f"std={np.std(final_ds):.6f} "
        f"min={np.min(final_ds):.6f} "
        f"max={np.max(final_ds):.6f}"
    )
    print(f"steps:   mean={np.mean(lengths):.2f}")

    venv.close()


if __name__ == "__main__":
    main()