#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from reversible_skills_wrapper import ReverseSkillEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)

    ap.add_argument("--model_path", required=True, help="Path to sac_reverse_skill.zip")
    ap.add_argument("--norm_path", required=True, help="Path to vecnormalize.pkl")

    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=None, help="Override env max steps if set.")
    ap.add_argument("--settle_steps", type=int, default=10)
    ap.add_argument("--open_map", type=str, default=None)
    ap.add_argument("--spatial_map", type=str, default=None)

    ap.add_argument("--curriculum", type=float, default=1.0,
                    help="Reset curriculum level in [0,1]. 1=start from post, 0=start from easiest keyframe.")
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic actions.")
    ap.add_argument("--sleep", type=float, default=0.03,
                    help="Seconds to sleep between steps for nicer visualization.")
    args = ap.parse_args()

    if args.spatial_map is not None and not os.path.isfile(args.spatial_map):
        raise FileNotFoundError(args.spatial_map)

    # Force GUI visualization
    headless = False

    def make_env():
        env = ReverseSkillEnv(
            demo_npz_path=args.demo_npz,
            task_name=args.task,
            variation=args.variation,
            headless=headless,
            max_episode_steps=(args.max_steps if args.max_steps is not None else 150),
            settle_steps_after_restore=args.settle_steps,
            open_map_path=args.open_map,
            spatial_map_path=args.spatial_map,
        )
        env.set_curriculum(args.curriculum)
        return Monitor(env)

    venv = DummyVecEnv([make_env])

    # Load VecNormalize stats (important!)
    venv = VecNormalize.load(args.norm_path, venv)
    venv.training = False
    venv.norm_reward = False

    model = SAC.load(args.model_path, env=venv)

    successes = 0

    for ep in range(args.episodes):
        obs = venv.reset()
        ep_return = 0.0
        ep_len = 0

        # show reset info from Monitor/env if available
        # (DummyVecEnv doesn't expose info directly, so we just run)

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, dones, infos = venv.step(action)

            ep_return += float(reward[0])
            ep_len += 1
            done = bool(dones[0])

            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            atoms = info.get("atoms", None)
            if atoms is not None:
                # Print a compact predicate status line
                sat = " ".join([f"{k}:{'1' if v else '0'}" for k, v in atoms.items()])
                print(f"[ep{ep:02d} step{ep_len:03d}] r={reward[0]:+.3f}  {sat}")

            if args.sleep > 0:
                time.sleep(args.sleep)

            if ep_len >= 5000:
                print("Safety break (too long).")
                break

        # termination reason
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        success = False
        atoms = info.get("atoms", {})

        success = bool(info.get("atoms")) and all(info["atoms"].values())

        successes += int(success)
        print(f"Episode {ep+1}/{args.episodes}: len={ep_len} return={ep_return:.2f} success={success}")

    print(f"\nSuccess rate: {successes}/{args.episodes} = {successes/args.episodes:.2f}")


if __name__ == "__main__":
    main()
