#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_wrapper_suffix import ReverseSuffixEnv, SplitSpec
from policy_io_suffix import DEFAULT_SUFFIX_OBS_SPEC
from train_suffix_sac import (
    load_demo_snaps,
    load_split_idx,
    load_bc_checkpoint,
    ResidualBCWrapper,
)


def make_base_env(
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


def load_run_config(model_dir: str):
    path = os.path.join(model_dir, "run_config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"run_config.json not found in {model_dir}")
    with open(path, "r") as f:
        return json.load(f)


def build_eval_env_from_run_config(cfg, render: bool):
    """
    Rebuild the exact environment type used in training.
    For residual mode, wrap the base env with ResidualBCWrapper.
    """
    bc_mode = cfg.get("bc_mode", "none")

    def _make():
        env = make_base_env(
            task=cfg["task"],
            variation=int(cfg["variation"]),
            prep_npz=cfg["prep_npz"],
            zspec_json=cfg["zspec_json"],
            consensus_json=cfg["consensus_json"],
            max_steps=int(cfg["max_steps"]),
            goal_tol=float(cfg["goal_tol"]),
            goal_hold=int(cfg["goal_hold"]),
            seed=int(cfg["seed"]),
            render=render,
        )

        if bc_mode == "residual":
            bc_ckpt = cfg.get("bc_ckpt", None)
            if not bc_ckpt:
                raise ValueError("Residual run_config.json has no bc_ckpt")
            residual_alpha = float(cfg.get("residual_alpha", 0.25))

            bc_device = torch.device("cpu")
            bc_model, bc_obs_mean, bc_obs_std, bc_ckpt_data = load_bc_checkpoint(
                bc_ckpt, bc_device
            )

            env_obs_dim = int(np.prod(env.observation_space.shape))
            env_act_dim = int(np.prod(env.action_space.shape))
            bc_obs_dim = int(bc_ckpt_data["obs_dim"])
            bc_act_dim = int(bc_ckpt_data["act_dim"])

            if env_obs_dim != bc_obs_dim:
                raise ValueError(
                    f"Residual eval obs mismatch: env obs_dim={env_obs_dim}, BC obs_dim={bc_obs_dim}"
                )
            if env_act_dim != bc_act_dim:
                raise ValueError(
                    f"Residual eval act mismatch: env act_dim={env_act_dim}, BC act_dim={bc_act_dim}"
                )

            env = ResidualBCWrapper(
                env=env,
                bc_model=bc_model,
                bc_obs_mean=bc_obs_mean,
                bc_obs_std=bc_obs_std,
                residual_alpha=residual_alpha,
                bc_device=bc_device,
            )

        return env

    return _make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        required=True,
        help="run directory containing sac_suffix.zip, vecnormalize.pkl, and run_config.json",
    )
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    model_path = os.path.join(args.model_dir, "sac_suffix.zip")
    norm_path = os.path.join(args.model_dir, "vecnormalize.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing VecNormalize stats: {norm_path}")

    cfg = load_run_config(args.model_dir)

    print(f"[info] task={cfg['task']} var={cfg['variation']}")
    print(f"[info] bc_mode={cfg.get('bc_mode', 'none')}")
    if cfg.get("bc_mode", "none") == "residual":
        print(f"[info] bc_ckpt={cfg.get('bc_ckpt')}")
        print(f"[info] residual_alpha={cfg.get('residual_alpha')}")

    make_env_fn = build_eval_env_from_run_config(cfg, render=args.render)

    # Probe once
    probe_env = make_env_fn()
    print(f"[info] obs_dim={probe_env.observation_space.shape[0]}")
    print(f"[info] act_dim={probe_env.action_space.shape[0]}")
    probe_env.close()

    # Build VecNormalize exactly like training
    venv = DummyVecEnv([make_env_fn])
    venv = VecNormalize.load(norm_path, venv)
    venv.training = False
    venv.norm_reward = False

    model = SAC.load(model_path, env=venv)

    successes = 0
    final_ds = []
    lengths = []
    returns = []

    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        ep_steps = 0
        ep_return = 0.0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done_arr, info = venv.step(action)

            done = bool(done_arr[0])
            ep_steps += 1
            ep_return += float(reward[0])
            last_info = info[0]

            if last_info.get("success", False):
                done = True

        d = float(last_info.get("d", np.nan))
        s = bool(last_info.get("success", False))
        successes += int(s)
        final_ds.append(d)
        lengths.append(ep_steps)
        returns.append(ep_return)

        print(
            f"[ep {ep:03d}] success={s} steps={ep_steps} "
            f"return={ep_return:.4f} final_d={d:.6f}"
        )

    print("\n=== Summary ===")
    print(f"success_rate = {successes}/{args.episodes} = {successes / args.episodes:.3f}")
    print(
        f"final_d: mean={np.mean(final_ds):.6f} "
        f"std={np.std(final_ds):.6f} "
        f"min={np.min(final_ds):.6f} "
        f"max={np.max(final_ds):.6f}"
    )
    print(f"return: mean={np.mean(returns):.4f} std={np.std(returns):.4f}")
    print(f"steps:  mean={np.mean(lengths):.2f}")

    venv.close()


if __name__ == "__main__":
    main()