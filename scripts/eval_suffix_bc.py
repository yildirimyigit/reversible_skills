#!/usr/bin/env python3
import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

from rl_wrapper_suffix import ReverseSuffixEnv, DemoSnapshots, SplitSpec
from rollback_triage import get_keyframe_rows
from policy_io_suffix import DEFAULT_SUFFIX_OBS_SPEC


class BCActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.backbone(obs)))


def load_demo_snaps(prep_npz_path: str) -> DemoSnapshots:
    d = np.load(prep_npz_path, allow_pickle=True)

    q_ref = np.asarray(d["joint_positions"], dtype=np.float32).reshape(-1, 7)
    g_ref = np.asarray(d["gripper_open"], dtype=np.float32).reshape(-1)

    trees_kR, kf, _root_names = get_keyframe_rows(d)

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
            "Choose a task/variation with an accepted split."
        )
    return int(rec)


def load_bc_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)

    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    obs_mean = np.asarray(ckpt["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(ckpt["obs_std"], dtype=np.float32)

    model = BCActor(obs_dim=obs_dim, act_dim=act_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, obs_mean, obs_std, ckpt


def make_env(args, demo_snaps, split_spec):
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
        render=args.visualize,
    )
    return env


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs, info
    return out, {}


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), info
    raise RuntimeError(f"Unexpected step() output length: {len(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc_ckpt", required=True)

    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)

    ap.add_argument("--prep_npz", required=True)
    ap.add_argument("--zspec_json", required=True)
    ap.add_argument("--consensus_json", required=True)

    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--goal_tol", type=float, default=0.05)
    ap.add_argument("--goal_hold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--out_json", default=None)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    demo_snaps = load_demo_snaps(args.prep_npz)
    split_idx = load_split_idx(args.consensus_json, args.task, args.variation)
    split_spec = SplitSpec(split_t=split_idx)

    env = make_env(args, demo_snaps, split_spec)
    model, obs_mean, obs_std, ckpt = load_bc_checkpoint(args.bc_ckpt, device)

    obs, reset_info = reset_env(env)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)

    env_obs_dim = int(obs.shape[0])
    bc_obs_dim = int(ckpt["obs_dim"])
    bc_act_dim = int(ckpt["act_dim"])
    env_act_dim = int(np.prod(env.action_space.shape))

    print(f"[info] split_idx={split_idx}")
    print(f"[info] env_obs_dim={env_obs_dim} | bc_obs_dim={bc_obs_dim}")
    print(f"[info] env_act_dim={env_act_dim} | bc_act_dim={bc_act_dim}")

    if "obs_spec" in ckpt:
        print(f"[info] bc obs_spec = {ckpt['obs_spec']}")

    if env_obs_dim != bc_obs_dim:
        env.close()
        raise ValueError(
            f"Observation mismatch: env emits {env_obs_dim} dims but BC checkpoint expects {bc_obs_dim}. "
            "Retrain BC with the same policy observation used by ReverseSuffixEnv."
        )

    if env_act_dim != bc_act_dim:
        env.close()
        raise ValueError(
            f"Action mismatch: env expects {env_act_dim} dims but BC checkpoint outputs {bc_act_dim}."
        )

    episode_rows = []
    num_success = 0

    for ep in range(args.episodes):
        obs, reset_info = reset_env(env)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        done = False
        ep_ret = 0.0
        ep_len = 0
        last_info = dict(reset_info) if isinstance(reset_info, dict) else {}

        while not done and ep_len < args.max_steps:
            obs_n = (obs - obs_mean) / obs_std
            obs_t = torch.from_numpy(obs_n).float().unsqueeze(0).to(device)

            with torch.no_grad():
                act = model(obs_t)[0].cpu().numpy().astype(np.float32)

            obs, rew, done, info = step_env(env, act)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)

            ep_ret += float(rew)
            ep_len += 1

            if isinstance(info, dict):
                last_info = info

            if args.visualize and args.sleep > 0:
                time.sleep(args.sleep)

        success = bool(
            last_info.get("success", False)
            or last_info.get("is_success", False)
        )
        num_success += int(success)

        row = {
            "episode": ep,
            "success": success,
            "return": ep_ret,
            "length": ep_len,
        }

        for k in ["d", "hold", "goal_dist", "z_dist", "final_dist", "hold_count", "obs_dim"]:
            if k in last_info:
                try:
                    row[k] = float(last_info[k])
                except Exception:
                    row[k] = last_info[k]

        episode_rows.append(row)
        print(
            f"[ep {ep:03d}] success={success} "
            f"return={ep_ret:.4f} len={ep_len} info={last_info}"
        )

    env.close()

    success_rate = num_success / max(1, args.episodes)
    mean_return = float(np.mean([r["return"] for r in episode_rows]))
    mean_len = float(np.mean([r["length"] for r in episode_rows]))

    print()
    print(f"success_rate = {success_rate:.3f}")
    print(f"mean_return  = {mean_return:.4f}")
    print(f"mean_len     = {mean_len:.2f}")


    # print("------------")
    # for i in range(50):
    #     print(row["d"])
    # print("------------")


    if args.out_json is not None:
        out = {
            "task": args.task,
            "variation": int(args.variation),
            "bc_ckpt": args.bc_ckpt,
            "split_idx": int(split_idx),
            "episodes": int(args.episodes),
            "success_rate": success_rate,
            "mean_return": mean_return,
            "mean_len": mean_len,
            "rows": episode_rows,
        }
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()