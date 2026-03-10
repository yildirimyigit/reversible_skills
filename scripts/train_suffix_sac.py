#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from rl_wrapper_suffix import ReverseSuffixEnv, DemoSnapshots, SplitSpec
from rollback_triage import get_keyframe_rows
from policy_io_suffix import DEFAULT_SUFFIX_OBS_SPEC


# -----------------------------
# BC model helpers
# -----------------------------

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
        x = self.backbone(obs)
        return torch.tanh(self.head(x))


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


def warmstart_sac_actor_from_bc(model: SAC, bc_ckpt_path: str, device: torch.device):
    bc_model, _obs_mean, _obs_std, ckpt = load_bc_checkpoint(bc_ckpt_path, device)

    actor = model.policy.actor

    env_obs_dim = int(np.prod(model.observation_space.shape))
    env_act_dim = int(np.prod(model.action_space.shape))
    bc_obs_dim = int(ckpt["obs_dim"])
    bc_act_dim = int(ckpt["act_dim"])

    if env_obs_dim != bc_obs_dim:
        raise ValueError(
            f"Warm-start obs mismatch: env obs_dim={env_obs_dim}, BC obs_dim={bc_obs_dim}"
        )
    if env_act_dim != bc_act_dim:
        raise ValueError(
            f"Warm-start act mismatch: env act_dim={env_act_dim}, BC act_dim={bc_act_dim}"
        )

    if not hasattr(actor, "latent_pi") or not hasattr(actor, "mu"):
        raise RuntimeError("SB3 SAC actor structure not recognized: missing latent_pi or mu")

    bc_linears = [m for m in bc_model.backbone if isinstance(m, nn.Linear)]
    sac_linears = [m for m in actor.latent_pi if isinstance(m, nn.Linear)]

    if len(bc_linears) != len(sac_linears):
        raise RuntimeError(
            f"Cannot warm-start: BC has {len(bc_linears)} hidden Linear layers, "
            f"SAC actor has {len(sac_linears)}"
        )

    for src, dst in zip(bc_linears, sac_linears):
        if src.weight.shape != dst.weight.shape or src.bias.shape != dst.bias.shape:
            raise RuntimeError(
                f"Layer shape mismatch during warm-start: "
                f"BC {tuple(src.weight.shape)} vs SAC {tuple(dst.weight.shape)}"
            )
        dst.weight.data.copy_(src.weight.data)
        dst.bias.data.copy_(src.bias.data)

    if bc_model.head.weight.shape != actor.mu.weight.shape or bc_model.head.bias.shape != actor.mu.bias.shape:
        raise RuntimeError(
            f"Output layer mismatch during warm-start: "
            f"BC {tuple(bc_model.head.weight.shape)} vs SAC {tuple(actor.mu.weight.shape)}"
        )

    actor.mu.weight.data.copy_(bc_model.head.weight.data)
    actor.mu.bias.data.copy_(bc_model.head.bias.data)

    print(f"[info] Warm-started SAC actor mean path from BC checkpoint: {bc_ckpt_path}")


class ResidualBCWrapper(gym.Wrapper):
    """
    Executes action = clip( BC(obs_raw) + alpha * residual_action, -1, 1 )

    The wrapped SAC policy only outputs the residual_action.
    BC normalization is taken from the BC checkpoint, independently of VecNormalize.
    """
    def __init__(
        self,
        env: gym.Env,
        bc_model: nn.Module,
        bc_obs_mean: np.ndarray,
        bc_obs_std: np.ndarray,
        residual_alpha: float,
        bc_device: torch.device,
    ):
        super().__init__(env)
        self.bc_model = bc_model
        self.bc_obs_mean = np.asarray(bc_obs_mean, dtype=np.float32)
        self.bc_obs_std = np.asarray(bc_obs_std, dtype=np.float32)
        self.residual_alpha = float(residual_alpha)
        self.bc_device = bc_device

        self._last_raw_obs = None

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def _bc_action(self, raw_obs: np.ndarray) -> np.ndarray:
        obs_n = (raw_obs - self.bc_obs_mean) / self.bc_obs_std
        obs_t = torch.from_numpy(obs_n.astype(np.float32)).unsqueeze(0).to(self.bc_device)
        with torch.no_grad():
            act = self.bc_model(obs_t)[0].cpu().numpy().astype(np.float32)
        return act

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            self._last_raw_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            return obs, info
        self._last_raw_obs = np.asarray(out, dtype=np.float32).reshape(-1)
        return out

    def step(self, residual_action):
        if self._last_raw_obs is None:
            raise RuntimeError("ResidualBCWrapper.step() called before reset().")

        residual_action = np.asarray(residual_action, dtype=np.float32).reshape(-1)
        bc_action = self._bc_action(self._last_raw_obs)
        exec_action = np.clip(bc_action + self.residual_alpha * residual_action, -1.0, 1.0)

        out = self.env.step(exec_action)

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            self._last_raw_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            if isinstance(info, dict):
                info = dict(info)
                info["bc_action_norm"] = float(np.linalg.norm(bc_action))
                info["residual_action_norm"] = float(np.linalg.norm(residual_action))
                info["exec_action_norm"] = float(np.linalg.norm(exec_action))
            return obs, reward, terminated, truncated, info

        if len(out) == 4:
            obs, reward, done, info = out
            self._last_raw_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            if isinstance(info, dict):
                info = dict(info)
                info["bc_action_norm"] = float(np.linalg.norm(bc_action))
                info["residual_action_norm"] = float(np.linalg.norm(residual_action))
                info["exec_action_norm"] = float(np.linalg.norm(exec_action))
            return obs, reward, done, info

        raise RuntimeError(f"Unexpected step() output length: {len(out)}")


# -----------------------------
# Demo / split loaders
# -----------------------------

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
            "Choose a task/variation with an accepted split "
            "(e.g., CloseDrawer var00 or PutItemInDrawer var00)."
        )
    return int(rec)


# -----------------------------
# Main
# -----------------------------

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

    ap.add_argument("--bc_mode", choices=["none", "warmstart", "residual"], default="none")
    ap.add_argument("--bc_ckpt", type=str, default=None)
    ap.add_argument("--residual_alpha", type=float, default=0.25)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.bc_mode in ("warmstart", "residual") and not args.bc_ckpt:
        raise ValueError("--bc_ckpt is required when --bc_mode is warmstart or residual")

    demo_snaps = load_demo_snaps(args.prep_npz)
    split_idx = load_split_idx(args.consensus_json, args.task, args.variation)
    split_spec = SplitSpec(split_t=split_idx)

    # Preload BC only if needed
    bc_model = None
    bc_obs_mean = None
    bc_obs_std = None
    bc_ckpt = None
    bc_device = torch.device("cpu")

    if args.bc_mode == "residual":
        bc_model, bc_obs_mean, bc_obs_std, bc_ckpt = load_bc_checkpoint(args.bc_ckpt, bc_device)

    def make_base_env():
        return ReverseSuffixEnv(
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

    def make_env():
        env = make_base_env()

        if args.bc_mode == "residual":
            env_obs_dim = int(np.prod(env.observation_space.shape))
            env_act_dim = int(np.prod(env.action_space.shape))
            bc_obs_dim = int(bc_ckpt["obs_dim"])
            bc_act_dim = int(bc_ckpt["act_dim"])

            if env_obs_dim != bc_obs_dim:
                raise ValueError(
                    f"Residual mode obs mismatch: env obs_dim={env_obs_dim}, BC obs_dim={bc_obs_dim}"
                )
            if env_act_dim != bc_act_dim:
                raise ValueError(
                    f"Residual mode act mismatch: env act_dim={env_act_dim}, BC act_dim={bc_act_dim}"
                )

            env = ResidualBCWrapper(
                env=env,
                bc_model=bc_model,
                bc_obs_mean=bc_obs_mean,
                bc_obs_std=bc_obs_std,
                residual_alpha=args.residual_alpha,
                bc_device=bc_device,
            )

        return Monitor(env)

    # Probe once
    probe_env = make_base_env()
    print(f"[info] split_idx={split_idx}")
    print(f"[info] obs_dim={probe_env.observation_space.shape[0]}")
    print(f"[info] act_dim={probe_env.action_space.shape[0]}")
    print(f"[info] bc_mode={args.bc_mode}")
    if args.bc_mode == "residual":
        print(f"[info] residual_alpha={args.residual_alpha}")
        print(f"[info] bc_ckpt={args.bc_ckpt}")
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

    if args.bc_mode == "warmstart":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warmstart_sac_actor_from_bc(model, args.bc_ckpt, device)

    config = {
        "task": args.task,
        "variation": int(args.variation),
        "prep_npz": args.prep_npz,
        "zspec_json": args.zspec_json,
        "consensus_json": args.consensus_json,
        "split_idx": int(split_idx),
        "timesteps": int(args.timesteps),
        "max_steps": int(args.max_steps),
        "goal_tol": float(args.goal_tol),
        "goal_hold": int(args.goal_hold),
        "seed": int(args.seed),
        "bc_mode": args.bc_mode,
        "bc_ckpt": args.bc_ckpt,
        "residual_alpha": float(args.residual_alpha),
    }
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    model.learn(total_timesteps=int(args.timesteps))

    model.save(os.path.join(args.outdir, "sac_suffix.zip"))
    venv.save(os.path.join(args.outdir, "vecnormalize.pkl"))
    print("Saved model + VecNormalize to", args.outdir)


if __name__ == "__main__":
    main()