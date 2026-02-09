#!/usr/bin/env python3
import os
import argparse
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from reversible_skills_wrapper import ReverseSkillEnv


class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.curriculum_steps = int(curriculum_steps)

    def _on_step(self) -> bool:
        level = min(1.0, float(self.num_timesteps) / float(max(1, self.curriculum_steps)))
        self.training_env.env_method("set_curriculum", level)
        return True


def seed_replay_buffer_with_reverse_demo_vec(model: SAC, venv, demo_npz_path: str, max_steps: int = 400):
    """
    Warm start 2 (vector-env compatible):
      - set curriculum to hardest (start from post)
      - execute reverse replay derived from demo action proxies
      - add transitions to replay buffer using the SAME venv (so obs match normalization)
    """
    demo = np.load(demo_npz_path, allow_pickle=True)
    if "action_arm" not in demo or "action_gripper" not in demo:
        print("[seed] demo missing action proxies; skip.")
        return

    action_arm = demo["action_arm"].astype(np.float32)
    action_gripper = demo["action_gripper"].astype(np.float32).reshape(-1)
    T = int(action_arm.shape[0])

    # Hardest curriculum
    venv.env_method("set_curriculum", 1.0)

    obs = venv.reset()
    n_envs = obs.shape[0]
    assert n_envs == 1, "Seeding helper assumes n_envs=1"

    steps = min(T, int(max_steps))
    for k in range(steps):
        t_src = T - 1 - k
        u_rev = -action_arm[t_src]
        g_rev = float(action_gripper[t_src])  # 1=open, 0=closed

        act = np.zeros((1, 8), dtype=np.float32)
        act[0, :7] = np.clip(u_rev, -1.0, 1.0)
        act[0, 7] = 1.0 if g_rev > 0.5 else -1.0  # sign -> wrapper converts to 1/0

        next_obs, rewards, dones, infos = venv.step(act)

        model.replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=act,
            reward=rewards,
            done=dones,
            infos=infos,
        )

        obs = next_obs
        if dones[0]:
            obs = venv.reset()

    print(f"[seed] seeded {steps} steps from reverse replay into replay buffer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.set_defaults(headless=True)

    ap.add_argument("--total_steps", type=int, default=300_000)
    ap.add_argument("--curriculum_steps", type=int, default=150_000)
    ap.add_argument("--max_episode_steps", type=int, default=150)
    ap.add_argument("--settle_steps", type=int, default=10)

    ap.add_argument("--logdir", type=str, default="runs/reverse_skill")
    ap.add_argument("--seed_from_demo", action="store_true")

    ap.add_argument("--open_map", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    def make_env():
        env = ReverseSkillEnv(
            demo_npz_path=args.demo_npz,
            task_name=args.task,
            variation=args.variation,
            headless=args.headless,
            max_episode_steps=args.max_episode_steps,
            settle_steps_after_restore=args.settle_steps,
            open_map_path=args.open_map,
        )
        return Monitor(env)

    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = SAC(
        policy="MlpPolicy",
        env=venv,
        verbose=1,
        tensorboard_log=args.logdir,
        learning_rate=3e-4,
        buffer_size=300_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=2_000,
    )

    if args.seed_from_demo:
        seed_replay_buffer_with_reverse_demo_vec(model, venv, args.demo_npz, max_steps=400)

    cb = CurriculumCallback(curriculum_steps=args.curriculum_steps, verbose=0)
    model.learn(total_timesteps=args.total_steps, callback=cb)

    model_path = os.path.join(args.logdir, "sac_reverse_skill.zip")
    norm_path = os.path.join(args.logdir, "vecnormalize.pkl")
    model.save(model_path)
    venv.save(norm_path)

    print("Saved:", model_path)
    print("Saved:", norm_path)


if __name__ == "__main__":
    main()
