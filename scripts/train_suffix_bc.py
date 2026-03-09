#!/usr/bin/env python3
"""
train_suffix_bc.py

PLAN-A Step 5:
Train a reverse suffix Behavior Cloning (BC) policy from recorded demos.

Design:
- Use only the irreversible suffix determined by consensus split
- Reverse-time samples: state at t, target action moves toward t-1
- Goal is the boundary snapshot/state at split index
- Observation is built ONLY through policy_io_suffix.py so BC and RL share one contract

Assumptions:
- prep npz contains joint positions under one of:
    joint_positions / qpos / arm_qpos
- optional gripper state under one of:
    gripper_open / gripper / gripper_state
- optional compact state under one of:
    z / z_traj / compact_state
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from policy_io_suffix import (
    DEFAULT_SUFFIX_OBS_SPEC,
    build_suffix_policy_obs,
    build_reverse_action_target,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_SENTINEL = object()


def pick_first(d: Dict[str, Any], keys: List[str], default=_SENTINEL):
    for k in keys:
        if k in d:
            return d[k]
    if default is not _SENTINEL:
        return default
    raise KeyError(f"None of these keys found: {keys}")


def load_npz_dict(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_consensus_split(consensus_json: str, task: str, variation: int) -> int:
    """
    Supports several possible schemas.
    Prefers recommended_split_idx, then consensus_snapped, then consensus_median.
    """
    with open(consensus_json, "r") as f:
        obj = json.load(f)

    var_key = f"var{variation:02d}"
    flat_key = f"{task}_{var_key}"
    group_key = f"{task}__{var_key}"

    def extract_split(v):
        if not isinstance(v, dict):
            return None
        for k in ["recommended_split_idx", "consensus_snapped", "consensus_median"]:
            if v.get(k) is not None:
                return int(round(v[k]))
        return None

    groups = obj.get("groups")
    if isinstance(groups, dict):
        if group_key in groups:
            s = extract_split(groups[group_key])
            if s is not None:
                return s

        for _, item in groups.items():
            if (
                isinstance(item, dict)
                and item.get("task") == task
                and int(item.get("variation", -1)) == variation
            ):
                s = extract_split(item)
                if s is not None:
                    return s

    if task in obj and isinstance(obj[task], dict) and var_key in obj[task]:
        s = extract_split(obj[task][var_key])
        if s is not None:
            return s

    if flat_key in obj and isinstance(obj[flat_key], dict):
        s = extract_split(obj[flat_key])
        if s is not None:
            return s

    if isinstance(obj, list):
        for item in obj:
            if (
                isinstance(item, dict)
                and item.get("task") == task
                and int(item.get("variation", -1)) == variation
            ):
                s = extract_split(item)
                if s is not None:
                    return s

    raise ValueError(
        f"Could not find consensus split for task={task}, variation={variation} "
        f"in {consensus_json}"
    )


def as_2d(arr: np.ndarray, T: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] == T:
            return arr.reshape(T, 1)
        return arr.reshape(1, -1).repeat(T, axis=0)
    return arr.astype(np.float32)


def load_demo_arrays(path: str) -> Dict[str, np.ndarray]:
    d = load_npz_dict(path)

    q = pick_first(d, ["joint_positions", "qpos", "arm_qpos"])
    q = np.asarray(q, dtype=np.float32)
    if q.ndim != 2:
        raise ValueError(f"{path}: q array must be [T, D], got shape {q.shape}")

    T = q.shape[0]

    g = pick_first(
        d,
        ["gripper_open", "gripper", "gripper_state"],
        default=np.zeros((T, 1), dtype=np.float32),
    )
    g = as_2d(g, T)

    z = pick_first(d, ["z", "z_traj", "compact_state"], default=None)
    if z is not None:
        z = np.asarray(z, dtype=np.float32)
        if z.ndim != 2 or z.shape[0] != T:
            raise ValueError(f"{path}: z must be [T, Dz], got shape {z.shape}")

    return {"q": q, "g": g, "z": z}


def build_reverse_suffix_dataset(
    prep_paths: List[str],
    split_idx: int,
    arm_k: float,
    include_gripper_action: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    meta_rows = []

    for path in prep_paths:
        demo = load_demo_arrays(path)
        q, g, z = demo["q"], demo["g"], demo["z"]
        T = q.shape[0]

        if split_idx <= 0 or split_idx >= T - 1:
            print(f"[warn] skipping {path}: split_idx={split_idx} invalid for T={T}")
            continue

        count_here = 0
        for t in range(T - 1, split_idx, -1):
            obs = build_suffix_policy_obs(
                q_t=q[t],
                g_t=g[t],
                q_goal=q[split_idx],
                g_goal=g[split_idx],
                z_t=None if z is None else z[t],
                z_goal=None if z is None else z[split_idx],
                spec=DEFAULT_SUFFIX_OBS_SPEC,
            )

            act = build_reverse_action_target(
                q_now=q[t],
                q_prev=q[t - 1],
                g_prev=g[t - 1],
                arm_k=arm_k,
                include_gripper_action=include_gripper_action,
            )

            obs_list.append(obs.astype(np.float32))
            act_list.append(act.astype(np.float32))
            count_here += 1

        meta_rows.append(
            {
                "prep_npz": path,
                "num_samples": count_here,
                "T": int(T),
            }
        )

    if not obs_list:
        raise RuntimeError("No BC samples created. Check split index and prep files.")

    X = np.stack(obs_list, axis=0).astype(np.float32)
    Y = np.stack(act_list, axis=0).astype(np.float32)

    info = {
        "num_demos": len(prep_paths),
        "num_samples": int(len(obs_list)),
        "obs_dim": int(X.shape[1]),
        "act_dim": int(Y.shape[1]),
        "split_idx": int(split_idx),
        "obs_spec": {
            "use_q": DEFAULT_SUFFIX_OBS_SPEC.use_q,
            "use_g": DEFAULT_SUFFIX_OBS_SPEC.use_g,
            "use_goal_q": DEFAULT_SUFFIX_OBS_SPEC.use_goal_q,
            "use_goal_g": DEFAULT_SUFFIX_OBS_SPEC.use_goal_g,
            "use_z": DEFAULT_SUFFIX_OBS_SPEC.use_z,
            "use_goal_z": DEFAULT_SUFFIX_OBS_SPEC.use_goal_z,
        },
        "include_gripper_action": bool(include_gripper_action),
        "per_demo": meta_rows,
    }
    return X, Y, info


class NpDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for obs, act in loader:
            obs = obs.to(device)
            act = act.to(device)
            pred = model(obs)
            loss = F.smooth_l1_loss(pred, act)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--variation", type=int, required=True)
    ap.add_argument("--prep_glob", type=str, required=True)
    ap.add_argument("--consensus_json", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--arm_k", type=float, default=4.0)
    ap.add_argument("--include_gripper_action", action="store_true")

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    prep_paths = sorted(glob.glob(args.prep_glob))
    if not prep_paths:
        raise FileNotFoundError(f"No files matched: {args.prep_glob}")

    split_idx = load_consensus_split(args.consensus_json, args.task, args.variation)
    print(f"[info] using consensus split: {split_idx}")

    X, Y, info = build_reverse_suffix_dataset(
        prep_paths=prep_paths,
        split_idx=split_idx,
        arm_k=args.arm_k,
        include_gripper_action=args.include_gripper_action,
    )

    print(f"[info] BC obs_dim = {X.shape[1]}")
    print(f"[info] BC act_dim = {Y.shape[1]}")
    print(f"[info] BC samples = {X.shape[0]}")

    obs_mean = X.mean(axis=0)
    obs_std = X.std(axis=0) + 1e-6
    Xn = (X - obs_mean) / obs_std

    np.savez_compressed(
        os.path.join(args.outdir, "bc_dataset.npz"),
        obs=Xn,
        act=Y,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )

    with open(os.path.join(args.outdir, "bc_dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    dataset = NpDataset(Xn, Y)

    n_total = len(dataset)
    n_val = max(1, int(round(args.val_ratio * n_total)))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCActor(obs_dim=X.shape[1], act_dim=Y.shape[1]).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for obs, act in train_loader:
            obs = obs.to(device)
            act = act.to(device)

            pred = model(obs)
            loss = F.smooth_l1_loss(pred, act)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(f"[epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f}")

        ckpt_last = {
            "model_state_dict": model.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "obs_dim": X.shape[1],
            "act_dim": Y.shape[1],
            "task": args.task,
            "variation": args.variation,
            "split_idx": split_idx,
            "include_gripper_action": args.include_gripper_action,
            "obs_spec": info["obs_spec"],
        }
        torch.save(ckpt_last, os.path.join(args.outdir, "bc_last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt_last, os.path.join(args.outdir, "bc_best.pt"))

    with open(os.path.join(args.outdir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"[done] best val loss = {best_val:.6f}")
    print(f"[done] saved to {args.outdir}")


if __name__ == "__main__":
    main()