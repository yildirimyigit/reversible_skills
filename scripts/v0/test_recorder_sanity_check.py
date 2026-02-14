#!/usr/bin/env python3
import argparse
import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _is_bytes(x):
    return isinstance(x, (bytes, bytearray))


def check_snapshot_arrays(d):
    _assert("snapshot_captured" in d.files, "missing snapshot_captured")
    _assert(int(d["snapshot_captured"][0]) == 1, "snapshot_captured != 1")

    _assert("snapshot_storage" in d.files, "missing snapshot_storage")
    storage = str(d["snapshot_storage"][0])
    print("snapshot_storage =", storage)

    _assert("snapshot_model_names" in d.files, "missing snapshot_model_names")
    _assert("snapshot_keyframe_trees" in d.files, "missing snapshot_keyframe_trees")
    _assert("snapshot_post_trees" in d.files, "missing snapshot_post_trees")

    names = d["snapshot_model_names"]
    kf = d["snapshot_keyframe_trees"]
    post = d["snapshot_post_trees"]

    _assert(kf.dtype == object, "snapshot_keyframe_trees must be dtype object")
    _assert(post.dtype == object, "snapshot_post_trees must be dtype object")

    M = int(names.shape[0])
    _assert(kf.ndim == 2 and kf.shape[1] == M, f"snapshot_keyframe_trees must be (K,M), got {kf.shape}")
    _assert(post.ndim == 1 and post.shape[0] == M, f"snapshot_post_trees must be (M,), got {post.shape}")

    K = int(kf.shape[0])
    print("M (models)   =", M)
    print("K (keyframes)=", K)
    print("snapshot_keyframe_trees shape:", kf.shape, "dtype:", kf.dtype)
    print("snapshot_post_trees shape:", post.shape, "dtype:", post.dtype)

    # Check sizes
    kf_sizes = []
    for r in range(K):
        for c in range(M):
            b = kf[r, c]
            _assert(_is_bytes(b), f"snapshot_keyframe_trees[{r},{c}] not bytes: {type(b)}")
            kf_sizes.append(len(b))
            _assert(len(b) > 8, f"snapshot_keyframe_trees[{r},{c}] too small: {len(b)} bytes")

    post_sizes = []
    for c in range(M):
        b = post[c]
        _assert(_is_bytes(b), f"snapshot_post_trees[{c}] not bytes: {type(b)}")
        post_sizes.append(len(b))
        _assert(len(b) > 8, f"post snapshot too small at model {c}: {len(b)} bytes")

    print(f"keyframe snapshot bytes: min={min(kf_sizes)} median={int(np.median(kf_sizes))} max={max(kf_sizes)}")
    print(f"post snapshot bytes:     min={min(post_sizes)} median={int(np.median(post_sizes))} max={max(post_sizes)}")

    return names, kf, post


def restore_snapshot(task_name, variation, headless, trees_row, settle_steps=30):
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.front_camera.set_all(True)

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=headless,
    )
    env.launch()
    try:
        task_cls = getattr(rlbench_tasks, task_name)
        task = env.get_task(task_cls)
        task.set_variation(variation)

        # Put sim into some known base (fresh reset)
        task.reset()
        env._pyrep.step()

        # Apply configuration trees (order must match your recorder order)
        for b in trees_row:
            env._pyrep.set_configuration_tree(b)

        for _ in range(int(settle_steps)):
            env._pyrep.step()

        print("[restore_test] restore applied and stepped.")
    finally:
        env.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--restore_test", action="store_true")
    ap.add_argument("--task", type=str, default="CloseDrawer")
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--keyframe_index", type=str, default="0",
                    help="integer keyframe row index, or 'post'")
    ap.add_argument("--settle_steps", type=int, default=30)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    print("NPZ keys:", d.files)

    # quick status fields if present
    if "snapshot_failed" in d.files:
        print("snapshot_failed   =", int(d["snapshot_failed"][0]))

    names, kf, post = check_snapshot_arrays(d)

    # Choose row to restore
    if args.keyframe_index.lower() == "post":
        row = post.tolist()
        print("[restore_test] using POST snapshot row")
    else:
        ki = int(args.keyframe_index)
        _assert(0 <= ki < kf.shape[0], f"keyframe_index out of range: {ki}")
        row = [kf[ki, c] for c in range(kf.shape[1])]
        print(f"[restore_test] using KEYFRAME row {ki}")

    if args.restore_test:
        restore_snapshot(
            task_name=args.task,
            variation=args.variation,
            headless=args.headless,
            trees_row=row,
            settle_steps=args.settle_steps,
        )


if __name__ == "__main__":
    main()
