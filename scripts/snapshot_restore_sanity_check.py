#!/usr/bin/env python3
import argparse
import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete


def get_top_level_models(pyrep):
    roots = pyrep.get_objects_in_tree(root_object=None, first_generation_only=True)
    models = [o for o in roots if o.is_model()]
    models.sort(key=lambda o: o.get_name())
    return models


def _restore_with_models_by_name(pyrep, model_names, row_bytes):
    """
    Preferred restore path: name-based mapping + model.set_configuration_tree(bytes).
    Falls back to pyrep.set_configuration_tree(bytes) if needed.
    """
    models = get_top_level_models(pyrep)
    name2m = {m.get_name(): m for m in models}

    missing = [n for n in model_names if n not in name2m]
    if missing:
        # Not always fatal, but usually indicates mismatch between saved snapshot set and current scene
        raise RuntimeError(
            "Missing models in current scene (first 20): "
            + ", ".join(missing[:20])
        )

    for name, b in zip(model_names, row_bytes):
        if not isinstance(b, (bytes, bytearray)):
            raise TypeError(f"Snapshot for model {name} is not bytes: {type(b)}")
        if len(b) <= 1:
            raise RuntimeError(f"Snapshot for model {name} too small: len={len(b)}")

        m = name2m[name]
        # Try model-level restore first
        try:
            m.set_configuration_tree(b)
        except Exception:
            # Fallback: some builds prefer applying via pyrep
            pyrep.set_configuration_tree(b)


def restore_row(pyrep, model_names, row_bytes, settle_steps=10):
    _restore_with_models_by_name(pyrep, model_names, row_bytes)
    for _ in range(int(settle_steps)):
        pyrep.step()


def l2(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n == 0:
        return float("nan")
    return float(np.linalg.norm(x[:n] - y[:n]))


def get_observation_any(task):
    """
    RLBench API differences helper.
    """
    if hasattr(task, "get_observation"):
        return task.get_observation()

    scene = getattr(task, "_scene", None)
    if scene is not None:
        if hasattr(scene, "get_observation"):
            return scene.get_observation()
        if hasattr(scene, "_get_observation"):
            return scene._get_observation()

    raise RuntimeError("Could not obtain observation (no get_observation / scene.get_observation found).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True, help="Path to recorded .npz demo")
    ap.add_argument("--task", required=True, help="RLBench task class name, e.g. CloseDrawer")
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--kf_index", type=int, default=0, help="Index into keyframe_indices")
    ap.add_argument("--settle_steps", type=int, default=20)
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.set_defaults(headless=True)
    args = ap.parse_args()

    data = np.load(args.demo_npz, allow_pickle=True)

    # Required snapshot fields
    for k in ("snapshot_model_names", "snapshot_keyframe_trees", "keyframe_indices"):
        if k not in data.files:
            raise RuntimeError(f"Missing '{k}' in {args.demo_npz}. Did you record with snapshots?")

    model_names = data["snapshot_model_names"].tolist()
    keyframe_ts = data["keyframe_indices"].astype(int).tolist()
    kf_trees = data["snapshot_keyframe_trees"]  # (K, M) object bytes

    K = len(keyframe_ts)
    if K == 0:
        raise RuntimeError("No keyframes in file (keyframe_indices empty).")
    if args.kf_index < 0 or args.kf_index >= K:
        raise ValueError(f"--kf_index out of range: {args.kf_index} (K={K})")

    t = int(keyframe_ts[args.kf_index])
    row = kf_trees[args.kf_index, :].tolist()

    # Minimal obs for comparing restored state
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = True
    obs_config.task_low_dim_state = True

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=args.headless,
    )
    env.launch()

    try:
        if not hasattr(rlbench_tasks, args.task):
            raise ValueError(f"Unknown RLBench task '{args.task}'.")
        task_cls = getattr(rlbench_tasks, args.task)
        task = env.get_task(task_cls)
        task.set_variation(args.variation)

        # IMPORTANT: use env._pyrep (your build does not expose scene._pyrep)
        pyrep = getattr(env, "_pyrep", None)
        if pyrep is None:
            raise RuntimeError("env._pyrep not found. Cannot restore snapshots.")

        # Make sure a valid scene is loaded before restore
        _ = task.reset()
        for _ in range(5):
            pyrep.step()

        # Restore keyframe snapshot
        restore_row(pyrep, model_names, row, settle_steps=args.settle_steps)

        # Read observation after restore
        obs = get_observation_any(task)

        # Recorded references at timestep t
        q_rec = data["joint_positions"][t]
        g_rec = float(np.asarray(data["gripper_open"][t]).reshape(-1)[0])

        pose_rec = data["gripper_pose"][t] if "gripper_pose" in data.files else None
        low_rec = data["task_low_dim_state"][t] if "task_low_dim_state" in data.files else None

        # Current after restore
        q_now = obs.joint_positions
        g_now = float(obs.gripper_open)
        pose_now = getattr(obs, "gripper_pose", None)
        low_now = getattr(obs, "task_low_dim_state", None)

        print(f"Restored keyframe index={args.kf_index}  timestep t={t}")
        print(f"  joint_positions L2 error: {l2(q_now, q_rec):.6f}")
        # Find best matching timestep (detect off-by-one / off-by-few alignment)
        q_all = data["joint_positions"].astype(np.float64)  # (T,7)
        q_now = np.asarray(obs.joint_positions, dtype=np.float64).reshape(1, -1)
        errs = np.linalg.norm(q_all - q_now, axis=1)  # (T,)
        best_t = int(np.argmin(errs))
        print(f"\nBest-matching recorded timestep for restored joints: best_t={best_t} (requested t={t}), L2={errs[best_t]:.6f}")
        print(f"  neighbor L2: t-1={errs[max(best_t-1,0)]:.6f}, t={errs[best_t]:.6f}, t+1={errs[min(best_t+1,len(errs)-1)]:.6f}")
        print(f"  gripper_open abs error:   {abs(g_now - g_rec):.6f}")

        if pose_rec is not None and pose_now is not None:
            print(f"  gripper_pose L2 error:    {l2(pose_now, pose_rec):.6f}")
        else:
            print("  gripper_pose: (missing in record or in observation)")

        if low_rec is not None and low_now is not None:
            print(f"  task_low_dim_state L2 error: {l2(low_now, low_rec):.6f}")
        else:
            print("  task_low_dim_state: (missing in record or in observation)")

        # Optional: test restoring final snapshot too
        if "snapshot_post_trees" in data.files:
            print("\nTesting restore of final (post) snapshot...")
            post_row = data["snapshot_post_trees"].tolist()
            restore_row(pyrep, model_names, post_row, settle_steps=args.settle_steps)

            obs2 = get_observation_any(task)
            t2 = int(data["joint_positions"].shape[0] - 1)

            q_rec2 = data["joint_positions"][t2]
            g_rec2 = float(np.asarray(data["gripper_open"][t2]).reshape(-1)[0])

            print(f"  final joint_positions L2 error: {l2(obs2.joint_positions, q_rec2):.6f}")
            print(f"  final gripper_open abs error:   {abs(float(obs2.gripper_open) - g_rec2):.6f}")

    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
