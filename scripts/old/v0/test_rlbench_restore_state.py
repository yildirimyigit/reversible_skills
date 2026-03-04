import argparse

from rlbench.environment import Environment
from rlbench import tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete


def get_top_level_models(pyrep):
    # First generation objects under the "world"
    roots = pyrep.get_objects_in_tree(root_object=None, first_generation_only=True)
    # Keep only models (task props are typically models)
    models = [o for o in roots if o.is_model()]
    # Deterministic order: by name
    models.sort(key=lambda o: o.get_name())
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="StackBlocks")
    args = ap.parse_args()

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.front_camera.set_all(True)

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=False,
    )
    env.launch()

    task = env.get_task(getattr(tasks, args.task))
    task.set_variation(0)

    pyrep = env._pyrep
    scene = task._scene  # RLBench 1.2.0

    saved = {"trees": None}

    # Hook end-of-demo
    orig_get_demo = scene.get_demo

    def get_demo_with_snapshot(*a, **kw):
        demo = orig_get_demo(*a, **kw)  # executes planner demo
        pyrep.step()

        models = get_top_level_models(pyrep)
        print("[snapshot] top-level models captured:")
        for m in models:
            print("  -", m.get_name())

        # Save config tree for every top-level model
        saved["trees"] = [(m.get_name(), m.get_configuration_tree()) for m in models]
        print(f"[snapshot] Saved {len(saved['trees'])} model trees (this should include task props).")
        return demo

    scene.get_demo = get_demo_with_snapshot
    try:
        _ = task.get_demos(amount=1, live_demos=True)
    finally:
        scene.get_demo = orig_get_demo

    if saved["trees"] is None:
        raise RuntimeError("Snapshot did not run. (scene.get_demo hook did not trigger.)")

    # Move away from whatever RLBench did after the demo
    task.reset()
    pyrep.step()

    input("Reset done. Press Enter to restore s1 (all model trees)...")

    # Restore ALL model trees
    for _, tree in saved["trees"]:
        pyrep.set_configuration_tree(tree)

    for _ in range(20):
        pyrep.step()

    input("Restored. Press Enter to quit...")
    env.shutdown()


if __name__ == "__main__":
    main()
