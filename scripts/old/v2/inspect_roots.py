#!/usr/bin/env python3
import argparse
from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--headless", action="store_true", default=False)
    args = ap.parse_args()

    oc = ObservationConfig()
    oc.set_all(False)
    oc.joint_positions = True

    env = Environment(MoveArmThenGripper(JointVelocity(), Discrete()), obs_config=oc, headless=args.headless)
    env.launch()
    try:
        task_cls = getattr(rlbench_tasks, args.task)
        task = env.get_task(task_cls)
        task.set_variation(args.variation)
        task.reset()

        pr = env._pyrep
        roots = pr.get_objects_in_tree(root_object=None, first_generation_only=True)

        print(f"Found {len(roots)} first-generation root objects:")
        for o in sorted(roots, key=lambda x: x.get_name()):
            name = o.get_name()
            try:
                is_model = bool(o.is_model())
            except Exception:
                is_model = False
            has_get = hasattr(o, "get_configuration_tree")
            has_set = hasattr(o, "set_configuration_tree")
            print(f"  {name:35s}  is_model={is_model}  get_tree={has_get}  set_tree={has_set}")
    finally:
        env.shutdown()

if __name__ == "__main__":
    main()