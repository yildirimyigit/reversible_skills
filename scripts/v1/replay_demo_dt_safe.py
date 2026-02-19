#!/usr/bin/env python3
import argparse
import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

from snapshot_utils import load_demo_npz, restore_keyframe, get_observation_strict

try:
    from pyrep.backend import sim
except Exception:
    sim = None

def get_sim_dt():
    if sim is None:
        return None
    try:
        return float(sim.simGetFloatParameter(sim.sim_floatparam_simulation_time_step))
    except Exception:
        return None

def set_sim_dt(dt):
    if sim is None:
        return False
    try:
        sim.simSetFloatParameter(sim.sim_floatparam_simulation_time_step, float(dt))
        return True
    except Exception:
        return False

def get_sim_time():
    if sim is None:
        return None
    try:
        return float(sim.simGetSimulationTime())
    except Exception:
        return None

def task_success(task):
    try:
        s = task._task.success()
        if isinstance(s, (tuple, list)):
            return bool(s[0])
        return bool(s)
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    data = load_demo_npz(args.demo_npz)
    actions = np.asarray(data["action"], dtype=np.float32)
    q = np.asarray(data["joint_positions"], dtype=np.float32)
    n_arm = q.shape[1]

    demo_dt = None
    if "sim_dt" in data.files:
        demo_dt = float(data["sim_dt"].item())
    print("[demo] sim_dt from file:", demo_dt)

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.task_low_dim_state = True

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=args.headless,
    )
    env.launch()

    try:
        task_cls = getattr(rlbench_tasks, args.task)
        task = env.get_task(task_cls)
        task.set_variation(args.variation)

        # Force sim dt BEFORE restoring snapshot / stepping
        dt_before = get_sim_dt()
        print("[sim] dt before set:", dt_before)
        if demo_dt is not None and np.isfinite(demo_dt):
            ok = set_sim_dt(demo_dt)
            print("[sim] set dt ->", ok, "dt now:", get_sim_dt())

        # Restore exact initial state from your recording
        restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)

        # Measure effective dt of one task.step
        zero = np.zeros((n_arm + 1,), dtype=np.float32)
        t0 = get_sim_time()
        task.step(zero)
        t1 = get_sim_time()
        if t0 is not None and t1 is not None:
            print("[sim] measured dt per task.step:", t1 - t0)

        # Restore again (the measurement step advanced time)
        restore_keyframe(env, task, data, kf_index=0, settle_steps=0, snap_offset=0)

        T = min(actions.shape[0], q.shape[0] - 1)
        if args.max_steps is not None:
            T = min(T, int(args.max_steps))

        # Plain open-loop replay: exactly the recorded actions
        for i in range(T):
            ret = task.step(actions[i])
            done = bool(ret[2]) if isinstance(ret, (tuple, list)) and len(ret) >= 3 else False
            if done or task_success(task):
                print(f"[replay] SUCCESS at step {i+1}/{T}")
                return

        print(f"[replay] FAILED (no success in {T} steps)")
    finally:
        env.shutdown()

if __name__ == "__main__":
    main()
