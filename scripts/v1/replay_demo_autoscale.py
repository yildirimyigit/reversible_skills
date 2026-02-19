#!/usr/bin/env python3
import argparse
import numpy as np

from rlbench.environment import Environment
from rlbench import tasks as rlbench_tasks
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

# sim-time access (multiple fallbacks)
try:
    from pyrep.backend import sim as sim_backend
except Exception:
    sim_backend = None

def sim_time() -> float:
    if sim_backend is None:
        return float("nan")
    for name in ("simGetSimulationTime", "getSimulationTime"):
        if hasattr(sim_backend, name):
            try:
                return float(getattr(sim_backend, name)())
            except Exception:
                pass
    return float("nan")

def task_success(task) -> bool:
    try:
        s = task._task.success()
        if isinstance(s, (tuple, list)):
            return bool(s[0])
        return bool(s)
    except Exception:
        return False

def infer_demo_dt(data: np.lib.npyio.NpzFile) -> float:
    # Prefer explicit if present+finite, otherwise infer from dq ~= v*dt
    if "control_dt" in data.files:
        try:
            v = float(np.asarray(data["control_dt"]).reshape(-1)[0])
            if np.isfinite(v) and v > 0:
                return v
        except Exception:
            pass

    q = np.asarray(data["joint_positions"], dtype=np.float64)
    vcmd = np.asarray(data["action_arm"], dtype=np.float64)
    dq = q[1:] - q[:-1]
    v0 = vcmd[:-1]
    num = (dq * v0).sum(axis=1)
    den = (v0 * v0).sum(axis=1) + 1e-12
    dt = num / den
    mask = (den > 1e-4) & np.isfinite(dt) & (dt > 0) & (dt < 1.0)
    if np.any(mask):
        return float(np.median(dt[mask]))
    raise RuntimeError("Could not infer demo dt (insufficient velocity signal).")

def measure_runtime_dt(task, n_arm: int, g_hold: float, n: int = 5) -> float:
    # measure sim-time advanced by task.step (NOT wall time)
    zero = np.zeros((n_arm,), dtype=np.float32)
    a = np.concatenate([zero, np.array([g_hold], dtype=np.float32)], axis=0)

    dts = []
    # warmup
    t0 = sim_time()
    _ = task.step(a)
    t1 = sim_time()
    if np.isfinite(t0) and np.isfinite(t1):
        dts.append(t1 - t0)

    for _ in range(int(n)):
        t0 = sim_time()
        _ = task.step(a)
        t1 = sim_time()
        if np.isfinite(t0) and np.isfinite(t1):
            dts.append(t1 - t0)

    dts = np.asarray(dts, dtype=np.float64)
    dts = dts[np.isfinite(dts) & (dts > 0) & (dts < 5.0)]
    if dts.size == 0:
        raise RuntimeError("Could not measure runtime dt from simulation time.")
    return float(np.median(dts))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    data = np.load(args.demo_npz, allow_pickle=True)
    actions = np.asarray(data["action"], dtype=np.float32)
    n_arm = actions.shape[1] - 1

    demo_dt = infer_demo_dt(data)
    print(f"[demo] inferred control_dt ~ {demo_dt:.6f} s")

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

        # reset once for dt measurement, then reset again for replay
        _ = task.reset()
        g_hold = float(actions[0, -1])
        runtime_dt = measure_runtime_dt(task, n_arm=n_arm, g_hold=g_hold, n=5)
        print(f"[sim] measured control_dt per task.step ~ {runtime_dt:.6f} s")

        vel_scale = demo_dt / runtime_dt
        print(f"[fix] velocity scale = demo_dt/runtime_dt = {vel_scale:.6f}")

        _ = task.reset()

        T = actions.shape[0]
        max_steps = int(T) if args.max_steps is None else int(min(T, args.max_steps))

        for i in range(max_steps):
            a = actions[i].copy()
            a[:n_arm] *= float(vel_scale)
            obs, reward, done = task.step(a)

            if done or task_success(task):
                print(f"[replay] SUCCESS at step {i+1}/{max_steps}")
                return

        print(f"[replay] FAILED (no success in {max_steps} steps)")

    finally:
        env.shutdown()

if __name__ == "__main__":
    main()
