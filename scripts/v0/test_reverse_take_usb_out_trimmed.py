import time
import numpy as np

from rlbench.environment import Environment
from rlbench.tasks import TakeUsbOutOfComputer
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete


def _set_if_exists(obj, attr, value=True):
    if hasattr(obj, attr):
        setattr(obj, attr, value)


def gripper_open_metric(obs) -> float:
    if hasattr(obs, "gripper_open_amount") and obs.gripper_open_amount is not None:
        return float(obs.gripper_open_amount)
    if hasattr(obs, "gripper_open") and obs.gripper_open is not None:
        return 1.0 if bool(obs.gripper_open) else 0.0
    if hasattr(obs, "gripper_joint_positions") and obs.gripper_joint_positions is not None:
        gj = np.array(obs.gripper_joint_positions, dtype=np.float32).ravel()
        return float(np.mean(gj)) if gj.size else 0.0
    return 0.0


def get_arm_joint_positions(obs) -> np.ndarray:
    if not hasattr(obs, "joint_positions") or obs.joint_positions is None:
        raise RuntimeError("Enable joint_positions in ObservationConfig.")
    return np.array(obs.joint_positions, dtype=np.float32).ravel()


def infer_discrete_gripper_mapping(task, n_arm: int):
    a0 = np.concatenate([np.zeros(n_arm, dtype=np.float32), [0.0]])
    a1 = np.concatenate([np.zeros(n_arm, dtype=np.float32), [1.0]])

    obs0, _, _ = task.step(a0)
    open0 = gripper_open_metric(obs0)

    obs1, _, _ = task.step(a1)
    open1 = gripper_open_metric(obs1)

    if open0 > open1:
        return 0.0, 1.0  # open_val, close_val
    return 1.0, 0.0


def vel_servo_to_joint_target(
    task,
    q_target,
    g_cmd,
    kp=4.0,
    vmax=1.5,
    tol=0.01,
    max_steps=40,
    sleep_s=0.0,
):
    q_target = np.array(q_target, dtype=np.float32).ravel()
    n_arm = q_target.size

    for _ in range(max_steps):
        obs, _, _ = task.step(np.concatenate([np.zeros(n_arm, dtype=np.float32), [g_cmd]]))
        q_cur = get_arm_joint_positions(obs)

        err = q_target - q_cur
        if float(np.linalg.norm(err)) < tol:
            return

        v = np.clip(kp * err, -vmax, vmax).astype(np.float32)
        task.step(np.concatenate([v, [g_cmd]]))

        if sleep_s > 0:
            time.sleep(sleep_s)


# -------------------------
# Parameters
# -------------------------
N_TRIM = 9                 # trim last N steps
GRIP_THRESH = 0.03         # same threshold you used for open/close
REVERSE_STRIDE = 3         # downsample reverse waypoints for speed (1 = no downsample)

# -------------------------
# Setup
# -------------------------
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.front_camera.set_all(True)

_set_if_exists(obs_config, "joint_velocities", True)
_set_if_exists(obs_config, "joint_positions", True)
_set_if_exists(obs_config, "gripper_open_amount", True)
_set_if_exists(obs_config, "gripper_open", True)
_set_if_exists(obs_config, "gripper_joint_positions", True)

env = Environment(
    MoveArmThenGripper(JointVelocity(), Discrete()),
    obs_config=obs_config,
    headless=False,
)
env.launch()

task = env.get_task(TakeUsbOutOfComputer)
task.set_variation(0)

input("Press Enter to run one live expert demo (this will execute in the sim)...")

# This will run the full expert once (may drop the USB at the end)
demos = task.get_demos(amount=1, live_demos=True)
demo = demos[0]
if N_TRIM >= len(demo):
    raise ValueError(f"N_TRIM={N_TRIM} is too large for demo length {len(demo)}")

demo_trim = demo[: len(demo) - N_TRIM]
print(f"Recorded demo length {len(demo)}. Using trimmed length {len(demo_trim)}.")

# Reset exactly to this demo’s initial state before replaying trimmed forward + reverse
task.reset_to_demo(demo)

# Infer discrete gripper mapping (this advances sim), then restore demo start again
n_arm = len(demo_trim[0].joint_velocities)
open_val, close_val = infer_discrete_gripper_mapping(task, n_arm=n_arm)
task.reset_to_demo(demo)

input("Press Enter to replay trimmed forward, then reverse back-to-back...")

# -------------------------
# Forward: replay trimmed actions (JointVelocity + Discrete)
# Also record waypoints for reverse
# -------------------------
waypoints = []
for ob in demo_trim:
    v = np.array(ob.joint_velocities, dtype=np.float32).ravel()
    g_cmd = open_val if gripper_open_metric(ob) > GRIP_THRESH else close_val
    q = get_arm_joint_positions(ob)

    waypoints.append((q, g_cmd))

    action = np.concatenate([v, [g_cmd]])
    _, _, terminate = task.step(action)
    if terminate:
        print("Episode terminated during trimmed forward replay.")
        break

print(f"Trimmed forward replay done. Waypoints stored: {len(waypoints)}")

# -------------------------
# Reverse: servo back through trimmed joint targets
# -------------------------
if len(waypoints) > 1:
    stride = max(1, int(REVERSE_STRIDE))
    indices = list(range(0, len(waypoints), stride))
    if indices[-1] != len(waypoints) - 1:
        indices.append(len(waypoints) - 1)

    for i in reversed(indices):  # include last index for robustness
        q_tgt, g_cmd = waypoints[i]
        vel_servo_to_joint_target(
            task,
            q_target=q_tgt,
            g_cmd=g_cmd,
            kp=4.0,
            vmax=1.5,
            tol=0.01,
            max_steps=30,
            sleep_s=0.0,
        )

print("Reverse playback finished.")
input("Press Enter to close...")
env.shutdown()
