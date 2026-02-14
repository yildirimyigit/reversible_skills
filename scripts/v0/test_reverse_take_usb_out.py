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
    # RLBench commonly exposes gripper_open_amount (0..1)
    if hasattr(obs, "gripper_open_amount") and obs.gripper_open_amount is not None:
        return float(obs.gripper_open_amount)
    # Some forks/versions expose a boolean.
    if hasattr(obs, "gripper_open") and obs.gripper_open is not None:
        return 1.0 if bool(obs.gripper_open) else 0.0
    # Fallback: infer from gripper joint positions if present.
    if hasattr(obs, "gripper_joint_positions") and obs.gripper_joint_positions is not None:
        gj = np.array(obs.gripper_joint_positions, dtype=np.float32).ravel()
        return float(np.mean(gj)) if gj.size else 0.0
    return 0.0


def get_arm_joint_positions(obs) -> np.ndarray:
    if not hasattr(obs, "joint_positions") or obs.joint_positions is None:
        raise RuntimeError(
            "Observation has no joint_positions. Enable low-dim fields in ObservationConfig."
        )
    return np.array(obs.joint_positions, dtype=np.float32).ravel()


def infer_discrete_gripper_mapping(task, n_arm: int):
    # Start from current state
    obs0, _, _ = task.step(np.concatenate([np.zeros(n_arm, dtype=np.float32), [0.0]]))
    open0 = gripper_open_metric(obs0)

    obs1, _, _ = task.step(np.concatenate([np.zeros(n_arm, dtype=np.float32), [1.0]]))
    open1 = gripper_open_metric(obs1)

    if open0 > open1:
        open_val, close_val = 0.0, 1.0
    else:
        open_val, close_val = 1.0, 0.0

    return open_val, close_val


def vel_servo_to_joint_target(task, q_target, g_cmd, kp=4.0, vmax=1.5, tol=0.01, max_steps=40, sleep_s=0.0):
    """
    Drive the arm toward q_target using joint-velocity actions:
        v = clip(kp*(q_target - q_current), -vmax, vmax)
    Apply g_cmd (discrete) at each step.
    """
    q_target = np.array(q_target, dtype=np.float32).ravel()
    n_arm = q_target.size

    for _ in range(max_steps):
        # We need the current observation; easiest is to do a "no-op" step to fetch it.
        # (RLBench only updates obs on step.)
        obs, _, _ = task.step(np.concatenate([np.zeros(n_arm, dtype=np.float32), [g_cmd]]))
        q_cur = get_arm_joint_positions(obs)

        err = q_target - q_cur
        if float(np.linalg.norm(err)) < tol:
            return

        v = np.clip(kp * err, -vmax, vmax).astype(np.float32)
        task.step(np.concatenate([v, [g_cmd]]))

        if sleep_s > 0:
            time.sleep(sleep_s)


# -----------------------------
# Main
# -----------------------------
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.front_camera.set_all(True)

# Make sure we actually get the low-dim fields we need for reverse playback.
_set_if_exists(obs_config, "joint_positions", True)
_set_if_exists(obs_config, "gripper_open_amount", True)
_set_if_exists(obs_config, "gripper_joint_positions", True)

env = Environment(
    MoveArmThenGripper(JointVelocity(), Discrete()),
    obs_config=obs_config,
    headless=False,
)
env.launch()

task = env.get_task(TakeUsbOutOfComputer)
task.set_variation(0)

# Ensure a clean initial episode state (safe even if get_demos resets internally).
task.reset()

# Infer how Discrete gripper is encoded (0/1 open/close) without guessing.
# We infer n_arm from a quick observation.
tmp_obs, _, _ = task.step(np.concatenate([np.zeros(7, dtype=np.float32), [0.0]]))
n_arm = get_arm_joint_positions(tmp_obs).size
open_val, close_val = infer_discrete_gripper_mapping(task, n_arm=n_arm)

# Reset again so calibration steps don't matter.
task.reset()

input("Press Enter to start the forward + reverse demo...")

print("Generating + running 1 live expert demo (forward)...")
demos = task.get_demos(amount=1, live_demos=True)  # leaves env in final state s
demo = demos[0]  # List[Observation]

# Extract (q, gripper_cmd) waypoints from the recorded forward demo.
waypoints = []
for ob in demo:
    q = get_arm_joint_positions(ob)
    amount = gripper_open_metric(ob)
    g_open = amount > 0.03
    g_cmd = open_val if g_open else close_val
    waypoints.append((q, g_cmd))

print(f"Forward demo finished. Recorded {len(waypoints)} waypoints.")
print("Now executing the demo in reverse starting from state s...")

# Reverse playback: downsample a bit so it doesn't take forever.
stride = 3  # increase to go faster (less faithful), decrease for more fidelity
indices = list(range(0, len(waypoints), stride))
if indices[-1] != len(waypoints) - 1:
    indices.append(len(waypoints) - 1)

for i in reversed(indices[:-1]):  # skip last because we're already at s
    q_tgt, g_cmd = waypoints[i]
    vel_servo_to_joint_target(
        task,
        q_target=q_tgt,
        g_cmd=g_cmd,
        kp=4.0,
        vmax=1.5,
        tol=0.01,
        max_steps=30,
        sleep_s=0.0,  # set e.g. 0.01 if you want it visually slower
    )

print("Reverse playback finished.")
input("Press Enter to close...")
env.shutdown()
