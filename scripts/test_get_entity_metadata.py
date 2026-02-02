import numpy as np
from pyrep.backend import sim

def _sim_const(sim_mod, *names, default=None):
    """Try multiple constant names across CoppeliaSim/PyRep variants."""
    for n in names:
        if hasattr(sim_mod, n):
            return getattr(sim_mod, n)
    return default

def list_scene_objects():
    """
    Returns a list of dicts: [{handle, name, type}, ...]
    Works inside RLBench because PyRep is embedded.
    """
    # These constant names vary a bit across versions/bindings:
    HANDLE_SCENE = _sim_const(sim, "handle_scene", "sim_handle_scene", default=-1)
    HANDLE_ALL   = _sim_const(sim, "handle_all", "sim_handle_all", default=-1)

    # sim.getObjectsInTree exists in the regular API; in PyRep it’s usually simGetObjectsInTree
    if hasattr(sim, "simGetObjectsInTree"):
        handles = sim.simGetObjectsInTree(HANDLE_SCENE, HANDLE_ALL, 0)
    elif hasattr(sim, "getObjectsInTree"):
        handles = sim.getObjectsInTree(HANDLE_SCENE, HANDLE_ALL, 0)
    else:
        raise RuntimeError("Could not find getObjectsInTree/simGetObjectsInTree in pyrep.backend.sim")

    out = []
    for h in handles:
        try:
            name = sim.simGetObjectName(h) if hasattr(sim, "simGetObjectName") else sim.getObjectName(h)
            otyp = sim.simGetObjectType(h) if hasattr(sim, "simGetObjectType") else sim.getObjectType(h)
            out.append({"handle": int(h), "name": name, "type": int(otyp)})
        except Exception:
            # Some handles can be invalid during transitions; ignore quietly
            pass
    return out

def mask_ids_to_names(mask_2d: np.ndarray):
    """
    RLBench masks are typically 2D arrays where each pixel stores an object handle.
    Returns: {id:int -> name:str}
    """
    ids = np.unique(mask_2d.astype(np.int64))
    ids = [int(i) for i in ids if i != 0]  # 0 is usually background
    mapping = {}
    for i in ids:
        try:
            mapping[i] = sim.simGetObjectName(i) if hasattr(sim, "simGetObjectName") else sim.getObjectName(i)
        except Exception:
            mapping[i] = "<unknown>"
    return mapping

# --- If (in some configs) you ever get a 3-channel "coded handles" image instead of 2D IDs:
def decode_coded_handles_rgb(rgb_img_uint8: np.ndarray) -> np.ndarray:
    """
    Convert an (H,W,3) uint8 coded-handles image into (H,W) integer handles.
    Handle encoding is effectively a 24-bit integer packed into RGB.
    """
    r = rgb_img_uint8[..., 0].astype(np.int32)
    g = rgb_img_uint8[..., 1].astype(np.int32)
    b = rgb_img_uint8[..., 2].astype(np.int32)
    return r + (g << 8) + (b << 16)


import re
from typing import Any, Dict, List, Tuple

def _flatten(x):
    if isinstance(x, (list, tuple)):
        out = []
        for y in x:
            out.extend(_flatten(y))
        return out
    return [x]

def _clean_name(n: str) -> str:
    # Strip common suffix noise
    n = re.sub(r'(_visibleElement|_visible|_visual)$', '', n)
    return n

def _is_robot_part(name: str) -> bool:
    return name.startswith("Panda_") or name.startswith("Sawyer_") or name.startswith("UR5_")

def _is_environment_junk(name: str) -> bool:
    junk = (
        "ResizableFloor", "Wall", "Floor", "Light", "Backdrop",
    )
    return any(j in name for j in junk)

def _is_region_like(name: str) -> bool:
    # Common RLBench naming patterns
    return any(k in name.lower() for k in ("success", "boundary", "spawn", "zone", "bin"))

def get_task_semantic_schema(task_env,
                             id2name: Dict[int, str],
                             include_static_context: bool = True) -> Dict[str, Any]:
    """
    Returns:
      {
        "entities": [str...],
        "regions": [str...],
        "groups": {"gripper": [handles...], "robot": [handles...]},
        "handles": {"entity_name": handle, ...}
      }
    """

    # 1) Grab the underlying RLBench task object (private API, but stable in practice)
    # TaskEnvironment typically has _task holding the task instance.
    t = getattr(task_env, "_task", None)
    if t is None:
        raise RuntimeError("Could not find underlying task object. Try inspecting task_env.__dict__ keys.")

    # 2) Find PyRep-like objects referenced by the task instance (only what the task cares about)
    # We detect by presence of get_handle() and get_name() methods.
    task_objs = []
    for k, v in vars(t).items():
        for obj in _flatten(v):
            if hasattr(obj, "get_handle") and hasattr(obj, "get_name"):
                task_objs.append(obj)

    task_handles = set()
    handle_to_clean = {}

    for obj in task_objs:
        try:
            h = int(obj.get_handle())
            nm = str(obj.get_name())
        except Exception:
            continue
        task_handles.add(h)
        handle_to_clean[h] = _clean_name(nm)

    # 3) Also add one high-level “gripper” entity (group robot visuals)
    # We can’t reliably get *all* robot link handles from the task alone, so we group from id2name.
    robot_handles = [h for h, nm in id2name.items() if _is_robot_part(nm)]
    # Heuristic: gripper is the finger visuals if present
    gripper_handles = [h for h, nm in id2name.items()
                       if ("finger" in nm.lower() or "gripper" in nm.lower()) and _is_robot_part(nm)]

    # 4) Build semantic lists
    entities = []
    regions = []
    handles_out = {}

    for h in sorted(task_handles):
        if h not in id2name:
            # If you want, you can still keep handle_to_clean[h]
            continue
        raw = id2name[h]
        name = _clean_name(raw)

        # Filter obvious junk
        if _is_environment_junk(raw):
            continue

        # Classify
        if _is_region_like(name):
            regions.append(name)
        else:
            entities.append(name)

        handles_out[name] = h

    # 5) Optionally include a bit of static context (table/workspace) as entities
    if include_static_context:
        for h, raw in id2name.items():
            nm = _clean_name(raw)
            if nm in entities or nm in regions:
                continue
            if _is_environment_junk(raw) or _is_robot_part(raw):
                continue
            if any(k in nm.lower() for k in ("table", "workspace", "drawer", "cabinet", "bin")):
                entities.append(nm)
                handles_out[nm] = h

    # 6) Add high-level grouped entities
    if gripper_handles:
        if "gripper" not in entities:
            entities.append("gripper")
    elif robot_handles:
        if "robot" not in entities:
            entities.append("robot")

    return {
        "entities": entities,
        "regions": regions,
        "groups": {
            "gripper": gripper_handles,
            "robot": robot_handles,
        },
        "handles": handles_out,
    }



from rlbench.environment import Environment
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import StackBlocks
from rlbench.observation_config import ObservationConfig

obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.front_camera.set_all(True)   # includes rgb/depth/mask if available

env = Environment(
    MoveArmThenGripper(JointVelocity(), Discrete()),
    obs_config=obs_config,
    headless=True,   # or False if you want to watch
)
env.launch()

task = env.get_task(StackBlocks)
descriptions, obs = task.reset()

# 1) dump scene object metadata
# objs = list_scene_objects()
# print(f"Scene objects: {len(objs)}")
# print(objs[:10])

# # 2) map segmentation IDs in the current observation back to names
# # (RLBench typically exposes obs.front_mask as (H,W))
# id2name = mask_ids_to_names(obs.front_mask)
# print("IDs in front_mask:", id2name)

# schema = get_task_semantic_schema(task, id2name)
# print("entities:", schema["entities"])
# print("regions:", schema["regions"])
# print("groups:", {k: len(v) for k, v in schema["groups"].items()})
print(descriptions)