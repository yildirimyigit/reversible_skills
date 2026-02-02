#!/usr/bin/env python3
import argparse
import io
import os
from typing import List, Literal, Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

MEDIA_RES_MAP = {
    "unspecified": "MEDIA_RESOLUTION_UNSPECIFIED",
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
}


# ----------------------------
# Minimal structured output schema
# ----------------------------
class ReversibilityAssessment(BaseModel):
    reversibility_verdict: Literal["LIKELY", "UNLIKELY", "UNCERTAIN"]
    reversibility_confidence: float = Field(ge=0.0, le=1.0)
    predicate_consistency: Literal["CONSISTENT", "INCONSISTENT", "UNCERTAIN"]
    key_evidence: List[str] = Field(
        min_length=3,
        max_length=6,
        description="3–6 grounded observations; mention forward_t and/or reverse_step k."
    )


# ----------------------------
# Image helpers (cheaper)
# ----------------------------
def rgb_to_jpeg_part(rgb_u8: np.ndarray, quality: int = 75, max_side: Optional[int] = None) -> types.Part:
    """
    Convert RGB uint8 to JPEG Part.
    max_side: if set, downscale so max(H,W) <= max_side (reduces cost).
    """
    if rgb_u8.dtype != np.uint8:
        rgb_u8 = rgb_u8.astype(np.uint8)

    img = Image.fromarray(rgb_u8, mode="RGB")

    if max_side is not None:
        w, h = img.size
        m = max(w, h)
        if m > max_side:
            scale = max_side / float(m)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


# ----------------------------
# Keyframe selection (robust, endpoint-safe)
# ----------------------------
def force_include_endpoints(k_idx: np.ndarray, T: int, max_k: int) -> np.ndarray:
    k_set = set(map(int, k_idx.tolist())) if k_idx.size > 0 else set()
    k_set.add(0)
    k_set.add(T - 1)
    k_sorted = sorted(k_set)

    if len(k_sorted) > max_k:
        middle = k_sorted[1:-1]
        if len(middle) == 0:
            k_sorted = [0, T - 1]
        else:
            n_mid = max_k - 2
            take = np.linspace(0, len(middle) - 1, num=n_mid, dtype=int)
            k_sorted = [k_sorted[0]] + [middle[i] for i in take] + [k_sorted[-1]]

    return np.array(k_sorted, dtype=np.int32)


def event_keyframes_from_state(state01: np.ndarray, T: int, max_k: int, event_window: int) -> np.ndarray:
    state01 = state01.reshape(T).astype(np.int32)
    flips = np.where(state01[1:] != state01[:-1])[0] + 1

    idx = {0, T - 1}
    for f in flips:
        for d in range(-event_window, event_window + 1):
            j = f + d
            if 0 <= j < T:
                idx.add(int(j))

    idx = sorted(idx)
    if len(idx) > max_k:
        keep = [idx[0]]
        mid = idx[1:-1]
        take = np.linspace(0, len(mid) - 1, num=max_k - 2, dtype=int)
        keep += [mid[i] for i in take]
        keep += [idx[-1]]
        idx = keep

    return np.array(idx, dtype=np.int32)


def choose_keyframes(npz, T: int, max_k: int, event_window: int) -> np.ndarray:
    if "keyframe_indices" in npz:
        k_idx = npz["keyframe_indices"].astype(np.int32)
    else:
        if "action_gripper" in npz:
            g = npz["action_gripper"].reshape(T)
            state = (g > 0.5).astype(np.int32)
            k_idx = event_keyframes_from_state(state, T, max_k=max(max_k, 2), event_window=event_window)
        elif "gripper_open" in npz:
            thr = float(npz["gripper_threshold"][0]) if "gripper_threshold" in npz else 0.03
            go = npz["gripper_open"].reshape(T)
            state = (go > thr).astype(np.int32)
            k_idx = event_keyframes_from_state(state, T, max_k=max(max_k, 2), event_window=event_window)
        else:
            k_idx = np.linspace(0, T - 1, num=min(max_k, T), dtype=np.int32)

    return force_include_endpoints(k_idx, T, max_k=max(max_k, 2))


# ----------------------------
# Compact tables (token-cheap) with OPEN/CLOSED semantics
# ----------------------------
def _g01_to_state(g01: float) -> str:
    if not np.isfinite(g01):
        return "UNKNOWN"
    return "OPEN" if g01 >= 0.5 else "CLOSED"


def _fmt_vec(x: Optional[np.ndarray], decimals: int = 2) -> str:
    if x is None:
        return "None"
    x = np.asarray(x).reshape(-1)
    fmt = f"{{:+.{decimals}f}}"
    return "[" + ",".join(fmt.format(float(v)) for v in x.tolist()) + "]"


def compact_keyframe_table(npz, idxs: np.ndarray, T: int) -> str:
    """
    Minimal table aligned with the reverse replay rule, but with gripper state as OPEN/CLOSED.
    For each FORWARD index t:
      - reverse_step k = T-1-t
      - g_rev at that reverse_step equals action_gripper[t]
      - u_rev at that reverse_step equals -action_arm[t]
    """
    lines = []
    lines.append("forward_t | reverse_step k | g_rev_state | g_rev01 | u_rev(7) | gripper_xyz")

    # Determine how to read g source
    have_ag = "action_gripper" in npz
    have_go = "gripper_open" in npz
    thr = float(npz["gripper_threshold"][0]) if "gripper_threshold" in npz else 0.03

    have_aa = "action_arm" in npz
    have_gp = "gripper_pose" in npz

    for t in idxs.tolist():
        k = T - 1 - t

        # g_rev01 at this keyframe
        if have_ag:
            g01 = float(npz["action_gripper"][t, 0])
        elif have_go:
            g01 = 1.0 if float(npz["gripper_open"][t, 0]) > thr else 0.0
        else:
            g01 = float("nan")

        g_state = _g01_to_state(g01)

        # u_rev = -action_arm[t]
        u_rev = None
        if have_aa:
            aa = npz["action_arm"][t].astype(np.float32).reshape(-1)
            u_rev = (-aa)

        # gripper xyz (if available)
        if have_gp:
            gp = npz["gripper_pose"][t].astype(np.float32).reshape(-1)
            xyz = gp[:3]
            xyz_s = f"[{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}]"
        else:
            xyz_s = "None"

        lines.append(
            f"{t:4d} | {k:4d} | {g_state:6s} | {int(g01) if np.isfinite(g01) else -1:>6d} | {_fmt_vec(u_rev, decimals=2)} | {xyz_s}"
        )

    return "\n".join(lines)


# ----------------------------
# Generic, compact header (already updated)
# ----------------------------
def build_header(
    task: str,
    variation: int,
    demo_index: int,
    T: int,
    control_dt: Optional[float],
    action_mode: str,
    arm_action_mode: str,
    gripper_action_mode: str,
    pre_core: str,
    post_core: str,
) -> str:
    dt_str = f"{control_dt:.6f}" if control_dt is not None else "unknown"
    pre_core = (pre_core or "").strip() or "(not provided)"
    post_core = (post_core or "").strip() or "(not provided)"

    return f"""You are given a FORWARD robot demonstration and must judge whether it is reversible by exact reverse replay.

Neurosymbolic:
- action := (precondition, effect)
- postcondition := precondition + effect

Metadata: task={task}, variation={variation}, demo_index={demo_index}, T={T}, dt={dt_str}s
Control: {action_mode} (arm={arm_action_mode}, gripper={gripper_action_mode})

CORE PRE (given, do not invent predicates):
{pre_core}

CORE POST (given, do not invent predicates):
{post_core}

What you will receive:
- Several keyframes from the FORWARD demo (RGB images) with labels forward_t and reverse_step k.
- A numeric table at the same keyframes showing g_rev (OPEN/CLOSED) and u_rev.

Exact reverse replay rule (MUST use):
- The reverse rollout starts from the EXACT forward final state (the environment is in CORE POST at reverse_step k=0).
- reverse_step index: k = 0..T-1, where k = T-1-forward_t.
- arm command: u_rev[k] = -action_arm[T-1-k].
- gripper target: g_rev[k] = action_gripper[T-1-k].

IMPORTANT gripper semantics (do not invert):
- g_rev[k] = OPEN means not holding / releasing.
- g_rev[k] = CLOSED means trying to grasp or hold.
- CLOSED can succeed by contact: the gripper may be commanded CLOSED before reaching the block and then grasp when contact occurs.

Goal:
Decide whether exact reverse replay plausibly takes the environment from CORE POST → CORE PRE.
Incidental differences are allowed as long as CORE PRE holds (do not require exact pixel-perfect restoration).

Also: check whether CORE PRE/CORE POST are consistent with what is visible in the keyframes.

Output ONLY valid JSON with EXACT keys:
- reversibility_verdict: LIKELY | UNLIKELY | UNCERTAIN
- reversibility_confidence: number in [0,1]
- predicate_consistency: CONSISTENT | INCONSISTENT | UNCERTAIN
- key_evidence: 3–6 short grounded observations referencing forward_t and/or reverse_step k
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to recorded demo .npz")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--max_keyframes", type=int, default=6)
    ap.add_argument("--event_window", type=int, default=2)
    ap.add_argument("--media_resolution", default="low", choices=["unspecified", "low", "medium", "high"])
    ap.add_argument("--thinking", default="LOW", choices=["LOW", "MEDIUM", "HIGH"])

    # Media cost controls
    ap.add_argument("--views", default="front",
                    choices=["front", "front+wrist", "front+overhead", "front+overhead+wrist"])
    ap.add_argument("--jpeg_quality", type=int, default=75)
    ap.add_argument("--max_side", type=int, default=112,
                    help="Downscale images so max(H,W)<=max_side. Use 0 to disable.")
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)

    if "joint_positions" not in npz:
        raise ValueError("Expected 'joint_positions' in the npz.")
    T = int(npz["joint_positions"].shape[0])

    # Metadata
    task = str(npz["task"][0]) if "task" in npz else "UnknownTask"
    variation = int(npz["variation"][0]) if "variation" in npz else -1
    demo_index = int(npz["demo_index"][0]) if "demo_index" in npz else -1

    control_dt = None
    if "control_dt" in npz:
        v = float(npz["control_dt"][0])
        if np.isfinite(v):
            control_dt = v

    action_mode = str(npz["action_mode"][0]) if "action_mode" in npz else "UnknownActionMode"
    arm_action_mode = str(npz["arm_action_mode"][0]) if "arm_action_mode" in npz else "UnknownArmMode"
    gripper_action_mode = str(npz["gripper_action_mode"][0]) if "gripper_action_mode" in npz else "UnknownGripperMode"

    pre_core = str(npz["preconditions_core"][0]) if "preconditions_core" in npz else ""
    post_core = str(npz["postconditions_core"][0]) if "postconditions_core" in npz else ""

    # Keyframes (endpoint-safe)
    k_idx = choose_keyframes(npz, T=T, max_k=args.max_keyframes, event_window=args.event_window)
    k_set = set(map(int, k_idx.tolist()))
    print(f"T={T} keyframes={k_idx.tolist()} includes_start={0 in k_set} includes_last={(T-1) in k_set}")

    # Required visuals
    if "front_rgb" not in npz:
        raise ValueError("Expected 'front_rgb' in the npz.")
    front_imgs = npz["front_rgb"][k_idx]

    have_wrist = ("wrist_rgb" in npz)
    have_overhead = ("overhead_rgb" in npz)

    wrist_imgs = npz["wrist_rgb"][k_idx] if have_wrist else None
    overhead_imgs = npz["overhead_rgb"][k_idx] if have_overhead else None

    want_wrist = ("wrist" in args.views)
    want_overhead = ("overhead" in args.views)

    # Build prompt
    header = build_header(
        task=task,
        variation=variation,
        demo_index=demo_index,
        T=T,
        control_dt=control_dt,
        action_mode=action_mode,
        arm_action_mode=arm_action_mode,
        gripper_action_mode=gripper_action_mode,
        pre_core=pre_core,
        post_core=post_core,
    )

    contents = [header]

    # Compact table (OPEN/CLOSED + u_rev)
    contents.append("\nKEYFRAME TABLE (g_rev shown as OPEN/CLOSED):\n" + compact_keyframe_table(npz, k_idx, T))

    # Frames (kept in forward order, as you requested)
    contents.append("\nKEYFRAMES (FORWARD order). Use labels in evidence. reverse_step k = T-1-forward_t.")
    max_side = None if args.max_side == 0 else args.max_side

    # Determine g source for labels
    have_ag = "action_gripper" in npz
    have_go = "gripper_open" in npz
    thr = float(npz["gripper_threshold"][0]) if "gripper_threshold" in npz else 0.03

    for j, t in enumerate(k_idx.tolist()):
        k = T - 1 - t

        if have_ag:
            g01 = float(npz["action_gripper"][t, 0])
        elif have_go:
            g01 = 1.0 if float(npz["gripper_open"][t, 0]) > thr else 0.0
        else:
            g01 = float("nan")

        g_state = _g01_to_state(g01)

        contents.append(
            f"\nKeyframe {j+1}/{len(k_idx)}: forward_t={t}/{T-1}, reverse_step={k}, g_rev={g_state}"
        )

        contents.append("Front RGB:")
        contents.append(rgb_to_jpeg_part(front_imgs[j], quality=args.jpeg_quality, max_side=max_side))

        if want_overhead and overhead_imgs is not None:
            contents.append("Overhead RGB:")
            contents.append(rgb_to_jpeg_part(overhead_imgs[j], quality=args.jpeg_quality, max_side=max_side))

        if want_wrist and wrist_imgs is not None:
            contents.append("Wrist RGB:")
            contents.append(rgb_to_jpeg_part(wrist_imgs[j], quality=args.jpeg_quality, max_side=max_side))

    # Gemini config
    cfg = {
        "response_mime_type": "application/json",
        "response_json_schema": ReversibilityAssessment.model_json_schema(),
        "thinking_config": {"thinking_level": args.thinking},
        "temperature": 0.2,
        "max_output_tokens": 450,
        "media_resolution": MEDIA_RES_MAP[args.media_resolution],
    }

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=args.model,
        contents=contents,
        config=cfg,
    )

    print(resp.text)
    try:
        print("finish_reason:", resp.candidates[0].finish_reason)
    except Exception:
        pass

    # Robust JSON extraction
    s = (resp.text or "").strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]

    parsed = ReversibilityAssessment.model_validate_json(s)
    print("\n--- VALIDATED JSON ---")
    print(parsed.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
