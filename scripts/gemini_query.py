#!/usr/bin/env python3
import argparse
import io
import os
import numpy as np
from PIL import Image
from typing import List, Literal
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
# Structured output schema
# ----------------------------
class DualReverseAssessmentFlat(BaseModel):
    topple_verdict: Literal["LIKELY", "UNLIKELY", "UNCERTAIN"]
    topple_confidence: float = Field(ge=0.0, le=1.0)
    topple_key_evidence: List[str]

    undo_verdict: Literal["LIKELY", "UNLIKELY", "UNCERTAIN"]
    undo_confidence: float = Field(ge=0.0, le=1.0)
    undo_key_evidence: List[str]

    inverse_success_verdict: Literal["LIKELY", "UNLIKELY", "UNCERTAIN"]
    inverse_success_confidence: float = Field(ge=0.0, le=1.0)
    inverse_success_rationale: List[str]

    expected_outcome: str
    likely_failure_modes: List[str]
    what_to_log_next: List[str]


# ----------------------------
# Image encoding
# ----------------------------
def rgb_to_jpeg_part(rgb_u8: np.ndarray, quality: int = 90) -> types.Part:
    if rgb_u8.dtype != np.uint8:
        rgb_u8 = rgb_u8.astype(np.uint8)
    img = Image.fromarray(rgb_u8, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


# ----------------------------
# Keyframe selection
# ----------------------------
def select_keyframes(gripper_open_T1: np.ndarray, T: int, max_k: int = 12, event_window: int = 3) -> np.ndarray:
    """
    Event-based keyframes around gripper open/close transitions.
    Always includes endpoints 0 and T-1.
    """
    go = gripper_open_T1.reshape(T)
    state = (go > 0.5).astype(np.int32)
    flips = np.where(state[1:] != state[:-1])[0] + 1

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


def force_include_endpoints(k_idx: np.ndarray, T: int, max_k: int) -> np.ndarray:
    """
    Guarantee that 0 and T-1 are present even after truncation.
    If too many, keep endpoints + evenly spaced from middle.
    """
    k_set = set(map(int, k_idx.tolist()))
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


# ----------------------------
# Text summaries
# ----------------------------
def numeric_timeline(npz, idxs: np.ndarray) -> str:
    lines = []
    lines.append("t | gripper_open | gripper_pose(7) | joint_pos(7) | action_arm(7) | action_gripper(1)")
    for t in idxs.tolist():
        go = float(npz["gripper_open"][t, 0]) if "gripper_open" in npz else float("nan")
        gp = npz["gripper_pose"][t].tolist() if "gripper_pose" in npz else None
        jp = npz["joint_positions"][t].tolist() if "joint_positions" in npz else None
        aa = npz["action_arm"][t].tolist() if "action_arm" in npz else None
        ag = float(npz["action_gripper"][t, 0]) if "action_gripper" in npz else float("nan")
        lines.append(f"{t} | {go:.3f} | {gp} | {jp} | {aa} | {ag:.0f}")
    return "\n".join(lines)


def numeric_timeline_reverse(npz, idxs: np.ndarray, T: int) -> str:
    lines = []
    lines.append("reverse_step k | source_forward_t | u_rev(7) | g_rev(1)")
    for t in idxs.tolist():
        k = T - 1 - t
        aa = npz["action_arm"][t].astype(np.float32) if "action_arm" in npz else None
        ag = float(npz["action_gripper"][t, 0]) if "action_gripper" in npz else float("nan")

        u_rev = (-aa).tolist() if aa is not None else None
        g_rev = ag

        lines.append(f"{k} | {t} | {u_rev} | {g_rev:.0f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to StackBlocks_varXX_demoYYYY.npz")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--max_keyframes", type=int, default=12)
    ap.add_argument("--media_resolution", default=None, choices=[None, "unspecified", "low", "medium", "high"])
    ap.add_argument("--event_window", type=int, default=3, help="Keyframe window around gripper open/close flips")
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)

    # Metadata (best effort)
    task = str(npz["task"][0]) if "task" in npz else "UnknownTask"
    variation = int(npz["variation"][0]) if "variation" in npz else -1
    demo_index = int(npz["demo_index"][0]) if "demo_index" in npz else -1
    control_dt = float(npz["control_dt"][0]) if "control_dt" in npz and np.isfinite(npz["control_dt"][0]) else None

    # Length
    if "joint_positions" not in npz:
        raise ValueError("Expected 'joint_positions' in the npz.")
    T = int(npz["joint_positions"].shape[0])

    # ----------------------------
    # Choose keyframes (robust)
    #   - If recorder stored keyframe_indices, use them as a hint.
    #   - ALWAYS force include 0 and T-1.
    #   - If after forcing endpoints we exceed max, downsample.
    # ----------------------------
    if "keyframe_indices" in npz:
        k_idx = npz["keyframe_indices"].astype(np.int32)
    else:
        if "gripper_open" not in npz:
            # Fallback: evenly spaced with endpoints
            k_idx = np.linspace(0, T - 1, num=min(args.max_keyframes, T), dtype=np.int32)
        else:
            k_idx = select_keyframes(npz["gripper_open"], T, max_k=max(args.max_keyframes, 2), event_window=args.event_window)

    # Force endpoints and downsample to max_keyframes
    k_idx = force_include_endpoints(k_idx, T, max_k=max(args.max_keyframes, 2))

    # ----------------------------
    # IMPORTANT FIX:
    # Always slice from full RGB arrays using the final k_idx.
    # Do NOT rely on *_rgb_keyframes saved by recorder (can mismatch when you change max_keyframes).
    # ----------------------------
    if "front_rgb" not in npz:
        raise ValueError("Expected 'front_rgb' in the npz.")
    front_imgs = npz["front_rgb"][k_idx]

    wrist_imgs = None
    if "wrist_rgb" in npz:
        wrist_imgs = npz["wrist_rgb"][k_idx]

    # Debug prints (helpful and cheap)
    print(f"T={T}  chosen_keyframes={k_idx.tolist()}  includes_last={(T-1) in set(k_idx.tolist())}")

    reverse_rule = (
        "Backward execution rule:\n"
        "- Arm: we treat action_arm[t] as joint-velocity control (proxy). We execute u_rev[k] = -action_arm[T-1-k].\n"
        "- Gripper: action_gripper[t] is a proxy for target open-state (1=open, 0=closed) derived from gripper_open.\n"
        "  In reverse, we command g_rev[k] = action_gripper[T-1-k].\n"
        "We start from the EXACT final state of the forward demo (tower built + final robot configuration) and use the same dt.\n"
        "Note: actions are proxies (derived), not guaranteed to be identical to the planner’s internal commands.\n"
    )

    header = (
        f"Task: {task}, variation={variation}, demo_index={demo_index}\n"
        + (f"control_dt={control_dt} seconds\n" if control_dt is not None else "")
        + "\nWe recorded a successful FORWARD stacking demonstration.\n"
        + "A 'tower' means at least 2 blocks stacked vertically in stable contact.\n"
        + "We will now test a REVERSE rollout starting from the EXACT final forward state.\n\n"
        + reverse_rule
        + "\nQuestion:\n"
        + "Given the scene geometry and contacts visible in the frames, will this reverse rollout:\n"
        + "1) topple the tower (tower falls), and/or\n"
        + "2) restore/unstack the tower (clean dismantle),\n"
        + "or is it likely to fail (collide, miss contacts, jam, leave tower intact)?\n\n"
        + "Inverse objective for this task is to disperse. Toppling or unstacking counts as achieving the inverse objective.\n"
        + "\nReturn ONLY JSON with EXACTLY these keys:\n"
        + "topple_verdict, topple_confidence, topple_key_evidence,\n"
        + "undo_verdict, undo_confidence, undo_key_evidence,\n"
        + "inverse_success_verdict, inverse_success_confidence, inverse_success_rationale,\n"
        + "expected_outcome, likely_failure_modes, what_to_log_next.\n"
    )

    print(header)

    contents = [header]
    contents.append("\nNUMERIC TIMELINE AT KEYFRAMES (forward indices):\n" + numeric_timeline(npz, k_idx))
    contents.append("\nREVERSE-STEP COMMANDS AT KEYFRAMES:\n" + numeric_timeline_reverse(npz, k_idx, T))

    contents.append("\nFRAMES at keyframe indices (forward time). Each frame label includes reverse_step = T-1-forward_t:")
    for j, t in enumerate(k_idx.tolist()):
        go = float(npz["gripper_open"][t, 0]) if "gripper_open" in npz else float("nan")
        k = T - 1 - t
        contents.append(f"\nKeyframe {j+1}/{len(k_idx)}: forward_t={t}/{T-1}, reverse_step={k}, gripper_open={go:.3f}")
        contents.append("Front RGB:")
        contents.append(rgb_to_jpeg_part(front_imgs[j]))
        if wrist_imgs is not None:
            contents.append("Wrist RGB:")
            contents.append(rgb_to_jpeg_part(wrist_imgs[j]))

    # Structured output via JSON schema
    cfg = {
        "response_mime_type": "application/json",
        "response_json_schema": DualReverseAssessmentFlat.model_json_schema(),
        "thinking_config": {"thinking_level": "LOW"},
        "temperature": 0.2,
        "max_output_tokens": 900,
    }
    if args.media_resolution:
        cfg["media_resolution"] = MEDIA_RES_MAP[args.media_resolution]

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

    parsed = DualReverseAssessmentFlat.model_validate_json(resp.text)
    print("\n--- VALIDATED JSON ---")
    print(parsed.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
