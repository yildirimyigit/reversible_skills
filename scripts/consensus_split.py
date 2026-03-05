#!/usr/bin/env python3
"""
consensus_split.py

Compute task-level consensus split points using multiple *_triage_hybrid.json files.

For each (task, variation) group:
- Extract per-demo candidate split s_d = k_prev of first failing segment, where
    segment_success_rate < fail_thresh
- support = (# demos with a candidate split) / N
- consensus split = median(s_d) (robust)
- spread = MAD(s_d) (median absolute deviation, robust)

Optionally, "snap" the consensus split to each demo's available keyframes:
- snapped_d = max{k in keyframes_sorted_d : k <= consensus}
- consensus_snapped = median(snapped_d)

Outputs:
- Prints a report
- Writes a JSON summary to --out_json
"""

from __future__ import annotations

import argparse
import json
import os
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from statistics import median


def mad(values: List[float]) -> float:
    """Median absolute deviation (MAD)."""
    if not values:
        return float("nan")
    m = median(values)
    return float(median([abs(v - m) for v in values]))


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def first_failing_split(
    triage: Dict[str, Any],
    fail_thresh: float,
    use_mean_zdist_gate: bool = False,
    mean_zdist_thresh: float = 0.0,
) -> Optional[int]:
    """
    Return k_prev of the first segment that is considered failing.

    Default: fail if success_rate < fail_thresh.

    Optional additional gate:
      also require mean z_dist of that segment >= mean_zdist_thresh
      (useful if you ever see spurious low success_rate due to tiny tolerance issues)
    """
    segs = triage.get("segment_results", []) or []
    for seg in segs:
        sr = safe_float(seg.get("success_rate", 1.0), 1.0)
        if sr < fail_thresh:
            if use_mean_zdist_gate:
                trials = seg.get("trials", []) or []
                if trials:
                    mean_dz = sum(safe_float(t.get("z_dist", 0.0), 0.0) for t in trials) / max(1, len(trials))
                else:
                    mean_dz = 0.0
                if mean_dz < mean_zdist_thresh:
                    continue
            return safe_int(seg.get("k_prev", None), None)
    return None


def snap_to_keyframes(consensus: int, keyframes_sorted: List[int]) -> Optional[int]:
    """Snap consensus split down to the nearest available keyframe in this demo."""
    if consensus is None:
        return None
    kfs = [int(k) for k in (keyframes_sorted or [])]
    kfs = sorted(set(kfs))
    if not kfs:
        return None
    eligible = [k for k in kfs if k <= int(consensus)]
    if eligible:
        return int(max(eligible))
    # if consensus is before first keyframe, snap to earliest
    return int(kfs[0])


@dataclass
class DemoSplit:
    path: str
    task: str
    variation: int
    split: Optional[int]
    full_success_rate: Optional[float]
    keyframes_sorted: List[int]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/demos", help="Directory to search recursively for *_triage_hybrid.json")
    ap.add_argument("--pattern", default="**/*_triage_hybrid.json", help="Glob pattern relative to root")
    ap.add_argument("--fail_thresh", type=float, default=0.5, help="Segment success_rate below this => failing")
    ap.add_argument("--min_support", type=float, default=0.6, help="Minimum support fraction to accept split")
    ap.add_argument("--use_mean_zdist_gate", action="store_true", default=False)
    ap.add_argument("--mean_zdist_thresh", type=float, default=0.0)
    ap.add_argument("--out_json", default="data/consensus_splits.json")
    ap.add_argument("--quiet", action="store_true", default=False)
    args = ap.parse_args()

    root = args.root
    files = sorted(glob.glob(os.path.join(root, args.pattern), recursive=True))
    if not files:
        raise SystemExit(f"No triage json files found under {root} with pattern {args.pattern}")

    demos: List[DemoSplit] = []
    for p in files:
        tri = load_json(p)
        task = str(tri.get("task", ""))
        variation = safe_int(tri.get("variation", 0), 0)

        split = first_failing_split(
            tri,
            fail_thresh=float(args.fail_thresh),
            use_mean_zdist_gate=bool(args.use_mean_zdist_gate),
            mean_zdist_thresh=float(args.mean_zdist_thresh),
        )

        full_sr = None
        fr = tri.get("full_reversal", None)
        if isinstance(fr, dict) and "success_rate" in fr:
            full_sr = safe_float(fr.get("success_rate", None), None)

        kfs = tri.get("keyframes_sorted", []) or []
        kfs = [safe_int(x, x) for x in kfs if x is not None]

        demos.append(DemoSplit(
            path=p,
            task=task,
            variation=variation,
            split=split,
            full_success_rate=full_sr,
            keyframes_sorted=kfs,
        ))

    # group by (task, variation)
    groups: Dict[Tuple[str, int], List[DemoSplit]] = {}
    for d in demos:
        groups.setdefault((d.task, d.variation), []).append(d)

    summary: Dict[str, Any] = {
        "root": root,
        "pattern": args.pattern,
        "fail_thresh": float(args.fail_thresh),
        "min_support": float(args.min_support),
        "use_mean_zdist_gate": bool(args.use_mean_zdist_gate),
        "mean_zdist_thresh": float(args.mean_zdist_thresh),
        "groups": {},
    }

    # report
    for (task, var), ds in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        N = len(ds)
        splits = [d.split for d in ds if d.split is not None]
        support = len(splits) / max(1, N)

        consensus = int(median(splits)) if splits else None
        spread = mad([float(s) for s in splits]) if splits else float("nan")

        # snapped consensus
        snapped_list: List[int] = []
        if consensus is not None:
            for d in ds:
                s = snap_to_keyframes(consensus, d.keyframes_sorted)
                if s is not None:
                    snapped_list.append(s)
        consensus_snapped = int(median(snapped_list)) if snapped_list else None
        spread_snapped = mad([float(x) for x in snapped_list]) if snapped_list else float("nan")
        recommended = consensus_snapped if consensus_snapped is not None else consensus

        accept = (consensus is not None) and (support >= float(args.min_support))

        if not args.quiet:
            print(f"\n=== {task} var{var:02d} ===")
            print(f"demos={N} | split_support={support:.2f} | accept={accept}")
            print(f"raw_splits={splits if splits else 'None'}")
            print(f"consensus_median={consensus} | MAD={spread:.2f}")
            print(f"consensus_snapped={consensus_snapped} | MAD_snapped={spread_snapped:.2f}")
            print(f"recommended_split_idx={recommended}")
            for d in sorted(ds, key=lambda x: os.path.basename(x.path)):
                b = os.path.basename(d.path)
                print(f"  - {b:50s} split={d.split} full_sr={d.full_success_rate}")

        summary["groups"][f"{task}__var{var:02d}"] = {
            "task": task,
            "variation": var,
            "n_demos": N,
            "support": support,
            "accepted": bool(accept),
            "splits": splits,
            "consensus_median": consensus,
            "mad": None if (spread != spread) else spread,  # NaN check: NaN != NaN
            "mad_snapped": None if (spread_snapped != spread_snapped) else spread_snapped,
            "consensus_snapped": consensus_snapped,
            "recommended_split_idx": recommended,
            "per_demo": [
                {
                    "path": d.path,
                    "split": d.split,
                    "full_success_rate": d.full_success_rate,
                    "keyframes_sorted": d.keyframes_sorted,
                }
                for d in sorted(ds, key=lambda x: x.path)
            ],
        }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    if not args.quiet:
        print(f"\n[ok] wrote {args.out_json}")


if __name__ == "__main__":
    main()