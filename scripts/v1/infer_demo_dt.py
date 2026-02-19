#!/usr/bin/env python3
import argparse
import numpy as np

def infer_dt_from_q_v(q, v, vth=0.05):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    dq = q[1:] - q[:-1]
    v = v[:-1]

    mask = np.abs(v) > vth
    dts = []
    for t in range(dq.shape[0]):
        m = mask[t]
        if m.sum() < 3:
            continue
        # keep only joints where dq and v have same sign (positive dt expectation)
        ok = np.sign(dq[t, m]) == np.sign(v[t, m])
        dtj = (dq[t, m] / v[t, m])[ok]
        dtj = dtj[np.isfinite(dtj)]
        if dtj.size == 0:
            continue
        dt = float(np.median(dtj))
        if 0.0 < dt < 0.5:
            dts.append(dt)

    if not dts:
        return float("nan"), 0
    arr = np.array(dts, dtype=np.float64)
    return float(np.median(arr)), int(arr.size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_npz", required=True)
    ap.add_argument("--vth", type=float, default=0.05)
    args = ap.parse_args()

    d = np.load(args.demo_npz, allow_pickle=True)
    sim_dt = d["sim_dt"].item() if "sim_dt" in d.files else None

    q = d["joint_positions"]
    n_arm = q.shape[1]
    a = d["action"]
    v = a[:, :n_arm]

    dt_med, n = infer_dt_from_q_v(q, v, vth=args.vth)

    print("=== Demo dt report ===")
    print("sim_dt (from file):", sim_dt)
    print("inferred control_dt median:", dt_med, f"(from {n} steps)")
    if sim_dt is not None and np.isfinite(dt_med):
        print("ratio control_dt / sim_dt:", dt_med / float(sim_dt))

if __name__ == "__main__":
    main()
