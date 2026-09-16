# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision F8: does B3 locate the sparse pool's active minute, or just cover it?

Main 10.2 argues that on Huawei's sparse pool no rate estimate can find the
one active minute, because every rate-based policy sees the same flat
8-per-day average. B3 is not a flat-average estimator -- its window is exactly
the 1,440-minute median gap of that pool -- and it does reach 7.73% CSR there
against the 18.6-22.2% band, so the mechanism has to be measured rather than
asserted.

The diagnostic is the one main 10.2 already uses for the mixed pool: prewarm
duty (fraction of function-minutes with a prewarm order), recall (fraction of
arrival-minutes that were prewarmed) and lift = recall / duty, which is 1.0
for untargeted prewarming at the same duty and >1 only if the policy places
its orders where the arrivals are.

Output: results/runs/revision_f8_sparse_duty.json
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RUNS = PROJECT_ROOT / "results" / "runs"
POOLS = ["h_sparse", "h_mixed"]
RHOS = [1.0, 10.0, 100.0]
ARMS = {"B4a_ewma": "des_jobs_huawei", "A5_full_system": "des_jobs_huawei",
        "B1_fixed_keepalive": "des_jobs_huawei", "Oracle": "des_jobs_huawei",
        "B3_fourier": "des_jobs_f3_huawei"}


def main():
    out = {"metric": "duty = P(prewarm>0); recall = P(prewarm>0 | arrival); "
                     "lift = recall/duty (1.0 = untargeted at equal duty)",
           "pools": {}}
    for pool in POOLS:
        counts = np.load(RUNS / "des_jobs_huawei" / pool / "shared.npz")["counts"]
        arrival = counts > 0
        n_arr = int(arrival.sum())
        pool_out = {"n_functions": int(counts.shape[0]),
                    "n_ticks": int(counts.shape[1]),
                    "arrival_minutes": n_arr,
                    "arrival_density": float(arrival.mean()),
                    "by_rho": {}}
        for rho in RHOS:
            rho_out = {}
            for arm, root in ARMS.items():
                p = RUNS / root / pool / f"{arm}__rho{rho}.npz"
                if not p.exists():
                    continue
                pw = np.load(p)["prewarm"] > 0
                duty = float(pw.mean())
                recall = float(pw[arrival].mean())
                rho_out[arm] = {
                    "duty": duty, "recall": recall,
                    "lift": float(recall / duty) if duty > 0 else 0.0,
                    "precision": float(arrival[pw].mean()) if pw.any() else 0.0}
            pool_out["by_rho"][str(rho)] = rho_out
            line = " | ".join(
                f"{a.split('_')[0]} duty {v['duty']:.3f} recall {v['recall']:.3f}"
                f" lift {v['lift']:.2f}" for a, v in rho_out.items())
            print(f"{pool} rho={rho:6.1f}: {line}", flush=True)
        out["pools"][pool] = pool_out

    path = RUNS / "revision_f8_sparse_duty.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")


if __name__ == "__main__":
    main()
