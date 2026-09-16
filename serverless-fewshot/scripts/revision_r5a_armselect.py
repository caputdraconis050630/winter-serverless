#!/usr/bin/env python3
"""R5a: online arm-selection diagnostic vs measured spectral/EWMA ground
truth (pre-registered in revision_r5a_PREREG.md). Analysis only.

Diagnostic: MAE(seasonal_naive)/MAE(lag1_persistence) per pool (f6).
Ground truth: matched-memory B3-vs-EWMA delta per (pool, rho) (f5 curves,
Table S18 interpolation convention). Tie if |delta| < 0.10 pp.
"""

import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent.parent / "results" / "runs"
MARGIN_PP = 0.10
RHOS = [1.0, 10.0, 100.0]

f6 = json.load(open(RUNS / "revision_f6_fourier_diag.json"))["surfaces"]
f5 = json.load(open(RUNS / "revision_f5_fourier_stats.json"))["surfaces"]

POOL_MAP = {  # f6 key -> (f5 surface, f5 split)
    "azure2021_S1": ("azure2021_steady", "S1"),
    "azure2021_S2": ("azure2021_steady", "S2"),
    "azure2021_S3": ("azure2021_steady", "S3"),
    "azure2019_S1": ("azure2019_steady", "S1"),
    "azure2019_S2": ("azure2019_steady", "S2"),
    "azure2019_S3": ("azure2019_steady", "S3"),
    "huawei_h_mixed": ("huawei_steady", "h_mixed"),
    "huawei_h_sparse": ("huawei_steady", "h_sparse"),
    "huawei_h_saturated": ("huawei_steady", "h_saturated"),
}


def interp_delta(curve_b3, wm_ewma, csr_ewma):
    """B3 CSR interpolated at EWMA's memory, minus EWMA's CSR (pp).
    Registered convention of make_b3_table.py; None if out of range."""
    wm = np.array([c["wm_per_1k"] for c in curve_b3])
    csr = np.array([c["csr_pp"] for c in curve_b3])
    o = np.argsort(wm)
    wm, csr = wm[o], csr[o]
    if wm_ewma < wm.min() or wm_ewma > wm.max():
        return None
    return float(np.interp(wm_ewma, wm, csr) - csr_ewma)


out = {"prereg": "revision_r5a_PREREG.md", "margin_pp": MARGIN_PP, "pools": {}}
correct = mis = ties_ok = unresolved = 0
for pool, (surf, split) in POOL_MAP.items():
    d6 = f6[pool]
    ratio = d6["seasonal_naive"]["mae"] / d6["lag1_persistence"]["mae"]
    pick = "spectral" if ratio < 1.0 else "ewma"
    curves = f5[surf][split]["curves"]
    by_rho = {}
    verdicts = []
    for rho in RHOS:
        ew = next(c for c in curves["B4a_ewma"] if c["rho"] == rho)
        delta = interp_delta(curves["B3_fourier"], ew["wm_per_1k"], ew["csr_pp"])
        if delta is None:
            truth = "unresolved(out-of-range)"
        elif abs(delta) < MARGIN_PP:
            truth = "tie"
        else:
            truth = "spectral" if delta < 0 else "ewma"
        verdicts.append(truth)
        by_rho[str(rho)] = {"matched_memory_delta_pp": delta, "truth": truth}
    resolved = [v for v in verdicts if v in ("spectral", "ewma")]
    if not resolved:
        pool_truth = "tie" if "tie" in verdicts else "unresolved"
    else:
        pool_truth = max(set(resolved), key=resolved.count)
    if pool_truth in ("tie",):
        align, ties_ok = "tie-either-arm-ok", ties_ok + 1
    elif pool_truth == "unresolved":
        align, unresolved = "unresolved", unresolved + 1
    elif pick == pool_truth:
        align, correct = "correct", correct + 1
    else:
        align, mis = "MISMATCH", mis + 1
    out["pools"][pool] = {
        "diag_ratio_seasonal_over_lag1": round(ratio, 4),
        "diag_pick": pick, "by_rho": by_rho,
        "pool_truth": pool_truth, "alignment": align,
    }
    print(f"{pool:22s} ratio={ratio:5.2f} pick={pick:8s} truth={pool_truth:10s} -> {align}")

out["summary"] = {"correct": correct, "mismatch": mis,
                  "tie_pools": ties_ok, "unresolved": unresolved}
path = RUNS / "revision_r5a_armselect.json"
json.dump(out, open(path, "w"), indent=1)
print(f"\nsummary: {out['summary']}")
print(f"wrote {path}")
