# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision F6: forecast-accuracy diagnostics behind the B3 verdicts.

The main text attributes B3's win on Huawei's mixed pool to the daily period
carrying the signal there, and its loss on Azure to over-prediction. Those are
claims about the forecasts, not the simulation, so they are measured directly
here and archived: per-pool mean predicted rate, mean absolute error and
correlation for B3, EWMA and two references (lag-1 persistence, seasonal
naive), plus the concentration of B3's cold-start advantage over EWMA.

Output: results/runs/revision_f6_fourier_diag.json
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import rates_ewma, rates_fourier  # noqa: E402

RUNS = PROJECT_ROOT / "results" / "runs"
WARMUP = 1440          # skip the first day: B3 has no full window before it

SURFACES = [
    ("azure2021_S1", "data/processed", "s1_test", None),
    ("azure2021_S2", "data/processed", "s2_test", None),
    ("azure2021_S3", "data/processed", "s3_test", "s3_test_t_start"),
    ("huawei_h_mixed", None, "des_jobs_huawei/h_mixed", None),
    ("huawei_h_sparse", None, "des_jobs_huawei/h_sparse", None),
    ("huawei_h_saturated", None, "des_jobs_huawei/h_saturated", None),
    ("azure2019_S1", None, "des_jobs_2019/S1", None),
    ("azure2019_S2", None, "des_jobs_2019/S2", None),
    ("azure2019_S3", None, "des_jobs_2019/S3", None),
]


def load_counts(proc, key, t0key):
    if proc is None:
        return np.load(RUNS / key / "shared.npz")["counts"].astype(np.float64)
    counts = np.load(PROJECT_ROOT / proc / "counts.npy")
    splits = np.load(PROJECT_ROOT / proc / "splits.npz")
    c = counts[splits[key]]
    if t0key:
        c = c[:, int(splits[t0key][0]):]
    return c.astype(np.float64)


def scores(pred, actual):
    return {"mean_predicted": float(pred.mean()),
            "mean_actual": float(actual.mean()),
            "mae": float(np.abs(pred - actual).mean()),
            "corr": float(np.corrcoef(pred.ravel(), actual.ravel())[0, 1])}


def main():
    out = {"warmup_ticks_skipped": WARMUP, "surfaces": {}}
    for name, proc, key, t0key in SURFACES:
        c = load_counts(proc, key, t0key)
        if c.shape[1] <= WARMUP + 60:
            print(f"{name}: too short ({c.shape[1]} ticks), skipped")
            continue
        rf, re_ = rates_fourier(c), rates_ewma(c)
        lag1 = np.zeros_like(c); lag1[:, 1:] = c[:, :-1]
        sea = np.zeros_like(c); sea[:, 1440:] = c[:, :-1440]
        act = c[:, WARMUP:]
        entry = {m: scores(p[:, WARMUP:], act) for m, p in
                 [("B3_fourier", rf), ("B4a_ewma", re_),
                  ("lag1_persistence", lag1), ("seasonal_naive", sea)]}
        entry["n_functions"] = int(c.shape[0])
        entry["n_ticks"] = int(c.shape[1])
        out["surfaces"][name] = entry
        print(f"{name}: B3 mae {entry['B3_fourier']['mae']:.4f} "
              f"r {entry['B3_fourier']['corr']:.3f} | EWMA mae "
              f"{entry['B4a_ewma']['mae']:.4f} r {entry['B4a_ewma']['corr']:.3f}",
              flush=True)

    # Mean forecast level on the exact matrix each DES campaign was run on
    # (F1 slices S3 at s3_test_t_start before computing rates, so B3's history
    # restarts there; the full-matrix diagnostics above answer a different
    # question and must not be quoted for the campaign).
    des_seg = {}
    for name, proc, key, t0key in SURFACES:
        if not name.startswith("azure2021"):
            continue
        counts = np.load(PROJECT_ROOT / proc / "counts.npy")
        splits = np.load(PROJECT_ROOT / proc / "splits.npz")
        c = counts[splits[key]]
        if t0key:
            c = c[:, int(splits[t0key][0]):]
        c = c.astype(np.float64)
        r = rates_fourier(c)
        des_seg[name] = {"mean_predicted": float(r.mean()),
                         "mean_actual": float(c.mean()),
                         "n_ticks": int(c.shape[1])}
        print(f"{name} (DES segment): predicted {r.mean():.4f} "
              f"actual {c.mean():.4f}", flush=True)
    out["des_eval_segment"] = des_seg

    # concentration of B3's advantage over EWMA on the Huawei mixed pool
    b3 = [r for r in json.load(open(RUNS / "revision_f3_fourier_huawei_h_mixed.json"))
          if r["cost_ratio"] == 10.0]
    ew = [r for r in json.load(open(RUNS / "revision_h2_huawei_des_h_mixed.json"))
          if r["cost_ratio"] == 10.0 and r["method"] == "B4a_ewma"]
    a = np.mean([r["func_cold"] for r in b3], axis=0)
    t = np.mean([r["func_total"] for r in b3], axis=0)
    b = np.mean([r["func_cold"] for r in ew], axis=0)
    gap = b - a
    order = np.argsort(-gap)
    out["huawei_mixed_concentration_rho10"] = {
        "frac_functions_b3_fewer_cold": float((a[t > 0] < b[t > 0]).mean()),
        "b3_total_cold": float(a.sum()), "ewma_total_cold": float(b.sum()),
        "top5_share_of_gap": float(gap[order[:5]].sum() / gap.sum()),
        "n_active": int((t > 0).sum())}
    print("huawei mixed:", out["huawei_mixed_concentration_rho10"])

    path = RUNS / "revision_f6_fourier_diag.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")


if __name__ == "__main__":
    main()
