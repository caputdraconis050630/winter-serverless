# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision F4: B3 (Fourier) on the natural onboarding cohorts.

Adds the registered spectral arm to the two zero-history cells:
  azure2019  -- the 100-function natural new-deployment cohort of
                revision_a4_crosstrace.py
  huawei     -- the 76-function cross-provider cohort of
                revision_h1_huawei_cohort.py

Cohort definition, evaluation window (W=240 from t0), DES seeds, per-function
seeding (seed*7919+f), rho grid, duration handling and the decision layer are
taken unchanged from those scripts, so B3's numbers are directly pairable with
the archived arms. Only the B3 arm is simulated here; every comparator is read
from the archived cohort JSON, including its per-function cold-start vector,
which is what the paired tests use.

B3 sees exactly the history a new deployment has: the segment starts at t0,
before which the function does not exist. With less than its 60-tick warm-up
it emits no forecast, and with less than one daily period its window is
whatever history exists -- the honest behavior of a spectral predictor on a
function too young to have a spectrum.

  SF_DATA_DIR=processed_2019   python3.13 scripts/revision_f4_fourier_cohort.py --trace azure2019
  SF_DATA_DIR=processed_huawei python3.13 scripts/revision_f4_fourier_cohort.py --trace huawei

Output: results/runs/revision_f4_fourier_cohort_{trace}.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRACES = {
    "azure2019": {"data": "processed_2019",
                  "archive": "revision_a4_crosstrace.json"},
    "huawei": {"data": "processed_huawei",
               "archive": "revision_h1_huawei_cohort.json"},
}

ap = argparse.ArgumentParser()
ap.add_argument("--trace", choices=sorted(TRACES), required=True)
args = ap.parse_args()
CFG = TRACES[args.trace]
os.environ["SF_DATA_DIR"] = CFG["data"]

from scripts.phase6_des import (  # noqa: E402
    COLD_INIT, decisions_from_rates, rates_fourier,
)
from scripts.phase6_stats import cliffs_delta, holm_bonferroni  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.sim.des import rolling_csr, simulate_function  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / CFG["data"]
RUNS_DIR = PROJECT_ROOT / "results" / "runs"

W = 240
RHOS = [1.0, 10.0, 100.0]
COHORT_CAP = 100
MIN_IDLE_PREFIX = 3 * 1440
MIN_DAY1_INV = 30
DES_SEEDS = [0, 1, 2]
COMPARATORS = ["A5_proto", "A5_proto_zero", "B4a_ewma", "gated_v3", "gated_v4",
               "B1_fixed_keepalive", "Oracle"]


def find_natural_cohort(counts):
    """Verbatim from revision_a4_crosstrace / revision_h1_huawei_cohort."""
    N, T = counts.shape
    picked = []
    for f in range(N):
        row = np.asarray(counts[f])
        nz = np.flatnonzero(row)
        if len(nz) == 0:
            continue
        t0 = int(nz[0])
        if t0 < MIN_IDLE_PREFIX or t0 >= T - 1440:
            continue
        if row[t0:t0 + 1440].sum() < MIN_DAY1_INV:
            continue
        picked.append((f, t0))
    rng = np.random.default_rng(42)
    if len(picked) > COHORT_CAP:
        idx = rng.choice(len(picked), COHORT_CAP, replace=False)
        picked = [picked[i] for i in sorted(idx)]
    return picked


def main():
    from scipy import stats as sps

    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    pts = find_natural_cohort(counts)
    F_n = len(pts)
    print(f"{args.trace} natural cohort: {F_n} functions "
          f"(t0 median {np.median([t for _, t in pts]):.0f} min)", flush=True)

    archive = json.load(open(RUNS_DIR / CFG["archive"]))
    assert archive["n_functions"] == F_n, (
        f"cohort size {F_n} != archived {archive['n_functions']}; the cohort "
        f"must reproduce exactly or the paired tests are meaningless")

    seg_counts = np.stack([np.asarray(counts[fi, t0:t0 + W], dtype=np.int64)
                           for (fi, t0) in pts])
    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"]
                                 for (fi, _) in pts]), nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"]
                                   for (fi, _) in pts]), nan=0.5)

    rates = rates_fourier(seg_counts)
    print(f"B3 rates: mean predicted {rates.mean():.4f}, "
          f"actual {seg_counts.mean():.4f}, "
          f"silent ticks {(rates.sum(axis=0) == 0).sum()}/{W}", flush=True)

    out = {"trace": args.trace, "arm": "B3_fourier",
           "registered_in": "src/models/baselines.py:FourierPredictor",
           "constants": {"n_harmonics": 10, "window_min": 1440,
                         "refit_ticks": 60, "min_history_ticks": 60},
           "n_functions": F_n, "window_min": W, "des_seeds": DES_SEEDS,
           "comparator_archive": CFG["archive"], "by_rho": {}}

    for rho in RHOS:
        tau = newsvendor_quantile(rho)
        pw_m, ka_m = decisions_from_rates(rates, tau)
        roll_c = np.zeros(W)
        roll_n = np.zeros(W)
        func_cold60 = np.zeros(F_n)
        func_cold_all = np.zeros(F_n)
        func_wm = np.zeros(F_n)
        for f in range(F_n):
            for seed in DES_SEEDS:
                r = simulate_function(seg_counts[f], pw_m[f], ka_m[f],
                                      float(dm[f]), float(dstd[f]),
                                      seed=seed * 7919 + f,
                                      cold_mu=COLD_INIT["mu"],
                                      cold_sigma=COLD_INIT["sigma"],
                                      track_rolling=True)
                roll_c += r["roll_cold"]
                roll_n += r["roll_total"]
                func_cold60[f] += r["roll_cold"][:60].sum()
                func_cold_all[f] += r["roll_cold"].sum()
                func_wm[f] += r["idle_mem_gb_s"]
        func_cold60 /= len(DES_SEEDS)
        func_cold_all /= len(DES_SEEDS)
        func_wm /= len(DES_SEEDS)

        entry = {"overall_csr": float(roll_c.sum() / max(roll_n.sum(), 1e-9)),
                 "func_cold60": func_cold60.tolist(),
                 "func_cold_all": func_cold_all.tolist(),
                 "wm_total": float(func_wm.sum()),
                 "rolling_csr": rolling_csr(roll_c, roll_n, window=15).tolist()}

        arch_rho = archive["by_rho"][str(rho)]
        paired = {}
        for base in COMPARATORS:
            if base not in arch_rho:
                continue
            b = np.array(arch_rho[base]["func_cold_all"])
            diff = b - func_cold_all           # >0 means B3 has fewer
            nz = diff[diff != 0]
            p = float(sps.wilcoxon(nz)[1]) if len(nz) >= 6 else 1.0
            d, mag = cliffs_delta(func_cold_all, b)   # >0: B3 has fewer
            paired[f"B3_fourier_vs_{base}"] = {
                "wilcoxon_p": p, "mean_diff": float(diff.mean()),
                "cliffs_delta": d, "magnitude": mag,
                "base_overall_csr": float(arch_rho[base]["overall_csr"]),
                "base_wm_total": float(arch_rho[base]["wm_total"])}
        keys = list(paired)
        for k, pa in zip(keys, holm_bonferroni([paired[k]["wilcoxon_p"]
                                                for k in keys])):
            paired[k]["p_holm"] = float(pa)
        entry["_paired_cold_all"] = paired
        out["by_rho"][str(rho)] = entry

        comp = " ".join(f"{m}={arch_rho[m]['overall_csr']*100:.2f}%"
                        for m in COMPARATORS if m in arch_rho)
        print(f"rho={rho}: B3={entry['overall_csr']*100:.2f}% "
              f"(WM {entry['wm_total']:.0f}) | {comp}", flush=True)

    path = RUNS_DIR / f"revision_f4_fourier_cohort_{args.trace}.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")


if __name__ == "__main__":
    main()
