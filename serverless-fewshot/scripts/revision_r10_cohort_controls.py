# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R10: minimal zero-history controls on both real-deployment cohorts
(pre-registered in revision_r10_PREREG.md). CPU-only arms; no model.

Torch env (for module imports only):
  PYTHONPATH=/data/260715/site-packages:. python3.13 scripts/revision_r10_cohort_controls.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import decisions_from_rates, COLD_INIT  # noqa: E402
from src.sim.des import simulate_function  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402

RUNS = PROJECT_ROOT / "results" / "runs"
W = 240
RHOS = [1.0, 10.0, 100.0]
DES_SEEDS = [0, 1, 2]
ALPHA = 0.1
OUT = RUNS / "revision_r10_cohort_controls.json"


def fleet_prior_log1p():
    """median over 2021 s2_train functions of mean per-minute rate."""
    P21 = PROJECT_ROOT / "data" / "processed"
    counts = np.load(P21 / "counts.npy")
    idx = np.load(P21 / "splits.npz")["s2_train"]
    r_fleet = float(np.median(counts[idx].mean(axis=1)))
    return float(np.log1p(r_fleet)), r_fleet


def ewma_rates(seg_counts, state0=0.0):
    F_n, T = seg_counts.shape
    rates = np.zeros((F_n, T), dtype=np.float32)
    st = np.full(F_n, state0, dtype=np.float64)
    for t in range(T):
        rates[:, t] = np.expm1(np.maximum(st, 0))
        st = ALPHA * np.log1p(seg_counts[:, t].astype(np.float64)) + (1 - ALPHA) * st
    return rates


def b1_prewarm(seg_counts):
    F_n, T = seg_counts.shape
    pw = np.zeros((F_n, T), dtype=np.int32)
    for f in range(F_n):
        act = (seg_counts[f] > 0).astype(np.int32)
        conv = np.convolve(act, np.ones(10, dtype=np.int32))[:T]
        pw[f, 1:] = (conv[:-1] > 0).astype(np.int32)
    return pw


def age_ka(seg_counts, ka_young=60.0, ka_old=10.0, window=60):
    F_n, T = seg_counts.shape
    ka = np.full((F_n, T), ka_old, dtype=np.float32)
    for f in range(F_n):
        nz = np.flatnonzero(seg_counts[f])
        if len(nz) == 0:
            continue
        t0 = nz[0]
        ka[f, t0:min(T, t0 + window)] = ka_young
    return ka


def run_cohort(name, seg_counts, dm, ds, ref_path, ref_ewma_key):
    prior_log, prior_rate = fleet_prior_log1p()
    arms = {}
    arms["B4a_ewma"] = decisions_from_rates_all(ewma_rates(seg_counts), RHOS)
    arms["EWMA_fleetprior"] = decisions_from_rates_all(
        ewma_rates(seg_counts, state0=prior_log), RHOS)
    pw = b1_prewarm(seg_counts)
    ka = age_ka(seg_counts)
    arms["AgeKA60"] = {rho: (pw, ka) for rho in RHOS}

    ref = json.load(open(ref_path))
    out = {"n_functions": int(seg_counts.shape[0]),
           "fleet_prior_rate_per_min": prior_rate, "by_rho": {}}
    F_n = seg_counts.shape[0]
    for rho in RHOS:
        rho_out = {}
        for arm, dec in arms.items():
            pw_m, ka_m = dec[rho]
            cold = np.zeros(F_n); wm = np.zeros(F_n); tot = 0.0
            for f in range(F_n):
                for seed in DES_SEEDS:
                    r = simulate_function(seg_counts[f], pw_m[f], ka_m[f],
                                          float(dm[f]), float(ds[f]),
                                          seed=seed * 7919 + f,
                                          cold_mu=COLD_INIT["mu"],
                                          cold_sigma=COLD_INIT["sigma"],
                                          track_rolling=True)
                    cold[f] += r["roll_cold"].sum()
                    tot += r["roll_total"].sum()
                    wm[f] += r["idle_mem_gb_s"]
            cold /= len(DES_SEEDS); wm /= len(DES_SEEDS); tot /= len(DES_SEEDS)
            rho_out[arm] = {"overall_csr": float(cold.sum() / max(tot, 1e-9)),
                            "func_cold_all": cold.tolist(),
                            "wm_total": float(wm.sum())}
            print(f"  {name} rho={rho} {arm}: csr={rho_out[arm]['overall_csr']*100:.3f}%",
                  flush=True)
        # context from archive
        rb = ref["by_rho"][str(rho)]
        rho_out["_archived"] = {
            "A5_proto_csr": rb["A5_proto"]["overall_csr"],
            "ewma_csr": rb[ref_ewma_key]["overall_csr"]}
        rho_out["_gate_ewma_delta_pp"] = abs(
            rho_out["B4a_ewma"]["overall_csr"] - rb[ref_ewma_key]["overall_csr"]) * 100
        out["by_rho"][str(rho)] = rho_out
    return out


def decisions_from_rates_all(rates, rhos):
    return {rho: decisions_from_rates(rates, newsvendor_quantile(rho))
            for rho in rhos}


def main():
    out = {"prereg": "revision_r10_PREREG.md", "seeds": DES_SEEDS, "cohorts": {}}

    # Azure 2019 cohort (a4 rule)
    from scripts.revision_a4_crosstrace import find_natural_cohort, PROCESSED_DIR
    import pandas as pd
    counts19 = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    dur19 = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    pts = find_natural_cohort(counts19)
    seg = np.stack([np.asarray(counts19[fi, t0:t0 + W], dtype=np.int64)
                    for (fi, t0) in pts])
    dm = np.nan_to_num(np.array([dur19.iloc[fi]["dur_mean"] for (fi, _) in pts]), nan=1.0)
    ds = np.nan_to_num(np.array([dur19.iloc[fi]["dur_std"] for (fi, _) in pts]), nan=0.5)
    out["cohorts"]["azure2019"] = run_cohort(
        "2019", seg, dm, ds, RUNS / "revision_a4_crosstrace.json", "B4a_ewma")

    # Huawei cohort (h1 rule)
    from scripts.revision_h1_huawei_cohort import find_natural_cohort as hw_cohort
    P_HW = PROJECT_ROOT / "data" / "processed_huawei"
    counts_hw = np.load(P_HW / "counts.npy", mmap_mode="r")
    dur_hw = pd.read_csv(P_HW / "duration_stats.csv")
    fp = np.load(P_HW / "first_present.npy")
    pts_hw, funnel = hw_cohort(counts_hw, fp)
    seg_hw = np.stack([np.asarray(counts_hw[fi, t0:t0 + W], dtype=np.int64)
                       for (fi, t0) in pts_hw])
    dm_hw = np.nan_to_num(dur_hw["dur_mean"].to_numpy()[[f for f, _ in pts_hw]], nan=1.0)
    ds_hw = np.nan_to_num(dur_hw["dur_std"].to_numpy()[[f for f, _ in pts_hw]], nan=0.5)
    out["cohorts"]["huawei"] = run_cohort(
        "huawei", seg_hw, dm_hw, ds_hw,
        RUNS / "revision_h1_huawei_cohort.json", "B4a_ewma")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
