# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision E5c: re-run the E5 Chronos cohort pass recording per-minute
rolling CSR and first-hour cold counts (the E5 archive kept only totals).

Everything decision-relevant is identical to revision_e5_chronos.py:
same cohort (find_natural_cohort, seed 42, W=240), same registered constants
(chronos-bolt-small, CONTEXT=512, quantile clamp 0.9), same DES seeds {0,1,2},
same newsvendor decision layer. The only change is bookkeeping: roll_cold /
roll_total accumulated as per-minute arrays (as in revision_a4_crosstrace.py)
so rolling_csr(window=15) and func_cold60 can be archived.

Sanity gate: overall_csr per rho must reproduce revision_e5_chronos.json
within 5e-5 absolute (same rates, same seeds -> identical decisions).

Output: results/runs/revision_e5c_chronos_rolling.json
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("SF_DATA_DIR", "processed_2019")
os.environ.setdefault("HF_HOME", "/data/260715/.hf-cache")

from scripts.revision_a4_crosstrace import find_natural_cohort  # noqa: E402
from scripts.phase63_onboarding_drift import DES_SEEDS  # noqa: E402
from scripts.phase6_des import decisions_from_rates, COLD_INIT  # noqa: E402
from src.sim.des import simulate_function, rolling_csr  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_2019"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
W = 240
RHOS = [1.0, 10.0, 100.0]
CHRONOS_MODEL = "amazon/chronos-bolt-small"
CHRONOS_CONTEXT = 512
TAU = newsvendor_quantile(10.0)
TAU_READ = min(TAU, 0.9)


def main():
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    pts = find_natural_cohort(counts)
    F_n = len(pts)
    print(f"cohort: {F_n} functions")

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        CHRONOS_MODEL, device_map="cuda", torch_dtype=torch.bfloat16)

    seg_counts = np.stack([np.asarray(counts[fi, t0:t0 + W], dtype=np.int64)
                           for (fi, t0) in pts])
    hist = np.stack([
        np.concatenate([
            np.zeros(max(0, CHRONOS_CONTEXT - t0), dtype=np.float32),
            np.asarray(counts[fi, max(0, t0 - CHRONOS_CONTEXT):t0 + W],
                       dtype=np.float32)])
        for (fi, t0) in pts])

    rates = np.zeros((F_n, W), dtype=np.float32)
    t_start = time.time()
    with torch.no_grad():
        for w in range(W):
            ctx = torch.from_numpy(hist[:, w:w + CHRONOS_CONTEXT])
            q, _ = pipe.predict_quantiles(ctx, prediction_length=1,
                                          quantile_levels=[TAU_READ])
            rates[:, w] = np.maximum(q[:, 0, 0].float().numpy(), 0.0)
            if (w + 1) % 60 == 0:
                print(f"  chronos {w+1}/{W} ({time.time()-t_start:.0f}s)",
                      flush=True)

    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] for (fi, _) in pts]), nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] for (fi, _) in pts]), nan=0.5)

    ref = json.load(open(RUNS_DIR / "revision_e5_chronos.json"))
    out = {"model": CHRONOS_MODEL, "context": CHRONOS_CONTEXT,
           "tau_read": TAU_READ, "n_functions": F_n,
           "reproduces": "revision_e5_chronos.json overall_csr per rho",
           "by_rho": {}}
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
        overall = float(roll_c.sum() / max(roll_n.sum(), 1e-9))
        ref_csr = ref["by_rho"][str(rho)]["overall_csr"]
        assert abs(overall - ref_csr) < 5e-5, (rho, overall, ref_csr)
        out["by_rho"][str(rho)] = {
            "overall_csr": overall,
            "ref_overall_csr_e5": ref_csr,
            "func_cold60": func_cold60.tolist(),
            "func_cold_all": func_cold_all.tolist(),
            "wm_total": float(func_wm.sum()),
            "rolling_csr": rolling_csr(roll_c, roll_n, window=15).tolist()}
        print(f"rho={rho}: CSR={overall*100:.3f}% (e5 ref {ref_csr*100:.3f}%) OK",
              flush=True)

    path = RUNS_DIR / "revision_e5c_chronos_rolling.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")


if __name__ == "__main__":
    main()
