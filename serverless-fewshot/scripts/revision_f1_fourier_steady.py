# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision F1: B3 (Fourier/harmonic extrapolation) in STEADY STATE, Azure 2021.

B3 is registered in src/models/baselines.py:FourierPredictor and was listed in
the baseline registry from WP2, but was never run through the decision layer.
Reviewer question (2026-08-03): the related work cites spectral prewarming
(IceBreaker), so the registered spectral arm has to be measured rather than
represented by proxy.

Protocol is the steady-state campaign's, unchanged: 2021 S1/S2/S3 test pools
(S3 evaluated from s3_test_t_start), REDUCED_RHOS, DES seeds 0..9, the
identical newsvendor decision layer, the identical cold-init calibration.
Comparators are read from the archived campaign (sim_results_des_revision.json)
so no other arm is re-simulated.

Output: results/runs/revision_f1_fourier_steady.json
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import (  # noqa: E402
    COLD_INIT, REDUCED_RHOS, decisions_from_rates, rates_fourier, run_config,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402

PROC = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
SEEDS = list(range(10))
SPLITS = ["S1", "S2", "S3"]
CAMPAIGN = "sim_results_des_revision.json"
COMPARATORS = ["A5_full_system", "A5_gated_v3", "B4a_ewma",
               "B1_fixed_keepalive", "Oracle"]


def main():
    workers = int(os.environ.get("F1_WORKERS", "6"))
    counts_all = np.load(PROC / "counts.npy")
    splits = np.load(PROC / "splits.npz")
    dur_df = pd.read_csv(PROC / "duration_stats.csv")

    all_results = []
    for split in SPLITS:
        test_idx = splits[{"S1": "s1_test", "S2": "s2_test",
                           "S3": "s3_test"}[split]]
        counts = counts_all[test_idx]
        if split == "S3":
            t0 = int(splits.get("s3_test_t_start", [10080])[0])
            counts = counts[:, t0:]
        dm = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
             for fi in test_idx]), nan=1.0)
        ds = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
             for fi in test_idx]), nan=0.5)
        print(f"\n### {split}: {counts.shape[0]} funcs, {counts.shape[1]} ticks",
              flush=True)

        t = time.time()
        rates = rates_fourier(counts)
        print(f"  B3 rates in {time.time()-t:.0f}s "
              f"(mean predicted rate {rates.mean():.4f}, "
              f"actual {counts.mean():.4f})", flush=True)

        jobs = []
        for rho in REDUCED_RHOS:
            pw, ka = decisions_from_rates(rates, newsvendor_quantile(rho))
            jobs.append(("B3_fourier", rho, SEEDS, counts, pw, ka,
                         dm, ds, COLD_INIT, split))
        t = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res_list in ex.map(run_config, jobs):
                all_results.extend(res_list)
                r = res_list[0]
                print(f"    B3 rho={r['cost_ratio']:6.1f} "
                      f"CSR={np.mean([x['csr'] for x in res_list]):.4f} "
                      f"WM={np.mean([x['wm_per_1k_inv'] for x in res_list]):.0f}",
                      flush=True)
        print(f"  {split} simulated in {time.time()-t:.0f}s", flush=True)

    camp = json.load(open(RUNS_DIR / CAMPAIGN))
    camp = camp["results"] if isinstance(camp, dict) else camp
    comp = {}
    for split in SPLITS:
        comp[split] = {}
        for m in COMPARATORS:
            comp[split][m] = {}
            for rho in REDUCED_RHOS:
                rows = [r for r in camp if r["split"] == split
                        and r["method"] == m
                        and abs(r["cost_ratio"] - rho) < 1e-9]
                if rows:
                    comp[split][m][str(rho)] = {
                        "csr": float(np.mean([r["csr"] for r in rows])),
                        "csr_std": float(np.std([r["csr"] for r in rows])),
                        "wm": float(np.mean([r["wm_per_1k_inv"] for r in rows])),
                        "n_seeds": len(rows)}

    out = {"arm": "B3_fourier",
           "registered_in": "src/models/baselines.py:FourierPredictor",
           "constants": {"n_harmonics": 10, "window_min": 1440,
                         "refit_ticks": 60, "min_history_ticks": 60},
           "seeds": SEEDS, "rhos": REDUCED_RHOS, "splits": SPLITS,
           "campaign_source": CAMPAIGN,
           "results": all_results, "comparators_from_campaign": comp}
    path = RUNS_DIR / "revision_f1_fourier_steady.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"\nSaved + verified {path}")

    for split in SPLITS:
        for rho in REDUCED_RHOS:
            rs = [r for r in all_results if r["split"] == split
                  and abs(r["cost_ratio"] - rho) < 1e-9]
            line = (f"{split} rho={rho:6.1f}: B3 "
                    f"{np.mean([r['csr'] for r in rs])*100:.2f}% / "
                    f"{np.mean([r['wm_per_1k_inv'] for r in rs]):.0f}")
            for m in COMPARATORS:
                c = comp[split][m].get(str(rho))
                if c:
                    line += f" | {m} {c['csr']*100:.2f}% / {c['wm']:.0f}"
            print(line)


if __name__ == "__main__":
    main()
