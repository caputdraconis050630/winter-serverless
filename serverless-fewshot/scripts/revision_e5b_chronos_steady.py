# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision E5b: Chronos-Bolt-small in STEADY STATE (2021 S3 temporal split).

Complement to E5 (cohort): the TSFM won the zero/few-shot regime — does it
also beat EWMA/A5 once functions are converged? Either answer sharpens the
paper's boundary claim; measured, not assumed.

Protocol mirrors the steady-state campaign: 2021 S3 test functions, eval
segment t >= s3_test_t_start, REDUCED_RHOS, DES seeds 0..9, same decision
layer (rate = predicted TAU(10)-fractile, clamped to Bolt's 0.9 cap).
Context = trailing 512 minutes (registered E5 constant).
Comparators read from sim_results_des_revision.json (identical protocol).
Output: results/runs/revision_e5b_chronos_steady.json
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
os.environ.setdefault("HF_HOME", "/data/260715/.hf-cache")

from scripts.phase6_des import (  # noqa: E402
    decisions_from_rates, COLD_INIT, REDUCED_RHOS, run_config,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402

PROC = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
CHRONOS_MODEL = "amazon/chronos-bolt-small"
CONTEXT = 512
TAU_READ = min(newsvendor_quantile(10.0), 0.9)
SEEDS = list(range(10))


def main():
    counts_all = np.load(PROC / "counts.npy")
    splits = np.load(PROC / "splits.npz")
    dur_df = pd.read_csv(PROC / "duration_stats.csv")
    test_idx = splits["s3_test"]
    t0 = int(splits.get("s3_test_t_start", [10080])[0])
    seg = counts_all[test_idx]          # full trace; eval from t0
    n, T = seg.shape
    eval_counts = seg[:, t0:]
    Te = eval_counts.shape[1]
    print(f"S3: {n} funcs, {Te} eval ticks")

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        CHRONOS_MODEL, device_map="cuda", torch_dtype=torch.bfloat16)

    segf = seg.astype(np.float32)
    rates = np.zeros((n, Te), dtype=np.float32)
    t_start = time.time()
    with torch.no_grad():
        for w in range(Te):
            t = t0 + w
            ctx = torch.from_numpy(segf[:, t - CONTEXT:t])
            q, _ = pipe.predict_quantiles(ctx, prediction_length=1,
                                          quantile_levels=[TAU_READ])
            rates[:, w] = np.maximum(q[:, 0, 0].float().numpy(), 0.0)
            if (w + 1) % 1000 == 0:
                print(f"  {w+1}/{Te} ({time.time()-t_start:.0f}s)", flush=True)
    print(f"forecasts done in {time.time()-t_start:.0f}s")

    dm = np.nan_to_num(np.array(
        [dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
         for fi in test_idx]), nan=1.0)
    ds = np.nan_to_num(np.array(
        [dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
         for fi in test_idx]), nan=0.5)

    jobs = []
    for rho in REDUCED_RHOS:
        tau = newsvendor_quantile(rho)
        pw, ka = decisions_from_rates(rates, tau)
        jobs.append(("B9_chronos_bolt_small", rho, SEEDS, eval_counts, pw, ka,
                     dm, ds, COLD_INIT, "S3"))
    all_results = []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as ex:
        for res_list in ex.map(run_config, jobs):
            all_results.extend(res_list)
            r = res_list[0]
            print(f"  chronos rho={r['cost_ratio']:6.1f} "
                  f"CSR={np.mean([x['csr'] for x in res_list]):.4f} "
                  f"WM={np.mean([x['wm_per_1k_inv'] for x in res_list]):.0f}",
                  flush=True)

    camp = json.load(open(RUNS_DIR / "sim_results_des_revision.json"))["results"]
    comp = {}
    for m in ["A5_full_system", "B4a_ewma", "Oracle"]:
        comp[m] = {}
        for rho in REDUCED_RHOS:
            rows = [r for r in camp if r["split"] == "S3" and r["method"] == m
                    and abs(r["cost_ratio"] - rho) < 1e-9]
            if rows:
                comp[m][str(rho)] = {
                    "csr": float(np.mean([r["csr"] for r in rows])),
                    "wm": float(np.mean([r["wm_per_1k_inv"] for r in rows]))}
    out = {"model": CHRONOS_MODEL, "context": CONTEXT, "seeds": SEEDS,
           "results": all_results, "comparators_from_campaign": comp}
    path = RUNS_DIR / "revision_e5b_chronos_steady.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")
    for rho in REDUCED_RHOS:
        rs = [r for r in all_results if abs(r["cost_ratio"] - rho) < 1e-9]
        line = f"rho={rho}: chronos {np.mean([r['csr'] for r in rs])*100:.2f}%"
        for m in comp:
            if str(rho) in comp[m]:
                line += f" | {m} {comp[m][str(rho)]['csr']*100:.2f}%"
        print(line)


if __name__ == "__main__":
    main()
