# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R4: intermediate-size Chronos-Bolt points (pre-registered in
revision_r4_PREREG.md). Protocol is a faithful parameterization of
revision_e5_chronos.py (cohort) and revision_e5b_chronos_steady.py
(S3 steady) — only the model checkpoint varies.

Torch env:
  PYTHONPATH=/data/260715/site-packages:. python3.13 \
      scripts/revision_r4_chronos_scale.py [--models tiny,mini,small]
Output: results/runs/revision_r4_chronos_scale.json
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HOME", "/data/260715/.hf-cache")

from scripts.revision_a4_crosstrace import find_natural_cohort  # noqa: E402
from scripts.phase63_onboarding_drift import DES_SEEDS  # noqa: E402
from scripts.phase6_des import (  # noqa: E402
    decisions_from_rates, COLD_INIT, REDUCED_RHOS, run_config,
)
from src.sim.des import simulate_function  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402

torch.set_num_threads(1)
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
P19 = PROJECT_ROOT / "data" / "processed_2019"
P21 = PROJECT_ROOT / "data" / "processed"
W = 240
COHORT_RHOS = [1.0, 10.0, 100.0]
CONTEXT = 512
TAU_READ = min(newsvendor_quantile(10.0), 0.9)
STEADY_SEEDS = list(range(10))
MODELS = {"tiny": "amazon/chronos-bolt-tiny",
          "mini": "amazon/chronos-bolt-mini",
          "small": "amazon/chronos-bolt-small"}
OUT = RUNS_DIR / "revision_r4_chronos_scale.json"


def load_pipe(model_id):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in pipe.model.parameters())
    return pipe, n_params


def cohort_eval(pipe):
    """e5 protocol (revision_e5_chronos.py:64-150), model swapped."""
    counts = np.load(P19 / "counts.npy", mmap_mode="r")
    dur_df = pd.read_csv(P19 / "duration_stats.csv")
    pts = find_natural_cohort(counts)
    F_n = len(pts)
    seg_counts = np.stack([np.asarray(counts[fi, t0:t0 + W], dtype=np.int64)
                           for (fi, t0) in pts])
    hist = np.stack([
        np.concatenate([
            np.zeros(max(0, CONTEXT - t0), dtype=np.float32),
            np.asarray(counts[fi, max(0, t0 - CONTEXT):t0 + W],
                       dtype=np.float32)])
        for (fi, t0) in pts])
    rates = np.zeros((F_n, W), dtype=np.float32)
    per_batch = []
    with torch.no_grad():
        for w in range(W):
            ctx = torch.from_numpy(hist[:, w:w + CONTEXT])
            tb = time.time()
            q, _ = pipe.predict_quantiles(ctx, prediction_length=1,
                                          quantile_levels=[TAU_READ])
            per_batch.append(time.time() - tb)
            rates[:, w] = np.maximum(q[:, 0, 0].float().numpy(), 0.0)
    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] for (fi, _) in pts]), nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] for (fi, _) in pts]), nan=0.5)
    ref = json.load(open(RUNS_DIR / "revision_a4_crosstrace.json"))
    from scipy import stats as sps
    by_rho = {}
    for rho in COHORT_RHOS:
        tau = newsvendor_quantile(rho)
        pw_m, ka_m = decisions_from_rates(rates, tau)
        roll_c = roll_n = 0.0
        func_cold = np.zeros(F_n); func_wm = np.zeros(F_n)
        for f in range(F_n):
            for seed in DES_SEEDS:
                r = simulate_function(seg_counts[f], pw_m[f], ka_m[f],
                                      float(dm[f]), float(dstd[f]),
                                      seed=seed * 7919 + f,
                                      cold_mu=COLD_INIT["mu"],
                                      cold_sigma=COLD_INIT["sigma"],
                                      track_rolling=True)
                roll_c += r["roll_cold"].sum(); roll_n += r["roll_total"].sum()
                func_cold[f] += r["roll_cold"].sum()
                func_wm[f] += r["idle_mem_gb_s"]
        func_cold /= len(DES_SEEDS); func_wm /= len(DES_SEEDS)
        paired = {}
        for arm in ["A5_proto", "A5_proto_zero", "B4a_ewma"]:
            base = np.array(ref["by_rho"][str(rho)][arm]["func_cold_all"])
            diff = base - func_cold
            nz = diff[diff != 0]
            p = float(sps.wilcoxon(nz)[1]) if len(nz) >= 6 else 1.0
            paired[arm] = {"wilcoxon_p": p, "mean_diff": float(diff.mean()),
                           "ref_csr": ref["by_rho"][str(rho)][arm]["overall_csr"]}
        by_rho[str(rho)] = {"overall_csr": float(roll_c / max(roll_n, 1e-9)),
                            "wm_total": float(func_wm.sum()),
                            "paired_vs": paired}
        print(f"    cohort rho={rho}: CSR={by_rho[str(rho)]['overall_csr']*100:.3f}%",
              flush=True)
    return by_rho, {"mean_batch100_sec": float(np.mean(per_batch))}


def steady_eval(pipe, tag):
    """e5b protocol (revision_e5b_chronos_steady.py:46-95), model swapped."""
    counts_all = np.load(P21 / "counts.npy")
    splits = np.load(P21 / "splits.npz")
    dur_df = pd.read_csv(P21 / "duration_stats.csv")
    test_idx = splits["s3_test"]
    t0 = int(splits.get("s3_test_t_start", [10080])[0])
    seg = counts_all[test_idx]
    n = seg.shape[0]
    eval_counts = seg[:, t0:]
    Te = eval_counts.shape[1]
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
            if (w + 1) % 2000 == 0:
                print(f"    steady {w+1}/{Te} ({time.time()-t_start:.0f}s)",
                      flush=True)
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
        jobs.append((f"B9_{tag}", rho, STEADY_SEEDS, eval_counts, pw, ka,
                     dm, ds, COLD_INIT, "S3"))
    results = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for res_list in ex.map(run_config, jobs):
            for r in res_list:
                r.pop("func_csr", None)
            results.extend(res_list)
            r = res_list[0]
            print(f"    steady rho={r['cost_ratio']:6.1f} "
                  f"CSR={np.mean([x['csr'] for x in res_list])*100:.3f}% "
                  f"WM={np.mean([x['wm_per_1k_inv'] for x in res_list]):.0f}",
                  flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tiny,mini,small")
    args = ap.parse_args()
    out = json.load(open(OUT)) if OUT.exists() else {
        "prereg": "revision_r4_PREREG.md", "context": CONTEXT,
        "tau_read": TAU_READ, "models": {}}
    for tag in args.models.split(","):
        model_id = MODELS[tag]
        print(f"### {model_id}", flush=True)
        pipe, n_params = load_pipe(model_id)
        rec = {"model": model_id, "params": int(n_params),
               "state_bytes_bf16": int(n_params * 2)}
        by_rho, infer = cohort_eval(pipe)
        rec["cohort_by_rho"] = by_rho
        rec["inference"] = infer
        rec["inference"]["amortized_us_per_func_per_tick"] = (
            infer["mean_batch100_sec"] / 100 * 1e6)
        if tag == "small":
            e5 = json.load(open(RUNS_DIR / "revision_e5_chronos.json"))
            rec["reproduction_gate"] = {
                str(rho): {
                    "ours": by_rho[str(rho)]["overall_csr"],
                    "e5": e5["by_rho"][str(rho)]["overall_csr"],
                    "delta_pp": abs(by_rho[str(rho)]["overall_csr"]
                                    - e5["by_rho"][str(rho)]["overall_csr"]) * 100}
                for rho in COHORT_RHOS}
            print("    gate:", json.dumps(
                {k: round(v["delta_pp"], 5)
                 for k, v in rec["reproduction_gate"].items()}), flush=True)
        else:
            rec["steady_results"] = steady_eval(pipe, tag)
        out["models"][tag] = rec
        del pipe
        torch.cuda.empty_cache()
        json.dump(out, open(OUT, "w"), default=float)
        print(f"  saved {tag}", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
