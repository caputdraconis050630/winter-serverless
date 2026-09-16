# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision E5: zero-shot time-series foundation model baseline on the
natural onboarding cohort (Azure-2019), through the SAME decision layer.

Model: amazon/chronos-bolt-small (zero-shot, no fine-tuning), bf16 on GPU.
Cohort/protocol identical to revision_a4_crosstrace.py (100 functions,
seed 42, W=240, DES seeds {0,1,2}, rhos {1,10,100}).

Rate semantics (decision-layer parity): every arm in this evaluation feeds
the shared newsvendor layer a per-tick rate equal to its predicted
TAU-fractile of next-minute arrivals, TAU = newsvendor_quantile(10) = 0.909
(the registered fixed rate-fractile of phase63/A4). Chronos-Bolt outputs
quantiles on the 0.1..0.9 grid; the 0.909 read is clamped to its native
maximum 0.9 (documented; same spirit as the registered clamp rule for our
0.05..0.95 head). Context: the platform-visible count history — trailing
CHRONOS_CONTEXT=512 minutes ending at the decision tick, zero-padded
(pre-onset silence is genuine platform knowledge, identical to what the
other arms see).

Registered pre-fixed constants (no tuning): CHRONOS_MODEL=chronos-bolt-small,
CHRONOS_CONTEXT=512, quantile clamp 0.9.

Comparison arms are read from revision_a4_crosstrace.json (identical
functions, seeds, decision layer) for paired per-function statistics.
Also records measured inference cost per 100-function tick batch and the
parameter count (platform-economics argument).
Output: results/runs/revision_e5_chronos.json
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
from src.sim.des import simulate_function  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_2019"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
W = 240
RHOS = [1.0, 10.0, 100.0]
CHRONOS_MODEL = "amazon/chronos-bolt-small"
CHRONOS_CONTEXT = 512
TAU = newsvendor_quantile(10.0)          # 0.909...
TAU_READ = min(TAU, 0.9)                 # Bolt native quantile cap


def main():
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    pts = find_natural_cohort(counts)
    F_n = len(pts)
    print(f"cohort: {F_n} functions")

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        CHRONOS_MODEL, device_map="cuda", torch_dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in pipe.model.parameters())
    print(f"model params: {n_params/1e6:.1f}M")

    seg_counts = np.stack([np.asarray(counts[fi, t0:t0 + W], dtype=np.int64)
                           for (fi, t0) in pts])
    # full context source: [t0-CONTEXT, t0+W) per function (pre-t0 is zero by
    # cohort construction, but fetch the real trace values anyway)
    hist = np.stack([
        np.concatenate([
            np.zeros(max(0, CHRONOS_CONTEXT - t0), dtype=np.float32),
            np.asarray(counts[fi, max(0, t0 - CHRONOS_CONTEXT):t0 + W],
                       dtype=np.float32)])
        for (fi, t0) in pts])                    # [F, CONTEXT + W]

    rates = np.zeros((F_n, W), dtype=np.float32)
    t_start = time.time()
    per_batch = []
    with torch.no_grad():
        for w in range(W):
            ctx = torch.from_numpy(hist[:, w:w + CHRONOS_CONTEXT])
            tb = time.time()
            q, _ = pipe.predict_quantiles(ctx, prediction_length=1,
                                          quantile_levels=[TAU_READ])
            per_batch.append(time.time() - tb)
            rates[:, w] = np.maximum(q[:, 0, 0].float().numpy(), 0.0)
            if (w + 1) % 60 == 0:
                print(f"  chronos {w+1}/{W} ({time.time()-t_start:.0f}s)",
                      flush=True)
    infer_stats = {"mean_batch100_sec": float(np.mean(per_batch)),
                   "total_sec": float(time.time() - t_start),
                   "params": int(n_params)}
    print("inference stats:", infer_stats)

    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] for (fi, _) in pts]), nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] for (fi, _) in pts]), nan=0.5)

    ref = json.load(open(RUNS_DIR / "revision_a4_crosstrace.json"))
    from scipy import stats as sps
    out = {"model": CHRONOS_MODEL, "context": CHRONOS_CONTEXT,
           "tau_read": TAU_READ, "n_functions": F_n,
           "inference": infer_stats, "by_rho": {}}
    for rho in RHOS:
        tau = newsvendor_quantile(rho)
        pw_m, ka_m = decisions_from_rates(rates, tau)
        roll_c = 0.0; roll_n = 0.0
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
        rec = {"overall_csr": float(roll_c / max(roll_n, 1e-9)),
               "func_cold_all": func_cold.tolist(),
               "wm_total": float(func_wm.sum())}
        paired = {}
        for arm in ["A5_proto", "A5_proto_zero", "B4a_ewma"]:
            base = np.array(ref["by_rho"][str(rho)][arm]["func_cold_all"])
            diff = base - func_cold          # >0: chronos better
            nz = diff[diff != 0]
            p = float(sps.wilcoxon(nz)[1]) if len(nz) >= 6 else 1.0
            paired[arm] = {"wilcoxon_p": p, "mean_diff": float(diff.mean()),
                           "ref_csr": ref["by_rho"][str(rho)][arm]["overall_csr"]}
        rec["paired_vs"] = paired
        out["by_rho"][str(rho)] = rec
        print(f"rho={rho}: chronos CSR={rec['overall_csr']*100:.2f}% "
              f"WM={rec['wm_total']:.0f} | " +
              " ".join(f"{a}:{v['ref_csr']*100:.2f}%(p={v['wilcoxon_p']:.1e},"
                       f"d={v['mean_diff']:+.1f})" for a, v in paired.items()),
              flush=True)

    path = RUNS_DIR / "revision_e5_chronos.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")


if __name__ == "__main__":
    main()
