#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Revision E5d: corrected native-grid Chronos control.

The archived E5/E5b Chronos arms read the 0.9 quantile as a scalar rate and
then pass that rate through the common Poisson decision layer. This script
keeps that fixed-read branch for reproducibility and adds the corrected
native-grid branch used in the manuscript revision:

  native count quantile at tau*(rho) -> ceil(q_tau) prewarm target
  native count quantile at tau*(rho) -> existing keep-alive block as rate proxy

Chronos-Bolt-small only exposes a useful grid up to 0.9 in the archived
environment, so tau reads are clamped to [0.1, 0.9]. The decision rule matches
revision_r2_directread.py for native quantile arms.

Output:
  results/runs/revision_e5d_chronos_native.json
  results/runs/revision_e5d_chronos_native_forecasts.npz
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

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("SF_DATA_DIR", "processed_2019")
os.environ.setdefault("HF_HOME", "/data/260715/.hf-cache")

from scripts.phase6_des import (  # noqa: E402
    COLD_INIT,
    REDUCED_RHOS,
    decisions_from_rates,
    run_config,
)
from scripts.phase63_onboarding_drift import DES_SEEDS  # noqa: E402
from scripts.revision_a4_crosstrace import find_natural_cohort  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.sim.des import rolling_csr, simulate_function  # noqa: E402
from src.sim.simulator import compute_adaptation_lag  # noqa: E402

PROC_2019 = PROJECT_ROOT / "data" / "processed_2019"
PROC_2021 = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
OUT_JSON = RUNS_DIR / "revision_e5d_chronos_native.json"
OUT_CACHE = RUNS_DIR / "revision_e5d_chronos_native_forecasts.npz"

MODEL = "amazon/chronos-bolt-small"
CONTEXT = 512
Q_LEVELS = np.array([0.1, 0.5, 0.9], dtype=np.float32)
TAU_FIXED = 0.9
COHORT_RHOS = [1.0, 10.0, 100.0]
STEADY_RHOS = REDUCED_RHOS
COHORT_W = 240
STEADY_SEEDS = list(range(10))
EPS_GRID = [0.05, 0.1, 0.2]


def _load_chronos():
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(
        MODEL, device_map="cuda", torch_dtype=torch.bfloat16
    )


def _quantile_at(qvals, tau):
    """Read qvals shaped [Q, N, T] at clamped tau."""
    tt = float(np.clip(tau, Q_LEVELS[0], Q_LEVELS[-1]))
    if tt <= Q_LEVELS[0]:
        return qvals[0], tt
    if tt >= Q_LEVELS[-1]:
        return qvals[-1], tt
    idx = int(np.searchsorted(Q_LEVELS, tt))
    lo, hi = idx - 1, idx
    frac = (tt - float(Q_LEVELS[lo])) / float(Q_LEVELS[hi] - Q_LEVELS[lo])
    return ((1.0 - frac) * qvals[lo] + frac * qvals[hi]).astype(np.float32), tt


def decisions_from_native_quantiles(qvals, tau):
    """Direct native-grid decision rule from revision_r2_directread.py."""
    counts_q, tau_read = _quantile_at(qvals, tau)
    prewarm = np.ceil(counts_q - 1e-9).astype(np.int32)
    _, keepalive = decisions_from_rates(counts_q.astype(np.float32), tau)
    return prewarm, keepalive, tau_read


def _forecast_batches(pipe, history, label):
    n, total = history.shape
    width = total - CONTEXT
    out = np.zeros((len(Q_LEVELS), n, width), dtype=np.float32)
    t_start = time.time()
    with torch.no_grad():
        for w in range(width):
            ctx = torch.from_numpy(history[:, w:w + CONTEXT])
            q, _ = pipe.predict_quantiles(
                ctx, prediction_length=1, quantile_levels=Q_LEVELS.tolist()
            )
            out[:, :, w] = np.maximum(q[:, 0, :].float().cpu().numpy(), 0.0).T
            if (w + 1) % 1000 == 0 or (label == "cohort" and (w + 1) % 60 == 0):
                print(f"  {label} forecast {w+1}/{width} "
                      f"({time.time()-t_start:.1f}s)", flush=True)
    return out, time.time() - t_start


def build_cohort_inputs():
    counts = np.load(PROC_2019 / "counts.npy", mmap_mode="r")
    dur_df = pd.read_csv(PROC_2019 / "duration_stats.csv")
    pts = find_natural_cohort(counts)
    seg_counts = np.stack([
        np.asarray(counts[fi, t0:t0 + COHORT_W], dtype=np.int64)
        for fi, t0 in pts
    ])
    history = np.stack([
        np.concatenate([
            np.zeros(max(0, CONTEXT - t0), dtype=np.float32),
            np.asarray(counts[fi, max(0, t0 - CONTEXT):t0 + COHORT_W],
                       dtype=np.float32),
        ])
        for fi, t0 in pts
    ])
    dur_mean = np.nan_to_num(
        np.array([dur_df.iloc[fi]["dur_mean"] for fi, _ in pts]), nan=1.0
    )
    dur_std = np.nan_to_num(
        np.array([dur_df.iloc[fi]["dur_std"] for fi, _ in pts]), nan=0.5
    )
    return pts, seg_counts, history, dur_mean, dur_std


def build_steady_inputs():
    counts_all = np.load(PROC_2021 / "counts.npy")
    splits = np.load(PROC_2021 / "splits.npz")
    dur_df = pd.read_csv(PROC_2021 / "duration_stats.csv")
    test_idx = splits["s3_test"]
    t0 = int(splits.get("s3_test_t_start", [10080])[0])
    seg = counts_all[test_idx]
    eval_counts = seg[:, t0:]
    history = seg[:, t0 - CONTEXT:].astype(np.float32)
    dur_mean = np.nan_to_num(
        np.array([
            dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
            for fi in test_idx
        ]),
        nan=1.0,
    )
    dur_std = np.nan_to_num(
        np.array([
            dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
            for fi in test_idx
        ]),
        nan=0.5,
    )
    return test_idx, int(t0), eval_counts, history, dur_mean, dur_std


def load_or_compute_forecasts(parts):
    cache = {}
    if OUT_CACHE.exists():
        data = np.load(OUT_CACHE, allow_pickle=True)
        cache = {k: data[k] for k in data.files}

    need_cohort = "cohort" in parts and "cohort_q" not in cache
    need_steady = "steady" in parts and "steady_q" not in cache
    timings = {}
    if need_cohort or need_steady:
        pipe = _load_chronos()
        if need_cohort:
            _, _, history, _, _ = build_cohort_inputs()
            cache["cohort_q"], timings["cohort_forecast_sec"] = _forecast_batches(
                pipe, history, "cohort"
            )
        if need_steady:
            _, _, _, history, _, _ = build_steady_inputs()
            cache["steady_q"], timings["steady_forecast_sec"] = _forecast_batches(
                pipe, history, "steady"
            )
        cache["q_levels"] = Q_LEVELS
        cache["model"] = np.array(MODEL)
        np.savez_compressed(OUT_CACHE, **cache)
        print(f"Saved forecast cache {OUT_CACHE}", flush=True)
    return cache, timings


def eval_cohort_mode(name, seg_counts, dur_mean, dur_std, prewarm, keepalive):
    n, width = seg_counts.shape
    roll_c = np.zeros(width)
    roll_n = np.zeros(width)
    func_cold60 = np.zeros(n)
    func_cold_all = np.zeros(n)
    func_wm = np.zeros(n)
    for f in range(n):
        for seed in DES_SEEDS:
            r = simulate_function(
                seg_counts[f], prewarm[f], keepalive[f],
                float(dur_mean[f]), float(dur_std[f]),
                seed=seed * 7919 + f,
                cold_mu=COLD_INIT["mu"], cold_sigma=COLD_INIT["sigma"],
                track_rolling=True,
            )
            roll_c += r["roll_cold"]
            roll_n += r["roll_total"]
            func_cold60[f] += r["roll_cold"][:60].sum()
            func_cold_all[f] += r["roll_cold"].sum()
            func_wm[f] += r["idle_mem_gb_s"]
    func_cold60 /= len(DES_SEEDS)
    func_cold_all /= len(DES_SEEDS)
    func_wm /= len(DES_SEEDS)
    curve = rolling_csr(roll_c, roll_n, window=15)
    pairs = [(t, v) for t, v in enumerate(curve) if not np.isnan(v)]
    return {
        "method": name,
        "overall_csr": float(roll_c.sum() / max(roll_n.sum(), 1e-9)),
        "wm_total": float(func_wm.sum()),
        "cold_all_total": float(func_cold_all.sum()),
        "cold60_total": float(func_cold60.sum()),
        "func_cold_all": func_cold_all.tolist(),
        "func_cold60": func_cold60.tolist(),
        "rolling_csr": [None if np.isnan(x) else float(x) for x in curve],
        "al": {
            str(eps): (
                None if compute_adaptation_lag(pairs, event_tick=0,
                                               epsilon=eps) is None
                else float(compute_adaptation_lag(pairs, event_tick=0,
                                                  epsilon=eps))
            )
            for eps in EPS_GRID
        },
    }


def run_cohort(qvals):
    pts, seg_counts, _, dur_mean, dur_std = build_cohort_inputs()
    ref = json.load(open(RUNS_DIR / "revision_e5_chronos.json"))
    a4 = json.load(open(RUNS_DIR / "revision_a4_crosstrace.json"))
    out = {
        "n_functions": len(pts),
        "window_min": COHORT_W,
        "des_seeds": DES_SEEDS,
        "by_rho": {},
    }
    fixed_rates, _ = _quantile_at(qvals, TAU_FIXED)
    from scipy import stats as sps

    for rho in COHORT_RHOS:
        tau = newsvendor_quantile(rho)
        fixed_pw, fixed_ka = decisions_from_rates(fixed_rates, tau)
        native_pw, native_ka, native_tau_read = decisions_from_native_quantiles(
            qvals, tau
        )
        rec = {
            "tau_star": float(tau),
            "fixed_tau_read": TAU_FIXED,
            "native_tau_read": native_tau_read,
            "fixed_read": eval_cohort_mode(
                "B9_chronos_fixed_read", seg_counts, dur_mean, dur_std,
                fixed_pw, fixed_ka
            ),
            "native_direct": eval_cohort_mode(
                "B9_chronos_native_direct", seg_counts, dur_mean, dur_std,
                native_pw, native_ka
            ),
        }
        old_csr = ref["by_rho"][str(rho)]["overall_csr"]
        assert abs(rec["fixed_read"]["overall_csr"] - old_csr) < 5e-5, (
            rho, rec["fixed_read"]["overall_csr"], old_csr
        )
        paired = {}
        native_cold = np.array(rec["native_direct"]["func_cold_all"])
        for arm in ["A5_proto", "A5_proto_zero", "B4a_ewma"]:
            base = np.array(a4["by_rho"][str(rho)][arm]["func_cold_all"])
            diff = base - native_cold
            nz = diff[diff != 0]
            paired[arm] = {
                "ref_csr": a4["by_rho"][str(rho)][arm]["overall_csr"],
                "ref_wm_total": a4["by_rho"][str(rho)][arm]["wm_total"],
                "mean_cold_diff_ref_minus_native": float(diff.mean()),
                "wilcoxon_p": float(sps.wilcoxon(nz)[1]) if len(nz) >= 6 else 1.0,
            }
        rec["native_paired_vs"] = paired
        out["by_rho"][str(rho)] = rec
        print("cohort rho={}: fixed {:.4f}% native {:.4f}%".format(
            rho,
            100.0 * rec["fixed_read"]["overall_csr"],
            100.0 * rec["native_direct"]["overall_csr"],
        ), flush=True)
    return out


def summarize_rows(rows):
    return {
        "csr": float(np.mean([r["csr"] for r in rows])),
        "wm_per_1k_inv": float(np.mean([r["wm_per_1k_inv"] for r in rows])),
        "cold_starts": float(np.mean([r["cold_starts"] for r in rows])),
        "results": rows,
    }


def run_steady(qvals):
    test_idx, t0, eval_counts, _, dur_mean, dur_std = build_steady_inputs()
    old = json.load(open(RUNS_DIR / "revision_e5b_chronos_steady.json"))["results"]
    out = {
        "split": "S3",
        "n_functions": int(len(test_idx)),
        "eval_start_tick": int(t0),
        "eval_ticks": int(eval_counts.shape[1]),
        "des_seeds": STEADY_SEEDS,
        "by_rho": {},
    }
    fixed_rates, _ = _quantile_at(qvals, TAU_FIXED)
    jobs = []
    job_meta = []
    for rho in STEADY_RHOS:
        tau = newsvendor_quantile(rho)
        fixed_pw, fixed_ka = decisions_from_rates(fixed_rates, tau)
        native_pw, native_ka, native_tau_read = decisions_from_native_quantiles(
            qvals, tau
        )
        jobs.append(("B9_chronos_fixed_read", rho, STEADY_SEEDS, eval_counts,
                     fixed_pw, fixed_ka, dur_mean, dur_std, COLD_INIT, "S3"))
        job_meta.append((rho, "fixed_read", TAU_FIXED))
        jobs.append(("B9_chronos_native_direct", rho, STEADY_SEEDS, eval_counts,
                     native_pw, native_ka, dur_mean, dur_std, COLD_INIT, "S3"))
        job_meta.append((rho, "native_direct", native_tau_read))

    with ProcessPoolExecutor(max_workers=4) as ex:
        for meta, rows in zip(job_meta, ex.map(run_config, jobs)):
            rho, branch, tau_read = meta
            rec = summarize_rows(rows)
            rec["tau_read"] = float(tau_read)
            out["by_rho"].setdefault(str(rho), {
                "tau_star": float(newsvendor_quantile(rho))
            })[branch] = rec

    for rho in STEADY_RHOS:
        fixed = out["by_rho"][str(rho)]["fixed_read"]
        old_rows = [r for r in old if abs(r["cost_ratio"] - rho) < 1e-9]
        old_csr = float(np.mean([r["csr"] for r in old_rows]))
        assert abs(fixed["csr"] - old_csr) < 5e-5, (rho, fixed["csr"], old_csr)
        native = out["by_rho"][str(rho)]["native_direct"]
        print("steady rho={}: fixed {:.4f}% native {:.4f}%".format(
            rho, 100.0 * fixed["csr"], 100.0 * native["csr"]
        ), flush=True)
    return out


def merge_output(new_data):
    if OUT_JSON.exists():
        out = json.load(open(OUT_JSON))
    else:
        out = {
            "model": MODEL,
            "context": CONTEXT,
            "quantile_levels": Q_LEVELS.tolist(),
            "clamp": [float(Q_LEVELS[0]), float(Q_LEVELS[-1])],
            "native_decision_rule": (
                "read clamped native count quantile at tau*(rho), "
                "ceil for prewarm, existing keep-alive block with q_tau "
                "as scalar rate proxy"
            ),
            "fixed_read_reproduces": [
                "revision_e5_chronos.json",
                "revision_e5b_chronos_steady.json",
            ],
        }
    out.update(new_data)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, default=float)
    blob = open(OUT_JSON, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(OUT_JSON))
    print(f"Saved + verified {OUT_JSON}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["all", "cohort", "steady"], default="all")
    args = ap.parse_args()

    parts = ["cohort", "steady"] if args.part == "all" else [args.part]
    cache, timings = load_or_compute_forecasts(parts)
    update = {"inference": timings}
    if "cohort" in parts:
        update["cohort"] = run_cohort(cache["cohort_q"])
    if "steady" in parts:
        update["steady"] = run_steady(cache["steady_q"])
    merge_output(update)


if __name__ == "__main__":
    main()
