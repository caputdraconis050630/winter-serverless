# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R5b: duty-calibrated read on the Huawei pools (pre-registered in
revision_r5b_PREREG.md — the design decisions live there).

  --export   (torch env)  recompute A5 rates, calibrate tau_eff per
             (pool, rho) so the binary gate's duty matches EWMA's archived
             duty, export A5_duty_matched__rho*.npz job trees
  --report   (any env, after des_runner_fast) join CSR + f8 triple

Runner stage (numba env):
  PYTHONPATH=/data/260715/site-packages-des:. python3.13 \
      scripts/des_runner_fast.py --jobs results/runs/des_jobs_r5b \
      --out results/runs/revision_r5b_des.json
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("SF_DATA_DIR", "processed_huawei")

RUNS_DIR = PROJECT_ROOT / "results" / "runs"
JOBS_ROOT = RUNS_DIR / "des_jobs_r5b"
POOLS = ["h_mixed", "h_sparse"]
RHOS = [1.0, 10.0, 100.0]
SEEDS = [0, 1, 2]
OUT = RUNS_DIR / "revision_r5b_dutymatch.json"

from scripts.phase6_des import (  # noqa: E402
    decisions_from_rates, rates_ewma, rates_a5, COLD_INIT,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402


def duty_of(rates, tau):
    """P(binary prewarm order) = P(1-exp(-rate) > 1-tau)."""
    p_arr = 1.0 - np.exp(-np.maximum(rates, 0.0))
    return float((p_arr > (1.0 - tau)).mean())


def calibrate_tau(rates, target_duty):
    """Bisect tau_eff so duty matches target (monotone). Returns
    (tau_eff, achieved, attainable)."""
    dmax = duty_of(rates, 1.0 - 1e-12)
    if dmax < target_duty - 1e-6:
        return 1.0 - 1e-12, dmax, False
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if duty_of(rates, mid) < target_duty:
            lo = mid
        else:
            hi = mid
    return hi, duty_of(rates, hi), True


def decisions_duty_matched(rates, tau_star, tau_eff):
    """Binary activity gate at tau_eff; Poisson concurrency sizing and
    keep-alive at the true tau_star (PREREG decision 2)."""
    pw_star, ka_star = decisions_from_rates(rates, tau_star)
    p_arr = 1.0 - np.exp(-np.maximum(rates, 0.0))
    gate_eff = (p_arr > (1.0 - tau_eff)).astype(np.int32)
    hi = rates > 2.0
    pw = gate_eff.copy()
    pw[hi] = np.maximum(pw[hi], pw_star[hi])   # keep tau*-priced sizing
    return pw, ka_star


def export():
    import pandas as pd
    import torch
    from src.meta.trainer import ANILMetaTrainer
    from src.models.heads import N_QUANTILES
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    PROCESSED_DIR = PROJECT_ROOT / "data" / os.environ["SF_DATA_DIR"]
    features_mm = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts_mm = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    splits = np.load(PROCESSED_DIR / "splits.npz")
    steady_t = int(splits["steady_t"][0])

    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features_mm.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1, device=device)
    ckpt = PROJECT_ROOT / "results_azure2021" / "runs" / "best_anil_ridge_s2_s0.pt"
    trainer.load(ckpt)
    print(f"checkpoint: {ckpt}", flush=True)

    f8 = json.load(open(RUNS_DIR / "revision_f8_sparse_duty.json"))["pools"]
    manifest = json.load(open(RUNS_DIR / "des_huawei_manifest.json"))

    calib = {}
    for pool in POOLS:
        rows = np.asarray(manifest["pools"][pool]["rows"])
        counts = np.asarray(counts_mm[rows][:, :steady_t]).astype(np.float32)
        feats = np.asarray(features_mm[rows][:, :steady_t, :], dtype=np.float32)
        print(f"### {pool}: {len(rows)} funcs", flush=True)
        t0 = time.time()
        a5 = rates_a5(counts, feats, trainer, device)
        print(f"    a5 rates {time.time()-t0:.0f}s", flush=True)

        # gate: our EWMA duty must reproduce f8's archived EWMA duty
        ew = rates_ewma(counts)
        jdir = JOBS_ROOT / pool
        jdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(RUNS_DIR / "des_jobs_huawei" / pool / "shared.npz",
                    jdir / "shared.npz")

        job_meta = []
        calib[pool] = {}
        for rho in RHOS:
            tau_star = newsvendor_quantile(rho)
            target = f8[pool]["by_rho"][str(rho)]["B4a_ewma"]["duty"]
            ew_pw, _ = decisions_from_rates(ew, tau_star)
            ew_duty = float((ew_pw > 0).mean())
            tau_eff, achieved, attainable = calibrate_tau(a5, target)
            pw, ka = decisions_duty_matched(a5, tau_star, tau_eff)
            fname = f"A5_duty_matched__rho{rho}.npz"
            np.savez_compressed(jdir / fname, prewarm=pw, keepalive=ka)
            job_meta.append({"method": "A5_duty_matched", "rho": rho, "file": fname})
            calib[pool][str(rho)] = {
                "target_duty_f8_ewma": target,
                "our_ewma_duty_check": ew_duty,
                "ewma_duty_gate_ok": bool(abs(ew_duty - target) < 1e-4),
                "tau_star": tau_star, "tau_eff": tau_eff,
                "achieved_duty": achieved, "attainable": attainable,
                "a5_native_duty": duty_of(a5, tau_star),
            }
            print(f"    rho={rho}: target={target:.4f} tau*={tau_star:.3f} "
                  f"tau_eff={tau_eff:.6f} achieved={achieved:.4f} "
                  f"(native {calib[pool][str(rho)]['a5_native_duty']:.4f}) "
                  f"ewma_gate={'OK' if calib[pool][str(rho)]['ewma_duty_gate_ok'] else 'FAIL'}",
                  flush=True)
        json.dump({"split": pool, "seeds": SEEDS, "cold_init": COLD_INIT,
                   "jobs": job_meta}, open(jdir / "jobs.json", "w"))

    out = {"prereg": "revision_r5b_PREREG.md", "checkpoint": str(ckpt),
           "seeds": SEEDS, "calibration": calib}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} (export stage)")


def report():
    res = json.load(open(RUNS_DIR / "revision_r5b_des.json"))
    rows = res["results"] if isinstance(res, dict) and "results" in res else res
    out = json.load(open(OUT))
    f8 = json.load(open(RUNS_DIR / "revision_f8_sparse_duty.json"))["pools"]
    out["by_pool"] = {}
    for pool in POOLS:
        ref = json.load(open(RUNS_DIR / f"revision_h2_huawei_des_{pool}.json"))
        ref_rows = ref["results"] if isinstance(ref, dict) and "results" in ref else ref
        shared = np.load(JOBS_ROOT / pool / "shared.npz")
        arrival = shared["counts"] > 0
        pool_out = {}
        for rho in RHOS:
            ours = [r["csr"] for r in rows
                    if r["split"] == pool and r["cost_ratio"] == rho]
            a5_ref = [r["csr"] for r in ref_rows
                      if r["method"] == "A5_full_system" and r["cost_ratio"] == rho]
            ew_ref = [r["csr"] for r in ref_rows
                      if r["method"] == "B4a_ewma" and r["cost_ratio"] == rho]
            pw = np.load(JOBS_ROOT / pool / f"A5_duty_matched__rho{rho}.npz")["prewarm"] > 0
            duty = float(pw.mean()); recall = float(pw[arrival].mean())
            gap_native = np.mean(a5_ref) - np.mean(ew_ref)
            gap_matched = np.mean(ours) - np.mean(ew_ref)
            pool_out[str(rho)] = {
                "csr_duty_matched_pct": float(np.mean(ours) * 100),
                "csr_a5_native_pct": float(np.mean(a5_ref) * 100),
                "csr_ewma_pct": float(np.mean(ew_ref) * 100),
                "gap_native_pp": float(gap_native * 100),
                "gap_duty_matched_pp": float(gap_matched * 100),
                "gap_closed_frac": float(1 - gap_matched / gap_native) if gap_native else None,
                "duty": duty, "recall": recall,
                "lift": recall / duty if duty else None,
                "precision": float(arrival[pw].mean()) if pw.any() else None,
                "ewma_lift_f8": f8[pool]["by_rho"][str(rho)]["B4a_ewma"]["lift"],
            }
            print(f"{pool} rho={rho}: native gap {gap_native*100:+.2f}pp -> "
                  f"matched gap {gap_matched*100:+.2f}pp "
                  f"(closed {pool_out[str(rho)]['gap_closed_frac']:.0%}); "
                  f"lift {pool_out[str(rho)]['lift']:.2f} vs EWMA {pool_out[str(rho)]['ewma_lift_f8']:.2f}",
                  flush=True)
        out["by_pool"][pool] = pool_out
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} (report stage)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.export:
        export()
    elif args.report:
        report()
    else:
        ap.error("pass --export or --report")
