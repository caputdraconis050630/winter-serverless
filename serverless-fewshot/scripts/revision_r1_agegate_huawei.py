# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R1: the age-qualified gate, evaluated on the Huawei pools as published
(pre-registered in revision_r1_PREREG.md — constants unchanged).

  --export   (torch env)  compose gate rates, export job trees
  --report   (any env, after des_runner_fast) join CSR + bootstrap CIs

Runner stage (numba env):
  PYTHONPATH=/data/260715/site-packages-des:. python3.13 \
      scripts/des_runner_fast.py --jobs results/runs/des_jobs_r1 \
      --out results/runs/revision_r1_des.json
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
JOBS_ROOT = RUNS_DIR / "des_jobs_r1"
POOLS = ["h_mixed", "h_sparse", "h_saturated"]
RHOS = [1.0, 10.0, 100.0]
SEEDS = [0, 1, 2]
AGE_MIN = 720          # registered constant, NOT refit
GATE_THRESHOLD = 100   # registered constant, NOT refit
BOOT_SEED = 260715
OUT = RUNS_DIR / "revision_r1_agegate_huawei.json"


def export():
    import torch
    from scripts.phase6_des import (
        decisions_from_rates, rates_ewma, rates_a5, rates_fourier, COLD_INIT)
    from scripts.revision_a2_des import rates_proto_prefix
    from scripts.phase63_onboarding_drift import build_prototypes
    from src.decision.newsvendor import newsvendor_quantile
    from src.meta.trainer import ANILMetaTrainer
    from src.models.heads import N_QUANTILES
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    P_HW = PROJECT_ROOT / "data" / "processed_huawei"
    P_21 = PROJECT_ROOT / "data" / "processed"
    features_mm = np.load(P_HW / "features.npy", mmap_mode="r")
    counts_mm = np.load(P_HW / "counts.npy", mmap_mode="r")
    splits = np.load(P_HW / "splits.npz")
    first_present = np.load(P_HW / "first_present.npy")
    steady_t = int(splits["steady_t"][0])

    ckpt = PROJECT_ROOT / "results_azure2021" / "runs" / "best_anil_ridge_s2_s0.pt"
    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features_mm.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1, device=device)
    trainer.load(ckpt)
    print(f"checkpoint: {ckpt}", flush=True)

    feat21 = np.load(P_21 / "features.npy")
    cnt21 = np.load(P_21 / "counts.npy")
    spl21 = np.load(P_21 / "splits.npz")
    pm = build_prototypes(trainer, feat21, cnt21, spl21["s2_train"])

    manifest = json.load(open(RUNS_DIR / "des_huawei_manifest.json"))
    out = {"prereg": "revision_r1_PREREG.md", "checkpoint": str(ckpt),
           "proto_source": "2021 s2_train (16 clusters, phase63 builder)",
           "age_min": AGE_MIN, "gate_threshold": GATE_THRESHOLD,
           "age_source": "first_present.npy (deployment age)",
           "seeds": SEEDS, "pools": {}}

    for pool in POOLS:
        rows = np.asarray(manifest["pools"][pool]["rows"])
        counts = np.asarray(counts_mm[rows][:, :steady_t]).astype(np.float32)
        feats = np.asarray(features_mm[rows][:, :steady_t, :], dtype=np.float32)
        fp = first_present[rows].astype(np.int64)
        N, T = counts.shape
        print(f"### {pool}: {N} funcs", flush=True)

        stage = time.time()
        ew = rates_ewma(counts)
        a5 = rates_a5(counts, feats, trainer, device)
        b3 = rates_fourier(counts)
        proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
        print(f"    rates done in {time.time()-stage:.0f}s", flush=True)

        ticks = np.arange(T)[None, :]
        age_fp = ticks - fp[:, None]                       # deployment age
        first_arr = np.where(counts.sum(axis=1) > 0,
                             (counts > 0).argmax(axis=1), T)
        age_fa = ticks - first_arr[:, None]                # first-arrival age

        zero_h = cum_prev == 0
        mid = (cum_prev > 0) & (cum_prev < GATE_THRESHOLD)
        conv = cum_prev >= GATE_THRESHOLD
        conv_young = conv & (age_fp < AGE_MIN)
        conv_old = conv & (age_fp >= AGE_MIN)

        gated_aq = np.where(zero_h, proto,
                   np.where(mid, ew,
                   np.where(conv_young, a5, ew))).astype(np.float32)
        gated_aq_sp = np.where(zero_h, proto,
                      np.where(mid, ew,
                      np.where(conv_young, a5, b3))).astype(np.float32)

        routing = {
            "frac_ticks_proto": float(zero_h.mean()),
            "frac_ticks_mid_ewma": float(mid.mean()),
            "frac_ticks_conv_young_learned": float(conv_young.mean()),
            "frac_ticks_conv_old": float(conv_old.mean()),
            "sensitivity_first_arrival_age": {
                "frac_ticks_conv_young_learned": float((conv & (age_fa < AGE_MIN)).mean()),
                "frac_ticks_conv_old": float((conv & (age_fa >= AGE_MIN)).mean()),
            },
        }
        # Huawei zero-history audit + internal consistency assert
        zh_share = float(zero_h.mean())
        assert abs(zh_share - routing["frac_ticks_proto"]) < 1e-12
        audit = {"zero_history_share": zh_share,
                 "n_functions": int(N),
                 "n_deployed_after_window": int((fp >= steady_t).sum()),
                 "prefix_minutes_median": float(np.median(
                     np.minimum(first_arr, T)))}
        print(f"    routing: {json.dumps({k: round(v,4) for k, v in routing.items() if isinstance(v, float)})}",
              flush=True)

        jdir = JOBS_ROOT / pool
        jdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(RUNS_DIR / "des_jobs_huawei" / pool / "shared.npz",
                    jdir / "shared.npz")
        job_meta = []
        for method, rates in [("A5_gated_aq", gated_aq),
                              ("A5_gated_aq_spectral", gated_aq_sp)]:
            for rho in RHOS:
                tau = newsvendor_quantile(rho)
                pw, ka = decisions_from_rates(rates, tau)
                fname = f"{method}__rho{rho}.npz"
                np.savez_compressed(jdir / fname, prewarm=pw, keepalive=ka)
                job_meta.append({"method": method, "rho": rho, "file": fname})
        json.dump({"split": pool, "seeds": SEEDS, "cold_init": COLD_INIT,
                   "jobs": job_meta}, open(jdir / "jobs.json", "w"))
        out["pools"][pool] = {"n": int(N), "routing": routing,
                              "zerohistory_audit": audit}
        print(f"    exported {len(job_meta)} jobs", flush=True)

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} (export stage)")


def boot_ci(diff, n=10000, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    means = diff[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def report():
    res = json.load(open(RUNS_DIR / "revision_r1_des.json"))
    rows = res["results"] if isinstance(res, dict) and "results" in res else res
    out = json.load(open(OUT))
    f5 = json.load(open(RUNS_DIR / "revision_f5_fourier_stats.json"))["surfaces"]["huawei_steady"]
    for pool in POOLS:
        ref = json.load(open(RUNS_DIR / f"revision_h2_huawei_des_{pool}.json"))
        ref_rows = ref["results"] if isinstance(ref, dict) and "results" in ref else ref
        pool_out = {}
        for method in ["A5_gated_aq", "A5_gated_aq_spectral"]:
            for rho in RHOS:
                ours = [r for r in rows if r["split"] == pool
                        and r["method"] == method and r["cost_ratio"] == rho]
                cell = {"csr_pct": float(np.mean([r["csr"] for r in ours]) * 100),
                        "csr_seed_std_pp": float(np.std([r["csr"] for r in ours]) * 100),
                        "wm_per_1k": float(np.mean([r["wm_per_1k_inv"] for r in ours]))}
                fc_ours = np.mean([np.asarray(r["func_cold"]) for r in ours], axis=0)
                for base in ["B4a_ewma", "A5_full_system"]:
                    br = [r for r in ref_rows if r["method"] == base
                          and r["cost_ratio"] == rho]
                    fc_base = np.mean([np.asarray(r["func_cold"]) for r in br], axis=0)
                    diff = fc_base - fc_ours          # >0: ours better
                    cell[f"vs_{base}"] = {
                        "csr_diff_pp": float(
                            (np.mean([r["csr"] for r in ours])
                             - np.mean([r["csr"] for r in br])) * 100),
                        "paired_cold_boot_ci": boot_ci(diff)}
                b3pt = next((c for c in f5[pool]["curves"]["B3_fourier"]
                             if c["rho"] == rho), None)
                if b3pt:
                    cell["b3_context"] = {"csr_pp": b3pt["csr_pp"],
                                          "wm_per_1k": b3pt["wm_per_1k"]}
                pool_out[f"{method}|{rho}"] = cell
                print(f"{pool} {method} rho={rho}: csr={cell['csr_pct']:.3f}% "
                      f"(vs EWMA {cell['vs_B4a_ewma']['csr_diff_pp']:+.3f}pp, "
                      f"vs A5 {cell['vs_A5_full_system']['csr_diff_pp']:+.3f}pp) "
                      f"wm={cell['wm_per_1k']:.0f}", flush=True)
        out["pools"][pool]["cells"] = pool_out
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
