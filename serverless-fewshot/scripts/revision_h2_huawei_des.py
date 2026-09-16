# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP-H3 (regime-map cells 2 and 3): steady-state DES on Huawei Public 2023.

Cell 2 (converged steady state): does the learned path stay at parity with a
per-function EWMA on a provider the predictor never saw?
Cell 3 (sparse / saturated): does every policy -- Oracle included -- converge
to the same cold-start rate where traffic is too sparse or too dense to act on?

Protocol is the Azure one, unchanged: identical decision layer for every
predictive arm (decisions_from_rates), identical cold-init calibration
(COLD_INIT), registered rho grid and seeds. The predictor is the Azure-2021
checkpoint -- no meta-training on Huawei, so this is transfer, not a refit.

Pools come from data/processed_huawei/splits.npz and are defined by the
registered frequency buckets on the 14-day steady window:
  h_mixed     1/h .. 1/min   (the S1-like middle where learning could pay)
  h_sparse    < 1/h          (sparse cell)
  h_saturated >= 1/min       (saturated cell)

Usage (export, then run with the numba runner as in run_revision_des_2019.sh):
  SF_DATA_DIR=processed_huawei SF_CKPT_DIR=results_azure2021/runs \
  python3.13 scripts/revision_h2_huawei_des.py \
      --export-jobs results/runs/des_jobs_huawei
"""

import os, sys, time, json, argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("SF_DATA_DIR", "processed_huawei")

from scripts.phase6_des import (  # noqa: E402
    PROCESSED_DIR, RUNS_DIR, COLD_INIT,
    decisions_from_rates, rates_ewma, rates_oracle, rates_a5,
    policy_b1, policy_b2,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402

POOLS = ["h_mixed", "h_sparse", "h_saturated"]
RHOS = [1.0, 10.0, 100.0]          # registered grid used for the cohort study
SEEDS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=",".join(POOLS))
    ap.add_argument("--export-jobs", default="results/runs/des_jobs_huawei")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--pilot", action="store_true",
                    help="30 functions of h_mixed, rho=10, seed 0 (timing)")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)     # MKL batched linalg thrashes on this VM
    from src.models.heads import N_QUANTILES
    from src.meta.trainer import ANILMetaTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_mm = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts_mm = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    splits = np.load(PROCESSED_DIR / "splits.npz")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    steady_t = int(splits["steady_t"][0])

    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features_mm.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=device,
    )
    ckpt_dir = Path(os.environ.get(
        "SF_CKPT_DIR", str(PROJECT_ROOT / "results_azure2021" / "runs")))
    trainer.load(ckpt_dir / "best_anil_ridge_s2_s0.pt")
    print(f"Loaded model: {ckpt_dir / 'best_anil_ridge_s2_s0.pt'} "
          f"(Azure 2021; no Huawei meta-training)")

    seeds = [0] if args.pilot else list(range(args.seeds))
    rhos = [10.0] if args.pilot else RHOS
    pools = ["h_mixed"] if args.pilot else args.pools.split(",")

    manifest = {"trace": "huawei_public_2023", "steady_t": steady_t,
                "checkpoint": str(ckpt_dir / "best_anil_ridge_s2_s0.pt"),
                "rhos": rhos, "seeds": seeds, "pools": {}}

    for pool in pools:
        rows = np.asarray(splits[pool])
        if args.pilot:
            rows = rows[:30]
        t0 = time.time()
        counts = np.asarray(counts_mm[rows][:, :steady_t]).astype(np.float32)
        feats = np.asarray(features_mm[rows][:, :steady_t, :], dtype=np.float32)
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        print(f"\n### {pool}: {len(rows)} functions, {counts.shape[1]} ticks, "
              f"{int(counts.sum()):,} invocations "
              f"(materialized in {time.time()-t0:.0f}s)", flush=True)

        stage = time.time()
        ew = rates_ewma(counts)
        print(f"    ewma {time.time()-stage:.0f}s", flush=True); stage = time.time()
        a5 = rates_a5(counts, feats, trainer, device)
        print(f"    a5 {time.time()-stage:.0f}s", flush=True); stage = time.time()
        pol_b1 = policy_b1(counts)
        print(f"    b1 {time.time()-stage:.0f}s", flush=True); stage = time.time()
        pol_b2 = policy_b2(counts)
        print(f"    b2 {time.time()-stage:.0f}s", flush=True)

        rate_mats = {"B4a_ewma": ew, "Oracle": rates_oracle(counts),
                     "A5_full_system": a5}
        policies = {"B1_fixed_keepalive": pol_b1, "B2_histogram": pol_b2}

        jdir = Path(args.export_jobs) / pool
        jdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(jdir / "shared.npz",
                            counts=counts.astype(np.int64),
                            dur_means=dm, dur_stds=ds)
        job_meta = []
        for method, (pw, ka) in policies.items():
            for rho in rhos:
                fname = f"{method}__rho{rho}.npz"
                np.savez_compressed(jdir / fname, prewarm=pw, keepalive=ka)
                job_meta.append({"method": method, "rho": rho, "file": fname})
        for method, rates in rate_mats.items():
            for rho in rhos:
                tau = newsvendor_quantile(rho)
                pw, ka = decisions_from_rates(rates, tau)
                fname = f"{method}__rho{rho}.npz"
                np.savez_compressed(jdir / fname, prewarm=pw, keepalive=ka)
                job_meta.append({"method": method, "rho": rho, "file": fname})
        with open(jdir / "jobs.json", "w") as f:
            json.dump({"split": pool, "seeds": seeds,
                       "cold_init": COLD_INIT, "jobs": job_meta}, f)
        manifest["pools"][pool] = {
            "rows": rows.tolist(), "n": int(len(rows)),
            "invocations": int(counts.sum()), "n_jobs": len(job_meta)}
        print(f"  Exported {len(job_meta)} jobs to {jdir}", flush=True)

    mpath = RUNS_DIR / ("des_huawei_manifest_pilot.json" if args.pilot
                        else "des_huawei_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f)
    print(f"\nManifest -> {mpath}")


if __name__ == "__main__":
    main()
