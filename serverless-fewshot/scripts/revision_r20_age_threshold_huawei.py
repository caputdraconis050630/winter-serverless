#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R20-C: faithful Huawei validation for the age-threshold grid.

Exports Huawei Public 2023 steady-window DES jobs for the dense R20 gate grid.
The learned rate matrix uses the Section-3-faithful head (no archived EWMA
blend), with the Azure-2021 checkpoint and prototypes. Huawei age is measured
from the release's presence mask / first_present timestamp, as in R1.

Run export under the torch env, then execute jobs with scripts/des_runner_fast.py:

  SF_DATA_DIR=processed_huawei PYTHONPATH=/data/260715/site-packages:. \
    python3.13 scripts/revision_r20_age_threshold_huawei.py --export

  PYTHONPATH=/data/260715/site-packages-des:. python3.13 \
    scripts/des_runner_fast.py --jobs results/runs/des_jobs_r20_huawei \
    --out results/runs/revision_r20_huawei_fast_runs.json
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

RUNS = PROJECT_ROOT / "results" / "runs"
JOBS_ROOT = RUNS / "des_jobs_r20_huawei"
P_HW = PROJECT_ROOT / "data" / "processed_huawei"
P_21 = PROJECT_ROOT / "data" / "processed"

POOLS = ["h_mixed", "h_sparse", "h_saturated"]
RHOS = [1.0, 10.0, 100.0]
SEEDS = [0, 1, 2]
MIN_INV = 100
AGE_GRID = [0, 30, 60, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, None]
FAMILIES = ["WE", "WL"]


def age_label(age_min):
    return "inf" if age_min is None else str(int(age_min))


def method_name(family, age_min):
    return f"G_{family}_A{age_label(age_min)}"


def compose_gate_rates(proto, ewma, faithful, zero, w_mask, conv, age, family, age_min):
    rates = np.empty_like(ewma, dtype=np.float32)
    rates[zero] = proto[zero]
    rates[w_mask] = faithful[w_mask] if family == "WL" else ewma[w_mask]
    if age_min is None:
        learned_conv = conv
    else:
        learned_conv = conv & (age < int(age_min))
    ewma_conv = conv & ~learned_conv
    rates[learned_conv] = faithful[learned_conv]
    rates[ewma_conv] = ewma[ewma_conv]
    return rates


def export(pools):
    import pandas as pd
    import torch

    torch.set_num_threads(1)
    from scripts.phase6_des import COLD_INIT, decisions_from_rates, rates_ewma
    from scripts.phase63_onboarding_drift import build_prototypes
    from scripts.revision_a2_des import rates_proto_prefix
    from scripts.revision_r11_faithful import rates_a5_faithful
    from src.decision.newsvendor import newsvendor_quantile
    from src.meta.trainer import ANILMetaTrainer
    from src.models.heads import N_QUANTILES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_mm = np.load(P_HW / "features.npy", mmap_mode="r")
    counts_mm = np.load(P_HW / "counts.npy", mmap_mode="r")
    splits = np.load(P_HW / "splits.npz")
    first_present = np.load(P_HW / "first_present.npy")
    steady_t = int(splits["steady_t"][0])
    dur_df = pd.read_csv(P_HW / "duration_stats.csv")

    ckpt = PROJECT_ROOT / "results_azure2021" / "runs" / "best_anil_ridge_s2_s0.pt"
    trainer = ANILMetaTrainer(
        body_type="tcn",
        head_type="ridge",
        in_features=features_mm.shape[2],
        embedding_dim=64,
        n_quantiles=N_QUANTILES,
        n_horizons=1,
        device=device,
    )
    trainer.load(ckpt)

    feat21 = np.load(P_21 / "features.npy")
    cnt21 = np.load(P_21 / "counts.npy")
    spl21 = np.load(P_21 / "splits.npz")
    pm = build_prototypes(trainer, feat21, cnt21, spl21["s2_train"])

    manifest = {
        "description": "R20 faithful Huawei age-threshold grid export",
        "checkpoint": str(ckpt),
        "proto_source": "Azure 2021 s2_train",
        "age_source": "first_present.npy deployment-presence mask",
        "min_invocations": MIN_INV,
        "age_grid_minutes": [age_label(a) for a in AGE_GRID],
        "rhos": RHOS,
        "seeds": SEEDS,
        "pools": {},
    }

    for pool in pools:
        rows = np.asarray(splits[pool])
        t0 = time.time()
        counts = np.asarray(counts_mm[rows][:, :steady_t]).astype(np.float32)
        feats = np.asarray(features_mm[rows][:, :steady_t, :], dtype=np.float32)
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        fp = first_present[rows].astype(np.int64)
        print(
            f"### {pool}: {len(rows)} funcs, {counts.shape[1]} ticks, "
            f"{int(counts.sum()):,} invocations",
            flush=True,
        )

        stage = time.time()
        ewma = rates_ewma(counts)
        faithful = rates_a5_faithful(counts.astype(np.float32), feats, trainer, pm)
        proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
        print(f"    rates done in {time.time() - stage:.0f}s", flush=True)

        ticks = np.arange(counts.shape[1])[None, :]
        age = ticks - fp[:, None]
        zero = cum_prev == 0
        w_mask = (cum_prev > 0) & (cum_prev < MIN_INV)
        conv = cum_prev >= MIN_INV

        rate_mats = {
            "B4a_ewma": ewma,
            "A5_faithful": faithful,
        }
        for family in FAMILIES:
            for age_min in AGE_GRID:
                rate_mats[method_name(family, age_min)] = compose_gate_rates(
                    proto, ewma, faithful, zero, w_mask, conv, age, family, age_min
                )

        jdir = JOBS_ROOT / pool
        jdir.mkdir(parents=True, exist_ok=True)
        shared_src = RUNS / "des_jobs_huawei" / pool / "shared.npz"
        if shared_src.exists():
            shutil.copy(shared_src, jdir / "shared.npz")
        else:
            np.savez_compressed(
                jdir / "shared.npz",
                counts=counts.astype(np.int64),
                dur_means=dm,
                dur_stds=ds,
            )
        job_meta = []
        for method, rates in rate_mats.items():
            for rho in RHOS:
                tau = newsvendor_quantile(rho)
                pw, ka = decisions_from_rates(rates, tau)
                fname = f"{method}__rho{rho:g}.npz"
                np.savez_compressed(jdir / fname, prewarm=pw, keepalive=ka)
                job_meta.append({"method": method, "rho": float(rho), "file": fname})
        with open(jdir / "jobs.json", "w") as f:
            json.dump(
                {
                    "split": pool,
                    "seeds": SEEDS,
                    "cold_init": COLD_INIT,
                    "jobs": job_meta,
                },
                f,
                indent=1,
            )
        manifest["pools"][pool] = {
            "n": int(len(rows)),
            "invocations": int(counts.sum()),
            "n_jobs": len(job_meta),
            "routing": {
                "frac_ticks_proto": float(zero.mean()),
                "frac_ticks_mid": float(w_mask.mean()),
                "frac_ticks_conv": float(conv.mean()),
                "frac_ticks_conv_young_A720": float((conv & (age < 720)).mean()),
            },
            "elapsed_sec": time.time() - t0,
        }
        print(f"    exported {len(job_meta)} jobs in {time.time() - t0:.0f}s", flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / "revision_r20_huawei_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {RUNS / 'revision_r20_huawei_manifest.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--pools", default=",".join(POOLS))
    args = ap.parse_args()
    if not args.export:
        ap.error("pass --export")
    export(args.pools.split(","))


if __name__ == "__main__":
    main()
