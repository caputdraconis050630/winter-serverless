# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Phase 6 (DES) on Azure-2019: sampled-population runs + 3-way hybrid gate.

Differences from phase6_des.py (2021 full-split driver):
  - Splits are too large for exhaustive DES (s3_test = 30k functions), so each
    split is subsampled with the validated estimator design
    (scripts/des_subsample_validation.py): take-all for the top TAKE_ALL_FRAC
    of the sample budget by trace volume + SRS over the remainder, with
    inverse-inclusion weights saved for population-level ratio estimates.
  - features.npy (25GB) is mmap'ed; only sampled rows are materialized.
  - Adds A5_gated_v2: per-function online routing among
    {EWMA (sparse), A5 adapted (learned), B5 global} by trailing prediction
    error with a conservative margin — the gate must observe A5 beating B5 by
    >=10% on trailing error before routing to adaptation. All routing inputs
    are strictly causal (trailing window, updated every GATE_INTERVAL ticks).

Usage:
  SF_DATA_DIR=processed_2019 python phase6_des_sampled.py [--pilot] \
      [--splits S1,S2,S3] [--seeds 3] [--out sim_results_des_2019.json]
"""

import os, sys, time, json, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import (
    PROCESSED_DIR, RUNS_DIR, COLD_INIT, REDUCED_RHOS,
    decisions_from_rates, rates_ewma, rates_oracle, rates_a5, rates_b5,
    policy_b1, policy_b2, run_config,
)
from src.decision.newsvendor import newsvendor_quantile

# Sampling design (validated on Azure-2021 full DES results)
SAMPLE_SIZES = {"S1": 1000, "S2": 1000, "S3": 1500}
TAKE_ALL_FRAC = 0.2
SAMPLE_SEED = 20260716

# Gate v2 design constants (fixed a priori, not tuned on test outcomes)
GATE_ALPHA = 0.9        # A5 must beat B5 by >=10% trailing error to be routed
GATE_WINDOW = 720       # 12h trailing-error window
GATE_INTERVAL = 240     # re-route every 4h
GATE_MIN_INV = 100      # sparse functions stay on the cheap EWMA path


def build_sample(pool_idx, counts_mmap, n_sample, rng):
    """Take-all top by trace volume + SRS remainder; returns (rows, weights)."""
    pool_idx = np.asarray(pool_idx)
    if len(pool_idx) <= n_sample:
        return pool_idx, np.ones(len(pool_idx))
    totals = np.array([counts_mmap[fi].sum() for fi in pool_idx], dtype=float)
    k = max(1, int(TAKE_ALL_FRAC * n_sample))
    top = pool_idx[np.argsort(totals)[::-1][:k]]
    rest_pool = np.setdiff1d(pool_idx, top)
    pick = rng.choice(rest_pool, n_sample - k, replace=False)
    rows = np.concatenate([top, np.sort(pick)])
    weights = np.concatenate([
        np.ones(k), np.full(n_sample - k, len(rest_pool) / (n_sample - k))])
    return rows, weights


def gated_rates_v2(a5, b5, ewma, counts,
                   alpha=GATE_ALPHA, window=GATE_WINDOW,
                   interval=GATE_INTERVAL, min_inv=GATE_MIN_INV):
    """3-way per-function routing by trailing prediction error (causal).

    Route at tick t uses errors from [t-window, t) only:
      cum_inv < min_inv        -> EWMA path (sparse)
      trail_a5 < alpha*trail_b5 -> A5 (learned adaptation demonstrably better)
      else                      -> B5 (global default)
    """
    N, T = counts.shape
    logc = np.log1p(counts.astype(np.float32))

    def trailing_err(rates):
        e = np.abs(np.log1p(np.maximum(rates, 0.0)) - logc)  # [N, T] f32
        cs = np.cumsum(e, axis=1, dtype=np.float64)
        trail = np.empty_like(e)
        trail[:, :window] = cs[:, :window] / np.arange(1, window + 1)
        trail[:, window:] = (cs[:, window:] - cs[:, :-window]) / window
        return trail

    tr_a5 = trailing_err(a5)
    tr_b5 = trailing_err(b5)
    cum = np.cumsum(counts, axis=1)
    cum_prev = np.concatenate([np.zeros((N, 1)), cum[:, :-1]], axis=1)

    # route codes: 0=EWMA, 1=A5, 2=B5; decisions refreshed every `interval`
    route = np.full((N, T), 2, dtype=np.int8)
    decision_ticks = np.arange(0, T, interval)
    cur = np.full(N, 0, dtype=np.int8)  # everyone starts sparse/EWMA
    for td in decision_ticks:
        tprev = max(td - 1, 0)
        sparse = cum_prev[:, td] < min_inv
        a5_wins = tr_a5[:, tprev] < alpha * tr_b5[:, tprev]
        warm = td >= window  # need a full trailing window to trust errors
        cur = np.where(sparse, 0, np.where(warm & a5_wins, 1, 2)).astype(np.int8)
        route[:, td:td + interval] = cur[:, None]

    routed = np.where(route == 0, ewma, np.where(route == 1, a5, b5))
    stats = {
        "frac_ticks_ewma": float((route == 0).mean()),
        "frac_ticks_a5": float((route == 1).mean()),
        "frac_ticks_b5": float((route == 2).mean()),
        "frac_funcs_ever_a5": float((route == 1).any(axis=1).mean()),
        "alpha": alpha, "window": window, "interval": interval,
        "min_inv": min_inv,
    }
    return routed.astype(np.float32), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="S1,S2,S3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pilot", action="store_true",
                    help="n=100, rho=10 only, seed 0 only, S2 only, timing run")
    ap.add_argument("--workers", type=int, default=max(2, os.cpu_count() - 2))
    ap.add_argument("--out", default="sim_results_des_2019.json")
    ap.add_argument("--export-jobs", default=None, metavar="DIR",
                    help="write per-(method,rho) job npz files to DIR instead "
                         "of running the DES inline (for the numba runner)")
    args = ap.parse_args()

    import torch
    from src.models.heads import N_QUANTILES
    from src.meta.trainer import ANILMetaTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SAMPLE_SEED)

    print("=" * 60)
    print(f"PHASE 6 (DES, sampled): {PROCESSED_DIR.name}"
          + (" [PILOT]" if args.pilot else ""))
    print("=" * 60)

    features_mm = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts_mm = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    splits_data = np.load(PROCESSED_DIR / "splits.npz")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")

    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features_mm.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=device,
    )
    ckpt_dir = Path(os.environ.get("SF_CKPT_DIR", str(RUNS_DIR)))
    trainer.load(ckpt_dir / "best_anil_ridge_s2_s0.pt")
    print(f"Loaded model: {ckpt_dir / 'best_anil_ridge_s2_s0.pt'}")

    seeds = [0] if args.pilot else list(range(args.seeds))
    rhos = [10.0] if args.pilot else REDUCED_RHOS
    split_names = ["S2"] if args.pilot else args.splits.split(",")

    all_results = []
    sample_manifest = {}
    gate_stats_all = {}

    for split in split_names:
        key = {"S1": "s1_test", "S2": "s2_test", "S3": "s3_test"}[split]
        n_sample = 100 if args.pilot else SAMPLE_SIZES[split]

        t0 = time.time()
        rows, weights = build_sample(splits_data[key], counts_mm, n_sample, rng)
        counts = np.asarray(counts_mm[rows])
        feats = np.asarray(features_mm[rows], dtype=np.float32)
        if split == "S3":
            ts = int(splits_data.get("s3_test_t_start", [10080])[0])
            counts = counts[:, ts:]
            feats = feats[:, ts:]
        print(f"\n### Split {split}: {len(rows)}/{len(splits_data[key])} funcs "
              f"sampled, {counts.shape[1]} ticks, rhos={rhos} "
              f"(materialized in {time.time()-t0:.0f}s)")
        sample_manifest[split] = {"rows": rows.tolist(),
                                  "weights": weights.tolist(),
                                  "pool_size": int(len(splits_data[key]))}

        dm = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
             for fi in rows]), nan=1.0)
        ds = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
             for fi in rows]), nan=0.5)

        print("  Computing predictions/policies...")
        stage = time.time()
        ew = rates_ewma(counts)
        print(f"    ewma {time.time()-stage:.0f}s"); stage = time.time()
        a5 = rates_a5(counts, feats, trainer, device)
        print(f"    a5 {time.time()-stage:.0f}s"); stage = time.time()
        b5 = rates_b5(counts, feats, trainer, device)
        print(f"    b5 {time.time()-stage:.0f}s"); stage = time.time()
        gated, gstats = gated_rates_v2(a5, b5, ew, counts)
        gate_stats_all[split] = gstats
        print(f"    gate_v2 {time.time()-stage:.0f}s "
              f"(ticks: ewma {gstats['frac_ticks_ewma']*100:.0f}% / "
              f"a5 {gstats['frac_ticks_a5']*100:.0f}% / "
              f"b5 {gstats['frac_ticks_b5']*100:.0f}%)"); stage = time.time()
        rate_mats = {
            "B4a_ewma": ew,
            "Oracle": rates_oracle(counts),
            "A5_full_system": a5,
            "B5_global": b5,
            "A5_gated_v2": gated,
        }
        pol_b1 = policy_b1(counts)
        print(f"    b1 {time.time()-stage:.0f}s"); stage = time.time()
        pol_b2 = policy_b2(counts)
        print(f"    b2 {time.time()-stage:.0f}s")
        policies = {"B1_fixed_keepalive": pol_b1, "B2_histogram": pol_b2}

        jobs = []
        for method, (pw, ka) in policies.items():
            for rho in rhos:
                jobs.append((method, rho, seeds, counts, pw, ka,
                             dm, ds, COLD_INIT, split))
        for method, rates in rate_mats.items():
            for rho in rhos:
                tau = newsvendor_quantile(rho)
                pw, ka = decisions_from_rates(rates, tau)
                jobs.append((method, rho, seeds, counts, pw, ka,
                             dm, ds, COLD_INIT, split))

        if args.export_jobs:
            jdir = Path(args.export_jobs) / split
            jdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(jdir / "shared.npz", counts=counts,
                                dur_means=dm, dur_stds=ds)
            job_meta = []
            for method, rho, _seeds, _c, pw, ka, *_ in jobs:
                fname = f"{method}__rho{rho}.npz"
                np.savez_compressed(jdir / fname, prewarm=pw, keepalive=ka)
                job_meta.append({"method": method, "rho": rho, "file": fname})
            with open(jdir / "jobs.json", "w") as f:
                json.dump({"split": split, "seeds": seeds,
                           "cold_init": COLD_INIT, "jobs": job_meta}, f)
            print(f"  Exported {len(job_meta)} jobs to {jdir}")
            continue

        print(f"  Running {len(jobs)} configs x {len(seeds)} seeds "
              f"on {args.workers} workers...")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res_list in ex.map(run_config, jobs):
                all_results.extend(res_list)
                r = res_list[0]
                print(f"    {r['method']:<20s} rho={r['cost_ratio']:6.1f} "
                      f"CSR={np.mean([x['csr'] for x in res_list]):.4f} "
                      f"WM={np.mean([x['wm_per_1k_inv'] for x in res_list]):.0f} "
                      f"({r['elapsed_sec']:.0f}s/seed)", flush=True)
        print(f"  Split {split} simulated in {time.time()-t0:.0f}s")

    suffix = "_pilot" if args.pilot else ""
    if not args.export_jobs:
        out_path = RUNS_DIR / args.out.replace(".json", f"{suffix}.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, default=float)
        print(f"\nSaved {len(all_results)} runs to {out_path}")
    with open(RUNS_DIR / f"des_sample_manifest{suffix}.json", "w") as f:
        json.dump(sample_manifest, f)
    with open(RUNS_DIR / f"gate_v2_routing_stats{suffix}.json", "w") as f:
        json.dump(gate_stats_all, f, indent=2)
    if args.export_jobs:
        return

    print(f"\nSummary at rho=10 (mean over seeds)")
    for split in split_names:
        print(f"  --- {split} ---")
        for method in sorted(set(r["method"] for r in all_results)):
            rs = [r for r in all_results
                  if r["method"] == method and r["split"] == split
                  and abs(r["cost_ratio"] - 10.0) < 0.01]
            if rs:
                print(f"    {method:<22s} "
                      f"CSR={np.mean([r['csr'] for r in rs]):.4f} "
                      f"WM={np.mean([r['wm_per_1k_inv'] for r in rs]):.0f}")


if __name__ == "__main__":
    main()
