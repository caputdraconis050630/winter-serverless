# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision E1: v4 "economic gate" — steady-state evaluation.

v4 is v3 with the EWMA and learned arms swapped (registered threshold 100,
no new tuning):
    cum_prev == 0        -> prototype zero-shot rate
    0 < cum_prev < 100   -> online-adapted A5 (learned, prototype-lineage)
    cum_prev >= 100      -> EWMA
The cohort-side evaluation (revision_a4_crosstrace.json) already showed v4
REGRESSES onboarding (count-convergence != time-convergence; see
REVISION_LOG). This script measures the steady-state side of the ledger.

--trace 2021: rate composition identical to revision_a2_des.py (same
  checkpoint best_anil_ridge_s1_s0.pt, same prototypes from s1_train, same
  rho grid incl. LOW_RHOS, DES seeds 0..9). Runs ONLY the new A5_gated_v4
  arm — every comparator is already archived in sim_results_des_revision.json
  under the identical protocol. Output: revision_e1_v4_2021.json.

--trace 2019: composes v3 AND v4 rates for the chain's exact split samples
  (des_sample_manifest.json rows; neither gate variant exists in the 2019
  campaign — the chain carries only legacy A5_gated_v2), 2019-trained
  checkpoint best_anil_ridge_s2_s0.pt as used by the chain, prototypes from
  2019 s2_train (steady-state same-trace training is split-separated;
  the V2 leakage concern applies to the onboarding cohort only). Exports
  job npzs to des_jobs_2019_e1/ (seeds mirror the chain's jobs.json); run
  des_runner_fast on them once the chain frees the numba env.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)  # MKL thread-thrashing fix

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import (  # noqa: E402
    rates_a5, rates_ewma, run_config,
    decisions_from_rates, COLD_INIT, FULL_RHOS, REDUCED_RHOS,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "results" / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L = 60
MIN_INV = 100
LOW_RHOS = [0.01, 0.02, 0.05]
MED_IDX = N_QUANTILES // 2


def compose_gates(counts, feats, trainer, pm):
    from scripts.revision_a2_des import rates_proto_prefix
    a5 = rates_a5(counts, feats, trainer, DEVICE)
    ew = rates_ewma(counts)
    proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
    zero = cum_prev == 0
    mid = (cum_prev > 0) & (cum_prev < MIN_INV)
    v3 = np.where(zero, proto, np.where(mid, ew, a5)).astype(np.float32)
    v4 = np.where(zero, proto, np.where(mid, a5, ew)).astype(np.float32)
    routing = {"frac_ticks_proto": float(zero.mean()),
               "frac_ticks_mid": float(mid.mean()),
               "frac_ticks_converged": float((cum_prev >= MIN_INV).mean())}
    return v3, v4, routing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", choices=["2021", "2019"], required=True)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    if args.trace == "2021":
        from scripts.phase63_onboarding_drift import build_prototypes
        PROC = PROJECT_ROOT / "data" / "processed"
        features = np.load(PROC / "features.npy")
        counts_all = np.load(PROC / "counts.npy")
        splits_data = np.load(PROC / "splits.npz")
        dur_df = pd.read_csv(PROC / "duration_stats.csv")
        trainer = ANILMetaTrainer(
            body_type="tcn", head_type="ridge",
            in_features=features.shape[2], embedding_dim=64,
            n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE)
        trainer.load(RUNS_DIR / "best_anil_ridge_s1_s0.pt")
        pm = build_prototypes(trainer, features, counts_all,
                              splits_data["s1_train"])
        SEEDS = list(range(10))

        all_results, routing_all = [], {}
        for split in ["S1", "S2", "S3"]:
            key = {"S1": "s1_test", "S2": "s2_test", "S3": "s3_test"}[split]
            test_idx = splits_data[key]
            counts = counts_all[test_idx]
            feats = features[test_idx]
            if split == "S3":
                t0 = int(splits_data.get("s3_test_t_start", [10080])[0])
                counts = counts[:, t0:]
                feats = feats[:, t0:]
            dm = np.nan_to_num(np.array(
                [dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
                 for fi in test_idx]), nan=1.0)
            ds = np.nan_to_num(np.array(
                [dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
                 for fi in test_idx]), nan=0.5)
            print(f"\n### {split}: composing gate rates...", flush=True)
            t0s = time.time()
            _v3, v4, routing = compose_gates(counts, feats, trainer, pm)
            routing_all[split] = routing
            print(f"    rates in {time.time()-t0s:.0f}s; routing {routing}",
                  flush=True)
            rhos = (FULL_RHOS if split == "S1" else REDUCED_RHOS) + LOW_RHOS
            jobs = []
            for rho in rhos:
                tau = newsvendor_quantile(rho)
                pw, ka = decisions_from_rates(v4, tau)
                jobs.append(("A5_gated_v4", rho, SEEDS, counts, pw, ka,
                             dm, ds, COLD_INIT, split))
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for res_list in ex.map(run_config, jobs):
                    all_results.extend(res_list)
                    r = res_list[0]
                    print(f"    v4 rho={r['cost_ratio']:6.2f} "
                          f"CSR={np.mean([x['csr'] for x in res_list]):.4f} "
                          f"WM={np.mean([x['wm_per_1k_inv'] for x in res_list]):.0f}",
                          flush=True)
        out_path = RUNS_DIR / "revision_e1_v4_2021.json"
        with open(out_path, "w") as f:
            json.dump({"results": all_results, "routing": routing_all,
                       "seeds": SEEDS}, f, default=float)
        blob = open(out_path, "rb").read()
        assert blob and blob.count(0) == 0
        json.load(open(out_path))
        print(f"Saved + verified {out_path} ({len(all_results)} rows)")

    else:  # 2019 export
        import os
        os.environ.setdefault("SF_DATA_DIR", "processed_2019")
        from scripts.phase63_onboarding_drift import build_prototypes
        PROC = PROJECT_ROOT / "data" / "processed_2019"
        features_mm = np.load(PROC / "features.npy", mmap_mode="r")
        counts_mm = np.load(PROC / "counts.npy", mmap_mode="r")
        splits_data = np.load(PROC / "splits.npz")
        manifest = json.load(open(RUNS_DIR / "des_sample_manifest.json"))
        trainer = ANILMetaTrainer(
            body_type="tcn", head_type="ridge",
            in_features=features_mm.shape[2], embedding_dim=64,
            n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE)
        trainer.load(RUNS_DIR / "best_anil_ridge_s2_s0.pt")
        pm = build_prototypes(trainer, features_mm, counts_mm,
                              splits_data["s2_train"])
        src_root = RUNS_DIR / "des_jobs_2019"
        jobs_root = RUNS_DIR / "des_jobs_2019_e1"
        for split in ["S1", "S2", "S3"]:
            src_jobs = json.load(open(src_root / split / "jobs.json"))
            shared = np.load(src_root / split / "shared.npz")
            counts = shared["counts"]
            rows = manifest[split]["rows"]
            feats = np.asarray(features_mm[rows], dtype=np.float32)
            if split == "S3":
                ts = int(splits_data.get("s3_test_t_start", [10080])[0])
                feats = feats[:, ts:]
            assert feats.shape[:2] == counts.shape, (feats.shape, counts.shape)
            print(f"\n### {split}: composing gate rates "
                  f"({counts.shape[0]} funcs x {counts.shape[1]} ticks)...",
                  flush=True)
            t0s = time.time()
            v3, v4, routing = compose_gates(counts, feats, trainer, pm)
            print(f"    rates in {time.time()-t0s:.0f}s; routing {routing}",
                  flush=True)
            jdir = jobs_root / split
            jdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(jdir / "shared.npz", counts=counts,
                                dur_means=shared["dur_means"],
                                dur_stds=shared["dur_stds"])
            job_meta = []
            rhos = sorted({j["rho"] for j in src_jobs["jobs"]})
            for method, rmat in [("A5_gated_v3", v3), ("A5_gated_v4", v4)]:
                for rho in rhos:
                    tau = newsvendor_quantile(rho)
                    pw, ka = decisions_from_rates(rmat, tau)
                    fname = f"{method}__rho{rho}.npz"
                    np.savez_compressed(jdir / fname, prewarm=pw, keepalive=ka)
                    job_meta.append({"method": method, "rho": rho, "file": fname})
            with open(jdir / "jobs.json", "w") as fp:
                json.dump({"split": split, "seeds": src_jobs["seeds"],
                           "cold_init": src_jobs["cold_init"],
                           "routing": routing, "jobs": job_meta}, fp)
            print(f"    exported {len(job_meta)} jobs to {jdir}", flush=True)
        print("Run after chain: des_runner_fast.py --jobs "
              "results/runs/des_jobs_2019_e1 --out "
              "results/runs/revision_e1_gates_2019.json")


if __name__ == "__main__":
    main()
