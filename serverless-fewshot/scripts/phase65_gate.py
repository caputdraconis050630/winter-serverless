# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP5: Hybrid gate — route functions between the learned predictor and the
cheap EWMA path, per HybridGate's sparse-function criterion (trailing
observed invocations < 100 -> EWMA; else learned A5).

Runs A5_gated through the same DES pipeline and appends results to
sim_results_des.json. Also reports routing statistics (what fraction of
function-ticks / functions use the learned path, by frequency bucket).
"""

import sys, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import (
    decisions_from_rates, rates_ewma, rates_a5, run_config,
    COLD_INIT, FULL_RHOS, REDUCED_RHOS,
)
from src.decision.newsvendor import newsvendor_quantile
from src.models.heads import N_QUANTILES
from src.meta.trainer import ANILMetaTrainer

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_INVOCATIONS = 100


def gated_rates(a5, ewma, counts):
    """Route per function-tick: learned once >=100 observed invocations."""
    N, T = counts.shape
    cum = np.cumsum(counts, axis=1)
    cum_prev = np.concatenate([np.zeros((N, 1)), cum[:, :-1]], axis=1)
    use_learned = cum_prev >= MIN_INVOCATIONS
    routed = np.where(use_learned, a5, ewma).astype(np.float32)
    return routed, use_learned


def main():
    features = np.load(PROCESSED_DIR / "features.npy")
    counts_all = np.load(PROCESSED_DIR / "counts.npy")
    splits_data = np.load(PROCESSED_DIR / "splits.npz")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")

    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE,
    )
    trainer.load(RUNS_DIR / "best_anil_ridge_s1_s0.pt")

    seeds = list(range(5))
    routing_stats = {}
    new_results = []

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

        print(f"\n### {split}: computing rates...")
        a5 = rates_a5(counts, feats, trainer, DEVICE)
        ew = rates_ewma(counts)
        routed, use_learned = gated_rates(a5, ew, counts)

        frac_ticks = float(use_learned.mean())
        frac_funcs_ever = float((use_learned.any(axis=1)).mean())
        frac_funcs_final = float(use_learned[:, -1].mean())
        routing_stats[split] = {
            "frac_function_ticks_learned": frac_ticks,
            "frac_functions_ever_learned": frac_funcs_ever,
            "frac_functions_learned_at_end": frac_funcs_final,
            "min_invocations": MIN_INVOCATIONS,
        }
        print(f"  routing: {frac_ticks*100:.1f}% of function-ticks learned, "
              f"{frac_funcs_final*100:.1f}% of functions on learned path at end")

        rhos = FULL_RHOS if split == "S1" else REDUCED_RHOS
        jobs = []
        for rho in rhos:
            tau = newsvendor_quantile(rho)
            pw, ka = decisions_from_rates(routed, tau)
            jobs.append(("A5_gated", rho, seeds, counts, pw, ka,
                         dm, ds, COLD_INIT, split))

        with ProcessPoolExecutor(max_workers=6) as ex:
            for res_list in ex.map(run_config, jobs):
                new_results.extend(res_list)
                r = res_list[0]
                print(f"    A5_gated rho={r['cost_ratio']:6.1f} "
                      f"CSR={np.mean([x['csr'] for x in res_list]):.4f} "
                      f"WM={np.mean([x['wm_per_1k_inv'] for x in res_list]):.0f}")

    # Append to the main results file
    path = RUNS_DIR / "sim_results_des.json"
    with open(path) as f:
        existing = json.load(f)
    existing = [r for r in existing if r["method"] != "A5_gated"]
    existing.extend(new_results)
    with open(path, "w") as f:
        json.dump(existing, f, default=float)

    with open(RUNS_DIR / "gate_routing_stats.json", "w") as f:
        json.dump(routing_stats, f, indent=2)
    print(f"\nAppended {len(new_results)} A5_gated runs; routing stats saved.")


if __name__ == "__main__":
    main()
