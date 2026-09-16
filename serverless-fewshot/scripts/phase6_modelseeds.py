# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP6 follow-up: steady-state robustness across meta-training seeds.

Runs the A5 pipeline on S1 with each of the three trained models
(seeds 0/1/2) and reports CSR/WM at rho=10. Shows the steady-state result
is not an artifact of a single training run.
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import rates_a5, decisions_from_rates, COLD_INIT
from src.sim.des import simulate_trace
from src.decision.newsvendor import newsvendor_quantile
from src.models.heads import N_QUANTILES
from src.meta.trainer import ANILMetaTrainer

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    features = np.load(PROCESSED_DIR / "features.npy")
    counts_all = np.load(PROCESSED_DIR / "counts.npy")
    splits_data = np.load(PROCESSED_DIR / "splits.npz")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")

    test_idx = splits_data["s1_test"]
    counts = counts_all[test_idx]
    feats = features[test_idx]
    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] for fi in test_idx]), nan=1.0)
    ds = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] for fi in test_idx]), nan=0.5)

    tau = newsvendor_quantile(10.0)
    ckpts = {
        0: "best_anil_ridge_s1_s0.pt",   # primary (trained on S1)
        1: "best_anil_ridge_s2_s1.pt",   # additional training seeds (S2)
        2: "best_anil_ridge_s2_s2.pt",
    }
    results = {}
    for seed, name in ckpts.items():
        p = RUNS_DIR / name
        if not p.exists():
            print(f"  [skip] {name}")
            continue
        trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                                  in_features=features.shape[2],
                                  embedding_dim=64, n_quantiles=N_QUANTILES,
                                  n_horizons=1, device=DEVICE)
        trainer.load(p)
        rates = rates_a5(counts, feats, trainer, DEVICE)
        pw, ka = decisions_from_rates(rates, tau)
        csrs, wms = [], []
        for s in range(3):
            m = simulate_trace(counts, pw, ka, dm, ds, seed=s,
                               cold_mu=COLD_INIT["mu"],
                               cold_sigma=COLD_INIT["sigma"])
            csrs.append(m["csr"]); wms.append(m["wm_per_1k_inv"])
        results[str(seed)] = {"model": name,
                              "csr_mean": float(np.mean(csrs)),
                              "csr_std": float(np.std(csrs)),
                              "wm_mean": float(np.mean(wms))}
        print(f"  model seed {seed} ({name}): CSR={np.mean(csrs):.4f}"
              f"+-{np.std(csrs):.4f} WM={np.mean(wms):.0f}")

    csr_across = [v["csr_mean"] for v in results.values()]
    summary = {"per_model": results,
               "csr_across_models_mean": float(np.mean(csr_across)),
               "csr_across_models_std": float(np.std(csr_across))}
    with open(RUNS_DIR / "modelseed_robustness.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAcross training seeds: CSR {np.mean(csr_across):.4f} "
          f"+- {np.std(csr_across):.4f}")


if __name__ == "__main__":
    main()
