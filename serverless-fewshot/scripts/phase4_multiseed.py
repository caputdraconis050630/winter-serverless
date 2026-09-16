# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP6: Meta-train with additional seeds (1, 2) for training robustness.

Reuses train_meta_learner / evaluate_on_split from phase4_train.
Reports GO-gate CRPS as mean +/- std across training seeds.
"""

import sys, json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase4_train import train_meta_learner, evaluate_on_split
from scripts.phase4_train import PROCESSED_DIR, RUNS_DIR


def main():
    print("=" * 60)
    print("WP6: Multi-Seed Meta-Training (seeds 1, 2)")
    print("=" * 60)

    # mmap: 2019 features.npy is ~25GB (> RAM); episode sampling only slices per-function windows
    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    splits_data = np.load(PROCESSED_DIR / "splits.npz")

    s2_splits = {
        "train": splits_data["s2_train"],
        "val": splits_data["s2_val"],
        "test": splits_data["s2_test"],
    }

    results = {}

    # Seed 0 already trained: load its eval result from the gate file
    gate_path = RUNS_DIR / "go_nogo_gate.json"
    if gate_path.exists():
        with open(gate_path) as f:
            gate = json.load(f)
        results["0"] = {"crps": gate["ours_crps"], "source": "existing"}
        print(f"Seed 0 (existing): CRPS={gate['ours_crps']:.4f}")

    for seed in (1, 2):
        trainer, best_crps = train_meta_learner(
            features, counts, s2_splits, seed=seed,
            body_type="tcn", head_type="ridge",
            n_steps=5000, batch_size=32, k_support=10, k_query=32,
            horizons=(1,), tag="anil_ridge_s2",
        )
        eval_res = evaluate_on_split(
            trainer, features, counts, s2_splits["test"],
            k_support=10, horizons=(1,), tag=f"Ours(S2,seed{seed})"
        )
        results[str(seed)] = {"crps": eval_res["crps"], "val_crps": best_crps,
                              "source": "trained"}

    crps_values = [r["crps"] for r in results.values()]
    summary = {
        "per_seed": results,
        "crps_mean": float(np.mean(crps_values)),
        "crps_std": float(np.std(crps_values)),
        "n_seeds": len(crps_values),
    }

    out = RUNS_DIR / "multiseed_training.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Multi-seed CRPS: {summary['crps_mean']:.4f} +/- {summary['crps_std']:.4f} (n={summary['n_seeds']})")
    print(f"Saved to {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
