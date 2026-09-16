#!/usr/bin/env python3
"""R15: matched-subset CRPS for the B7 LSTM comparison.

Table S31's archived B7 LSTM CRPS was computed on the first 20 S2
hold-out functions. This script evaluates the WINTER ridge head on the
same first-20 subset using the same support/query times as the archived
B7 predictor-quality run, so the LSTM/WINTER CRPS ratio is a direct
matched-population comparison.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.meta.trainer import ANILMetaTrainer  # noqa: E402
from src.models.heads import N_QUANTILES, QUANTILES, crps_from_quantiles  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results_azure2021" / "runs"
OUT = RUNS_DIR / "revision_r15_lstm_matched_crps.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L = 60
K_SUPPORT = 10
N_QUERY = 32
B7_CRPS_ARCHIVED = 0.0045499455067329105
B7_N_EVAL_ARCHIVED = 20


def evaluate_ridge_subset(features, counts, func_indices, ckpt):
    trainer = ANILMetaTrainer(
        body_type="tcn",
        head_type="ridge",
        in_features=features.shape[2],
        embedding_dim=64,
        n_quantiles=N_QUANTILES,
        n_horizons=1,
        device=DEVICE,
    )
    trainer.load(ckpt)
    trainer.body.eval()

    quantiles = QUANTILES.to(DEVICE)
    T = features.shape[1]
    support_times = np.linspace(L, T // 2, K_SUPPORT, dtype=int)
    query_times = np.linspace(T // 2 + 1, T - 2, N_QUERY, dtype=int)
    per_func = []

    with torch.no_grad():
        for fi in func_indices:
            sx = torch.stack([
                torch.from_numpy(np.asarray(features[fi, t - L:t], dtype=np.float32))
                for t in support_times
            ]).to(DEVICE)
            sy = torch.tensor([
                np.log1p(counts[fi, t])
                for t in support_times
            ], dtype=torch.float32, device=DEVICE)
            qx = torch.stack([
                torch.from_numpy(np.asarray(features[fi, t - L:t], dtype=np.float32))
                for t in query_times
            ]).to(DEVICE)
            qy = torch.tensor([
                np.log1p(counts[fi, t])
                for t in query_times
            ], dtype=torch.float32, device=DEVICE)

            phi_s = trainer.body(sx)
            phi_q = trainer.body(qx)
            sy_exp = sy.unsqueeze(-1).expand(K_SUPPORT, N_QUANTILES)
            W = trainer.head.adapt(phi_s, sy_exp)
            pred = trainer.head.predict(phi_q, W)
            crps = crps_from_quantiles(pred[:, :N_QUANTILES], qy, quantiles)
            per_func.append(float(crps.item()))

    return {
        "checkpoint": str(ckpt),
        "mean": float(np.mean(per_func)),
        "std_functions": float(np.std(per_func, ddof=1)),
        "per_function": per_func,
    }


def main():
    torch.set_num_threads(1)
    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    splits = np.load(PROCESSED_DIR / "splits.npz")
    subset = np.asarray(splits["s2_test"][:B7_N_EVAL_ARCHIVED], dtype=int)

    per_seed = {}
    for seed in (0, 1, 2):
        ckpt = RUNS_DIR / f"best_anil_ridge_s2_s{seed}.pt"
        per_seed[str(seed)] = evaluate_ridge_subset(features, counts, subset, ckpt)

    seed_means = [v["mean"] for v in per_seed.values()]
    winter_mean = float(np.mean(seed_means))
    winter_std = float(np.std(seed_means, ddof=1))
    out = {
        "purpose": "matched first-20 S2 CRPS for Table S31 B7 comparison",
        "data_dir": str(PROCESSED_DIR),
        "runs_dir": str(RUNS_DIR),
        "split": "s2_test",
        "func_indices": [int(x) for x in subset],
        "support_times": [int(x) for x in np.linspace(L, features.shape[1] // 2, K_SUPPORT, dtype=int)],
        "query_times": [int(x) for x in np.linspace(features.shape[1] // 2 + 1, features.shape[1] - 2, N_QUERY, dtype=int)],
        "winter": {
            "crps_mean_over_seeds": winter_mean,
            "crps_std_over_seed_means": winter_std,
            "n_model_seeds": len(seed_means),
            "per_seed": per_seed,
        },
        "b7_lstm_archived": {
            "crps": B7_CRPS_ARCHIVED,
            "n_eval": B7_N_EVAL_ARCHIVED,
            "source": "results_azure2021/runs/extended_results.json",
        },
        "matched_ratio_winter_to_b7": winter_mean / B7_CRPS_ARCHIVED,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({
        "winter_crps": winter_mean,
        "winter_seed_std": winter_std,
        "b7_crps": B7_CRPS_ARCHIVED,
        "ratio": out["matched_ratio_winter_to_b7"],
        "out": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
