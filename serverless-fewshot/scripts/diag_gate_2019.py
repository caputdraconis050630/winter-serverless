#!/usr/bin/env python3
"""Diagnose the 2019 GO/NO-GO PIVOT: K-sweep + per-function CRPS distribution.

Loads the seed-0 checkpoint and re-runs the gate evaluation protocol
(identical support/query placement and eval-cap sampling) at multiple K,
then correlates per-function CRPS with function traffic statistics.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase4_train import (
    PROCESSED_DIR, RUNS_DIR, DEVICE, N_QUANTILES, QUANTILES,
    cap_eval_indices, crps_from_quantiles,
)
from src.meta.trainer import ANILMetaTrainer
from src.models.bodies import build_body

K_SWEEP = (5, 10, 20, 50, 100)
HORIZONS = (1,)


def evaluate_k(trainer, features, counts, func_indices, k_support):
    quantiles = QUANTILES.to(DEVICE)
    n_horizons = len(HORIZONS)
    n_outputs = N_QUANTILES * n_horizons
    trainer.body.eval()
    L = 60
    T = features.shape[1]

    def win(fi, t):
        return torch.from_numpy(np.asarray(features[fi, t - L:t], dtype=np.float32))

    crps_scores = []
    with torch.no_grad():
        for fi in func_indices:
            support_times = np.linspace(L, T // 2, k_support, dtype=int)
            query_times = np.linspace(T // 2 + 1, T - max(HORIZONS) - 1, 32, dtype=int)

            sx = torch.stack([win(fi, t) for t in support_times]).to(DEVICE)
            sy = torch.stack([
                torch.tensor([np.log1p(counts[fi, min(t + h - 1, T - 1)]) for h in HORIZONS])
                for t in support_times]).float().to(DEVICE)
            qx = torch.stack([win(fi, t) for t in query_times]).to(DEVICE)
            qy = torch.stack([
                torch.tensor([np.log1p(counts[fi, min(t + h - 1, T - 1)]) for h in HORIZONS])
                for t in query_times]).float().to(DEVICE)

            phi_s = trainer.body(sx)
            phi_q = trainer.body(qx)
            sy_exp = sy.unsqueeze(-1).expand(k_support, n_horizons, N_QUANTILES)
            sy_exp = sy_exp.reshape(k_support, n_outputs)
            W = trainer.head.adapt(phi_s, sy_exp)
            pred = trainer.head.predict(phi_q, W)
            crps = crps_from_quantiles(pred[:, :N_QUANTILES], qy[:, 0], quantiles)
            crps_scores.append(crps.item())
    return np.array(crps_scores)


def evaluate_b5(body, head, features, counts, func_indices):
    quantiles = QUANTILES.to(DEVICE)
    body.eval()
    head.eval()
    L = 60
    T = features.shape[1]
    crps_scores = []
    with torch.no_grad():
        for fi in func_indices:
            query_times = np.linspace(T // 2 + 1, T - max(HORIZONS) - 1, 32, dtype=int)
            qx = torch.stack([
                torch.from_numpy(np.asarray(features[fi, t - L:t], dtype=np.float32))
                for t in query_times]).to(DEVICE)
            qy = torch.tensor([np.log1p(counts[fi, min(t, T - 1)])
                               for t in query_times]).float().to(DEVICE)
            phi = body(qx)
            pred = head(phi)
            crps = crps_from_quantiles(pred[:, :N_QUANTILES], qy, quantiles)
            crps_scores.append(crps.item())
    return np.array(crps_scores)


def main():
    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    splits = np.load(PROCESSED_DIR / "splits.npz")
    func_df = pd.read_csv(PROCESSED_DIR / "active_functions.csv")

    test_idx = cap_eval_indices(splits["s2_test"])
    print(f"Eval functions: {len(test_idx)} (same cap seed as gate)")

    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=len(HORIZONS),
        use_amp=True, device=DEVICE,
    )
    trainer.load(RUNS_DIR / "best_anil_ridge_s2_s0.pt")
    lam = trainer.head.ridge_lambda.item()
    print(f"Loaded seed-0 checkpoint; ridge lambda = {lam:.4f}")

    out = {"ridge_lambda": lam, "n_eval": int(len(test_idx)), "k_sweep": {}}
    per_k = {}
    for k in K_SWEEP:
        scores = evaluate_k(trainer, features, counts, test_idx, k)
        per_k[k] = scores
        out["k_sweep"][str(k)] = {
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "p90": float(np.percentile(scores, 90)),
            "p99": float(np.percentile(scores, 99)),
            "max": float(scores.max()),
        }
        print(f"K={k:3d}: mean={scores.mean():.4f} median={np.median(scores):.4f} "
              f"p90={np.percentile(scores, 90):.4f} p99={np.percentile(scores, 99):.4f} "
              f"max={scores.max():.4f}")

    # Correlate K=10 per-function CRPS with traffic stats (row order of
    # active_functions.csv matches feature/count row indices)
    s10 = per_k[10]
    stats = func_df.iloc[test_idx]
    rate_col = next((c for c in ("mean_rate_per_min", "rate_per_min", "mean_rate")
                     if c in stats.columns), None)
    if rate_col is not None:
        rates = stats[rate_col].to_numpy()
        top = np.argsort(s10)[::-1][:20]
        out["worst20_k10"] = [
            {"func_row": int(test_idx[i]), "crps": float(s10[i]),
             "rate_per_min": float(rates[i])}
            for i in top
        ]
        # share of mean CRPS carried by worst 5% of functions
        n5 = max(1, int(0.05 * len(s10)))
        worst5_share = float(np.sort(s10)[::-1][:n5].sum() / s10.sum())
        out["worst5pct_share_of_total_crps_k10"] = worst5_share
        log_rates = np.log10(rates + 1e-6)
        out["spearman_crps_vs_lograte_k10"] = float(
            pd.Series(s10).corr(pd.Series(log_rates), method="spearman"))
        print(f"\nworst-5% functions carry {worst5_share*100:.1f}% of total CRPS (K=10)")
        print(f"Spearman(CRPS, log10 rate) = {out['spearman_crps_vs_lograte_k10']:.3f}")

    # ---- Paired comparison vs B5 on the same functions ----
    ckpt = torch.load(RUNS_DIR / "b5_global_s0.pt", map_location=DEVICE,
                      weights_only=False)
    b5_body = build_body("tcn", features.shape[2]).to(DEVICE)
    b5_head = torch.nn.Linear(64, N_QUANTILES * len(HORIZONS)).to(DEVICE)
    b5_body.load_state_dict(ckpt["body"])
    b5_head.load_state_dict(ckpt["head"])
    b5 = evaluate_b5(b5_body, b5_head, features, counts, test_idx)

    diff = s10 - b5  # positive = ours worse
    from scipy import stats as sstats
    w_stat, w_p = sstats.wilcoxon(s10, b5)
    out["paired_vs_b5"] = {
        "b5_mean": float(b5.mean()), "b5_median": float(np.median(b5)),
        "b5_p90": float(np.percentile(b5, 90)), "b5_p99": float(np.percentile(b5, 99)),
        "ours_k10_mean": float(s10.mean()), "ours_k10_median": float(np.median(s10)),
        "ours_win_rate": float((diff < 0).mean()),
        "median_diff": float(np.median(diff)),
        "wilcoxon_p": float(w_p),
    }
    print(f"\nB5 per-function: mean={b5.mean():.4f} median={np.median(b5):.4f} "
          f"p90={np.percentile(b5, 90):.4f} p99={np.percentile(b5, 99):.4f}")
    print(f"Ours(K=10) win rate vs B5: {(diff < 0).mean()*100:.1f}% of functions, "
          f"median diff={np.median(diff):+.4f}, Wilcoxon p={w_p:.2e}")

    # Stratify the paired diff by traffic rate quartile
    if rate_col is not None:
        q = pd.qcut(np.log10(rates + 1e-6), 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])
        strat = {}
        for name in q.categories:
            m = np.asarray(q == name)
            strat[str(name)] = {
                "n": int(m.sum()),
                "ours_mean": float(s10[m].mean()), "b5_mean": float(b5[m].mean()),
                "ours_median": float(np.median(s10[m])), "b5_median": float(np.median(b5[m])),
                "win_rate": float((diff[m] < 0).mean()),
            }
            print(f"  {name}: n={m.sum()} ours_mean={s10[m].mean():.4f} "
                  f"b5_mean={b5[m].mean():.4f} win_rate={(diff[m] < 0).mean()*100:.0f}%")
        out["paired_by_rate_quartile"] = strat

    out_path = RUNS_DIR / "diag_gate2019_ksweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
