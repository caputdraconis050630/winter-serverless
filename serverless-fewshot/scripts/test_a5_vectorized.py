# -*- coding: utf-8 -*-
"""WP-C regression: vectorized rates_a5 must match the original loop version."""
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import rates_a5
from src.models.heads import N_QUANTILES
from src.meta.trainer import ANILMetaTrainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rates_a5_loop(counts, features, trainer, device):
    """Original per-function-loop implementation (reference)."""
    N, T = counts.shape
    L, d = 60, 64
    n_raw = features.shape[2]
    total_d = d + n_raw
    rates = np.zeros((N, T), dtype=np.float32)
    features_t = torch.from_numpy(features).float()
    buf_x = [[] for _ in range(N)]
    buf_y = [[] for _ in range(N)]
    func_w = [None] * N
    ewma = np.zeros(N)
    alpha = 0.15
    for t in range(min(L, T)):
        rates[:, t] = np.expm1(np.maximum(ewma, 0))
        prev = counts[:, max(t - 1, 0)].astype(np.float64)
        ewma = alpha * np.log1p(prev) + (1 - alpha) * ewma
    trainer.body.eval()
    with torch.no_grad():
        for t in range(L, T):
            bx = features_t[:, t - L:t, :].to(device)
            phi = trainer.body(bx).cpu().numpy()
            raw = features[:, min(t, T - 1), :]
            combined = np.concatenate([phi, raw], axis=1)
            prev = counts[:, max(t - 1, 0)].astype(np.float64)
            ewma = alpha * np.log1p(prev) + (1 - alpha) * ewma
            for fi in range(N):
                buf_x[fi].append(combined[fi])
                buf_y[fi].append(np.log1p(prev[fi]))
                if len(buf_x[fi]) > 180:
                    buf_x[fi] = buf_x[fi][-180:]
                    buf_y[fi] = buf_y[fi][-180:]
            if t % 60 == 0 and t > L + 30:
                for fi in range(N):
                    if len(buf_x[fi]) < 20:
                        continue
                    X = np.array(buf_x[fi]); y = np.array(buf_y[fi])
                    func_w[fi] = np.linalg.solve(
                        X.T @ X + 1.0 * np.eye(total_d), X.T @ y)
            for fi in range(N):
                if func_w[fi] is not None:
                    blended = 0.6 * float(combined[fi] @ func_w[fi]) + 0.4 * ewma[fi]
                    rates[fi, t] = np.expm1(max(blended, 0))
                else:
                    rates[fi, t] = np.expm1(max(ewma[fi], 0))
    return rates


def main():
    import time
    features = np.load("data/processed/features.npy")
    counts = np.load("data/processed/counts.npy")
    splits = np.load("data/processed/splits.npz")
    idx = splits["s1_test"][:10]          # 10 funcs, shortened horizon
    T = 3000
    sub_c = counts[idx, :T]
    sub_f = features[idx, :T]

    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1,
                              device=DEVICE)
    trainer.load(PROJECT_ROOT / "results/runs/best_anil_ridge_s1_s0.pt")

    t0 = time.time()
    r_loop = rates_a5_loop(sub_c, sub_f, trainer, DEVICE)
    t_loop = time.time() - t0
    t0 = time.time()
    r_vec = rates_a5(sub_c, sub_f, trainer, DEVICE)
    t_vec = time.time() - t0

    diff = np.abs(r_loop - r_vec)
    rel = diff / np.maximum(np.abs(r_loop), 1e-6)
    print(f"loop: {t_loop:.1f}s, vectorized: {t_vec:.1f}s "
          f"(speedup {t_loop/t_vec:.1f}x at N=10; scales with N)")
    print(f"max abs diff: {diff.max():.2e}, max rel diff: {rel.max():.2e}")
    assert diff.max() < 1e-3, "REGRESSION MISMATCH"
    print("PASS: vectorized rates_a5 matches loop implementation")


if __name__ == "__main__":
    main()
