# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Phase 6: Trace-Driven Simulation.

Runs vectorized simulation with all methods across rho-sweep.
Key: prewarm decision = P(arrival in next tick) vs newsvendor cost threshold.
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.heads import N_QUANTILES, QUANTILES
from src.meta.trainer import ANILMetaTrainer
from src.decision.newsvendor import newsvendor_quantile

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prewarm_from_rate(pred_rate, tau_star):
    """Newsvendor-optimal binary prewarm + concurrency sizing.

    Prewarm >= 1 if P(arrival in next tick) > cost threshold (1-tau*).
    Scale up for high concurrency via Poisson quantile.
    """
    if pred_rate <= 0:
        return 0
    # P(at least 1 arrival) = 1 - e^(-rate)
    p_arrival = 1.0 - np.exp(-pred_rate)
    cost_threshold = 1.0 - tau_star
    if p_arrival <= cost_threshold:
        return 0
    # At least 1 container; for high rates, use Poisson quantile
    from scipy import stats as sp_stats
    return max(1, int(sp_stats.poisson.ppf(tau_star, pred_rate)))


def keepalive_from_rate(pred_rate, tau_star):
    """Newsvendor-optimal keepalive: keep alive while expected benefit > cost.

    Scale by tau_star: high tau = cold starts expensive = keep longer.
    Minimum keepalive ensures pool survives brief traffic gaps.
    """
    # Minimum keepalive to ride out brief gaps (tau-responsive)
    # Low rho (tau~0.1): ~5.7 min (less WM than B1's fixed 10)
    # rho=1 (tau~0.5): ~9 min (comparable to B1)
    # High rho (tau~0.9): ~12 min (fewer cold starts than B1)
    min_ka = 5.0 + tau_star * 8.0

    if pred_rate < 0.001:
        return max(2.0, min_ka * 0.5)
    expected_iat = 1.0 / pred_rate
    ka = expected_iat * (0.3 + tau_star * 1.2)
    return max(min_ka, min(ka, 30.0))


def fast_simulate(counts, prewarm_matrix, keepalive_matrix, dur_means, dur_stds,
                  memory_mb=256, seed=0):
    """Fast vectorized simulation."""
    np.random.seed(seed)
    N, T = counts.shape

    warm_pool = np.zeros(N, dtype=np.float64)
    last_use = np.full(N, -1000.0)

    total_cold = 0
    total_inv = 0
    total_idle_mem_sec = 0.0
    total_busy_mem_sec = 0.0
    func_cold = np.zeros(N, dtype=np.int64)
    func_total = np.zeros(N, dtype=np.int64)
    latencies = []

    for t in range(T):
        arrivals = counts[:, t]

        # Evict containers past keep-alive
        idle_duration = t - last_use
        ka = keepalive_matrix[:, min(t, T - 1)]
        evict_mask = idle_duration > ka
        idle_before_evict = warm_pool.copy()
        warm_pool[evict_mask] = 0

        # Record wasted memory for evicted containers
        evict_funcs = np.where(evict_mask & (idle_before_evict > 0))[0]
        for fi in evict_funcs:
            idle_mins = min(idle_duration[fi], ka[fi] + 1)  # Cap to keepalive
            total_idle_mem_sec += idle_before_evict[fi] * idle_mins * 60 * memory_mb / 1024

        # Set prewarm target
        target = prewarm_matrix[:, min(t, T - 1)]
        warm_pool = np.maximum(warm_pool, target)

        # Process arrivals
        active = arrivals > 0
        active_funcs = np.where(active)[0]

        for fi in active_funcs:
            n_arr = int(arrivals[fi])
            func_total[fi] += n_arr
            total_inv += n_arr

            n_warm = int(warm_pool[fi])
            n_cold = max(0, n_arr - n_warm)

            func_cold[fi] += n_cold
            total_cold += n_cold

            # Warm invocations
            n_warm_used = min(n_arr, n_warm)
            for _ in range(n_warm_used):
                dur = max(0.01, np.random.normal(dur_means[fi], max(dur_stds[fi], 0.01)))
                latencies.append(dur)
                total_busy_mem_sec += dur * memory_mb / 1024

            # Cold invocations
            for _ in range(n_cold):
                cold_lat = np.random.lognormal(np.log(1.0), 0.3)
                dur = max(0.01, np.random.normal(dur_means[fi], max(dur_stds[fi], 0.01)))
                latencies.append(cold_lat + dur)
                total_busy_mem_sec += (cold_lat + dur) * memory_mb / 1024

            # Update warm pool: containers stay warm after use
            warm_pool[fi] = max(warm_pool[fi], n_arr)
            last_use[fi] = t

        # Accumulate idle memory for non-active functions with warm containers
        idle_funcs = np.where((~active) & (warm_pool > 0))[0]
        total_idle_mem_sec += np.sum(warm_pool[idle_funcs]) * 60 * memory_mb / 1024

    lats = np.array(latencies) if latencies else np.array([0])
    csr = total_cold / max(1, total_inv)
    wm_per_1k = total_idle_mem_sec / max(1, total_inv / 1000)
    wm_frac = total_idle_mem_sec / max(1e-10, total_idle_mem_sec + total_busy_mem_sec)

    return {
        "csr": csr,
        "cold_starts": int(total_cold),
        "total_invocations": int(total_inv),
        "wm_total_gb_s": total_idle_mem_sec,
        "wm_per_1k_inv": wm_per_1k,
        "wm_fraction": wm_frac,
        "latency_p50": float(np.percentile(lats, 50)),
        "latency_p95": float(np.percentile(lats, 95)),
        "latency_p99": float(np.percentile(lats, 99)),
    }


def compute_prewarm_keepalive_matrices(method, counts, features, dur_means,
                                        tau_star, trainer=None, quantile_levels=None):
    """Compute prewarm and keep-alive matrices for a method."""
    N, T = counts.shape
    prewarm = np.zeros((N, T), dtype=np.int32)
    keepalive = np.full((N, T), 10.0, dtype=np.float32)

    if method == "B1_fixed_keepalive":
        # Fixed 10-min keep-alive, prewarm 1 if recent activity (policy baseline)
        for fi in range(N):
            for t in range(T):
                recent = counts[fi, max(0, t - 10):t].sum()
                prewarm[fi, t] = 1 if recent > 0 else 0
                keepalive[fi, t] = 10.0

    elif method == "B2_histogram":
        # Histogram-based (Shahrad et al., ATC'20): rolling 4h IAT histogram
        for fi in range(N):
            for t in range(T):
                window = counts[fi, max(0, t - 240):t]
                if window.sum() > 2:
                    nonzero = np.nonzero(window)[0]
                    if len(nonzero) > 1:
                        iats = np.diff(nonzero)
                        ka = float(min(np.percentile(iats, 95), 30.0))
                        pw = 1 if np.median(iats) < 5 else 0
                    else:
                        ka = 10.0
                        pw = 1
                else:
                    recent = counts[fi, max(0, t - 30):t].sum()
                    ka = 5.0
                    pw = 1 if recent > 0 else 0
                prewarm[fi, t] = pw
                keepalive[fi, t] = ka

    elif method == "B4a_ewma":
        # EWMA predictor with newsvendor-based prewarm/keepalive
        alpha = 0.1
        state = np.zeros(N)
        for t in range(T):
            rates = np.log1p(counts[:, t].astype(np.float64))
            state = alpha * rates + (1 - alpha) * state
            pred_rates = np.expm1(np.maximum(state, 0))
            for fi in range(N):
                prewarm[fi, t] = prewarm_from_rate(pred_rates[fi], tau_star)
                keepalive[fi, t] = keepalive_from_rate(pred_rates[fi], tau_star)

    elif method == "Oracle":
        # Oracle: perfect next-tick knowledge
        for fi in range(N):
            for t in range(T):
                actual = float(counts[fi, t])
                prewarm[fi, t] = prewarm_from_rate(actual, tau_star)
                # Oracle also looks ahead for keepalive
                if actual > 0:
                    keepalive[fi, t] = keepalive_from_rate(actual, tau_star)
                else:
                    # Look ahead to decide keepalive
                    future_window = counts[fi, t + 1:min(t + 30, T)]
                    if future_window.sum() > 0:
                        next_t = np.argmax(future_window > 0) + 1
                        keepalive[fi, t] = float(next_t + 0.5)
                    else:
                        keepalive[fi, t] = 1.0

    elif method == "A5_full_system" and trainer is not None:
        # Full system: reactive (recent activity) + proactive (learned predictions)
        # Reactive base: like B1, prewarm if recent activity
        # Proactive enhancement: model predictions for upcoming activity & keepalive
        L = 60
        d = 64
        n_raw_feat = features.shape[2]
        total_d = d + n_raw_feat
        adapt_interval = 60
        features_t = torch.from_numpy(features).float()

        buf_x = [[] for _ in range(N)]
        buf_y = [[] for _ in range(N)]
        func_w = [None] * N
        func_adapted = np.zeros(N, dtype=bool)

        ewma_state = np.zeros(N)
        ewma_alpha = 0.15

        # Phase 1: EWMA-only for first L ticks
        for t in range(min(L, T)):
            prev = counts[:, max(t - 1, 0)].astype(np.float64)
            ewma_state = ewma_alpha * np.log1p(prev) + (1 - ewma_alpha) * ewma_state
            for fi in range(N):
                recent = counts[fi, max(0, t - 10):t].sum()
                pred_rate = np.expm1(max(ewma_state[fi], 0))
                # Reactive base + predictive enhancement
                pw = 1 if recent > 0 else 0
                if pred_rate > 0.3:
                    pw = max(pw, 1)
                prewarm[fi, t] = pw
                keepalive[fi, t] = keepalive_from_rate(pred_rate, tau_star)

        # Phase 2: Body embeddings + ridge adaptation
        trainer.body.eval()
        with torch.no_grad():
            for t in range(L, T):
                batch_x = features_t[:, t - L:t, :].to(DEVICE)
                phi = trainer.body(batch_x).cpu().numpy()
                raw_feat = features[:, min(t, T - 1), :]
                combined = np.concatenate([phi, raw_feat], axis=1)

                prev = counts[:, max(t - 1, 0)].astype(np.float64)
                ewma_state = ewma_alpha * np.log1p(prev) + (1 - ewma_alpha) * ewma_state

                for fi in range(N):
                    buf_x[fi].append(combined[fi])
                    buf_y[fi].append(np.log1p(prev[fi]))
                    if len(buf_x[fi]) > 180:
                        buf_x[fi] = buf_x[fi][-180:]
                        buf_y[fi] = buf_y[fi][-180:]

                if t % adapt_interval == 0 and t > L + 30:
                    for fi in range(N):
                        if len(buf_x[fi]) < 20:
                            continue
                        X = np.array(buf_x[fi])
                        y = np.array(buf_y[fi])
                        lam = 1.0
                        try:
                            w = np.linalg.solve(X.T @ X + lam * np.eye(total_d), X.T @ y)
                            func_w[fi] = w
                            func_adapted[fi] = True
                        except np.linalg.LinAlgError:
                            pass

                for fi in range(N):
                    if func_adapted[fi] and func_w[fi] is not None:
                        ridge_pred = combined[fi] @ func_w[fi]
                        blended = 0.6 * ridge_pred + 0.4 * ewma_state[fi]
                        pred_rate = np.expm1(max(blended, 0))
                    else:
                        pred_rate = np.expm1(max(ewma_state[fi], 0))

                    # Reactive base: prewarm if recent activity (like B1)
                    recent = counts[fi, max(0, t - 10):t].sum()
                    pw = 1 if recent > 0 else 0

                    # Proactive enhancement: model predicts upcoming activity
                    if pred_rate > 0.3:
                        pw = max(pw, 1)
                    # Concurrency scaling for high-rate functions
                    if pred_rate > 2.0:
                        pw = max(pw, prewarm_from_rate(pred_rate, tau_star))
                    prewarm[fi, t] = pw

                    # Adaptive keepalive: tau-responsive, predictive
                    keepalive[fi, t] = keepalive_from_rate(pred_rate, tau_star)

                if t % 5000 == 0:
                    print(f"    tick {t}/{T} ({func_adapted.sum()}/{N} adapted)", flush=True)

    elif method == "B5_global" and trainer is not None:
        # Global model: body embeddings, single shared head, no per-function adapt
        L = 60
        features_t = torch.from_numpy(features).float()
        trainer.body.eval()

        with torch.no_grad():
            all_phi = []
            all_y = []
            for t in range(L, min(L + 1000, T)):
                batch_x = features_t[:, t - L:t, :].to(DEVICE)
                phi_b = trainer.body(batch_x).cpu()
                y = torch.from_numpy(np.log1p(counts[:, t])).float()
                all_phi.append(phi_b)
                all_y.append(y)

            Phi_global = torch.cat(all_phi, dim=0)
            Y_global = torch.cat(all_y, dim=0).unsqueeze(-1).expand(-1, N_QUANTILES)
            PhiTPhi = Phi_global.T @ Phi_global
            PhiTY = Phi_global.T @ Y_global
            lam = trainer.head.ridge_lambda.item()
            A = PhiTPhi + lam * torch.eye(64)
            W_global = torch.linalg.solve(A, PhiTY).to(DEVICE)

            ewma_b5 = np.zeros(N)
            for t in range(min(L, T)):
                prev = counts[:, max(t - 1, 0)].astype(np.float64)
                ewma_b5 = 0.1 * np.log1p(prev) + 0.9 * ewma_b5
                for fi in range(N):
                    pr = np.expm1(max(ewma_b5[fi], 0))
                    prewarm[fi, t] = prewarm_from_rate(pr, tau_star)
                    keepalive[fi, t] = keepalive_from_rate(pr, tau_star)

            for t in range(L, T):
                batch_x = features_t[:, t - L:t, :].to(DEVICE)
                phi_b = trainer.body(batch_x)
                pred = (phi_b @ W_global).cpu().numpy()

                for fi in range(N):
                    pred_rate = np.expm1(max(
                        np.interp(tau_star, quantile_levels, pred[fi]), 0
                    ))
                    prewarm[fi, t] = prewarm_from_rate(pred_rate, tau_star)
                    keepalive[fi, t] = keepalive_from_rate(pred_rate, tau_star)

                if t % 5000 == 0:
                    print(f"    tick {t}/{T}", flush=True)

    return prewarm, keepalive


def main():
    print("=" * 60)
    print("PHASE 6: Trace-Driven Simulation")
    print("=" * 60)

    features = np.load(PROCESSED_DIR / "features.npy")
    counts = np.load(PROCESSED_DIR / "counts.npy")
    splits_data = np.load(PROCESSED_DIR / "splits.npz")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")

    test_indices = splits_data["s1_test"]
    sub_counts = counts[test_indices]
    sub_features = features[test_indices]
    N_test = len(test_indices)

    dur_means = np.array([dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
                          for fi in test_indices])
    dur_stds = np.array([dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
                         for fi in test_indices])
    dur_means = np.nan_to_num(dur_means, nan=1.0)
    dur_stds = np.nan_to_num(dur_stds, nan=0.5)

    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE,
    )
    model_path = RUNS_DIR / "best_anil_ridge_s1_s0.pt"
    if not model_path.exists():
        model_path = RUNS_DIR / "best_anil_ridge_s2_s0.pt"
    if model_path.exists():
        trainer.load(model_path)
        print(f"Loaded model from {model_path}")

    print(f"Simulating {N_test} test functions, {sub_counts.shape[1]} ticks")

    quantile_levels = np.linspace(0.05, 0.95, N_QUANTILES)

    methods = ["B1_fixed_keepalive", "B2_histogram", "B4a_ewma",
               "A5_full_system", "B5_global", "Oracle"]

    cost_ratios = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0]

    all_results = []

    for method in methods:
        print(f"\n--- {method} ---")

        for rho in cost_ratios:
            tau = newsvendor_quantile(rho)
            print(f"  rho={rho:6.1f} (tau*={tau:.3f})", end=" ", flush=True)

            t0 = time.time()

            prewarm_mat, keepalive_mat = compute_prewarm_keepalive_matrices(
                method, sub_counts, sub_features, dur_means,
                tau, trainer=trainer, quantile_levels=quantile_levels,
            )

            metrics = fast_simulate(
                sub_counts, prewarm_mat, keepalive_mat, dur_means, dur_stds, seed=0
            )
            elapsed = time.time() - t0

            metrics["method"] = method
            metrics["cost_ratio"] = rho
            metrics["elapsed_sec"] = elapsed
            all_results.append(metrics)

            print(f"CSR={metrics['csr']:.4f} WM={metrics['wm_per_1k_inv']:.1f} "
                  f"P95={metrics['latency_p95']:.2f}s [{elapsed:.1f}s]")

    # Save all results
    results_path = RUNS_DIR / "sim_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    # Build T3: Main comparison table at rho=10
    rho_main = 10.0
    print(f"\n{'='*70}")
    print(f"TABLE T3: Main Comparison (rho={rho_main})")
    print(f"{'='*70}")
    print(f"{'Method':<22s} {'CSR':>8s} {'ColdStarts':>10s} {'WM/1k':>10s} {'P50':>6s} {'P95':>6s} {'P99':>6s}")
    print("-" * 70)
    for r in all_results:
        if abs(r["cost_ratio"] - rho_main) < 0.01:
            print(f"{r['method']:<22s} {r['csr']:>8.4f} {r['cold_starts']:>10d} "
                  f"{r['wm_per_1k_inv']:>10.1f} {r['latency_p50']:>6.3f} "
                  f"{r['latency_p95']:>6.3f} {r['latency_p99']:>6.3f}")

    t3_data = [r for r in all_results if abs(r["cost_ratio"] - rho_main) < 0.01]
    t3_df = pd.DataFrame(t3_data)
    t3_cols = ["method", "csr", "cold_starts", "total_invocations",
               "wm_per_1k_inv", "wm_fraction", "latency_p50", "latency_p95", "latency_p99"]
    t3_df[[c for c in t3_cols if c in t3_df.columns]].to_csv(
        TABLES_DIR / "T3_main_comparison.csv", index=False)
    print(f"\nSaved T3 to T3_main_comparison.csv")

    print("\n" + "=" * 60)
    print("Phase 6 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
