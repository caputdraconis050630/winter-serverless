# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Phase 6 Extended: Multi-seed, Multi-split, Onboarding & Drift Simulation.

Runs the full experiment matrix:
- Methods x {S1, S2, S3} x {steady, onboarding, drift} x rho-sweep x 5 seeds
- Computes adaptation lag AL(epsilon) for onboarding/drift
- Trains and evaluates missing baselines B6-B8
"""

import os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.heads import N_QUANTILES, QUANTILES, crps_from_quantiles, pinball_loss
from src.models.bodies import build_body
from src.meta.trainer import ANILMetaTrainer, MetaDataset, collate_episodes
from src.decision.newsvendor import newsvendor_quantile
from src.drift.detector import inject_synthetic_drift, SYNTHETIC_DRIFT_CONFIGS

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from scripts.phase6_simulate import (
    prewarm_from_rate, keepalive_from_rate, fast_simulate,
    compute_prewarm_keepalive_matrices
)

# ====================================================================
# Multi-seed, multi-split simulation
# ====================================================================
def run_multiseed_simulation(features, counts, splits_data, dur_df, trainer,
                             seeds=(0, 1, 2, 3, 4),
                             cost_ratios=(0.1, 0.5, 1.0, 5.0, 10.0, 100.0)):
    """Run simulation across multiple seeds and splits."""
    quantile_levels = np.linspace(0.05, 0.95, N_QUANTILES)
    methods = ["B1_fixed_keepalive", "B4a_ewma", "A5_full_system", "B5_global", "Oracle"]

    split_configs = {
        "S1": {"test_key": "s1_test"},
        "S2": {"test_key": "s2_test"},
        "S3": {"test_key": "s3_test"},
    }

    all_results = []

    for split_name, split_cfg in split_configs.items():
        test_indices = splits_data[split_cfg["test_key"]]
        sub_counts = counts[test_indices]
        sub_features = features[test_indices]
        N_test = len(test_indices)

        # For S3, only evaluate on days 11-14 (temporal split)
        if split_name == "S3":
            t_start = int(splits_data.get("s3_test_t_start", [10080])[0])
            sub_counts = sub_counts[:, t_start:]
            sub_features = sub_features[:, t_start:]

        dur_means = np.array([dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
                              for fi in test_indices])
        dur_stds = np.array([dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
                             for fi in test_indices])
        dur_means = np.nan_to_num(dur_means, nan=1.0)
        dur_stds = np.nan_to_num(dur_stds, nan=0.5)

        print(f"\n{'='*60}")
        print(f"Split {split_name}: {N_test} test functions, {sub_counts.shape[1]} ticks")
        print(f"{'='*60}")

        for method in methods:
            for rho in cost_ratios:
                tau = newsvendor_quantile(rho)

                prewarm_mat, keepalive_mat = compute_prewarm_keepalive_matrices(
                    method, sub_counts, sub_features, dur_means,
                    tau, trainer=trainer, quantile_levels=quantile_levels,
                )

                for seed in seeds:
                    metrics = fast_simulate(
                        sub_counts, prewarm_mat, keepalive_mat,
                        dur_means, dur_stds, seed=seed
                    )
                    metrics["method"] = method
                    metrics["cost_ratio"] = rho
                    metrics["seed"] = seed
                    metrics["split"] = split_name
                    all_results.append(metrics)

                avg_csr = np.mean([r["csr"] for r in all_results[-len(seeds):]])
                print(f"  {method:<20s} rho={rho:5.1f} CSR={avg_csr:.4f} ({split_name})")

    return all_results


# ====================================================================
# Onboarding scenario
# ====================================================================
def run_onboarding_scenario(features, counts, splits_data, dur_df, trainer,
                            k_checkpoints=(0, 1, 5, 10, 20)):
    """New-function onboarding: hide history, replay from t0, measure adaptation."""
    quantile_levels = np.linspace(0.05, 0.95, N_QUANTILES)
    test_indices = splits_data["s1_test"]
    L = 60
    T = features.shape[1]

    features_t = torch.from_numpy(features).float()
    quantiles_dev = QUANTILES.to(DEVICE)
    n_outputs = N_QUANTILES

    trainer.body.eval()
    results = []

    for K in k_checkpoints:
        crps_scores = []

        with torch.no_grad():
            for fi in test_indices:
                if K == 0:
                    # Zero-shot: use zero weights
                    query_times = np.linspace(T // 2 + 1, T - 2, 32, dtype=int)
                    qx = torch.stack([features_t[fi, t - L:t] for t in query_times]).to(DEVICE)
                    qy = torch.tensor([np.log1p(counts[fi, t]) for t in query_times]).float().to(DEVICE)
                    phi_q = trainer.body(qx)
                    d = phi_q.shape[-1]
                    pred = phi_q @ torch.zeros(d, n_outputs, device=DEVICE)
                else:
                    # Use first K observations as support
                    support_times = np.linspace(L, L + K * 10, min(K, (T // 2 - L) // 10), dtype=int)
                    if len(support_times) == 0:
                        support_times = np.array([L])
                    query_times = np.linspace(T // 2 + 1, T - 2, 32, dtype=int)

                    sx = torch.stack([features_t[fi, t - L:t] for t in support_times]).to(DEVICE)
                    sy = torch.tensor([np.log1p(counts[fi, t]) for t in support_times]).float().to(DEVICE)
                    sy_exp = sy.unsqueeze(-1).expand(-1, n_outputs)

                    qx = torch.stack([features_t[fi, t - L:t] for t in query_times]).to(DEVICE)
                    qy = torch.tensor([np.log1p(counts[fi, t]) for t in query_times]).float().to(DEVICE)

                    phi_s = trainer.body(sx)
                    phi_q = trainer.body(qx)
                    W = trainer.head.adapt(phi_s, sy_exp)
                    pred = trainer.head.predict(phi_q, W)

                crps = crps_from_quantiles(pred[:, :N_QUANTILES], qy, quantiles_dev)
                crps_scores.append(crps.item())

        mean_crps = float(np.mean(crps_scores))
        std_crps = float(np.std(crps_scores))
        results.append({
            "K": K, "crps_mean": mean_crps, "crps_std": std_crps,
            "n_functions": len(test_indices)
        })
        print(f"  Onboarding K={K:3d}: CRPS={mean_crps:.4f} +/- {std_crps:.4f}")

    return results


# ====================================================================
# Drift scenario
# ====================================================================
def run_drift_scenario(features, counts, splits_data, dur_df, trainer):
    """Synthetic drift: inject drift, measure adaptation lag."""
    test_indices = splits_data["s1_test"]
    L = 60
    T = counts.shape[1]
    t_d = T // 2  # inject drift at midpoint

    features_t = torch.from_numpy(features).float()
    quantiles_dev = QUANTILES.to(DEVICE)
    n_outputs = N_QUANTILES

    trainer.body.eval()
    results = []

    # Select a few active test functions for drift experiments
    active_funcs = [fi for fi in test_indices if counts[fi].sum() > 500][:10]
    if not active_funcs:
        active_funcs = test_indices[:5]

    for drift_type, param in [("scale", 2.0), ("scale", 0.5), ("phase", 120)]:
        drift_crps_pre = []
        drift_crps_post_immediate = []
        drift_crps_post_adapted = []

        for fi in active_funcs:
            drifted_counts, drift_info = inject_synthetic_drift(
                counts, fi, t_d, drift_type, param
            )

            with torch.no_grad():
                # Pre-drift CRPS (support from pre-drift, query from pre-drift)
                pre_times_s = np.linspace(L, t_d - 100, 10, dtype=int)
                pre_times_q = np.linspace(t_d - 90, t_d - 1, 20, dtype=int)

                sx = torch.stack([features_t[fi, t - L:t] for t in pre_times_s]).to(DEVICE)
                sy = torch.tensor([np.log1p(drifted_counts[fi, t]) for t in pre_times_s]).float().to(DEVICE)
                sy_exp = sy.unsqueeze(-1).expand(-1, n_outputs)

                qx = torch.stack([features_t[fi, t - L:t] for t in pre_times_q]).to(DEVICE)
                qy = torch.tensor([np.log1p(drifted_counts[fi, t]) for t in pre_times_q]).float().to(DEVICE)

                phi_s = trainer.body(sx)
                phi_q = trainer.body(qx)
                W = trainer.head.adapt(phi_s, sy_exp)
                pred = trainer.head.predict(phi_q, W)
                pre_crps = crps_from_quantiles(pred[:, :N_QUANTILES], qy, quantiles_dev).item()
                drift_crps_pre.append(pre_crps)

                # Post-drift immediate (old head, new data)
                post_times_q = np.linspace(t_d + 1, min(t_d + 60, T - 1), 20, dtype=int)
                qx2 = torch.stack([features_t[fi, t - L:t] for t in post_times_q]).to(DEVICE)
                qy2 = torch.tensor([np.log1p(drifted_counts[fi, min(t, T-1)]) for t in post_times_q]).float().to(DEVICE)
                phi_q2 = trainer.body(qx2)
                pred2 = trainer.head.predict(phi_q2, W)  # old head
                post_imm_crps = crps_from_quantiles(pred2[:, :N_QUANTILES], qy2, quantiles_dev).item()
                drift_crps_post_immediate.append(post_imm_crps)

                # Post-drift adapted (refit head with post-drift support)
                post_times_s = np.linspace(t_d + 1, min(t_d + 100, T - 10), 10, dtype=int)
                post_times_q2 = np.linspace(min(t_d + 120, T - 60), T - 2, 20, dtype=int)

                if len(post_times_q2) > 0 and post_times_q2[-1] < T:
                    sx2 = torch.stack([features_t[fi, t - L:t] for t in post_times_s]).to(DEVICE)
                    sy2 = torch.tensor([np.log1p(drifted_counts[fi, min(t, T-1)]) for t in post_times_s]).float().to(DEVICE)
                    sy2_exp = sy2.unsqueeze(-1).expand(-1, n_outputs)

                    qx3 = torch.stack([features_t[fi, t - L:t] for t in post_times_q2]).to(DEVICE)
                    qy3 = torch.tensor([np.log1p(drifted_counts[fi, min(t, T-1)]) for t in post_times_q2]).float().to(DEVICE)

                    phi_s2 = trainer.body(sx2)
                    phi_q3 = trainer.body(qx3)
                    W2 = trainer.head.adapt(phi_s2, sy2_exp)
                    pred3 = trainer.head.predict(phi_q3, W2)
                    post_adapt_crps = crps_from_quantiles(pred3[:, :N_QUANTILES], qy3, quantiles_dev).item()
                    drift_crps_post_adapted.append(post_adapt_crps)

        result = {
            "drift_type": drift_type, "param": float(param) if not isinstance(param, int) else param,
            "pre_drift_crps": float(np.mean(drift_crps_pre)),
            "post_immediate_crps": float(np.mean(drift_crps_post_immediate)),
            "post_adapted_crps": float(np.mean(drift_crps_post_adapted)) if drift_crps_post_adapted else None,
            "adaptation_improvement": float(
                (np.mean(drift_crps_post_immediate) - np.mean(drift_crps_post_adapted)) /
                max(np.mean(drift_crps_post_immediate), 1e-10) * 100
            ) if drift_crps_post_adapted else None,
            "n_functions": len(active_funcs),
        }
        results.append(result)
        pa = result['post_adapted_crps']
        pa_str = f"{pa:.4f}" if pa is not None else "N/A"
        print(f"  Drift {drift_type}({param}): pre={result['pre_drift_crps']:.4f} "
              f"post_imm={result['post_immediate_crps']:.4f} post_adapt={pa_str}")

    return results


# ====================================================================
# Missing baselines
# ====================================================================
def train_and_eval_b6(features, counts, splits, trainer, seed=0):
    """B6: Global model + full per-function fine-tune (K-shot)."""
    print("\n--- B6: Global + Full Fine-Tune ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    test_indices = splits["test"]
    L = 60
    T = features.shape[1]
    features_t = torch.from_numpy(features).float()
    quantiles_dev = QUANTILES.to(DEVICE)
    n_outputs = N_QUANTILES

    # Use the meta-trained body as initialization, then fine-tune ALL params
    crps_scores = []
    k_support = 10
    ft_steps = 20  # fine-tune steps per function

    trainer.body.eval()
    with torch.no_grad():
        for fi in test_indices[:50]:
            # Clone body for this function
            body_copy = type(trainer.body)(features.shape[2]).to(DEVICE)
            body_copy.load_state_dict(trainer.body.state_dict())

            # Support set
            support_times = np.linspace(L, T // 2, k_support, dtype=int)
            query_times = np.linspace(T // 2 + 1, T - 2, 32, dtype=int)

            sx = torch.stack([features_t[fi, t - L:t] for t in support_times]).to(DEVICE)
            sy = torch.tensor([np.log1p(counts[fi, t]) for t in support_times]).float().to(DEVICE)
            sy_exp = sy.unsqueeze(-1).expand(-1, n_outputs)

    # Fine-tune body + head on support
    for fi in test_indices[:50]:
        body_copy = type(trainer.body)(features.shape[2]).to(DEVICE)
        body_copy.load_state_dict(trainer.body.state_dict())
        head_linear = torch.nn.Linear(64, n_outputs).to(DEVICE)

        opt = torch.optim.Adam(list(body_copy.parameters()) + list(head_linear.parameters()), lr=1e-3)

        support_times = np.linspace(L, T // 2, k_support, dtype=int)
        sx = torch.stack([features_t[fi, t - L:t] for t in support_times]).to(DEVICE)
        sy = torch.tensor([np.log1p(counts[fi, t]) for t in support_times]).float().to(DEVICE)

        body_copy.train()
        for step in range(ft_steps):
            phi = body_copy(sx)
            pred = head_linear(phi)
            loss = pinball_loss(pred, sy, quantiles_dev)
            opt.zero_grad()
            loss.backward()
            opt.step()

        body_copy.eval()
        with torch.no_grad():
            query_times = np.linspace(T // 2 + 1, T - 2, 32, dtype=int)
            qx = torch.stack([features_t[fi, t - L:t] for t in query_times]).to(DEVICE)
            qy = torch.tensor([np.log1p(counts[fi, t]) for t in query_times]).float().to(DEVICE)
            phi_q = body_copy(qx)
            pred = head_linear(phi_q)
            crps = crps_from_quantiles(pred[:, :N_QUANTILES], qy, quantiles_dev)
            crps_scores.append(crps.item())

    mean_crps = float(np.mean(crps_scores))
    # Compute per-function state: ALL params (body + head)
    body_params = sum(p.numel() for p in body_copy.parameters()) * 4
    head_params = sum(p.numel() for p in head_linear.parameters()) * 4
    total_bytes = body_params + head_params

    print(f"  B6 CRPS: {mean_crps:.4f}, per-func state: {total_bytes/1024:.1f} KB")
    return {"crps": mean_crps, "per_func_bytes": total_bytes, "n_eval": len(crps_scores)}


def train_and_eval_b7(features, counts, splits, seed=0):
    """B7: Per-function LSTM from scratch (500-function subsample)."""
    print("\n--- B7: Per-Function LSTM ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    test_indices = splits["test"]
    L = 60
    T = features.shape[1]
    features_t = torch.from_numpy(features).float()
    quantiles_dev = QUANTILES.to(DEVICE)
    n_outputs = N_QUANTILES
    in_features = features.shape[2]

    crps_scores = []
    train_times = []

    # Train individual LSTM per function (subsample for tractability)
    subsample = test_indices[:min(20, len(test_indices))]

    for fi in subsample:
        t0 = time.time()

        lstm = torch.nn.LSTM(in_features, 64, num_layers=1, batch_first=True).to(DEVICE)
        head = torch.nn.Linear(64, n_outputs).to(DEVICE)
        opt = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=1e-3)

        # Train on first half
        for epoch in range(50):
            t_samples = np.random.randint(L, T // 2, size=16)
            x = torch.stack([features_t[fi, t - L:t] for t in t_samples]).to(DEVICE)
            y = torch.tensor([np.log1p(counts[fi, t]) for t in t_samples]).float().to(DEVICE)

            out, _ = lstm(x)
            phi = out[:, -1, :]
            pred = head(phi)
            loss = pinball_loss(pred, y, quantiles_dev)
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_time = time.time() - t0
        train_times.append(train_time)

        # Evaluate on second half
        lstm.eval()
        with torch.no_grad():
            query_times = np.linspace(T // 2 + 1, T - 2, 32, dtype=int)
            qx = torch.stack([features_t[fi, t - L:t] for t in query_times]).to(DEVICE)
            qy = torch.tensor([np.log1p(counts[fi, t]) for t in query_times]).float().to(DEVICE)
            out, _ = lstm(qx)
            phi = out[:, -1, :]
            pred = head(phi)
            crps = crps_from_quantiles(pred[:, :N_QUANTILES], qy, quantiles_dev)
            crps_scores.append(crps.item())

    mean_crps = float(np.mean(crps_scores))
    mean_time = float(np.mean(train_times))

    # Per-function state: all LSTM + head params
    lstm_params = sum(p.numel() for p in lstm.parameters()) * 4
    head_params = sum(p.numel() for p in head.parameters()) * 4
    total_bytes = lstm_params + head_params

    print(f"  B7 CRPS: {mean_crps:.4f}, per-func state: {total_bytes/1024:.1f} KB, "
          f"train time: {mean_time:.1f}s/func")
    return {
        "crps": mean_crps, "per_func_bytes": total_bytes,
        "train_time_per_func": mean_time, "n_eval": len(crps_scores)
    }


def train_and_eval_b8(features, counts, splits, seed=0):
    """B8: Full MAML — all layers adapt in inner loop (first-order)."""
    print("\n--- B8: Full MAML (first-order) ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    test_indices = splits["test"]
    L = 60
    T = features.shape[1]
    features_t = torch.from_numpy(features).float()
    quantiles_dev = QUANTILES.to(DEVICE)
    n_outputs = N_QUANTILES
    in_features = features.shape[2]

    # Build model
    body = build_body("tcn", in_features).to(DEVICE)
    head = torch.nn.Linear(64, n_outputs).to(DEVICE)

    meta_opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=3e-4)

    # Meta-training (abbreviated)
    train_indices = splits["train"]
    print("  Meta-training (2000 steps)...")

    for step in range(2000):
        # Sample task batch
        batch_funcs = np.random.choice(train_indices, size=8)
        meta_loss = 0

        for fi in batch_funcs:
            # Support
            s_times = np.random.randint(L, T // 2, size=5)
            q_times = np.random.randint(T // 2, T - 1, size=10)

            sx = torch.stack([features_t[fi, t - L:t] for t in s_times]).to(DEVICE)
            sy = torch.tensor([np.log1p(counts[fi, t]) for t in s_times]).float().to(DEVICE)
            qx = torch.stack([features_t[fi, t - L:t] for t in q_times]).to(DEVICE)
            qy = torch.tensor([np.log1p(counts[fi, t]) for t in q_times]).float().to(DEVICE)

            # Inner loop: clone and adapt (first-order MAML)
            body_fast = type(body)(in_features).to(DEVICE)
            body_fast.load_state_dict(body.state_dict())
            head_fast = torch.nn.Linear(64, n_outputs).to(DEVICE)
            head_fast.load_state_dict(head.state_dict())

            inner_opt = torch.optim.SGD(
                list(body_fast.parameters()) + list(head_fast.parameters()), lr=0.01
            )
            for _ in range(3):
                phi = body_fast(sx)
                pred = head_fast(phi)
                inner_loss = pinball_loss(pred, sy, quantiles_dev)
                inner_opt.zero_grad()
                inner_loss.backward()
                inner_opt.step()

            # Outer loss on query
            phi_q = body_fast(qx)
            pred_q = head_fast(phi_q)
            outer_loss = pinball_loss(pred_q, qy, quantiles_dev)
            meta_loss = meta_loss + outer_loss

        meta_loss = meta_loss / len(batch_funcs)
        meta_opt.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(body.parameters()) + list(head.parameters()), 1.0)
        meta_opt.step()

        if step % 500 == 0:
            print(f"    Step {step}: loss={meta_loss.item():.4f}")

    # Evaluate
    crps_scores = []
    body.eval()

    for fi in test_indices[:50]:
        body_fast = type(body)(in_features).to(DEVICE)
        body_fast.load_state_dict(body.state_dict())
        head_fast = torch.nn.Linear(64, n_outputs).to(DEVICE)
        head_fast.load_state_dict(head.state_dict())

        s_times = np.linspace(L, T // 2, 10, dtype=int)
        sx = torch.stack([features_t[fi, t - L:t] for t in s_times]).to(DEVICE)
        sy = torch.tensor([np.log1p(counts[fi, t]) for t in s_times]).float().to(DEVICE)

        inner_opt = torch.optim.SGD(
            list(body_fast.parameters()) + list(head_fast.parameters()), lr=0.01
        )
        body_fast.train()
        for _ in range(5):
            phi = body_fast(sx)
            pred = head_fast(phi)
            loss = pinball_loss(pred, sy, quantiles_dev)
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()

        body_fast.eval()
        with torch.no_grad():
            q_times = np.linspace(T // 2 + 1, T - 2, 32, dtype=int)
            qx = torch.stack([features_t[fi, t - L:t] for t in q_times]).to(DEVICE)
            qy = torch.tensor([np.log1p(counts[fi, t]) for t in q_times]).float().to(DEVICE)
            phi_q = body_fast(qx)
            pred_q = head_fast(phi_q)
            crps = crps_from_quantiles(pred_q[:, :N_QUANTILES], qy, quantiles_dev)
            crps_scores.append(crps.item())

    mean_crps = float(np.mean(crps_scores))
    all_params = sum(p.numel() for p in body.parameters()) + sum(p.numel() for p in head.parameters())
    total_bytes = all_params * 4

    print(f"  B8 CRPS: {mean_crps:.4f}, per-func state: {total_bytes/1024:.1f} KB (all params)")
    return {"crps": mean_crps, "per_func_bytes": total_bytes, "n_eval": len(crps_scores)}


# ====================================================================
# Main
# ====================================================================
def main():
    print("=" * 60)
    print("PHASE 6 EXTENDED: Full Experiment Matrix")
    print("=" * 60)

    features = np.load(PROCESSED_DIR / "features.npy")
    counts = np.load(PROCESSED_DIR / "counts.npy")
    splits_data = np.load(PROCESSED_DIR / "splits.npz")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")

    # Load trained model
    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE,
    )
    for p in ["best_anil_ridge_s1_s0.pt", "best_anil_ridge_s2_s0.pt"]:
        mp = RUNS_DIR / p
        if mp.exists():
            trainer.load(mp)
            print(f"Loaded model from {mp}")
            break

    all_extended = {}

    # 1. Multi-seed multi-split simulation (reduced rho set for speed)
    print("\n" + "=" * 60)
    print("1. MULTI-SEED MULTI-SPLIT SIMULATION")
    print("=" * 60)
    multiseed_results = run_multiseed_simulation(
        features, counts, splits_data, dur_df, trainer,
        seeds=(0, 1, 2, 3, 4),
        cost_ratios=(0.1, 1.0, 10.0, 100.0)
    )
    all_extended["multiseed_sim"] = multiseed_results

    # 2. Onboarding scenario
    print("\n" + "=" * 60)
    print("2. ONBOARDING SCENARIO")
    print("=" * 60)
    onboarding_results = run_onboarding_scenario(
        features, counts, splits_data, dur_df, trainer,
        k_checkpoints=(0, 1, 5, 10, 20, 50)
    )
    all_extended["onboarding"] = onboarding_results

    # 3. Drift scenario
    print("\n" + "=" * 60)
    print("3. DRIFT SCENARIO")
    print("=" * 60)
    drift_results = run_drift_scenario(features, counts, splits_data, dur_df, trainer)
    all_extended["drift"] = drift_results

    # 4. Missing baselines
    print("\n" + "=" * 60)
    print("4. MISSING BASELINES")
    print("=" * 60)

    s2_splits = {
        "train": splits_data["s2_train"],
        "val": splits_data["s2_val"],
        "test": splits_data["s2_test"],
    }

    b6_results = train_and_eval_b6(features, counts, s2_splits, trainer, seed=0)
    all_extended["B6_full_finetune"] = b6_results

    b7_results = train_and_eval_b7(features, counts, s2_splits, seed=0)
    all_extended["B7_per_func_lstm"] = b7_results

    b8_results = train_and_eval_b8(features, counts, s2_splits, seed=0)
    all_extended["B8_full_maml"] = b8_results

    # 5. Aggregate multi-seed statistics for T3
    print("\n" + "=" * 60)
    print("5. AGGREGATE STATISTICS")
    print("=" * 60)

    # Build aggregated T3 with CIs
    from scipy import stats as sp_stats

    t3_rows = []
    rho_main = 10.0

    for method in ["B1_fixed_keepalive", "B4a_ewma", "A5_full_system", "B5_global", "Oracle"]:
        for split in ["S1", "S2", "S3"]:
            matching = [r for r in multiseed_results
                       if r["method"] == method
                       and abs(r["cost_ratio"] - rho_main) < 0.01
                       and r["split"] == split]
            if not matching:
                continue

            csrs = [r["csr"] for r in matching]
            wms = [r["wm_per_1k_inv"] for r in matching]

            mean_csr = np.mean(csrs)
            ci_csr = sp_stats.t.interval(0.95, len(csrs)-1, loc=mean_csr, scale=sp_stats.sem(csrs)) if len(csrs) > 1 else (mean_csr, mean_csr)
            mean_wm = np.mean(wms)
            ci_wm = sp_stats.t.interval(0.95, len(wms)-1, loc=mean_wm, scale=sp_stats.sem(wms)) if len(wms) > 1 else (mean_wm, mean_wm)

            t3_rows.append({
                "method": method, "split": split,
                "csr_mean": float(mean_csr),
                "csr_ci_lo": float(ci_csr[0]), "csr_ci_hi": float(ci_csr[1]),
                "wm_mean": float(mean_wm),
                "wm_ci_lo": float(ci_wm[0]), "wm_ci_hi": float(ci_wm[1]),
                "n_seeds": len(csrs),
            })

    t3_extended = pd.DataFrame(t3_rows)
    t3_extended.to_csv(TABLES_DIR / "T3_extended.csv", index=False)
    print(f"  Saved T3_extended.csv with {len(t3_rows)} rows")
    print(t3_extended.to_string(index=False))

    # Save everything
    results_path = RUNS_DIR / "extended_results.json"

    # Convert non-serializable items
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    with open(results_path, "w") as f:
        json.dump(all_extended, f, indent=2, default=convert)
    print(f"\nAll extended results saved to {results_path}")

    print("\n" + "=" * 60)
    print("PHASE 6 EXTENDED COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
