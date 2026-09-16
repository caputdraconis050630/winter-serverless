#!/usr/bin/env python3
"""Phase 4: Meta-Training & Baseline Training.

Trains ANIL meta-learner and baselines, then runs GO/NO-GO gate.
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.meta.trainer import ANILMetaTrainer, MetaDataset, collate_episodes
from src.models.heads import N_QUANTILES, QUANTILES, crps_from_quantiles, pinball_loss
from src.models.bodies import build_body
from src.models.baselines import (
    FourierPredictor, EWMAPredictor, SeasonalNaive, OraclePredictor
)

# SF_DATA_DIR=processed_2019 switches to the Azure-2019 dataset
PROCESSED_DIR = PROJECT_ROOT / "data" / os.environ.get("SF_DATA_DIR", "processed")
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Cap per-function evaluation loops at large scale (stratified subsample,
# same subsample shared by all methods via fixed seed)
EVAL_CAP = int(os.environ.get("SF_EVAL_CAP", "500"))


def cap_eval_indices(func_indices, cap=None, seed=123):
    cap = cap or EVAL_CAP
    func_indices = np.asarray(func_indices)
    if len(func_indices) <= cap:
        return func_indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(func_indices, size=cap, replace=False))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_meta_learner(features, counts, splits, seed=0,
                       body_type="tcn", head_type="ridge",
                       n_steps=5000, batch_size=32, k_support=10, k_query=32,
                       horizons=(1,), lr=3e-4, eval_every=500,
                       tag="default"):
    """Train ANIL meta-learner."""
    print(f"\n{'='*60}")
    print(f"Meta-training: body={body_type}, head={head_type}, K={k_support}, seed={seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_horizons = len(horizons)
    in_features = features.shape[2]

    # Build datasets
    train_ds = MetaDataset(features, counts, splits["train"],
                           k_support=k_support, k_query=k_query, horizons=horizons)
    val_ds = MetaDataset(features, counts, splits["val"],
                         k_support=k_support, k_query=k_query, horizons=horizons)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_episodes, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_episodes, num_workers=2,
                            pin_memory=True, drop_last=True)

    # Build trainer
    trainer = ANILMetaTrainer(
        body_type=body_type, head_type=head_type,
        in_features=in_features, embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=n_horizons,
        lr=lr, weight_decay=0.01, grad_clip=1.0,
        use_amp=True, device=DEVICE,
    )

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=n_steps, eta_min=1e-6
    )

    # Training loop
    best_val_crps = float("inf")
    patience = 0
    max_patience = 500
    train_log = []

    step = 0
    t0 = time.time()

    for epoch in range(1000):  # outer epochs
        for sx, sy, qx, qy in train_loader:
            loss, metrics = trainer.meta_train_step(sx, sy, qx, qy)
            scheduler.step()
            step += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Step {step:5d} | loss={loss:.4f} | crps={metrics['crps']:.4f} | "
                      f"mae={metrics['mae']:.4f} | λ={metrics['lambda']:.4f} | "
                      f"lr={lr_now:.2e} | {elapsed:.0f}s")

            if step % eval_every == 0:
                val_metrics = trainer.evaluate(val_loader)
                print(f"  >> Val: loss={val_metrics['loss']:.4f} crps={val_metrics['crps']:.4f}")
                train_log.append({
                    "step": step, "train_loss": loss,
                    "val_loss": val_metrics["loss"],
                    "val_crps": val_metrics["crps"],
                    **metrics
                })

                if val_metrics["crps"] < best_val_crps:
                    best_val_crps = val_metrics["crps"]
                    patience = 0
                    save_path = RUNS_DIR / f"best_{tag}_s{seed}.pt"
                    trainer.save(save_path)
                    print(f"  >> New best! Saved to {save_path}")
                else:
                    patience += 1

                if patience >= max_patience // eval_every:
                    print(f"  >> Early stopping at step {step}")
                    break

            if step >= n_steps:
                break
        if step >= n_steps or patience >= max_patience // eval_every:
            break

    elapsed = time.time() - t0
    print(f"\nTraining complete: {step} steps in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Best val CRPS: {best_val_crps:.4f}")

    # Save training log
    log_path = RUNS_DIR / f"log_{tag}_s{seed}.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f)

    return trainer, best_val_crps


def train_global_model(features, counts, splits, seed=0, n_steps=5000,
                       body_type="tcn", horizons=(1,)):
    """B5: Global model with no per-function adaptation.

    Uses a shared head across all functions (K=0 adaptation).
    """
    print(f"\n{'='*60}")
    print(f"Training B5: Global model (no adaptation), seed={seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    in_features = features.shape[2]
    n_horizons = len(horizons)
    n_outputs = N_QUANTILES * n_horizons

    body = build_body(body_type, in_features).to(DEVICE)
    # Global head: a single linear layer
    head = torch.nn.Linear(64, n_outputs).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(body.parameters()) + list(head.parameters()),
        lr=3e-4, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # Build flat dataset (no episodic structure)
    train_indices = splits["train"]
    T = features.shape[1]
    L = 60
    quantiles = QUANTILES.to(DEVICE)

    best_val_loss = float("inf")
    step = 0
    t0 = time.time()

    for epoch in range(100):
        # Shuffle and iterate
        perm = np.random.permutation(len(train_indices))
        for batch_start in range(0, len(perm) - 32, 32):
            batch_funcs = train_indices[perm[batch_start:batch_start + 32]]

            # Sample random times for each function
            batch_x, batch_y = [], []
            for fi in batch_funcs:
                t = np.random.randint(L, T - max(horizons))
                x = torch.from_numpy(features[fi, t - L:t]).float()
                y = torch.tensor([np.log1p(counts[fi, min(t + h - 1, T - 1)])
                                  for h in horizons]).float()
                batch_x.append(x)
                batch_y.append(y)

            batch_x = torch.stack(batch_x).to(DEVICE)  # [B, L, F]
            batch_y = torch.stack(batch_y).to(DEVICE)  # [B, H]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                phi = body(batch_x)    # [B, d]
                pred = head(phi)       # [B, n_outputs]
                loss = pinball_loss(pred, batch_y, quantiles, n_horizons=n_horizons)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(body.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 500 == 0:
                print(f"  B5 Step {step}: loss={loss.item():.4f}")

            if step >= n_steps:
                break
        if step >= n_steps:
            break

    elapsed = time.time() - t0
    print(f"B5 training: {step} steps in {elapsed:.0f}s")

    # Save
    save_path = RUNS_DIR / f"b5_global_s{seed}.pt"
    torch.save({"body": body.state_dict(), "head": head.state_dict()}, save_path)

    return body, head


def evaluate_on_split(trainer, features, counts, func_indices, k_support=10,
                      horizons=(1,), n_eval_episodes=200, tag=""):
    """Evaluate a meta-learner on test functions."""
    func_indices = cap_eval_indices(func_indices)
    print(f"\nEvaluating on {len(func_indices)} test functions (K={k_support})...")

    quantiles = QUANTILES.to(DEVICE)
    n_horizons = len(horizons)
    n_outputs = N_QUANTILES * n_horizons

    trainer.body.eval()
    L = 60
    T = features.shape[1]

    def win(fi, t):
        return torch.from_numpy(np.asarray(features[fi, t - L:t],
                                           dtype=np.float32))

    crps_scores = []
    mae_scores = []

    with torch.no_grad():
        for fi in func_indices:
            # Build support set
            support_times = np.linspace(L, T // 2, k_support, dtype=int)
            query_times = np.linspace(T // 2 + 1, T - max(horizons) - 1, 32, dtype=int)

            sx = torch.stack([win(fi, t) for t in support_times]).to(DEVICE)
            sy = torch.stack([
                torch.tensor([np.log1p(counts[fi, min(t + h - 1, T - 1)])
                              for h in horizons])
                for t in support_times
            ]).float().to(DEVICE)

            qx = torch.stack([win(fi, t) for t in query_times]).to(DEVICE)
            qy = torch.stack([
                torch.tensor([np.log1p(counts[fi, min(t + h - 1, T - 1)])
                              for h in horizons])
                for t in query_times
            ]).float().to(DEVICE)

            phi_s = trainer.body(sx)  # [K, d]
            phi_q = trainer.body(qx)  # [Q, d]

            sy_exp = sy.unsqueeze(-1).expand(k_support, n_horizons, N_QUANTILES)
            sy_exp = sy_exp.reshape(k_support, n_outputs)

            W = trainer.head.adapt(phi_s, sy_exp)
            pred = trainer.head.predict(phi_q, W)  # [Q, n_outputs]

            crps = crps_from_quantiles(
                pred[:, :N_QUANTILES], qy[:, 0], quantiles
            )
            median_pred = pred[:, N_QUANTILES // 2]
            mae = (median_pred - qy[:, 0]).abs().mean()

            crps_scores.append(crps.item())
            mae_scores.append(mae.item())

    trainer.body.train()

    mean_crps = np.mean(crps_scores)
    mean_mae = np.mean(mae_scores)
    print(f"  {tag} CRPS: {mean_crps:.4f} ± {np.std(crps_scores):.4f}")
    print(f"  {tag} MAE:  {mean_mae:.4f} ± {np.std(mae_scores):.4f}")
    return {"crps": mean_crps, "mae": mean_mae, "crps_list": crps_scores}


def evaluate_global_on_split(body, head, features, counts, func_indices,
                              horizons=(1,)):
    """Evaluate B5 global model."""
    func_indices = cap_eval_indices(func_indices)
    print(f"\nEvaluating B5 global model on {len(func_indices)} functions...")

    quantiles = QUANTILES.to(DEVICE)
    n_horizons = len(horizons)

    body.eval()
    head.eval()
    L = 60
    T = features.shape[1]

    crps_scores = []

    with torch.no_grad():
        for fi in func_indices:
            query_times = np.linspace(T // 2 + 1, T - max(horizons) - 1, 32, dtype=int)
            qx = torch.stack([
                torch.from_numpy(np.asarray(features[fi, t - L:t],
                                            dtype=np.float32))
                for t in query_times]).to(DEVICE)
            qy = torch.tensor([np.log1p(counts[fi, min(t, T - 1)])
                                for t in query_times]).float().to(DEVICE)

            phi = body(qx)
            pred = head(phi)  # [Q, n_outputs]

            crps = crps_from_quantiles(pred[:, :N_QUANTILES], qy, quantiles)
            crps_scores.append(crps.item())

    mean_crps = np.mean(crps_scores)
    print(f"  B5 CRPS: {mean_crps:.4f} ± {np.std(crps_scores):.4f}")
    return {"crps": mean_crps, "crps_list": crps_scores}


def fit_statistical_baselines(counts, func_indices, horizons=(1,)):
    """Fit and evaluate statistical baselines (B3, B4a, B4b)."""
    func_indices = cap_eval_indices(func_indices)
    print("\nFitting statistical baselines...")
    T = counts.shape[1]
    L = 60
    quantile_levels = np.linspace(0.05, 0.95, N_QUANTILES)

    results = {}

    # B3: Fourier
    fourier = FourierPredictor(n_harmonics=10)
    fourier_crps = []
    for fi in func_indices:
        # Fit on first half
        fourier.fit(fi, counts[fi, :T // 2])
        # Evaluate on second half
        query_times = np.linspace(T // 2 + 1, T - 2, 32, dtype=int)
        for t in query_times:
            pred_q = fourier.predict_quantiles(fi, t, quantile_levels, n_steps=1)
            actual = np.log1p(counts[fi, t])
            pred_q_log = np.log1p(pred_q[0])
            errors = actual - pred_q_log
            pb = np.maximum(quantile_levels * errors, (quantile_levels - 1) * errors)
            fourier_crps.append(2 * pb.mean())
    results["B3_fourier"] = {"crps": np.mean(fourier_crps)}
    print(f"  B3 Fourier CRPS: {np.mean(fourier_crps):.4f}")

    # B4a: EWMA
    ewma = EWMAPredictor(alpha=0.1)
    ewma_crps = []
    for fi in func_indices:
        for t in range(L, T // 2):
            ewma.update(fi, np.log1p(counts[fi, t]))
        query_times = np.linspace(T // 2 + 1, T - 2, 32, dtype=int)
        for t in query_times:
            ewma.update(fi, np.log1p(counts[fi, t]))
            pred_q = ewma.predict_quantiles(fi, quantile_levels)
            actual = np.log1p(counts[fi, min(t + 1, T - 1)])
            errors = actual - np.log1p(pred_q)
            pb = np.maximum(quantile_levels * errors, (quantile_levels - 1) * errors)
            ewma_crps.append(2 * pb.mean())
    results["B4a_ewma"] = {"crps": np.mean(ewma_crps)}
    print(f"  B4a EWMA CRPS: {np.mean(ewma_crps):.4f}")

    # B4b: Seasonal naive
    snaive = SeasonalNaive(season=1440)
    snaive_crps = []
    for fi in func_indices:
        for t in range(T):
            snaive.update(fi, t, np.log1p(counts[fi, t]))
        query_times = np.linspace(max(1441, T // 2 + 1), T - 2, 32, dtype=int)
        for t in query_times:
            pred_val = snaive.predict(fi, t)
            pred_q = snaive.predict_quantiles(fi, t, quantile_levels)
            actual = np.log1p(counts[fi, t])
            errors = actual - np.log1p(pred_q)
            pb = np.maximum(quantile_levels * errors, (quantile_levels - 1) * errors)
            snaive_crps.append(2 * pb.mean())
    results["B4b_seasonal_naive"] = {"crps": np.mean(snaive_crps)}
    print(f"  B4b Seasonal Naive CRPS: {np.mean(snaive_crps):.4f}")

    # Oracle
    oracle = OraclePredictor(counts)
    print(f"  Oracle: CRPS = 0.0000 (by definition)")
    results["Oracle"] = {"crps": 0.0}

    return results


def go_nogo_gate(ours_crps, global_crps):
    """Phase 4.4 GO/NO-GO gate.

    GO if relative CRPS improvement >= 10%.
    PIVOT if < 5%.
    """
    if global_crps == 0:
        improvement = 1.0
    else:
        improvement = (global_crps - ours_crps) / global_crps

    print(f"\n{'='*60}")
    print(f"GO/NO-GO GATE (Phase 4.4)")
    print(f"{'='*60}")
    print(f"  Ours (ANIL, K=10) CRPS:  {ours_crps:.4f}")
    print(f"  B5 (Global, no adapt) CRPS: {global_crps:.4f}")
    print(f"  Relative improvement: {improvement*100:.1f}%")

    if improvement >= 0.10:
        decision = "GO"
        print(f"  Decision: ✓ GO (≥10% improvement)")
    elif improvement >= 0.05:
        decision = "MARGINAL"
        print(f"  Decision: ~ MARGINAL (5-10% improvement)")
    else:
        decision = "PIVOT"
        print(f"  Decision: ✗ PIVOT (<5% improvement)")

    print(f"{'='*60}")
    return decision, improvement


def main():
    print("=" * 60)
    print("PHASE 4: Meta-Training & Baselines")
    print("=" * 60)

    # Load data
    # mmap: 2019 features.npy is ~25GB (> RAM); episode sampling only slices per-function windows
    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    splits_data = np.load(PROCESSED_DIR / "splits.npz")

    # Use S2 (cluster hold-out) for GO/NO-GO gate as specified
    s2_splits = {
        "train": splits_data["s2_train"],
        "val": splits_data["s2_val"],
        "test": splits_data["s2_test"],
    }

    horizons = (1,)  # Start with 1-min horizon
    seed = 0

    # 1. Train our meta-learner (ANIL + ridge head)
    trainer, best_crps = train_meta_learner(
        features, counts, s2_splits, seed=seed,
        body_type="tcn", head_type="ridge",
        n_steps=5000, batch_size=32, k_support=10, k_query=32,
        horizons=horizons, tag="anil_ridge_s2",
    )

    # 2. Evaluate on S2 test (cluster hold-out)
    ours_results = evaluate_on_split(
        trainer, features, counts, s2_splits["test"],
        k_support=10, horizons=horizons, tag="Ours(S2)"
    )

    # 3. Train global model (B5)
    b5_body, b5_head = train_global_model(
        features, counts, s2_splits, seed=seed,
        n_steps=5000, body_type="tcn", horizons=horizons,
    )

    # 4. Evaluate B5
    b5_results = evaluate_global_on_split(
        b5_body, b5_head, features, counts, s2_splits["test"], horizons=horizons
    )

    # 5. Statistical baselines
    stat_results = fit_statistical_baselines(counts, s2_splits["test"], horizons=horizons)

    # 6. GO/NO-GO gate
    decision, improvement = go_nogo_gate(ours_results["crps"], b5_results["crps"])

    # Save gate results
    gate_results = {
        "decision": decision,
        "improvement_pct": round(improvement * 100, 2),
        "ours_crps": ours_results["crps"],
        "global_crps": b5_results["crps"],
        "baselines": {k: {"crps": v["crps"]} for k, v in stat_results.items()},
    }

    gate_path = RUNS_DIR / "go_nogo_gate.json"
    with open(gate_path, "w") as f:
        json.dump(gate_results, f, indent=2)
    print(f"\nGate results saved to {gate_path}")

    return decision


if __name__ == "__main__":
    decision = main()
    sys.exit(0 if decision == "GO" else 1)
