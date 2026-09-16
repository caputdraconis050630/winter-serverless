# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP8: Real head-type ablation (removes the estimated-factor placeholder).

Meta-trains ANIL-GD (H1) and Bayesian neural-linear (H3) heads with an
unbatched episodic loop (their adapt/predict APIs are unbatched), then
evaluates the K x head CRPS grid alongside the already-trained Ridge (H2).

- ANIL-GD: multi-quantile linear head adapted by 5 inner SGD steps;
  K=0 uses the meta-learned initialization directly.
- Bayesian: single-output neural-linear trained with Gaussian NLL;
  quantiles derived from the predictive normal (honest CRPS).
Updates ablation_results.json: heatmap_k_head.
"""

import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bodies import build_body
from src.models.heads import (
    ANILGDHead, BayesianLinearHead, N_QUANTILES, QUANTILES, pinball_loss,
    crps_from_quantiles,
)
from src.meta.trainer import ANILMetaTrainer

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

L = 60
K_VALUES = [0, 1, 5, 10, 20, 50]
N_STEPS = 1200
EPISODES_PER_STEP = 4
QL = QUANTILES.to(DEVICE)
Z_QL = torch.distributions.Normal(0, 1).icdf(QL.cpu()).to(DEVICE)  # [Q]


def sample_episode(features_t, counts, func_indices, rng, k_support=10, k_query=16):
    T = features_t.shape[1]
    fi = int(rng.choice(func_indices))
    times = rng.integers(L, T - 1, size=k_support + k_query)
    xs = torch.stack([features_t[fi, t - L:t] for t in times]).to(DEVICE)
    ys = torch.tensor([np.log1p(counts[fi, t]) for t in times],
                      dtype=torch.float32, device=DEVICE)
    return (xs[:k_support], ys[:k_support], xs[k_support:], ys[k_support:])


def train_anil_gd(features_t, counts, train_idx, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    body = build_body("tcn", features_t.shape[2]).to(DEVICE)
    head = ANILGDHead(64, N_QUANTILES, 1).to(DEVICE)
    opt = torch.optim.AdamW(list(body.parameters()) + list(head.parameters()),
                            lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS)

    t0 = time.time()
    for step in range(N_STEPS):
        loss_acc = 0.0
        opt.zero_grad()
        for _ in range(EPISODES_PER_STEP):
            sx, sy, qx, qy = sample_episode(features_t, counts, train_idx, rng)
            phi_s = body(sx)
            phi_q = body(qx)
            sy_exp = sy.unsqueeze(-1).expand(-1, N_QUANTILES)
            W, b = head.adapt(phi_s, sy_exp)
            pred = head.predict(phi_q, W, b)
            loss = pinball_loss(pred, qy, QL)
            (loss / EPISODES_PER_STEP).backward()
            loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_(
            list(body.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"    anil_gd step {step}: loss={loss_acc/EPISODES_PER_STEP:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return body, head


def train_bayesian(features_t, counts, train_idx, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    body = build_body("tcn", features_t.shape[2]).to(DEVICE)
    head = BayesianLinearHead(64, n_outputs=1, n_horizons=1).to(DEVICE)
    opt = torch.optim.AdamW(list(body.parameters()) + list(head.parameters()),
                            lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS)

    t0 = time.time()
    for step in range(N_STEPS):
        loss_acc = 0.0
        opt.zero_grad()
        for _ in range(EPISODES_PER_STEP):
            sx, sy, qx, qy = sample_episode(features_t, counts, train_idx, rng)
            phi_s = body(sx)
            phi_q = body(qx)
            mu, Sigma = head.adapt(phi_s, sy.unsqueeze(-1))
            mean, var = head.predict(phi_q, mu, Sigma)
            # Gaussian NLL on query
            nll = 0.5 * (torch.log(2 * np.pi * var)
                         + (qy - mean.squeeze(-1)) ** 2 / var).mean()
            (nll / EPISODES_PER_STEP).backward()
            loss_acc += nll.item()
        torch.nn.utils.clip_grad_norm_(
            list(body.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"    bayesian step {step}: nll={loss_acc/EPISODES_PER_STEP:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return body, head


@torch.no_grad()
def eval_grid(head_type, body, head, features_t, counts, test_idx):
    """CRPS at each K on the test split."""
    T = features_t.shape[1]
    res = {}
    for K in K_VALUES:
        scores = []
        for fi in test_idx:
            q_times = np.linspace(T // 2 + 1, T - 2, 32, dtype=int)
            qx = torch.stack([features_t[fi, t - L:t] for t in q_times]).to(DEVICE)
            qy = torch.tensor([np.log1p(counts[fi, t]) for t in q_times],
                              dtype=torch.float32, device=DEVICE)
            phi_q = body(qx)

            if K > 0:
                s_times = np.linspace(L, T // 2, K, dtype=int)
                sx = torch.stack([features_t[fi, t - L:t] for t in s_times]).to(DEVICE)
                sy = torch.tensor([np.log1p(counts[fi, t]) for t in s_times],
                                  dtype=torch.float32, device=DEVICE)
                phi_s = body(sx)

            if head_type == "anil_gd":
                if K == 0:
                    pred = head.predict(phi_q, head.W_init, head.b_init)
                else:
                    with torch.enable_grad():
                        sy_exp = sy.unsqueeze(-1).expand(-1, N_QUANTILES)
                        W, b = head.adapt(phi_s, sy_exp)
                    pred = head.predict(phi_q, W.detach(), b.detach())
                crps = crps_from_quantiles(pred[:, :N_QUANTILES], qy, QL)
            else:  # bayesian
                if K == 0:
                    mean = torch.zeros(len(q_times), device=DEVICE)
                    var = (head.sigma_noise ** 2
                           + (head.sigma_prior ** 2) * (phi_q ** 2).sum(-1))
                else:
                    mu, Sigma = head.adapt(phi_s, sy.unsqueeze(-1))
                    m, var = head.predict(phi_q, mu, Sigma)
                    mean = m.squeeze(-1)
                # quantiles from predictive normal
                pred = mean.unsqueeze(-1) + Z_QL.unsqueeze(0) * var.sqrt().unsqueeze(-1)
                crps = crps_from_quantiles(pred, qy, QL)
            scores.append(crps.item())
        res[K] = float(np.mean(scores))
        print(f"    {head_type} K={K:3d}: CRPS={res[K]:.4f}", flush=True)
    return res


def main():
    features = np.load(PROCESSED_DIR / "features.npy")
    counts = np.load(PROCESSED_DIR / "counts.npy")
    splits_data = np.load(PROCESSED_DIR / "splits.npz")
    train_idx = splits_data["s2_train"]
    test_idx = splits_data["s2_test"]
    features_t = torch.from_numpy(features).float()

    grid = {}

    # Ridge (H2): reuse the primary trained model
    print("  Evaluating ridge (existing model)...")
    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE)
    trainer.load(RUNS_DIR / "best_anil_ridge_s2_s0.pt")
    trainer.body.eval()
    ridge_res = {}
    with torch.no_grad():
        for K in K_VALUES:
            scores = []
            T = features.shape[1]
            for fi in test_idx:
                q_times = np.linspace(T // 2 + 1, T - 2, 32, dtype=int)
                qx = torch.stack([features_t[fi, t - L:t] for t in q_times]).to(DEVICE)
                qy = torch.tensor([np.log1p(counts[fi, t]) for t in q_times],
                                  dtype=torch.float32, device=DEVICE)
                phi_q = trainer.body(qx)
                if K == 0:
                    pred = phi_q @ torch.zeros(64, N_QUANTILES, device=DEVICE)
                else:
                    s_times = np.linspace(L, T // 2, K, dtype=int)
                    sx = torch.stack([features_t[fi, t - L:t] for t in s_times]).to(DEVICE)
                    sy = torch.tensor([np.log1p(counts[fi, t]) for t in s_times],
                                      dtype=torch.float32, device=DEVICE)
                    phi_s = trainer.body(sx)
                    sy_exp = sy.unsqueeze(-1).expand(-1, N_QUANTILES)
                    W = trainer.head.adapt(phi_s, sy_exp)
                    pred = trainer.head.predict(phi_q, W)
                scores.append(crps_from_quantiles(
                    pred[:, :N_QUANTILES], qy, QL).item())
            ridge_res[K] = float(np.mean(scores))
            print(f"    ridge K={K:3d}: CRPS={ridge_res[K]:.4f}", flush=True)
    grid["ridge"] = ridge_res

    print("  Training anil_gd head...")
    body_g, head_g = train_anil_gd(features_t, counts, train_idx)
    body_g.eval()
    grid["anil_gd"] = eval_grid("anil_gd", body_g, head_g, features_t, counts, test_idx)

    print("  Training bayesian head...")
    body_b, head_b = train_bayesian(features_t, counts, train_idx)
    body_b.eval()
    grid["bayesian"] = eval_grid("bayesian", body_b, head_b, features_t, counts, test_idx)

    # Update ablation_results.json heatmap with measured values
    path = RUNS_DIR / "ablation_results.json"
    with open(path) as f:
        abl = json.load(f)
    head_types = ["ridge", "anil_gd", "bayesian"]
    matrix = [[grid[h].get(k, float("nan")) for h in head_types]
              for k in K_VALUES]
    abl["heatmap_k_head"] = {
        "k_values": K_VALUES, "head_types": head_types,
        "crps_matrix": matrix, "source": "measured (phase8_heads.py)",
    }
    with open(path, "w") as f:
        json.dump(abl, f, indent=2, default=str)
    print(f"\nUpdated heatmap in {path}")


if __name__ == "__main__":
    main()
