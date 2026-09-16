#!/usr/bin/env python3
"""WP3a on Azure-2019: prototype-biased ridge (adapt_biased) under the exact
GO/NO-GO gate protocol, vs plain ridge adaptation and B5.

The designed component (PrototypeManager + BiasedRidgeHead.adapt_biased)
targets the diagnosed 2019 failure mode: lambda~1.08 ridge shrinks W toward 0
and underestimates high-scale targets; biased ridge shrinks toward the
cluster prototype W_proto instead.

Differences from phase63's prototype build (2021):
  - features.npy is 25GB on 2019 -> all reads go through np.memmap slices;
    prototypes are built from a random subsample of s2_train functions
    (PROTO_TRAIN_CAP, seed 42) — compute_prototype_heads capped per-cluster
    at 50 functions anyway.
  - Prototype assignment for an eval function uses ONLY its K support
    windows (mean phi_s), never query-period data.

Env: SF_DATA_DIR, SF_RUNS_DIR, SF_TAG as in router_analysis.py.
Output: results/runs/adapt_biased_<tag>.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.cluster import KMeans  # noqa: E402

from scripts.phase4_train import (  # noqa: E402
    PROCESSED_DIR, RUNS_DIR, DEVICE, N_QUANTILES, QUANTILES,
    cap_eval_indices, crps_from_quantiles,
)
from scripts.diag_gate_2019 import HORIZONS  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402
from src.models.heads import BiasedRidgeHead  # noqa: E402

CKPT_RUNS = Path(os.environ.get("SF_RUNS_DIR", str(RUNS_DIR)))
TAG = os.environ.get("SF_TAG", os.environ.get("SF_DATA_DIR", "processed"))
K_SUPPORT = 10
L = 60
N_CLUSTERS = 16
# Support functions pooled into one prototype head. Was an inline `[:50]`;
# named here so an ablation can lift it without editing the builder. The
# default is unchanged, so every archived result still reproduces exactly.
# It matters for interpreting the k=1 arm: with this cap a "single global
# prototype" is fitted on 50 functions, not on the whole training split.
PROTO_FUNC_CAP = 50
PROTO_TRAIN_CAP = 2000
SEEDS = (0, 1, 2)


def load_trainer(in_features, seed):
    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=in_features, embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=len(HORIZONS),
        use_amp=True, device=DEVICE,
    )
    trainer.load(CKPT_RUNS / f"best_anil_ridge_s2_s{seed}.pt")
    biased = BiasedRidgeHead(64, n_outputs=N_QUANTILES,
                             n_horizons=len(HORIZONS)).to(DEVICE)
    biased.load_state_dict(trainer.head.state_dict())
    return trainer, biased


@torch.no_grad()
def embed_functions(body, features, func_indices, n_windows=50, batch=256):
    """Mean-pooled embeddings from mmap features, windows over full trace."""
    body.eval()
    T = features.shape[1]
    starts = np.linspace(L, T - 1, n_windows, dtype=int)
    out = []
    for i in range(0, len(func_indices), batch):
        idx = np.asarray(func_indices[i:i + batch])
        accum = None
        for t in starts:
            x = torch.from_numpy(
                np.asarray(features[idx, t - L:t, :], dtype=np.float32)).to(DEVICE)
            phi = body(x)
            accum = phi if accum is None else accum + phi
        out.append((accum / len(starts)).cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def build_prototypes(trainer, biased_head, features, counts, train_idx):
    rng = np.random.default_rng(42)
    if len(train_idx) > PROTO_TRAIN_CAP:
        train_idx = np.sort(rng.choice(train_idx, PROTO_TRAIN_CAP, replace=False))
    print(f"  building {N_CLUSTERS} prototypes from {len(train_idx)} train funcs")
    emb = embed_functions(trainer.body, features, train_idx)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(emb.numpy())
    centroids = torch.from_numpy(km.cluster_centers_).float()

    T = features.shape[1]
    n_outputs = N_QUANTILES * len(HORIZONS)
    proto_w = torch.zeros(N_CLUSTERS, 64, n_outputs)
    win_starts = np.linspace(L, T - 2, 20, dtype=int)
    for c in range(N_CLUSTERS):
        cfuncs = train_idx[labels == c][:PROTO_FUNC_CAP]
        if len(cfuncs) == 0:
            continue
        phis, ys = [], []
        for fi in cfuncs:
            for t in win_starts:
                x = torch.from_numpy(np.asarray(
                    features[fi:fi + 1, t - L:t, :], dtype=np.float32)).to(DEVICE)
                phis.append(trainer.body(x).cpu())
                ys.append(float(np.log1p(counts[fi, min(t, T - 1)])))
        phi_pool = torch.cat(phis, dim=0).to(DEVICE)
        y_pool = torch.tensor(ys).float().unsqueeze(1).to(DEVICE)
        W = trainer.head.adapt(phi_pool, y_pool.expand(-1, n_outputs))
        proto_w[c] = W.detach().cpu()
    return centroids, proto_w


@torch.no_grad()
def evaluate_biased(trainer, biased_head, centroids, proto_w,
                    features, counts, func_indices, k_support=K_SUPPORT,
                    also_zero_shot=True):
    """Gate protocol (identical placement to diag/gate) with adapt_biased.
    Prototype assigned from mean support embedding only."""
    quantiles = QUANTILES.to(DEVICE)
    n_outputs = N_QUANTILES * len(HORIZONS)
    trainer.body.eval()
    T = features.shape[1]
    cen = centroids.to(DEVICE)
    pw = proto_w.to(DEVICE)

    def win(fi, t):
        return torch.from_numpy(np.asarray(features[fi, t - L:t], dtype=np.float32))

    crps_biased, crps_zero, proto_ids = [], [], []
    for fi in func_indices:
        support_times = np.linspace(L, T // 2, k_support, dtype=int)
        query_times = np.linspace(T // 2 + 1, T - max(HORIZONS) - 1, 32, dtype=int)

        sx = torch.stack([win(fi, t) for t in support_times]).to(DEVICE)
        sy = torch.stack([
            torch.tensor([np.log1p(counts[fi, min(t + h - 1, T - 1)])
                          for h in HORIZONS])
            for t in support_times]).float().to(DEVICE)
        qx = torch.stack([win(fi, t) for t in query_times]).to(DEVICE)
        qy = torch.stack([
            torch.tensor([np.log1p(counts[fi, min(t + h - 1, T - 1)])
                          for h in HORIZONS])
            for t in query_times]).float().to(DEVICE)

        phi_s = trainer.body(sx)
        phi_q = trainer.body(qx)
        sims = F.cosine_similarity(phi_s.mean(0, keepdim=True), cen, dim=1)
        cid = int(sims.argmax().item())
        W_proto = pw[cid]
        proto_ids.append(cid)

        sy_exp = sy.unsqueeze(-1).expand(k_support, len(HORIZONS), N_QUANTILES)
        sy_exp = sy_exp.reshape(k_support, n_outputs)
        W = biased_head.adapt_biased(phi_s, sy_exp, W_proto)
        pred = biased_head.predict(phi_q, W)
        crps_biased.append(crps_from_quantiles(
            pred[:, :N_QUANTILES], qy[:, 0], quantiles).item())
        if also_zero_shot:
            pred0 = biased_head.predict(phi_q, W_proto)
            crps_zero.append(crps_from_quantiles(
                pred0[:, :N_QUANTILES], qy[:, 0], quantiles).item())
    return (np.array(crps_biased), np.array(crps_zero) if also_zero_shot else None,
            np.array(proto_ids))


def summarize(s):
    return {"mean": float(s.mean()), "median": float(np.median(s)),
            "p90": float(np.percentile(s, 90))}


def main():
    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    splits = np.load(PROCESSED_DIR / "splits.npz")
    test_idx = cap_eval_indices(splits["s2_test"])
    T = counts.shape[1]
    print(f"[{TAG}] adapt_biased eval, test n={len(test_idx)}")

    # Paired reference scores from router analysis (same protocol/functions)
    ref = np.load(RUNS_DIR / f"router_perfunc_{TAG}.npz")
    assert np.array_equal(ref["test_idx"], test_idx), "eval set mismatch"
    b5_test = ref["b5_test"]

    rates = np.asarray(counts[test_idx, :T // 2]).mean(axis=1)
    q = pd.qcut(np.log10(rates + 1e-6), 4, duplicates="drop")
    q = q.rename_categories([f"Q{i+1}" for i in range(len(q.categories))])

    out = {"tag": TAG, "n_clusters": N_CLUSTERS,
           "proto_train_cap": PROTO_TRAIN_CAP, "per_seed": {}}
    for seed in SEEDS:
        ckpt = CKPT_RUNS / f"best_anil_ridge_s2_s{seed}.pt"
        if not ckpt.exists():
            print(f"  seed {seed}: missing checkpoint, skipping")
            continue
        trainer, biased_head = load_trainer(features.shape[2], seed)
        centroids, proto_w = build_prototypes(
            trainer, biased_head, features, counts, splits["s2_train"])
        biased, zero, pids = evaluate_biased(
            trainer, biased_head, centroids, proto_w, features, counts, test_idx)
        plain = ref[f"ours_test_s{seed}"]

        strat = {}
        for name in q.categories:
            m = np.asarray(q == name)
            strat[str(name)] = {
                "n": int(m.sum()),
                "plain_mean": float(plain[m].mean()),
                "biased_mean": float(biased[m].mean()),
                "b5_mean": float(b5_test[m].mean()),
                "biased_win_rate_vs_b5": float((biased[m] < b5_test[m]).mean()),
            }
        res = {
            "plain_ridge": summarize(plain),
            "biased_ridge": summarize(biased),
            "zero_shot_proto": summarize(zero),
            "b5": summarize(b5_test),
            "biased_vs_plain_mean_diff": float(biased.mean() - plain.mean()),
            "biased_win_rate_vs_b5": float((biased < b5_test).mean()),
            "by_rate_quartile": strat,
            "proto_usage": np.bincount(pids, minlength=N_CLUSTERS).tolist(),
        }
        out["per_seed"][str(seed)] = res
        np.savez(RUNS_DIR / f"adapt_biased_perfunc_{TAG}_s{seed}.npz",
                 biased=biased, zero_shot=zero, proto_ids=pids)
        print(f"  seed {seed}: plain={plain.mean():.4f} biased={biased.mean():.4f} "
              f"zero-shot={zero.mean():.4f} b5={b5_test.mean():.4f} "
              f"win_vs_b5={(biased < b5_test).mean()*100:.0f}%")

    out_path = RUNS_DIR / f"adapt_biased_{TAG}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
