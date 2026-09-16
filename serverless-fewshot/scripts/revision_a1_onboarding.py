# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision experiments A1 + A2(onboarding side) + B2(a,b,c), batched.

A1(a): distribution of nearest-centroid assignments for the 33 onboarding
       functions at K=0 (histogram over 16 clusters, entropy, stability).
A1(b): prototype-assignment ablation ladder:
       {noproto, random, global-pooled(k=1), nearest(current), oracle(post-hoc)}
       -> CRPS at t=1..10 min + full curves + first-hour CSR per rho.
A2   : integrated-gate onboarding == 'nearest' arm by construction (every
       onboarding function is brand-new -> prototype branch); recorded as such.
B2a  : rho extended to {1, 10, 100}.
B2b  : pinball loss at each rho's critical fractile (not just full CRPS).
B2c  : zero-shot gain stratified by early-traffic burstiness (Fano, first 30m).

Semantics identical to phase63_onboarding_drift.onboarding_experiment
(same TAU-rate extraction, decisions_from_rates, simulate_function, seeds)
so numbers are directly comparable to onboarding_drift_results.json.

New a-priori constants (documented for Table 1): RANDOM_ARM_SEED=123,
ORACLE_SELECT_WINDOW=10 min. Both fixed before evaluation.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase63_onboarding_drift import (  # noqa: E402
    L, RHO_MAIN, QL, ONBOARD_WINDOW, DES_SEEDS, DEVICE,
    PROCESSED_DIR, RUNS_DIR,
    build_prototypes, assign_proto_W, rate_from_logq,
    find_onboarding_points,
)
from scripts.phase6_des import decisions_from_rates, COLD_INIT  # noqa: E402
from src.sim.des import simulate_function, rolling_csr  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402
from src.models.prototypes import PrototypeManager  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402

RHOS = [1.0, 10.0, 100.0]          # B2a: adds 100 to phase63's {1, 10}
RANDOM_ARM_SEED = 123              # fixed a priori
ORACLE_SELECT_WINDOW = 10          # minutes of post-hoc data for oracle arm
ARMS = ["noproto", "random", "global", "nearest", "oracle"]


def build_global_prototype(trainer, features, counts, train_idx):
    """k=1 'pooled' prototype: same pipeline, single cluster."""
    pm1 = PrototypeManager(n_clusters=1, device=DEVICE)
    feats_t = torch.from_numpy(features).float()
    counts_t = torch.from_numpy(counts).float()
    emb = pm1.compute_embeddings(trainer.body, feats_t, train_idx)
    labels = pm1.fit_clusters(emb)
    pm1.compute_prototype_heads(trainer.body, trainer.head, feats_t, counts_t,
                                train_idx, labels,
                                n_quantiles=N_QUANTILES, n_horizons=1)
    return pm1.proto_weights[0].cpu()   # [d, O]


def main():
    features = np.load(PROCESSED_DIR / "features.npy")
    counts = np.load(PROCESSED_DIR / "counts.npy")
    splits = np.load(PROCESSED_DIR / "splits.npz")

    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE,
    )
    trainer.load(RUNS_DIR / "best_anil_ridge_s1_s0.pt")
    train_idx, test_idx = splits["s1_train"], splits["s1_test"]

    pm = build_prototypes(trainer, features, counts, train_idx)
    W_global = build_global_prototype(trainer, features, counts, train_idx)

    pts = find_onboarding_points(counts, test_idx)
    F_n, W = len(pts), ONBOARD_WINDOW
    n_out = N_QUANTILES
    print(f"{F_n} onboarding functions, window {W} min, arms {ARMS}, rhos {RHOS}")

    # ---- embeddings and targets (identical to phase63) ----
    feats_t = torch.from_numpy(features).float()
    trainer.body.eval()
    phi_all = torch.zeros(F_n, W, 64)
    with torch.no_grad():
        for w in range(W):
            batch = []
            for (fi, t0) in pts:
                t = t0 + w
                if t < L:
                    pad = torch.zeros(L - t, features.shape[2])
                    win = torch.cat([pad, feats_t[fi, :t]], dim=0)
                else:
                    win = feats_t[fi, t - L:t]
                batch.append(win)
            x = torch.stack(batch).to(DEVICE)
            phi_all[:, w] = trainer.body(x).cpu()
    y_all = np.stack([np.log1p(counts[fi, t0:t0 + W]) for (fi, t0) in pts])
    y_t = torch.from_numpy(y_all).float()

    pw_all = pm.proto_weights.cpu()               # [16, d, O]
    K = pw_all.shape[0]

    # ---- A1(a): assignment distribution at K=0 + stability ----
    _, ids0 = assign_proto_W(pm, phi_all[:, 0])
    ids0 = ids0.cpu().numpy()
    hist = np.bincount(ids0, minlength=K)
    p = hist / hist.sum()
    entropy_bits = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    stab = np.zeros(F_n)
    for w in range(min(60, W)):
        _, idw = assign_proto_W(pm, phi_all[:, w])
        stab += (idw.cpu().numpy() == ids0)
    stab /= min(60, W)
    print(f"A1a: assignment hist={hist.tolist()} entropy={entropy_bits:.2f} bits "
          f"(uniform max={np.log2(K):.1f}) stability(first hour)={stab.mean():.2f}")

    # ---- arm-specific prototype priors ----
    rng = np.random.default_rng(RANDOM_ARM_SEED)
    rand_ids = rng.integers(0, K, size=F_n)
    # oracle: post-hoc best cluster by zero-shot CRPS over first 10 min
    oracle_ids = np.zeros(F_n, dtype=int)
    with torch.no_grad():
        for f in range(F_n):
            phi10 = phi_all[f, :ORACLE_SELECT_WINDOW]        # [10, d]
            y10 = y_all[f, :ORACLE_SELECT_WINDOW]
            best, best_c = None, 0
            for c in range(K):
                pred = (phi10 @ pw_all[c]).numpy()           # [10, Q]
                err = y10[:, None] - pred
                pb = np.maximum(QL * err, (QL - 1) * err).mean()
                if best is None or pb < best:
                    best, best_c = pb, c
            oracle_ids[f] = best_c
    oracle_match = float((oracle_ids == ids0).mean())
    print(f"A1a: oracle-vs-nearest agreement at K=0: {oracle_match:.2f}")

    def prior_for(arm, w, phi_now):
        if arm == "noproto":
            return None
        if arm == "random":
            return pw_all[rand_ids]
        if arm == "global":
            return W_global.unsqueeze(0).expand(F_n, -1, -1)
        if arm == "nearest":
            Wp, _ = assign_proto_W(pm, phi_now)              # per-tick, as phase63
            return Wp.cpu()
        if arm == "oracle":
            return pw_all[oracle_ids]
        raise ValueError(arm)

    # ---- online adaptation per arm ----
    lam = trainer.head.ridge_lambda.item()
    eye = torch.eye(64)
    preds = {a: np.zeros((F_n, W, n_out), dtype=np.float32) for a in ARMS}
    for arm in ARMS:
        for w in range(W):
            phi_now = phi_all[:, w]
            Wp = prior_for(arm, w, phi_now)
            if w == 0:
                W_head = Wp if Wp is not None else torch.zeros(F_n, 64, n_out)
            else:
                phi_s = phi_all[:, :w]
                y_s = y_t[:, :w].unsqueeze(-1).expand(F_n, w, n_out)
                PhiT = phi_s.transpose(1, 2)
                A = PhiT @ phi_s + lam * eye.unsqueeze(0)
                PhiTY = PhiT @ y_s
                rhs = PhiTY + (lam * Wp if Wp is not None else 0.0)
                W_head = torch.linalg.solve(A, rhs)
            preds[arm][:, w] = (phi_now.unsqueeze(1) @ W_head).squeeze(1).numpy()
        print(f"  arm {arm}: online loop done")

    # ---- CRPS / fractile pinball vs time ----
    def crps_curve(pred):
        err = y_all[:, :, None] - pred
        pb = np.maximum(QL * err, (QL - 1) * err)
        return 2 * pb.mean(axis=(0, 2))                      # [W]

    def fractile_pinball_curve(pred, tau):
        v = np.stack([np.interp(tau, QL, pred[f, w])
                      for f in range(F_n) for w in range(W)]
                     ).reshape(F_n, W)
        err = y_all - v
        return np.maximum(tau * err, (tau - 1) * err).mean(axis=0)  # [W]

    out = {
        "a1a": {"assignment_hist": hist.tolist(),
                "entropy_bits": entropy_bits,
                "max_entropy_bits": float(np.log2(K)),
                "stability_first_hour_mean": float(stab.mean()),
                "oracle_agreement": oracle_match,
                "oracle_ids": oracle_ids.tolist(),
                "nearest_ids_k0": ids0.tolist()},
        "constants": {"random_arm_seed": RANDOM_ARM_SEED,
                      "oracle_select_window_min": ORACLE_SELECT_WINDOW,
                      "rhos": RHOS, "des_seeds": DES_SEEDS,
                      "note_a2": "integrated-gate onboarding == nearest arm "
                                 "(all onboarding functions are brand-new)"},
        "crps_vs_time": {}, "crps_t1_10": {}, "fractile_pinball": {},
        "by_rho": {}, "burstiness": {},
    }
    for arm in ARMS:
        c = crps_curve(preds[arm])
        out["crps_vs_time"][arm] = c.tolist()
        out["crps_t1_10"][arm] = c[1:11].tolist()
        out["fractile_pinball"][arm] = {
            str(r): fractile_pinball_curve(preds[arm],
                                           newsvendor_quantile(r))[:60].tolist()
            for r in RHOS}

    # ---- B2c: burstiness stratification (Fano of first 30 min counts) ----
    seg_counts = np.stack([counts[fi, t0:t0 + W] for (fi, t0) in pts])
    c30 = seg_counts[:, :30].astype(np.float64)
    fano = c30.var(axis=1) / np.maximum(c30.mean(axis=1), 1e-9)
    terc = pd.qcut(fano, 3, labels=["smooth", "mid", "bursty"])
    def early_crps(pred, f_mask):
        err = y_all[f_mask, :10, None] - pred[f_mask, :10]
        return float(2 * np.maximum(QL * err, (QL - 1) * err).mean())
    for name in ["smooth", "mid", "bursty"]:
        m = np.asarray(terc == name)
        out["burstiness"][name] = {
            "n": int(m.sum()), "fano_median": float(np.median(fano[m])),
            "early_crps": {a: early_crps(preds[a], m) for a in ARMS}}

    # ---- DES per arm x rho (identical machinery to phase63) ----
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] for (fi, _) in pts]), nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] for (fi, _) in pts]), nan=0.5)

    rates = {a: np.zeros((F_n, W), dtype=np.float32) for a in ARMS}
    for a in ARMS:
        for w in range(W):
            rates[a][:, w] = rate_from_logq(preds[a][:, w])
    # baselines, same as phase63
    ewma = np.zeros(F_n)
    rates["B4a_ewma"] = np.zeros((F_n, W), dtype=np.float32)
    for w in range(W):
        rates["B4a_ewma"][:, w] = np.expm1(np.maximum(ewma, 0))
        ewma = 0.1 * np.log1p(seg_counts[:, w].astype(np.float64)) + 0.9 * ewma
    rates["Oracle"] = seg_counts.astype(np.float32)
    pw_b1 = np.zeros((F_n, W), dtype=np.int32)
    for f in range(F_n):
        act = (seg_counts[f] > 0).astype(np.int32)
        conv = np.convolve(act, np.ones(10, dtype=np.int32))[:W]
        pw_b1[f, 1:] = (conv[:-1] > 0).astype(np.int32)
    b1_dec = (pw_b1, np.full((F_n, W), 10.0, np.float32))

    for rho in RHOS:
        tau = newsvendor_quantile(rho)
        decisions = {m: decisions_from_rates(r, tau) for m, r in rates.items()}
        decisions["B1_fixed_keepalive"] = b1_dec
        rho_out = {}
        for m, (pw_m, ka_m) in decisions.items():
            roll_c = np.zeros(W); roll_n = np.zeros(W)
            func_cold60 = np.zeros(F_n); func_wm = np.zeros(F_n)
            for f in range(F_n):
                for seed in DES_SEEDS:
                    r = simulate_function(seg_counts[f], pw_m[f], ka_m[f],
                                          float(dm[f]), float(dstd[f]),
                                          seed=seed * 7919 + f,
                                          cold_mu=COLD_INIT["mu"],
                                          cold_sigma=COLD_INIT["sigma"],
                                          track_rolling=True)
                    roll_c += r["roll_cold"]; roll_n += r["roll_total"]
                    func_cold60[f] += r["roll_cold"][:60].sum()
                    func_wm[f] += r["idle_mem_gb_s"]
            func_cold60 /= len(DES_SEEDS); func_wm /= len(DES_SEEDS)
            rho_out[m] = {
                "overall_csr": float(roll_c.sum() / max(roll_n.sum(), 1e-9)),
                "cold60_total": float(func_cold60.sum()),
                "func_cold60": func_cold60.tolist(),
                "wm_total": float(func_wm.sum()),
                "rolling_csr": rolling_csr(roll_c, roll_n, window=15).tolist(),
            }
        out["by_rho"][str(rho)] = rho_out
        print(f"  rho={rho}: " + " ".join(
            f"{m}={v['overall_csr']*100:.2f}%" for m, v in rho_out.items()))

    path = RUNS_DIR / "revision_a1_onboarding.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    # read-back verify (sdb NUL-corruption guard)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")


if __name__ == "__main__":
    main()
