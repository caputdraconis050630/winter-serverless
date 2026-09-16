# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision E6: (A) k-means k sensitivity; (B) S2 OOD embedding-distance
diagnostic.

A) k in {1, 4, 16, 64}: rebuild prototypes from 2021 s2_train with the EXACT
   crosstrace builder (eval_adapt_biased.build_prototypes; its module-global
   N_CLUSTERS is set per k, KMeans seed 42), zero-shot evaluation on the
   2019 natural cohort's first hour (clean cross-trace config, 2021
   checkpoint). Metrics: log-space CRPS over ticks 1-10 and 0-59 (the A1
   ladder metric), fractile pinball at tau(rho=10). k=16 must reproduce the
   crosstrace prototype arm (sanity anchor); k=1 is the global-pooled prior
   (ladder self-containment: expect ~ the A1 'global' arm).

B) OOD diagnostic on the 2021 steady-state S2 losing cell: per-function
   Delta cold = (EWMA cold - A5 cold) from sim_results_des_revision.json
   (10-seed means, per rho), correlated (Spearman) against cosine distance
   of the function's mean-pooled embedding to the nearest prototype
   centroid — computed with the SAME checkpoint (best_anil_ridge_s1_s0) and
   s1_train prototypes used by that campaign (revision_a2_des mirror).
   Positive correlation => the S2 cell is measured prior-misspecification
   cost; either way, reported plainly.

Output: results/runs/revision_e6_kmeans_ood.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("SF_DATA_DIR", "processed_2019")
os.environ["SF_RUNS_DIR"] = str(PROJECT_ROOT / "results_azure2021" / "runs")

import scripts.eval_adapt_biased as eab  # noqa: E402
from scripts.revision_a4_crosstrace import find_natural_cohort  # noqa: E402
from scripts.phase63_onboarding_drift import DEVICE  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "results" / "runs"
PROC_2019 = PROJECT_ROOT / "data" / "processed_2019"
PROC_2021 = PROJECT_ROOT / "data" / "processed"
L = 60
W_EVAL = 60
KS = [1, 4, 16, 64]
QL = np.linspace(0.05, 0.95, N_QUANTILES)
TAU10 = newsvendor_quantile(10.0)


def main():
    import torch.nn.functional as Fun
    features = np.load(PROC_2019 / "features.npy", mmap_mode="r")
    counts = np.load(PROC_2019 / "counts.npy", mmap_mode="r")
    nF = features.shape[2]
    feat21 = np.load(PROC_2021 / "features.npy", mmap_mode="r")
    cnt21 = np.load(PROC_2021 / "counts.npy", mmap_mode="r")
    spl21 = np.load(PROC_2021 / "splits.npz")

    pts = find_natural_cohort(counts)
    F_n = len(pts)
    trainer, biased_head = eab.load_trainer(nF, seed=0)  # 2021 ckpt
    trainer.body.eval()

    # cohort embeddings, first hour
    phi_all = torch.zeros(F_n, W_EVAL, 64)
    with torch.no_grad():
        for w in range(W_EVAL):
            batch = []
            for (fi, t0) in pts:
                t = t0 + w
                if t < L:
                    win = np.zeros((L, nF), dtype=np.float32)
                    win[L - t:] = features[fi, :t]
                else:
                    win = np.asarray(features[fi, t - L:t], dtype=np.float32)
                batch.append(torch.from_numpy(win))
            phi_all[:, w] = trainer.body(torch.stack(batch).to(DEVICE)).cpu()
    y = np.stack([np.log1p(np.asarray(counts[fi, t0:t0 + W_EVAL], dtype=np.float64))
                  for (fi, t0) in pts])

    out = {"part_a": {}, "part_b": {}}
    for k in KS:
        eab.N_CLUSTERS = k
        centroids, proto_w = eab.build_prototypes(
            trainer, biased_head, feat21, cnt21, spl21["s2_train"])
        cen = centroids
        preds = np.zeros((F_n, W_EVAL, N_QUANTILES), dtype=np.float32)
        with torch.no_grad():
            for w in range(W_EVAL):
                phi_now = phi_all[:, w]
                sims = Fun.cosine_similarity(phi_now.unsqueeze(1),
                                             cen.unsqueeze(0), dim=2)
                Wp = proto_w[sims.argmax(dim=1)]
                preds[:, w] = (phi_now.unsqueeze(1) @ Wp).squeeze(1).numpy()
        err = y[:, :, None] - preds
        pb = np.maximum(QL * err, (QL - 1) * err)
        crps_t = 2 * pb.mean(axis=(0, 2))                     # [W]
        v = np.stack([np.interp(TAU10, QL, preds[f, w])
                      for f in range(F_n) for w in range(W_EVAL)]).reshape(F_n, W_EVAL)
        e2 = y - v
        fp = np.maximum(TAU10 * e2, (TAU10 - 1) * e2).mean()
        out["part_a"][str(k)] = {
            "crps_t1_10": float(crps_t[1:11].mean()),
            "crps_first_hour": float(crps_t.mean()),
            "fractile_pinball_tau10": float(fp),
        }
        print(f"k={k:3d}: CRPS[1:10]={out['part_a'][str(k)]['crps_t1_10']:.4f} "
              f"CRPS[0:60]={out['part_a'][str(k)]['crps_first_hour']:.4f} "
              f"pinball(tau10)={fp:.4f}", flush=True)

    # ---- Part B ----
    from scripts.phase63_onboarding_drift import build_prototypes as p63_build
    from src.meta.trainer import ANILMetaTrainer
    f21 = np.asarray(feat21, dtype=np.float32)
    c21 = np.asarray(cnt21)
    spl = spl21
    tr2 = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                          in_features=f21.shape[2], embedding_dim=64,
                          n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE)
    tr2.load(RUNS_DIR / "best_anil_ridge_s1_s0.pt")
    pm = p63_build(tr2, f21, c21, spl["s1_train"])
    s2_test = spl["s2_test"]
    emb = pm.compute_embeddings(tr2.body, torch.from_numpy(f21).float(), s2_test)
    cen = pm.centroids.to(emb.device)
    sims = Fun.cosine_similarity(emb.unsqueeze(1), cen.unsqueeze(0), dim=2)
    dist = (1.0 - sims.max(dim=1).values).cpu().numpy()      # [n_s2]

    camp = json.load(open(RUNS_DIR / "sim_results_des_revision.json"))["results"]
    from scipy import stats as sps
    out["part_b"]["n_s2"] = int(len(s2_test))
    for rho in [0.1, 1.0, 10.0, 100.0]:
        def fc(method):
            rows = [r for r in camp if r["split"] == "S2" and r["method"] == method
                    and abs(r["cost_ratio"] - rho) < 1e-9]
            return (np.mean([r["func_cold"] for r in rows], axis=0),
                    np.mean([r["func_total"] for r in rows], axis=0))
        try:
            cold_a5, tot = fc("A5_full_system")
            cold_ew, _ = fc("B4a_ewma")
        except Exception:
            continue
        delta = (cold_ew - cold_a5) / np.maximum(tot, 1)     # >0: learned wins
        rho_s, p_s = sps.spearmanr(dist, delta)
        out["part_b"][str(rho)] = {"spearman_r": float(rho_s),
                                   "p": float(p_s),
                                   "median_dist": float(np.median(dist))}
        print(f"OOD rho={rho}: spearman(dist, ewma-a5 percold)={rho_s:+.3f} "
              f"(p={p_s:.3f})", flush=True)

    path = RUNS_DIR / "revision_e6_kmeans_ood.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")


if __name__ == "__main__":
    main()
