# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R11: faithful-head steady campaign + channel-0 zeroing
(pre-registered in revision_r11_PREREG.md).

Torch env: PYTHONPATH=/data/260715/site-packages:. python3.13 scripts/revision_r11_faithful.py
Output: results/runs/revision_r11_faithful.json
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import (  # noqa: E402
    rates_ewma, rates_a5, run_config, decisions_from_rates, COLD_INIT,
    REDUCED_RHOS,
)
from scripts.phase63_onboarding_drift import build_prototypes  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402

torch.set_num_threads(1)
P21 = PROJECT_ROOT / "data" / "processed"
RUNS = PROJECT_ROOT / "results" / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L = 60
SEEDS = list(range(10))
MED = N_QUANTILES // 2
OUT = RUNS / "revision_r11_faithful.json"


def rates_a5_faithful(counts, features, trainer, pm):
    """Section-3-faithful serving: phi-only prior-biased ridge, meta lambda,
    no EWMA blend after the first refit (PREREG item 1)."""
    N, T = counts.shape
    d = 64
    BUF, adapt_interval = 180, 60
    lam = float(trainer.head.ridge_lambda.item())
    rates = np.zeros((N, T), dtype=np.float32)
    features_t = torch.from_numpy(np.ascontiguousarray(features)).float()
    Xbuf = torch.zeros(N, BUF, d, dtype=torch.float64)
    ybuf = torch.zeros(N, BUF, dtype=torch.float64)
    Wcur = None
    eye = torch.eye(d, dtype=torch.float64, device=DEVICE)
    cen = pm.centroids.to(DEVICE)                       # [K, d]
    pw_med = pm.proto_weights[:, :, MED].double().to(DEVICE)  # [K, d]

    ewma = np.zeros(N)
    alpha = 0.15
    for t in range(min(L, T)):
        rates[:, t] = np.expm1(np.maximum(ewma, 0))
        prev = counts[:, max(t - 1, 0)].astype(np.float64)
        ewma = alpha * np.log1p(prev) + (1 - alpha) * ewma

    trainer.body.eval()
    with torch.no_grad():
        for t in range(L, T):
            phis = []
            for s in range(0, N, 4096):
                bx = features_t[s:s + 4096, t - L:t, :].to(DEVICE)
                phis.append(trainer.body(bx).cpu())
            phi = torch.cat(phis, 0)                    # [N, d] float32
            prev = counts[:, max(t - 1, 0)].astype(np.float64)
            ewma = alpha * np.log1p(prev) + (1 - alpha) * ewma
            slot = (t - L) % BUF
            Xbuf[:, slot] = phi.double()
            ybuf[:, slot] = torch.from_numpy(np.log1p(prev))
            n_filled = min(t - L + 1, BUF)
            if t % adapt_interval == 0 and t > L + 30 and n_filled >= 20:
                phi_g = phi.to(DEVICE)
                sims = torch.nn.functional.cosine_similarity(
                    phi_g.unsqueeze(1), cen.unsqueeze(0), dim=2)
                wp = pw_med[sims.argmax(dim=1)]          # [N, d] prior column
                Xv = Xbuf[:, :n_filled].to(DEVICE)
                yv = ybuf[:, :n_filled].to(DEVICE)
                A = Xv.transpose(1, 2) @ Xv + lam * eye
                b = (Xv.transpose(1, 2) @ yv.unsqueeze(-1)).squeeze(-1) + lam * wp
                Wcur = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1).cpu()
            if Wcur is not None:
                pred = (phi.double() * Wcur).sum(dim=1).numpy()
                rates[:, t] = np.expm1(np.maximum(pred, 0))
            else:
                rates[:, t] = np.expm1(np.maximum(ewma, 0))
            if t % 5000 == 0:
                print(f"      faithful tick {t}/{T}", flush=True)
    return rates


def main():
    features = np.load(P21 / "features.npy")
    counts_all = np.load(P21 / "counts.npy")
    splits = np.load(P21 / "splits.npz")
    dur_df = pd.read_csv(P21 / "duration_stats.csv")
    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1,
                              device=DEVICE)
    ckpt = RUNS / "best_anil_ridge_s1_s0.pt"
    trainer.load(ckpt)
    print(f"checkpoint: {ckpt} lambda={trainer.head.ridge_lambda.item():.3f}",
          flush=True)
    pm = build_prototypes(trainer, features, counts_all, splits["s1_train"])

    out = json.load(open(OUT)) if OUT.exists() else {
        "prereg": "revision_r11_PREREG.md", "checkpoint": str(ckpt),
        "meta_lambda": float(trainer.head.ridge_lambda.item()),
        "seeds": SEEDS, "results": []}
    done = {(r["method"], r["split"], r["cost_ratio"], r["seed"])
            for r in out["results"]}

    for split in ["S1", "S2", "S3"]:
        key = {"S1": "s1_test", "S2": "s2_test", "S3": "s3_test"}[split]
        test_idx = splits[key]
        counts = counts_all[test_idx]
        feats = features[test_idx]
        if split == "S3":
            t0 = int(splits.get("s3_test_t_start", [10080])[0])
            counts = counts[:, t0:]
            feats = feats[:, t0:, :]
        dm = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
             for fi in test_idx]), nan=1.0)
        ds = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
             for fi in test_idx]), nan=0.5)
        print(f"### {split}: {len(test_idx)} funcs", flush=True)

        t_start = time.time()
        r_faith = rates_a5_faithful(counts.astype(np.float32), feats, trainer, pm)
        feats0 = feats.copy()
        feats0[:, :, 0] = 0.0
        r_chan0 = rates_a5(counts.astype(np.float32), feats0, trainer, DEVICE)
        print(f"  rates done in {time.time()-t_start:.0f}s", flush=True)

        jobs = []
        for rho in REDUCED_RHOS:
            tau = newsvendor_quantile(rho)
            for method, rates in [("A5_faithful", r_faith),
                                  ("A5_chan0zero", r_chan0)]:
                if (method, split, rho, SEEDS[-1]) in done:
                    continue
                pw, ka = decisions_from_rates(rates, tau)
                jobs.append((method, rho, SEEDS, counts, pw, ka,
                             dm, ds, COLD_INIT, split))
        with ProcessPoolExecutor(max_workers=4) as ex:
            for res in ex.map(run_config, jobs):
                for r in res:
                    r.pop("func_csr", None)
                out["results"].extend(res)
                r0 = res[0]
                print(f"    {split} {r0['method']:13s} rho={r0['cost_ratio']:<5} "
                      f"csr={np.mean([x['csr'] for x in res])*100:.3f}% "
                      f"wm={np.mean([x['wm_per_1k_inv'] for x in res]):.0f}",
                      flush=True)
                json.dump(out, open(OUT, "w"), default=float)

    # context: archived comparators
    camp = json.load(open(RUNS / "sim_results_des_revision.json"))["results"]
    ctx = {}
    for m in ["A5_full_system", "B4a_ewma"]:
        ctx[m] = {}
        for split in ["S1", "S2", "S3"]:
            for rho in REDUCED_RHOS:
                rows = [r for r in camp if r["method"] == m
                        and r["split"] == split and r["cost_ratio"] == rho]
                if rows:
                    ctx[m][f"{split}|{rho}"] = {
                        "csr": float(np.mean([r["csr"] for r in rows])),
                        "wm": float(np.mean([r["wm_per_1k_inv"] for r in rows]))}
    out["archived_comparators"] = ctx
    json.dump(out, open(OUT, "w"), default=float)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
