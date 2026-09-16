# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision E2: natural-onboarding cohort robustness sweep (Azure-2019).

Grid: idle-gap in {1, 3, 7} days x day-1 activity threshold in {10, 30, 100}
invocations. Cohort(gap, thr) = functions whose FIRST invocation in the
entire trace occurs at t0 in [gap*1440, T-1440) with >= thr invocations in
[t0, t0+1440); capped at 100 per cell (rng seed 42, as in A4-2). Note the
definition is first-appearance-only by construction (zero invocations before
t0 anywhere in the trace) — the "first-appearance subset" of the directive
IS the cohort; the selection funnel documents this.

Configuration: CLEAN cross-trace setup (V2) — 2021-trained checkpoint +
2021-s2_train prototypes, evaluated on 2019 cohorts. Arms: A5_proto (learned
path), B4a_ewma, B1 fixed keep-alive. rhos {1,10,100}, DES seeds {0,1,2}.
Registered constants throughout; no new tuning.

Output: results/runs/revision_e2_cohort_sweep.json
  per cell: n_candidates, funnel, overall CSR per arm/rho, paired Wilcoxon
  (cold over full 4h window) A5_proto vs EWMA and vs B1, mean_diff.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)  # MKL thread-thrashing fix (see REVISION_LOG)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("SF_DATA_DIR", "processed_2019")
os.environ["SF_RUNS_DIR"] = str(PROJECT_ROOT / "results_azure2021" / "runs")

from scripts.eval_adapt_biased import (  # noqa: E402
    load_trainer, build_prototypes, PROCESSED_DIR, RUNS_DIR,
)
from scripts.phase63_onboarding_drift import DES_SEEDS, DEVICE  # noqa: E402
from scripts.phase6_des import decisions_from_rates, COLD_INIT  # noqa: E402
from src.sim.des import simulate_function  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402

L = 60
W = 240
RHOS = [1.0, 10.0, 100.0]
COHORT_CAP = 100
GAPS_D = [1, 3, 7]
THRS = [10, 30, 100]
QL = np.linspace(0.05, 0.95, N_QUANTILES)
TAU = newsvendor_quantile(10.0)
PROC_2021 = PROJECT_ROOT / "data" / "processed"


def build_cells(counts):
    N, T = counts.shape
    # one pass: first arrival + day-1 volume for every function
    first = np.full(N, -1, dtype=np.int64)
    day1 = np.zeros(N, dtype=np.int64)
    for f in range(N):
        row = np.asarray(counts[f])
        nz = np.flatnonzero(row)
        if len(nz) == 0:
            continue
        first[f] = nz[0]
        day1[f] = row[nz[0]:nz[0] + 1440].sum()
    cells = {}
    for gap in GAPS_D:
        for thr in THRS:
            mask = ((first >= gap * 1440) & (first < T - 1440) & (day1 >= thr))
            cand = np.flatnonzero(mask)
            rng = np.random.default_rng(42)
            if len(cand) > COHORT_CAP:
                idx = rng.choice(len(cand), COHORT_CAP, replace=False)
                picked = sorted(cand[i] for i in idx)
            else:
                picked = list(cand)
            cells[(gap, thr)] = {
                "funcs": [(int(f), int(first[f])) for f in picked],
                "funnel": {
                    "total_functions": int(N),
                    "ever_invoked": int((first >= 0).sum()),
                    "idle_prefix_ok": int(((first >= gap * 1440) &
                                           (first < T - 1440)).sum()),
                    "day1_threshold_ok": int(mask.sum()),
                    "capped": len(picked),
                },
            }
    return cells


def main():
    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    nF = features.shape[2]

    feat21 = np.load(PROC_2021 / "features.npy", mmap_mode="r")
    cnt21 = np.load(PROC_2021 / "counts.npy", mmap_mode="r")
    spl21 = np.load(PROC_2021 / "splits.npz")

    cells = build_cells(counts)
    union = sorted({pt for c in cells.values() for pt in c["funcs"]})
    print(f"cells built; union of cohort functions = {len(union)}", flush=True)

    trainer, biased_head = load_trainer(nF, seed=0)  # 2021 checkpoint
    centroids, proto_w = build_prototypes(
        trainer, biased_head, feat21, cnt21, spl21["s2_train"])

    # ---- embed union once: phi[U, W, 64] ----
    trainer.body.eval()
    U = len(union)
    phi_all = torch.zeros(U, W, 64)
    with torch.no_grad():
        for w in range(W):
            batch = []
            for (fi, t0) in union:
                t = t0 + w
                if t < L:
                    win = np.zeros((L, nF), dtype=np.float32)
                    win[L - t:] = features[fi, :t]
                else:
                    win = np.asarray(features[fi, t - L:t], dtype=np.float32)
                batch.append(torch.from_numpy(win))
            x = torch.stack(batch).to(DEVICE)
            phi_all[:, w] = trainer.body(x).cpu()
            if (w + 1) % 60 == 0:
                print(f"  embed {w+1}/{W}", flush=True)
    y_all = np.stack([np.log1p(np.asarray(counts[fi, t0:t0 + W], dtype=np.float64))
                      for (fi, t0) in union])
    seg_union = np.stack([np.asarray(counts[fi, t0:t0 + W], dtype=np.int64)
                          for (fi, t0) in union])

    # ---- prototype-biased few-shot preds for the union (GPU) ----
    import torch.nn.functional as Fun
    lam = trainer.head.ridge_lambda.item()
    n_out = N_QUANTILES
    phi_g = phi_all.to(DEVICE)
    y_g = torch.from_numpy(y_all).float().to(DEVICE)
    cen_g = centroids.to(DEVICE)
    pw_g = proto_w.to(DEVICE)
    eye_g = torch.eye(64, device=DEVICE)
    preds = np.zeros((U, W, n_out), dtype=np.float32)
    for w in range(W):
        phi_now = phi_g[:, w]
        sims = Fun.cosine_similarity(phi_now.unsqueeze(1), cen_g.unsqueeze(0), dim=2)
        Wp = pw_g[sims.argmax(dim=1)]
        if w == 0:
            W_h = Wp
        else:
            phi_s = phi_g[:, :w]
            y_s = y_g[:, :w].unsqueeze(-1).expand(U, w, n_out)
            PhiT = phi_s.transpose(1, 2)
            A = PhiT @ phi_s + lam * eye_g.unsqueeze(0)
            W_h = torch.linalg.solve(A, PhiT @ y_s + lam * Wp)
        preds[:, w] = (phi_now.unsqueeze(1) @ W_h).squeeze(1).cpu().numpy()
    print("union preds done", flush=True)

    rate_proto = np.zeros((U, W), dtype=np.float32)
    for w in range(W):
        val = np.array([np.interp(TAU, QL, preds[u, w]) for u in range(U)])
        rate_proto[:, w] = np.expm1(np.maximum(val, 0.0))
    ewma = np.zeros(U)
    rate_ewma = np.zeros((U, W), dtype=np.float32)
    for w in range(W):
        rate_ewma[:, w] = np.expm1(np.maximum(ewma, 0))
        ewma = 0.1 * np.log1p(seg_union[:, w].astype(np.float64)) + 0.9 * ewma

    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] for (fi, _) in union]), nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] for (fi, _) in union]), nan=0.5)
    upos = {pt: i for i, pt in enumerate(union)}

    from scipy import stats as sps
    out = {"config": "cross-trace clean (2021 ckpt+protos -> 2019 cohorts)",
           "grid": {"gaps_days": GAPS_D, "thresholds": THRS},
           "union_size": U, "cells": {}}
    for (gap, thr), cell in cells.items():
        rows = [upos[pt] for pt in cell["funcs"]]
        n = len(rows)
        key = f"gap{gap}d_thr{thr}"
        if n < 10:
            out["cells"][key] = {"funnel": cell["funnel"], "skipped_small_n": n}
            continue
        seg = seg_union[rows]
        cell_out = {"funnel": cell["funnel"], "n": n, "by_rho": {}}
        for rho in RHOS:
            tau = newsvendor_quantile(rho)
            dec = {"A5_proto": decisions_from_rates(rate_proto[rows], tau),
                   "B4a_ewma": decisions_from_rates(rate_ewma[rows], tau)}
            pw_b1 = np.zeros((n, W), dtype=np.int32)
            for i in range(n):
                act = (seg[i] > 0).astype(np.int32)
                conv = np.convolve(act, np.ones(10, dtype=np.int32))[:W]
                pw_b1[i, 1:] = (conv[:-1] > 0).astype(np.int32)
            dec["B1_fixed_keepalive"] = (pw_b1, np.full((n, W), 10.0, np.float32))
            r_out = {}
            for m, (pw_m, ka_m) in dec.items():
                tot_c = 0.0; tot_n = 0.0
                func_cold = np.zeros(n)
                for i in range(n):
                    gi = rows[i]
                    for seed in DES_SEEDS:
                        r = simulate_function(seg[i], pw_m[i], ka_m[i],
                                              float(dm[gi]), float(dstd[gi]),
                                              seed=seed * 7919 + i,
                                              cold_mu=COLD_INIT["mu"],
                                              cold_sigma=COLD_INIT["sigma"],
                                              track_rolling=True)
                        tot_c += r["roll_cold"].sum(); tot_n += r["roll_total"].sum()
                        func_cold[i] += r["roll_cold"].sum()
                func_cold /= len(DES_SEEDS)
                r_out[m] = {"overall_csr": float(tot_c / max(tot_n, 1e-9)),
                            "func_cold": func_cold.tolist()}
            paired = {}
            ours = np.array(r_out["A5_proto"]["func_cold"])
            for b in ["B4a_ewma", "B1_fixed_keepalive"]:
                diff = np.array(r_out[b]["func_cold"]) - ours
                nz = diff[diff != 0]
                p = float(sps.wilcoxon(nz)[1]) if len(nz) >= 6 else 1.0
                paired[b] = {"wilcoxon_p": p, "mean_diff": float(diff.mean())}
            cell_out["by_rho"][str(rho)] = {
                "csr": {m: r_out[m]["overall_csr"] for m in r_out},
                "paired_vs_A5_proto": paired}
            print(f"{key} rho={rho}: " +
                  " ".join(f"{m}={r_out[m]['overall_csr']*100:.2f}%" for m in r_out) +
                  f" | p(EWMA)={paired['B4a_ewma']['wilcoxon_p']:.2e}", flush=True)
        out["cells"][key] = cell_out

    path = RUNS_DIR / "revision_e2_cohort_sweep.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")


if __name__ == "__main__":
    main()
