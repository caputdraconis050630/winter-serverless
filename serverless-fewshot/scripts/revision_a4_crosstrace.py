# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision V2/V3/E1(cohort): CROSS-TRACE natural onboarding on Azure-2019.

V2 audit (2026-07-18) found the original revision_a4_natural.py configuration
leaks: 82/100 cohort functions are members of the 2019 s2_train split used to
meta-train best_anil_ridge_s2_s0.pt (2019) and to build its prototypes, and
MetaDataset samples windows over the full timeline (no temporal cutoff), so
training saw post-onset windows of evaluated functions.

This script re-runs the evaluation in the clean configuration sanctioned by
the revision directive: body + prototypes meta-learned on the *2021 trace*
(results_azure2021/runs/best_anil_ridge_s2_s0.pt, prototypes from 2021
s2_train), evaluated on the 2019 natural-onboarding cohort. Cross-trace
transfer by construction: the two traces share no functions (different years,
hashed ids), so no cohort function can appear in meta-training.

Additionally evaluates the deployed gate configurations on the same cohort
(V3 + E1-cohort):
  gated_v3: cum==0 -> prototype zero-shot; 0<cum<100 -> EWMA; cum>=100 -> learned
  gated_v4: cum==0 -> prototype zero-shot; 1<=cum<=100 -> prototype-biased
            few-shot learned; cum>100 -> EWMA
(threshold 100 = registered constant; v4 is v3 with the EWMA/learned arms
swapped). Gating selects among per-tick rate forecasts; the shared decision
layer is applied to the composed rate matrix.

Cohort definition, arms, seeds, TAU semantics identical to
revision_a4_natural.py (registered constants; no new tuning).
Output: results/runs/revision_a4_crosstrace.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# MKL batched linalg on this VM thrashes threads (6.5s vs 0.02s per solve);
# GPU handles the heavy lifting, CPU ops stay single-threaded.
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("SF_DATA_DIR", "processed_2019")
# checkpoint loading redirected to the 2021-trained runs (V2 fix)
os.environ["SF_RUNS_DIR"] = str(PROJECT_ROOT / "results_azure2021" / "runs")

from scripts.eval_adapt_biased import (  # noqa: E402
    load_trainer, build_prototypes, PROCESSED_DIR, RUNS_DIR,
)
from scripts.phase63_onboarding_drift import DES_SEEDS, DEVICE  # noqa: E402
from scripts.phase6_des import decisions_from_rates, COLD_INIT  # noqa: E402
from src.sim.des import simulate_function, rolling_csr  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402

L = 60
W = 240
RHOS = [1.0, 10.0, 100.0]
COHORT_CAP = 100
MIN_IDLE_PREFIX = 3 * 1440
MIN_DAY1_INV = 30
GATE_THRESHOLD = 100          # registered constant (v1/v3/v4 share it)
QL = np.linspace(0.05, 0.95, N_QUANTILES)
TAU = newsvendor_quantile(10.0)

PROC_2021 = PROJECT_ROOT / "data" / "processed"


def find_natural_cohort(counts):
    N, T = counts.shape
    picked = []
    for f in range(N):
        row = np.asarray(counts[f])
        nz = np.flatnonzero(row)
        if len(nz) == 0:
            continue
        t0 = int(nz[0])
        if t0 < MIN_IDLE_PREFIX or t0 >= T - 1440:
            continue
        if row[t0:t0 + 1440].sum() < MIN_DAY1_INV:
            continue
        picked.append((f, t0))
    rng = np.random.default_rng(42)
    if len(picked) > COHORT_CAP:
        idx = rng.choice(len(picked), COHORT_CAP, replace=False)
        picked = [picked[i] for i in sorted(idx)]
    return picked


def rate_from_logq_rows(pred_q):
    val = np.array([np.interp(TAU, QL, row) for row in np.atleast_2d(pred_q)])
    return np.expm1(np.maximum(val, 0.0))


def main():
    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    nF = features.shape[2]

    # 2021 trace: prototype source (clean, cross-trace)
    feat21 = np.load(PROC_2021 / "features.npy", mmap_mode="r")
    cnt21 = np.load(PROC_2021 / "counts.npy", mmap_mode="r")
    spl21 = np.load(PROC_2021 / "splits.npz")
    assert feat21.shape[2] == nF, "feature dim mismatch between traces"

    pts = find_natural_cohort(counts)
    F_n = len(pts)
    print(f"natural cohort: {F_n} functions "
          f"(t0 median {np.median([t for _, t in pts]):.0f} min)")

    trainer, biased_head = load_trainer(nF, seed=0)   # 2021 checkpoint
    centroids, proto_w = build_prototypes(
        trainer, biased_head, feat21, cnt21, spl21["s2_train"])
    pw_all = proto_w  # [16, d, O] cpu

    trainer.body.eval()
    phi_all = torch.zeros(F_n, W, 64)
    with torch.no_grad():
        for w in range(W):
            batch = []
            for (fi, t0) in pts:
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
                      for (fi, t0) in pts])
    y_t = torch.from_numpy(y_all).float()

    lam = trainer.head.ridge_lambda.item()
    n_out = N_QUANTILES
    arms_pred = ["A5_proto", "A5_noproto", "A5_proto_zero"]
    preds = {a: np.zeros((F_n, W, n_out), dtype=np.float32) for a in arms_pred}
    import torch.nn.functional as Fun
    assign0 = None
    # all ridge algebra on GPU (CPU MKL batched solve is pathological here)
    phi_g = phi_all.to(DEVICE)
    y_g = y_t.to(DEVICE)
    cen_g = centroids.to(DEVICE)
    pw_g = pw_all.to(DEVICE)
    eye_g = torch.eye(64, device=DEVICE)
    for w in range(W):
        phi_now = phi_g[:, w]
        sims = Fun.cosine_similarity(phi_now.unsqueeze(1),
                                     cen_g.unsqueeze(0), dim=2)
        ids = sims.argmax(dim=1)
        if w == 0:
            assign0 = ids.cpu().numpy().copy()
        Wp = pw_g[ids]
        if w == 0:
            W_p, W_n = Wp.clone(), torch.zeros(F_n, 64, n_out, device=DEVICE)
        else:
            phi_s = phi_g[:, :w]
            y_s = y_g[:, :w].unsqueeze(-1).expand(F_n, w, n_out)
            PhiT = phi_s.transpose(1, 2)
            A = PhiT @ phi_s + lam * eye_g.unsqueeze(0)
            PhiTY = PhiT @ y_s
            W_p = torch.linalg.solve(A, PhiTY + lam * Wp)
            W_n = torch.linalg.solve(A, PhiTY)
        preds["A5_proto"][:, w] = (phi_now.unsqueeze(1) @ W_p).squeeze(1).cpu().numpy()
        preds["A5_noproto"][:, w] = (phi_now.unsqueeze(1) @ W_n).squeeze(1).cpu().numpy()
        preds["A5_proto_zero"][:, w] = (phi_now.unsqueeze(1) @ Wp).squeeze(1).cpu().numpy()
    print("ridge/pred stage done", flush=True)

    def crps_curve(pred):
        err = y_all[:, :, None] - pred
        pb = np.maximum(QL * err, (QL - 1) * err)
        return 2 * pb.mean(axis=(0, 2))

    seg_counts = np.stack([np.asarray(counts[fi, t0:t0 + W], dtype=np.int64)
                           for (fi, t0) in pts])
    rates = {a: np.zeros((F_n, W), dtype=np.float32) for a in preds}
    for a in preds:
        for w in range(W):
            rates[a][:, w] = rate_from_logq_rows(preds[a][:, w])
    ewma = np.zeros(F_n)
    rates["B4a_ewma"] = np.zeros((F_n, W), dtype=np.float32)
    for w in range(W):
        rates["B4a_ewma"][:, w] = np.expm1(np.maximum(ewma, 0))
        ewma = 0.1 * np.log1p(seg_counts[:, w].astype(np.float64)) + 0.9 * ewma
    rates["Oracle"] = seg_counts.astype(np.float32)

    # ---- gate composition (V3 / E1-cohort) ----
    # cum[f,w] = invocations strictly before tick w (history available at w)
    cum = np.zeros((F_n, W), dtype=np.int64)
    cum[:, 1:] = np.cumsum(seg_counts[:, :-1], axis=1)
    zero_h = cum == 0
    mid = (cum > 0) & (cum < GATE_THRESHOLD)
    conv = cum >= GATE_THRESHOLD
    rates["gated_v3"] = np.where(zero_h, rates["A5_proto_zero"],
                        np.where(mid, rates["B4a_ewma"], rates["A5_proto"]))
    rates["gated_v4"] = np.where(zero_h, rates["A5_proto_zero"],
                        np.where(mid, rates["A5_proto"], rates["B4a_ewma"]))
    routing = {"zero_history": float(zero_h.mean()),
               "midband_lt100": float(mid.mean()),
               "converged_ge100": float(conv.mean())}
    print("routing fractions:", routing)

    pw_b1 = np.zeros((F_n, W), dtype=np.int32)
    for f in range(F_n):
        act = (seg_counts[f] > 0).astype(np.int32)
        conv1 = np.convolve(act, np.ones(10, dtype=np.int32))[:W]
        pw_b1[f, 1:] = (conv1[:-1] > 0).astype(np.int32)
    b1_dec = (pw_b1, np.full((F_n, W), 10.0, np.float32))

    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] for (fi, _) in pts]), nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] for (fi, _) in pts]), nan=0.5)

    from scipy import stats as sps
    out = {"config": "cross-trace (2021-trained body+prototypes -> 2019 cohort)",
           "checkpoint": "results_azure2021/runs/best_anil_ridge_s2_s0.pt",
           "proto_source": "2021 s2_train",
           "n_functions": F_n, "cohort_cap": COHORT_CAP,
           "gate_threshold": GATE_THRESHOLD,
           "routing_fractions": routing,
           "assign_hist_k0": np.bincount(assign0, minlength=16).tolist(),
           "crps_vs_time": {a: crps_curve(preds[a]).tolist() for a in preds},
           "by_rho": {}}
    for rho in RHOS:
        tau = newsvendor_quantile(rho)
        dec = {m: decisions_from_rates(r, tau) for m, r in rates.items()}
        dec["B1_fixed_keepalive"] = b1_dec
        rho_out = {}
        for m, (pw_m, ka_m) in dec.items():
            roll_c = np.zeros(W); roll_n = np.zeros(W)
            func_cold60 = np.zeros(F_n); func_wm = np.zeros(F_n)
            func_cold_all = np.zeros(F_n)
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
                    func_cold_all[f] += r["roll_cold"].sum()
                    func_wm[f] += r["idle_mem_gb_s"]
            func_cold60 /= len(DES_SEEDS); func_wm /= len(DES_SEEDS)
            func_cold_all /= len(DES_SEEDS)
            print(f"  rho={rho} arm {m} simulated", flush=True)
            rho_out[m] = {"overall_csr": float(roll_c.sum() / max(roll_n.sum(), 1e-9)),
                          "func_cold60": func_cold60.tolist(),
                          "func_cold_all": func_cold_all.tolist(),
                          "wm_total": float(func_wm.sum()),
                          "rolling_csr": rolling_csr(roll_c, roll_n, window=15).tolist()}
        paired = {}
        for ours_name, base_name in [
                ("A5_proto", "A5_noproto"), ("A5_proto", "B4a_ewma"),
                ("A5_proto", "B1_fixed_keepalive"),
                ("gated_v3", "B4a_ewma"), ("gated_v4", "B4a_ewma"),
                ("gated_v4", "gated_v3"), ("gated_v4", "A5_proto")]:
            ours = np.array(rho_out[ours_name]["func_cold_all"])
            base = np.array(rho_out[base_name]["func_cold_all"])
            diff = base - ours
            nz = diff[diff != 0]
            p = float(sps.wilcoxon(nz)[1]) if len(nz) >= 6 else 1.0
            paired[f"{ours_name}_vs_{base_name}"] = {
                "wilcoxon_p": p, "mean_diff": float(diff.mean())}
        rho_out["_paired_cold_all"] = paired
        out["by_rho"][str(rho)] = rho_out
        print(f"rho={rho}: " + " ".join(
            f"{m}={v['overall_csr']*100:.2f}%" for m, v in rho_out.items()
            if not m.startswith("_")), flush=True)

    path = RUNS_DIR / "revision_a4_crosstrace.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")


if __name__ == "__main__":
    main()
