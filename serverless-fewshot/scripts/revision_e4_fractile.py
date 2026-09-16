# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision E4: mechanism audits for the CRPS != CSR result (B4).

(1) Fractile decomposition (primary). On 2021 S3 (temporal split — the
LSTM's cleanest, history-rich condition, as in revision_b4_lstm_des), compare
per-fractile pinball loss of count-quantile forecasts:
  - B7 per-function LSTM: native 19-quantile head (log1p space), count
    quantile = expm1(q). Training protocol identical to revision_b4_lstm_des
    (seed 0, 50 epochs x 16 windows, pinball loss).
  - A5 (deployed learned path, steady state): online ridge point rate
    (rates_a5) -> decision-layer-implied count quantile poisson.ppf(tau, rate).
  - B4a EWMA: same Poisson mapping on the EWMA rate.
Reported at the 19-point grid tau in {0.05..0.95} plus the decision-relevant
fractiles tau* = rho/(1+rho) for rho in {1, 10, 100} (0.5, 0.909, 0.990).
Registered clamp rule: quantile-head reads above 0.95 are clamped to the 0.95
quantile (the deployed behavior); Poisson ppf is evaluated at exact tau*.
Hypothesis: the LSTM's CRPS advantage concentrates in central quantiles and
vanishes (or reverses) at the tail fractiles the decision layer actually reads.

(2) Lag-shift / persistence audit (secondary, a la Christofidi SoCC'23).
For each method's rate series: per-function Pearson correlation of pred(t)
with y(t-delta) for delta in 0..30; distribution of the argmax delta; and MSE
vs the lag-1 persistence forecast. Tests whether forecasts are approximately
time-shifted inputs, explaining steady-state parity with EWMA.

No new constants: seeds/protocols reused from revision_b4_lstm_des and the
registered clamp rule. Source data: 2021 processed. GPU.
Output: results/runs/revision_e4_fractile.json

CHECKPOINT FIX (2026-08-01). The original run loaded
results/runs/best_anil_ridge_s2_s0.pt, which is wrong twice over:
 (a) split-lineage mismatch -- the name s{N}_s{seed} denotes the meta-
     training SPLIT, not the trace vintage: best_anil_ridge_s2_s0.pt is by
     origin a 2021 checkpoint meta-trained on s2_train (phase4_train.py:474,
     tag="anil_ridge_s2"). The codebase convention is checkpoint matched to
     the evaluated split, and no s3 checkpoint exists, so every 2021 S3
     campaign uses s1_s0: phase6_des.py:305, revision_a2_des.py:98,
     revision_e1_v4gate.py:93, revision_b3_severity.py:59,
     revision_e6_kmeans_ood.py:128, revision_e7_agegate.py:89.
     revision_a2_des.py is the decisive one -- it produces
     sim_results_des_revision.json, the very S3 CSR result this script
     exists to explain. Auditing a different body than the one that
     produced those CSRs audits the wrong model.
 (b) contamination -- the 7/16 2019 recovery chain (run_recovery.sh:40 ->
     phase4_train.py, which hardcodes RUNS_DIR and ignores SF_RUNS_DIR)
     overwrote the FILE AT THAT PATH with a 2019-trained model
     (md5 ff148276..., vs the clean 2021 original a2aad046... preserved in
     results_azure2021/runs), and this script's output postdates that
     write. Same failure mode that voided revision_m1_kproto v1/v2.
     Note this is an overwritten path, not "the 2019 chain's checkpoint" --
     s2_s0 by origin is 2021.
Checkpoint reads now honour SF_RUNS_DIR (defaulting to the clean
results_azure2021/runs tree) like the sibling scripts, and the resolved
path + md5 are recorded in the output JSON. Outputs still go to
results/runs.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)  # MKL batched linalg thrashes threads on this VM

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import rates_ewma, rates_a5  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES, QUANTILES, pinball_loss  # noqa: E402
from scipy import stats as sps  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"          # outputs
CKPT_DIR = Path(os.environ.get(                        # checkpoint reads
    "SF_RUNS_DIR", PROJECT_ROOT / "results_azure2021" / "runs"))
DEFAULT_CKPT = "best_anil_ridge_s1_s0.pt"              # 2021 lineage (see header)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L = 60
TRAIN_SEED = 0
RHOS_STAR = [1.0, 10.0, 100.0]
QL = np.linspace(0.05, 0.95, N_QUANTILES)
MAX_LAG = 30


def train_predict_lstm_quantiles(feats_f, counts_f, train_end, pred_ticks):
    """B4 protocol, but returns the full quantile matrix [len(ticks), Q]
    in log1p space."""
    in_features = feats_f.shape[1]
    ft = torch.from_numpy(feats_f).float()
    lstm = torch.nn.LSTM(in_features, 64, num_layers=1, batch_first=True).to(DEVICE)
    head = torch.nn.Linear(64, N_QUANTILES).to(DEVICE)
    opt = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=1e-3)
    qdev = QUANTILES.to(DEVICE)
    for _ in range(50):
        t_samples = np.random.randint(L, train_end, size=16)
        x = torch.stack([ft[t - L:t] for t in t_samples]).to(DEVICE)
        y = torch.tensor([np.log1p(counts_f[t]) for t in t_samples]).float().to(DEVICE)
        out, _ = lstm(x)
        pred = head(out[:, -1, :])
        loss = pinball_loss(pred, y, qdev)
        opt.zero_grad(); loss.backward(); opt.step()
    lstm.eval()
    qs = np.zeros((len(pred_ticks), N_QUANTILES), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(pred_ticks), 2048):
            ticks = pred_ticks[s:s + 2048]
            x = torch.stack([
                torch.cat([torch.zeros(max(0, L - t), in_features), ft[max(0, t - L):t]])
                for t in ticks]).to(DEVICE)
            out, _ = lstm(x)
            qs[s:s + 2048] = head(out[:, -1, :]).cpu().numpy()
    return qs


def pinball_count(y, q, tau):
    err = y - q
    return float(np.maximum(tau * err, (tau - 1) * err).mean())


def main(args):
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    features = np.load(PROCESSED_DIR / "features.npy")
    counts_all = np.load(PROCESSED_DIR / "counts.npy")
    splits = np.load(PROCESSED_DIR / "splits.npz")

    test_idx = splits["s3_test"]
    T_full = counts_all.shape[1]
    t0 = int(splits.get("s3_test_t_start", [10080])[0])
    n = len(test_idx)
    seg_counts = counts_all[test_idx]
    seg_feats = features[test_idx]
    y_eval = seg_counts[:, t0:].astype(np.float64)          # [n, Te] counts
    Te = y_eval.shape[1]
    print(f"S3: {n} funcs, eval ticks {Te} (t0={t0})")

    # ---- B7 LSTM quantiles ----
    t_start = time.time()
    lstm_q = np.zeros((n, Te, N_QUANTILES), dtype=np.float32)
    pred_ticks = list(range(t0, T_full))
    for i in range(n):
        lstm_q[i] = train_predict_lstm_quantiles(
            seg_feats[i], seg_counts[i], t0, pred_ticks)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n} LSTMs ({time.time()-t_start:.0f}s)", flush=True)
    lstm_cq = np.expm1(np.maximum(lstm_q, 0.0))             # count-space quantiles

    # ---- A5 / EWMA rates (honest online, full trace; slice eval) ----
    from src.meta.trainer import ANILMetaTrainer
    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1,
                              use_amp=True, device=DEVICE)
    ckpt = CKPT_DIR / args.checkpoint
    ckpt_md5 = hashlib.md5(ckpt.read_bytes()).hexdigest()
    print(f"A5 checkpoint: {ckpt} (md5 {ckpt_md5})", flush=True)
    trainer.load(ckpt)
    print("computing A5 online rates...", flush=True)
    a5_rates = rates_a5(seg_counts, seg_feats, trainer, DEVICE)[:, t0:]
    ew_rates = rates_ewma(seg_counts)[:, t0:]

    # ---- fractile pinball ----
    report_taus = list(QL) + [newsvendor_quantile(r) for r in RHOS_STAR]
    report_taus = sorted(set(round(t, 6) for t in report_taus))

    def lstm_quantile_at(tau):
        # registered clamp: reads above 0.95 use the 0.95 quantile
        tt = min(tau, QL[-1])
        w = np.interp(tt, QL, np.arange(N_QUANTILES))
        lo, hi = int(np.floor(w)), int(np.ceil(w))
        frac = w - lo
        return (1 - frac) * lstm_cq[:, :, lo] + frac * lstm_cq[:, :, hi]

    out = {"split": "S3", "n_functions": int(n), "eval_ticks": int(Te),
           "protocol": "revision_b4_lstm_des seed0; A5=rates_a5 online; "
                       "count-space pinball; poisson-implied quantiles for "
                       "rate methods; LSTM clamp>0.95 (registered rule)",
           "a5_checkpoint": str(ckpt), "a5_checkpoint_md5": ckpt_md5,
           "fractile_pinball": {}, "crps_grid_mean": {}, "lag_audit": {}}
    for tau in report_taus:
        rec = {}
        rec["B7_lstm"] = pinball_count(y_eval, lstm_quantile_at(tau), tau)
        for name, rates in [("A5_learned", a5_rates), ("B4a_ewma", ew_rates)]:
            q = sps.poisson.ppf(tau, np.maximum(rates, 1e-9))
            rec[name] = pinball_count(y_eval, q, tau)
        out["fractile_pinball"][f"{tau:.6f}"] = rec
        print(f"tau={tau:.3f}: " + " ".join(f"{k}={v:.5f}" for k, v in rec.items()),
              flush=True)

    for m in ["B7_lstm", "A5_learned", "B4a_ewma"]:
        grid_vals = [out["fractile_pinball"][f"{t:.6f}"][m] for t in QL]
        out["crps_grid_mean"][m] = 2 * float(np.mean(grid_vals))

    # ---- lag-shift audit ----
    def lag_audit(pred):
        best_lags = np.full(n, -1, dtype=int)
        r_at_0 = np.zeros(n); r_best = np.zeros(n)
        for f in range(n):
            y = y_eval[f]; p = pred[f]
            if y.std() < 1e-9 or p.std() < 1e-9:
                continue
            rs = []
            for d in range(MAX_LAG + 1):
                # pred(t) vs y(t-d)
                yy = y[:Te - d] if d else y
                pp = p[d:] if d else p
                rs.append(np.corrcoef(pp, yy)[0, 1] if yy.std() > 1e-9 else np.nan)
            rs = np.array(rs)
            if np.all(np.isnan(rs)):
                continue
            best_lags[f] = int(np.nanargmax(rs))
            r_at_0[f] = rs[0] if not np.isnan(rs[0]) else 0.0
            r_best[f] = np.nanmax(rs)
        valid = best_lags >= 0
        mse_pred = float(np.mean((pred[:, 1:] - y_eval[:, 1:]) ** 2))
        mse_persist = float(np.mean((y_eval[:, :-1] - y_eval[:, 1:]) ** 2))
        return {"median_best_lag": float(np.median(best_lags[valid])) if valid.any() else None,
                "frac_best_lag_ge1": float((best_lags[valid] >= 1).mean()) if valid.any() else None,
                "lag_hist_0_5": np.bincount(
                    np.clip(best_lags[valid], 0, 5), minlength=6).tolist() if valid.any() else None,
                "mean_r_lag0": float(r_at_0[valid].mean()) if valid.any() else None,
                "mean_r_best": float(r_best[valid].mean()) if valid.any() else None,
                "mse_forecast": mse_pred, "mse_lag1_persistence": mse_persist,
                "n_valid": int(valid.sum())}

    lstm_med_rate = lstm_cq[:, :, N_QUANTILES // 2].astype(np.float64)
    for name, pred in [("B7_lstm_median", lstm_med_rate),
                       ("A5_learned", a5_rates.astype(np.float64)),
                       ("B4a_ewma", ew_rates.astype(np.float64))]:
        out["lag_audit"][name] = lag_audit(pred)
        print(name, out["lag_audit"][name], flush=True)

    path = RUNS_DIR / args.out
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT,
                    help="A5 checkpoint filename inside SF_RUNS_DIR")
    ap.add_argument("--out", default="revision_e4_fractile.json",
                    help="output filename inside results/runs")
    main(ap.parse_args())
