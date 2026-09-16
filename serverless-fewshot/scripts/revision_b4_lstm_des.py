# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision B4: run the per-function LSTM (B7) through the DES at matched
decision logic, S2 and S3.

Training protocol identical to phase6_extended (1-layer LSTM(64) + linear
quantile head, 50 epochs x 16 windows, Adam 1e-3, pinball loss, seed 0).

Rate semantics (documented):
  S3 (temporal split): LSTM trained on each function's pre-test history
      (t < s3_test_t_start); rates = median-quantile prediction over the
      evaluated segment. Clean protocol: no leakage, abundant history.
  S2: LSTM trained on the first half of the trace; rates = EWMA for
      t < T/2 (training period), LSTM prediction for t >= T/2.

DES: REDUCED_RHOS x seeds 0..9 (matches revision campaign).
Output: results/runs/revision_b4_lstm_des.json
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
    rates_ewma, run_config, decisions_from_rates, COLD_INIT, REDUCED_RHOS,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES, QUANTILES, pinball_loss  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L = 60
SEEDS = list(range(10))
MED_IDX = N_QUANTILES // 2
TRAIN_SEED = 0


def train_predict_lstm(feats_f, counts_f, train_end, pred_ticks):
    """Train one per-function LSTM on [L, train_end); predict median rate
    for each tick in pred_ticks. Returns rates [len(pred_ticks)]."""
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
    rates = np.zeros(len(pred_ticks), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(pred_ticks), 2048):
            ticks = pred_ticks[s:s + 2048]
            x = torch.stack([
                torch.cat([torch.zeros(max(0, L - t), in_features), ft[max(0, t - L):t]])
                for t in ticks]).to(DEVICE)
            out, _ = lstm(x)
            med = head(out[:, -1, :])[:, MED_IDX].cpu().numpy()
            rates[s:s + 2048] = np.expm1(np.maximum(med, 0.0))
    return rates


def main():
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    features = np.load(PROCESSED_DIR / "features.npy")
    counts_all = np.load(PROCESSED_DIR / "counts.npy")
    splits = np.load(PROCESSED_DIR / "splits.npz")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")

    all_results = []
    for split in ["S2", "S3"]:
        key = {"S2": "s2_test", "S3": "s3_test"}[split]
        test_idx = splits[key]
        T_full = counts_all.shape[1]
        if split == "S3":
            t0 = int(splits.get("s3_test_t_start", [10080])[0])
        else:
            t0 = T_full // 2

        n = len(test_idx)
        seg_counts = counts_all[test_idx]
        seg_feats = features[test_idx]
        if split == "S3":
            eval_counts = seg_counts[:, t0:]
        else:
            eval_counts = seg_counts   # S2 DES runs the full trace

        print(f"### {split}: {n} funcs, training per-function LSTMs "
              f"(train_end={t0})...")
        t_start = time.time()
        Tval = eval_counts.shape[1]
        rates_b7 = np.zeros((n, Tval), dtype=np.float32)
        ew_full = rates_ewma(seg_counts)
        for i, fi in enumerate(test_idx):
            if split == "S3":
                pred_ticks = list(range(t0, T_full))
                r = train_predict_lstm(seg_feats[i], seg_counts[i], t0, pred_ticks)
                rates_b7[i] = r
            else:
                pred_ticks = list(range(t0, T_full))
                r = train_predict_lstm(seg_feats[i], seg_counts[i], t0, pred_ticks)
                rates_b7[i, :t0] = ew_full[i, :t0]
                rates_b7[i, t0:] = r
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{n} LSTMs ({time.time()-t_start:.0f}s)")
        print(f"  all LSTMs done in {time.time()-t_start:.0f}s")

        dm = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
             for fi in test_idx]), nan=1.0)
        ds = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
             for fi in test_idx]), nan=0.5)

        jobs = []
        for rho in REDUCED_RHOS:
            tau = newsvendor_quantile(rho)
            pw, ka = decisions_from_rates(rates_b7, tau)
            jobs.append(("B7_per_func_lstm", rho, SEEDS, eval_counts, pw, ka,
                         dm, ds, COLD_INIT, split))
        with ProcessPoolExecutor(max_workers=5) as ex:
            for res_list in ex.map(run_config, jobs):
                all_results.extend(res_list)
                r = res_list[0]
                print(f"    B7 rho={r['cost_ratio']:6.1f} "
                      f"CSR={np.mean([x['csr'] for x in res_list]):.4f} "
                      f"WM={np.mean([x['wm_per_1k_inv'] for x in res_list]):.0f}")

    path = RUNS_DIR / "revision_b4_lstm_des.json"
    with open(path, "w") as f:
        json.dump({"results": all_results, "train_seed": TRAIN_SEED,
                   "protocol": {"S3": "train pre-test, predict eval segment",
                                "S2": "EWMA before T/2, LSTM after"}},
                  f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")


if __name__ == "__main__":
    main()
