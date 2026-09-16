# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP-H5 diagnostic: is the Huawei steady-state failure a *scale* problem?

Cell 2 did not replicate: on Huawei `h_mixed` the transferred learned path is
~7 pp worse than a per-function EWMA while holding *less* warm memory -- the
signature of systematic under-forecasting of an out-of-distribution traffic
scale.  Two explanations are observationally equivalent in the WP-H3 data:

  (a) cross-provider transfer degradation -- the Azure-2021 body mis-scales
      Huawei traffic, and the regime boundary itself is untouched;
  (b) a genuine property of the converged regime.

This script separates them without any meta-training.  A5's prediction is a
blend in log1p space, `z_t = 0.6 * ridge_t + 0.4 * ewma_t`.  We add one causal,
per-function additive correction in that same space,

      b_i(t) = beta * (log1p(y_i(t-1)) - z_i(t-1)) + (1 - beta) * b_i(t-1),
      z_cal_i(t) = z_i(t) + b_i(t),

i.e. a multiplicative rate recalibration, fitted online from exactly the
history EWMA already uses -- no future leakage, no new information, no
retraining.  beta is fixed a priori to the registered EWMA alpha (0.1).

If recalibration closes most of the gap, the failure is diagnosed as transfer
mis-scaling (a).  If the gap survives, it is evidence for (b).

The same diagnostic runs on the Azure-2021 steady pool (`--trace azure2021`),
where the body *was* meta-trained, to give the in-distribution reference for
the residual bias.

  export:  PYTHONPATH=/data/260715/site-packages:. python3.13 \
               scripts/revision_h5_scale_recal.py --trace huawei --export
  run:     PYTHONPATH=/data/260715/site-packages-des:. python3.13 \
               scripts/des_runner_fast.py \
               --jobs results/runs/split_jobs_recal_huawei \
               --out results/runs/revision_h5_recal_huawei.json
  report:  PYTHONPATH=/data/260715/site-packages:. python3.13 \
               scripts/revision_h5_scale_recal.py --report
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Arms of the diagnostic.  `ckpt` matters: the Azure steady-state campaign in
# the paper runs on best_anil_ridge_s1_s0, while the cross-trace transfer
# protocol (revision_a4_crosstrace.py, and therefore the whole H-series) uses
# best_anil_ridge_s2_s0.  Both are run on Huawei so the provider effect is not
# confounded with the checkpoint choice.  The Azure reference arms are S1 and
# S3, where the paper claims parity -- S2 is the intentionally adversarial
# cluster hold-out where the paper already reports EWMA winning.
TRACES = {
    "huawei": {"data_dir": "processed_huawei", "pool": "h_mixed",
               "split_name": "h_mixed_recal", "ckpt": "s2_s0"},
    "huawei_s1ckpt": {"data_dir": "processed_huawei", "pool": "h_mixed",
                      "split_name": "h_mixed_recal_s1", "ckpt": "s1_s0"},
    "azure2021": {"data_dir": "processed", "pool": "s2_test",
                  "split_name": "s2_test_recal", "ckpt": "s2_s0"},
    "azure2021_s1": {"data_dir": "processed", "pool": "s1_test",
                     "split_name": "s1_test_recal", "ckpt": "s1_s0"},
    "azure2021_s3": {"data_dir": "processed", "pool": "s3_test",
                     "split_name": "s3_test_recal", "ckpt": "s1_s0",
                     "t_start_key": "s3_test_t_start"},
}
CKPTS = {"s1_s0": "results_azure2021/runs/best_anil_ridge_s1_s0.pt",
         "s2_s0": "results_azure2021/runs/best_anil_ridge_s2_s0.pt"}
RHOS = [1.0, 10.0, 100.0]
SEEDS = [0, 1, 2]
STEADY_T = 20160
BETA = 0.1              # registered EWMA alpha; not tuned on either trace
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def rates_a5_with_recal(counts, features, trainer, device, batch_gpu=4096):
    """rates_a5, re-emitted alongside its bias-corrected twin.

    Identical ridge fits and identical blend as scripts.phase6_des.rates_a5 --
    the only addition is the causal residual EWMA `bias`, so the two rate
    matrices differ by exactly the recalibration under test.
    """
    import torch
    N, T = counts.shape
    L, d = 60, 64
    n_raw = features.shape[2]
    D = d + n_raw
    adapt_interval, BUF = 60, 180

    rates = np.zeros((N, T), dtype=np.float32)
    rates_cal = np.zeros((N, T), dtype=np.float32)
    features_t = torch.from_numpy(np.ascontiguousarray(features)).float()

    Xbuf = torch.zeros(N, BUF, D, dtype=torch.float64)
    ybuf = torch.zeros(N, BUF, dtype=torch.float64)
    W = None
    eyeD = torch.eye(D, dtype=torch.float64, device=device)

    ewma = np.zeros(N)
    alpha = 0.15
    bias = np.zeros(N)              # additive correction in log1p space
    prev_z = np.zeros(N)            # last emitted blend, for the residual
    resid_sum = np.zeros(N)         # diagnostics: mean residual before recal
    resid_sum_cal = np.zeros(N)
    resid_n = 0
    bias_trace = []

    for t in range(min(L, T)):
        rates[:, t] = np.expm1(np.maximum(ewma, 0))
        rates_cal[:, t] = rates[:, t]      # warmup: EWMA only, nothing to correct
        prev_z = ewma.copy()
        prev = counts[:, max(t - 1, 0)].astype(np.float64)
        ewma = alpha * np.log1p(prev) + (1 - alpha) * ewma

    trainer.body.eval()
    with torch.no_grad():
        for t in range(L, T):
            phis = []
            for s in range(0, N, batch_gpu):
                bx = features_t[s:s + batch_gpu, t - L:t, :].to(device)
                phis.append(trainer.body(bx).cpu())
            phi = torch.cat(phis, 0)
            raw = features_t[:, min(t, T - 1), :]
            combined = torch.cat([phi, raw], dim=1).double()

            prev = counts[:, max(t - 1, 0)].astype(np.float64)
            ewma = alpha * np.log1p(prev) + (1 - alpha) * ewma

            # --- causal bias update on the residual of the *previous* emit ---
            resid = np.log1p(prev) - prev_z
            bias = BETA * resid + (1 - BETA) * bias
            resid_sum += resid
            resid_sum_cal += resid - bias
            resid_n += 1

            slot = (t - L) % BUF
            Xbuf[:, slot] = combined
            ybuf[:, slot] = torch.from_numpy(np.log1p(prev))
            n_filled = min(t - L + 1, BUF)

            if t % adapt_interval == 0 and t > L + 30 and n_filled >= 20:
                Xv = Xbuf[:, :n_filled].to(device)
                yv = ybuf[:, :n_filled].to(device)
                A = Xv.transpose(1, 2) @ Xv + eyeD
                b = (Xv.transpose(1, 2) @ yv.unsqueeze(-1))
                W = torch.linalg.solve(A, b).squeeze(-1).cpu()

            if W is not None:
                ridge_pred = (combined * W).sum(dim=1).numpy()
                z = 0.6 * ridge_pred + 0.4 * ewma
            else:
                z = ewma
            rates[:, t] = np.expm1(np.maximum(z, 0))
            rates_cal[:, t] = np.expm1(np.maximum(z + bias, 0))
            prev_z = z

            if t % 5000 == 0:
                bias_trace.append({"t": t, "bias_median": float(np.median(bias)),
                                   "bias_mean": float(bias.mean())})
                print(f"      tick {t}/{T}  bias median {np.median(bias):+.4f}",
                      flush=True)

    diag = {
        "beta": BETA,
        "mean_log_residual_per_fn": (resid_sum / max(resid_n, 1)).tolist(),
        "mean_log_residual_per_fn_recal": (resid_sum_cal / max(resid_n, 1)).tolist(),
        "final_bias_per_fn": bias.tolist(),
        "bias_trace": bias_trace,
    }
    return rates, rates_cal, diag


def export(trace):
    cfg = TRACES[trace]
    os.environ["SF_DATA_DIR"] = cfg["data_dir"]
    from scripts.phase6_des import (PROCESSED_DIR, COLD_INIT,
                                    decisions_from_rates, rates_ewma)
    from src.decision.newsvendor import newsvendor_quantile

    import torch
    torch.set_num_threads(1)
    from src.models.heads import N_QUANTILES
    from src.meta.trainer import ANILMetaTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_mm = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts_mm = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    splits = np.load(PROCESSED_DIR / "splits.npz")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    rows = np.asarray(splits[cfg["pool"]])

    ckpt = CKPTS[cfg.get("ckpt", "s2_s0")]
    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features_mm.shape[2],
                              embedding_dim=64, n_quantiles=N_QUANTILES,
                              n_horizons=1, device=device)
    trainer.load(PROJECT_ROOT / ckpt)

    t0 = int(splits[cfg["t_start_key"]][0]) if "t_start_key" in cfg else 0
    counts = np.asarray(counts_mm[rows][:, t0:STEADY_T]).astype(np.float32)
    feats = np.asarray(features_mm[rows][:, t0:STEADY_T, :], dtype=np.float32)
    dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
    ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
    print(f"{trace}/{cfg['pool']}: {len(rows)} functions, "
          f"ticks {t0}..{STEADY_T}, {int(counts.sum()):,} invocations, "
          f"ckpt {Path(ckpt).name}", flush=True)

    ew = rates_ewma(counts)
    a5, a5_cal, diag = rates_a5_with_recal(counts, feats, trainer, device)

    r = np.asarray(diag["mean_log_residual_per_fn"])
    rc = np.asarray(diag["mean_log_residual_per_fn_recal"])
    print(f"  mean log-residual (actual - predicted), per function:")
    print(f"    baseline     median {np.median(r):+.4f}  mean {r.mean():+.4f}  "
          f"frac>0 {float((r > 0).mean()):.3f}")
    print(f"    recalibrated median {np.median(rc):+.4f}  mean {rc.mean():+.4f}  "
          f"frac>0 {float((rc > 0).mean()):.3f}")

    jdir = RUNS_DIR / f"split_jobs_recal_{trace}" / cfg["split_name"]
    jdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(jdir / "shared.npz", counts=counts.astype(np.int64),
                        dur_means=dm, dur_stds=ds)
    job_meta = []
    for method, rates in {"B4a_ewma": ew, "A5_full_system": a5,
                          "A5_recalibrated": a5_cal}.items():
        for rho in RHOS:
            pw, ka = decisions_from_rates(rates, newsvendor_quantile(rho))
            fname = f"{method}__rho{rho}.npz"
            np.savez_compressed(jdir / fname, prewarm=pw, keepalive=ka)
            job_meta.append({"method": method, "rho": rho, "file": fname})
    json.dump({"split": cfg["split_name"], "seeds": SEEDS,
               "cold_init": COLD_INIT, "jobs": job_meta},
              open(jdir / "jobs.json", "w"))
    diag.update({"trace": trace, "pool": cfg["pool"], "n_functions": len(rows),
                 "checkpoint": ckpt, "t_start": t0, "steady_t": STEADY_T})
    json.dump(diag, open(RUNS_DIR / f"revision_h5_recal_diag_{trace}.json", "w"),
              default=float)
    print(f"  exported {len(job_meta)} jobs -> {jdir}", flush=True)


def report():
    out, lines = {}, ["trace,rho,method,csr_pct,wm_per_1k"]
    for trace in TRACES:
        p = RUNS_DIR / f"revision_h5_recal_{trace}.json"
        dp = RUNS_DIR / f"revision_h5_recal_diag_{trace}.json"
        if not p.exists():
            print(f"  missing {p.name} -- skipped")
            continue
        rows = json.load(open(p))
        agg = {}
        for r in rows:
            agg.setdefault((r["cost_ratio"], r["method"]), []).append(r)
        cell = {k: {"csr": float(np.mean([x["csr"] for x in v])) * 100,
                    "wm": float(np.mean([x["wm_per_1k_inv"] for x in v]))}
                for k, v in agg.items()}
        for (rho, m), v in sorted(cell.items()):
            lines.append(f"{trace},{rho},{m},{v['csr']:.4f},{v['wm']:.1f}")
        t = {"by_rho": {}}
        for rho in RHOS:
            e = cell.get((rho, "B4a_ewma"))
            a = cell.get((rho, "A5_full_system"))
            c = cell.get((rho, "A5_recalibrated"))
            if not (e and a and c):
                continue
            gap, gap_cal = a["csr"] - e["csr"], c["csr"] - e["csr"]
            t["by_rho"][str(rho)] = {
                "ewma_csr_pct": e["csr"], "a5_csr_pct": a["csr"],
                "a5_recal_csr_pct": c["csr"],
                "gap_pp": gap, "gap_recal_pp": gap_cal,
                "gap_closed_frac": (1 - gap_cal / gap) if gap else float("nan"),
                "ewma_wm_per_1k": e["wm"], "a5_wm_per_1k": a["wm"],
                "a5_recal_wm_per_1k": c["wm"]}
        if dp.exists():
            d = json.load(open(dp))
            r = np.asarray(d["mean_log_residual_per_fn"])
            t["mean_log_residual_median"] = float(np.median(r))
            t["mean_log_residual_frac_positive"] = float((r > 0).mean())
            t["n_functions"] = d["n_functions"]
        out[trace] = t

    json.dump(out, open(RUNS_DIR / "revision_h5_scale_recal.json", "w"),
              indent=2, default=float)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "T_huawei_scale_recal.csv").write_text("\n".join(lines) + "\n")

    for trace, t in out.items():
        print(f"\n--- {trace} ({t.get('n_functions')} functions) ---")
        print(f"  median per-function log-residual "
              f"{t.get('mean_log_residual_median', float('nan')):+.4f} "
              f"(frac>0 {t.get('mean_log_residual_frac_positive', float('nan')):.3f})")
        for rho, v in t["by_rho"].items():
            print(f"  rho={rho:<6}: EWMA {v['ewma_csr_pct']:.4f}%  "
                  f"A5 {v['a5_csr_pct']:.4f}%  A5+recal {v['a5_recal_csr_pct']:.4f}%"
                  f"   gap {v['gap_pp']:+.4f} -> {v['gap_recal_pp']:+.4f} pp "
                  f"({v['gap_closed_frac']*100:.0f}% closed)")
    print("\nSaved revision_h5_scale_recal.json + T_huawei_scale_recal.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", choices=list(TRACES), default="huawei")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.export:
        export(a.trace)
    if a.report:
        report()
    if not (a.export or a.report):
        ap.error("pass --export or --report")
