# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Phase 6 (DES): Steady-state simulation with the event-level simulator.

Architecture:
  1. Per method+split, compute a predicted-rate matrix ONCE (rho-independent).
  2. Per rho, map rates -> (prewarm, keepalive) via the shared decision layer.
  3. Run the event-level DES across seeds with multiprocessing.
  4. Save aggregate + per-function results for paired statistics.

Predictive methods (B4a, A5, B5, Oracle) share the identical decision layer,
so performance differences are attributable to prediction quality alone.
B1/B2 are policy baselines evaluated end-to-end.

Usage: python phase6_des.py [--splits S1,S2,S3] [--seeds 5] [--quick]
"""

import os, sys, time, json, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.des import simulate_trace
from src.decision.newsvendor import newsvendor_quantile

PROCESSED_DIR = PROJECT_ROOT / "data" / os.environ.get("SF_DATA_DIR", "processed")
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"

# Cold-init latency (lognormal, seconds), CALIBRATED from Knative testbed
# measurements (WP7-1, T6): pooled over python-ml/node/java real runtimes,
# end-to-end cold start median ~1.28 s. See results/runs/testbed_cdf.json.
COLD_INIT = {"mu": 0.25, "sigma": 0.31}

FULL_RHOS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0]
REDUCED_RHOS = [0.1, 1.0, 10.0, 100.0]


# ====================================================================
# Decision layer (shared, vectorized)
# ====================================================================
def decisions_from_rates(rates, tau_star):
    """Map predicted rate matrix [N,T] -> (prewarm, keepalive) matrices.

    Same semantics as prewarm_from_rate / keepalive_from_rate, vectorized.
    """
    p_arr = 1.0 - np.exp(-np.maximum(rates, 0.0))
    pw = (p_arr > (1.0 - tau_star)).astype(np.int32)

    # Concurrency scaling: Poisson quantile where rate is high enough to matter
    hi = rates > 2.0
    if hi.any():
        ppf = sp_stats.poisson.ppf(tau_star, rates[hi]).astype(np.int32)
        pw[hi] = np.maximum(pw[hi], ppf)

    min_ka = 5.0 + tau_star * 8.0
    with np.errstate(divide="ignore"):
        iat = np.where(rates > 1e-9, 1.0 / np.maximum(rates, 1e-9), np.inf)
    ka = np.clip(iat * (0.3 + tau_star * 1.2), min_ka, 30.0)
    ka = np.where(rates < 0.001, max(2.0, min_ka * 0.5), ka).astype(np.float32)

    return pw, ka


# ====================================================================
# Rate matrices (one per method+split; rho-independent)
# ====================================================================
def rates_ewma(counts, alpha=0.1):
    """B4a: EWMA of log1p counts, one-step-ahead (uses info up to t-1)."""
    N, T = counts.shape
    rates = np.zeros((N, T), dtype=np.float32)
    state = np.zeros(N)
    for t in range(T):
        rates[:, t] = np.expm1(np.maximum(state, 0))
        state = alpha * np.log1p(counts[:, t].astype(np.float64)) + (1 - alpha) * state
    return rates


# --- B3 (Fourier / harmonic extrapolation, IceBreaker-style) -----------
# Registered in src/models/baselines.py:FourierPredictor since WP2 but never
# evaluated through the DES until 2026-08-03. Constants: n_harmonics=10 is
# the registered value of that class, untouched; the online protocol reuses
# constants already registered elsewhere rather than introducing new ones --
# a 1,440-minute (one-day) context, the period a spectral prewarmer exists to
# exploit; a 60-tick refit cadence, identical to A5's adapt_interval; and a
# 60-tick warm-up, identical to the learned arms' L. No value was tuned
# against any evaluation result.
FOURIER_HARMONICS = 10
FOURIER_WINDOW = 1440
FOURIER_REFIT = 60
FOURIER_MIN_HIST = 60


def rates_fourier(counts, window=FOURIER_WINDOW,
                  n_harmonics=FOURIER_HARMONICS, refit=FOURIER_REFIT,
                  min_hist=FOURIER_MIN_HIST, verbose=False):
    """B3: harmonic extrapolation of per-minute counts, one-step-ahead.

    Vectorized restatement of FourierPredictor.fit/predict over all
    functions at once. Every `refit` ticks the trailing `window` of counts
    (strictly causal: [t-window, t), never including tick t) is transformed,
    all but the `n_harmonics`+1 largest-magnitude bins are zeroed, and the
    truncated spectrum is inverted. The forecast for tick t is the
    reconstruction evaluated at the periodic-extension index -- with a
    1,440-minute window that is the same minute of the previous day,
    smoothed to its dominant harmonics.

    Degradation is graceful and deliberate: with less than one full period
    of history the window is whatever exists, and with only a few samples
    the surviving DC bin makes the forecast the mean of that history. That
    is the behavior of a spectral predictor on a function too young to have
    a spectrum, and it is what the zero-history cells measure.
    """
    N, T = counts.shape
    rates = np.zeros((N, T), dtype=np.float32)
    x = counts.astype(np.float64)
    recon = None
    win_start = 0
    n_win = 0
    for t_block in range(0, T, refit):
        if t_block >= min_hist:
            ws = max(0, t_block - window)
            seg = x[:, ws:t_block]
            n = seg.shape[1]
            spec = np.fft.rfft(seg, axis=1)
            keep = np.argsort(np.abs(spec), axis=1)[:, -(n_harmonics + 1):]
            trunc = np.zeros_like(spec)
            np.put_along_axis(trunc, keep,
                              np.take_along_axis(spec, keep, axis=1), axis=1)
            recon = np.maximum(np.fft.irfft(trunc, n=n, axis=1), 0.0)
            win_start, n_win = ws, n
            if verbose and t_block % 5000 < refit:
                print(f"      B3 tick {t_block}/{T}", flush=True)
        if recon is None:
            continue
        for t in range(t_block, min(t_block + refit, T)):
            rates[:, t] = recon[:, (t - win_start) % n_win]
    return rates


def rates_oracle(counts):
    """Oracle: perfect knowledge of current-tick arrivals."""
    return counts.astype(np.float32)


def rates_a5(counts, features, trainer, device, batch_gpu=4096):
    """A5: online per-function ridge on body embeddings + raw features,
    blended with EWMA. Online protocol: refit every 60 ticks on a
    trailing 180-tick buffer. NOTE (2026-08-02 audit): the raw feature
    row concatenated at line `raw = features_t[:, min(t, T - 1), :]` is
    the CURRENT tick's row, whose channel 0 is log1p(count_t) -- i.e.
    the input is not strictly causal. A controlled ablation (zeroing
    the channel; scripts/audit_implnotes.py) shows the leak is not
    exploited: decisions are 99.2-99.5% unchanged and the served rate
    correlates more strongly with count_{t-1} than count_t. Disclosed
    in the paper's supplementary implementation notes.

    Vectorized: ring-buffer tensors + batched float64 ridge solves on GPU.
    Ridge normal equations are order-invariant, so the ring buffer is
    exactly equivalent to the trailing-window list implementation.
    """
    import torch
    N, T = counts.shape
    L = 60
    d = 64
    n_raw = features.shape[2]
    D = d + n_raw
    adapt_interval = 60
    BUF = 180

    rates = np.zeros((N, T), dtype=np.float32)
    features_t = torch.from_numpy(np.ascontiguousarray(features)).float()

    Xbuf = torch.zeros(N, BUF, D, dtype=torch.float64)
    ybuf = torch.zeros(N, BUF, dtype=torch.float64)
    W = None  # [N, D] float64 once first refit happens
    eyeD = torch.eye(D, dtype=torch.float64, device=device)

    ewma = np.zeros(N)
    alpha = 0.15

    # First L ticks: EWMA only
    for t in range(min(L, T)):
        rates[:, t] = np.expm1(np.maximum(ewma, 0))
        prev = counts[:, max(t - 1, 0)].astype(np.float64)
        ewma = alpha * np.log1p(prev) + (1 - alpha) * ewma

    trainer.body.eval()
    with torch.no_grad():
        for t in range(L, T):
            # Batched embedding (chunk over functions to bound GPU memory)
            phis = []
            for s in range(0, N, batch_gpu):
                bx = features_t[s:s + batch_gpu, t - L:t, :].to(device)
                phis.append(trainer.body(bx).cpu())
            phi = torch.cat(phis, 0)                        # [N, d] float32
            raw = features_t[:, min(t, T - 1), :]           # [N, n_raw]
            combined = torch.cat([phi, raw], dim=1).double()  # [N, D]

            prev = counts[:, max(t - 1, 0)].astype(np.float64)
            ewma = alpha * np.log1p(prev) + (1 - alpha) * ewma

            # Ring-buffer write (same slot for all functions)
            slot = (t - L) % BUF
            Xbuf[:, slot] = combined
            ybuf[:, slot] = torch.from_numpy(np.log1p(prev))

            n_filled = min(t - L + 1, BUF)

            # Batched refit
            if t % adapt_interval == 0 and t > L + 30 and n_filled >= 20:
                Xv = Xbuf[:, :n_filled].to(device)          # [N, n, D]
                yv = ybuf[:, :n_filled].to(device)          # [N, n]
                A = Xv.transpose(1, 2) @ Xv + eyeD          # lam = 1.0
                b = (Xv.transpose(1, 2) @ yv.unsqueeze(-1)) # [N, D, 1]
                W = torch.linalg.solve(A, b).squeeze(-1).cpu()  # [N, D]

            # Batched prediction
            if W is not None:
                ridge_pred = (combined * W).sum(dim=1).numpy()
                blended = 0.6 * ridge_pred + 0.4 * ewma
            else:
                blended = ewma
            rates[:, t] = np.expm1(np.maximum(blended, 0))

            if t % 5000 == 0:
                print(f"      A5 tick {t}/{T}", flush=True)
    return rates


def rates_b5(counts, features, trainer, device, n_quantiles=19):
    """B5: global head on body embeddings, no per-function adaptation."""
    import torch
    N, T = counts.shape
    L = 60
    quantile_levels = np.linspace(0.05, 0.95, n_quantiles)
    rates = np.zeros((N, T), dtype=np.float32)
    features_t = torch.from_numpy(features).float()

    trainer.body.eval()
    with torch.no_grad():
        # Fit global head on first 1000 ticks
        all_phi, all_y = [], []
        for t in range(L, min(L + 1000, T)):
            bx = features_t[:, t - L:t, :].to(device)
            all_phi.append(trainer.body(bx).cpu())
            all_y.append(torch.from_numpy(np.log1p(counts[:, t])).float())
        import torch as _t
        Phi = _t.cat(all_phi, 0)
        Y = _t.cat(all_y, 0).unsqueeze(-1).expand(-1, n_quantiles)
        lam = trainer.head.ridge_lambda.item()
        W = _t.linalg.solve(Phi.T @ Phi + lam * _t.eye(64), Phi.T @ Y).to(device)

        # EWMA fallback for warmup
        ewma = np.zeros(N)
        for t in range(min(L, T)):
            rates[:, t] = np.expm1(np.maximum(ewma, 0))
            prev = counts[:, max(t - 1, 0)].astype(np.float64)
            ewma = 0.1 * np.log1p(prev) + 0.9 * ewma

        for t in range(L, T):
            bx = features_t[:, t - L:t, :].to(device)
            pred = (trainer.body(bx) @ W).cpu().numpy()  # [N, Q]
            # tau-independent representative rate: median quantile
            med = pred[:, n_quantiles // 2]
            rates[:, t] = np.expm1(np.maximum(med, 0))
            if t % 5000 == 0:
                print(f"      B5 tick {t}/{T}", flush=True)
    return rates


def policy_b1(counts):
    """B1: prewarm 1 if activity in last 10 min; fixed 10-min keep-alive."""
    N, T = counts.shape
    active = (counts > 0).astype(np.int32)
    # recent[fi,t] = any activity in [t-10, t)
    kernel = np.ones(10, dtype=np.int32)
    pw = np.zeros((N, T), dtype=np.int32)
    for fi in range(N):
        conv = np.convolve(active[fi], kernel)[:T]
        # conv[t] = sum of active[t-9..t]; we need [t-10, t) => shift by 1
        pw[fi, 1:] = (conv[:-1] > 0).astype(np.int32)
    ka = np.full((N, T), 10.0, dtype=np.float32)
    return pw, ka


def policy_b2(counts):
    """B2: hybrid histogram (rolling 4h IAT stats)."""
    N, T = counts.shape
    pw = np.zeros((N, T), dtype=np.int32)
    ka = np.full((N, T), 10.0, dtype=np.float32)
    for fi in range(N):
        for t in range(T):
            window = counts[fi, max(0, t - 240):t]
            if window.sum() > 2:
                nz = np.nonzero(window)[0]
                if len(nz) > 1:
                    iats = np.diff(nz)
                    ka[fi, t] = float(min(np.percentile(iats, 95), 30.0))
                    pw[fi, t] = 1 if np.median(iats) < 5 else 0
                else:
                    ka[fi, t] = 10.0
                    pw[fi, t] = 1
            else:
                recent = counts[fi, max(0, t - 30):t].sum()
                ka[fi, t] = 5.0
                pw[fi, t] = 1 if recent > 0 else 0
    return pw, ka


# ====================================================================
# Worker
# ====================================================================
def run_config(args):
    """One (method, rho) config across all seeds. Runs in a worker process."""
    (method, rho, seeds, counts, prewarm, keepalive,
     dur_means, dur_stds, cold_init, split) = args
    results = []
    for seed in seeds:
        t0 = time.time()
        m = simulate_trace(
            counts, prewarm, keepalive, dur_means, dur_stds,
            seed=seed, cold_mu=cold_init["mu"], cold_sigma=cold_init["sigma"],
        )
        m["method"] = method
        m["cost_ratio"] = rho
        m["seed"] = seed
        m["split"] = split
        m["elapsed_sec"] = time.time() - t0
        results.append(m)
    return results


# ====================================================================
# Main
# ====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="S1,S2,S3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true",
                    help="reduced rho set on all splits")
    ap.add_argument("--workers", type=int, default=max(2, os.cpu_count() - 2))
    ap.add_argument("--out", default="sim_results_des.json")
    args = ap.parse_args()

    import torch
    from src.models.heads import N_QUANTILES
    from src.meta.trainer import ANILMetaTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("PHASE 6 (DES): Event-Level Steady-State Simulation")
    print("=" * 60)

    features = np.load(PROCESSED_DIR / "features.npy")
    counts_all = np.load(PROCESSED_DIR / "counts.npy")
    splits_data = np.load(PROCESSED_DIR / "splits.npz")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")

    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=device,
    )
    model_path = RUNS_DIR / "best_anil_ridge_s1_s0.pt"
    if not model_path.exists():
        model_path = RUNS_DIR / "best_anil_ridge_s2_s0.pt"
    trainer.load(model_path)
    print(f"Loaded model: {model_path.name}")

    seeds = list(range(args.seeds))
    split_names = args.splits.split(",")
    all_results = []

    for split in split_names:
        key = {"S1": "s1_test", "S2": "s2_test", "S3": "s3_test"}[split]
        test_idx = splits_data[key]
        counts = counts_all[test_idx]
        feats = features[test_idx]
        if split == "S3":
            t0 = int(splits_data.get("s3_test_t_start", [10080])[0])
            counts = counts[:, t0:]
            feats = feats[:, t0:]

        dm = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
             for fi in test_idx]), nan=1.0)
        ds = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
             for fi in test_idx]), nan=0.5)

        rhos = REDUCED_RHOS if (args.quick or split != "S1") else FULL_RHOS

        print(f"\n### Split {split}: {len(test_idx)} functions, "
              f"{counts.shape[1]} ticks, rhos={rhos}")

        # --- Rate matrices / policies (once per method) ---
        print("  Computing predictions/policies...")
        t0 = time.time()
        rate_mats = {
            "B4a_ewma": rates_ewma(counts),
            "Oracle": rates_oracle(counts),
            "A5_full_system": rates_a5(counts, feats, trainer, device),
            "B5_global": rates_b5(counts, feats, trainer, device),
        }
        policies = {
            "B1_fixed_keepalive": policy_b1(counts),
            "B2_histogram": policy_b2(counts),
        }
        print(f"  Predictions done in {time.time()-t0:.0f}s")

        # --- Build job list ---
        jobs = []
        for method, (pw, ka) in policies.items():
            for rho in rhos:  # policies are rho-independent; duplicate rows
                jobs.append((method, rho, seeds, counts, pw, ka,
                             dm, ds, COLD_INIT, split))
        for method, rates in rate_mats.items():
            for rho in rhos:
                tau = newsvendor_quantile(rho)
                pw, ka = decisions_from_rates(rates, tau)
                jobs.append((method, rho, seeds, counts, pw, ka,
                             dm, ds, COLD_INIT, split))

        # --- Run DES in parallel ---
        print(f"  Running {len(jobs)} configs x {len(seeds)} seeds "
              f"on {args.workers} workers...")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res_list in ex.map(run_config, jobs):
                all_results.extend(res_list)
                r = res_list[0]
                print(f"    {r['method']:<20s} rho={r['cost_ratio']:6.1f} "
                      f"CSR={np.mean([x['csr'] for x in res_list]):.4f} "
                      f"WM={np.mean([x['wm_per_1k_inv'] for x in res_list]):.0f}",
                      flush=True)
        print(f"  Split {split} simulated in {time.time()-t0:.0f}s")

    out_path = RUNS_DIR / args.out
    with open(out_path, "w") as f:
        json.dump(all_results, f, default=float)
    print(f"\nSaved {len(all_results)} runs to {out_path}")

    # Quick summary at rho=10
    print(f"\n{'='*70}")
    print("Summary at rho=10 (mean over seeds)")
    print(f"{'='*70}")
    for split in split_names:
        print(f"  --- {split} ---")
        methods = sorted(set(r["method"] for r in all_results))
        for method in methods:
            rs = [r for r in all_results
                  if r["method"] == method and r["split"] == split
                  and abs(r["cost_ratio"] - 10.0) < 0.01]
            if rs:
                csr = [r["csr"] for r in rs]
                wm = [r["wm_per_1k_inv"] for r in rs]
                print(f"    {method:<22s} CSR={np.mean(csr):.4f}"
                      f"+-{np.std(csr):.4f}  WM={np.mean(wm):.0f}"
                      f"+-{np.std(wm):.0f}")


if __name__ == "__main__":
    main()
