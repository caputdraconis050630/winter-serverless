# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision V1: FULL Shahrad'20 hybrid histogram baseline (B2f).

The existing B2 (`policy_b2` in phase6_des.py) is percentile-only: rolling
4h IAT window, ka = 95th pct capped 30 min, pw = median IAT < 5 min. The
paper will rename it "histogram (percentile-only)". This module implements
the complete ATC'20 policy, verified against the primary source
(arXiv:2003.03423, Section 4.2; details logged in REVISION_LOG.md):

  - Per-function idle-time (IT) histogram, 1-minute bins, range 240 min;
    ITs >= range are out-of-bounds (OOB).
  - Head/tail cutoffs at the 5th/99th percentiles (head rounded down, tail
    rounded up); pre-warming window = head, keep-alive until tail. 10%
    margin: pre-warm window reduced 10%, keep-alive extended 10%.
    Head == 0 -> no unloading after execution (pre-warm window 0).
  - Pattern-representativeness: CV of histogram bin counts; CV < 2 (their
    default) OR too few ITs -> standard keep-alive fallback: no pre-warming,
    keep-alive = histogram range (240 min).
  - Too many OOB ITs -> ARIMA tier: predict the next IT, pre-warm window =
    prediction - 15%, keep-alive covers +/-15% of the prediction; model
    updated at every invocation.

Constants from Shahrad'20 defaults (not tuned by us): RANGE=240, HEAD=5th,
TAIL=99th, MARGIN=10%, CV_THRESHOLD=2, ARIMA_MARGIN=15%.
Registered pre-fixed constants where ATC'20 states no number:
  HH_MIN_ITS=10        (min IT samples before the histogram is trusted;
                        smallest count giving stable 5/99th pct estimates)
  HH_OOB_FRAC=0.5      ("histogram does not capture most ITs" read literally:
                        majority of ITs out of bounds)
  HH_ARIMA_MAXHIST=50  (IT history cap for refits; ARIMA-tier functions are
                        sparse by construction, bounded compute)
Implementation substitution (documented): pmdarima.auto_arima is unavailable
in this environment; we use statsmodels ARIMA with AIC selection over the
small order grid p<=2, d<=1, q<=2 (auto_arima's default stepwise space),
order searched on first ARIMA-tier entry per function, parameters refit at
every subsequent invocation (Shahrad'20 refits per invocation).

Modes:
  --trace 2021   inline DES (172 funcs), seeds 0..9, REDUCED_RHOS ->
                 results/runs/revision_v1_hybridfull_2021.json
  --trace 2019   read des_sample_manifest.json rows, compute policy per split,
                 export job npzs to results/runs/des_jobs_2019_v1/<SPLIT>/
                 (run des_runner_fast on them after the recovery chain frees
                 the numba env), seeds mirror the chain's jobs.json
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Shahrad'20 defaults (verified against arXiv:2003.03423)
HH_RANGE = 240
HH_HEAD_PCT = 0.05
HH_TAIL_PCT = 0.99
HH_MARGIN = 0.10
HH_CV_THRESHOLD = 2.0
HH_ARIMA_MARGIN = 0.15
# Registered pre-fixed constants (no ATC'20 numeric default; see docstring)
HH_MIN_ITS = 10
HH_OOB_FRAC = 0.5
HH_ARIMA_MAXHIST = 50
ARIMA_ORDER_GRID = [(p, d, q) for p in (0, 1, 2) for d in (0, 1)
                    for q in (0, 1, 2) if (p, q) != (0, 0)]


def _fit_arima_predict(its, state):
    """AIC order search on first call (cached in state), refit + one-step
    forecast on subsequent calls. Returns predicted next IT (minutes) or
    None if fitting is impossible."""
    from statsmodels.tsa.arima.model import ARIMA
    y = np.asarray(its[-HH_ARIMA_MAXHIST:], dtype=float)
    if len(y) < 4 or np.all(y == y[0]):
        return float(y.mean()) if len(y) else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if state.get("order") is None:
            best = None
            for order in ARIMA_ORDER_GRID:
                try:
                    res = ARIMA(y, order=order).fit()
                    if best is None or res.aic < best[0]:
                        best = (res.aic, order, res)
                except Exception:
                    continue
            if best is None:
                return float(y.mean())
            state["order"] = best[1]
            fitted = best[2]
        else:
            try:
                fitted = ARIMA(y, order=state["order"]).fit()
            except Exception:
                return float(y.mean())
        try:
            pred = float(fitted.forecast(1)[0])
        except Exception:
            return float(y.mean())
    return max(pred, 1.0)


def policy_b2_full(counts, verbose_every=200):
    """Full hybrid histogram policy -> (prewarm, keepalive) tick matrices.

    Tick mapping (engine contract: pw = pool target at tick, ka = idle
    eviction horizon in minutes at tick):
      histogram mode after an execution at tick a:
        alive window [a + 0.9*head, a + 1.1*tail]; pw=1 and ka=window length
        inside it; pw=0, ka=0 outside (immediate unload / post-window kill).
        head==0: no unload, alive from a.
      standard keep-alive: pw=0, ka=HH_RANGE.
      ARIMA: alive window [a + 0.85*pred, a + 1.15*pred], same channel use.
    Idle time is approximated by the inter-arrival gap at 1-min ticks
    (executions last <<1 tick in these traces; same approximation as the
    percentile-only baseline).
    """
    N, T = counts.shape
    pw = np.zeros((N, T), dtype=np.int32)
    ka = np.zeros((N, T), dtype=np.float32)
    n_arima_funcs = 0
    t_start = time.time()

    for f in range(N):
        nz = np.flatnonzero(counts[f])
        # before any observation: standard keep-alive (conservative)
        end0 = nz[0] + 1 if len(nz) else T
        ka[f, :end0] = HH_RANGE
        if len(nz) == 0:
            continue

        hist = np.zeros(HH_RANGE, dtype=np.int64)
        n_in = 0
        n_oob = 0
        its = []
        arima_state = {}
        used_arima = False
        # current windows: (mode, start_off, end_off); standard => ka=RANGE
        mode = "standard"
        start_off = end_off = 0

        arrivals = list(nz) + [None]
        for i, a in enumerate(arrivals[:-1]):
            nxt = arrivals[i + 1] if arrivals[i + 1] is not None else T
            if i > 0:
                it = a - arrivals[i - 1]
                its.append(it)
                if it >= HH_RANGE:
                    n_oob += 1
                else:
                    hist[it] += 1
                    n_in += 1
                ntot = n_in + n_oob
                if ntot < HH_MIN_ITS:
                    mode = "standard"
                elif n_oob / ntot > HH_OOB_FRAC:
                    pred = _fit_arima_predict(its, arima_state)
                    if pred is None:
                        mode = "standard"
                    else:
                        mode = "arima"
                        used_arima = True
                        start_off = (1 - HH_ARIMA_MARGIN) * pred
                        end_off = (1 + HH_ARIMA_MARGIN) * pred
                else:
                    m = hist.mean()
                    cv = hist.std() / m if m > 0 else 0.0
                    if cv < HH_CV_THRESHOLD:
                        mode = "standard"
                    else:
                        cum = np.cumsum(hist)
                        head = int(np.searchsorted(cum, HH_HEAD_PCT * n_in,
                                                   side="left"))       # round down
                        tail = int(np.searchsorted(cum, HH_TAIL_PCT * n_in,
                                                   side="left")) + 1   # round up
                        mode = "hist"
                        start_off = (1 - HH_MARGIN) * head
                        end_off = (1 + HH_MARGIN) * tail

            # fill (a, nxt]: decisions for the idle period starting at a.
            # Includes tick nxt itself — the eviction check at the start of
            # tick nxt still belongs to the idle period that began at a
            # (the next cycle overwrites from nxt+1 on).
            s, e = a + 1, nxt + 1
            if s >= T:
                break
            e = min(e, T)
            if mode == "standard":
                ka[f, s:e] = HH_RANGE
            else:
                w_start = a + int(np.floor(start_off))
                w_end = a + int(np.ceil(end_off))
                w_len = max(w_end - w_start, 1)
                ws, we = max(s, w_start), min(e, w_end + 1)
                if ws < we:
                    pw[f, ws:we] = 1
                    ka[f, ws:we] = w_len
                # outside the alive window pw=0, ka=0 (unload phases)

        if used_arima:
            n_arima_funcs += 1
        if verbose_every and (f + 1) % verbose_every == 0:
            print(f"    b2_full {f+1}/{N} funcs ({time.time()-t_start:.0f}s, "
                  f"arima funcs so far {n_arima_funcs})", flush=True)

    print(f"  policy_b2_full done: {N} funcs, {n_arima_funcs} used ARIMA tier "
          f"({time.time()-t_start:.0f}s)", flush=True)
    return pw, ka


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", choices=["2021", "2019"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from scripts.phase6_des import COLD_INIT, REDUCED_RHOS, run_config
    RUNS_DIR = PROJECT_ROOT / "results" / "runs"

    if args.trace == "2021":
        PROC = PROJECT_ROOT / "data" / "processed"
        counts = np.load(PROC / "counts.npy")
        dur_df = pd.read_csv(PROC / "duration_stats.csv")
        N = counts.shape[0]
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[:N], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[:N], nan=0.5)
        splits = np.load(PROC / "splits.npz")
        seeds = list(range(10))

        all_results = []
        for split, key in [("S1", "s1_test"), ("S2", "s2_test"), ("S3", "s3_test")]:
            idx = splits[key]
            seg = counts[idx]
            if split == "S3":
                ts = int(splits.get("s3_test_t_start", [10080])[0])
                seg = seg[:, ts:]
            print(f"### {split}: {len(idx)} funcs, {seg.shape[1]} ticks")
            pwm, kam = policy_b2_full(seg, verbose_every=50)
            jobs = [("B2f_hybrid_full", rho, seeds, seg, pwm, kam,
                     dm[idx], ds[idx], COLD_INIT, split)
                    for rho in REDUCED_RHOS]
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for res_list in ex.map(run_config, jobs):
                    all_results.extend(res_list)
                    r = res_list[0]
                    print(f"    {r['method']} rho={r['cost_ratio']} "
                          f"CSR={np.mean([x['csr'] for x in res_list]):.4f} "
                          f"WM={np.mean([x['wm_per_1k_inv'] for x in res_list]):.0f}",
                          flush=True)
        out = RUNS_DIR / "revision_v1_hybridfull_2021.json"
        with open(out, "w") as fp:
            json.dump(all_results, fp, default=float)
        blob = open(out, "rb").read()
        assert blob and blob.count(0) == 0
        json.load(open(out))
        print(f"Saved + verified {out} ({len(all_results)} rows)")

    else:  # 2019: export jobs for the numba runner
        manifest = json.load(open(RUNS_DIR / "des_sample_manifest.json"))
        jobs_root = RUNS_DIR / "des_jobs_2019_v1"
        src_root = RUNS_DIR / "des_jobs_2019"
        for split, m in manifest.items():
            src_jobs = json.load(open(src_root / split / "jobs.json"))
            shared = np.load(src_root / split / "shared.npz")
            seg = shared["counts"]
            print(f"### {split}: {seg.shape[0]} funcs, {seg.shape[1]} ticks "
                  f"(from chain shared.npz; seeds {src_jobs['seeds']})")
            pwm, kam = policy_b2_full(seg, verbose_every=200)
            jdir = jobs_root / split
            jdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(jdir / "shared.npz", counts=seg,
                                dur_means=shared["dur_means"],
                                dur_stds=shared["dur_stds"])
            job_meta = []
            for rho in [j["rho"] for j in src_jobs["jobs"]
                        if j["method"] == "B2_histogram"]:
                fname = f"B2f_hybrid_full__rho{rho}.npz"
                np.savez_compressed(jdir / fname, prewarm=pwm, keepalive=kam)
                job_meta.append({"method": "B2f_hybrid_full", "rho": rho,
                                 "file": fname})
            with open(jdir / "jobs.json", "w") as fp:
                json.dump({"split": split, "seeds": src_jobs["seeds"],
                           "cold_init": src_jobs["cold_init"],
                           "jobs": job_meta}, fp)
            print(f"  exported {len(job_meta)} jobs to {jdir}")
        print("Run after chain completes: PYTHONPATH=<des env> python3.13 "
              "scripts/des_runner_fast.py --jobs results/runs/des_jobs_2019_v1 "
              "--out results/runs/revision_v1_hybridfull_2019.json")


if __name__ == "__main__":
    main()
