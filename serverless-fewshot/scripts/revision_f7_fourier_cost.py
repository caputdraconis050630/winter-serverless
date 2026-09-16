# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision F7: B3's cost--quality accounting, in the protocols of Table 4.

Two measurements, each replicating the protocol the existing rows were
produced with, so the B3 row is comparable rather than merely adjacent:

  CRPS (S2 hold-out)  the EWMA protocol of phase8_ablations.py 8.3 verbatim
                      -- 32 query times in the second half of the trace, a
                      Poisson band around the point forecast, pinball loss in
                      log1p space, averaged per function. Only the forecast
                      changes: B3's causal harmonic extrapolation instead of
                      the EWMA state.
  State and adapt     the scalability protocol of phase8_ablations.py 8.4 --
                      persistent bytes per function and batched refit time
                      per function at N=10,000.

Output: results/runs/revision_f7_fourier_cost.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import (  # noqa: E402
    FOURIER_HARMONICS, FOURIER_MIN_HIST, FOURIER_REFIT, FOURIER_WINDOW,
)
from src.models.heads import N_QUANTILES  # noqa: E402

PROC = PROJECT_ROOT / "data" / "processed"
RUNS = PROJECT_ROOT / "results" / "runs"
L = 60
N_SCALE = 10000


def b3_point(hist, n_harmonics=FOURIER_HARMONICS, window=FOURIER_WINDOW):
    """One causal B3 forecast from a history vector (the deployed rule)."""
    seg = hist[-window:]
    n = len(seg)
    if n < FOURIER_MIN_HIST:
        return 0.0
    spec = np.fft.rfft(seg.astype(np.float64))
    keep = np.argsort(np.abs(spec))[-(n_harmonics + 1):]
    trunc = np.zeros_like(spec)
    trunc[keep] = spec[keep]
    recon = np.maximum(np.fft.irfft(trunc, n=n), 0.0)
    return float(recon[0])          # periodic extension one step past the window


def crps_s2():
    counts = np.load(PROC / "counts.npy")
    splits = np.load(PROC / "splits.npz")
    test_funcs = splits["s2_test"]
    T = counts.shape[1]
    q = np.linspace(0.05, 0.95, N_QUANTILES)
    scores = []
    for fi in test_funcs:
        query_times = np.linspace(T // 2 + 1, T - 2, 32, dtype=int)
        func_crps = []
        for t in query_times:
            pred_mean = max(b3_point(counts[fi, max(0, t - FOURIER_WINDOW):t]),
                            0.01)
            pred_q = np.array([sp_stats.poisson.ppf(x, pred_mean) for x in q])
            actual = np.log1p(counts[fi, min(t + 1, T - 1)])
            err = actual - np.log1p(pred_q)
            pb = np.maximum(q * err, (q - 1) * err)
            func_crps.append(2 * pb.mean())
        scores.append(float(np.mean(func_crps)))
    return scores, len(test_funcs)


def scalability():
    rng = np.random.default_rng(0)
    hist = rng.poisson(0.5, size=(N_SCALE, FOURIER_WINDOW)).astype(np.float64)
    for _ in range(2):                       # warm the FFT plan
        np.fft.rfft(hist[:64], axis=1)
    t0 = time.perf_counter()
    spec = np.fft.rfft(hist, axis=1)
    keep = np.argsort(np.abs(spec), axis=1)[:, -(FOURIER_HARMONICS + 1):]
    trunc = np.zeros_like(spec)
    np.put_along_axis(trunc, keep, np.take_along_axis(spec, keep, axis=1),
                      axis=1)
    np.fft.irfft(trunc, n=FOURIER_WINDOW, axis=1)
    total_ms = (time.perf_counter() - t0) * 1e3
    return {"N": N_SCALE, "refit_ms_total": total_ms,
            "refit_ms_per_func": total_ms / N_SCALE,
            "refit_us_per_func": total_ms * 1e3 / N_SCALE,
            "amortized_us_per_func_per_tick":
                total_ms * 1e3 / N_SCALE / FOURIER_REFIT,
            "per_func_bytes": FOURIER_WINDOW * 4,
            "per_func_kb": FOURIER_WINDOW * 4 / 1024,
            "state_note": "trailing 1,440-minute count ring buffer, uint32"}


def main():
    sc = scalability()
    print("scalability:", {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in sc.items()})
    scores, n = crps_s2()
    print(f"CRPS S2 (B3, n={n}): mean {np.mean(scores):.4f} "
          f"median {np.median(scores):.4f}")

    ab = json.load(open(RUNS / "ablation_results.json"))["statistics"]
    out = {"arm": "B3_fourier",
           "crps_protocol": "phase8_ablations.py 8.3 (EWMA path), S2 hold-out",
           "crps_mean": float(np.mean(scores)),
           "crps_scores": scores, "n_eval": n,
           "comparators": {"ours": ab["ours_vs_ewma"]["ours_ci"]["mean"],
                           "ewma": ab["ours_vs_ewma"]["ewma_ci"]["mean"]},
           "scalability_protocol": "phase8_ablations.py 8.4",
           "scalability": sc}
    path = RUNS / "revision_f7_fourier_cost.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")
    print(f"  B3 {sc['per_func_kb']:.1f} KB state, "
          f"{sc['refit_us_per_func']:.2f} us/refit, "
          f"CRPS {np.mean(scores):.4f} "
          f"(ours {out['comparators']['ours']:.4f}, "
          f"EWMA {out['comparators']['ewma']:.4f})")


if __name__ == "__main__":
    main()
