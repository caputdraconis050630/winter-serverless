# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision B3: drift severity sweep.

Extends the published drift grid (scale x2/x0.5, phase +2h, splice) with
higher severities to locate the crossover where (i) triggered refit beats a
frozen head and (ii) the learned path beats EWMA:

    scale x5, x10        (load surges)
    scale x0.2, x0.1     (load collapses, mirror severities)
    phase +6h, +12h      (schedule shifts)

Arms, DES machinery, rhos {1,10}, seeds, and refit semantics are identical to
phase63_onboarding_drift.drift_experiment (A5_refit = conformal-trigger +
periodic refit; A5_frozen = no refits after onset). Severity values fixed a
priori. Output: results/runs/revision_b3_severity.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from scripts.phase63_onboarding_drift import (  # noqa: E402
    L, RHOS_EVAL, DES_SEEDS, DRIFT_WINDOW, DEVICE, PROCESSED_DIR, RUNS_DIR,
    a5_online_rates,
)
from scripts.phase6_des import decisions_from_rates, COLD_INIT  # noqa: E402
from src.sim.des import simulate_function, rolling_csr  # noqa: E402
from src.sim.simulator import compute_adaptation_lag  # noqa: E402
from src.data.features import features_from_counts  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.drift.detector import inject_synthetic_drift  # noqa: E402
from src.drift.triggers import ConformalMonitor  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402

SEVERITY_GRID = [("scale", 5.0), ("scale", 10.0),
                 ("scale", 0.2), ("scale", 0.1),
                 ("phase", 360), ("phase", 720)]


def main():
    features = np.load(PROCESSED_DIR / "features.npy")
    counts = np.load(PROCESSED_DIR / "counts.npy")
    splits = np.load(PROCESSED_DIR / "splits.npz")
    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE,
    )
    trainer.load(RUNS_DIR / "best_anil_ridge_s1_s0.pt")
    test_idx, train_idx = splits["s1_test"], splits["s1_train"]

    T = counts.shape[1]
    t_d = T // 2
    seg_lo, seg_hi = t_d - DRIFT_WINDOW, t_d + DRIFT_WINDOW
    rel_td = DRIFT_WINDOW
    Tseg = seg_hi - seg_lo
    active = [fi for fi in test_idx
              if counts[fi, seg_lo:t_d].sum() >= 100
              and counts[fi, t_d:seg_hi].sum() >= 100][:12]
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    print(f"{len(active)} functions, severities {SEVERITY_GRID}")

    out = {"severity_curves": {}, "grid": [list(map(str, g)) for g in SEVERITY_GRID]}
    for dtype, param in SEVERITY_GRID:
        key = f"{dtype}({param})"
        print(f"--- {key} ---")
        agg = {}
        for fi in active:
            drifted, _ = inject_synthetic_drift(counts, fi, t_d, dtype, param)
            seg_counts = drifted[fi, seg_lo:seg_hi].astype(np.int64)
            feats_full = features_from_counts(
                drifted[fi], t_offset=0, app_corate=features[fi, :, 10])
            feats_seg = feats_full[seg_lo:seg_hi]
            dm = float(np.nan_to_num(dur_df.iloc[fi]["dur_mean"], nan=1.0))
            dstd = float(np.nan_to_num(dur_df.iloc[fi]["dur_std"], nan=0.5))

            variants = {}
            r_refit, _ = a5_online_rates(trainer, feats_seg, seg_counts, seg_lo,
                                         trigger=ConformalMonitor())
            variants["A5_refit"] = r_refit
            r_frozen, _ = a5_online_rates(trainer, feats_seg, seg_counts, seg_lo,
                                          frozen_after=rel_td - 1)
            variants["A5_frozen"] = r_frozen
            ew = np.zeros(Tseg, dtype=np.float32)
            st = 0.0
            for t in range(Tseg):
                ew[t] = np.expm1(max(st, 0))
                st = 0.1 * np.log1p(seg_counts[t]) + 0.9 * st
            variants["B4a_ewma"] = ew
            variants["Oracle"] = seg_counts.astype(np.float32)
            act = (seg_counts > 0).astype(np.int32)
            conv = np.convolve(act, np.ones(10, dtype=np.int32))[:Tseg]
            pw_b1 = np.zeros(Tseg, dtype=np.int32)
            pw_b1[1:] = (conv[:-1] > 0).astype(np.int32)
            ka_b1 = np.full(Tseg, 10.0, np.float32)

            for rho in RHOS_EVAL:
                tau = newsvendor_quantile(rho)
                dec = {n: decisions_from_rates(r[None, :], tau)
                       for n, r in variants.items()}
                dec["B1_fixed_keepalive"] = (pw_b1[None, :], ka_b1[None, :])
                for name, (pw, ka) in dec.items():
                    rc_sum = np.zeros(Tseg); rn_sum = np.zeros(Tseg)
                    for seed in DES_SEEDS:
                        r = simulate_function(seg_counts, pw[0], ka[0], dm, dstd,
                                              seed=seed * 104729 + fi,
                                              cold_mu=COLD_INIT["mu"],
                                              cold_sigma=COLD_INIT["sigma"],
                                              track_rolling=True)
                        rc_sum += r["roll_cold"]; rn_sum += r["roll_total"]
                    a = agg.setdefault((rho, name),
                                       [np.zeros(Tseg), np.zeros(Tseg)])
                    a[0] += rc_sum; a[1] += rn_sum

        curves = {}
        for (rho, name), (rc, rn) in agg.items():
            c = rolling_csr(rc, rn, window=15)
            pairs = [(t, v) for t, v in enumerate(c) if not np.isnan(v)]
            al = compute_adaptation_lag(pairs, event_tick=rel_td)
            curves.setdefault(str(rho), {})[name] = {
                "rolling_csr": [None if np.isnan(x) else float(x) for x in c],
                "al": float(al) if al is not None else None,
                "csr_pre": float(np.nansum(rc[:rel_td]) / max(1, np.nansum(rn[:rel_td]))),
                "csr_post60": float(np.nansum(rc[rel_td:rel_td + 60]) /
                                    max(1, np.nansum(rn[rel_td:rel_td + 60]))),
                "csr_post_all": float(np.nansum(rc[rel_td:]) /
                                      max(1, np.nansum(rn[rel_td:]))),
            }
        for rho in curves:
            row = curves[rho]
            print(f"  rho={rho}: " + " ".join(
                f"{n}:post60={v['csr_post60']*100:.2f}%" for n, v in row.items()))
        out["severity_curves"][key] = curves

    path = RUNS_DIR / "revision_b3_severity.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")


if __name__ == "__main__":
    main()
