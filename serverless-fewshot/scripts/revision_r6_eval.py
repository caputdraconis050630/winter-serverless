# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R6 stage 2: evaluate the drift arms on MINED natural collapse events
(pre-registered in revision_r6_PREREG.md). The harness is
revision_b3_severity.py with injection removed — real counts, real
features, drift time = mined change point.

Torch env:
  PYTHONPATH=/data/260715/site-packages:. python3.13 scripts/revision_r6_eval.py
Output: results/runs/revision_r6_natural_collapse.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from scripts.phase63_onboarding_drift import (  # noqa: E402
    RHOS_EVAL, DES_SEEDS, DRIFT_WINDOW, DEVICE,
    a5_online_rates,
)
from scripts.phase6_des import decisions_from_rates, COLD_INIT  # noqa: E402
from src.sim.des import simulate_function, rolling_csr  # noqa: E402
from src.sim.simulator import compute_adaptation_lag  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.drift.triggers import ConformalMonitor  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402

torch.set_num_threads(1)
RUNS = PROJECT_ROOT / "results" / "runs"
CAP = 40                      # per-trace cap (see PREREG AMENDMENT 1)
INV_CAP = 200_000             # window-invocation eligibility cap (~p90)
SEL_SEED = 260715
W = DRIFT_WINDOW              # 240 min each side


def select(events):
    """AMENDMENT 1: random CAP among events with window invocations
    <= INV_CAP; 2021 (16 events) passes through unmodified."""
    if len(events) <= 16:
        return events
    elig = [e for e in events
            if e["pre_inv_240"] + e["post_inv_240"] <= INV_CAP]
    rng = np.random.default_rng(SEL_SEED)
    idx = rng.choice(len(elig), min(CAP, len(elig)), replace=False)
    return [elig[i] for i in sorted(idx)]

TRACES = {
    "azure2021": {
        "data": PROJECT_ROOT / "data" / "processed",
        "ckpt": RUNS / "best_anil_ridge_s1_s0.pt",
        "mmap": False},
    "azure2019": {
        "data": PROJECT_ROOT / "data" / "processed_2019",
        "ckpt": PROJECT_ROOT / "results_azure2021" / "runs" / "best_anil_ridge_s2_s0.pt",
        "mmap": True},
    "huawei": {
        "data": PROJECT_ROOT / "data" / "processed_huawei",
        "ckpt": PROJECT_ROOT / "results_azure2021" / "runs" / "best_anil_ridge_s2_s0.pt",
        "mmap": True},
}


def main():
    catalog = json.load(open(RUNS / "revision_r6_catalog.json"))
    out = {"prereg": "revision_r6_PREREG.md", "cap_per_trace": CAP,
           "des_seeds": DES_SEEDS, "rhos": RHOS_EVAL, "traces": {}}

    for trace, cfg in TRACES.items():
        events = select(catalog["traces"][trace]["events"])
        if not events:
            out["traces"][trace] = {"n_events": 0}
            continue
        mode = "r" if cfg["mmap"] else None
        counts = np.load(cfg["data"] / "counts.npy", mmap_mode=mode)
        features = np.load(cfg["data"] / "features.npy", mmap_mode=mode)
        dur_df = pd.read_csv(cfg["data"] / "duration_stats.csv")
        trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                                  in_features=features.shape[2], embedding_dim=64,
                                  n_quantiles=N_QUANTILES, n_horizons=1,
                                  device=DEVICE)
        trainer.load(cfg["ckpt"])
        print(f"### {trace}: {len(events)} events, ckpt={cfg['ckpt'].name}",
              flush=True)

        agg = {}
        t_start = time.time()
        for k, ev in enumerate(events):
            fi, cp = ev["row"], ev["cp_minute"]
            seg_lo, seg_hi = cp - W, cp + W
            seg_counts = np.asarray(counts[fi, seg_lo:seg_hi]).astype(np.int64)
            feats_seg = np.asarray(features[fi, seg_lo:seg_hi, :],
                                   dtype=np.float32)
            dm = float(np.nan_to_num(dur_df.iloc[fi]["dur_mean"], nan=1.0)) \
                if fi < len(dur_df) else 1.0
            dstd = float(np.nan_to_num(dur_df.iloc[fi]["dur_std"], nan=0.5)) \
                if fi < len(dur_df) else 0.5
            Tseg = seg_hi - seg_lo
            rel_td = W

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
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(events)} events "
                      f"({time.time()-t_start:.0f}s)", flush=True)

        curves = {}
        for (rho, name), (rc, rn) in agg.items():
            c = rolling_csr(rc, rn, window=15)
            pairs = [(t, v) for t, v in enumerate(c) if not np.isnan(v)]
            al = compute_adaptation_lag(pairs, event_tick=W)
            curves.setdefault(str(rho), {})[name] = {
                "rolling_csr": [None if np.isnan(x) else float(x) for x in c],
                "al": float(al) if al is not None else None,
                "csr_pre": float(np.nansum(rc[:W]) / max(1, np.nansum(rn[:W]))),
                "csr_post60": float(np.nansum(rc[W:W + 60]) /
                                    max(1, np.nansum(rn[W:W + 60]))),
                "csr_post_all": float(np.nansum(rc[W:]) /
                                      max(1, np.nansum(rn[W:]))),
            }
        for rho in curves:
            print(f"  rho={rho}: " + " ".join(
                f"{n}:post60={v['csr_post60']*100:.2f}%"
                for n, v in sorted(curves[rho].items())), flush=True)
        out["traces"][trace] = {
            "n_events": len(events), "checkpoint": str(cfg["ckpt"]),
            "median_magnitude": float(np.median(
                [e["magnitude"] for e in events])),
            "curves": curves}

    path = RUNS / "revision_r6_natural_collapse.json"
    json.dump(out, open(path, "w"))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
