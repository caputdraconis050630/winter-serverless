# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R13: integrated age-qualified gate with live conformal triggers on 2021 S3
(pre-registered in revision_r13_PREREG.md).

Torch env: PYTHONPATH=/data/260715/site-packages:. python3.13 scripts/revision_r13_gate_trig.py
Output: results/runs/revision_r13_gate_trig.json
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
    rates_ewma, rates_a5, run_config, decisions_from_rates, COLD_INIT,
)
from scripts.revision_a2_des import rates_proto_prefix  # noqa: E402
from scripts.phase63_onboarding_drift import build_prototypes  # noqa: E402
from src.drift.triggers import ConformalMonitor  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402

torch.set_num_threads(1)
RUNS = PROJECT_ROOT / "results" / "runs"
AGE_MIN, GATE_THRESHOLD = 720, 100
REFIT_WINDOW = 240
RHOS = [1.0, 10.0]
SEEDS = list(range(10))
OUT = RUNS / "revision_r13_gate_trig.json"


def main():
    P21 = PROJECT_ROOT / "data" / "processed"
    features = np.load(P21 / "features.npy")
    counts_all = np.load(P21 / "counts.npy")
    splits = np.load(P21 / "splits.npz")
    dur_df = pd.read_csv(P21 / "duration_stats.csv")
    idx = splits["s3_test"]
    t0 = int(splits["s3_test_t_start"][0])
    counts = counts_all[idx][:, t0:].astype(np.float32)
    feats = features[idx][:, t0:, :]
    N, T = counts.shape
    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df)
                                 else 1.0 for fi in idx]), nan=1.0)
    ds = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] if fi < len(dur_df)
                                 else 0.5 for fi in idx]), nan=0.5)

    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1, device="cuda")
    ckpt = RUNS / "best_anil_ridge_s1_s0.pt"
    trainer.load(ckpt)
    print(f"checkpoint: {ckpt}", flush=True)
    pm = build_prototypes(trainer, features, counts_all, splits["s1_train"])

    t_s = time.time()
    ew = rates_ewma(counts)
    a5 = rates_a5(counts, feats, trainer, "cuda")
    proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
    print(f"rates done {time.time()-t_s:.0f}s", flush=True)

    # E7 composition (first-arrival age, 2021 rule)
    first_arr = np.where(counts.sum(axis=1) > 0, (counts > 0).argmax(axis=1), T)
    age = np.arange(T)[None, :] - first_arr[:, None]
    zero_h = cum_prev == 0
    mid = (cum_prev > 0) & (cum_prev < GATE_THRESHOLD)
    conv_young = (cum_prev >= GATE_THRESHOLD) & (age < AGE_MIN)
    base = np.where(zero_h, proto, np.where(mid, ew,
                    np.where(conv_young, a5, ew))).astype(np.float32)

    # live triggers on the served-rate residual stream
    resid = np.abs(np.log1p(counts) - np.log1p(np.maximum(base, 0.0)))
    fire_mask = np.zeros((N, T), dtype=bool)
    n_fires = 0
    for f in range(N):
        mon = ConformalMonitor()
        t = 0
        while t < T:
            fired = mon.update(float(resid[f, t]))
            if fired:
                n_fires += 1
                end = min(T, t + 1 + REFIT_WINDOW)
                fire_mask[f, t + 1:end] = True
                mon.reset()
                t = end - 1
            t += 1
    gated_trig = np.where(fire_mask, a5, base).astype(np.float32)
    fire_stats = {
        "n_fires": int(n_fires),
        "fires_per_1k_func_min": float(n_fires / (N * T) * 1000),
        "share_ticks_in_refit_window": float(fire_mask.mean()),
    }
    print("fire stats:", fire_stats, flush=True)

    jobs = []
    for rho in RHOS:
        tau = newsvendor_quantile(rho)
        pw, ka = decisions_from_rates(gated_trig, tau)
        jobs.append(("A5_gated_aq_trig", rho, SEEDS, counts, pw, ka,
                     dm, ds, COLD_INIT, "S3"))
    results = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for res in ex.map(run_config, jobs):
            for r in res:
                for k in ("func_csr", "func_cold", "func_total", "func_wm"):
                    r.pop(k, None)
            results.extend(res)
            r0 = res[0]
            print(f"  rho={r0['cost_ratio']}: csr={np.mean([x['csr'] for x in res])*100:.4f}% "
                  f"wm={np.mean([x['wm_per_1k_inv'] for x in res]):.0f}", flush=True)

    e7 = json.load(open(RUNS / "revision_e7_agegate.json"))
    ref = {}
    for rho in RHOS:
        rows = [r for r in e7["steady_2021"]["results"]
                if r["split"] == "S3" and r["cost_ratio"] == rho]
        ref[str(rho)] = {"csr": float(np.mean([r["csr"] for r in rows])),
                         "wm": float(np.mean([r["wm_per_1k_inv"] for r in rows]))}
    out = {"prereg": "revision_r13_PREREG.md", "checkpoint": str(ckpt),
           "constants": {"age_min": AGE_MIN, "gate_threshold": GATE_THRESHOLD,
                         "refit_window": REFIT_WINDOW,
                         "trigger": "conformal alpha=0.1 delta=0.15 m=3"},
           "fire_stats": fire_stats, "results": results,
           "trigger_free_aq_ref": ref}
    for rho in RHOS:
        ours = np.mean([r["csr"] for r in results if r["cost_ratio"] == rho])
        print(f"delta vs trigger-free aq rho={rho}: "
              f"{(ours - ref[str(rho)]['csr'])*100:+.4f}pp")
    json.dump(out, open(OUT, "w"), default=float)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
