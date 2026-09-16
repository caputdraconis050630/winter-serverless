# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""M2-B: age-qualifier sensitivity sweep on the 2021 steady-state protocol.

Pre-registration: results/runs/revision_m2_PREREG.md.

Machinery is revision_e7_agegate.py's steady_2021() unchanged; the only
difference is that the age qualifier A is swept over {180, 360, 720, 1440,
2880} instead of being fixed at the registered 720. Everything else --
checkpoint, prototypes, splits, rho grid, the 10 DES seeds, the shared
decision layer -- is identical, so the A=720 arm must reproduce E7.

Measured before running (pre-registration section I): the share of
function-minutes whose routing changes anywhere in the swept range is 7.1%
(S1), 5.3% (S2), 20.6% (S3), so S3 is the primary split.

Run: PYTHONPATH=/data/260715/site-packages:. python3.13 -u \
     scripts/revision_m2b_agesweep_2021.py
Output: results/runs/revision_m2b_agesweep.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision_e7_agegate import age_matrix  # noqa: E402

GATE_THRESHOLD = 100
AGE_GRID = [180, 360, 720, 1440, 2880]
STEADY_RHOS = [1.0, 10.0, 100.0]
STEADY_SEEDS = list(range(10))


def main():
    from scripts.phase6_des import (rates_a5, rates_ewma, run_config,
                                    decisions_from_rates, COLD_INIT)
    from scripts.revision_a2_des import rates_proto_prefix
    from scripts.phase63_onboarding_drift import build_prototypes
    from src.decision.newsvendor import newsvendor_quantile
    from src.models.heads import N_QUANTILES
    from src.meta.trainer import ANILMetaTrainer
    from concurrent.futures import ProcessPoolExecutor

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RUNS_DIR = PROJECT_ROOT / "results" / "runs"
    PROC = PROJECT_ROOT / "data" / "processed"
    features = np.load(PROC / "features.npy")
    counts_all = np.load(PROC / "counts.npy")
    splits_data = np.load(PROC / "splits.npz")
    dur_df = pd.read_csv(PROC / "duration_stats.csv")

    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE)
    trainer.load(RUNS_DIR / "best_anil_ridge_s1_s0.pt")
    pm = build_prototypes(trainer, features, counts_all,
                          splits_data["s1_train"])

    out = {"prereg": "revision_m2_PREREG.md", "age_grid": AGE_GRID,
           "gate_threshold": GATE_THRESHOLD, "seeds": STEADY_SEEDS,
           "rhos": STEADY_RHOS, "results": [], "routing": {}}

    for split in ["S3", "S1", "S2"]:          # S3 first: primary split
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

        print(f"### {split}: composing rates...", flush=True)
        t = time.time()
        a5 = rates_a5(counts, feats, trainer, DEVICE)
        ew = rates_ewma(counts)
        proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
        age = age_matrix(counts)
        zero = cum_prev == 0
        mid = (cum_prev > 0) & (cum_prev < GATE_THRESHOLD)
        conv = cum_prev >= GATE_THRESHOLD
        print(f"    rates in {time.time()-t:.0f}s", flush=True)

        out["routing"][split] = {"frac_proto": float(zero.mean()),
                                 "frac_mid_ewma": float(mid.mean()),
                                 "frac_conv": float(conv.mean())}
        jobs = []
        for A in AGE_GRID:
            young = conv & (age < A)
            out["routing"][split][f"frac_conv_young_learned_A{A}"] = \
                float(young.mean())
            aq = np.where(zero, proto,
                 np.where(mid, ew,
                 np.where(young, a5, ew))).astype(np.float32)
            for rho in STEADY_RHOS:
                pw, ka = decisions_from_rates(aq, newsvendor_quantile(rho))
                jobs.append((f"A5_gated_aq_{A}", rho, STEADY_SEEDS, counts,
                             pw, ka, dm, ds, COLD_INIT, split))
        print(f"    routing {out['routing'][split]}", flush=True)

        t = time.time()
        with ProcessPoolExecutor(max_workers=8) as ex:
            for res_list in ex.map(run_config, jobs):
                out["results"].extend(res_list)
                r = res_list[0]
                print(f"    {r['method']} rho={r['cost_ratio']:6.1f} "
                      f"CSR={np.mean([x['csr'] for x in res_list]):.4f} "
                      f"WM={np.mean([x['wm_per_1k_inv'] for x in res_list]):.0f}",
                      flush=True)
        print(f"    DES {split} in {time.time()-t:.0f}s", flush=True)

        path = RUNS_DIR / "revision_m2b_agesweep.json"
        with open(path, "w") as f:
            json.dump(out, f, default=float)
        print(f"  checkpointed after {split} -> {path}", flush=True)

    json.load(open(RUNS_DIR / "revision_m2b_agesweep.json"))
    print("Saved + verified", flush=True)


if __name__ == "__main__":
    main()
