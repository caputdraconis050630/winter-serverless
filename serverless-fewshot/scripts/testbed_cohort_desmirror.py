# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""E6: DES mirror of the E5 cohort -- predicted vs measured, arm by arm.

The reviewer question behind the testbed section is not "is the testbed
large?" but "does the simulator reflect reality?".  Answering it needs the
simulator run on *exactly* the same function-segments and the same policy
arms as the hardware measurement, so that every number has a matched pair.

This script therefore:
  1. reads the cohort actually measured (results/runs/testbed_cohort.json),
  2. rebuilds the same four policies with the same code path E5 used
     (testbed_cohort.build_schedules -> the paper's decisions_from_rates),
  3. runs the event-level DES over those segments with the registered seeds,
  4. reports the pairs: per-arm CSR (predicted, measured), the sign of every
     pairwise arm difference in both worlds, and the per-function rank
     correlation.

Two cold-start models are simulated: the registered COLD_INIT prior, and a
lognormal refit to the cold latencies actually observed in E5, so that
"the simulator's magnitudes are off" can be separated from "the simulator's
cold-start distribution was mis-specified".

Claim scope is fixed in advance: with 10 functions there is no power to
resolve tenths of a percentage point, so only the ORDERING of arms and the
ORDER OF MAGNITUDE of their gaps are claimed.

Writes results/runs/testbed_cohort_desmirror.json.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase63_onboarding_drift import DES_SEEDS, PROCESSED_DIR, RUNS_DIR  # noqa: E402
from scripts.phase6_des import COLD_INIT                        # noqa: E402
from src.sim.des import simulate_function                       # noqa: E402
from src.models.heads import N_QUANTILES                        # noqa: E402
from src.meta.trainer import ANILMetaTrainer                    # noqa: E402
from src.models.prototypes import PrototypeManager              # noqa: E402
import scripts.testbed_cohort as tc                             # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_des(seg, des_dec, arms, dur_df, func_ids, cold_mu, cold_sigma):
    F_n, T = seg.shape
    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] for fi in func_ids]),
                       nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] for fi in func_ids]),
                         nan=0.5)
    out = {}
    for arm in arms:
        pw, ka = des_dec[arm]
        cold = np.zeros(F_n)
        total = np.zeros(F_n)
        for f in range(F_n):
            for seed in DES_SEEDS:
                r = simulate_function(seg[f], pw[f], ka[f],
                                      float(dm[f]), float(dstd[f]),
                                      seed=seed * 7919 + f,
                                      cold_mu=cold_mu, cold_sigma=cold_sigma,
                                      track_rolling=True)
                cold[f] += r["roll_cold"].sum()
                total[f] += r["roll_total"].sum()
        cold /= len(DES_SEEDS)
        total /= len(DES_SEEDS)
        out[arm] = {"csr": float(cold.sum() / max(total.sum(), 1e-9)),
                    "func_cold": cold.tolist(),
                    "func_total": total.tolist()}
    return out


def main():
    meas_path = RUNS_DIR / "testbed_cohort.json"
    if not meas_path.exists():
        raise SystemExit("run scripts/testbed_cohort.py first")
    meas = json.load(open(meas_path))
    func_ids = meas["func_ids"]
    onboard_t = meas["onboard_t"]
    pts = list(zip(func_ids, onboard_t))
    arms = [a for a in tc.ARMS if a in meas["arms"]]
    tc.SEG_MIN = meas["constants"]["seg_min"]

    counts = np.load(PROCESSED_DIR / "counts.npy")
    features = np.load(PROCESSED_DIR / "features.npy")
    splits = np.load(PROCESSED_DIR / "splits.npz")
    import pandas as pd
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")

    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1,
                              device=DEVICE)
    trainer.load(RUNS_DIR / "best_anil_ridge_s1_s0.pt")
    pm = PrototypeManager(n_clusters=tc.N_CLUSTERS, device=DEVICE)
    feats_t = torch.from_numpy(features).float()
    cnts_t = torch.from_numpy(counts).float()
    emb = pm.compute_embeddings(trainer.body, feats_t, splits["s1_train"])
    labels = pm.fit_clusters(emb)
    pm.compute_prototype_heads(trainer.body, trainer.head, feats_t, cnts_t,
                               splits["s1_train"], labels,
                               n_quantiles=N_QUANTILES, n_horizons=1)

    seg, sched, des_dec = tc.build_schedules(counts, features, pts, trainer, pm)
    assert seg.tolist() == meas["segment_counts"], \
        "segment mismatch: mirror is not simulating the measured cohort"
    # Simulate the counts the load generator actually issued, not the raw
    # trace counts: the per-tick cap applies to the hardware run, so feeding
    # the DES uncapped counts would compare two different workloads.
    issued = np.array(meas["issued_counts"], dtype=np.int64)

    # cold-start model 2: refit to the latencies E5 actually observed.
    # Per-invocation latencies are summarised per function, so the lognormal
    # median is taken from the measured per-function p50s.
    med = [pf["lat_p50"] for arm in arms
           for pf in meas["arms"][arm]["per_function"]
           if pf.get("lat_p50")]
    refit = None
    if med:
        m = float(np.median(med))
        refit = {"mu": float(np.log(max(m, 1e-3))), "sigma": COLD_INIT["sigma"],
                 "source": "median of measured per-function p50 latency"}

    models = {"registered": {"mu": COLD_INIT["mu"], "sigma": COLD_INIT["sigma"]}}
    if refit:
        models["refit_to_testbed"] = {"mu": refit["mu"], "sigma": refit["sigma"]}

    out = {"config": "E6 DES mirror of the E5 cohort",
           "n_functions": len(pts), "func_ids": func_ids,
           "arms": arms, "cold_models": models,
           "claim_scope": "ordering of arms and order of magnitude of gaps "
                          "only; n=10 cannot resolve sub-pp differences",
           "predicted": {}, "measured": {}, "comparison": {}}

    for name, mdl in models.items():
        out["predicted"][name] = run_des(issued, des_dec, arms, dur_df,
                                         func_ids, mdl["mu"], mdl["sigma"])

    for arm in arms:
        a = meas["arms"][arm]
        out["measured"][arm] = {
            "csr_event": a["csr_event"], "csr_latency": a["csr_latency"],
            "invocations": a["invocations"], "pod_seconds": a["pod_seconds"],
            "func_csr_event": [pf["csr_event"] for pf in a["per_function"]]}

    for name in models:
        pred = out["predicted"][name]
        rows = []
        for x, y in combinations(arms, 2):
            dp = pred[x]["csr"] - pred[y]["csr"]
            dm_ = out["measured"][x]["csr_event"] - out["measured"][y]["csr_event"]
            rows.append({"pair": f"{x}-{y}",
                         "delta_predicted_pp": 100 * dp,
                         "delta_measured_pp": 100 * dm_,
                         "sign_agrees": bool(np.sign(dp) == np.sign(dm_)),
                         "magnitude_ratio": (dm_ / dp) if abs(dp) > 1e-12 else None})
        agree = sum(r["sign_agrees"] for r in rows)
        # per-function Spearman on the learned arm, predicted vs measured
        from scipy import stats as sps
        rho_rows = {}
        for arm in arms:
            p = np.array(pred[arm]["func_cold"], float)
            m2 = np.array(out["measured"][arm]["func_csr_event"], float)
            if p.std() > 0 and m2.std() > 0:
                r_, pv = sps.spearmanr(p, m2)
                rho_rows[arm] = {"spearman_r": float(r_), "p": float(pv)}
        out["comparison"][name] = {
            "pairwise": rows,
            "sign_agreement": f"{agree}/{len(rows)}",
            "sign_agreement_frac": agree / max(1, len(rows)),
            "per_function_spearman": rho_rows,
            "arm_order_predicted": sorted(arms, key=lambda a: pred[a]["csr"]),
            "arm_order_measured": sorted(
                arms, key=lambda a: out["measured"][a]["csr_event"]),
        }

    path = RUNS_DIR / "testbed_cohort_desmirror.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0, "corrupt write (sdb guard)"
    json.load(open(path))

    for name in models:
        c = out["comparison"][name]
        print(f"\n[{name}] sign agreement {c['sign_agreement']}")
        print(f"  predicted order: {c['arm_order_predicted']}")
        print(f"  measured  order: {c['arm_order_measured']}")
        for r in c["pairwise"]:
            print(f"   {r['pair']:26s} pred {r['delta_predicted_pp']:+7.3f}pp "
                  f"meas {r['delta_measured_pp']:+7.3f}pp "
                  f"{'OK' if r['sign_agrees'] else 'MISMATCH'}")
    print(f"\nSaved + verified {path}")


if __name__ == "__main__":
    main()
