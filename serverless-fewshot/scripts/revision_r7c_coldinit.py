# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R7c: per-runtime cold-init sensitivity (pre-registered in
revision_r7c_PREREG.md). Zero simulator changes — only jobs.json cold_init
varies across byte-identical decision trees.

  --prepare  (torch env)  export the 2021 S3 mini-tree (A5/EWMA/Oracle,
             rho=10) and stage the four cold_init variants for S3 + h_mixed
  --report   (any env, after des_runner_fast on each variant)

Runner stage (numba env), once per variant:
  for v in pooled python-ml node-api-real java-svc; do
    PYTHONPATH=/data/260715/site-packages-des:. python3.13 \
      scripts/des_runner_fast.py --jobs results/runs/des_jobs_r7c/$v \
      --out results/runs/revision_r7c_des_$v.json
  done
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RUNS = PROJECT_ROOT / "results" / "runs"
JOBS_ROOT = RUNS / "des_jobs_r7c"
RHO = 10.0
SEEDS = [0, 1, 2]
VARIANTS = {
    "pooled": {"mu": 0.25, "sigma": 0.31},
    "python-ml": {"mu": 0.2683, "sigma": 0.3176},
    "node-api-real": {"mu": 0.1828, "sigma": 0.3221},
    "java-svc": {"mu": 0.2961, "sigma": 0.2919},
}
S3_METHODS = ["A5_full_system", "B4a_ewma", "Oracle"]
HW_METHODS = ["A5_full_system", "B4a_ewma", "Oracle", "B1_fixed_keepalive",
              "B2_histogram"]
OUT = RUNS / "revision_r7c_coldinit.json"


def prepare():
    import pandas as pd
    import torch
    from scripts.phase6_des import (
        decisions_from_rates, rates_ewma, rates_a5, rates_oracle)
    from src.decision.newsvendor import newsvendor_quantile
    from src.meta.trainer import ANILMetaTrainer
    from src.models.heads import N_QUANTILES
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    P21 = PROJECT_ROOT / "data" / "processed"
    features = np.load(P21 / "features.npy")
    counts_all = np.load(P21 / "counts.npy")
    splits = np.load(P21 / "splits.npz")
    dur_df = pd.read_csv(P21 / "duration_stats.csv")
    test_idx = splits["s3_test"]
    t0 = int(splits.get("s3_test_t_start", [10080])[0])
    counts = counts_all[test_idx][:, t0:].astype(np.float32)
    feats = features[test_idx][:, t0:, :]
    dm = np.nan_to_num(np.array(
        [dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
         for fi in test_idx]), nan=1.0)
    ds = np.nan_to_num(np.array(
        [dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
         for fi in test_idx]), nan=0.5)

    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1, device=device)
    ckpt = RUNS / "best_anil_ridge_s1_s0.pt"     # clean 2021 (989b0ef6)
    trainer.load(ckpt)
    print(f"checkpoint: {ckpt}", flush=True)

    tau = newsvendor_quantile(RHO)
    rate_mats = {"B4a_ewma": rates_ewma(counts),
                 "A5_full_system": rates_a5(counts, feats, trainer, device),
                 "Oracle": rates_oracle(counts)}
    s3_stage = JOBS_ROOT / "_s3_decisions"
    s3_stage.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(s3_stage / "shared.npz",
                        counts=counts.astype(np.int64),
                        dur_means=dm, dur_stds=ds)
    for m, rates in rate_mats.items():
        pw, ka = decisions_from_rates(rates, tau)
        np.savez_compressed(s3_stage / f"{m}__rho{RHO}.npz",
                            prewarm=pw, keepalive=ka)
    print("S3 decisions exported", flush=True)

    hw_src = RUNS / "des_jobs_huawei" / "h_mixed"
    for vname, ci in VARIANTS.items():
        for pool, methods, src in [("S3_2021", S3_METHODS, s3_stage),
                                   ("h_mixed", HW_METHODS, hw_src)]:
            jdir = JOBS_ROOT / vname / pool
            jdir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src / "shared.npz", jdir / "shared.npz")
            job_meta = []
            for m in methods:
                fname = f"{m}__rho{RHO}.npz"
                shutil.copy(src / fname, jdir / fname)
                job_meta.append({"method": m, "rho": RHO, "file": fname})
            json.dump({"split": pool, "seeds": SEEDS, "cold_init": ci,
                       "jobs": job_meta}, open(jdir / "jobs.json", "w"))
        print(f"staged variant {vname} {ci}", flush=True)
    json.dump({"prereg": "revision_r7c_PREREG.md", "variants": VARIANTS,
               "rho": RHO, "seeds": SEEDS, "checkpoint": str(ckpt)},
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT} (prepare stage)")


def report():
    out = json.load(open(OUT))
    per = {}
    for vname in VARIANTS:
        res = json.load(open(RUNS / f"revision_r7c_des_{vname}.json"))
        rows = res["results"] if isinstance(res, dict) and "results" in res else res
        for r in rows:
            per.setdefault((r["split"], r["method"]), {}).setdefault(
                vname, []).append(r["csr"])
    tables = {}
    for (pool, method), by_v in sorted(per.items()):
        csr = {v: float(np.mean(x) * 100) for v, x in by_v.items()}
        rts = [csr[v] for v in ("python-ml", "node-api-real", "java-svc")]
        tables[f"{pool}|{method}"] = {
            "csr_pct_by_variant": csr,
            "spread_pp": float(max(rts) - min(rts)),
            "mixture_iid_uniform_pct": float(np.mean(rts)),
            "mixture_minus_pooled_pp": float(np.mean(rts) - csr["pooled"]),
        }
        print(f"{pool:8s} {method:18s} pooled={csr['pooled']:.3f}% "
              f"spread={tables[f'{pool}|{method}']['spread_pp']:.4f}pp "
              f"mix-pooled={tables[f'{pool}|{method}']['mixture_minus_pooled_pp']:+.4f}pp",
              flush=True)
    # reproduction gate: pooled h_mixed must match the archived h2 campaign
    ref = json.load(open(RUNS / "revision_h2_huawei_des_h_mixed.json"))
    ref_rows = ref["results"] if isinstance(ref, dict) and "results" in ref else ref
    gates = {}
    for m in HW_METHODS:
        refv = np.mean([r["csr"] for r in ref_rows
                        if r["method"] == m and r["cost_ratio"] == RHO])
        ours = tables[f"h_mixed|{m}"]["csr_pct_by_variant"]["pooled"] / 100
        gates[m] = {"ours_pct": ours * 100, "ref_pct": float(refv * 100),
                    "delta_pp": float(abs(ours - refv) * 100)}
    out["tables"] = tables
    out["pooled_reproduction_gate"] = gates
    json.dump(out, open(OUT, "w"), indent=1)
    print("gates:", json.dumps({k: round(v["delta_pp"], 6) for k, v in gates.items()}))
    print(f"wrote {OUT} (report stage)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.report:
        report()
    else:
        ap.error("pass --prepare or --report")
