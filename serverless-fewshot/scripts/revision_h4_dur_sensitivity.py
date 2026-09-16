# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP-H4 control: are the Huawei cell verdicts an artefact of the imputed
execution durations?

Huawei Public 2023 ships request counts only, so durations were imputed from
the Huawei Private 2023 `function_delay_minute` pool (i.i.d. mode, selected by
the pre-registered Spearman rule).  Durations never enter the decision layer --
prewarm/keep-alive matrices are functions of counts and features alone -- so
this control re-simulates the *identical* decisions under three duration models
and checks that the ordering of methods (the cell verdict) is invariant:

  huawei_private   the primary model (Huawei Private empirical pool)
  azure2021        Azure 2021 measured durations, same i.i.d. assignment rule
  const1s          degenerate control: every function exactly 1 s, zero variance

Scope (pre-stated): 300-function subsample of h_mixed, rho in {1, 10}, seed 0.

  export:  PYTHONPATH=/data/260715/site-packages:. python3.13 \
               scripts/revision_h4_dur_sensitivity.py --export
  run:     PYTHONPATH=/data/260715/site-packages-des:. python3.13 \
               scripts/des_runner_fast.py \
               --jobs results/runs/split_jobs_hdur_<variant> \
               --out results/runs/revision_h4_dur_<variant>.json
  report:  PYTHONPATH=/data/260715/site-packages:. python3.13 \
               scripts/revision_h4_dur_sensitivity.py --report
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

SRC_JOBS = RUNS_DIR / "split_jobs_huawei_h_mixed" / "h_mixed"
SPLIT_NAME = "h_mixed_dur300"
N_SUB = 300
SUB_SEED = 42
DUR_SEED = 42            # same registered seed as phase1c_huawei.assign_durations
RHOS = [1.0, 10.0]
SEEDS = [0]
VARIANTS = ["huawei_private", "azure2021", "const1s"]

LEARNED = "A5_full_system"
EWMA = "B4a_ewma"
ORACLE = "Oracle"


def azure_duration_pool():
    """Azure 2021 measured per-function durations (minutes)."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "duration_stats.csv")
    dm = np.nan_to_num(df["dur_mean"].to_numpy(), nan=1.0)
    ds = np.nan_to_num(df["dur_std"].to_numpy(), nan=0.5)
    ok = np.isfinite(dm) & (dm > 0)
    return dm[ok], ds[ok]


def export():
    shared = np.load(SRC_JOBS / "shared.npz")
    meta = json.load(open(SRC_JOBS / "jobs.json"))
    counts = shared["counts"]
    n = counts.shape[0]

    rng = np.random.default_rng(SUB_SEED)
    rows = np.sort(rng.choice(n, min(N_SUB, n), replace=False))
    counts_s = counts[rows]
    print(f"Subsample: {len(rows)}/{n} h_mixed functions, "
          f"{int(counts_s.sum()):,} invocations")

    # --- the three duration models, all on the same 300 functions ---
    dur_rng = np.random.default_rng(DUR_SEED)
    az_m, az_s = azure_duration_pool()
    az_idx = dur_rng.integers(0, len(az_m), size=len(rows))   # i.i.d., as primary
    durations = {
        "huawei_private": (shared["dur_means"][rows], shared["dur_stds"][rows]),
        "azure2021": (az_m[az_idx], az_s[az_idx]),
        "const1s": (np.ones(len(rows)), np.zeros(len(rows))),
    }
    for v, (dm, ds) in durations.items():
        # dur_means feed src/sim/des_fast.py, which works in seconds
        # (ka_sec = keepalive * 60; durations are compared against it raw)
        print(f"  {v:<16s} mean {dm.mean():.4f} s  median {np.median(dm):.4f}"
              f"  std-of-means {dm.std():.4f}")

    jobs_keep = [j for j in meta["jobs"] if j["rho"] in RHOS]
    for v, (dm, ds) in durations.items():
        jdir = RUNS_DIR / f"split_jobs_hdur_{v}" / SPLIT_NAME
        if jdir.exists():
            shutil.rmtree(jdir)
        jdir.mkdir(parents=True)
        np.savez_compressed(jdir / "shared.npz", counts=counts_s,
                            dur_means=dm, dur_stds=ds)
        for job in jobs_keep:
            jz = np.load(SRC_JOBS / job["file"])
            np.savez_compressed(jdir / job["file"],
                                prewarm=jz["prewarm"][rows],
                                keepalive=jz["keepalive"][rows])
        json.dump({"split": SPLIT_NAME, "seeds": SEEDS,
                   "cold_init": meta["cold_init"], "jobs": jobs_keep,
                   "duration_model": v, "rows_in_h_mixed": rows.tolist()},
                  open(jdir / "jobs.json", "w"))
        print(f"  exported {len(jobs_keep)} jobs -> {jdir}")


def report():
    out = {"n_functions": N_SUB, "rhos": RHOS, "seeds": SEEDS, "variants": {}}
    lines = ["duration_model,rho,method,csr_pct,wm_per_1k"]
    verdict_rows = []
    for v in VARIANTS:
        p = RUNS_DIR / f"revision_h4_dur_{v}.json"
        if not p.exists():
            print(f"  missing {p.name} -- skipped")
            continue
        rows = json.load(open(p))
        cell = {}
        for r in rows:
            cell[(r["cost_ratio"], r["method"])] = r
            lines.append(f"{v},{r['cost_ratio']},{r['method']},"
                         f"{r['csr']*100:.4f},{r['wm_per_1k_inv']:.1f}")
        out["variants"][v] = {
            f"{rho}|{m}": {"csr_pct": cell[(rho, m)]["csr"] * 100,
                           "wm_per_1k": cell[(rho, m)]["wm_per_1k_inv"]}
            for (rho, m) in cell}
        for rho in RHOS:
            if (rho, LEARNED) in cell and (rho, EWMA) in cell:
                d = (cell[(rho, LEARNED)]["csr"] - cell[(rho, EWMA)]["csr"]) * 100
                orc = cell[(rho, ORACLE)]["csr"] * 100 if (rho, ORACLE) in cell else float("nan")
                verdict_rows.append((v, rho, d, orc))

    out["learned_minus_ewma_pp"] = {f"{v}|{rho}": d for v, rho, d, _ in verdict_rows}
    out["oracle_csr_pct"] = {f"{v}|{rho}": o for v, rho, _, o in verdict_rows}
    signs = {rho: {np.sign(round(d, 3)) for v, r, d, _ in verdict_rows if r == rho}
             for rho in RHOS}
    out["verdict_invariant"] = {str(rho): (len(s) == 1) for rho, s in signs.items()}

    json.dump(out, open(RUNS_DIR / "revision_h4_dur_sensitivity.json", "w"),
              indent=2, default=float)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "T_huawei_dur_sensitivity.csv").write_text("\n".join(lines) + "\n")

    print("\nlearned - EWMA (pp), by duration model:")
    for v, rho, d, orc in verdict_rows:
        print(f"  {v:<16s} rho={rho:<6}: {d:+.4f} pp   (Oracle CSR {orc:.4f}%)")
    print(f"sign-invariant per rho: {out['verdict_invariant']}")
    print("Saved revision_h4_dur_sensitivity.json + T_huawei_dur_sensitivity.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.export:
        export()
    if a.report:
        report()
    if not (a.export or a.report):
        ap.error("pass --export or --report")
