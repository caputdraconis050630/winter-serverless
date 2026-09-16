#!/usr/bin/env python3
"""WP-H3 statistics: Huawei steady-state DES -> regime-map cell verdicts.

Reads results/runs/revision_h2_huawei_des.json (produced by des_runner_fast.py
over des_jobs_huawei) and reports, per pool x rho x method:
  - pool CSR and warm memory per 1k invocations (pools are exhaustive, so no
    inclusion weights are needed -- every active function in the bucket is
    simulated)
  - learned-vs-EWMA: function-level paired bootstrap CI of the CSR difference
    and the CI-based TOST verdict at the registered +/-0.10 pp margin
    (equivalence iff the 95% CI lies strictly inside the margin)
  - Oracle-vs-fixed-keep-alive gap: the cell-3 statistic ("nothing helps")

Output: results/runs/revision_h2_huawei_stats.json
        results/tables/T_huawei_des.csv
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sstats

PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

# One results file per pool (des_runner_fast.py is launched once per split);
# they are concatenated here into the single row list the aggregation expects.
RESULT_GLOB = "revision_h2_huawei_des_*.json"
TOST_MARGIN_PP = 0.10          # registered equivalence margin
N_BOOT = 5000
LEARNED = "A5_full_system"
EWMA = "B4a_ewma"
ORACLE = "Oracle"
KEEPALIVE = "B1_fixed_keepalive"


def csr(cold, total):
    return float(cold.sum() / max(total.sum(), 1e-9))


def paired_boot(cold_a, cold_b, total, seed=11):
    """95% CI for CSR_a - CSR_b in percentage points (function resampling)."""
    rng = np.random.default_rng(seed)
    n = len(total)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = csr(cold_a[idx], total[idx]) - csr(cold_b[idx], total[idx])
    d = diffs * 100.0
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def wilcoxon_fn(cold_a, cold_b, total):
    fa = cold_a / np.maximum(total, 1)
    fb = cold_b / np.maximum(total, 1)
    mask = ~np.isclose(fa, fb)
    if mask.sum() <= 10:
        return 1.0, float((fa < fb).mean())
    try:
        p = float(sstats.wilcoxon(fa[mask], fb[mask]).pvalue)
    except ValueError:
        p = 1.0
    return p, float((fa < fb).mean())


def main():
    paths = sorted(RUNS_DIR.glob(RESULT_GLOB))
    if not paths:
        raise SystemExit(f"no results matching {RESULT_GLOB} in {RUNS_DIR}")
    rows = []
    for p in paths:
        part = json.load(open(p))
        rows.extend(part)
        print(f"  {p.name}: {len(part)} runs")
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["split"], r["cost_ratio"], r["method"])].append(r)

    agg = {k: {"cold": np.mean([r["func_cold"] for r in rs], axis=0),
               "total": np.mean([r["func_total"] for r in rs], axis=0),
               "wm": np.mean([r["func_wm"] for r in rs], axis=0),
               "csr_seeds": [r["csr"] for r in rs]}
           for k, rs in grouped.items()}

    out = {"tost_margin_pp": TOST_MARGIN_PP, "sources": [p.name for p in paths],
           "cells": {}, "verdicts": {}}
    csv_lines = ["pool,rho,method,csr_pct,csr_seed_std_pct,wm_per_1k,"
                 "diff_vs_ewma_pp,ci_lo_pp,ci_hi_pp,wilcoxon_p_vs_ewma,"
                 "win_rate_vs_ewma"]

    pools = sorted({k[0] for k in agg})
    for pool in pools:
        rhos = sorted({k[1] for k in agg if k[0] == pool})
        for rho in rhos:
            ref = agg.get((pool, rho, EWMA))
            for m in sorted({k[2] for k in agg if k[0] == pool and k[1] == rho}):
                a = agg[(pool, rho, m)]
                c = csr(a["cold"], a["total"]) * 100.0
                wm1k = float(a["wm"].sum()
                             / max(a["total"].sum() / 1000.0, 1e-9))
                entry = {"csr_pct": c,
                         "csr_seed_std_pct": float(np.std(a["csr_seeds"]) * 100),
                         "wm_per_1k": wm1k}
                dif = lo = hi = wp = wr = ""
                if ref is not None and m != EWMA:
                    d = c - csr(ref["cold"], ref["total"]) * 100.0
                    lo_v, hi_v = paired_boot(a["cold"], ref["cold"], a["total"])
                    wp_v, wr_v = wilcoxon_fn(a["cold"], ref["cold"], a["total"])
                    entry.update({"diff_vs_ewma_pp": d,
                                  "ci95_pp": [lo_v, hi_v],
                                  "wilcoxon_p_vs_ewma": wp_v,
                                  "win_rate_vs_ewma": wr_v})
                    dif, lo, hi = f"{d:.4f}", f"{lo_v:.4f}", f"{hi_v:.4f}"
                    wp, wr = f"{wp_v:.2e}", f"{wr_v:.3f}"
                out["cells"][f"{pool}|{rho}|{m}"] = entry
                csv_lines.append(
                    f"{pool},{rho},{m},{c:.4f},{entry['csr_seed_std_pct']:.4f},"
                    f"{wm1k:.1f},{dif},{lo},{hi},{wp},{wr}")

            # ---- cell verdicts ----
            v = {}
            if (pool, rho, LEARNED) in agg and ref is not None:
                e = out["cells"][f"{pool}|{rho}|{LEARNED}"]
                lo_v, hi_v = e["ci95_pp"]
                v["learned_vs_ewma_pp"] = e["diff_vs_ewma_pp"]
                v["learned_vs_ewma_ci_pp"] = [lo_v, hi_v]
                v["tost_equivalent"] = bool(abs(lo_v) < TOST_MARGIN_PP
                                            and abs(hi_v) < TOST_MARGIN_PP)
            if (pool, rho, ORACLE) in agg and (pool, rho, KEEPALIVE) in agg:
                o = agg[(pool, rho, ORACLE)]
                k = agg[(pool, rho, KEEPALIVE)]
                gap = (csr(k["cold"], k["total"]) - csr(o["cold"], o["total"])) * 100
                v["oracle_below_keepalive_pp"] = float(gap)
            out["verdicts"][f"{pool}|{rho}"] = v

    with open(RUNS_DIR / "revision_h2_huawei_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / "T_huawei_des.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    print("Pool CSR (%) by rho:")
    for pool in pools:
        print(f"  --- {pool} ---")
        for rho in sorted({k[1] for k in agg if k[0] == pool}):
            line = "  ".join(
                f"{m}={out['cells'][f'{pool}|{rho}|{m}']['csr_pct']:.4f}"
                for m in sorted({k[2] for k in agg
                                 if k[0] == pool and k[1] == rho}))
            v = out["verdicts"][f"{pool}|{rho}"]
            print(f"    rho={rho:<6}: {line}")
            if v:
                print(f"        learned-EWMA {v.get('learned_vs_ewma_pp', float('nan')):+.4f} pp "
                      f"CI {v.get('learned_vs_ewma_ci_pp')} "
                      f"TOST_equiv={v.get('tost_equivalent')} | "
                      f"keepalive-Oracle {v.get('oracle_below_keepalive_pp', float('nan')):+.4f} pp")
    print("Saved revision_h2_huawei_stats.json + T_huawei_des.csv")


if __name__ == "__main__":
    main()
