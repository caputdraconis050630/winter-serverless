#!/usr/bin/env python3
"""Azure 2021 TOST archive (2026-08-02): formalizes the equivalence
computations quoted in the main text for the 10-seed 2021 steady-state
campaign, and states the analysis unit explicitly.

Unit (disclosed in the supplementary implementation notes): the Azure
equivalence claims are *across-seed paired* TOST on aggregate CSR
(n = 10 DES seeds, two one-sided t-tests against the registered
+/-0.10 pp margin). The function-level paired bootstrap CI (the unit
used for the Huawei pools, cf. revision_h2_huawei_stats.py) is archived
alongside for reference; at +/-0.10 pp that unit cannot pass for any
arm because per-function CSR differences are an order of magnitude
larger than the margin, which is calibrated to aggregate CSR.

Arms: A5_full_system vs B4a_ewma, A5_gated_v3 vs A5_full_system
(sim_results_des_revision.json) and A5_gated_v4 vs B4a_ewma
(revision_e1_v4_2021.json), all splits, all cost ratios with 10
seeds on both arms.

Output: results/runs/revision_tost_azure.json
        results/tables/T_tost_azure.csv
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sstats

PROJECT_ROOT = Path(__file__).parent.parent
RUNS = PROJECT_ROOT / "results" / "runs"
TABLES = PROJECT_ROOT / "results" / "tables"

TOST_MARGIN_PP = 0.10
N_BOOT = 5000
BOOT_SEED = 11
EWMA = "B4a_ewma"


def load_rows(name):
    j = json.load(open(RUNS / name))
    return j["results"] if isinstance(j, dict) and "results" in j else j


def csr(cold, total):
    return float(np.sum(cold) / max(np.sum(total), 1e-9))


def paired_boot(cold_a, cold_b, total, rng):
    n = len(total)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        k = rng.integers(0, n, n)
        diffs[i] = csr(cold_a[k], total[k]) - csr(cold_b[k], total[k])
    d = diffs * 100.0
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def seed_tost(diffs_pp):
    n = len(diffs_pp)
    m = float(np.mean(diffs_pp))
    se = float(np.std(diffs_pp, ddof=1)) / np.sqrt(n)
    t_lo = (m + TOST_MARGIN_PP) / se
    t_hi = (m - TOST_MARGIN_PP) / se
    p = max(1.0 - sstats.t.cdf(t_lo, n - 1), sstats.t.cdf(t_hi, n - 1))
    return m, float(p)


def main():
    comparisons = [
        ("A5_full_system", EWMA, "sim_results_des_revision.json"),
        ("A5_gated_v4", EWMA, "revision_e1_v4_2021.json"),
        ("A5_gated_v3", "A5_full_system", "sim_results_des_revision.json"),
    ]
    grouped = defaultdict(dict)
    for name in {"sim_results_des_revision.json", "revision_e1_v4_2021.json"}:
        for r in load_rows(name):
            grouped[(r["split"], r["cost_ratio"], r["method"])][r["seed"]] = r

    out = {"tost_margin_pp": TOST_MARGIN_PP,
           "unit": "across-seed paired TOST on aggregate CSR (n=10)",
           "cells": {}}
    csv = ["arm,reference,split,rho,mean_diff_pp,tost_p,tost_equivalent,"
           "func_boot_ci_lo_pp,func_boot_ci_hi_pp,func_boot_inside_margin"]

    for arm, ref, _src in comparisons:
        keys = sorted(k for k in grouped
                      if k[2] == arm and (k[0], k[1], ref) in grouped)
        for split, rho, _ in keys:
            a, e = grouped[(split, rho, arm)], grouped[(split, rho, ref)]
            seeds = sorted(set(a) & set(e))
            if len(seeds) < 10:
                continue
            diffs = np.array([(a[s]["csr"] - e[s]["csr"]) * 100 for s in seeds])
            m, p = seed_tost(diffs)
            cold_a = np.mean([a[s]["func_cold"] for s in seeds], axis=0)
            cold_e = np.mean([e[s]["func_cold"] for s in seeds], axis=0)
            tot = np.mean([a[s]["func_total"] for s in seeds], axis=0)
            rng = np.random.default_rng(BOOT_SEED)
            lo, hi = paired_boot(cold_a, cold_e, tot, rng)
            cell = {"n_seeds": len(seeds), "mean_diff_pp": m, "tost_p": p,
                    "tost_equivalent": bool(p < 0.05),
                    "func_boot_ci_pp": [lo, hi],
                    "func_boot_inside_margin":
                        bool(lo > -TOST_MARGIN_PP and hi < TOST_MARGIN_PP)}
            out["cells"][f"{arm}|vs|{ref}|{split}|{rho}"] = cell
            csv.append(f"{arm},{ref},{split},{rho},{m:.5f},{p:.3e},"
                       f"{cell['tost_equivalent']},{lo:.4f},{hi:.4f},"
                       f"{cell['func_boot_inside_margin']}")
            print(f"{arm:15s} vs {ref:15s} {split} rho={rho:<6} diff={m:+.4f}pp "
                  f"p={p:.1e} equiv={cell['tost_equivalent']} "
                  f"funcCI=[{lo:+.3f},{hi:+.3f}]")

    with open(RUNS / "revision_tost_azure.json", "w") as f:
        json.dump(out, f, indent=1)
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "T_tost_azure.csv").write_text("\n".join(csv) + "\n")
    n_eq = sum(c["tost_equivalent"] for c in out["cells"].values())
    print(f"\n{len(out['cells'])} cells, seed-level TOST equivalent: {n_eq}")


if __name__ == "__main__":
    main()
