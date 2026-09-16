#!/usr/bin/env python3
"""Validate the stratified-subsampling estimator for DES population metrics
against the FULL Azure-2021 DES results (results_azure2021/runs/
sim_results_des.json), before applying the same policy to Azure-2019.

Estimator: functions stratified by log10(total trace invocations) into
quartile strata (observable pre-simulation); equal allocation per stratum;
population CSR estimated as a weighted ratio-of-sums:
    CSR_hat = sum_s w_s * sum_{i in sample_s} cold_i
            / sum_s w_s * sum_{i in sample_s} total_i,   w_s = N_s / n_s
(WM analogous). Compared against simple random sampling (SRS).

Monte-Carlo over draws; reports per-method relative error and pairwise
method-ordering preservation (same function sample shared by all methods,
as it would be in a real subsampled DES run).

Output: results/runs/des_subsample_validation.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
FULL = PROJECT_ROOT / "results_azure2021" / "runs" / "sim_results_des.json"
OUT = PROJECT_ROOT / "results" / "runs" / "des_subsample_validation.json"

SPLIT = "S3"          # largest 2021 split (172 functions)
N_DRAWS = 2000
SAMPLE_SIZES = (20, 40, 80)
N_STRATA = 4
RNG = np.random.default_rng(20260716)


def ratio_estimate(cold, total, idx, weights):
    return (weights * cold[idx]).sum() / max((weights * total[idx]).sum(), 1e-9)


def main():
    rows = json.load(open(FULL))
    rows = [r for r in rows if r["split"] == SPLIT]
    # group: (method, cost_ratio) -> averaged-over-seed per-function arrays
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["cost_ratio"])].append(r)

    # Strata from trace totals (identical across methods/seeds; take any row)
    any_row = rows[0]
    totals = np.array(any_row["func_total"], dtype=float)
    n_funcs = len(totals)
    order = np.argsort(np.log10(totals + 1))
    strata = np.zeros(n_funcs, dtype=int)
    for s, chunk in enumerate(np.array_split(order, N_STRATA)):
        strata[chunk] = s
    strata_sizes = np.bincount(strata, minlength=N_STRATA)

    # Per (method, rho): population CSR/WM and per-function cold/total/wm
    pop = {}
    arrays = {}
    for (method, rho), rs in groups.items():
        cold = np.mean([r["func_cold"] for r in rs], axis=0)
        tot = np.mean([r["func_total"] for r in rs], axis=0)
        wm = np.mean([r["func_wm"] for r in rs], axis=0)
        pop[(method, rho)] = (cold.sum() / tot.sum(), wm.sum())
        arrays[(method, rho)] = (cold, tot, wm)

    rhos = sorted({rho for (_, rho) in pop})
    methods = sorted({m for (m, _) in pop})
    print(f"{SPLIT}: {n_funcs} funcs, methods={methods}, rhos={rhos}")

    pps_p = totals / totals.sum()

    def draw_indices(n, scheme):
        if scheme == "stratified":
            per = max(1, n // N_STRATA)
            idx, w = [], []
            for s in range(N_STRATA):
                pool = np.where(strata == s)[0]
                take = min(per, len(pool))
                pick = RNG.choice(pool, take, replace=False)
                idx.append(pick)
                w.append(np.full(take, strata_sizes[s] / take))
            return np.concatenate(idx), np.concatenate(w)
        if scheme == "pps":
            # Hansen-Hurwitz: with-replacement, p ~ size; weight = 1/(n*p)
            pick = RNG.choice(n_funcs, n, replace=True, p=pps_p)
            return pick, 1.0 / (n * pps_p[pick])
        if scheme == "topk_srs":
            # take-all for the heaviest 20% of the sample budget by size,
            # SRS over the remainder
            k = max(1, n // 5)
            top = np.argsort(totals)[::-1][:k]
            rest_pool = np.setdiff1d(np.arange(n_funcs), top)
            pick = RNG.choice(rest_pool, n - k, replace=False)
            idx = np.concatenate([top, pick])
            w = np.concatenate([np.ones(k),
                                np.full(n - k, len(rest_pool) / (n - k))])
            return idx, w
        pick = RNG.choice(n_funcs, n, replace=False)
        return pick, np.full(n, n_funcs / n)

    out = {"split": SPLIT, "n_funcs": int(n_funcs), "n_draws": N_DRAWS,
           "results": {}}
    for scheme in ("stratified", "srs", "pps", "topk_srs"):
        for n in SAMPLE_SIZES:
            rel_errs = defaultdict(list)
            order_ok = defaultdict(int)
            order_n = defaultdict(int)
            for _ in range(N_DRAWS):
                idx, w = draw_indices(n, scheme)
                est = {}
                for key, (cold, tot, wm) in arrays.items():
                    csr_hat = ratio_estimate(cold, tot, idx, w)
                    est[key] = csr_hat
                    csr_pop = pop[key][0]
                    if csr_pop > 0:
                        rel_errs[key].append(abs(csr_hat - csr_pop) / csr_pop)
                # pairwise ordering preservation at each rho
                for rho in rhos:
                    for i, m1 in enumerate(methods):
                        for m2 in methods[i + 1:]:
                            if (m1, rho) not in pop or (m2, rho) not in pop:
                                continue
                            true_order = pop[(m1, rho)][0] < pop[(m2, rho)][0]
                            est_order = est[(m1, rho)] < est[(m2, rho)]
                            order_n[rho] += 1
                            order_ok[rho] += int(true_order == est_order)
            med_rel = float(np.median([e for v in rel_errs.values() for e in v]))
            p90_rel = float(np.percentile(
                [e for v in rel_errs.values() for e in v], 90))
            ord_frac = {str(r): order_ok[r] / order_n[r] for r in rhos}
            key = f"{scheme}_n{n}"
            out["results"][key] = {
                "median_rel_err_csr": med_rel,
                "p90_rel_err_csr": p90_rel,
                "ordering_preserved_frac_by_rho": ord_frac,
            }
            print(f"{key}: median rel err={med_rel*100:.1f}% "
                  f"p90={p90_rel*100:.1f}% ordering={ord_frac}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
