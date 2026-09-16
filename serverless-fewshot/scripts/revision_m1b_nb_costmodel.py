# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""M1(b): does the always-on net-negative verdict survive a non-Poisson band?

Reviewer M1(b) / M5. R12 showed that replacing the decision layer's Poisson
predictive band with NB2 moves CSR by at most +/-0.06 pp but raises warm
memory 5-37%. The paper's negative headline -- always-on learning is
net-negative -- is an accounting in GB.s in which memory is one of the two
terms, so a 37% move in that term is not automatically harmless: ordering
robustness is not the same as accounting robustness.

This is arithmetic over archived runs, not a new simulation. It reuses the
registered cost model of revision_e3_costsens.py:

    net = ( kappa*rho*cold_ewma + wm_ewma )
        - ( kappa*rho*cold_learned + wm_learned + state )
    net > 0  =>  learning pays

Coverage limit, stated up front: R12 archived S3 at rho=10 only, so this
recomputation covers that cell alone. It cannot speak for "every split at
every rho"; it can only show whether the one cell that was measured under NB2
changes sign.

Run: PYTHONPATH=/data/260715/site-packages:. python3.13 -u \
     scripts/revision_m1b_nb_costmodel.py
Output: results/runs/revision_m1b_nb_costmodel.json
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
RUNS = PROJECT_ROOT / "results" / "runs"

KAPPAS = [5.0, 15.0, 50.0]          # registered 10x range
RHO = 10.0
SPLIT = "S3"
STATE_GBS = 8e-6 * 10080 * 60       # 8 KB/function, S3 segment (e3 constant)
LEARNED = "A5_full_system"
EWMA = "B4a_ewma"
BANDS = ["poisson", "nb_phi2", "nb_phi5", "nb_phihat"]


def rows_of(results, method, split=SPLIT, rho=RHO):
    return {r["seed"]: r for r in results
            if r["method"] == method and r["split"] == split
            and abs(r["cost_ratio"] - rho) < 1e-9}


def main():
    ref = json.load(open(RUNS / "sim_results_des_revision.json"))["results"]
    r12 = json.load(open(RUNS / "revision_r12_dispersion.json"))["results"]

    arms = {}
    arms[("poisson", LEARNED)] = rows_of(ref, LEARNED)
    arms[("poisson", EWMA)] = rows_of(ref, EWMA)
    for band in BANDS[1:]:
        arms[(band, LEARNED)] = rows_of(r12, f"{LEARNED}__{band}")
        arms[(band, EWMA)] = rows_of(r12, f"{EWMA}__{band}")

    for k, v in arms.items():
        assert v, f"no archived rows for {k}"
    print("seeds per arm:", {f"{b}/{m}": len(v) for (b, m), v in arms.items()},
          flush=True)

    out = {"source": ["sim_results_des_revision.json",
                      "revision_r12_dispersion.json"],
           "cost_model": "net = (k*rho*cold_e + wm_e) - "
                         "(k*rho*cold_a + wm_a + state); net>0 => learning pays",
           "split": SPLIT, "rho": RHO, "kappas": KAPPAS,
           "state_gbs": STATE_GBS,
           "coverage_limit": "R12 archived S3 at rho=10 only",
           "bands": {}}

    for band in BANDS:
        la, ea = arms[(band, LEARNED)], arms[(band, EWMA)]
        seeds = sorted(set(la) & set(ea))
        cold_a = np.array([la[s]["cold_starts"] for s in seeds], float)
        wm_a = np.array([la[s]["wm_total_gb_s"] for s in seeds], float)
        cold_e = np.array([ea[s]["cold_starts"] for s in seeds], float)
        wm_e = np.array([ea[s]["wm_total_gb_s"] for s in seeds], float)
        csr_a = np.array([la[s]["csr"] for s in seeds], float)
        csr_e = np.array([ea[s]["csr"] for s in seeds], float)

        rec = {"n_seeds": len(seeds),
               "csr_learned_pct": float(csr_a.mean() * 100),
               "csr_ewma_pct": float(csr_e.mean() * 100),
               "d_csr_pp": float((csr_a.mean() - csr_e.mean()) * 100),
               "wm_learned": float(wm_a.mean()),
               "wm_ewma": float(wm_e.mean()),
               "by_kappa": {}}
        for kappa in KAPPAS:
            net = (kappa * RHO * cold_e + wm_e) - \
                  (kappa * RHO * cold_a + wm_a + STATE_GBS)
            rec["by_kappa"][str(kappa)] = {
                "net_gbs_mean": float(net.mean()),
                "net_gbs_min": float(net.min()),
                "net_gbs_max": float(net.max()),
                "sign": "pays" if net.mean() > 0 else "does_not_pay",
                "sign_consistent_across_seeds":
                    bool((net > 0).all() or (net < 0).all()),
                "frac_seeds_pay": float((net > 0).mean())}
        out["bands"][band] = rec
        k15 = rec["by_kappa"]["15.0"]
        print(f"{band:10s} dCSR {rec['d_csr_pp']:+.4f}pp  "
              f"wm_learned/wm_ewma {rec['wm_learned']/rec['wm_ewma']:.3f}  "
              f"net(k=15) {k15['net_gbs_mean']:+.3e} -> {k15['sign']}",
              flush=True)

    ref_sign = out["bands"]["poisson"]["by_kappa"]["15.0"]["sign"]
    flips = [b for b in BANDS
             if out["bands"][b]["by_kappa"]["15.0"]["sign"] != ref_sign]
    all_signs = {(b, k) for b in BANDS for k in map(str, KAPPAS)
                 if out["bands"][b]["by_kappa"][k]["sign"] != ref_sign}
    out["verdict"] = {
        "poisson_sign": ref_sign,
        "bands_flipping_at_kappa15": flips,
        "cells_flipping_any_kappa": sorted(f"{b}/k={k}" for b, k in all_signs),
        "conclusion": ("net-negative survives the NB2 band at every kappa"
                       if not all_signs else
                       "SIGN CHANGES under a non-Poisson band -- see cells")}
    print(json.dumps(out["verdict"], indent=1), flush=True)

    path = RUNS / "revision_m1b_nb_costmodel.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    json.load(open(path))
    print(f"Saved + verified {path}", flush=True)


if __name__ == "__main__":
    main()
