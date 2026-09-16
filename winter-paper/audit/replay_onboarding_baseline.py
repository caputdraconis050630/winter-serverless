"""Add B2f to the existing four-hour cohorts without modifying source archives.

Run with PYTHONPATH=/data/260715/site-packages python3.13.
The control replay must reproduce the archived EWMA endpoint before accepting
the additional baseline. No model training or threshold selection occurs.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent / "serverless-fewshot"
sys.path.insert(0, str(ROOT))

from scripts.revision_r22_ewma_alpha_frontier import cohort_arrays
from scripts.revision_v1_hybridfull import policy_b2_full
from scripts.phase6_des import COLD_INIT, decisions_from_rates, rates_ewma
from src.decision.newsvendor import newsvendor_quantile
from src.sim.des import rolling_csr, simulate_function


def replay(counts, dm, ds, pw, ka):
    n, horizon = counts.shape
    cold = np.zeros(n)
    cold60 = np.zeros(n)
    wm = np.zeros(n)
    roll_c = np.zeros(horizon)
    roll_n = np.zeros(horizon)
    for f in range(n):
        for seed in (0, 1, 2):
            result = simulate_function(
                counts[f], pw[f], ka[f], float(dm[f]), float(ds[f]),
                seed=seed * 7919 + f, cold_mu=COLD_INIT["mu"],
                cold_sigma=COLD_INIT["sigma"], track_rolling=True,
            )
            cold[f] += result["roll_cold"].sum() / 3
            cold60[f] += result["roll_cold"][:60].sum() / 3
            wm[f] += result["idle_mem_gb_s"] / 3
            roll_c += result["roll_cold"]
            roll_n += result["roll_total"]
    return {
        "overall_csr": float(cold.sum() / counts.sum()),
        "func_cold_all": cold.tolist(), "func_cold60": cold60.tolist(),
        "func_wm": wm.tolist(), "wm_total": float(wm.sum()),
        "rolling_csr": rolling_csr(roll_c, roll_n, window=15).tolist(),
    }


def main():
    out = {"description": "B2f onboarding addition, 2026-09-07",
           "seeds": [0, 1, 2], "window_minutes": 240, "cohorts": {}}
    archives = {"azure2019": "revision_a4_crosstrace.json",
                "huawei": "revision_h1_huawei_cohort.json"}
    for cohort, filename in archives.items():
        pts, counts, dm, ds = cohort_arrays(cohort)
        reference = json.loads((ROOT / "results/runs" / filename).read_text())
        rates = rates_ewma(counts)
        pw, ka = decisions_from_rates(rates, newsvendor_quantile(10.0))
        control = replay(counts, dm, ds, pw, ka)
        old = reference["by_rho"]["10.0"]["B4a_ewma"]
        np.testing.assert_allclose(control["func_cold_all"], old["func_cold_all"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(control["wm_total"], old["wm_total"], atol=1e-7, rtol=0)
        print(cohort, "EWMA control matches", flush=True)
        pw, ka = policy_b2_full(counts)
        baseline = replay(counts, dm, ds, pw, ka)
        out["cohorts"][cohort] = {
            "source_reference": filename, "points": pts,
            "invocations_per_function": counts.sum(axis=1).tolist(),
            "control_max_cold_residual": float(np.max(np.abs(np.asarray(control["func_cold_all"]) - old["func_cold_all"]))),
            "control_wm_residual": control["wm_total"] - old["wm_total"],
            "B2f_hybrid_full": baseline,
        }
        (HERE / "onboarding_b2f.json").write_text(json.dumps(out, indent=2))
        print(cohort, "B2f CSR", baseline["overall_csr"] * 100,
              "WM/1k", baseline["wm_total"] / counts.sum() * 1000, flush=True)


if __name__ == "__main__":
    main()
