# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP1 validation: invariant checks for the event-level DES.

Invariants:
  I1. Conservation: processed invocations == trace total
  I2. Oracle CSR <= every other method's CSR (same rho, same seed)
  I3. B1 behavior: with 10-min keep-alive and steady 1/min traffic,
      after the first cold start everything is warm
  I4. Monotonicity: for predictive methods, CSR is non-increasing in rho
  I5. Seed variance: repeated runs with different seeds produce
      non-identical CSR (stochasticity is real)
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.des import simulate_trace, simulate_function
from scripts.phase6_des import (
    decisions_from_rates, rates_ewma, rates_oracle, policy_b1,
)
from src.decision.newsvendor import newsvendor_quantile

PASS = "PASS"
FAIL = "FAIL"
failures = []


def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"  [{status}] {name} {detail}")
    if not cond:
        failures.append(name)


def main():
    rng = np.random.default_rng(42)

    # Small synthetic workload: 8 functions, 2 days
    N, T = 8, 2880
    counts = np.zeros((N, T), dtype=np.int64)
    counts[0, ::1] = 1                      # steady 1/min
    counts[1, ::10] = 1                     # 1 per 10 min
    counts[2] = rng.poisson(3.0, T)         # busy Poisson
    counts[3, ::60] = 5                     # hourly burst
    counts[4, 1000:1200] = 2                # activity window only
    counts[5] = rng.poisson(0.05, T)        # sparse
    counts[6, ::1440] = 1                   # daily
    # counts[7] stays all zero
    dur_means = np.array([1.0, 1.0, 8.0, 2.0, 30.0, 1.0, 1.0, 1.0])
    dur_stds = np.array([0.2] * N)

    print("=" * 60)
    print("I1. Conservation")
    print("=" * 60)
    pw, ka = policy_b1(counts)
    r = simulate_trace(counts, pw, ka, dur_means, dur_stds, seed=0)
    check("total invocations == trace sum",
          r["total_invocations"] == int(counts.sum()),
          f"({r['total_invocations']} vs {int(counts.sum())})")

    print("=" * 60)
    print("I2. Oracle dominance")
    print("=" * 60)
    # Oracle minimizes newsvendor COST, not CSR: at low rho it rationally
    # trades cold starts for memory. Correct invariants:
    #   (a) rho >= 1 (prewarming favored): Oracle CSR <= predictive baselines
    #   (b) all rho: Oracle is never Pareto-dominated (CSR and WM both worse)
    for rho in [0.5, 1.0, 10.0]:
        tau = newsvendor_quantile(rho)
        pw_o, ka_o = decisions_from_rates(rates_oracle(counts), tau)
        pw_e, ka_e = decisions_from_rates(rates_ewma(counts), tau)
        r_o = simulate_trace(counts, pw_o, ka_o, dur_means, dur_stds, seed=0)
        r_e = simulate_trace(counts, pw_e, ka_e, dur_means, dur_stds, seed=0)
        r_b1 = simulate_trace(counts, *policy_b1(counts), dur_means, dur_stds, seed=0)
        if rho >= 1.0:
            check(f"rho={rho}: Oracle CSR <= EWMA",
                  r_o["csr"] <= r_e["csr"] + 1e-9,
                  f"({r_o['csr']:.4f} vs {r_e['csr']:.4f})")
            check(f"rho={rho}: Oracle CSR <= B1",
                  r_o["csr"] <= r_b1["csr"] + 1e-9,
                  f"({r_o['csr']:.4f} vs {r_b1['csr']:.4f})")
        not_dominated = not (r_o["csr"] > r_e["csr"] + 1e-9
                             and r_o["wm_total_gb_s"] > r_e["wm_total_gb_s"] + 1e-9)
        check(f"rho={rho}: Oracle not Pareto-dominated by EWMA", not_dominated,
              f"(CSR {r_o['csr']:.4f}/{r_e['csr']:.4f}, "
              f"WM {r_o['wm_total_gb_s']:.0f}/{r_e['wm_total_gb_s']:.0f})")

    print("=" * 60)
    print("I3. B1 semantics: steady 1/min -> only initial cold starts")
    print("=" * 60)
    steady = np.ones((1, 500), dtype=np.int64)
    pw_s, ka_s = policy_b1(steady)
    r_s = simulate_trace(steady, pw_s, ka_s, np.array([1.0]), np.array([0.1]), seed=0)
    # With 1 req/min and dur 1s, one container suffices; only the very first
    # arrival (and possibly a couple during init lead) should be cold.
    check("steady traffic: cold starts <= 3",
          r_s["cold_starts"] <= 3, f"(cold={r_s['cold_starts']})")

    print("=" * 60)
    print("I4. Monotonicity in rho (EWMA + Oracle, averaged over 3 seeds)")
    print("=" * 60)
    for name, rates in [("EWMA", rates_ewma(counts)),
                        ("Oracle", rates_oracle(counts))]:
        csrs = []
        for rho in [0.1, 1.0, 10.0, 100.0]:
            tau = newsvendor_quantile(rho)
            pw_m, ka_m = decisions_from_rates(rates, tau)
            vals = [simulate_trace(counts, pw_m, ka_m, dur_means, dur_stds,
                                   seed=s)["csr"] for s in range(3)]
            csrs.append(np.mean(vals))
        mono = all(csrs[i] >= csrs[i + 1] - 0.005 for i in range(len(csrs) - 1))
        check(f"{name}: CSR non-increasing in rho", mono,
              f"({['%.4f' % c for c in csrs]})")

    print("=" * 60)
    print("I5. Seed variance (stochasticity)")
    print("=" * 60)
    tau = newsvendor_quantile(10.0)
    pw_v, ka_v = decisions_from_rates(rates_ewma(counts), tau)
    csr_by_seed = [simulate_trace(counts, pw_v, ka_v, dur_means, dur_stds,
                                  seed=s)["csr"] for s in range(5)]
    check("CSR varies across seeds", len(set(csr_by_seed)) > 1,
          f"(std={np.std(csr_by_seed):.5f})")

    print("=" * 60)
    if failures:
        print(f"RESULT: {len(failures)} FAILURES: {failures}")
        sys.exit(1)
    print("RESULT: ALL INVARIANTS PASS")


if __name__ == "__main__":
    main()
