#!/usr/bin/env python3
"""Statistical validation: numba simulator (des_fast) vs pure-python simulator
(src/sim/des.py) on identical exported job inputs (Azure-2021 scale).

The two are stochastically — not bitwise — equivalent (per-function RNG
streams, histogram percentiles), so acceptance is statistical:
  per (method, rho): |CSR_fast_mean - CSR_py_mean| <= 3 * sqrt(var_py + var_fast)
  over N_SEEDS seeds each, plus WM relative difference <= 2%.

Run under the DES env:
  PYTHONPATH=/data/260715/site-packages-des:. python3.13 \
      scripts/validate_des_fast.py --jobs results/runs/des_jobs_val2021
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.des import simulate_trace  # noqa: E402  (pure numpy, env-agnostic)
from src.sim.des_fast import simulate_trace_fast  # noqa: E402

N_SEEDS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    args = ap.parse_args()

    jobs_root = Path(args.jobs)
    failures = 0
    report = {}
    for split_dir in sorted(jobs_root.iterdir()):
        if not (split_dir / "jobs.json").exists():
            continue
        meta = json.load(open(split_dir / "jobs.json"))
        shared = np.load(split_dir / "shared.npz")
        counts = shared["counts"]
        dm, ds = shared["dur_means"], shared["dur_stds"]
        ci = meta["cold_init"]
        for job in meta["jobs"]:
            jz = np.load(split_dir / job["file"])
            pw, ka = jz["prewarm"], jz["keepalive"]

            csr_py, wm_py, csr_fa, wm_fa = [], [], [], []
            t0 = time.time()
            for seed in range(N_SEEDS):
                m = simulate_trace(counts, pw, ka, dm, ds, seed=seed,
                                   cold_mu=ci["mu"], cold_sigma=ci["sigma"])
                csr_py.append(m["csr"]); wm_py.append(m["wm_per_1k_inv"])
            t_py = time.time() - t0
            t0 = time.time()
            for seed in range(N_SEEDS):
                m = simulate_trace_fast(counts, pw, ka, dm, ds, seed=seed,
                                        cold_mu=ci["mu"], cold_sigma=ci["sigma"])
                csr_fa.append(m["csr"]); wm_fa.append(m["wm_per_1k_inv"])
            t_fa = time.time() - t0

            d = abs(np.mean(csr_fa) - np.mean(csr_py))
            tol = 3 * np.sqrt(np.var(csr_py) + np.var(csr_fa) + 1e-12)
            wm_rel = abs(np.mean(wm_fa) - np.mean(wm_py)) / max(np.mean(wm_py), 1e-9)
            ok = (d <= max(tol, 1e-4)) and (wm_rel <= 0.02)
            failures += 0 if ok else 1
            key = f"{meta['split']}|{job['method']}|rho{job['rho']}"
            report[key] = {
                "csr_py": float(np.mean(csr_py)), "csr_fast": float(np.mean(csr_fa)),
                "csr_diff": float(d), "csr_tol": float(max(tol, 1e-4)),
                "wm_rel_diff": float(wm_rel),
                "t_py_s": t_py, "t_fast_s": t_fa, "ok": bool(ok),
            }
            print(f"{'OK ' if ok else 'FAIL'} {key}: "
                  f"CSR py={np.mean(csr_py):.5f} fast={np.mean(csr_fa):.5f} "
                  f"(diff {d:.5f} tol {max(tol,1e-4):.5f}) "
                  f"WM rel {wm_rel*100:.2f}% | speedup x{t_py/max(t_fa,1e-9):.0f}",
                  flush=True)

    out = PROJECT_ROOT / "results" / "runs" / "des_fast_validation.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{'ALL OK' if failures == 0 else f'{failures} FAILURES'}; saved {out}")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
