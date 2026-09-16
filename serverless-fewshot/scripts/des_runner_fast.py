#!/usr/bin/env python3
"""Numba DES runner: executes job npz files exported by
phase6_des_sampled.py --export-jobs.

Run under the DES env (numpy 2.4 + numba):
  PYTHONPATH=/data/260715/site-packages-des:. python3.13 \
      scripts/des_runner_fast.py --jobs results/runs/des_jobs_2019 \
      --out results/runs/sim_results_des_2019.json

Parallelism comes from numba prange inside simulate_trace_fast (over
functions), so jobs run sequentially here.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.des_fast import simulate_trace_fast  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    jobs_root = Path(args.jobs)
    all_results = []
    for split_dir in sorted(jobs_root.iterdir()):
        if not (split_dir / "jobs.json").exists():
            continue
        meta = json.load(open(split_dir / "jobs.json"))
        shared = np.load(split_dir / "shared.npz")
        counts = shared["counts"]
        dm, ds = shared["dur_means"], shared["dur_stds"]
        ci = meta["cold_init"]
        print(f"### {meta['split']}: {counts.shape[0]} funcs, "
              f"{len(meta['jobs'])} jobs x {len(meta['seeds'])} seeds")
        for job in meta["jobs"]:
            jz = np.load(split_dir / job["file"])
            pw, ka = jz["prewarm"], jz["keepalive"]
            for seed in meta["seeds"]:
                t0 = time.time()
                m = simulate_trace_fast(counts, pw, ka, dm, ds, seed=seed,
                                        cold_mu=ci["mu"], cold_sigma=ci["sigma"])
                m.update({"method": job["method"], "cost_ratio": job["rho"],
                          "seed": seed, "split": meta["split"],
                          "elapsed_sec": time.time() - t0})
                all_results.append(m)
            rs = [r for r in all_results
                  if r["method"] == job["method"]
                  and r["cost_ratio"] == job["rho"]
                  and r["split"] == meta["split"]]
            print(f"    {job['method']:<20s} rho={job['rho']:6.1f} "
                  f"CSR={np.mean([x['csr'] for x in rs]):.4f} "
                  f"WM={np.mean([x['wm_per_1k_inv'] for x in rs]):.0f} "
                  f"({rs[-1]['elapsed_sec']:.1f}s/seed)", flush=True)

    with open(args.out, "w") as f:
        json.dump(all_results, f, default=float)
    print(f"Saved {len(all_results)} runs to {args.out}")


if __name__ == "__main__":
    main()
