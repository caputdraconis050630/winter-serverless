#!/usr/bin/env python3
"""Process-parallel wrapper for DES jobs exported for simulate_trace_fast.

This runner preserves the metric path used by des_runner_fast.py but schedules
individual (split, job, seed) simulations across worker processes and writes a
checkpoint after each completed task.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.des_fast import simulate_trace_fast  # noqa: E402


def _run_one(task):
    split_dir = Path(task["split_dir"])
    meta = json.load(open(split_dir / "jobs.json"))
    shared = np.load(split_dir / "shared.npz")
    counts = shared["counts"]
    dm, ds = shared["dur_means"], shared["dur_stds"]
    ci = meta["cold_init"]

    job = task["job"]
    jz = np.load(split_dir / job["file"])
    pw, ka = jz["prewarm"], jz["keepalive"]

    t0 = time.time()
    m = simulate_trace_fast(
        counts, pw, ka, dm, ds, seed=task["seed"],
        cold_mu=ci["mu"], cold_sigma=ci["sigma"])
    m.update({
        "method": job["method"],
        "cost_ratio": job["rho"],
        "seed": task["seed"],
        "split": meta["split"],
        "elapsed_sec": time.time() - t0,
    })
    return m


def _task_key(r):
    return (r["split"], r["method"], float(r["cost_ratio"]), int(r["seed"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    jobs_root = Path(args.jobs)
    out = Path(args.out)
    all_results = []
    done = set()
    if args.resume and out.exists():
        all_results = json.load(open(out))
        done = {_task_key(r) for r in all_results}
        print(f"resuming with {len(done)} completed tasks from {out}", flush=True)

    tasks = []
    for split_dir in sorted(jobs_root.iterdir()):
        if not (split_dir / "jobs.json").exists():
            continue
        meta = json.load(open(split_dir / "jobs.json"))
        print(f"### {meta['split']}: {len(meta['jobs'])} jobs x "
              f"{len(meta['seeds'])} seeds", flush=True)
        for job in meta["jobs"]:
            for seed in meta["seeds"]:
                key = (meta["split"], job["method"], float(job["rho"]), int(seed))
                if key in done:
                    continue
                tasks.append({"split_dir": str(split_dir), "job": job, "seed": seed})

    print(f"running {len(tasks)} pending tasks with {args.workers} workers", flush=True)
    if not tasks:
        return

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run_one, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            all_results.append(r)
            json.dump(all_results, open(out, "w"), default=float)
            print(f"    {r['split']:<12s} {r['method']:<20s} "
                  f"rho={float(r['cost_ratio']):6.1f} seed={int(r['seed'])} "
                  f"CSR={100.0 * float(r['csr']):.4f} "
                  f"WM={float(r['wm_per_1k_inv']):.0f} "
                  f"({float(r['elapsed_sec']):.1f}s)", flush=True)

    print(f"Saved {len(all_results)} runs to {out}", flush=True)


if __name__ == "__main__":
    main()
