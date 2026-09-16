#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incremental fast-DES runner for long R21 job bundles.

This is intentionally a thin wrapper around src.sim.des_fast.simulate_trace_fast.
It writes the output JSON after every seed so long cross-provider reruns can be
resumed without discarding completed rows.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.des_fast import simulate_trace_fast  # noqa: E402


def load_existing(path):
    if not path.exists():
        return []
    data = json.load(open(path))
    return data.get("results", data) if isinstance(data, dict) else data


def row_key(row):
    return (
        row["split"],
        row["method"],
        f"{float(row['cost_ratio']):.12g}",
        int(row["seed"]),
    )


def atomic_write(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(rows, f, default=float)
    os.replace(tmp, path)


def validate_decisions(prewarm, keepalive, split, job_file):
    for name, arr in [("prewarm", prewarm), ("keepalive", keepalive)]:
        if not np.isfinite(arr).all():
            raise ValueError(f"{split}/{job_file}: {name} contains non-finite values")
        if np.min(arr) < 0:
            raise ValueError(f"{split}/{job_file}: {name} contains negative values")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    jobs_root = Path(args.jobs)
    out = Path(args.out)
    rows = load_existing(out)
    done = {row_key(r) for r in rows}
    print(f"Loaded {len(rows)} existing rows from {out}" if rows else "No existing rows")

    for split_dir in sorted(jobs_root.iterdir()):
        if not (split_dir / "jobs.json").exists():
            continue
        meta = json.load(open(split_dir / "jobs.json"))
        shared = np.load(split_dir / "shared.npz")
        counts = shared["counts"]
        dm, ds = shared["dur_means"], shared["dur_stds"]
        ci = meta["cold_init"]
        print(
            f"### {meta['split']}: {counts.shape[0]} funcs, "
            f"{len(meta['jobs'])} jobs x {len(meta['seeds'])} seeds",
            flush=True,
        )
        for job in meta["jobs"]:
            jz = np.load(split_dir / job["file"])
            pw, ka = jz["prewarm"], jz["keepalive"]
            validate_decisions(pw, ka, meta["split"], job["file"])
            for seed in meta["seeds"]:
                key = (meta["split"], job["method"], f"{float(job['rho']):.12g}", int(seed))
                if key in done:
                    continue
                t0 = time.time()
                m = simulate_trace_fast(
                    counts,
                    pw,
                    ka,
                    dm,
                    ds,
                    seed=seed,
                    cold_mu=ci["mu"],
                    cold_sigma=ci["sigma"],
                )
                m.update(
                    {
                        "method": job["method"],
                        "cost_ratio": job["rho"],
                        "seed": seed,
                        "split": meta["split"],
                        "elapsed_sec": time.time() - t0,
                    }
                )
                rows.append(m)
                done.add(key)
                atomic_write(out, rows)
                print(
                    f"    {meta['split']} {job['method']:<20s} rho={job['rho']:6.1f} "
                    f"seed={seed} CSR={m['csr']:.4f} WM={m['wm_per_1k_inv']:.0f} "
                    f"({m['elapsed_sec']:.1f}s)",
                    flush=True,
                )
    atomic_write(out, rows)
    print(f"Saved {len(rows)} runs to {out}")


if __name__ == "__main__":
    main()
