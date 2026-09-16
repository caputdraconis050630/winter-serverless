# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision F2: export B3 (Fourier) DES jobs for Azure 2019 and Huawei 2023.

Reuses the exact count matrices the archived campaigns were run on
(results/runs/des_jobs_*/<split>/shared.npz), so the B3 arm is simulated on
the same functions, the same ticks, the same durations, the same seeds and
the same rho grid as every comparator already in those tables. Only the
(prewarm, keepalive) decision matrices are new.

  python3.13 scripts/revision_f2_fourier_export.py            # both traces
  python3.13 scripts/revision_f2_fourier_export.py --only 2019

Then run each exported split dir with the numba runner:
  PYTHONPATH=/data/260715/site-packages-des:. python3.13 \
      scripts/des_runner_fast.py --jobs results/runs/des_jobs_f3_2019 \
      --out results/runs/revision_f3_fourier_2019.json
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import (  # noqa: E402
    decisions_from_rates, rates_fourier,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "results" / "runs"
SOURCES = {
    "2019": ("des_jobs_2019", ["S1", "S2", "S3"], "des_jobs_f3_2019"),
    "huawei": ("des_jobs_huawei", ["h_mixed", "h_sparse", "h_saturated"],
               "des_jobs_f3_huawei"),
}
METHOD = "B3_fourier"


def export(src_root, splits, dst_root):
    for split in splits:
        src = RUNS_DIR / src_root / split
        dst = RUNS_DIR / dst_root / split
        dst.mkdir(parents=True, exist_ok=True)
        meta = json.load(open(src / "jobs.json"))
        shared = np.load(src / "shared.npz")
        counts = shared["counts"]
        rhos = sorted({j["rho"] for j in meta["jobs"]})

        t = time.time()
        rates = rates_fourier(counts)
        print(f"### {split}: {counts.shape[0]} funcs, {counts.shape[1]} ticks, "
              f"rhos={rhos}, seeds={meta['seeds']} | B3 rates in "
              f"{time.time()-t:.0f}s (mean predicted {rates.mean():.4f}, "
              f"actual {counts.mean():.4f})", flush=True)

        # shared.npz is identical to the source campaign's; copy so the dir
        # is self-contained for the runner (a dangling symlink would be a
        # silent wrong-counts hazard).
        if not (dst / "shared.npz").exists():
            shutil.copy2(src / "shared.npz", dst / "shared.npz")

        jobs = []
        for rho in rhos:
            pw, ka = decisions_from_rates(rates, newsvendor_quantile(rho))
            fname = f"{METHOD}__rho{rho}.npz"
            np.savez_compressed(dst / fname, prewarm=pw, keepalive=ka)
            jobs.append({"method": METHOD, "rho": rho, "file": fname})
            print(f"    rho={rho:6.1f} prewarm mean {pw.mean():.4f}, "
                  f"keepalive mean {ka.mean():.2f}", flush=True)
        with open(dst / "jobs.json", "w") as f:
            json.dump({"split": meta["split"], "seeds": meta["seeds"],
                       "cold_init": meta["cold_init"], "jobs": jobs}, f)
        print(f"  exported {len(jobs)} jobs -> {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(SOURCES), default=None)
    args = ap.parse_args()
    for key, (src_root, splits, dst_root) in SOURCES.items():
        if args.only and key != args.only:
            continue
        print(f"\n===== {key} =====", flush=True)
        export(src_root, splits, dst_root)


if __name__ == "__main__":
    main()
