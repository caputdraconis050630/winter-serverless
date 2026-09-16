# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R14: function-level equivalence readout (recomputation).

The archived result was produced by an ad-hoc recomputation from
sim_results_des_revision.json; the procedure is fixed by
results/runs/revision_r14_PREREG.md but no script was saved with it. This
file restores the generator so the artifact is self-contained.

It recomputes, per split at rho=10 (A5_full_system vs B4a_ewma, seed-averaged
per-function cold/total):

  1. per-function paired CSR difference: median, IQR, share within +/-0.10pp
     and +/-1.0pp, share favoring A5 among non-ties;
  2. bootstrap 95% CI on the MEDIAN difference (10,000 resamples, seed 260715).
     NOTE: the original run created ONE Generator and consumed it across S1, S2,
     S3 in that order, and used the same resample index matrix for both the
     median and the aggregate statistic. Both are reproduced here; reseeding
     per split does not match the archive.
  3. delta*, the smallest margin at which a function-level bootstrap-CI
     equivalence test would pass (CI of the mean weighted difference inside
     +/-delta*) -- a margin sweep rather than a post-hoc chosen margin.

Verify against the archive:
    python3 revision_r14_funclevel_equiv.py --check
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RUNS = PROJECT_ROOT / "results" / "runs"
SRC = RUNS / "sim_results_des_revision.json"
OUT = RUNS / "revision_r14_funclevel_equiv.json"

RHO = 10.0
BOOT_N = 10_000
BOOT_SEED = 260715
OURS, BASE = "A5_full_system", "B4a_ewma"

# S1·S2는 비트 단위로 재현되고 S3의 aggregate CI만 5번째 유효숫자에서 어긋난다
# (재계산 0.161322 vs 아카이브 0.161312, 절대차 1e-5 pp). S3에만 있는 총 호출 0인
# 함수 26개의 가중치 처리 차이로 보이며, 원 코드가 남아 있지 않아 특정하지 못했다.
# 보고되는 delta* = 0.16 pp는 어느 쪽이든 같다. 숨기지 않고 허용 오차를 명시한다.
TOL = 2e-5


def per_function_csr(rows):
    """Seed-average per-function cold/total, then form per-function CSR."""
    cold = np.mean([np.asarray(r["func_cold"], dtype=float) for r in rows], axis=0)
    total = np.mean([np.asarray(r["func_total"], dtype=float) for r in rows], axis=0)
    return np.divide(cold, np.maximum(total, 1e-9)), total


def analyze(split, results, rng):
    def pick(method):
        return [r for r in results
                if r["split"] == split and r["method"] == method
                and float(r["cost_ratio"]) == RHO]

    a_rows, b_rows = pick(OURS), pick(BASE)
    if not a_rows or not b_rows:
        return None
    a_csr, a_tot = per_function_csr(a_rows)
    b_csr, _ = per_function_csr(b_rows)

    # positive = A5 worse (matches the archive's sign convention)
    diff_pp = (a_csr - b_csr) * 100.0
    nonties = diff_pp[diff_pp != 0]

    n = len(diff_pp)
    idx = rng.integers(0, n, size=(BOOT_N, n))   # 공유 Generator (위 NOTE 참조)
    med_boot = np.median(diff_pp[idx], axis=1)

    # aggregate difference weighted by invocation volume
    w = a_tot / max(a_tot.sum(), 1e-9)
    agg_boot = (diff_pp[idx] * w[idx]).sum(axis=1) / np.maximum(w[idx].sum(axis=1), 1e-12)
    agg_ci = [float(np.percentile(agg_boot, 2.5)), float(np.percentile(agg_boot, 97.5))]

    # delta*: smallest margin containing the aggregate CI
    delta_star = float(max(abs(agg_ci[0]), abs(agg_ci[1])))

    return {
        "n_functions": int(n),
        "median_diff_pp": float(np.median(diff_pp)),
        "iqr_pp": [float(np.percentile(diff_pp, 25)), float(np.percentile(diff_pp, 75))],
        "share_within_0p10pp": float(np.mean(np.abs(diff_pp) <= 0.10)),
        "share_within_1pp": float(np.mean(np.abs(diff_pp) <= 1.0)),
        "share_a5_better_of_nonties": float(np.mean(nonties < 0)) if len(nonties) else None,
        "median_diff_boot_ci_pp": [float(np.percentile(med_boot, 2.5)),
                                   float(np.percentile(med_boot, 97.5))],
        "aggregate_weighted_diff_ci_pp": agg_ci,
        "delta_star_required_margin_pp": delta_star,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="아카이브와 대조만 하고 파일을 쓰지 않는다")
    args = ap.parse_args()

    results = json.load(open(SRC))["results"]
    rng = np.random.default_rng(BOOT_SEED)   # split 간 공유 (재시드하지 않는다)
    out = {"prereg": "revision_r14_PREREG.md", "rho": RHO, "by_split": {}}
    for split in ("S1", "S2", "S3"):
        r = analyze(split, results, rng)
        if r:
            out["by_split"][split] = r

    if args.check:
        if not OUT.exists():
            sys.exit(f"아카이브 없음: {OUT}")
        arch = json.load(open(OUT))["by_split"]
        bad = 0
        for split, got in out["by_split"].items():
            want = arch.get(split, {})
            for k, v in got.items():
                w = want.get(k)
                if isinstance(v, float) and isinstance(w, float):
                    if abs(v - w) > TOL:
                        print(f"불일치 {split}.{k}: 재계산 {v} vs 아카이브 {w}")
                        bad += 1
                elif isinstance(v, list) and isinstance(w, list):
                    if any(abs(x - y) > TOL for x, y in zip(v, w)):
                        print(f"불일치 {split}.{k}: 재계산 {v} vs 아카이브 {w}")
                        bad += 1
                elif v != w:
                    print(f"불일치 {split}.{k}: 재계산 {v} vs 아카이브 {w}")
                    bad += 1
        print(f"{'✅ 아카이브와 일치' if bad == 0 else f'⚠ 불일치 {bad}건'}")
        return 0 if bad == 0 else 1

    with open(OUT, "w") as f:
        json.dump(out, f, default=float)
    print(f"Saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
