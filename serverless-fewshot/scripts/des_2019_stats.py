#!/usr/bin/env python3
"""Population-level statistics for the sampled Azure-2019 DES runs.

Reads sim_results_des_2019.json + des_sample_manifest.json and produces,
per split x rho x method:
  - weighted ratio estimate of population CSR (take-all + SRS weights)
  - weighted WM per 1k invocations
  - function-level paired comparisons vs B5_global and B1 at each rho:
    weighted paired bootstrap CI of the CSR difference + Wilcoxon (unweighted,
    sample-level) + win rate.
Seeds are averaged per function before pairing (function is the unit).

Output: results/runs/des_2019_stats.json + results/tables/T3_des_2019.csv
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sstats

PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

RESULTS = RUNS_DIR / "sim_results_des_2019.json"
MANIFEST = RUNS_DIR / "des_sample_manifest.json"
BASELINES = ("B5_global", "B1_fixed_keepalive", "B4a_ewma")
TARGET = "A5_gated_v2"
N_BOOT = 5000


def weighted_csr(cold, total, w):
    return float((w * cold).sum() / max((w * total).sum(), 1e-9))


def weighted_paired_boot(cold_a, cold_b, total, w, seed=11):
    """CI for CSR_a - CSR_b under function resampling (weights kept)."""
    rng = np.random.default_rng(seed)
    n = len(total)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = (weighted_csr(cold_a[idx], total[idx], w[idx])
                    - weighted_csr(cold_b[idx], total[idx], w[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    rows = json.load(open(RESULTS))
    manifest = json.load(open(MANIFEST))

    # (split, rho, method) -> per-function arrays averaged over seeds
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["split"], r["cost_ratio"], r["method"])].append(r)

    agg = {}
    for key, rs in grouped.items():
        agg[key] = {
            "cold": np.mean([r["func_cold"] for r in rs], axis=0),
            "total": np.mean([r["func_total"] for r in rs], axis=0),
            "wm": np.mean([r["func_wm"] for r in rs], axis=0),
            "csr_seeds": [r["csr"] for r in rs],
        }

    out = {}
    csv_lines = ["split,rho,method,pop_csr,csr_sample_mean,csr_sample_std,"
                 "wm_per_1k_weighted,diff_vs_b5,ci_lo,ci_hi,wilcoxon_p_vs_b5,"
                 "win_rate_vs_b5"]
    splits = sorted({k[0] for k in agg})
    for split in splits:
        w = np.array(manifest[split]["weights"])
        rhos = sorted({k[1] for k in agg if k[0] == split})
        for rho in rhos:
            methods = sorted({k[2] for k in agg
                              if k[0] == split and k[1] == rho})
            ref = agg.get((split, rho, "B5_global"))
            for m in methods:
                a = agg[(split, rho, m)]
                pop_csr = weighted_csr(a["cold"], a["total"], w)
                wm1k = float((w * a["wm"]).sum()
                             / max((w * a["total"]).sum() / 1000.0, 1e-9))
                entry = {
                    "pop_csr": pop_csr,
                    "csr_sample_mean": float(np.mean(a["csr_seeds"])),
                    "csr_sample_std": float(np.std(a["csr_seeds"])),
                    "wm_per_1k_weighted": wm1k,
                }
                dif = ci = wp = wr = ""
                if ref is not None and m != "B5_global":
                    d = (weighted_csr(a["cold"], a["total"], w)
                         - weighted_csr(ref["cold"], ref["total"], w))
                    lo, hi = weighted_paired_boot(
                        a["cold"], ref["cold"], a["total"], w)
                    fa = a["cold"] / np.maximum(a["total"], 1)
                    fb = ref["cold"] / np.maximum(ref["total"], 1)
                    mask = ~np.isclose(fa, fb)
                    try:
                        wp_v = float(sstats.wilcoxon(fa[mask], fb[mask]).pvalue) \
                            if mask.sum() > 10 else 1.0
                    except ValueError:
                        wp_v = 1.0
                    wr_v = float((fa < fb).mean())
                    entry.update({"csr_diff_vs_b5": d,
                                  "csr_diff_ci95": [lo, hi],
                                  "wilcoxon_p_vs_b5": wp_v,
                                  "win_rate_vs_b5": wr_v})
                    dif, ci, wp, wr = (f"{d:.5f}", f"{lo:.5f}|{hi:.5f}",
                                       f"{wp_v:.2e}", f"{wr_v:.3f}")
                out[f"{split}|{rho}|{m}"] = entry
                csv_lines.append(
                    f"{split},{rho},{m},{pop_csr:.5f},"
                    f"{entry['csr_sample_mean']:.5f},"
                    f"{entry['csr_sample_std']:.5f},{wm1k:.1f},"
                    f"{dif},{ci},{wp},{wr}")

    with open(RUNS_DIR / "des_2019_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / "T3_des_2019.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    print("Population CSR (weighted) at rho=10:")
    for split in splits:
        print(f"  --- {split} ---")
        for m in sorted({k[2] for k in agg if k[0] == split}):
            e = out.get(f"{split}|10.0|{m}")
            if e:
                extra = ""
                if "csr_diff_vs_b5" in e:
                    lo, hi = e["csr_diff_ci95"]
                    extra = (f"  dCSR_vs_B5={e['csr_diff_vs_b5']:+.5f} "
                             f"[{lo:+.5f},{hi:+.5f}] p={e['wilcoxon_p_vs_b5']:.1e}")
                print(f"    {m:<22s} CSR={e['pop_csr']:.5f} "
                      f"WM/1k={e['wm_per_1k_weighted']:.0f}{extra}")
    print(f"Saved {RUNS_DIR / 'des_2019_stats.json'} and T3_des_2019.csv")


if __name__ == "__main__":
    main()
