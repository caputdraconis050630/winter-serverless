# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP2: Per-function paired statistics + headline decision gate.

Reads sim_results_des.json. For each split:
- Per-function CSR/WM (averaged over seeds) -> paired Wilcoxon vs A5,
  Cliff's delta, bootstrap CIs, Holm-Bonferroni across baselines.
- Matched-WM comparison: interpolate A5's Pareto curve to B1's WM level,
  compare CSR there (this is the AC3 test done right).
- Decision gate: steady-state co-headline iff CSR reduction >= 10% at
  matched WM with p < 0.01.
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "results" / "runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OURS = "A5_full_system"
BASELINES = ["B1_fixed_keepalive", "B2_histogram", "B4a_ewma", "B5_global",
             "A5_gated"]


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    more = sum((x < y).sum() for x, y in [(a[:, None], b[None, :])])
    less = sum((x > y).sum() for x, y in [(a[:, None], b[None, :])])
    d = (more - less) / (len(a) * len(b))
    mag = ("negligible" if abs(d) < 0.147 else "small" if abs(d) < 0.33
           else "medium" if abs(d) < 0.474 else "large")
    return float(d), mag


def bootstrap_ci_mean(x, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = np.array([rng.choice(x, len(x), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(x.mean()), float(lo), float(hi)


def holm_bonferroni(p_values):
    m = len(p_values)
    order = np.argsort(p_values)
    adj = np.ones(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(p_values[idx] * (m - rank), 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist()


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def per_function_table(results, split, rho):
    """func-level CSR/WM per method, averaged over seeds."""
    out = {}
    for method in [OURS] + BASELINES + ["Oracle"]:
        runs = [r for r in results
                if r["method"] == method and r["split"] == split
                and abs(r["cost_ratio"] - rho) < 1e-9]
        if not runs:
            continue
        csr_mat = np.array([r["func_csr"] for r in runs])   # [seeds, F]
        wm_mat = np.array([r["func_wm"] for r in runs])
        tot = np.array(runs[0]["func_total"], dtype=float)
        out[method] = {
            "func_csr": csr_mat.mean(axis=0),
            "func_wm": wm_mat.mean(axis=0),
            "func_total": tot,
            "agg_csr": float(np.mean([r["csr"] for r in runs])),
            "agg_csr_std": float(np.std([r["csr"] for r in runs])),
            "agg_wm": float(np.mean([r["wm_per_1k_inv"] for r in runs])),
            "agg_wm_std": float(np.std([r["wm_per_1k_inv"] for r in runs])),
        }
    return out


def pareto_curve(results, method, split):
    """(rho, mean CSR, mean WM/1k) sorted by WM."""
    pts = {}
    for r in results:
        if r["method"] == method and r["split"] == split:
            pts.setdefault(r["cost_ratio"], []).append(r)
    curve = []
    for rho, runs in sorted(pts.items()):
        curve.append({
            "rho": rho,
            "csr": float(np.mean([x["csr"] for x in runs])),
            "wm": float(np.mean([x["wm_per_1k_inv"] for x in runs])),
        })
    return curve


def matched_wm_comparison(results, split, target_method="B1_fixed_keepalive"):
    """Interpolate OURS Pareto curve at target's WM; compare CSR."""
    tgt = pareto_curve(results, target_method, split)
    ours = pareto_curve(results, OURS, split)
    if not tgt or len(ours) < 2:
        return None
    tgt_wm = float(np.mean([p["wm"] for p in tgt]))   # policy: rho-invariant
    tgt_csr = float(np.mean([p["csr"] for p in tgt]))

    ours_sorted = sorted(ours, key=lambda p: p["wm"])
    wms = [p["wm"] for p in ours_sorted]
    csrs = [p["csr"] for p in ours_sorted]

    if tgt_wm < wms[0]:
        ours_at = csrs[0]; note = "extrapolated_low(conservative)"
    elif tgt_wm > wms[-1]:
        ours_at = csrs[-1]; note = "extrapolated_high(conservative)"
    else:
        ours_at = float(np.interp(tgt_wm, wms, csrs)); note = "interpolated"

    rel_reduction = (tgt_csr - ours_at) / max(tgt_csr, 1e-12)
    return {
        "target": target_method, "target_wm": tgt_wm, "target_csr": tgt_csr,
        "ours_csr_at_matched_wm": ours_at,
        "relative_csr_reduction": float(rel_reduction),
        "note": note,
    }


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "sim_results_des.json"
    with open(RUNS_DIR / src) as f:
        results = json.load(f)

    splits = sorted(set(r["split"] for r in results))
    rho_main = 10.0
    report = {}

    for split in splits:
        print(f"\n{'='*70}\nSPLIT {split} (rho={rho_main})\n{'='*70}")
        tab = per_function_table(results, split, rho_main)
        if OURS not in tab:
            continue
        ours_csr = tab[OURS]["func_csr"]
        # Only functions with traffic contribute to CSR pairing
        active = tab[OURS]["func_total"] > 0

        split_report = {"paired": {}, "aggregate": {}}
        p_raw = []
        base_list = [b for b in BASELINES if b in tab]

        for b in base_list:
            base_csr = tab[b]["func_csr"]
            diffs = base_csr[active] - ours_csr[active]
            nz = diffs[diffs != 0]
            if len(nz) >= 6:
                _, p = sp_stats.wilcoxon(nz)
            else:
                p = 1.0
            d, mag = cliffs_delta(ours_csr[active], base_csr[active])
            p_raw.append(p)
            split_report["paired"][b] = {
                "wilcoxon_p": float(p), "cliffs_delta": d, "magnitude": mag,
                "mean_diff": float(diffs.mean()),
                "n_functions": int(active.sum()),
            }

        adj = holm_bonferroni(p_raw)
        for b, pa in zip(base_list, adj):
            split_report["paired"][b]["p_holm"] = pa

        for m, v in tab.items():
            mean, lo, hi = bootstrap_ci_mean(v["func_csr"][active])
            split_report["aggregate"][m] = {
                "csr": v["agg_csr"], "csr_std_seeds": v["agg_csr_std"],
                "csr_func_ci": [lo, hi],
                "wm": v["agg_wm"], "wm_std_seeds": v["agg_wm_std"],
            }
            print(f"  {m:<22s} CSR={v['agg_csr']:.4f}+-{v['agg_csr_std']:.4f} "
                  f"funcCI=[{lo:.4f},{hi:.4f}]  WM={v['agg_wm']:.0f}")

        print("\n  Paired tests (A5 vs baseline, per-function CSR):")
        for b in base_list:
            pr = split_report["paired"][b]
            print(f"    vs {b:<20s} p={pr['wilcoxon_p']:.4f}"
                  f"{stars(pr['p_holm'])} (holm={pr['p_holm']:.4f}) "
                  f"delta={pr['cliffs_delta']:.3f} ({pr['magnitude']}) "
                  f"meanDiff={pr['mean_diff']:+.4f}")

        mw = matched_wm_comparison(results, split)
        split_report["matched_wm"] = mw
        if mw:
            print(f"\n  Matched-WM vs B1: B1 (CSR={mw['target_csr']:.4f}, "
                  f"WM={mw['target_wm']:.0f}) vs Ours CSR="
                  f"{mw['ours_csr_at_matched_wm']:.4f} at same WM "
                  f"-> relative reduction {mw['relative_csr_reduction']*100:+.1f}% "
                  f"[{mw['note']}]")

        report[split] = split_report

    # ---- Decision gate ----
    print(f"\n{'='*70}\nDECISION GATE (steady-state headline)\n{'='*70}")
    gate = {"criteria": "CSR reduction >=10% at matched WM AND p_holm<0.01 vs B1"}
    votes = []
    for split in splits:
        mw = report[split].get("matched_wm")
        pr = report[split]["paired"].get("B1_fixed_keepalive", {})
        ok = (mw and mw["relative_csr_reduction"] >= 0.10
              and pr.get("p_holm", 1.0) < 0.01)
        votes.append(ok)
        print(f"  {split}: reduction={mw['relative_csr_reduction']*100:+.1f}% "
              f"p_holm={pr.get('p_holm', 1):.4f} -> {'PASS' if ok else 'FAIL'}")
    decision = ("steady_state_coheadline" if all(votes)
                else "adaptation_scalability_headline")
    gate["per_split_pass"] = votes
    gate["decision"] = decision
    report["_decision_gate"] = gate
    print(f"\n  HEADLINE DECISION: {decision}")

    out = RUNS_DIR / "steady_state_stats.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
