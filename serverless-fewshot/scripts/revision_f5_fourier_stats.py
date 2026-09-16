# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision F5: statistics for the B3 (Fourier) arm across every comparison
surface the paper reports.

Inputs (all produced by F1-F4 or already archived):
  2021 steady   revision_f1_fourier_steady.json   vs sim_results_des_revision.json
  2019 steady   revision_f3_fourier_2019_S*.json  vs sim_results_des_2019.json
                                                     + des_sample_manifest.json
  Huawei steady revision_f3_fourier_huawei_*.json vs revision_h2_huawei_des_*.json
  cohorts       revision_f4_fourier_cohort_*.json (paired stats computed there)

Per surface x split x rho this reports, against EWMA and the learned path:
  - CSR (population-weighted on 2019) and warm memory per 1k invocations
  - function-level paired Wilcoxon (Holm-corrected within a split) + Cliff's
    delta, the protocol of phase6_stats.py
  - the pre-registered cost accounting C = 15*rho*cold + wm (GB.s), which is
    the only fair single number here: B3 buys cold starts with memory, so a
    CSR comparison at equal rho is not a comparison at equal cost
  - matched-memory CSR: each arm's (WM, CSR) curve interpolated at the other's
    operating points, strictly inside the overlapping WM range (no
    extrapolation)

Output: results/runs/revision_f5_fourier_stats.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sstats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_stats import cliffs_delta, holm_bonferroni  # noqa: E402

RUNS = PROJECT_ROOT / "results" / "runs"
B3 = "B3_fourier"
KAPPA = 15.0                      # registered cost constant
N_BOOT = 5000


# ------------------------------------------------------------------ helpers
def per_function(rows, weights=None):
    """Average per-function arrays over seeds; also aggregate CSR/WM.

    With inclusion weights (the 2019 stratified sample) the aggregates are the
    population estimates of des_2019_stats.py -- weighted ratio estimators
    computed per seed, then averaged -- not the sample means the DES rows
    carry. Reporting r["csr"] there would label an unweighted sample mean as a
    population CSR.
    """
    def _agg(r):
        cold = np.asarray(r["func_cold"], dtype=float)
        tot = np.asarray(r["func_total"], dtype=float)
        wm = np.asarray(r["func_wm"], dtype=float)
        if weights is None:
            return r["csr"], r["wm_per_1k_inv"]
        return (float((weights * cold).sum() / max((weights * tot).sum(), 1e-9)),
                float((weights * wm).sum()
                      / max((weights * tot).sum() / 1000.0, 1e-9)))

    aggs = [_agg(r) for r in rows]
    return {
        "cold": np.mean([r["func_cold"] for r in rows], axis=0),
        "total": np.mean([r["func_total"] for r in rows], axis=0),
        "wm": np.mean([r["func_wm"] for r in rows], axis=0),
        "csr_seeds": [a[0] for a in aggs],
        "wm1k_seeds": [a[1] for a in aggs],
    }


def paired_stats(a, b):
    """a, b: per_function dicts. Positive delta/mean_diff favors a."""
    active = a["total"] > 0
    fa = a["cold"][active] / np.maximum(a["total"][active], 1)
    fb = b["cold"][active] / np.maximum(b["total"][active], 1)
    diffs = fb - fa
    nz = diffs[diffs != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else 1.0
    d, mag = cliffs_delta(fa, fb)
    return {"wilcoxon_p": p, "cliffs_delta": d, "magnitude": mag,
            "mean_csr_diff_pp": float(diffs.mean() * 100),
            "n_functions": int(active.sum())}


def cost_stats(a, b, rho, weights=None):
    """Pre-registered accounting C = KAPPA*rho*cold + wm, per function."""
    Ca = KAPPA * rho * a["cold"] + a["wm"]
    Cb = KAPPA * rho * b["cold"] + b["wm"]
    w = np.ones_like(Ca) if weights is None else weights
    cheaper = Ca < Cb
    return {"total_cost_a": float((w * Ca).sum()),
            "total_cost_b": float((w * Cb).sum()),
            "cost_ratio_a_over_b": float((w * Ca).sum()
                                         / max((w * Cb).sum(), 1e-9)),
            "frac_functions_a_cheaper": float(cheaper.mean())}


def matched_memory(curve_a, curve_b):
    """CSR of curve_a interpolated at curve_b's WM points, inside the overlap.

    curve_*: list of (wm_per_1k, csr, rho) sorted by wm. Returns one entry per
    comparator operating point that falls inside curve_a's measured WM range;
    points outside it are reported as skipped rather than extrapolated.
    """
    wa = np.array([c[0] for c in curve_a])
    ca = np.array([c[1] for c in curve_a])
    order = np.argsort(wa)
    wa, ca = wa[order], ca[order]
    out = {"points": [], "skipped_outside_range": []}
    for wm_b, csr_b, rho_b in curve_b:
        if wm_b < wa.min() or wm_b > wa.max():
            out["skipped_outside_range"].append(
                {"rho": rho_b, "wm": wm_b,
                 "a_wm_range": [float(wa.min()), float(wa.max())]})
            continue
        csr_a_at = float(np.interp(wm_b, wa, ca))
        out["points"].append(
            {"rho": rho_b, "wm": float(wm_b),
             "csr_a_interp_pp": csr_a_at * 100,
             "csr_b_pp": csr_b * 100,
             "a_minus_b_pp": (csr_a_at - csr_b) * 100})
    return out


def surface(rows_b3, rows_comp, comparators, weights_by_split=None,
            label=""):
    """One comparison surface: dict split -> rho -> stats."""
    g = defaultdict(list)
    for r in rows_b3 + rows_comp:
        g[(r["split"], float(r["cost_ratio"]), r["method"])].append(r)
    # weights are per split, so resolve them before aggregating
    agg = {}
    for k, v in g.items():
        w_k = (None if weights_by_split is None
               else np.asarray(weights_by_split[k[0]], dtype=float))
        agg[k] = per_function(v, w_k)

    splits = sorted({k[0] for k in agg})
    out = {}
    for sp in splits:
        w = None if weights_by_split is None else np.array(weights_by_split[sp])
        rhos = sorted({k[1] for k in agg if k[0] == sp and k[2] == B3})
        curves = defaultdict(list)
        for m in [B3] + comparators:
            for rho in rhos:
                a = agg.get((sp, rho, m))
                if a is None:
                    continue
                csr_v = float(np.mean(a["csr_seeds"]))
                wm1k = float(np.mean(a["wm1k_seeds"]))
                curves[m].append((wm1k, csr_v, rho))

        sp_out = {"curves": {m: [{"rho": r, "wm_per_1k": wm, "csr_pp": c * 100}
                                 for (wm, c, r) in v]
                             for m, v in curves.items()},
                  "by_rho": {}}
        for rho in rhos:
            b3 = agg.get((sp, rho, B3))
            if b3 is None:
                continue
            entry = {"b3_csr_pp": float(np.mean(b3["csr_seeds"]) * 100),
                     "b3_csr_std_pp": float(np.std(b3["csr_seeds"]) * 100),
                     "b3_wm_per_1k": float(np.mean(b3["wm1k_seeds"])),
                     "vs": {}}
            ps, keys = [], []
            for m in comparators:
                c = agg.get((sp, rho, m))
                if c is None:
                    continue
                st = paired_stats(b3, c)
                st.update(cost_stats(b3, c, rho, w))
                st["comparator_csr_pp"] = float(np.mean(c["csr_seeds"]) * 100)
                st["comparator_wm_per_1k"] = float(np.mean(c["wm1k_seeds"]))
                entry["vs"][m] = st
                ps.append(st["wilcoxon_p"])
                keys.append(m)
            for k, pa in zip(keys, holm_bonferroni(ps)):
                entry["vs"][k]["p_holm"] = float(pa)
            sp_out["by_rho"][str(rho)] = entry

        sp_out["matched_memory"] = {
            m: matched_memory(curves[B3], curves[m])
            for m in comparators if m in curves}
        out[sp] = sp_out
        for rho in rhos:
            e = sp_out["by_rho"].get(str(rho))
            if not e:
                continue
            line = (f"{label}{sp} rho={rho:6.1f}: B3 {e['b3_csr_pp']:.2f}% / "
                    f"{e['b3_wm_per_1k']:.0f}")
            for m, st in e["vs"].items():
                line += (f" | {m} {st['comparator_csr_pp']:.2f}% /"
                         f" {st['comparator_wm_per_1k']:.0f}"
                         f" (cost x{st['cost_ratio_a_over_b']:.2f},"
                         f" p_holm {st['p_holm']:.1e})")
            print(line, flush=True)
    return out


def load(path):
    d = json.load(open(path))
    return d["results"] if isinstance(d, dict) and "results" in d else d


def main():
    out = {"cost_model": f"C = {KAPPA:g}*rho*cold + wm_gbs",
           "arm": B3, "surfaces": {}}

    # ---- 2021 steady state -------------------------------------------
    print("\n===== Azure 2021 steady state =====")
    b3 = load(RUNS / "revision_f1_fourier_steady.json")
    comp = load(RUNS / "sim_results_des_revision.json")
    keep = ["A5_full_system", "A5_gated_v3", "B4a_ewma",
            "B1_fixed_keepalive", "Oracle"]
    rhos_b3 = {float(r["cost_ratio"]) for r in b3}
    comp = [r for r in comp if r["method"] in keep
            and float(r["cost_ratio"]) in rhos_b3]
    out["surfaces"]["azure2021_steady"] = surface(b3, comp, keep,
                                                  label="  2021 ")

    # ---- 2019 steady state (stratified sample -> population weights) --
    print("\n===== Azure 2019 steady state (population-weighted) =====")
    b3 = []
    for sp in ["S1", "S2", "S3"]:
        p = RUNS / f"revision_f3_fourier_2019_{sp}.json"
        if p.exists():
            b3.extend(load(p))
    if b3:
        comp = load(RUNS / "sim_results_des_2019.json")
        keep19 = ["A5_full_system", "A5_gated_v2", "B4a_ewma", "B5_global",
                  "B1_fixed_keepalive", "Oracle"]
        rhos_b3 = {float(r["cost_ratio"]) for r in b3}
        comp = [r for r in comp if r["method"] in keep19
                and float(r["cost_ratio"]) in rhos_b3]
        manifest = json.load(open(RUNS / "des_sample_manifest.json"))
        weights = {sp: manifest[sp]["weights"] for sp in ["S1", "S2", "S3"]}
        out["surfaces"]["azure2019_steady"] = surface(
            b3, comp, keep19, weights_by_split=weights, label="  2019 ")
    else:
        print("  (no 2019 B3 results yet -- skipped)")

    # ---- Huawei steady state -----------------------------------------
    print("\n===== Huawei 2023 steady state =====")
    b3 = [r for p in sorted(RUNS.glob("revision_f3_fourier_huawei_*.json"))
          for r in load(p)]
    if b3:
        comp = [r for p in sorted(RUNS.glob("revision_h2_huawei_des_*.json"))
                for r in load(p)]
        keephw = ["A5_full_system", "B4a_ewma", "B2_histogram",
                  "B1_fixed_keepalive", "Oracle"]
        rhos_b3 = {float(r["cost_ratio"]) for r in b3}
        comp = [r for r in comp if r["method"] in keephw
                and float(r["cost_ratio"]) in rhos_b3]
        out["surfaces"]["huawei_steady"] = surface(b3, comp, keephw,
                                                   label="  HW ")
    else:
        print("  (no Huawei B3 results yet -- skipped)")

    # ---- cohorts (paired stats already computed by F4) ----------------
    print("\n===== Onboarding cohorts =====")
    for trace in ["azure2019", "huawei"]:
        p = RUNS / f"revision_f4_fourier_cohort_{trace}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        out["surfaces"][f"{trace}_cohort"] = {
            "n_functions": d["n_functions"],
            "by_rho": {rho: {"b3_csr_pp": v["overall_csr"] * 100,
                             "b3_wm_total": v["wm_total"],
                             "vs": v["_paired_cold_all"]}
                       for rho, v in d["by_rho"].items()}}
        for rho, v in d["by_rho"].items():
            base = v["_paired_cold_all"]
            line = f"  {trace} cohort rho={rho}: B3 {v['overall_csr']*100:.2f}%"
            for k, s in base.items():
                line += (f" | {k.split('_vs_')[1]} "
                         f"{s['base_overall_csr']*100:.2f}%"
                         f" (p_holm {s['p_holm']:.1e})")
            print(line, flush=True)

    path = RUNS / "revision_f5_fourier_stats.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"\nSaved + verified {path}")


if __name__ == "__main__":
    main()
