#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R22: post-hoc EWMA-alpha frontier audit for onboarding cohorts.

The script replays only EWMA variants over existing onboarding cohorts and
uses the released WINTER onboarding JSONs as fixed references. It does not
retrain or re-estimate WINTER.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import COLD_INIT, decisions_from_rates, rates_ewma
from scripts.revision_m2a_holdout_cohort import (
    GATE_THRESHOLD,
    MIN_DAY1_INV,
    MIN_IDLE_PREFIX,
    W as HOLDOUT_W,
    e2_exclusion_union,
    scan_trace,
)
from src.decision.newsvendor import newsvendor_quantile
from src.sim.des import rolling_csr, simulate_function
from src.sim.simulator import compute_adaptation_lag

RUNS = PROJECT_ROOT / "results" / "runs"
TABLES = PROJECT_ROOT / "results" / "tables"
P19 = PROJECT_ROOT / "data" / "processed_2019"
PHW = PROJECT_ROOT / "data" / "processed_huawei"

ALPHAS = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]
RHOS = [1.0, 10.0, 100.0]
SEEDS = [0, 1, 2]
W = 240
COHORT_CAP = 100
KAPPA = 15.0
BOOT_N = 5000
BOOT_SEED = 260715


def find_azure2019_cohort(counts):
    picked = []
    n, t_len = counts.shape
    for f in range(n):
        row = np.asarray(counts[f])
        nz = np.flatnonzero(row)
        if len(nz) == 0:
            continue
        t0 = int(nz[0])
        if t0 < MIN_IDLE_PREFIX or t0 >= t_len - 1440:
            continue
        if row[t0:t0 + 1440].sum() < MIN_DAY1_INV:
            continue
        picked.append((f, t0))
    rng = np.random.default_rng(42)
    if len(picked) > COHORT_CAP:
        idx = rng.choice(len(picked), COHORT_CAP, replace=False)
        picked = [picked[i] for i in sorted(idx)]
    return picked


def find_huawei_cohort(counts, first_present):
    picked = []
    funnel = {"universe": counts.shape[0], "qualifying": 0}
    for f in range(counts.shape[0]):
        row = np.asarray(counts[f])
        nz = np.flatnonzero(row)
        if len(nz) == 0:
            continue
        t0 = int(nz[0])
        if t0 < MIN_IDLE_PREFIX or t0 >= counts.shape[1] - 1440:
            continue
        if row[t0:t0 + 1440].sum() < MIN_DAY1_INV:
            continue
        picked.append((f, t0))
    funnel["qualifying"] = len(picked)
    funnel["absent_from_trace_before_t0"] = int(
        sum(1 for (f, t0) in picked if first_present[f] >= t0)
    )
    rng = np.random.default_rng(42)
    if len(picked) > COHORT_CAP:
        idx = rng.choice(len(picked), COHORT_CAP, replace=False)
        picked = [picked[i] for i in sorted(idx)]
    funnel["cohort_size"] = len(picked)
    return picked, funnel


def find_validation_holdout(counts):
    first, day1, _ = scan_trace(counts)
    t_len = counts.shape[1]
    union = e2_exclusion_union(first, day1, t_len)
    assert len(union) == 772, f"exclusion union {len(union)} != 772"
    mask = (first >= MIN_IDLE_PREFIX) & (first < t_len - HOLDOUT_W) & (day1 >= MIN_DAY1_INV)
    cand = [int(f) for f in np.flatnonzero(mask)]
    holdout = [f for f in cand if f not in union]
    return [(f, int(first[f])) for f in holdout]


def cohort_arrays(name, pilot=0):
    if name in ("azure2019", "azure2019_validation"):
        counts = np.load(P19 / "counts.npy", mmap_mode="r")
        dur_df = pd.read_csv(P19 / "duration_stats.csv")
        pts = find_azure2019_cohort(counts) if name == "azure2019" else find_validation_holdout(counts)
    elif name == "huawei":
        counts = np.load(PHW / "counts.npy", mmap_mode="r")
        dur_df = pd.read_csv(PHW / "duration_stats.csv")
        first_present = np.load(PHW / "first_present.npy")
        pts, _ = find_huawei_cohort(counts, first_present)
    else:
        raise ValueError(name)
    if pilot:
        pts = pts[:pilot]
    seg_counts = np.stack([np.asarray(counts[f, t0:t0 + W], dtype=np.int64) for f, t0 in pts])
    dm = np.nan_to_num(np.array([dur_df.iloc[f]["dur_mean"] for f, _ in pts]), nan=1.0)
    ds = np.nan_to_num(np.array([dur_df.iloc[f]["dur_std"] for f, _ in pts]), nan=0.5)
    return pts, seg_counts, dm, ds


def run_ewma_alpha(seg_counts, dm, ds, alpha, rho, track_rolling=True):
    rates = rates_ewma(seg_counts, alpha=alpha)
    tau = newsvendor_quantile(rho)
    pw, ka = decisions_from_rates(rates, tau)
    f_n, w_len = seg_counts.shape
    if not track_rolling:
        from src.sim.des_fast import simulate_trace_fast

        seed_rows = [
            simulate_trace_fast(
                seg_counts,
                pw,
                ka,
                dm,
                ds,
                seed=seed,
                cold_mu=COLD_INIT["mu"],
                cold_sigma=COLD_INIT["sigma"],
            )
            for seed in SEEDS
        ]
        inv_per_seed = float(np.mean([r["total_invocations"] for r in seed_rows]))
        cold_per_seed = float(np.mean([r["cold_starts"] for r in seed_rows]))
        func_cold = np.mean([np.asarray(r["func_cold"], dtype=float) for r in seed_rows], axis=0)
        func_wm = np.mean([np.asarray(r["func_wm"], dtype=float) for r in seed_rows], axis=0)
        wm_per_seed = float(func_wm.sum())
        return {
            "alpha": float(alpha),
            "rho": float(rho),
            "n_functions": int(f_n),
            "invocations": inv_per_seed,
            "cold_starts": cold_per_seed,
            "overall_csr": cold_per_seed / max(inv_per_seed, 1e-9),
            "csr_pct": 100.0 * cold_per_seed / max(inv_per_seed, 1e-9),
            "func_cold_all": func_cold.tolist(),
            "func_cold60": [],
            "wm_total": wm_per_seed,
            "wm_per_1k_inv": wm_per_seed / max(inv_per_seed / 1000.0, 1e-9),
            "cost_per_1k_inv": (KAPPA * rho * cold_per_seed + wm_per_seed)
            / max(inv_per_seed / 1000.0, 1e-9),
            "rolling_csr": [],
            "al_0p1_min": None,
            "fast_des_no_rolling": True,
        }

    roll_c = np.zeros(w_len)
    roll_n = np.zeros(w_len)
    func_cold = np.zeros(f_n)
    func_cold60 = np.zeros(f_n)
    func_wm = np.zeros(f_n)
    for f in range(f_n):
        for seed in SEEDS:
            res = simulate_function(
                seg_counts[f],
                pw[f],
                ka[f],
                float(dm[f]),
                float(ds[f]),
                seed=seed * 7919 + f,
                cold_mu=COLD_INIT["mu"],
                cold_sigma=COLD_INIT["sigma"],
                track_rolling=True,
            )
            roll_c += res["roll_cold"]
            roll_n += res["roll_total"]
            func_cold[f] += res["roll_cold"].sum()
            func_cold60[f] += res["roll_cold"][:60].sum()
            func_wm[f] += res["idle_mem_gb_s"]
    func_cold /= len(SEEDS)
    func_cold60 /= len(SEEDS)
    func_wm /= len(SEEDS)
    inv_per_seed = float(seg_counts.sum())
    cold_per_seed = float(func_cold.sum())
    wm_per_seed = float(func_wm.sum())
    curve = rolling_csr(roll_c, roll_n, window=15)
    pairs = [(i, float(x)) for i, x in enumerate(curve) if not np.isnan(x)]
    al = compute_adaptation_lag(pairs, event_tick=0)
    return {
        "alpha": float(alpha),
        "rho": float(rho),
        "n_functions": int(f_n),
        "invocations": inv_per_seed,
        "cold_starts": cold_per_seed,
        "overall_csr": cold_per_seed / max(inv_per_seed, 1e-9),
        "csr_pct": 100.0 * cold_per_seed / max(inv_per_seed, 1e-9),
        "func_cold_all": func_cold.tolist(),
        "func_cold60": func_cold60.tolist(),
        "wm_total": wm_per_seed,
        "wm_per_1k_inv": wm_per_seed / max(inv_per_seed / 1000.0, 1e-9),
        "cost_per_1k_inv": (KAPPA * rho * cold_per_seed + wm_per_seed)
        / max(inv_per_seed / 1000.0, 1e-9),
        "rolling_csr": [None if np.isnan(x) else float(x) for x in curve],
        "al_0p1_min": None if al is None else float(al),
    }


def reference_rows(cohort, seg_counts):
    path = {
        "azure2019": RUNS / "revision_a4_crosstrace.json",
        "huawei": RUNS / "revision_h1_huawei_cohort.json",
    }.get(cohort)
    if path is None or not path.exists():
        return {}
    data = json.load(open(path))
    refs = {}
    inv_per_seed = float(seg_counts.sum())
    for rho in RHOS:
        rkey = str(rho)
        rho_out = data["by_rho"][rkey]
        refs[str(rho)] = {}
        for arm in ["A5_proto", "B4a_ewma", "B1_fixed_keepalive", "Oracle"]:
            if arm not in rho_out:
                continue
            rec = rho_out[arm]
            curve = rec.get("rolling_csr", [])
            pairs = [(i, float(x)) for i, x in enumerate(curve) if x is not None and not np.isnan(x)]
            al = compute_adaptation_lag(pairs, event_tick=0)
            cold = float(rec["overall_csr"]) * inv_per_seed
            wm = float(rec.get("wm_total", np.nan))
            refs[str(rho)][arm] = {
                "overall_csr": float(rec["overall_csr"]),
                "csr_pct": 100.0 * float(rec["overall_csr"]),
                "invocations": inv_per_seed,
                "cold_starts": cold,
                "func_cold_all": rec.get("func_cold_all", []),
                "func_cold60": rec.get("func_cold60", []),
                "wm_total": wm,
                "wm_per_1k_inv": wm / max(inv_per_seed / 1000.0, 1e-9),
                "cost_per_1k_inv": (KAPPA * rho * cold + wm) / max(inv_per_seed / 1000.0, 1e-9),
                "al_0p1_min": None if al is None else float(al),
                "source_json": str(path),
            }
    return refs


def boot_ci(diff):
    if len(diff) == 0:
        return [None, None]
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(diff), size=(BOOT_N, len(diff)))
    means = diff[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def selected_alphas(validation):
    selected = {"by_csr": {}, "by_cost": {}}
    for rho in RHOS:
        rows = validation["by_rho"][str(rho)]["ewma_alpha"]
        by_csr = min(rows, key=lambda r: (rows[r]["csr_pct"], rows[r]["cost_per_1k_inv"], float(r)))
        by_cost = min(rows, key=lambda r: (rows[r]["cost_per_1k_inv"], rows[r]["csr_pct"], float(r)))
        selected["by_csr"][str(rho)] = float(by_csr)
        selected["by_cost"][str(rho)] = float(by_cost)
    return selected


def write_tables(out, prefix):
    TABLES.mkdir(parents=True, exist_ok=True)
    header = (
        "cohort,selection,rho,arm,alpha,csr_pct,wm_per_1k_inv,cost_per_1k_inv,"
        "al_0p1_min,delta_csr_pp_vs_winter,mean_diff_cold_per_fn_vs_winter,"
        "wilcoxon_p_vs_winter,boot_ci95_lo,boot_ci95_hi"
    )
    lines = [header]
    selected = out["selected_alpha"]
    for cohort in ["azure2019", "huawei"]:
        if cohort not in out["cohorts"]:
            continue
        cdat = out["cohorts"][cohort]
        for rho in RHOS:
            rkey = str(rho)
            winter = cdat["references"].get(rkey, {}).get("A5_proto")
            for sel_name, amap in [
                ("default_alpha_0.1", {rkey: 0.1}),
                ("validation_min_csr", selected["by_csr"]),
                ("validation_min_cost", selected["by_cost"]),
                ("expost_min_csr", {}),
            ]:
                alpha = (
                    min(
                        cdat["by_rho"][rkey]["ewma_alpha"],
                        key=lambda a: cdat["by_rho"][rkey]["ewma_alpha"][a]["csr_pct"],
                    )
                    if sel_name == "expost_min_csr"
                    else amap[rkey]
                )
                alpha = float(alpha)
                rec = cdat["by_rho"][rkey]["ewma_alpha"][str(alpha)]
                dcsr = np.nan
                mdiff = np.nan
                wp = np.nan
                ci = [None, None]
                if (
                    winter
                    and winter.get("func_cold_all")
                    and len(winter["func_cold_all"]) == len(rec["func_cold_all"])
                ):
                    diff = np.asarray(rec["func_cold_all"], dtype=float) - np.asarray(winter["func_cold_all"], dtype=float)
                    dcsr = rec["csr_pct"] - winter["csr_pct"]
                    mdiff = float(diff.mean())
                    nz = diff[diff != 0]
                    wp = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else 1.0
                    ci = boot_ci(diff)
                lines.append(
                    f"{cohort},{sel_name},{rho:g},EWMA,{alpha:g},"
                    f"{rec['csr_pct']:.6f},{rec['wm_per_1k_inv']:.3f},"
                    f"{rec['cost_per_1k_inv']:.3f},{rec['al_0p1_min']},"
                    f"{dcsr:.6f},{mdiff:.6f},{wp:.6g},"
                    f"{'' if ci[0] is None else f'{ci[0]:.6f}'},"
                    f"{'' if ci[1] is None else f'{ci[1]:.6f}'}"
                )
            if winter:
                lines.append(
                    f"{cohort},reference,{rho:g},WINTER,A5_proto,"
                    f"{winter['csr_pct']:.6f},{winter['wm_per_1k_inv']:.3f},"
                    f"{winter['cost_per_1k_inv']:.3f},{winter['al_0p1_min']},"
                    ",,,,"
                )
    (TABLES / f"{prefix}.csv").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / f'{prefix}.csv'}")


def write_json_checked(path, out):
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    blob = path.read_bytes()
    assert blob and blob.count(0) == 0
    json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--cohorts", default="azure2019,huawei,azure2019_validation")
    ap.add_argument("--out", default="revision_r22_ewma_alpha_frontier.json")
    ap.add_argument("--table-prefix", default="T_r22_ewma_alpha_frontier")
    args = ap.parse_args()

    path = RUNS / args.out
    if path.exists():
        out = json.load(open(path))
        out.setdefault("cohorts", {})
    else:
        out = {
            "analysis": "R22 post-hoc EWMA alpha frontier audit",
            "alphas": ALPHAS,
            "rhos": RHOS,
            "seeds": SEEDS,
            "window_minutes": W,
            "selection_rule": {
                "by_csr": "minimum validation-cohort CSR, tie by cost then lower alpha",
                "by_cost": "minimum validation-cohort registered cost, tie by CSR then lower alpha",
            },
            "cohorts": {},
        }

    for cohort in args.cohorts.split(","):
        t0 = time.time()
        pts, seg_counts, dm, ds = cohort_arrays(cohort, pilot=args.pilot)
        print(f"### {cohort}: {len(pts)} functions, {int(seg_counts.sum())} invocations", flush=True)
        cdat = out["cohorts"].get(cohort, {
            "n_functions": len(pts),
            "invocations": int(seg_counts.sum()),
            "by_rho": {},
            "references": reference_rows(cohort, seg_counts),
        })
        cdat["n_functions"] = len(pts)
        cdat["invocations"] = int(seg_counts.sum())
        cdat["references"] = reference_rows(cohort, seg_counts)
        for rho in RHOS:
            cdat["by_rho"].setdefault(str(rho), {"ewma_alpha": {}})
            cdat["by_rho"][str(rho)].setdefault("ewma_alpha", {})
            for alpha in ALPHAS:
                if str(alpha) in cdat["by_rho"][str(rho)]["ewma_alpha"]:
                    print(f"  rho={rho:g} alpha={alpha:g} already present; skipping", flush=True)
                    continue
                rec = run_ewma_alpha(
                    seg_counts,
                    dm,
                    ds,
                    alpha,
                    rho,
                    track_rolling=(cohort != "azure2019_validation"),
                )
                cdat["by_rho"][str(rho)]["ewma_alpha"][str(alpha)] = rec
                out["cohorts"][cohort] = cdat
                write_json_checked(path, out)
                print(
                    f"  rho={rho:g} alpha={alpha:g} CSR={rec['csr_pct']:.4f}% "
                    f"WM/1k={rec['wm_per_1k_inv']:.1f} AL={rec['al_0p1_min']}",
                    flush=True,
                )
        cdat["elapsed_sec"] = time.time() - t0
        out["cohorts"][cohort] = cdat
        write_json_checked(path, out)

    if "azure2019_validation" not in out["cohorts"]:
        raise SystemExit("azure2019_validation cohort is required for alpha selection")
    out["selected_alpha"] = selected_alphas(out["cohorts"]["azure2019_validation"])

    write_json_checked(path, out)
    print(f"wrote {path}")
    write_tables(out, args.table_prefix)


if __name__ == "__main__":
    main()
