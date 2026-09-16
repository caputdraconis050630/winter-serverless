#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R20-B: cross-trace 2019 validation for the age-threshold grid.

This reuses the M2-A held-out Azure-2019 48h cohort and cross-trace
2021-trained prototype-biased ridge machinery, but evaluates the dense R20
EWMA-switch age grid and both W-routing families.

The experiment is validation/frontier characterization for the R20 threshold
diagnostic. It does not make the selected threshold confirmatory because the
held-out cohort has already appeared in prior age-gate audits.

Output:
  results/runs/revision_r20_age_threshold_2019_holdout.json
  results/tables/T_r20_2019_holdout_rho10.csv
  results/tables/T_r20_2019_holdout_fullrho.csv
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("SF_DATA_DIR", "processed_2019")
os.environ["SF_RUNS_DIR"] = str(PROJECT_ROOT / "results_azure2021" / "runs")

from scripts.eval_adapt_biased import (  # noqa: E402
    CKPT_RUNS,
    PROCESSED_DIR,
    build_prototypes,
    load_trainer,
)
from scripts.phase6_des import COLD_INIT, decisions_from_rates  # noqa: E402
from scripts.phase63_onboarding_drift import DES_SEEDS  # noqa: E402
from scripts.revision_m2a_holdout_cohort import (  # noqa: E402
    AGE_PRIMARY,
    CKPT_MD5,
    GATE_THRESHOLD,
    MIN_DAY1_INV,
    MIN_IDLE_PREFIX,
    PROC_2021,
    W,
    check_incremental,
    e2_exclusion_union,
    embed_cohort,
    ridge_rates,
    run_arm_seed,
    scan_trace,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402

RUNS = PROJECT_ROOT / "results" / "runs"
TABLES = PROJECT_ROOT / "results" / "tables"
OUT = RUNS / "revision_r20_age_threshold_2019_holdout.json"

RHOS = [1.0, 10.0, 100.0]
AGE_GRID = [0, 30, 60, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, None]
FAMILIES = ["WE", "WL"]
KAPPA = 15.0
PRIMARY_RHO = 10.0


def age_label(age_min):
    return "inf" if age_min is None else str(int(age_min))


def method_name(family, age_min):
    return f"G_{family}_A{age_label(age_min)}"


def result_key(row):
    return row["method"], float(row["rho"]), int(row["seed"])


def dedupe(rows):
    by_key = {}
    for row in rows:
        by_key[result_key(row)] = row
    return [by_key[k] for k in sorted(by_key, key=lambda x: (x[1], x[2], x[0]))]


def load_output():
    if OUT.exists():
        out = json.load(open(OUT))
    else:
        out = {
            "description": "R20 cross-trace 2019 holdout age-threshold validation",
            "created_by": "scripts/revision_r20_age_threshold_2019_holdout.py",
            "config": "2021-trained body+prototypes -> held-out Azure 2019 48h cohort",
            "rhos": RHOS,
            "age_grid_minutes": [age_label(a) for a in AGE_GRID],
            "families": {
                "WE": "W=EWMA, learned after K>=100 until age A",
                "WL": "W=learned, learned after K>=100 until age A",
            },
            "aliases": {
                method_name("WE", 720): "current age-qualified WINTER-G",
                method_name("WE", None): "v3",
                method_name("WL", 0): "v4",
            },
            "results": [],
            "routing": {},
        }
    out["results"] = dedupe(out.get("results", []))
    return out


def save(out):
    out["results"] = dedupe(out.get("results", []))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    json.load(open(OUT))


def compose_gate_rates(proto, ewma, learned, zero_h, w_mask, conv, age, family, age_min):
    rates = np.empty_like(ewma, dtype=np.float32)
    rates[zero_h] = proto[zero_h]
    rates[w_mask] = learned[w_mask] if family == "WL" else ewma[w_mask]
    if age_min is None:
        learned_conv = conv
    else:
        learned_conv = conv & (age < int(age_min))
    ewma_conv = conv & ~learned_conv
    rates[learned_conv] = learned[learned_conv]
    rates[ewma_conv] = ewma[ewma_conv]
    return rates


def make_ewma(seg_counts):
    f_n, w_len = seg_counts.shape
    ew = np.zeros(f_n)
    rates = np.zeros((f_n, w_len), dtype=np.float32)
    for w in range(w_len):
        rates[:, w] = np.expm1(np.maximum(ew, 0))
        ew = 0.1 * np.log1p(seg_counts[:, w].astype(np.float64)) + 0.9 * ew
    return rates


def summarize(rows):
    if not rows:
        return None
    inv = np.array([r["inv"] for r in rows], dtype=np.float64)
    cold = np.array([r["cold"] for r in rows], dtype=np.float64)
    wm_total = np.array([np.sum(r["func_wm"]) for r in rows], dtype=np.float64)
    rho = float(rows[0]["rho"])
    csr = cold / np.maximum(inv, 1e-10)
    wm_per_1k = wm_total / np.maximum(inv / 1000.0, 1e-10)
    cost_per_1k = (KAPPA * rho * cold + wm_total) / np.maximum(inv / 1000.0, 1e-10)
    return {
        "n": int(len(rows)),
        "invocations": float(inv.mean()),
        "cold_starts": float(cold.mean()),
        "csr_pct": float(100.0 * csr.mean()),
        "csr_std_pct": float(100.0 * csr.std()),
        "wm_per_1k_inv": float(wm_per_1k.mean()),
        "cost_per_1k_inv": float(cost_per_1k.mean()),
        "cost_std_per_1k_inv": float(cost_per_1k.std()),
    }


def method_meta(method):
    if method in ("B4a_ewma", "A5_proto"):
        return "", "", method
    parts = method.split("_")
    family = parts[1]
    age = parts[2][1:]
    alias = {
        ("WE", "720"): "current_WINTRG",
        ("WE", "inf"): "v3",
        ("WL", "0"): "v4",
    }.get((family, age), "")
    return family, age, alias


def write_tables(out):
    TABLES.mkdir(parents=True, exist_ok=True)
    header = (
        "method,family,age_min,alias,rho,csr_pct,csr_std_pct,wm_per_1k_inv,"
        "cost_per_1k_inv,delta_cost_per_1k_vs_ewma,delta_csr_pp_vs_ewma,n"
    )
    full = [header]
    rho10 = [header]
    methods = ["B4a_ewma", "A5_proto"] + [
        method_name(family, age) for family in FAMILIES for age in AGE_GRID
    ]
    for rho in RHOS:
        ref = summarize([r for r in out["results"] if r["method"] == "B4a_ewma" and r["rho"] == rho])
        for method in methods:
            rec = summarize([r for r in out["results"] if r["method"] == method and r["rho"] == rho])
            if not rec:
                continue
            family, age, alias = method_meta(method)
            dcost = rec["cost_per_1k_inv"] - ref["cost_per_1k_inv"] if ref else np.nan
            dcsr = rec["csr_pct"] - ref["csr_pct"] if ref else np.nan
            line = (
                f"{method},{family},{age},{alias},{rho:g},"
                f"{rec['csr_pct']:.6f},{rec['csr_std_pct']:.6f},"
                f"{rec['wm_per_1k_inv']:.3f},{rec['cost_per_1k_inv']:.3f},"
                f"{dcost:.3f},{dcsr:.6f},{rec['n']}"
            )
            full.append(line)
            if abs(rho - PRIMARY_RHO) < 1e-9:
                rho10.append(line)
    (TABLES / "T_r20_2019_holdout_fullrho.csv").write_text("\n".join(full) + "\n")
    (TABLES / "T_r20_2019_holdout_rho10.csv").write_text("\n".join(rho10) + "\n")


def print_primary(out):
    print("\n=== R20 2019 holdout rho=10 summary ===")
    rows = out["results"]
    ref = summarize([r for r in rows if r["method"] == "B4a_ewma" and abs(float(r["rho"]) - 10.0) < 1e-9])
    ranked = []
    for method in ["A5_proto"] + [method_name(f, a) for f in FAMILIES for a in AGE_GRID]:
        rec = summarize([r for r in rows if r["method"] == method and abs(float(r["rho"]) - 10.0) < 1e-9])
        if rec:
            ranked.append((rec["cost_per_1k_inv"], method, rec))
    ranked.sort(key=lambda x: x[0])
    for cost, method, rec in ranked[:12]:
        print(
            f"  {method:12s} CSR={rec['csr_pct']:.4f}% "
            f"dCSR={rec['csr_pct'] - ref['csr_pct']:+.4f}pp "
            f"WM={rec['wm_per_1k_inv']:.1f} C/1k={cost:.1f} "
            f"dC={cost - ref['cost_per_1k_inv']:+.1f}"
        )
    cur = summarize([r for r in rows if r["method"] == method_name("WE", 720) and abs(float(r["rho"]) - 10.0) < 1e-9])
    if cur:
        print(
            f"  current A720: CSR={cur['csr_pct']:.4f}% "
            f"WM={cur['wm_per_1k_inv']:.1f} C/1k={cur['cost_per_1k_inv']:.1f} "
            f"dC={cur['cost_per_1k_inv'] - ref['cost_per_1k_inv']:+.1f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--export-jobs", default="",
                    help="Write decision matrices for scripts/des_runner_fast.py and exit")
    args = ap.parse_args()

    global OUT
    if args.pilot:
        OUT = RUNS / f"revision_r20_age_threshold_2019_holdout_pilot{args.pilot}.json"

    out = load_output()
    save(out)

    ck = CKPT_RUNS / "best_anil_ridge_s2_s0.pt"
    md5 = hashlib.md5(open(ck, "rb").read()).hexdigest()
    assert md5 == CKPT_MD5, f"checkpoint md5 {md5} != registered {CKPT_MD5}"
    out["checkpoint"] = str(ck)
    out["checkpoint_md5"] = md5
    print(f"checkpoint OK ({ck}, md5 {md5})", flush=True)

    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    nF = features.shape[2]
    T = counts.shape[1]

    cache = Path("/tmp/claude-1000/-data-260715/d735e123-b9b7-473b-8a34-6b51820575ec/scratchpad/c2019.npz")
    if cache.exists():
        z = np.load(cache)
        first, day1, t100 = z["first"], z["day1"], z["t100"]
    else:
        first, day1, t100 = scan_trace(counts)

    union = e2_exclusion_union(first, day1, T)
    assert len(union) == 772, f"exclusion union {len(union)} != 772"
    mask = (first >= MIN_IDLE_PREFIX) & (first < T - W) & (day1 >= MIN_DAY1_INV)
    cand = [int(f) for f in np.flatnonzero(mask)]
    holdout = [f for f in cand if f not in union]
    if args.pilot:
        rng = np.random.default_rng(42)
        holdout = sorted(rng.choice(holdout, args.pilot, replace=False).tolist())
    pts = [(f, int(first[f])) for f in holdout]
    F_n = len(pts)
    print(f"candidates {len(cand)} | holdout {F_n} | pilot={bool(args.pilot)}", flush=True)

    trainer, biased_head = load_trainer(nF, seed=0)
    feat21 = np.load(PROC_2021 / "features.npy", mmap_mode="r")
    cnt21 = np.load(PROC_2021 / "counts.npy", mmap_mode="r")
    spl21 = np.load(PROC_2021 / "splits.npz")
    centroids, proto_w = build_prototypes(trainer, biased_head, feat21, cnt21, spl21["s2_train"])

    t0 = time.time()
    phi = embed_cohort(features, pts, trainer.body, nF)
    print(f"embed stage {time.time() - t0:.0f}s", flush=True)

    seg_counts = np.stack([np.asarray(counts[fi, t0:t0 + W], dtype=np.int64) for (fi, t0) in pts])
    y = torch.from_numpy(np.log1p(seg_counts.astype(np.float32)))
    lam = trainer.head.ridge_lambda.item()
    da, db, dr = check_incremental(phi, y, lam, F_n)
    assert da < 1e-4 and db < 1e-4 and dr < 1e-4, "incremental ridge check failed"
    out["incremental_check"] = {"rel_dA": da, "rel_db": db, "max_rate_diff": dr}

    t0 = time.time()
    r_learn, r_zero, assign0 = ridge_rates(phi, y, centroids, proto_w, lam, F_n)
    print(f"ridge stage {time.time() - t0:.0f}s", flush=True)
    del phi
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ewma = make_ewma(seg_counts)
    cum = np.zeros((F_n, W), dtype=np.int64)
    cum[:, 1:] = np.cumsum(seg_counts[:, :-1], axis=1)
    age = np.tile(np.arange(W, dtype=np.int64), (F_n, 1))
    zero_h = cum == 0
    w_mask = (cum > 0) & (cum < GATE_THRESHOLD)
    conv = cum >= GATE_THRESHOLD
    out["n_candidates"] = len(cand)
    out["n_exclusion_union"] = len(union)
    out["n_functions"] = F_n
    out["horizon_ticks"] = W
    out["gate_threshold"] = GATE_THRESHOLD
    out["age_primary_previous"] = AGE_PRIMARY
    out["routing"] = {
        "zero_history": float(zero_h.mean()),
        "midband_lt100": float(w_mask.mean()),
        "converged_ge100": float(conv.mean()),
        "assign_hist_k0": np.bincount(assign0, minlength=16).tolist(),
    }
    for family in FAMILIES:
        for age_min in AGE_GRID:
            if age_min is None:
                young = conv
            else:
                young = conv & (age < int(age_min))
            learned_mask = young.copy()
            if family == "WL":
                learned_mask |= w_mask
            out["routing"][f"{method_name(family, age_min)}_learned_fraction"] = float(
                learned_mask.mean()
            )
    save(out)

    rate_mats = {
        "B4a_ewma": ewma,
        "A5_proto": r_learn,
    }
    for family in FAMILIES:
        for age_min in AGE_GRID:
            rate_mats[method_name(family, age_min)] = compose_gate_rates(
                r_zero, ewma, r_learn, zero_h, w_mask, conv, age, family, age_min
            )

    seg_counts = seg_counts.astype(np.int32)
    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] for (fi, _) in pts]), nan=1.0)
    ds = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] for (fi, _) in pts]), nan=0.5)

    if args.export_jobs:
        jdir = Path(args.export_jobs) / "azure2019_holdout"
        jdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            jdir / "shared.npz",
            counts=seg_counts.astype(np.int64),
            dur_means=dm,
            dur_stds=ds,
        )
        job_meta = []
        for rho in RHOS:
            tau = newsvendor_quantile(rho)
            for method, rates in rate_mats.items():
                pw, ka = decisions_from_rates(rates, tau)
                fname = f"{method}__rho{rho:g}.npz"
                np.savez_compressed(jdir / fname, prewarm=pw, keepalive=ka)
                job_meta.append({"method": method, "rho": float(rho), "file": fname})
        with open(jdir / "jobs.json", "w") as f:
            json.dump(
                {
                    "split": "azure2019_holdout",
                    "seeds": list(DES_SEEDS),
                    "cold_init": COLD_INIT,
                    "jobs": job_meta,
                },
                f,
                indent=1,
            )
        out["export_jobs"] = str(jdir)
        out["n_export_jobs"] = len(job_meta)
        save(out)
        print(f"exported {len(job_meta)} jobs to {jdir}", flush=True)
        return

    done = {result_key(r) for r in out["results"]}
    methods = list(rate_mats)
    for rho in RHOS:
        tau = newsvendor_quantile(rho)
        tasks = []
        for method in methods:
            pw, ka = decisions_from_rates(rate_mats[method], tau)
            for seed in DES_SEEDS:
                if (method, float(rho), int(seed)) in done:
                    continue
                tasks.append((method, float(rho), int(seed), seg_counts, pw, ka, dm, ds))
        if not tasks:
            print(f"rho={rho:g}: all complete", flush=True)
            continue
        print(f"rho={rho:g}: running {len(tasks)} arm-seed tasks on {args.workers} workers", flush=True)
        t0 = time.time()
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for arm, rrho, res in ex.map(run_arm_seed, tasks):
                row = {
                    "method": arm,
                    "rho": float(rrho),
                    "seed": int(tasks[completed][2]),
                    "cold": float(res["cold"]),
                    "inv": float(res["inv"]),
                    "pre": float(res["pre"]),
                    "post": float(res["post"]),
                    "func_cold": res["func_cold"].tolist(),
                    "func_wm": res["func_wm"].tolist(),
                }
                out["results"].append(row)
                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    save(out)
                    print(f"  [{completed}/{len(tasks)}] rho={rrho:g} {arm} ({time.time() - t0:.0f}s)", flush=True)
        save(out)

    write_tables(out)
    save(out)
    print_primary(out)
    print(f"wrote {OUT}")
    print(f"wrote {TABLES / 'T_r20_2019_holdout_rho10.csv'}")
    print(f"wrote {TABLES / 'T_r20_2019_holdout_fullrho.csv'}")


if __name__ == "__main__":
    main()
