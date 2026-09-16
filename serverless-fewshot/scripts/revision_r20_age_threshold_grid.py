#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R20: faithful EWMA-switch age-threshold grid for the Azure-2021 gates.

This diagnostic reruns the faithful Section-3 WINTER head in the active-pool
steady-window protocol while sweeping the wall-clock age A at which the gate
hands back to EWMA after the 100-invocation threshold.

Families:
  W=E: current WINTER-G family. Prototype on zero history, EWMA for
       1<=K<100, learned for K>=100 and age<A, EWMA for age>=A.
  W=L: joint W/A family. Same A rule after K>=100, but learned is also used
       in the pre-count-threshold W state. A=0 reproduces v4.

A=720 in W=E is the current age-qualified WINTER-G. A=inf in W=E is v3.
A=0 in W=L is v4. The selection readout uses registered cost at rho=10.

Outputs:
  results/runs/revision_r20_age_threshold_grid.json
  results/tables/T_r20_age_grid_azure2021_rho10.csv
  results/tables/T_r20_age_frontier_fullrho.csv
  results/tables/T_r20_selection_summary.csv
  results/tables/T_r20_reconciliation.csv
  results/tables/T_r20_routing_exposure.csv
"""

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import (  # noqa: E402
    COLD_INIT,
    FULL_RHOS,
    decisions_from_rates,
    rates_ewma,
    run_config,
)
from scripts.phase63_onboarding_drift import build_prototypes  # noqa: E402
from scripts.revision_a2_des import rates_proto_prefix  # noqa: E402
from scripts.revision_r11_faithful import rates_a5_faithful  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402

torch.set_num_threads(1)

P21 = PROJECT_ROOT / "data" / "processed"
RUNS = PROJECT_ROOT / "results" / "runs"
TABLES = PROJECT_ROOT / "results" / "tables"
OUT = RUNS / "revision_r20_age_threshold_grid.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = list(range(10))
LOW_RHOS = [0.01, 0.02, 0.05]
RHOS = sorted({float(r) for r in list(FULL_RHOS) + LOW_RHOS})
MIN_INV = 100
AGE_GRID = [0, 30, 60, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, None]
KAPPA = 15.0
PRIMARY_RHO = 10.0
SPLITS = ["S1", "S2", "S3"]
FAMILIES = ["WE", "WL"]


def age_label(age_min):
    return "inf" if age_min is None else str(int(age_min))


def age_value_for_sort(label):
    return math.inf if str(label) == "inf" else float(label)


def method_name(family, age_min):
    return f"G_{family}_A{age_label(age_min)}"


def result_key(row):
    return (
        row["method"],
        row["split"],
        float(row["cost_ratio"]),
        int(row["seed"]),
    )


def dedupe(rows):
    by_key = {}
    for row in rows:
        by_key[result_key(row)] = row
    return [by_key[k] for k in sorted(by_key, key=lambda x: (x[1], x[2], x[3], x[0]))]


def load_output():
    if OUT.exists():
        out = json.load(open(OUT))
    else:
        out = {
            "description": "Faithful Azure-2021 EWMA-switch age-threshold grid",
            "created_by": "scripts/revision_r20_age_threshold_grid.py",
            "seeds": SEEDS,
            "rhos": RHOS,
            "primary_rho": PRIMARY_RHO,
            "min_invocations": MIN_INV,
            "age_grid_minutes": [age_label(a) for a in AGE_GRID],
            "families": {
                "WE": "W=EWMA, learned only after K>=100 until age A",
                "WL": "W=learned, learned after K>=100 until age A",
            },
            "aliases": {
                method_name("WE", 720): "current age-qualified WINTER-G",
                method_name("WE", None): "v3",
                method_name("WL", 0): "v4",
            },
            "results": [],
            "routing": {},
            "reconciliation": {},
        }
    out["results"] = dedupe(out.get("results", []))
    out.setdefault("routing", {})
    out.setdefault("reconciliation", {})
    return out


def save(out):
    out["results"] = dedupe(out.get("results", []))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    json.load(open(OUT))


def split_arrays(features, counts_all, splits, dur_df, split):
    key = {"S1": "s1_test", "S2": "s2_test", "S3": "s3_test"}[split]
    test_idx = splits[key]
    counts = counts_all[test_idx]
    feats = features[test_idx]
    if split == "S3":
        t0 = int(splits.get("s3_test_t_start", [10080])[0])
        counts = counts[:, t0:]
        feats = feats[:, t0:, :]
    dm = np.nan_to_num(
        np.array(
            [
                dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
                for fi in test_idx
            ]
        ),
        nan=1.0,
    )
    ds = np.nan_to_num(
        np.array(
            [
                dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
                for fi in test_idx
            ]
        ),
        nan=0.5,
    )
    return test_idx, counts, feats, dm, ds


def age_matrix(counts):
    n, t_len = counts.shape
    age = np.zeros((n, t_len), dtype=np.int64)
    for f in range(n):
        nz = np.flatnonzero(counts[f])
        if len(nz):
            t0 = int(nz[0])
            age[f, t0:] = np.arange(t_len - t0)
    return age


def compose_gate_rates(proto, ewma, faithful, zero, w_mask, conv_mask, age, family, age_min):
    rates = np.empty_like(ewma, dtype=np.float32)
    rates[zero] = proto[zero]
    rates[w_mask] = faithful[w_mask] if family == "WL" else ewma[w_mask]

    if age_min is None:
        learned_conv = conv_mask
    else:
        learned_conv = conv_mask & (age < int(age_min))
    ewma_conv = conv_mask & ~learned_conv
    rates[learned_conv] = faithful[learned_conv]
    rates[ewma_conv] = ewma[ewma_conv]
    return rates


def routing_exposure(counts, cum_prev, age):
    zero = cum_prev == 0
    w_mask = (cum_prev > 0) & (cum_prev < MIN_INV)
    conv = cum_prev >= MIN_INV
    total_ticks = counts.size
    total_inv = int(counts.sum())
    rows = [
        {
            "region": "Z0",
            "condition": "K=0",
            "function_minutes": int(zero.sum()),
            "share_function_minutes": float(zero.mean()),
            "invocations": int(counts[zero].sum()),
            "share_invocations": float(counts[zero].sum() / max(total_inv, 1)),
        },
        {
            "region": "W",
            "condition": "1<=K<100",
            "function_minutes": int(w_mask.sum()),
            "share_function_minutes": float(w_mask.sum() / max(total_ticks, 1)),
            "invocations": int(counts[w_mask].sum()),
            "share_invocations": float(counts[w_mask].sum() / max(total_inv, 1)),
        },
    ]
    bands = [
        (0, 30),
        (30, 60),
        (60, 120),
        (120, 180),
        (180, 240),
        (240, 360),
        (360, 480),
        (480, 720),
        (720, 960),
        (960, 1440),
        (1440, 2880),
        (2880, None),
    ]
    for lo, hi in bands:
        if hi is None:
            mask = conv & (age >= lo)
            label = f"K>=100_age>={lo}"
            cond = f"K>=100, age>={lo}"
        else:
            mask = conv & (age >= lo) & (age < hi)
            label = f"K>=100_age{lo}_{hi}"
            cond = f"K>=100, {lo}<=age<{hi}"
        rows.append(
            {
                "region": label,
                "condition": cond,
                "function_minutes": int(mask.sum()),
                "share_function_minutes": float(mask.sum() / max(total_ticks, 1)),
                "invocations": int(counts[mask].sum()),
                "share_invocations": float(counts[mask].sum() / max(total_inv, 1)),
            }
        )
    return rows


def mean_cell(rows):
    if not rows:
        return None
    total_inv = np.array([r["total_invocations"] for r in rows], dtype=np.float64)
    cold = np.array([r["cold_starts"] for r in rows], dtype=np.float64)
    csr = np.array([r["csr"] for r in rows], dtype=np.float64)
    wm = np.array([r["wm_per_1k_inv"] for r in rows], dtype=np.float64)
    cost_total = np.array(
        [KAPPA * float(r["cost_ratio"]) * r["cold_starts"] + r["wm_total_gb_s"] for r in rows],
        dtype=np.float64,
    )
    cost_per_1k = cost_total / np.maximum(total_inv / 1000.0, 1e-10)
    return {
        "n": int(len(rows)),
        "total_invocations": float(total_inv.mean()),
        "cold_starts": float(cold.mean()),
        "csr_pct": float(csr.mean() * 100.0),
        "csr_std_pct": float(csr.std() * 100.0),
        "wm_per_1k_inv": float(wm.mean()),
        "cost_per_1k_inv": float(cost_per_1k.mean()),
        "cost_std_per_1k_inv": float(cost_per_1k.std()),
    }


def method_meta(method):
    if method in ("B4a_ewma", "A5_faithful"):
        return {"family": "", "age_min": "", "alias": method}
    # G_WE_A720 -> family WE, age 720
    parts = method.split("_")
    family = parts[1]
    age = parts[2][1:]
    alias = {
        ("WE", "720"): "current_WINTRG",
        ("WE", "inf"): "v3",
        ("WL", "0"): "v4",
    }.get((family, age), "")
    return {"family": family, "age_min": age, "alias": alias}


def readout(out):
    rows = out["results"]
    methods = ["B4a_ewma", "A5_faithful"] + [
        method_name(family, age) for family in FAMILIES for age in AGE_GRID
    ]
    summary = {}
    for split in SPLITS:
        summary[split] = {}
        for rho in RHOS:
            summary[split][f"{rho:g}"] = {}
            for method in methods:
                cell = [
                    r
                    for r in rows
                    if r["split"] == split
                    and r["method"] == method
                    and abs(float(r["cost_ratio"]) - rho) < 1e-9
                ]
                rec = mean_cell(cell)
                if rec:
                    summary[split][f"{rho:g}"][method] = rec
    out["readout"] = summary


def write_tables(out):
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = out["results"]
    methods = ["B4a_ewma", "A5_faithful"] + [
        method_name(family, age) for family in FAMILIES for age in AGE_GRID
    ]

    def table_line(split, rho, method, ref):
        rec = mean_cell(
            [
                r
                for r in rows
                if r["split"] == split
                and r["method"] == method
                and abs(float(r["cost_ratio"]) - rho) < 1e-9
            ]
        )
        if not rec:
            return None
        meta = method_meta(method)
        dcost = rec["cost_per_1k_inv"] - ref["cost_per_1k_inv"] if ref else np.nan
        dcsr = rec["csr_pct"] - ref["csr_pct"] if ref else np.nan
        return (
            f"{split},{method},{meta['family']},{meta['age_min']},{meta['alias']},"
            f"{rho:g},{rec['csr_pct']:.6f},{rec['csr_std_pct']:.6f},"
            f"{rec['wm_per_1k_inv']:.3f},{rec['cost_per_1k_inv']:.3f},"
            f"{dcost:.3f},{dcsr:.6f},{rec['n']}"
        )

    header = (
        "split,method,family,age_min,alias,rho,csr_pct,csr_std_pct,"
        "wm_per_1k_inv,cost_per_1k_inv,delta_cost_per_1k_vs_ewma,"
        "delta_csr_pp_vs_ewma,n"
    )
    rho10_lines = [header]
    full_lines = [header]
    for split in SPLITS:
        for rho in RHOS:
            ref = mean_cell(
                [
                    r
                    for r in rows
                    if r["split"] == split
                    and r["method"] == "B4a_ewma"
                    and abs(float(r["cost_ratio"]) - rho) < 1e-9
                ]
            )
            for method in methods:
                line = table_line(split, rho, method, ref)
                if line is None:
                    continue
                full_lines.append(line)
                if abs(rho - PRIMARY_RHO) < 1e-9:
                    rho10_lines.append(line)
    (TABLES / "T_r20_age_grid_azure2021_rho10.csv").write_text("\n".join(rho10_lines) + "\n")
    (TABLES / "T_r20_age_frontier_fullrho.csv").write_text("\n".join(full_lines) + "\n")

    # Selection summary at rho=10.
    sel_header = (
        "rank,method,family,age_min,alias,mean_delta_cost_per_1k_vs_ewma,"
        "max_delta_cost_per_1k_vs_ewma,mean_delta_csr_pp_vs_ewma,"
        "max_delta_csr_pp_vs_ewma,splits_with_lower_cost"
    )
    candidates = [m for m in methods if m not in ("B4a_ewma", "A5_faithful")]
    scored = []
    for method in candidates:
        deltas_cost = []
        deltas_csr = []
        lower = 0
        complete = True
        for split in SPLITS:
            ref = mean_cell(
                [
                    r
                    for r in rows
                    if r["split"] == split
                    and r["method"] == "B4a_ewma"
                    and abs(float(r["cost_ratio"]) - PRIMARY_RHO) < 1e-9
                ]
            )
            rec = mean_cell(
                [
                    r
                    for r in rows
                    if r["split"] == split
                    and r["method"] == method
                    and abs(float(r["cost_ratio"]) - PRIMARY_RHO) < 1e-9
                ]
            )
            if not ref or not rec:
                complete = False
                break
            dc = rec["cost_per_1k_inv"] - ref["cost_per_1k_inv"]
            ds = rec["csr_pct"] - ref["csr_pct"]
            deltas_cost.append(dc)
            deltas_csr.append(ds)
            if dc < 0:
                lower += 1
        if complete:
            scored.append(
                {
                    "method": method,
                    "mean_dc": float(np.mean(deltas_cost)),
                    "max_dc": float(np.max(deltas_cost)),
                    "mean_dcsr": float(np.mean(deltas_csr)),
                    "max_dcsr": float(np.max(deltas_csr)),
                    "lower": lower,
                }
            )
    scored.sort(key=lambda r: (-r["lower"], r["mean_dc"], r["max_dc"], r["max_dcsr"]))
    sel_lines = [sel_header]
    for rank, row in enumerate(scored, 1):
        meta = method_meta(row["method"])
        sel_lines.append(
            f"{rank},{row['method']},{meta['family']},{meta['age_min']},{meta['alias']},"
            f"{row['mean_dc']:.3f},{row['max_dc']:.3f},"
            f"{row['mean_dcsr']:.6f},{row['max_dcsr']:.6f},{row['lower']}"
        )
    (TABLES / "T_r20_selection_summary.csv").write_text("\n".join(sel_lines) + "\n")

    # Routing exposure is deterministic and split-level.
    route_lines = [
        "split,region,condition,function_minutes,share_function_minutes,invocations,share_invocations"
    ]
    for split in SPLITS:
        for rec in out["routing"][split]["exposure"]:
            route_lines.append(
                f"{split},{rec['region']},{rec['condition']},"
                f"{rec['function_minutes']},{rec['share_function_minutes']:.8f},"
                f"{rec['invocations']},{rec['share_invocations']:.8f}"
            )
    (TABLES / "T_r20_routing_exposure.csv").write_text("\n".join(route_lines) + "\n")

    # Reconciliation table against prior faithful experiments where labels match.
    rec_lines = [
        "mapped_method,split,rho,n,max_abs_cold_diff,max_abs_total_diff,max_abs_wm_diff_gbs,max_abs_csr_diff"
    ]
    for key, val in sorted(out.get("reconciliation", {}).items()):
        method, split, rho = key.split("|")
        rec_lines.append(
            f"{method},{split},{rho},{val['n']},{val['max_abs_cold_diff']},"
            f"{val['max_abs_total_diff']},{val['max_abs_wm_diff_gbs']:.9f},"
            f"{val['max_abs_csr_diff']:.12f}"
        )
    (TABLES / "T_r20_reconciliation.csv").write_text("\n".join(rec_lines) + "\n")


def load_expected_rows():
    rows = []
    for name in [
        "sim_results_des_revision.json",
        "revision_r16_faithful_primary_experiment.json",
        "revision_r17_faithful_gates_2021.json",
    ]:
        path = RUNS / name
        if not path.exists():
            continue
        data = json.load(open(path))
        rows.extend(data.get("results", data if isinstance(data, list) else []))
    return rows


def reconcile(out):
    expected = load_expected_rows()
    exp_by = {
        (r["method"], r["split"], float(r["cost_ratio"]), int(r["seed"])): r
        for r in expected
    }
    method_map = {
        "B4a_ewma": "B4a_ewma",
        "A5_faithful": "A5_faithful",
        method_name("WE", 720): "A5_gated_aq_faithful",
        method_name("WE", None): "A5_gated_v3_faithful",
        method_name("WL", 0): "A5_gated_v4_faithful",
    }
    checks = {}
    for ours in out["results"]:
        mapped = method_map.get(ours["method"])
        if not mapped:
            continue
        key = (mapped, ours["split"], float(ours["cost_ratio"]), int(ours["seed"]))
        exp = exp_by.get(key)
        if not exp:
            continue
        ckey = f"{ours['method']}|{ours['split']}|{ours['cost_ratio']:g}"
        rec = checks.setdefault(
            ckey,
            {
                "n": 0,
                "max_abs_cold_diff": 0,
                "max_abs_total_diff": 0,
                "max_abs_wm_diff_gbs": 0.0,
                "max_abs_csr_diff": 0.0,
            },
        )
        rec["n"] += 1
        rec["max_abs_cold_diff"] = max(
            rec["max_abs_cold_diff"],
            abs(int(ours["cold_starts"]) - int(exp["cold_starts"])),
        )
        rec["max_abs_total_diff"] = max(
            rec["max_abs_total_diff"],
            abs(int(ours["total_invocations"]) - int(exp["total_invocations"])),
        )
        rec["max_abs_wm_diff_gbs"] = max(
            rec["max_abs_wm_diff_gbs"],
            abs(float(ours["wm_total_gb_s"]) - float(exp["wm_total_gb_s"])),
        )
        rec["max_abs_csr_diff"] = max(
            rec["max_abs_csr_diff"],
            abs(float(ours["csr"]) - float(exp["csr"])),
        )
    out["reconciliation"] = checks


def print_primary_summary(out):
    print("\n=== R20 primary rho=10 summary ===")
    rows = out["results"]
    for split in SPLITS:
        print(f"\n{split}")
        ref = mean_cell(
            [
                r
                for r in rows
                if r["split"] == split
                and r["method"] == "B4a_ewma"
                and abs(float(r["cost_ratio"]) - PRIMARY_RHO) < 1e-9
            ]
        )
        ranked = []
        for family in FAMILIES:
            for age in AGE_GRID:
                method = method_name(family, age)
                rec = mean_cell(
                    [
                        r
                        for r in rows
                        if r["split"] == split
                        and r["method"] == method
                        and abs(float(r["cost_ratio"]) - PRIMARY_RHO) < 1e-9
                    ]
                )
                if rec:
                    ranked.append((rec["cost_per_1k_inv"], method, rec))
        ranked.sort(key=lambda x: x[0])
        for cost, method, rec in ranked[:8]:
            dc = cost - ref["cost_per_1k_inv"]
            ds = rec["csr_pct"] - ref["csr_pct"]
            print(
                f"  {method:12s} CSR={rec['csr_pct']:.4f}% "
                f"dCSR={ds:+.4f}pp WM={rec['wm_per_1k_inv']:.1f} "
                f"C/1k={cost:.1f} dC={dc:+.1f}"
            )
        cur = mean_cell(
            [
                r
                for r in rows
                if r["split"] == split
                and r["method"] == method_name("WE", 720)
                and abs(float(r["cost_ratio"]) - PRIMARY_RHO) < 1e-9
            ]
        )
        if cur:
            print(
                f"  current A720: CSR={cur['csr_pct']:.4f}% "
                f"WM={cur['wm_per_1k_inv']:.1f} "
                f"C/1k={cur['cost_per_1k_inv']:.1f} "
                f"dC={cur['cost_per_1k_inv'] - ref['cost_per_1k_inv']:+.1f}"
            )


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    out = load_output()
    save(out)

    features = np.load(P21 / "features.npy")
    counts_all = np.load(P21 / "counts.npy")
    splits = np.load(P21 / "splits.npz")
    dur_df = pd.read_csv(P21 / "duration_stats.csv")

    trainer = ANILMetaTrainer(
        body_type="tcn",
        head_type="ridge",
        in_features=features.shape[2],
        embedding_dim=64,
        n_quantiles=N_QUANTILES,
        n_horizons=1,
        device=DEVICE,
    )
    ckpt = RUNS / "best_anil_ridge_s1_s0.pt"
    trainer.load(ckpt)
    pm = build_prototypes(trainer, features, counts_all, splits["s1_train"])
    out["checkpoint"] = str(ckpt)
    out["meta_lambda"] = float(trainer.head.ridge_lambda.item())
    save(out)

    all_methods = ["B4a_ewma", "A5_faithful"] + [
        method_name(family, age) for family in FAMILIES for age in AGE_GRID
    ]

    for split in SPLITS:
        test_idx, counts, feats, dm, ds = split_arrays(features, counts_all, splits, dur_df, split)
        print(f"### {split}: {len(test_idx)} funcs, {counts.shape[1]} ticks", flush=True)
        t0 = time.time()
        faithful = rates_a5_faithful(counts.astype(np.float32), feats, trainer, pm)
        ewma = rates_ewma(counts)
        proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
        age = age_matrix(counts)
        zero = cum_prev == 0
        w_mask = (cum_prev > 0) & (cum_prev < MIN_INV)
        conv = cum_prev >= MIN_INV
        out["routing"][split] = {
            "n_functions": int(counts.shape[0]),
            "n_ticks": int(counts.shape[1]),
            "total_invocations": int(counts.sum()),
            "exposure": routing_exposure(counts, cum_prev, age),
            "note": "K and age are evaluation-window-local and computed before tick t.",
        }
        rate_mats = {
            "B4a_ewma": ewma,
            "A5_faithful": faithful,
        }
        for family in FAMILIES:
            for age_min in AGE_GRID:
                rate_mats[method_name(family, age_min)] = compose_gate_rates(
                    proto, ewma, faithful, zero, w_mask, conv, age, family, age_min
                )
        print(f"  rates/gates done in {time.time() - t0:.0f}s", flush=True)
        save(out)

        done = {result_key(r) for r in out["results"]}
        for rho in RHOS:
            tau = newsvendor_quantile(rho)
            jobs = []
            for method in all_methods:
                seeds = [
                    seed
                    for seed in SEEDS
                    if (method, split, float(rho), seed) not in done
                ]
                if not seeds:
                    continue
                pw, ka = decisions_from_rates(rate_mats[method], tau)
                jobs.append((method, float(rho), seeds, counts, pw, ka, dm, ds, COLD_INIT, split))
            if not jobs:
                print(f"  rho={rho:g}: all complete", flush=True)
                continue
            print(f"  rho={rho:g}: running {len(jobs)} configs on 8 workers", flush=True)
            with ProcessPoolExecutor(max_workers=8) as ex:
                for res in ex.map(run_config, jobs):
                    for row in res:
                        row.pop("func_csr", None)
                    out["results"].extend(res)
                    r0 = res[0]
                    print(
                        f"    {split} {r0['method']} rho={r0['cost_ratio']:g} "
                        f"CSR={np.mean([r['csr'] for r in res]) * 100:.4f}% "
                        f"WM={np.mean([r['wm_per_1k_inv'] for r in res]):.1f}",
                        flush=True,
                    )
                    save(out)
            done = {result_key(r) for r in out["results"]}

    readout(out)
    reconcile(out)
    save(out)
    write_tables(out)
    print_primary_summary(out)
    print(f"wrote {OUT}")
    print(f"wrote {TABLES / 'T_r20_age_grid_azure2021_rho10.csv'}")
    print(f"wrote {TABLES / 'T_r20_age_frontier_fullrho.csv'}")
    print(f"wrote {TABLES / 'T_r20_selection_summary.csv'}")
    print(f"wrote {TABLES / 'T_r20_reconciliation.csv'}")
    print(f"wrote {TABLES / 'T_r20_routing_exposure.csv'}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", "/data/260715/site-packages:.")
    main()
