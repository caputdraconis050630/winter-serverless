#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize R20 fast-DES JSON outputs into CSV tables."""

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
TABLES = PROJECT_ROOT / "results" / "tables"
KAPPA = 15.0
PRIMARY_RHO = 10.0


def summarize(rows):
    if not rows:
        return None
    total = np.array([r["total_invocations"] for r in rows], dtype=np.float64)
    cold = np.array([r["cold_starts"] for r in rows], dtype=np.float64)
    wm = np.array([r["wm_total_gb_s"] for r in rows], dtype=np.float64)
    rho = float(rows[0]["cost_ratio"])
    csr = cold / np.maximum(total, 1e-10)
    wm_per_1k = wm / np.maximum(total / 1000.0, 1e-10)
    cost_per_1k = (KAPPA * rho * cold + wm) / np.maximum(total / 1000.0, 1e-10)
    return {
        "n": len(rows),
        "csr_pct": float(100.0 * csr.mean()),
        "csr_std_pct": float(100.0 * csr.std()),
        "wm_per_1k_inv": float(wm_per_1k.mean()),
        "cost_per_1k_inv": float(cost_per_1k.mean()),
        "cost_std_per_1k_inv": float(cost_per_1k.std()),
    }


def method_meta(method):
    if method in ("B4a_ewma", "A5_proto", "A5_faithful"):
        return "", "", method
    parts = method.split("_")
    if len(parts) < 3:
        return "", "", ""
    family = parts[1]
    age = parts[2][1:]
    alias = {
        ("WE", "720"): "current_WINTRG",
        ("WE", "inf"): "v3",
        ("WL", "0"): "v4",
    }.get((family, age), "")
    return family, age, alias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    rows = json.load(open(args.input))
    TABLES.mkdir(parents=True, exist_ok=True)
    splits = sorted({r["split"] for r in rows})
    rhos = sorted({float(r["cost_ratio"]) for r in rows})
    methods = sorted({r["method"] for r in rows}, key=lambda m: (method_meta(m)[0], method_meta(m)[1], m))

    header = (
        "split,method,family,age_min,alias,rho,csr_pct,csr_std_pct,"
        "wm_per_1k_inv,cost_per_1k_inv,delta_cost_per_1k_vs_ewma,"
        "delta_csr_pp_vs_ewma,n"
    )
    full = [header]
    rho10 = [header]
    for split in splits:
        for rho in rhos:
            ref = summarize(
                [
                    r
                    for r in rows
                    if r["split"] == split
                    and r["method"] == "B4a_ewma"
                    and abs(float(r["cost_ratio"]) - rho) < 1e-9
                ]
            )
            for method in methods:
                rec = summarize(
                    [
                        r
                        for r in rows
                        if r["split"] == split
                        and r["method"] == method
                        and abs(float(r["cost_ratio"]) - rho) < 1e-9
                    ]
                )
                if not rec:
                    continue
                family, age, alias = method_meta(method)
                dcost = rec["cost_per_1k_inv"] - ref["cost_per_1k_inv"] if ref else np.nan
                dcsr = rec["csr_pct"] - ref["csr_pct"] if ref else np.nan
                line = (
                    f"{split},{method},{family},{age},{alias},{rho:g},"
                    f"{rec['csr_pct']:.6f},{rec['csr_std_pct']:.6f},"
                    f"{rec['wm_per_1k_inv']:.3f},{rec['cost_per_1k_inv']:.3f},"
                    f"{dcost:.3f},{dcsr:.6f},{rec['n']}"
                )
                full.append(line)
                if abs(rho - PRIMARY_RHO) < 1e-9:
                    rho10.append(line)

    full_path = TABLES / f"{args.prefix}_fullrho.csv"
    rho10_path = TABLES / f"{args.prefix}_rho10.csv"
    full_path.write_text("\n".join(full) + "\n")
    rho10_path.write_text("\n".join(rho10) + "\n")
    print(f"wrote {full_path}")
    print(f"wrote {rho10_path}")

    print("\n=== rho=10 best by registered cost ===")
    for split in splits:
        ref = summarize(
            [
                r
                for r in rows
                if r["split"] == split
                and r["method"] == "B4a_ewma"
                and abs(float(r["cost_ratio"]) - PRIMARY_RHO) < 1e-9
            ]
        )
        ranked = []
        for method in methods:
            if method == "B4a_ewma":
                continue
            rec = summarize(
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
        print(f"\n{split}")
        for cost, method, rec in ranked[:10]:
            print(
                f"  {method:12s} CSR={rec['csr_pct']:.4f}% "
                f"dCSR={rec['csr_pct'] - ref['csr_pct']:+.4f}pp "
                f"WM={rec['wm_per_1k_inv']:.1f} C/1k={cost:.1f} "
                f"dC={cost - ref['cost_per_1k_inv']:+.1f}"
            )


if __name__ == "__main__":
    main()
