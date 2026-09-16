#!/usr/bin/env python3
"""R27 paper-claim summary for drift-aware WINTER-G."""

from __future__ import annotations

from pathlib import Path
import json
from typing import List, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def exists(name: str) -> bool:
    return (TABLES / name).exists()


def fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|")


def md_table(df: pd.DataFrame, cols: Sequence[str]) -> str:
    data = df.loc[:, list(cols)].copy()
    lines = [
        "| " + " | ".join(data.columns) + " |",
        "| " + " | ".join(["---"] * len(data.columns)) + " |",
    ]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in data.columns) + " |")
    return "\n".join(lines)


def manifest_rows() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "results" / "runs").glob("revision_r26_*_manifest.json")):
        if "smoke" in path.name:
            continue
        data = json.loads(path.read_text())
        trace = str(data.get("trace", path.stem.replace("revision_r26_", "").replace("_manifest", "")))
        for split, meta in dict(data.get("splits", {})).items():
            routing = dict(meta.get("routing", {}))
            rec = {
                "trace": trace,
                "split": split,
                "pool_size": int(meta.get("pool_size", 0)),
                "invocations": float(meta.get("invocations", 0.0)),
                "recovery_candidate_share": float(routing.get("recovery_candidate", 0.0)),
                "mean_fires_per_function": float(routing.get("mean_fires_per_function", 0.0)),
                "mean_recovery_ticks_per_function": float(routing.get("mean_recovery_ticks_per_function", 0.0)),
                "served_recovery_share_rho10": float(routing.get("G_WE_A720D_served_recovery_rho10", 0.0)),
                "guard_blocked_share_rho10": float(routing.get("G_WE_A720D_guard_blocked_rho10", 0.0)),
                "unguarded_recovery_share_rho10": float(routing.get("G_WE_A720D_NG_served_recovery_rho10", 0.0)),
            }
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    lines: List[str] = [
        "# R27 paper-claim summary",
        "",
        "Negative deltas favor the drift-aware gate. Dual wins require both lower CSR and lower registered cost.",
        "",
    ]

    if exists("T_r25_drift_integrated_gate_dual_counts.csv"):
        dual = pd.read_csv(TABLES / "T_r25_drift_integrated_gate_dual_counts.csv")
        pooled = dual[(dual["provider"] == "pooled") & (dual["arm"].isin(["G_WE_A720D", "G_WE_A720D_NG"]))]
        lines += [
            "## R25 integrated drift gate",
            "",
            md_table(
                pooled,
                [
                    "provider",
                    "source",
                    "arm",
                    "n_condition_cells",
                    "csr_lower_than_ewma_0p1",
                    "cost_lower_than_ewma_0p1",
                    "dual_win_vs_ewma_0p1",
                    "dual_no_worse_vs_ewma_0p1",
                ],
            ),
            "",
        ]
    else:
        lines += ["## R25 integrated drift gate", "", "R25 cross-provider outputs not found yet.", ""]

    if exists("T_r25_drift_integrated_gate_cross_provider_overall.csv"):
        overall = pd.read_csv(TABLES / "T_r25_drift_integrated_gate_cross_provider_overall.csv")
        pooled = overall[(overall["provider"] == "pooled") & (overall["arm"].isin(["G_WE_A720D", "G_WE_A720D_NG"]))]
        lines += [
            "Pooled source/rho aggregates:",
            "",
            md_table(
                pooled,
                [
                    "source",
                    "rho",
                    "arm",
                    "csr_240_pct",
                    "cost_240",
                    "delta_vs_ewma_0p1_csr_240_pp",
                    "delta_vs_ewma_0p1_cost_240",
                    "route_learned_ticks_mean",
                    "guard_blocked_ticks_mean",
                ],
            ),
            "",
        ]

    r26_files = sorted(TABLES.glob("T_r26_*_fullrho.csv"))
    if r26_files:
        frames = []
        for path in r26_files:
            if path.name.startswith("T_r26_smoke"):
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "artifact", path.stem)
            frames.append(frame)
        if frames:
            r26 = pd.concat(frames, ignore_index=True)
            primary = r26[(r26["method"].isin(["G_WE_A720_current", "G_WE_A720D", "G_WE_A720D_NG"])) & (r26["rho"] == 10)]
            lines += [
                "## R26 steady false-positive audit",
                "",
                md_table(
                    primary,
                    [
                        "artifact",
                        "split",
                        "method",
                        "csr_pct",
                        "cost_per_1k_inv",
                        "delta_csr_pp_vs_ewma",
                        "delta_cost_per_1k_vs_ewma",
                        "route_recovery_share",
                        "served_recovery_share",
                        "guard_blocked_share",
                    ],
                ),
                "",
            ]
        else:
            lines += ["## R26 steady false-positive audit", "", "Only smoke outputs found so far.", ""]
    else:
        lines += ["## R26 steady false-positive audit", "", "R26 outputs not found yet.", ""]

    mr = manifest_rows()
    if not mr.empty:
        lines += [
            "R26 manifest routing exposure:",
            "",
            md_table(
                mr.sort_values(["trace", "split"]),
                [
                    "trace",
                    "split",
                    "pool_size",
                    "invocations",
                    "recovery_candidate_share",
                    "mean_fires_per_function",
                    "mean_recovery_ticks_per_function",
                    "served_recovery_share_rho10",
                    "guard_blocked_share_rho10",
                    "unguarded_recovery_share_rho10",
                ],
            ),
            "",
        ]

    if exists("T_r24_ewma_alpha_frontier_readout.csv"):
        alpha = pd.read_csv(TABLES / "T_r24_ewma_alpha_frontier_readout.csv")
        lines += [
            "## R22/R24 EWMA alpha frontier",
            "",
            md_table(
                alpha,
                [
                    "cohort",
                    "rho",
                    "winter_csr_pct",
                    "validation_alpha",
                    "validation_delta_csr_pp_vs_winter",
                    "expost_alpha",
                    "expost_delta_csr_pp_vs_winter",
                ],
            ),
            "",
        ]

    lines += [
        "## Claim rule",
        "",
        "Use the strong wording only where G_WE_A720D has a dual win against EWMA alpha=0.1. Where CSR improves but cost does not, describe the result as a CSR recovery frontier point rather than economic dominance.",
    ]
    out = TABLES / "T_r27_paper_claim_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
