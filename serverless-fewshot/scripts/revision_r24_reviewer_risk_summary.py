#!/usr/bin/env python3
"""Reviewer-risk summary tables for the FGCS WINTER revision.

This script does not run new simulations. It consolidates the already executed
R21/R22/R23 experiments into reviewer-facing readouts:

* R23 drift: practical WINTER refits vs EWMA arms, with CSR- and cost-selected
  envelopes separated.
* R23 drift: per-condition counts for the best practical refit envelope and for
  each deployable single arm.
* R22 onboarding: fixed EWMA alpha, validation-selected EWMA, and ex-post EWMA
  frontier against WINTER.
* R21 cross-provider faithful head/gate audit against EWMA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"

WINDOW = 240
CSR = f"csr_{WINDOW}_pct"
COST = f"cost_{WINDOW}"
PRACTICAL_WINTER = ["winter_scheduled", "winter_trigger", "winter_sched_trigger"]
EWMA_ARMS = ["ewma_0.1", "ewma_0.3", "ewma_0.5"]
EPS = 1e-12


def _as_float_label(value: object) -> str:
    try:
        fval = float(value)
    except Exception:
        return str(value)
    return f"{fval:g}"


def _best_row(group: pd.DataFrame, arms: Sequence[str], metric: str) -> pd.Series:
    sub = group[group["arm"].isin(arms)].copy()
    if sub.empty:
        raise ValueError(f"missing arms {arms} for group")
    return sub.sort_values([metric, "arm"]).iloc[0]


def _row_for_arm(group: pd.DataFrame, arm: str) -> pd.Series:
    sub = group[group["arm"] == arm]
    if sub.empty:
        raise ValueError(f"missing arm {arm} for group")
    return sub.iloc[0]


def drift_overall() -> pd.DataFrame:
    df = pd.read_csv(TABLES / "T_r23_drift_v2_cross_provider_overall.csv")
    rows: List[dict] = []
    for (provider, source, rho), group in df.groupby(["provider", "source", "rho"], sort=True):
        best_w_csr = _best_row(group, PRACTICAL_WINTER, CSR)
        best_e_csr = _best_row(group, EWMA_ARMS, CSR)
        best_w_cost = _best_row(group, PRACTICAL_WINTER, COST)
        best_e_cost = _best_row(group, EWMA_ARMS, COST)
        ewma01 = _row_for_arm(group, "ewma_0.1")
        frozen = _row_for_arm(group, "winter_frozen")
        rows.append(
            {
                "provider": provider,
                "source": source,
                "rho": _as_float_label(rho),
                "n_events": int(best_w_csr["n_events"]),
                "best_winter_csr_arm": best_w_csr["arm"],
                "best_winter_csr_pct": best_w_csr[CSR],
                "best_winter_cost_for_csr": best_w_csr[COST],
                "best_ewma_csr_arm": best_e_csr["arm"],
                "best_ewma_csr_pct": best_e_csr[CSR],
                "best_ewma_cost_for_csr": best_e_csr[COST],
                "ewma_0p1_csr_pct": ewma01[CSR],
                "ewma_0p1_cost": ewma01[COST],
                "winter_minus_best_ewma_csr_pp": best_w_csr[CSR] - best_e_csr[CSR],
                "winter_minus_ewma_0p1_csr_pp": best_w_csr[CSR] - ewma01[CSR],
                "winter_minus_best_ewma_cost_for_csr": best_w_csr[COST] - best_e_csr[COST],
                "winter_minus_ewma_0p1_cost_for_csr": best_w_csr[COST] - ewma01[COST],
                "frozen_csr_pct": frozen[CSR],
                "frozen_cost": frozen[COST],
                "winter_minus_frozen_csr_pp": best_w_csr[CSR] - frozen[CSR],
                "winter_minus_frozen_cost_for_csr": best_w_csr[COST] - frozen[COST],
                "best_winter_cost_arm": best_w_cost["arm"],
                "best_winter_cost_csr_pct": best_w_cost[CSR],
                "best_winter_cost": best_w_cost[COST],
                "best_ewma_cost_arm": best_e_cost["arm"],
                "best_ewma_cost_csr_pct": best_e_cost[CSR],
                "best_ewma_cost": best_e_cost[COST],
                "cost_selected_winter_minus_ewma_cost": best_w_cost[COST] - best_e_cost[COST],
                "cost_selected_winter_minus_ewma_csr_pp": best_w_cost[CSR] - best_e_cost[CSR],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "T_r24_drift_vs_ewma_overall.csv", index=False)
    return out


def _condition_readout(group: pd.DataFrame) -> dict:
    best_w = _best_row(group, PRACTICAL_WINTER, CSR)
    best_e = _best_row(group, EWMA_ARMS, CSR)
    ewma01 = _row_for_arm(group, "ewma_0.1")
    frozen = _row_for_arm(group, "winter_frozen")
    out = {
        "best_winter_arm": best_w["arm"],
        "best_winter_csr_pct": best_w[CSR],
        "best_winter_cost": best_w[COST],
        "best_ewma_arm": best_e["arm"],
        "best_ewma_csr_pct": best_e[CSR],
        "best_ewma_cost": best_e[COST],
        "ewma_0p1_csr_pct": ewma01[CSR],
        "ewma_0p1_cost": ewma01[COST],
        "frozen_csr_pct": frozen[CSR],
        "frozen_cost": frozen[COST],
    }
    for arm in PRACTICAL_WINTER:
        arm_row = _row_for_arm(group, arm)
        out[f"{arm}_csr_pct"] = arm_row[CSR]
        out[f"{arm}_cost"] = arm_row[COST]
    return out


def drift_condition_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(TABLES / "T_r23_drift_v2_cross_provider_condition_summary.csv")
    rows: List[dict] = []
    for key, group in df.groupby(["provider", "source", "kind", "rho"], sort=True):
        provider, source, kind, rho = key
        rec = {
            "provider": provider,
            "source": source,
            "kind": kind,
            "rho": _as_float_label(rho),
            "n_events": int(group["n_events"].iloc[0]),
        }
        rec.update(_condition_readout(group))
        rows.append(rec)
    per_condition = pd.DataFrame(rows)
    per_condition.to_csv(TABLES / "T_r24_drift_vs_ewma_conditions.csv", index=False)

    count_rows: List[dict] = []
    for provider in sorted(per_condition["provider"].unique()):
        pdat = per_condition[per_condition["provider"] == provider]
        for source in ["all"] + sorted(pdat["source"].unique()):
            sdat = pdat if source == "all" else pdat[pdat["source"] == source]
            total = len(sdat)
            count_rows.append(
                {
                    "provider": provider,
                    "source": source,
                    "n_condition_cells": total,
                    "best_practical_lower_than_frozen": int(
                        (sdat["best_winter_csr_pct"] < sdat["frozen_csr_pct"] - EPS).sum()
                    ),
                    "best_practical_no_worse_than_frozen": int(
                        (sdat["best_winter_csr_pct"] <= sdat["frozen_csr_pct"] + EPS).sum()
                    ),
                    "best_practical_lower_than_ewma_0p1": int(
                        (sdat["best_winter_csr_pct"] < sdat["ewma_0p1_csr_pct"] - EPS).sum()
                    ),
                    "best_practical_no_worse_than_ewma_0p1": int(
                        (sdat["best_winter_csr_pct"] <= sdat["ewma_0p1_csr_pct"] + EPS).sum()
                    ),
                    "best_practical_lower_than_best_ewma": int(
                        (sdat["best_winter_csr_pct"] < sdat["best_ewma_csr_pct"] - EPS).sum()
                    ),
                    "best_practical_no_worse_than_best_ewma": int(
                        (sdat["best_winter_csr_pct"] <= sdat["best_ewma_csr_pct"] + EPS).sum()
                    ),
                    "best_practical_cost_lower_than_best_ewma": int(
                        (sdat["best_winter_cost"] < sdat["best_ewma_cost"] - EPS).sum()
                    ),
                    "best_practical_cost_no_worse_than_best_ewma": int(
                        (sdat["best_winter_cost"] <= sdat["best_ewma_cost"] + EPS).sum()
                    ),
                }
            )
    count_df = pd.DataFrame(count_rows)
    count_df.to_csv(TABLES / "T_r24_drift_condition_counts.csv", index=False)

    arm_rows: List[dict] = []
    for provider in sorted(per_condition["provider"].unique()):
        pdat = per_condition[per_condition["provider"] == provider]
        for source in ["all"] + sorted(pdat["source"].unique()):
            sdat = pdat if source == "all" else pdat[pdat["source"] == source]
            for arm in PRACTICAL_WINTER:
                csr_col = f"{arm}_csr_pct"
                cost_col = f"{arm}_cost"
                arm_rows.append(
                    {
                        "provider": provider,
                        "source": source,
                        "arm": arm,
                        "n_condition_cells": len(sdat),
                        "lower_than_frozen": int((sdat[csr_col] < sdat["frozen_csr_pct"] - EPS).sum()),
                        "no_worse_than_frozen": int((sdat[csr_col] <= sdat["frozen_csr_pct"] + EPS).sum()),
                        "lower_than_ewma_0p1": int((sdat[csr_col] < sdat["ewma_0p1_csr_pct"] - EPS).sum()),
                        "no_worse_than_ewma_0p1": int((sdat[csr_col] <= sdat["ewma_0p1_csr_pct"] + EPS).sum()),
                        "lower_than_best_ewma": int((sdat[csr_col] < sdat["best_ewma_csr_pct"] - EPS).sum()),
                        "no_worse_than_best_ewma": int((sdat[csr_col] <= sdat["best_ewma_csr_pct"] + EPS).sum()),
                        "cost_lower_than_best_ewma": int((sdat[cost_col] < sdat["best_ewma_cost"] - EPS).sum()),
                        "cost_no_worse_than_best_ewma": int((sdat[cost_col] <= sdat["best_ewma_cost"] + EPS).sum()),
                    }
                )
    arm_df = pd.DataFrame(arm_rows)
    arm_df.to_csv(TABLES / "T_r24_drift_single_arm_counts.csv", index=False)
    return per_condition, count_df, arm_df


def ewma_alpha_readout() -> pd.DataFrame:
    df = pd.read_csv(TABLES / "T_r22_ewma_alpha_frontier.csv")
    rows: List[dict] = []
    for (cohort, rho), group in df.groupby(["cohort", "rho"], sort=True):
        ref = group[group["selection"] == "reference"].iloc[0]
        default = group[group["selection"] == "default_alpha_0.1"].iloc[0]
        val = group[group["selection"] == "validation_min_csr"].iloc[0]
        expost = group[group["selection"] == "expost_min_csr"].iloc[0]
        rows.append(
            {
                "cohort": cohort,
                "rho": _as_float_label(rho),
                "winter_csr_pct": ref["csr_pct"],
                "winter_cost_per_1k_inv": ref["cost_per_1k_inv"],
                "winter_al_0p1_min": ref["al_0p1_min"],
                "default_alpha": default["alpha"],
                "default_ewma_csr_pct": default["csr_pct"],
                "default_delta_csr_pp_vs_winter": default["delta_csr_pp_vs_winter"],
                "default_wilcoxon_p_vs_winter": default["wilcoxon_p_vs_winter"],
                "validation_alpha": val["alpha"],
                "validation_ewma_csr_pct": val["csr_pct"],
                "validation_delta_csr_pp_vs_winter": val["delta_csr_pp_vs_winter"],
                "validation_wilcoxon_p_vs_winter": val["wilcoxon_p_vs_winter"],
                "expost_alpha": expost["alpha"],
                "expost_ewma_csr_pct": expost["csr_pct"],
                "expost_delta_csr_pp_vs_winter": expost["delta_csr_pp_vs_winter"],
                "expost_wilcoxon_p_vs_winter": expost["wilcoxon_p_vs_winter"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "T_r24_ewma_alpha_frontier_readout.csv", index=False)
    return out


def r21_gate_readout() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(TABLES / "T_r21_reviewer_full_faithful_fullrho.csv")
    methods = ["A5_faithful", "G_WE_A720"]
    detail = df[df["method"].isin(methods)].copy()
    detail["provider_family"] = np.where(detail["split"].str.startswith("h_"), "huawei", "azure2019")
    detail = detail[
        [
            "provider_family",
            "split",
            "rho",
            "method",
            "csr_pct",
            "cost_per_1k_inv",
            "delta_csr_pp_vs_ewma",
            "delta_cost_per_1k_vs_ewma",
            "boot_ci95_lo_pp",
            "boot_ci95_hi_pp",
            "route_learned_share",
        ]
    ]
    detail["rho"] = detail["rho"].map(_as_float_label)
    detail.to_csv(TABLES / "T_r24_r21_faithful_gate_detail.csv", index=False)

    rows: List[dict] = []
    for (provider_family, method), group in detail.groupby(["provider_family", "method"], sort=True):
        rows.append(
            {
                "provider_family": provider_family,
                "method": method,
                "n_cells": len(group),
                "csr_lower_than_ewma": int((group["delta_csr_pp_vs_ewma"] < -EPS).sum()),
                "csr_no_worse_than_ewma": int((group["delta_csr_pp_vs_ewma"] <= EPS).sum()),
                "cost_lower_than_ewma": int((group["delta_cost_per_1k_vs_ewma"] < -EPS).sum()),
                "cost_no_worse_than_ewma": int((group["delta_cost_per_1k_vs_ewma"] <= EPS).sum()),
            }
        )
    counts = pd.DataFrame(rows)
    counts.to_csv(TABLES / "T_r24_r21_faithful_gate_counts.csv", index=False)
    return detail, counts


def _md_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int | None = None) -> str:
    data = df.loc[:, list(cols)].copy()
    if max_rows is not None:
        data = data.head(max_rows)
    headers = list(data.columns)

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def _fmt_count(row: pd.Series, field: str) -> str:
    return f"{int(row[field])}/{int(row['n_condition_cells'])}"


def write_markdown(
    drift_overall_df: pd.DataFrame,
    drift_counts: pd.DataFrame,
    single_arm_counts: pd.DataFrame,
    alpha_df: pd.DataFrame,
    r21_counts: pd.DataFrame,
) -> None:
    pooled_counts = drift_counts[(drift_counts["provider"] == "pooled") & (drift_counts["source"] == "all")].iloc[0]
    pooled_overall = drift_overall_df[drift_overall_df["provider"] == "pooled"].copy()
    pooled_single = single_arm_counts[
        (single_arm_counts["provider"] == "pooled") & (single_arm_counts["source"] == "all")
    ].copy()

    lines = [
        "# R24 reviewer-risk experiment summary",
        "",
        "Generated from R21/R22/R23 artifacts. Negative deltas mean the WINTER arm is lower.",
        "",
        "## Drift vs EWMA",
        "",
        (
            "Per-condition best practical WINTER refit envelope vs frozen: "
            f"{_fmt_count(pooled_counts, 'best_practical_lower_than_frozen')} strict CSR wins "
            f"({_fmt_count(pooled_counts, 'best_practical_no_worse_than_frozen')} no-worse)."
        ),
        (
            "Per-condition best practical WINTER refit envelope vs EWMA alpha=0.1: "
            f"{_fmt_count(pooled_counts, 'best_practical_lower_than_ewma_0p1')} strict CSR wins."
        ),
        (
            "Per-condition best practical WINTER refit envelope vs best EWMA alpha in {0.1,0.3,0.5}: "
            f"{_fmt_count(pooled_counts, 'best_practical_lower_than_best_ewma')} strict CSR wins "
            f"({_fmt_count(pooled_counts, 'best_practical_no_worse_than_best_ewma')} no-worse)."
        ),
        (
            "Cost check vs best EWMA, using the CSR-selected best practical WINTER envelope: "
            f"{_fmt_count(pooled_counts, 'best_practical_cost_lower_than_best_ewma')} strict cost wins."
        ),
        "",
        _md_table(
            pooled_overall,
            [
                "source",
                "rho",
                "best_winter_csr_arm",
                "best_winter_csr_pct",
                "best_ewma_csr_arm",
                "best_ewma_csr_pct",
                "winter_minus_best_ewma_csr_pp",
                "winter_minus_best_ewma_cost_for_csr",
                "cost_selected_winter_minus_ewma_cost",
            ],
        ),
        "",
        "Single deployable practical arms on pooled 28 condition cells:",
        "",
        _md_table(
            pooled_single,
            [
                "arm",
                "lower_than_frozen",
                "lower_than_best_ewma",
                "cost_lower_than_best_ewma",
                "n_condition_cells",
            ],
        ),
        "",
        "## EWMA alpha frontier",
        "",
        _md_table(
            alpha_df,
            [
                "cohort",
                "rho",
                "winter_csr_pct",
                "default_alpha",
                "default_delta_csr_pp_vs_winter",
                "validation_alpha",
                "validation_delta_csr_pp_vs_winter",
                "expost_alpha",
                "expost_delta_csr_pp_vs_winter",
            ],
        ),
        "",
        "## R21 faithful cross-provider audit",
        "",
        _md_table(
            r21_counts,
            [
                "provider_family",
                "method",
                "n_cells",
                "csr_lower_than_ewma",
                "csr_no_worse_than_ewma",
                "cost_lower_than_ewma",
                "cost_no_worse_than_ewma",
            ],
        ),
        "",
        "## Interpretation",
        "",
        (
            "R23 closes the missing drift-vs-EWMA CSR comparison: the practical refit envelope is "
            "lower than the best EWMA alpha in all pooled condition cells. However, the cost frontier "
            "does not uniformly favor WINTER, especially under synthetic drifts, so manuscript wording "
            "should separate CSR improvement from economic break-even claims."
        ),
        (
            "R22 shows that an EWMA alpha sweep does not erase the onboarding CSR advantage of WINTER "
            "in the reported cohorts. The validation-selected alpha is 0.3 for every rho in these "
            "readouts; ex-post alpha selection narrows but does not remove the gap."
        ),
        (
            "R21 confirms that the current Section-3 faithful head/gate audit should remain scoped as "
            "a diagnostic: it does not establish broad active-pool cross-provider dominance over EWMA."
        ),
        (
            "No existing experiment implements a mature EWMA -> drift detected -> refitted learned "
            "WINTER-G state transition. That is a new integrated policy experiment, not a rerun of the "
            "current R23 drift extension."
        ),
        "",
        "## Output files",
        "",
        "- results/tables/T_r24_drift_vs_ewma_overall.csv",
        "- results/tables/T_r24_drift_vs_ewma_conditions.csv",
        "- results/tables/T_r24_drift_condition_counts.csv",
        "- results/tables/T_r24_drift_single_arm_counts.csv",
        "- results/tables/T_r24_ewma_alpha_frontier_readout.csv",
        "- results/tables/T_r24_r21_faithful_gate_detail.csv",
        "- results/tables/T_r24_r21_faithful_gate_counts.csv",
    ]
    (TABLES / "T_r24_reviewer_experiment_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    overall = drift_overall()
    _, drift_counts, single_arm_counts = drift_condition_tables()
    alpha = ewma_alpha_readout()
    _, r21_counts = r21_gate_readout()
    write_markdown(overall, drift_counts, single_arm_counts, alpha, r21_counts)
    print(f"wrote R24 reviewer-risk summaries under {TABLES}")


if __name__ == "__main__":
    main()
