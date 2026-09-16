#!/usr/bin/env python3
"""Aggregate Revision R23 drift-v2 experiment outputs.

The raw event table has one row per event, arm, rho, and simulator seed.
This script compares each arm against winter_frozen on paired event-seed
records and also emits invocation-weighted overall summaries.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


WINDOWS = (15, 30, 60, 240)
BASELINE_ARM = "winter_frozen"
PRACTICAL_WINTER_ARMS = (
    "winter_scheduled",
    "winter_trigger",
    "winter_sched_trigger",
)
DEFAULT_PROVIDERS = (
    ("azure2021", ""),
    ("azure2019", "_azure2019"),
    ("huawei", "_huawei"),
)


def wilcoxon_pvalue(delta: pd.Series) -> float:
    values = pd.to_numeric(delta, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size < 2 or np.allclose(values, 0.0):
        return math.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return float(wilcoxon(values, zero_method="wilcox").pvalue)
    except ValueError:
        return math.nan


def count_unique_events(frame: pd.DataFrame) -> int:
    provider_col = "origin_provider" if "origin_provider" in frame.columns else "provider"
    return frame[[provider_col, "event_id"]].drop_duplicates().shape[0]


def count_unique_event_seeds(frame: pd.DataFrame) -> int:
    provider_col = "origin_provider" if "origin_provider" in frame.columns else "provider"
    return frame[[provider_col, "event_id", "seed"]].drop_duplicates().shape[0]


def load_events(tables_dir: Path, provider_specs: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for provider, suffix in provider_specs:
        path = tables_dir / f"T_r23_drift_v2_events{suffix}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "origin_provider", provider)
        frame.insert(0, "provider", provider)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_weighted(events: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in events.groupby(list(group_cols), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = dict(zip(group_cols, keys))
        row["n_events"] = count_unique_events(group)
        row["n_event_seed"] = count_unique_event_seeds(group)
        row["post_sum_actual_mean"] = group["post_sum_actual"].mean()
        row["post_active_frac_mean"] = group["post_active_frac"].mean()
        row["fires_post_mean"] = group["fires_post"].mean()
        row["fit_count_post_mean"] = group["fit_count_post"].mean()
        row["first10_active_csr_pct"] = group["first10_active_csr_pct"].mean()
        row["first10_active_inv"] = group["first10_active_inv"].sum()
        for window in WINDOWS:
            cold = group[f"cold_{window}"].sum()
            inv = group[f"inv_{window}"].sum()
            row[f"cold_{window}"] = cold
            row[f"inv_{window}"] = inv
            row[f"csr_{window}_pct"] = 100.0 * cold / inv if inv else math.nan
            row[f"idle_gb_s_{window}"] = group[f"idle_gb_s_{window}"].sum()
            row[f"busy_gb_s_{window}"] = group[f"busy_gb_s_{window}"].sum()
            row[f"cost_{window}"] = group[f"cost_{window}"].sum()
        rows.append(row)

    out = pd.DataFrame(rows)
    baseline_cols = [c for c in group_cols if c != "arm"]
    base = out[out["arm"] == BASELINE_ARM].copy()
    rename = {}
    for window in WINDOWS:
        rename[f"csr_{window}_pct"] = f"frozen_csr_{window}_pct"
        rename[f"cost_{window}"] = f"frozen_cost_{window}"
        rename[f"idle_gb_s_{window}"] = f"frozen_idle_gb_s_{window}"
    base = base[list(baseline_cols) + list(rename)].rename(columns=rename)
    out = out.merge(base, on=list(baseline_cols), how="left")
    for window in WINDOWS:
        out[f"delta_vs_frozen_csr_{window}_pp"] = (
            out[f"csr_{window}_pct"] - out[f"frozen_csr_{window}_pct"]
        )
        out[f"delta_vs_frozen_cost_{window}"] = (
            out[f"cost_{window}"] - out[f"frozen_cost_{window}"]
        )
        out[f"delta_vs_frozen_idle_gb_s_{window}"] = (
            out[f"idle_gb_s_{window}"] - out[f"frozen_idle_gb_s_{window}"]
        )
    drop_cols = [
        c
        for c in out.columns
        if c.startswith("frozen_csr_")
        or c.startswith("frozen_cost_")
        or c.startswith("frozen_idle_")
    ]
    return out.drop(columns=drop_cols)


def pairwise_against_frozen(events: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    key_cols = ["provider", "source", "kind", "rho", "event_id", "seed"]
    if "origin_provider" in events.columns:
        key_cols.insert(1, "origin_provider")
    metric_cols = []
    for window in WINDOWS:
        metric_cols += [
            f"cold_{window}",
            f"inv_{window}",
            f"csr_{window}_pct",
            f"idle_gb_s_{window}",
            f"cost_{window}",
        ]
    metric_cols += ["first10_active_csr_pct", "fires_post", "fit_count_post"]

    base = events[events["arm"] == BASELINE_ARM][key_cols + metric_cols].copy()
    base = base.rename(columns={col: f"{col}_frozen" for col in metric_cols})

    rows: List[Dict[str, object]] = []
    arms = sorted(a for a in events["arm"].unique() if a != BASELINE_ARM)
    for arm in arms:
        arm_frame = events[events["arm"] == arm]
        merged = arm_frame.merge(base, on=key_cols, how="inner")
        for keys, group in merged.groupby(list(group_cols), sort=True, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: Dict[str, object] = dict(zip(group_cols, keys))
            row["arm"] = arm
            row["n_pairs"] = group.shape[0]
            row["n_events"] = count_unique_events(group)
            row["mean_fires_post"] = group["fires_post"].mean()
            row["mean_fit_count_post"] = group["fit_count_post"].mean()
            for window in WINDOWS:
                arm_cold = group[f"cold_{window}"].sum()
                arm_inv = group[f"inv_{window}"].sum()
                base_cold = group[f"cold_{window}_frozen"].sum()
                base_inv = group[f"inv_{window}_frozen"].sum()
                delta = group[f"csr_{window}_pct"] - group[f"csr_{window}_pct_frozen"]
                cost_delta = group[f"cost_{window}"] - group[f"cost_{window}_frozen"]
                idle_delta = group[f"idle_gb_s_{window}"] - group[f"idle_gb_s_{window}_frozen"]
                row[f"arm_csr_{window}_pct"] = 100.0 * arm_cold / arm_inv if arm_inv else math.nan
                row[f"frozen_csr_{window}_pct"] = (
                    100.0 * base_cold / base_inv if base_inv else math.nan
                )
                row[f"weighted_delta_csr_{window}_pp"] = (
                    row[f"arm_csr_{window}_pct"] - row[f"frozen_csr_{window}_pct"]
                )
                row[f"mean_delta_csr_{window}_pp"] = delta.mean()
                row[f"median_delta_csr_{window}_pp"] = delta.median()
                row[f"better_frac_csr_{window}"] = (delta < 0).mean()
                row[f"worse_frac_csr_{window}"] = (delta > 0).mean()
                row[f"wilcoxon_p_csr_{window}"] = wilcoxon_pvalue(delta)
                row[f"sum_delta_cost_{window}"] = cost_delta.sum()
                row[f"mean_delta_cost_{window}"] = cost_delta.mean()
                row[f"sum_delta_idle_gb_s_{window}"] = idle_delta.sum()
            rows.append(row)
    return pd.DataFrame(rows)


def add_best_winter_columns(summary: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    keys = ["provider", "source", "kind", "rho"]
    practical = summary[summary["arm"].isin(PRACTICAL_WINTER_ARMS)].copy()
    idx = practical.groupby(keys)["delta_vs_frozen_csr_240_pp"].idxmin()
    best = practical.loc[idx, keys + ["arm", "n_events", "csr_240_pct", "delta_vs_frozen_csr_240_pp"]]
    best = best.rename(
        columns={
            "arm": "best_winter_arm",
            "csr_240_pct": "best_csr240_pct",
            "delta_vs_frozen_csr_240_pp": "best_delta240_pp",
        }
    )

    for arm in PRACTICAL_WINTER_ARMS:
        cols = keys + ["delta_vs_frozen_csr_240_pp", "csr_240_pct", "fires_post_mean"]
        arm_summary = summary[summary["arm"] == arm][cols].rename(
            columns={
                "delta_vs_frozen_csr_240_pp": f"{arm}_delta240_pp",
                "csr_240_pct": f"{arm}_csr240_pct",
                "fires_post_mean": f"{arm}_fires_post_mean",
            }
        )
        best = best.merge(arm_summary, on=keys, how="left")

        arm_pairwise = pairwise[pairwise["arm"] == arm][
            keys + ["wilcoxon_p_csr_240", "better_frac_csr_240", "worse_frac_csr_240"]
        ].rename(
            columns={
                "wilcoxon_p_csr_240": f"{arm}_wilcoxon_p240",
                "better_frac_csr_240": f"{arm}_better_frac240",
                "worse_frac_csr_240": f"{arm}_worse_frac240",
            }
        )
        best = best.merge(arm_pairwise, on=keys, how="left")
    return best.sort_values(keys).reset_index(drop=True)


def write_provider_outputs(tables_dir: Path, provider: str, suffix: str, events: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    provider_events = events[events["provider"] == provider].copy()
    summary = summarize_weighted(provider_events, ["provider", "source", "kind", "rho", "arm"])
    overall = summarize_weighted(provider_events, ["provider", "source", "rho", "arm"])
    pairwise = pairwise_against_frozen(provider_events, ["provider", "source", "kind", "rho"])
    pairwise_overall = pairwise_against_frozen(provider_events, ["provider", "source", "rho"])
    best = add_best_winter_columns(summary, pairwise)

    summary.to_csv(tables_dir / f"T_r23_drift_v2_condition_summary{suffix}.csv", index=False)
    overall.to_csv(tables_dir / f"T_r23_drift_v2_overall{suffix}.csv", index=False)
    pairwise.to_csv(tables_dir / f"T_r23_drift_v2_pairwise{suffix}.csv", index=False)
    pairwise_overall.to_csv(tables_dir / f"T_r23_drift_v2_pairwise_overall{suffix}.csv", index=False)
    best.to_csv(tables_dir / f"T_r23_drift_v2_winter_best{suffix}.csv", index=False)
    return {
        "summary": summary,
        "overall": overall,
        "pairwise": pairwise,
        "pairwise_overall": pairwise_overall,
        "best": best,
    }


def pooled_provider(events: pd.DataFrame) -> pd.DataFrame:
    pooled = events.copy()
    pooled["provider"] = "pooled"
    return pooled


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", default="results/tables")
    args = parser.parse_args(argv)

    tables_dir = Path(args.tables_dir)
    events = load_events(tables_dir, DEFAULT_PROVIDERS)

    all_outputs: Dict[str, List[pd.DataFrame]] = {
        "summary": [],
        "overall": [],
        "pairwise": [],
        "pairwise_overall": [],
        "best": [],
    }
    for provider, suffix in DEFAULT_PROVIDERS:
        outputs = write_provider_outputs(tables_dir, provider, suffix, events)
        for name, frame in outputs.items():
            all_outputs[name].append(frame)

    pooled_events = pooled_provider(events)
    pooled_summary = summarize_weighted(pooled_events, ["provider", "source", "kind", "rho", "arm"])
    pooled_overall = summarize_weighted(pooled_events, ["provider", "source", "rho", "arm"])
    pooled_pairwise = pairwise_against_frozen(pooled_events, ["provider", "source", "kind", "rho"])
    pooled_pairwise_overall = pairwise_against_frozen(pooled_events, ["provider", "source", "rho"])
    pooled_best = add_best_winter_columns(pooled_summary, pooled_pairwise)

    all_outputs["summary"].append(pooled_summary)
    all_outputs["overall"].append(pooled_overall)
    all_outputs["pairwise"].append(pooled_pairwise)
    all_outputs["pairwise_overall"].append(pooled_pairwise_overall)
    all_outputs["best"].append(pooled_best)

    pd.concat(all_outputs["summary"], ignore_index=True).to_csv(
        tables_dir / "T_r23_drift_v2_cross_provider_condition_summary.csv", index=False
    )
    pd.concat(all_outputs["overall"], ignore_index=True).to_csv(
        tables_dir / "T_r23_drift_v2_cross_provider_overall.csv", index=False
    )
    pd.concat(all_outputs["pairwise"], ignore_index=True).to_csv(
        tables_dir / "T_r23_drift_v2_cross_provider_pairwise.csv", index=False
    )
    pd.concat(all_outputs["pairwise_overall"], ignore_index=True).to_csv(
        tables_dir / "T_r23_drift_v2_cross_provider_pairwise_overall.csv", index=False
    )
    pd.concat(all_outputs["best"], ignore_index=True).to_csv(
        tables_dir / "T_r23_drift_v2_cross_provider_winter_best.csv", index=False
    )

    manifest = pd.DataFrame(
        [
            {
                "provider": provider,
                "suffix": suffix,
                "event_rows": int(events[events["provider"] == provider].shape[0]),
                "n_events": int(count_unique_events(events[events["provider"] == provider])),
                "n_event_seeds": int(count_unique_event_seeds(events[events["provider"] == provider])),
            }
            for provider, suffix in DEFAULT_PROVIDERS
        ]
    )
    manifest.to_csv(tables_dir / "T_r23_drift_v2_cross_provider_manifest.csv", index=False)

    print("[done] providers:", ", ".join(p for p, _ in DEFAULT_PROVIDERS))
    print("[done] wrote cross-provider summary, overall, pairwise, best, and manifest tables")


if __name__ == "__main__":
    main()
