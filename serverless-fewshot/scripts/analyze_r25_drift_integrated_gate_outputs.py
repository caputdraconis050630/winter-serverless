#!/usr/bin/env python3
"""Aggregate R25 integrated drift-aware WINTER-G outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


WINDOWS = (15, 30, 60, 240)
BASELINES = ("ewma_0.1", "G_WE_A720_current", "winter_frozen")
PRIMARY_ARMS = ("G_WE_A720D", "G_WE_A720D_NG")
DEFAULT_PROVIDERS = (
    ("azure2021", ""),
    ("azure2019", "_azure2019"),
    ("huawei", "_huawei"),
)
EPS = 1e-12


def count_unique_events(frame: pd.DataFrame) -> int:
    provider_col = "origin_provider" if "origin_provider" in frame.columns else "provider"
    return frame[[provider_col, "event_id"]].drop_duplicates().shape[0]


def count_unique_event_seeds(frame: pd.DataFrame) -> int:
    provider_col = "origin_provider" if "origin_provider" in frame.columns else "provider"
    return frame[[provider_col, "event_id", "seed"]].drop_duplicates().shape[0]


def load_events(tables_dir: Path, provider_specs: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for provider, suffix in provider_specs:
        path = tables_dir / f"T_r25_drift_integrated_gate_events{suffix}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "origin_provider", provider)
        frame.insert(0, "provider", provider)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_weighted(events: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    meta_cols = [
        "fires_post",
        "fit_count_post",
        "recovery_entries_post",
        "recovery_ticks",
        "route_candidate_ticks",
        "route_learned_ticks",
        "guard_blocked_ticks",
        "recovery_active_at_onset",
        "mature_eligible",
    ]
    rows: List[Dict[str, object]] = []
    for keys, group in events.groupby(list(group_cols), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = dict(zip(group_cols, keys))
        row["n_events"] = count_unique_events(group)
        row["n_event_seed"] = count_unique_event_seeds(group)
        row["post_sum_actual_mean"] = group["post_sum_actual"].mean()
        row["post_active_frac_mean"] = group["post_active_frac"].mean()
        for col in meta_cols:
            row[f"{col}_mean"] = group[col].mean() if col in group else 0.0
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
    return add_baseline_deltas(out, group_cols)


def add_baseline_deltas(summary: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    baseline_cols = [c for c in group_cols if c != "arm"]
    out = summary.copy()
    for baseline in BASELINES:
        base = out[out["arm"] == baseline].copy()
        if base.empty:
            continue
        rename = {}
        for window in WINDOWS:
            rename[f"csr_{window}_pct"] = f"{baseline}_csr_{window}_pct"
            rename[f"cost_{window}"] = f"{baseline}_cost_{window}"
        base = base[list(baseline_cols) + list(rename)].rename(columns=rename)
        out = out.merge(base, on=list(baseline_cols), how="left")
        safe = baseline.replace(".", "p")
        for window in WINDOWS:
            out[f"delta_vs_{safe}_csr_{window}_pp"] = (
                out[f"csr_{window}_pct"] - out[f"{baseline}_csr_{window}_pct"]
            )
            out[f"delta_vs_{safe}_cost_{window}"] = (
                out[f"cost_{window}"] - out[f"{baseline}_cost_{window}"]
            )
    drop_cols = [c for c in out.columns if any(c.startswith(f"{b}_") for b in BASELINES)]
    return out.drop(columns=drop_cols)


def dual_counts(summary: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in summary.groupby(list(group_cols), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        prefix = dict(zip(group_cols, keys))
        total_cells = group[["source", "kind", "rho"]].drop_duplicates().shape[0]
        for arm in PRIMARY_ARMS:
            sub = group[group["arm"] == arm]
            if sub.empty:
                continue
            dcsr = sub["delta_vs_ewma_0p1_csr_240_pp"]
            dcost = sub["delta_vs_ewma_0p1_cost_240"]
            rows.append(
                {
                    **prefix,
                    "arm": arm,
                    "n_condition_cells": int(total_cells),
                    "csr_lower_than_ewma_0p1": int((dcsr < -EPS).sum()),
                    "cost_lower_than_ewma_0p1": int((dcost < -EPS).sum()),
                    "dual_win_vs_ewma_0p1": int(((dcsr < -EPS) & (dcost < -EPS)).sum()),
                    "csr_no_worse_than_ewma_0p1": int((dcsr <= EPS).sum()),
                    "cost_no_worse_than_ewma_0p1": int((dcost <= EPS).sum()),
                    "dual_no_worse_vs_ewma_0p1": int(((dcsr <= EPS) & (dcost <= EPS)).sum()),
                }
            )
    return pd.DataFrame(rows)


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
    events_all = pd.concat([events, pooled_provider(events)], ignore_index=True)

    condition = summarize_weighted(events_all, ["provider", "source", "kind", "rho", "arm"])
    overall = summarize_weighted(events_all, ["provider", "source", "rho", "arm"])
    counts_all = dual_counts(condition, ["provider"])
    counts_by_source = dual_counts(condition, ["provider", "source"])
    counts = pd.concat([counts_all, counts_by_source], ignore_index=True)

    condition.to_csv(
        tables_dir / "T_r25_drift_integrated_gate_cross_provider_condition_summary.csv",
        index=False,
    )
    overall.to_csv(
        tables_dir / "T_r25_drift_integrated_gate_cross_provider_overall.csv",
        index=False,
    )
    counts.to_csv(
        tables_dir / "T_r25_drift_integrated_gate_dual_counts.csv",
        index=False,
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
    manifest.to_csv(tables_dir / "T_r25_drift_integrated_gate_manifest.csv", index=False)
    print("[done] providers:", ", ".join(p for p, _ in DEFAULT_PROVIDERS))
    print("[done] wrote R25 cross-provider condition, overall, dual-count, and manifest tables")


if __name__ == "__main__":
    main()
