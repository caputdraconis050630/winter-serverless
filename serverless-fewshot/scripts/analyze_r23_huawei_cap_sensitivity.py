#!/usr/bin/env python3
"""Summarize Huawei natural-drift max-window-inv sensitivity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analyze_r23_drift_v2_outputs import (
    add_best_winter_columns,
    pairwise_against_frozen,
    summarize_weighted,
)


INPUTS = (
    ("huawei_cap50k", "T_r23_drift_v2_events_huawei.csv"),
    ("huawei_cap100k", "T_r23_drift_v2_events_huawei_natcap100k.csv"),
    ("huawei_cap200k", "T_r23_drift_v2_events_huawei_natcap200k.csv"),
)


def main() -> None:
    tables_dir = Path("results/tables")
    frames = []
    manifest_rows = []
    for provider, filename in INPUTS:
        path = tables_dir / filename
        frame = pd.read_csv(path)
        frame = frame[frame["source"] == "natural"].copy()
        frame.insert(0, "origin_provider", provider)
        frame.insert(0, "provider", provider)
        frames.append(frame)
        events = frame[["event_id"]].drop_duplicates().shape[0]
        event_seeds = frame[["event_id", "seed"]].drop_duplicates().shape[0]
        manifest_rows.append(
            {
                "provider": provider,
                "filename": filename,
                "event_rows": int(frame.shape[0]),
                "n_events": int(events),
                "n_event_seeds": int(event_seeds),
            }
        )

    events = pd.concat(frames, ignore_index=True)
    condition = summarize_weighted(events, ["provider", "source", "kind", "rho", "arm"])
    overall = summarize_weighted(events, ["provider", "source", "rho", "arm"])
    pairwise = pairwise_against_frozen(events, ["provider", "source", "kind", "rho"])
    pairwise_overall = pairwise_against_frozen(events, ["provider", "source", "rho"])
    best = add_best_winter_columns(condition, pairwise)

    condition.to_csv(tables_dir / "T_r23_drift_v2_huawei_cap_sensitivity_condition_summary.csv", index=False)
    overall.to_csv(tables_dir / "T_r23_drift_v2_huawei_cap_sensitivity_overall.csv", index=False)
    pairwise.to_csv(tables_dir / "T_r23_drift_v2_huawei_cap_sensitivity_pairwise.csv", index=False)
    pairwise_overall.to_csv(
        tables_dir / "T_r23_drift_v2_huawei_cap_sensitivity_pairwise_overall.csv", index=False
    )
    best.to_csv(tables_dir / "T_r23_drift_v2_huawei_cap_sensitivity_winter_best.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(
        tables_dir / "T_r23_drift_v2_huawei_cap_sensitivity_manifest.csv", index=False
    )

    print("[done] wrote Huawei cap sensitivity tables")


if __name__ == "__main__":
    main()
