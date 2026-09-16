#!/usr/bin/env python3
"""Phase 2: Task Construction & Benchmark Scenarios.

Builds splits, drift catalog, validates episode construction.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.task_builder import TaskBuilder
from src.drift.detector import detect_natural_drift, SYNTHETIC_DRIFT_CONFIGS

PROCESSED_DIR = PROJECT_ROOT / "data" / os.environ.get("SF_DATA_DIR", "processed")
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"


def main():
    print("=" * 60)
    print("PHASE 2: Task Construction & Benchmark Scenarios")
    print("=" * 60)

    # Load processed data
    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    func_df = pd.read_csv(PROCESSED_DIR / "active_functions.csv")
    func_ids = np.load(PROCESSED_DIR / "func_ids.npy", allow_pickle=True)

    print(f"Loaded: {features.shape[0]} functions, {features.shape[1]} timesteps, {features.shape[2]} features")

    # Build task builder
    tb = TaskBuilder(features, counts, func_df, context_len=60, horizons=(1, 5, 15))

    # -------- Splits --------
    print("\n--- Split S1: Random hold-out ---")
    s1 = tb.split_s1_random(seed=42)
    print(f"  Train: {len(s1['train'])}, Val: {len(s1['val'])}, Test: {len(s1['test'])}")

    print("\n--- Split S2: Cluster hold-out ---")
    s2 = tb.split_s2_cluster_holdout(n_clusters=12, holdout_clusters=3, seed=42)
    print(f"  Train: {len(s2['train'])}, Val: {len(s2['val'])}, Test: {len(s2['test'])}")
    print(f"  Held-out clusters: {s2['held_out_clusters']}")

    # Print cluster distribution
    labels = s2["cluster_labels"]
    for c in range(12):
        n = (labels == c).sum()
        held = "** HELD OUT **" if c in s2["held_out_clusters"] else ""
        print(f"    Cluster {c:2d}: {n:3d} functions {held}")

    print("\n--- Split S3: Temporal ---")
    s3 = tb.split_s3_temporal(train_days=10, seed=42)
    print(f"  Train functions: {len(s3['train'])}, Val: {len(s3['val'])}")
    print(f"  Train period: 0-{s3['train_t_end']} min ({s3['train_t_end']/1440:.0f} days)")
    print(f"  Test period: {s3['test_t_start']}-{features.shape[1]} min")

    # Save splits
    np.savez(PROCESSED_DIR / "splits.npz",
             s1_train=s1["train"], s1_val=s1["val"], s1_test=s1["test"],
             s2_train=s2["train"], s2_val=s2["val"], s2_test=s2["test"],
             s2_labels=s2["cluster_labels"], s2_held_out=np.array(s2["held_out_clusters"]),
             s3_train=s3["train"], s3_val=s3["val"], s3_test=s3["test"],
             s3_train_t_end=np.array([s3["train_t_end"]]),
             s3_test_t_start=np.array([s3["test_t_start"]]))

    # -------- Test episode building --------
    print("\n--- Episode construction test ---")
    for k in [1, 5, 10, 20]:
        ep = tb.build_episode(0, k_support=k, k_query=32)
        print(f"  K={k:2d}: support {ep.support_x.shape}, query {ep.query_x.shape}, "
              f"target {ep.query_y.shape}")

    # -------- Onboarding scenario test --------
    print("\n--- Onboarding scenario test ---")
    episodes = tb.onboarding_scenario(0, k_checkpoints=(0, 1, 5, 10, 20))
    for ep in episodes:
        print(f"  K={ep.meta['k']:2d}: support {ep.support_x.shape}, "
              f"query {ep.query_x.shape}")

    # -------- Drift detection --------
    print("\n--- Natural drift detection (PELT) ---")
    catalog = detect_natural_drift(counts, func_ids)
    print(f"  Detected {len(catalog)} change-points across {len(set(d['func_idx'] for d in catalog))} functions")

    if catalog:
        cat_df = pd.DataFrame(catalog)
        # Summary
        print(f"\n  Shift types:")
        for st, cnt in cat_df["shift_type"].value_counts().items():
            print(f"    {st}: {cnt}")
        print(f"  Mean magnitude: {cat_df['magnitude'].mean():.2f}")
        print(f"  Change-points per function: {cat_df.groupby('func_idx').size().mean():.1f}")

        # Save catalog
        cat_df.to_csv(RESULTS_DIR / "T2_drift_catalog.csv", index=False)
        print(f"\n  Saved drift catalog to T2_drift_catalog.csv")
    else:
        print("  No change-points detected (functions may be too short/sparse)")
        # Create empty catalog
        cat_df = pd.DataFrame(columns=["func_idx", "func_id", "change_point_minute",
                                        "shift_type", "magnitude"])
        cat_df.to_csv(RESULTS_DIR / "T2_drift_catalog.csv", index=False)

    # -------- Synthetic drift configs --------
    print(f"\n--- Synthetic drift configurations ---")
    for dtype, param in SYNTHETIC_DRIFT_CONFIGS:
        print(f"  {dtype}: {param}")

    print("\n" + "=" * 60)
    print("Phase 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
