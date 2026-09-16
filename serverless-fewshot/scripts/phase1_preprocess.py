#!/usr/bin/env python3
"""Phase 1: Data Acquisition & Preprocessing.

Builds per-function time series at 1-minute granularity from Azure 2021 traces,
computes per-function descriptors, partitions into active/sparse pools,
performs feature engineering, and persists as memory-mappable arrays.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import fft as scipy_fft
from sklearn.preprocessing import StandardScaler

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = Path("/data/260427/dataset/azure")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
TRACE_DURATION_SEC = 14 * 24 * 3600  # 14 days
GRANULARITY_SEC = 60  # 1 minute
N_MINUTES = TRACE_DURATION_SEC // GRANULARITY_SEC  # 20160
MIN_INVOCATIONS = 100  # active pool threshold
MAX_FUNCTIONS = 50000  # sample cap

FREQ_BUCKETS = {
    "<1/day": (0, 1 / 1440),
    "1/day-1/h": (1 / 1440, 1 / 60),
    "1/h-1/min": (1 / 60, 1.0),
    ">1/min": (1.0, float("inf")),
}


def load_raw_trace():
    """Load the Azure 2021 invocation trace."""
    print("Loading raw trace...")
    t0 = time.time()
    trace_file = RAW_DIR / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
    df = pd.read_csv(trace_file, dtype={
        "app": str, "func": str, "end_timestamp": np.float64, "duration": np.float64
    })
    print(f"  Loaded {len(df):,} invocations in {time.time()-t0:.1f}s")
    return df


def build_per_function_timeseries(df):
    """Build per-function 1-minute count time series."""
    print("Building per-function time series...")
    t0 = time.time()

    # Compute start timestamp (arrival = end - duration)
    df["arrival"] = df["end_timestamp"] - df["duration"]
    df["minute_bin"] = (df["arrival"] / GRANULARITY_SEC).astype(np.int64).clip(0, N_MINUTES - 1)

    # Group by function, build count series
    func_ids = df["func"].unique()
    n_funcs = len(func_ids)
    func_to_idx = {f: i for i, f in enumerate(func_ids)}

    # Also track app mapping
    func_app_map = df.groupby("func")["app"].first().to_dict()

    print(f"  {n_funcs} unique functions")

    # Build count matrix [n_funcs, N_MINUTES]
    counts = np.zeros((n_funcs, N_MINUTES), dtype=np.float32)
    for func_id, minute_bin in zip(df["func"].values, df["minute_bin"].values):
        counts[func_to_idx[func_id], minute_bin] += 1

    # Per-function duration statistics (for concurrency sim later)
    dur_stats = df.groupby("func")["duration"].agg(["mean", "std", "median", "count"])
    dur_stats = dur_stats.reindex(func_ids)

    print(f"  Built count matrix {counts.shape} in {time.time()-t0:.1f}s")
    return counts, func_ids, func_to_idx, func_app_map, dur_stats


def compute_descriptors(counts, func_ids, func_app_map, dur_stats):
    """Compute per-function descriptors for clustering and stratification."""
    print("Computing per-function descriptors...")
    t0 = time.time()

    n_funcs = len(func_ids)
    total_invocations = counts.sum(axis=1)
    mean_rate = total_invocations / N_MINUTES  # per minute

    descriptors = []
    for i in range(n_funcs):
        series = counts[i]
        total = total_invocations[i]

        # Inter-arrival time stats (from non-zero minutes)
        nonzero_idx = np.nonzero(series)[0]
        if len(nonzero_idx) > 1:
            iats = np.diff(nonzero_idx).astype(np.float64)
            iat_mean = iats.mean()
            iat_std = iats.std()
            iat_cv = iat_std / iat_mean if iat_mean > 0 else 0
        else:
            iat_cv = 0
            iat_mean = N_MINUTES

        # Burstiness (Fano factor) = var/mean
        fano = series.var() / series.mean() if series.mean() > 0 else 0

        # Sparsity
        sparsity = (series == 0).mean()

        # Spectral energy at 24h and 7d periods
        if total > 10:
            fft_vals = np.abs(scipy_fft.rfft(series.astype(np.float64)))
            freqs = scipy_fft.rfftfreq(N_MINUTES, d=1.0)  # in cycles/minute

            # 24h period = 1/1440 cycles/min
            daily_idx = np.argmin(np.abs(freqs - 1 / 1440))
            daily_energy = fft_vals[daily_idx] / (fft_vals.sum() + 1e-10)

            # 7d period = 1/10080 cycles/min
            weekly_idx = np.argmin(np.abs(freqs - 1 / 10080))
            weekly_energy = fft_vals[weekly_idx] / (fft_vals.sum() + 1e-10)
        else:
            daily_energy = 0
            weekly_energy = 0

        descriptors.append({
            "func_id": func_ids[i],
            "app_id": func_app_map.get(func_ids[i], ""),
            "total_invocations": float(total),
            "mean_rate_per_min": float(mean_rate[i]),
            "iat_cv": float(iat_cv),
            "fano_factor": float(fano),
            "sparsity": float(sparsity),
            "daily_spectral_energy": float(daily_energy),
            "weekly_spectral_energy": float(weekly_energy),
            "mean_duration": float(dur_stats.iloc[i]["mean"]) if not np.isnan(dur_stats.iloc[i]["mean"]) else 0,
            "n_invocations_in_trace": int(dur_stats.iloc[i]["count"]),
        })

    desc_df = pd.DataFrame(descriptors)
    print(f"  Computed descriptors for {n_funcs} functions in {time.time()-t0:.1f}s")
    return desc_df


def assign_frequency_bucket(mean_rate):
    """Assign a function to a frequency bucket based on its mean rate."""
    for name, (lo, hi) in FREQ_BUCKETS.items():
        if lo <= mean_rate < hi:
            return name
    return ">1/min"


def partition_and_sample(desc_df, counts):
    """Partition into active/sparse pools and stratified-sample if needed."""
    print("Partitioning and sampling...")

    desc_df["freq_bucket"] = desc_df["mean_rate_per_min"].apply(assign_frequency_bucket)

    # Active vs sparse
    active_mask = desc_df["total_invocations"] >= MIN_INVOCATIONS
    active_df = desc_df[active_mask].copy()
    sparse_df = desc_df[~active_mask].copy()

    print(f"  Active pool: {len(active_df)} functions")
    print(f"  Sparse tail: {len(sparse_df)} functions")

    # Stratified sampling if needed
    if len(active_df) > MAX_FUNCTIONS:
        print(f"  Stratified sampling to {MAX_FUNCTIONS} functions...")
        bucket_counts = active_df["freq_bucket"].value_counts()
        bucket_fracs = bucket_counts / bucket_counts.sum()

        sampled_indices = []
        for bucket, frac in bucket_fracs.items():
            n_sample = max(1, int(frac * MAX_FUNCTIONS))
            bucket_df = active_df[active_df["freq_bucket"] == bucket]
            if len(bucket_df) > n_sample:
                sampled = bucket_df.sample(n=n_sample, random_state=42)
            else:
                sampled = bucket_df
            sampled_indices.extend(sampled.index.tolist())

        active_df = active_df.loc[sampled_indices]
        print(f"  Sampled active pool: {len(active_df)} functions")

    # Get indices into counts array
    active_indices = active_df.index.values
    sparse_indices = sparse_df.index.values

    return active_df, sparse_df, active_indices, sparse_indices


def feature_engineering(counts, active_indices):
    """Feature engineering per window: log1p counts, cyclical encodings, rolling stats, etc."""
    print("Feature engineering...")
    t0 = time.time()

    active_counts = counts[active_indices]  # [n_active, N_MINUTES]
    n_funcs, n_time = active_counts.shape

    # 1. log1p counts
    log_counts = np.log1p(active_counts)  # [n_funcs, n_time]

    # 2. Cyclical encodings
    minutes = np.arange(n_time)
    hour_of_day = (minutes % 1440) / 1440.0
    day_of_week = (minutes // 1440) / 7.0

    sin_hour = np.sin(2 * np.pi * hour_of_day).astype(np.float32)
    cos_hour = np.cos(2 * np.pi * hour_of_day).astype(np.float32)
    sin_dow = np.sin(2 * np.pi * day_of_week).astype(np.float32)
    cos_dow = np.cos(2 * np.pi * day_of_week).astype(np.float32)

    # Broadcast temporal features to [n_funcs, n_time]
    sin_hour_b = np.broadcast_to(sin_hour, (n_funcs, n_time))
    cos_hour_b = np.broadcast_to(cos_hour, (n_funcs, n_time))
    sin_dow_b = np.broadcast_to(sin_dow, (n_funcs, n_time))
    cos_dow_b = np.broadcast_to(cos_dow, (n_funcs, n_time))

    # 3. Rolling mean/std (15 and 60 min)
    def rolling_stat(arr, window):
        """Compute rolling mean and std along time axis."""
        n_f, n_t = arr.shape
        # Use cumsum for efficiency
        cumsum = np.cumsum(arr, axis=1)
        cumsum2 = np.cumsum(arr ** 2, axis=1)

        roll_mean = np.zeros_like(arr)
        roll_std = np.zeros_like(arr)

        for w in range(n_t):
            start = max(0, w - window + 1)
            n = w - start + 1
            if w >= window:
                s = cumsum[:, w] - cumsum[:, start - 1]
                s2 = cumsum2[:, w] - cumsum2[:, start - 1]
            else:
                s = cumsum[:, w]
                s2 = cumsum2[:, w]
            roll_mean[:, w] = s / n
            var = s2 / n - (s / n) ** 2
            roll_std[:, w] = np.sqrt(np.maximum(var, 0))

        return roll_mean, roll_std

    roll_mean_15, roll_std_15 = rolling_stat(log_counts, 15)
    roll_mean_60, roll_std_60 = rolling_stat(log_counts, 60)

    # 4. Time since last invocation (in minutes, capped)
    time_since_last = np.full((n_funcs, n_time), 1440, dtype=np.float32)
    for i in range(n_funcs):
        last_inv = -1440
        for t in range(n_time):
            if active_counts[i, t] > 0:
                last_inv = t
            time_since_last[i, t] = min(t - last_inv, 1440)

    # Normalize time_since_last
    time_since_last = time_since_last / 1440.0

    # 5. Stack features: [n_funcs, n_time, n_features]
    # Features: log_count, sin_h, cos_h, sin_d, cos_d, rm15, rs15, rm60, rs60, tsl
    features = np.stack([
        log_counts,
        sin_hour_b, cos_hour_b,
        sin_dow_b, cos_dow_b,
        roll_mean_15, roll_std_15,
        roll_mean_60, roll_std_60,
        time_since_last,
    ], axis=-1).astype(np.float32)

    print(f"  Feature tensor shape: {features.shape} ({features.nbytes / 1e9:.2f} GB)")
    print(f"  Features: log_count, sin_h, cos_h, sin_d, cos_d, rm15, rs15, rm60, rs60, tsl")
    print(f"  Feature engineering done in {time.time()-t0:.1f}s")
    return features


def compute_app_corates(counts, active_indices, desc_df):
    """Compute app-level aggregate rate (co-invocation signal) for active functions."""
    print("Computing app-level co-invocation rates...")

    active_desc = desc_df.iloc[active_indices]
    app_ids = active_desc["app_id"].values
    unique_apps = np.unique(app_ids)

    # Build app aggregate: sum of all functions in the same app (excluding self)
    app_rates = np.zeros((len(active_indices), N_MINUTES), dtype=np.float32)
    app_to_funcs = {}
    for i, app in enumerate(app_ids):
        if app not in app_to_funcs:
            app_to_funcs[app] = []
        app_to_funcs[app].append(i)

    for app, func_indices in app_to_funcs.items():
        if len(func_indices) > 1:
            app_total = counts[active_indices[func_indices]].sum(axis=0)
            for fi in func_indices:
                app_rates[fi] = np.log1p(app_total - counts[active_indices[fi]])

    return app_rates


def save_dataset_stats(active_df, sparse_df, desc_df):
    """Write T1 dataset statistics table."""
    print("Saving dataset statistics...")

    stats = []
    all_df = pd.concat([active_df, sparse_df])
    for bucket_name in FREQ_BUCKETS:
        bucket_data = all_df[all_df["freq_bucket"] == bucket_name]
        active_in_bucket = active_df[active_df["freq_bucket"] == bucket_name]
        stats.append({
            "frequency_bucket": bucket_name,
            "total_functions": len(bucket_data),
            "active_functions": len(active_in_bucket),
            "sparse_functions": len(bucket_data) - len(active_in_bucket),
            "total_invocations": int(bucket_data["total_invocations"].sum()),
            "mean_invocations": float(bucket_data["total_invocations"].mean()),
            "mean_rate_per_min": float(bucket_data["mean_rate_per_min"].mean()),
            "mean_sparsity": float(bucket_data["sparsity"].mean()),
            "mean_fano": float(bucket_data["fano_factor"].mean()),
        })

    stats.append({
        "frequency_bucket": "TOTAL",
        "total_functions": len(all_df),
        "active_functions": len(active_df),
        "sparse_functions": len(sparse_df),
        "total_invocations": int(all_df["total_invocations"].sum()),
        "mean_invocations": float(all_df["total_invocations"].mean()),
        "mean_rate_per_min": float(all_df["mean_rate_per_min"].mean()),
        "mean_sparsity": float(all_df["sparsity"].mean()),
        "mean_fano": float(all_df["fano_factor"].mean()),
    })

    stats_df = pd.DataFrame(stats)
    stats_path = RESULTS_DIR / "T1_dataset_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved to {stats_path}")
    print(stats_df.to_string(index=False))
    return stats_df


def main():
    print("=" * 60)
    print("PHASE 1: Data Acquisition & Preprocessing")
    print("=" * 60)

    # 1. Load raw trace
    df = load_raw_trace()

    # 2. Build per-function time series
    counts, func_ids, func_to_idx, func_app_map, dur_stats = build_per_function_timeseries(df)

    # 3. Compute descriptors
    desc_df = compute_descriptors(counts, func_ids, func_app_map, dur_stats)

    # 4. Partition and sample
    active_df, sparse_df, active_indices, sparse_indices = partition_and_sample(desc_df, counts)

    # 5. Feature engineering
    features = feature_engineering(counts, active_indices)

    # 6. App-level co-rates (extra feature)
    app_corates = compute_app_corates(counts, active_indices, desc_df)

    # Append co-rate as additional feature
    app_corates_expanded = app_corates[:, :, np.newaxis]
    features = np.concatenate([features, app_corates_expanded], axis=-1)
    print(f"  Final feature tensor: {features.shape} (11 features incl. app co-rate)")

    # 7. Save everything
    print("\nSaving processed data...")

    # Feature tensor
    np.save(PROCESSED_DIR / "features.npy", features)
    print(f"  features.npy: {features.shape}")

    # Raw counts for active functions
    active_counts = counts[active_indices]
    np.save(PROCESSED_DIR / "counts.npy", active_counts)
    print(f"  counts.npy: {active_counts.shape}")

    # Function metadata
    active_df = active_df.reset_index(drop=True)
    active_df.to_csv(PROCESSED_DIR / "active_functions.csv", index=False)
    sparse_df = sparse_df.reset_index(drop=True)
    sparse_df.to_csv(PROCESSED_DIR / "sparse_functions.csv", index=False)

    # Duration stats for active functions
    active_dur = dur_stats.iloc[active_indices].reset_index()
    active_dur.columns = ["func_id", "dur_mean", "dur_std", "dur_median", "dur_count"]
    active_dur.to_csv(PROCESSED_DIR / "duration_stats.csv", index=False)

    # Function ID mapping
    active_func_ids = func_ids[active_indices]
    np.save(PROCESSED_DIR / "func_ids.npy", active_func_ids)

    # All counts (including sparse) for potential later use
    np.save(PROCESSED_DIR / "all_counts.npy", counts)
    np.save(PROCESSED_DIR / "all_func_ids.npy", func_ids)

    # Save raw timestamps for D2 simulation (arrival, duration per function)
    print("  Saving raw arrival data for simulation...")
    arrival_data = df[["func", "arrival", "duration"]].copy()
    # Filter to active functions
    active_set = set(active_func_ids)
    active_arrivals = arrival_data[arrival_data["func"].isin(active_set)]
    active_arrivals.to_parquet(PROCESSED_DIR / "arrivals.parquet", index=False)
    print(f"  arrivals.parquet: {len(active_arrivals):,} invocations")

    # 8. Dataset statistics table
    stats_df = save_dataset_stats(active_df, sparse_df, desc_df)

    print("\n" + "=" * 60)
    print("Phase 1 COMPLETE")
    print(f"  Active functions: {len(active_df)}")
    print(f"  Feature shape: {features.shape}")
    print(f"  Time span: {N_MINUTES} minutes ({N_MINUTES/1440:.0f} days)")
    print("=" * 60)


if __name__ == "__main__":
    main()
