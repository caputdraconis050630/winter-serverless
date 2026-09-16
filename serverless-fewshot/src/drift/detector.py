"""Drift detection: PELT change-point detection, synthetic drift injection."""

import numpy as np
import pandas as pd
import ruptures


def detect_natural_drift(counts, func_ids, penalty="auto", min_size=60, max_cp_per_3days=1):
    """Run PELT change-point detection on each function's rate series.

    Args:
        counts: [N, T] count matrix
        func_ids: array of function IDs
        penalty: PELT penalty, or "auto" to tune
        min_size: minimum segment length in minutes
        max_cp_per_3days: target max change-points per 3-day window

    Returns:
        catalog: list of dicts with change-point info
    """
    n_funcs, n_time = counts.shape
    catalog = []

    for i in range(n_funcs):
        series = counts[i].astype(np.float64)

        # Skip very sparse functions
        if series.sum() < 50:
            continue

        # Downsample to hourly (60x reduction) for tractable PELT
        downsample_factor = 60
        hourly = series.reshape(-1, downsample_factor).sum(axis=1).astype(np.float64)
        n_hourly = len(hourly)

        # Smooth for stability (3-hour rolling mean)
        kernel = np.ones(3) / 3
        smoothed = np.convolve(hourly, kernel, mode="same")

        # Auto-tune penalty
        if penalty == "auto":
            pen = np.log(n_hourly) * smoothed.var()
            pen = max(pen, 1.0)
        else:
            pen = float(penalty)

        try:
            algo = ruptures.Pelt(model="l2", min_size=max(1, min_size // downsample_factor))
            result = algo.fit_predict(smoothed.reshape(-1, 1), pen=pen)
            # Map back to minute resolution
            change_points = [cp * downsample_factor for cp in result if cp < n_hourly]
        except Exception:
            change_points = []

        for cp in change_points:
            # Classify the shift
            pre_start = max(0, cp - 120)
            pre = series[pre_start:cp]
            post_end = min(n_time, cp + 120)
            post = series[cp:post_end]

            if len(pre) < 10 or len(post) < 10:
                continue

            pre_mean = pre.mean()
            post_mean = post.mean()
            pre_var = pre.var()
            post_var = post.var()

            # Classify shift type
            if pre_mean > 0 and abs(post_mean - pre_mean) / (pre_mean + 1e-10) > 0.3:
                shift_type = "level"
                magnitude = post_mean / (pre_mean + 1e-10)
            elif pre_var > 0 and abs(post_var - pre_var) / (pre_var + 1e-10) > 0.5:
                shift_type = "variance"
                magnitude = post_var / (pre_var + 1e-10)
            else:
                shift_type = "other"
                magnitude = 1.0

            catalog.append({
                "func_idx": i,
                "func_id": func_ids[i] if i < len(func_ids) else str(i),
                "change_point_minute": cp,
                "change_point_hour": cp / 60,
                "change_point_day": cp / 1440,
                "shift_type": shift_type,
                "magnitude": float(magnitude),
                "pre_mean": float(pre_mean),
                "post_mean": float(post_mean),
                "pre_var": float(pre_var),
                "post_var": float(post_var),
            })

    return catalog


def inject_synthetic_drift(counts, func_idx, t_d, drift_type, param):
    """Inject synthetic drift at time t_d.

    Args:
        counts: [N, T] count matrix (modified in place on a copy)
        func_idx: function index
        t_d: drift time (minute)
        drift_type: "scale", "phase", "period", "splice"
        param: drift parameter

    Returns:
        modified_counts: copy with drift injected
        drift_info: dict
    """
    modified = counts.copy()
    series = modified[func_idx].copy()
    n_time = len(series)

    if drift_type == "scale":
        # Scale shift: multiply post-drift by param
        series[t_d:] = series[t_d:] * param
        drift_info = {"type": "scale", "factor": param}

    elif drift_type == "phase":
        # Phase shift: shift the series by param minutes
        shift = int(param)
        post = series[t_d:].copy()
        if shift > 0:
            shifted = np.concatenate([np.zeros(shift), post[:len(post) - shift]])
        else:
            shifted = np.concatenate([post[-shift:], np.zeros(-shift)])
        series[t_d:t_d + len(shifted)] = shifted
        drift_info = {"type": "phase", "shift_minutes": shift}

    elif drift_type == "period":
        # Period change: resample post-drift to simulate periodicity change
        post = series[t_d:]
        # Stretch/compress to change dominant period
        factor = param  # e.g., 0.5 to double frequency, 2.0 to halve
        new_len = int(len(post) * factor)
        if new_len > 0 and new_len < n_time * 2:
            resampled = np.interp(
                np.linspace(0, len(post) - 1, len(post)),
                np.linspace(0, len(post) - 1, new_len),
                post[:new_len] if new_len <= len(post) else np.tile(post, 3)[:new_len]
            )
            series[t_d:t_d + len(resampled)] = resampled
        drift_info = {"type": "period", "factor": factor}

    elif drift_type == "splice":
        # Splice: replace post-drift with another function's pattern
        donor_idx = param  # index of donor function
        donor_series = counts[donor_idx]
        remaining = n_time - t_d
        series[t_d:] = donor_series[:remaining]
        drift_info = {"type": "splice", "donor_idx": donor_idx}

    else:
        raise ValueError(f"Unknown drift type: {drift_type}")

    modified[func_idx] = series
    return modified, drift_info


SYNTHETIC_DRIFT_CONFIGS = [
    # Scale shifts
    ("scale", 0.5), ("scale", 2.0), ("scale", 5.0),
    # Phase shifts (in minutes)
    ("phase", 120), ("phase", -120), ("phase", 360), ("phase", -360),
    # Period changes
    ("period", 0.5), ("period", 2.0),
]
