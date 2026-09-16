# -*- coding: utf-8 -*-
"""Feature engineering utility, refactored from phase1_preprocess so features
can be regenerated for modified count series (drift injection, onboarding).

Channel order (must match features.npy):
  log_count, sin_h, cos_h, sin_d, cos_d, rm15, rs15, rm60, rs60, tsl, app_corate

app_corate (channel 10) is the log1p app-level aggregate excluding self; it is
unaffected by single-function drift injection, so callers pass it through from
the original feature tensor via `app_corate=`.
"""

import numpy as np


def _rolling_stat(arr, window):
    n_f, n_t = arr.shape
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


def features_from_counts(counts, t_offset=0, app_corate=None):
    """Rebuild the feature tensor from a count matrix.

    Args:
        counts: [N, T] (or [T] for a single function) invocation counts
        t_offset: absolute minute index of counts[:, 0] (for correct
            time-of-day / day-of-week encodings on trace segments)
        app_corate: [N, T] channel 10 passed through from the original
            feature tensor (zeros if None)

    Returns:
        features: [N, T, 11] float32 (or [T, 11] if input was 1-D)
    """
    single = counts.ndim == 1
    if single:
        counts = counts[None, :]
        if app_corate is not None and app_corate.ndim == 1:
            app_corate = app_corate[None, :]
    n_funcs, n_time = counts.shape

    log_counts = np.log1p(counts.astype(np.float64))

    minutes = np.arange(t_offset, t_offset + n_time)
    hour_of_day = (minutes % 1440) / 1440.0
    day_of_week = (minutes // 1440) / 7.0
    sin_hour = np.sin(2 * np.pi * hour_of_day).astype(np.float32)
    cos_hour = np.cos(2 * np.pi * hour_of_day).astype(np.float32)
    sin_dow = np.sin(2 * np.pi * day_of_week).astype(np.float32)
    cos_dow = np.cos(2 * np.pi * day_of_week).astype(np.float32)

    sin_hour_b = np.broadcast_to(sin_hour, (n_funcs, n_time))
    cos_hour_b = np.broadcast_to(cos_hour, (n_funcs, n_time))
    sin_dow_b = np.broadcast_to(sin_dow, (n_funcs, n_time))
    cos_dow_b = np.broadcast_to(cos_dow, (n_funcs, n_time))

    rm15, rs15 = _rolling_stat(log_counts, 15)
    rm60, rs60 = _rolling_stat(log_counts, 60)

    # time-since-last-invocation, vectorized:
    # last_inv[t] = running max of indices where count>0 (-1440 before first)
    t_idx = np.arange(n_time)
    marks = np.where(counts > 0, t_idx[None, :], -1440)
    last_inv = np.maximum.accumulate(marks, axis=1)
    tsl = np.minimum(t_idx[None, :] - last_inv, 1440).astype(np.float64)
    tsl = tsl / 1440.0

    if app_corate is None:
        app_corate = np.zeros((n_funcs, n_time), dtype=np.float32)

    feats = np.stack([
        log_counts, sin_hour_b, cos_hour_b, sin_dow_b, cos_dow_b,
        rm15, rs15, rm60, rs60, tsl, app_corate,
    ], axis=-1).astype(np.float32)

    return feats[0] if single else feats
