# -*- coding: utf-8 -*-
"""Online drift triggers for head-refit decisions (WP4).

All triggers consume a per-tick scalar signal (prediction residual or
nonconformity score) via update(value) -> bool (True = drift detected).
Implementations are self-contained (no external streaming libraries).
"""

import numpy as np
from collections import deque


class PageHinkley:
    """Page-Hinkley test on the residual stream.

    Detects an increase in the mean of |residuals|.
    """

    def __init__(self, delta=0.005, threshold=1.0, min_samples=30):
        self.delta = delta
        self.threshold = threshold
        self.min_samples = min_samples
        self.reset()

    def reset(self):
        self.mean = 0.0
        self.n = 0
        self.cum = 0.0
        self.min_cum = 0.0

    def update(self, value):
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.cum += value - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        if self.n >= self.min_samples and (self.cum - self.min_cum) > self.threshold:
            self.reset()
            return True
        return False


class AdwinLite:
    """Simplified ADWIN: compare recent-window mean vs reference-window mean
    with a Hoeffding-style bound.
    """

    def __init__(self, window=120, recent=30, delta=0.002, min_samples=60):
        self.buf = deque(maxlen=window)
        self.recent = recent
        self.delta = delta
        self.min_samples = min_samples

    def reset(self):
        self.buf.clear()

    def update(self, value):
        self.buf.append(value)
        n = len(self.buf)
        if n < self.min_samples:
            return False
        arr = np.array(self.buf)
        ref, rec = arr[:-self.recent], arr[-self.recent:]
        n0, n1 = len(ref), len(rec)
        if n0 < 10:
            return False
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        var = max(arr.var(), 1e-12)
        eps = np.sqrt(2.0 / m * var * np.log(2.0 / self.delta)) \
            + 2.0 / (3.0 * m) * np.log(2.0 / self.delta)
        if abs(rec.mean() - ref.mean()) > eps:
            self.reset()
            return True
        return False


class ConformalMonitor:
    """Split-conformal coverage monitor.

    Maintains a calibration set of nonconformity scores (|residual|).
    Each tick, checks whether the realized residual falls inside the
    (1-alpha) conformal interval. Triggers when the rolling coverage drops
    below (1 - alpha - delta) for m consecutive windows.
    """

    def __init__(self, alpha=0.1, delta=0.15, m_consecutive=3,
                 calib_size=120, window=30, min_calib=30):
        self.alpha = alpha
        self.delta = delta
        self.m = m_consecutive
        self.calib = deque(maxlen=calib_size)
        self.covered = deque(maxlen=window)
        self.window = window
        self.min_calib = min_calib
        self.violations = 0

    def reset(self):
        self.calib.clear()
        self.covered.clear()
        self.violations = 0

    def update(self, value):
        """value = |residual| for the current tick."""
        if len(self.calib) >= self.min_calib:
            q = np.quantile(np.array(self.calib), 1.0 - self.alpha)
            self.covered.append(1.0 if value <= q else 0.0)
            if len(self.covered) >= self.window:
                cov = float(np.mean(self.covered))
                if cov < 1.0 - self.alpha - self.delta:
                    self.violations += 1
                    if self.violations >= self.m:
                        self.reset()
                        return True
                else:
                    self.violations = 0
        self.calib.append(value)
        return False


class PeriodicTrigger:
    """Fires unconditionally every `interval` ticks (baseline)."""

    def __init__(self, interval=360):
        self.interval = interval
        self.n = 0

    def reset(self):
        self.n = 0

    def update(self, value):
        self.n += 1
        if self.n >= self.interval:
            self.n = 0
            return True
        return False


def build_triggers():
    """Standard trigger set for the T5 comparison."""
    return {
        "Periodic (6h)": PeriodicTrigger(interval=360),
        "Conformal coverage": ConformalMonitor(alpha=0.1, delta=0.15,
                                               m_consecutive=3),
        "ADWIN-lite": AdwinLite(),
        "Page-Hinkley": PageHinkley(),
    }
