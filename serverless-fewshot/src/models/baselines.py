"""Baselines B1-B10 and Oracle.

All forecasting baselines must produce quantile predictions for fair comparison
via the shared decision layer. Policy baselines (B1, B2) are evaluated end-to-end.
"""

import numpy as np
from scipy import stats
from collections import defaultdict


class FixedKeepAlive:
    """B1: Fixed keep-alive (OpenWhisk default, 10 min)."""
    def __init__(self, keep_alive_min=10):
        self.keep_alive = keep_alive_min

    def predict(self, func_idx, t, history=None):
        """Returns: keep_alive duration in minutes."""
        return self.keep_alive


class HybridHistogram:
    """B2: Hybrid histogram (Shahrad et al., ATC'20).

    Uses 4-hour IAT histogram + prewarm window + keep-alive.
    """
    def __init__(self, hist_window_min=240, n_bins=50):
        self.hist_window = hist_window_min
        self.n_bins = n_bins
        self.histograms = {}  # func_idx -> histogram

    def update(self, func_idx, arrival_times):
        """Update histogram with new arrival times."""
        if len(arrival_times) < 2:
            self.histograms[func_idx] = None
            return

        iats = np.diff(arrival_times)
        iats = iats[iats > 0]
        if len(iats) < 3:
            self.histograms[func_idx] = None
            return

        hist, bin_edges = np.histogram(iats, bins=self.n_bins, density=True)
        self.histograms[func_idx] = (hist, bin_edges)

    def predict_iat_quantile(self, func_idx, quantile=0.95):
        """Predict inter-arrival time at given quantile."""
        if func_idx not in self.histograms or self.histograms[func_idx] is None:
            return 10.0  # fallback

        hist, bin_edges = self.histograms[func_idx]
        cdf = np.cumsum(hist * np.diff(bin_edges))
        idx = np.searchsorted(cdf, quantile)
        if idx >= len(bin_edges) - 1:
            return bin_edges[-1]
        return bin_edges[idx]

    def predict_prewarm_keepalive(self, func_idx, quantile=0.05):
        """Returns (prewarm_window, keep_alive) in minutes."""
        prewarm = self.predict_iat_quantile(func_idx, quantile)
        keep_alive = self.predict_iat_quantile(func_idx, 0.95)
        return max(0.5, prewarm), min(keep_alive, 30.0)


class FourierPredictor:
    """B3: Fourier/harmonic extrapolation (IceBreaker-style)."""
    def __init__(self, n_harmonics=10):
        self.n_harmonics = n_harmonics
        self.models = {}

    def fit(self, func_idx, series):
        """Fit harmonic model to a count series."""
        n = len(series)
        fft_vals = np.fft.rfft(series.astype(np.float64))
        freqs = np.fft.rfftfreq(n)

        # Keep top harmonics by magnitude
        magnitudes = np.abs(fft_vals)
        top_indices = np.argsort(magnitudes)[-self.n_harmonics - 1:]

        # Zero out non-top harmonics
        filtered = np.zeros_like(fft_vals)
        filtered[top_indices] = fft_vals[top_indices]

        self.models[func_idx] = (filtered, n)

    def predict(self, func_idx, t_start, n_steps=1):
        """Predict next n_steps values."""
        if func_idx not in self.models:
            return np.zeros(n_steps)

        filtered, n = self.models[func_idx]
        reconstructed = np.fft.irfft(filtered, n=n)

        predictions = []
        for dt in range(n_steps):
            t = (t_start + dt) % n
            predictions.append(max(0, reconstructed[t]))

        return np.array(predictions)

    def predict_quantiles(self, func_idx, t, quantiles, n_steps=1):
        """Produce quantile predictions (spread around point forecast)."""
        point = self.predict(func_idx, t, n_steps)
        # Use Poisson-like spread
        results = np.zeros((n_steps, len(quantiles)))
        for i, p in enumerate(point):
            rate = max(p, 0.01)
            for j, q in enumerate(quantiles):
                results[i, j] = stats.poisson.ppf(q, rate)
        return results


class EWMAPredictor:
    """B4a: Exponentially Weighted Moving Average."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.states = {}

    def update(self, func_idx, value):
        if func_idx not in self.states:
            self.states[func_idx] = value
        else:
            self.states[func_idx] = self.alpha * value + (1 - self.alpha) * self.states[func_idx]

    def predict(self, func_idx):
        return self.states.get(func_idx, 0)

    def predict_quantiles(self, func_idx, quantiles):
        point = max(self.predict(func_idx), 0.01)
        return np.array([stats.poisson.ppf(q, point) for q in quantiles])


class SeasonalNaive:
    """B4b: Seasonal naive (t - 1440, i.e., same minute yesterday)."""
    def __init__(self, season=1440):
        self.season = season
        self.histories = {}

    def update(self, func_idx, t, value):
        if func_idx not in self.histories:
            self.histories[func_idx] = {}
        self.histories[func_idx][t] = value

    def predict(self, func_idx, t):
        h = self.histories.get(func_idx, {})
        t_prev = t - self.season
        return h.get(t_prev, 0)

    def predict_quantiles(self, func_idx, t, quantiles):
        point = max(self.predict(func_idx, t), 0.01)
        return np.array([stats.poisson.ppf(q, point) for q in quantiles])


class OraclePredictor:
    """Oracle: perfect knowledge of next-Δ arrivals."""
    def __init__(self, counts):
        self.counts = counts

    def predict(self, func_idx, t, horizon=1):
        t_target = min(t + horizon - 1, self.counts.shape[1] - 1)
        return self.counts[func_idx, t_target]

    def predict_quantiles(self, func_idx, t, quantiles, horizon=1):
        val = self.predict(func_idx, t, horizon)
        # Oracle returns degenerate distribution at the true value
        return np.full(len(quantiles), val)


class GlobalModelBaseline:
    """B5: Global model, no per-function adaptation.

    Uses the same body architecture but with a single shared head for all functions.
    Trained as a standard supervised model (no meta-learning).
    """
    pass  # Implemented as a variant of the meta-trainer with K=0


class PerFunctionLSTM:
    """B7: Per-function LSTM from scratch (cost comparison baseline).

    Only trained on a 500-function subsample.
    """
    pass  # Implemented separately for cost comparison
