"""Cost-asymmetric decision layer.

Newsvendor pre-warming, concurrency-aware warm-pool sizing,
predictive keep-alive, and hybrid gate.
"""

import numpy as np
import torch
from scipy import stats


# Default cost ratios for Pareto sweep
DEFAULT_COST_RATIOS = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0])


def newsvendor_quantile(cost_ratio):
    """Compute optimal service quantile τ* = c_u / (c_u + c_o).

    Args:
        cost_ratio: ρ = c_u / c_o (cold-start cost / idle-memory cost)

    Returns:
        tau_star: optimal quantile level in [0, 1]
    """
    return cost_ratio / (1.0 + cost_ratio)


def quantile_to_prediction(pred_quantiles, quantile_levels, tau_star):
    """Extract the prediction at quantile tau_star from predicted quantiles.

    Linearly interpolates between the two nearest quantile levels.

    Args:
        pred_quantiles: [N, Q] predicted quantile values
        quantile_levels: [Q] quantile levels (e.g., 0.05, 0.10, ..., 0.95)
        tau_star: target quantile

    Returns:
        prediction: [N] interpolated prediction at tau_star
    """
    if isinstance(quantile_levels, torch.Tensor):
        quantile_levels = quantile_levels.numpy()
    if isinstance(pred_quantiles, torch.Tensor):
        pred_quantiles = pred_quantiles.detach().cpu().numpy()

    # Find bracketing indices
    idx = np.searchsorted(quantile_levels, tau_star)
    if idx == 0:
        return pred_quantiles[:, 0]
    if idx >= len(quantile_levels):
        return pred_quantiles[:, -1]

    # Linear interpolation
    q_lo = quantile_levels[idx - 1]
    q_hi = quantile_levels[idx]
    frac = (tau_star - q_lo) / (q_hi - q_lo + 1e-10)
    return pred_quantiles[:, idx - 1] * (1 - frac) + pred_quantiles[:, idx] * frac


def warm_pool_size_mginf(predicted_rate, mean_duration, tau_star, max_pool=20):
    """M/G/∞ approximation for warm pool sizing.

    Offered load a = λ̂ · E[D]
    Size w(t) = min{w : P(Poisson(a) > w) < 1 - τ*}

    Args:
        predicted_rate: predicted arrival rate (per minute)
        mean_duration: mean function execution duration (seconds)
        tau_star: service quantile
        max_pool: maximum pool size

    Returns:
        pool_size: recommended warm pool size
    """
    # Offered load in concurrent units
    offered_load = predicted_rate * (mean_duration / 60.0)  # convert duration to minutes

    if offered_load <= 0:
        return 0

    # Find w such that P(Poisson(a) > w) < 1 - τ*
    threshold = 1 - tau_star
    for w in range(max_pool + 1):
        p_exceed = 1.0 - stats.poisson.cdf(w, offered_load)
        if p_exceed < threshold:
            return w

    return max_pool


def warm_pool_size_montecarlo(pred_quantiles, quantile_levels, duration_samples,
                               tau_star, n_mc=1000, max_pool=20):
    """Monte Carlo warm pool sizing.

    Sample arrival counts from predicted distribution, sample durations,
    simulate overlap to estimate required concurrency.

    Args:
        pred_quantiles: [Q] quantile values for predicted rate
        quantile_levels: [Q] quantile levels
        duration_samples: array of empirical duration samples for this function
        tau_star: service quantile
        n_mc: number of MC samples
        max_pool: max pool size

    Returns:
        pool_size: recommended warm pool size
    """
    # Sample rates from predicted distribution (inverse CDF)
    u = np.random.uniform(0, 1, n_mc)
    rates = np.interp(u, quantile_levels, pred_quantiles)
    rates = np.maximum(rates, 0)

    # For each sample, compute max concurrency
    concurrencies = []
    for rate in rates:
        # Sample number of arrivals (Poisson with predicted rate)
        n_arrivals = np.random.poisson(max(rate, 0))
        if n_arrivals == 0:
            concurrencies.append(0)
            continue

        # Sample arrival times uniformly within the minute
        arrivals = np.sort(np.random.uniform(0, 60, n_arrivals))
        # Sample durations
        durations = np.random.choice(duration_samples, n_arrivals, replace=True)
        # Compute max concurrent at any arrival point
        max_conc = 0
        for i in range(n_arrivals):
            # How many previous invocations are still running?
            still_running = np.sum(
                (arrivals[:i + 1] + durations[:i + 1]) > arrivals[i]
            )
            max_conc = max(max_conc, still_running)
        concurrencies.append(max_conc)

    # Use tau_star quantile of concurrency distribution
    pool_size = int(np.quantile(concurrencies, tau_star))
    return min(pool_size, max_pool)


def predictive_keep_alive(pred_quantiles, quantile_levels, tau_star,
                           min_keep_alive=0.5, max_keep_alive=30.0):
    """Predictive keep-alive: retain container while P(next arrival within τ) > threshold.

    Uses time-to-next-invocation quantile from the predicted distribution.

    Args:
        pred_quantiles: [Q] quantile values for predicted rate
        quantile_levels: [Q]
        tau_star: service quantile (derived from cost ratio)
        min_keep_alive: minimum keep-alive in minutes
        max_keep_alive: maximum keep-alive in minutes

    Returns:
        keep_alive_min: keep-alive duration in minutes
    """
    # Predicted rate at tau_star
    rate = np.interp(tau_star, quantile_levels, pred_quantiles)
    rate = max(rate, 1e-6)

    # Expected time to next invocation: 1/rate (in minutes)
    expected_iat = 1.0 / rate

    # Keep alive for the (1-tau_star) quantile of the IAT
    # Assuming exponential IAT: quantile = -ln(1-q) / rate
    keep_alive = -np.log(1 - tau_star + 1e-10) / rate

    return np.clip(keep_alive, min_keep_alive, max_keep_alive)


class HybridGate:
    """Routes functions between learned predictor and histogram fallback.

    Sparse-tail functions (< min_invocations) or functions with too-wide
    prediction intervals → route to histogram baseline.
    """
    def __init__(self, min_invocations=100, max_interval_width=5.0):
        self.min_invocations = min_invocations
        self.max_interval_width = max_interval_width

    def should_use_learned(self, func_total_invocations, pred_quantiles=None,
                            quantile_levels=None):
        """Decide whether to use the learned predictor or histogram fallback.

        Returns:
            True if learned predictor should be used, False for histogram
        """
        if func_total_invocations < self.min_invocations:
            return False

        if pred_quantiles is not None and quantile_levels is not None:
            # Check prediction interval width (95% - 5%)
            low_idx = 0   # 0.05
            high_idx = -1  # 0.95
            interval_width = pred_quantiles[high_idx] - pred_quantiles[low_idx]
            if interval_width > self.max_interval_width:
                return False

        return True


class DecisionLayer:
    """Complete decision layer combining newsvendor, pool sizing, and keep-alive."""

    def __init__(self, cost_ratio=10.0, method="mginf", hybrid_gate=None):
        self.cost_ratio = cost_ratio
        self.tau_star = newsvendor_quantile(cost_ratio)
        self.method = method
        self.hybrid_gate = hybrid_gate or HybridGate()

    def decide(self, pred_quantiles, quantile_levels, mean_duration,
               func_total_invocations=1000, duration_samples=None):
        """Make pre-warming and keep-alive decisions.

        Args:
            pred_quantiles: [Q] predicted quantile values (in log1p space)
            quantile_levels: [Q] quantile levels
            mean_duration: mean execution duration (seconds)
            func_total_invocations: total invocations (for hybrid gate)
            duration_samples: empirical duration samples (for MC method)

        Returns:
            dict with:
                - prewarm_count: number of containers to prewarm
                - keep_alive_min: keep-alive duration in minutes
                - use_learned: whether learned predictor was used
                - predicted_rate: the predicted rate at tau_star
        """
        # Hybrid gate check
        use_learned = self.hybrid_gate.should_use_learned(
            func_total_invocations, pred_quantiles, quantile_levels
        )

        # Convert from log1p space to rate
        pred_rate_quantiles = np.expm1(np.maximum(pred_quantiles, 0))
        predicted_rate = np.interp(self.tau_star, quantile_levels, pred_rate_quantiles)

        # Pool sizing
        if self.method == "mginf":
            prewarm_count = warm_pool_size_mginf(
                predicted_rate, mean_duration, self.tau_star
            )
        elif self.method == "montecarlo" and duration_samples is not None:
            prewarm_count = warm_pool_size_montecarlo(
                pred_rate_quantiles, quantile_levels, duration_samples,
                self.tau_star
            )
        else:
            prewarm_count = warm_pool_size_mginf(
                predicted_rate, mean_duration, self.tau_star
            )

        # Keep-alive
        keep_alive = predictive_keep_alive(
            pred_rate_quantiles, quantile_levels, self.tau_star
        )

        return {
            "prewarm_count": prewarm_count,
            "keep_alive_min": keep_alive,
            "use_learned": use_learned,
            "predicted_rate": predicted_rate,
            "tau_star": self.tau_star,
        }
