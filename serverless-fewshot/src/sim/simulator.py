"""Trace-driven discrete-event simulator for serverless cold-start evaluation.

Entities: arrivals (from trace), containers (cold-init -> warm-idle -> busy).
Decision loop each tick (Δ=60s): batched inference -> decision layer -> events.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Container:
    func_idx: int
    state: str  # "cold_init", "warm_idle", "busy"
    memory_mb: float = 256.0
    created_at: float = 0.0
    busy_until: float = 0.0
    idle_since: float = 0.0
    cold_init_done_at: float = 0.0


@dataclass
class SimMetrics:
    """Accumulates simulation metrics."""
    total_invocations: int = 0
    cold_starts: int = 0
    # Per-invocation latencies
    latencies: list = field(default_factory=list)
    # Wasted memory tracking: (start_time, end_time, memory_mb) for idle periods
    idle_periods: list = field(default_factory=list)
    # Per-function cold starts for CSR breakdown
    func_cold_starts: dict = field(default_factory=lambda: defaultdict(int))
    func_total_invocations: dict = field(default_factory=lambda: defaultdict(int))
    # For adaptation lag
    func_rolling_csr: dict = field(default_factory=lambda: defaultdict(list))
    # Total allocated memory-seconds
    total_allocated_mem_sec: float = 0.0
    total_idle_mem_sec: float = 0.0


def sample_cold_init_latency(runtime_class="python"):
    """Sample cold-init latency from lognormal distribution."""
    params = {
        "node": (np.log(0.8), 0.3),     # median ~0.8s
        "python": (np.log(1.0), 0.3),   # median ~1.0s
        "java": (np.log(3.5), 0.3),     # median ~3.5s
    }
    mu, sigma = params.get(runtime_class, params["python"])
    return np.random.lognormal(mu, sigma)


class ServerlessSimulator:
    """Discrete-event simulator for serverless platform."""

    def __init__(self, n_functions, counts, duration_stats, tick_sec=60,
                 default_memory_mb=256, runtime_class="python"):
        """
        Args:
            n_functions: number of functions to simulate
            counts: [N, T] per-minute invocation counts
            duration_stats: dict with 'mean' and 'std' per function
            tick_sec: simulation tick in seconds
            default_memory_mb: default container memory
            runtime_class: for cold-init latency sampling
        """
        self.n_functions = n_functions
        self.counts = counts
        self.duration_stats = duration_stats
        self.tick_sec = tick_sec
        self.memory_mb = default_memory_mb
        self.runtime_class = runtime_class
        self.n_ticks = counts.shape[1]

        # Container pool per function
        self.containers = defaultdict(list)  # func_idx -> [Container]
        self.metrics = SimMetrics()

    def _get_warm_containers(self, func_idx, current_time):
        """Get list of warm-idle containers for a function."""
        return [c for c in self.containers[func_idx]
                if c.state == "warm_idle" and c.cold_init_done_at <= current_time]

    def _get_all_containers(self, func_idx):
        """Get all containers (any state) for a function."""
        return self.containers[func_idx]

    def process_tick(self, tick, prewarm_decisions, keep_alive_decisions):
        """Process one simulation tick.

        Args:
            tick: current tick index (0-based)
            prewarm_decisions: dict {func_idx: n_containers_to_prewarm}
            keep_alive_decisions: dict {func_idx: keep_alive_minutes}

        Returns:
            tick_metrics: dict with per-tick stats
        """
        current_time = tick * self.tick_sec / 60.0  # in minutes
        tick_cold = 0
        tick_total = 0

        for fi in range(self.n_functions):
            n_arrivals = int(self.counts[fi, tick])
            if n_arrivals == 0 and fi not in prewarm_decisions:
                continue

            # 1. Apply keep-alive expiry: evict idle containers past keep-alive
            keep_alive = keep_alive_decisions.get(fi, 10.0)  # default 10 min
            alive_containers = []
            for c in self.containers[fi]:
                if c.state == "warm_idle":
                    idle_duration = current_time - c.idle_since
                    if idle_duration > keep_alive:
                        # Evict: record idle period
                        self.metrics.idle_periods.append(
                            (c.idle_since, current_time, c.memory_mb)
                        )
                        self.metrics.total_idle_mem_sec += (
                            idle_duration * 60 * c.memory_mb / 1024
                        )
                        continue  # remove from pool
                alive_containers.append(c)
            self.containers[fi] = alive_containers

            # 2. Apply prewarm decisions: create new containers
            n_prewarm = prewarm_decisions.get(fi, 0)
            current_warm = len(self._get_warm_containers(fi, current_time))
            n_to_create = max(0, n_prewarm - current_warm)
            for _ in range(n_to_create):
                cold_lat = sample_cold_init_latency(self.runtime_class)
                c = Container(
                    func_idx=fi,
                    state="cold_init",
                    memory_mb=self.memory_mb,
                    created_at=current_time,
                    cold_init_done_at=current_time + cold_lat / 60.0,
                )
                self.containers[fi].append(c)

            # 3. Transition cold-init containers that are ready
            for c in self.containers[fi]:
                if c.state == "cold_init" and c.cold_init_done_at <= current_time:
                    c.state = "warm_idle"
                    c.idle_since = current_time

            # 4. Process arrivals
            for _ in range(n_arrivals):
                tick_total += 1
                self.metrics.total_invocations += 1
                self.metrics.func_total_invocations[fi] += 1

                # Sample duration
                dur_mean = self.duration_stats.get(fi, {}).get("mean", 1.0)
                dur_std = self.duration_stats.get(fi, {}).get("std", 0.5)
                duration = max(0.01, np.random.normal(dur_mean, max(dur_std, 0.01)))

                # Try to find a warm container
                warm = self._get_warm_containers(fi, current_time)
                busy_done = [c for c in self.containers[fi]
                             if c.state == "busy" and c.busy_until <= current_time]
                for c in busy_done:
                    c.state = "warm_idle"
                    c.idle_since = current_time
                    warm.append(c)

                warm = self._get_warm_containers(fi, current_time)

                if warm:
                    # Use warm container
                    c = warm[0]
                    if c.state == "warm_idle":
                        idle_dur = current_time - c.idle_since
                        self.metrics.total_idle_mem_sec += (
                            idle_dur * 60 * c.memory_mb / 1024
                        )
                    c.state = "busy"
                    c.busy_until = current_time + duration / 60.0
                    self.metrics.latencies.append(duration)
                    self.metrics.total_allocated_mem_sec += duration * c.memory_mb / 1024
                else:
                    # Cold start!
                    tick_cold += 1
                    self.metrics.cold_starts += 1
                    self.metrics.func_cold_starts[fi] += 1
                    cold_lat = sample_cold_init_latency(self.runtime_class)
                    total_lat = cold_lat + duration
                    self.metrics.latencies.append(total_lat)

                    c = Container(
                        func_idx=fi, state="busy",
                        memory_mb=self.memory_mb,
                        created_at=current_time,
                        busy_until=current_time + total_lat / 60.0,
                        cold_init_done_at=current_time + cold_lat / 60.0,
                    )
                    self.containers[fi].append(c)
                    self.metrics.total_allocated_mem_sec += total_lat * c.memory_mb / 1024

            # Record rolling CSR for this function
            fc = self.metrics.func_cold_starts[fi]
            ft = self.metrics.func_total_invocations[fi]
            if ft > 0:
                self.metrics.func_rolling_csr[fi].append((tick, fc / ft))

        return {"cold_starts": tick_cold, "total": tick_total}

    def compute_final_metrics(self):
        """Compute final simulation metrics."""
        m = self.metrics

        # CSR
        csr = m.cold_starts / max(1, m.total_invocations)

        # WM (wasted memory)
        wm_total_gb_s = m.total_idle_mem_sec  # already in GB·s
        wm_per_1k = wm_total_gb_s / max(1, m.total_invocations / 1000)
        wm_fraction = m.total_idle_mem_sec / max(1e-10,
            m.total_idle_mem_sec + m.total_allocated_mem_sec)

        # Latency stats
        if m.latencies:
            lats = np.array(m.latencies)
            p50 = np.percentile(lats, 50)
            p95 = np.percentile(lats, 95)
            p99 = np.percentile(lats, 99)
        else:
            p50 = p95 = p99 = 0

        # Per-bucket CSR
        bucket_csr = {}
        for fi in range(self.n_functions):
            total = m.func_total_invocations.get(fi, 0)
            cold = m.func_cold_starts.get(fi, 0)
            if total > 0:
                bucket_csr[fi] = cold / total

        return {
            "csr": csr,
            "cold_starts": m.cold_starts,
            "total_invocations": m.total_invocations,
            "wm_total_gb_s": wm_total_gb_s,
            "wm_per_1k_inv": wm_per_1k,
            "wm_fraction": wm_fraction,
            "latency_p50": p50,
            "latency_p95": p95,
            "latency_p99": p99,
            "per_func_csr": bucket_csr,
        }


def compute_adaptation_lag(rolling_csr_list, event_tick, epsilon=0.1,
                           window=15, stability_window=30):
    """Compute adaptation lag AL(ε).

    AL = inf{t >= t_event : rolling_CSR(t') <= (1+ε)·CSR_steady for all t' in [t, t+W]}

    Args:
        rolling_csr_list: list of (tick, csr) pairs
        event_tick: tick of onboarding/drift event
        epsilon: tolerance
        window: rolling window in ticks (minutes)
        stability_window: stability window W in ticks

    Returns:
        al_ticks: adaptation lag in ticks (minutes), or None if never converged
    """
    if not rolling_csr_list:
        return None

    ticks = np.array([t for t, _ in rolling_csr_list])
    csrs = np.array([c for _, c in rolling_csr_list])

    # Steady-state CSR: last 25% of the series
    n = len(csrs)
    if n < 10:
        return None

    steady_csr = np.mean(csrs[int(0.75 * n):])
    threshold = (1 + epsilon) * steady_csr

    # Find first tick after event where CSR stays below threshold
    post_event = ticks >= event_tick
    if not post_event.any():
        return None

    post_ticks = ticks[post_event]
    post_csrs = csrs[post_event]

    for i in range(len(post_ticks)):
        # Check if all values in [i, i+stability_window] are below threshold
        window_end = post_ticks[i] + stability_window
        mask = (post_ticks >= post_ticks[i]) & (post_ticks <= window_end)
        if mask.sum() >= stability_window // 2:  # need sufficient coverage
            if np.all(post_csrs[mask] <= threshold + 1e-10):
                return post_ticks[i] - event_tick

    return None
