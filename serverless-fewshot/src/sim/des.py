# -*- coding: utf-8 -*-
"""Event-level discrete-event simulator for serverless cold-start evaluation.

Improvements over the tick-granularity fast_simulate:
- Arrivals within a minute get uniform random offsets (seconds resolution).
- Busy containers cannot serve concurrent requests (real concurrency).
- Cold-init latency is on the request path; prewarmed containers need
  init lead time before they can serve.
- Duration / cold-init / offsets are seed-driven -> real cross-seed variance.
- Per-function metrics returned for paired statistics and stratified tables.

All internal times are in SECONDS. keepalive matrices are in MINUTES
(converted internally). counts/prewarm/keepalive are [T] arrays per function.
"""

import numpy as np

# Container list encoding: each container is [ready_at, free_at]
# - ready_at: when cold init completes (container can serve from then on)
# - free_at:  when current request finishes (>= ready_at once serving)
# idle_start = max(ready_at, free_at); container idle from idle_start onward.

MAX_POOL = 200  # safety cap on per-function pool size


def simulate_function(counts, prewarm, keepalive, dur_mean, dur_std,
                      memory_mb=256.0, seed=0, tick_sec=60.0,
                      cold_mu=0.0, cold_sigma=0.3,
                      track_rolling=False):
    """Simulate one function's full timeline.

    Args:
        counts: [T] int arrivals per tick
        prewarm: [T] int prewarm pool targets per tick
        keepalive: [T] float keep-alive in MINUTES per tick
        dur_mean, dur_std: execution duration stats (seconds)
        cold_mu, cold_sigma: lognormal params for cold-init latency (seconds);
            median init = exp(cold_mu)
        track_rolling: if True, also return per-tick (cold, total) arrays

    Returns:
        dict with per-function totals; latencies as a list (seconds).
    """
    rng = np.random.default_rng(seed)
    T = len(counts)
    mem_gb = memory_mb / 1024.0
    end_time = T * tick_sec

    containers = []  # list of [ready_at, free_at]
    total_inv = 0
    cold = 0
    idle_mem_s = 0.0   # GB*s spent warm-idle (wasted)
    busy_mem_s = 0.0   # GB*s spent initializing or serving (allocated)
    latencies = []
    roll_cold = np.zeros(T, dtype=np.int32) if track_rolling else None
    roll_total = np.zeros(T, dtype=np.int32) if track_rolling else None

    dur_sigma = max(dur_std, 0.01)

    def evict_expired(now, ka_sec):
        """Remove containers idle longer than ka_sec before `now`."""
        nonlocal idle_mem_s
        kept = []
        for c in containers:
            idle_start = c[0] if c[0] > c[1] else c[1]
            if idle_start <= now - ka_sec:
                idle_mem_s += ka_sec * mem_gb  # sat idle exactly ka before dying
            else:
                kept.append(c)
        containers[:] = kept

    for t in range(T):
        now0 = t * tick_sec
        ka_sec = float(keepalive[t]) * 60.0

        # 1. Evict containers whose idle period expired before this tick
        evict_expired(now0, ka_sec)

        # 2. Prewarm to target (never scale down here; keep-alive handles that)
        target = int(prewarm[t])
        deficit = target - len(containers)
        if deficit > 0 and len(containers) < MAX_POOL:
            n_new = min(deficit, MAX_POOL - len(containers))
            inits = rng.lognormal(cold_mu, cold_sigma, n_new)
            for init in inits:
                containers.append([now0 + init, now0 + init])
                busy_mem_s += init * mem_gb  # init period = allocated

        # 3. Arrivals
        n = int(counts[t])
        if n > 0:
            offsets = np.sort(rng.uniform(0.0, tick_sec, n))
            durs = rng.normal(dur_mean, dur_sigma, n)
            tick_cold = 0

            for i in range(n):
                tau = now0 + offsets[i]
                dur = durs[i] if durs[i] > 0.01 else 0.01

                # Lazy eviction up to arrival time
                evict_expired(tau, ka_sec)

                # Find MRU warm-idle container (ready and free at tau)
                best = None
                best_idle = -1.0
                for c in containers:
                    if c[0] <= tau and c[1] <= tau:
                        idle_start = c[0] if c[0] > c[1] else c[1]
                        if idle_start > best_idle:
                            best_idle = idle_start
                            best = c

                if best is not None:
                    # Warm hit
                    idle_mem_s += (tau - best_idle) * mem_gb
                    best[1] = tau + dur
                    busy_mem_s += dur * mem_gb
                    latencies.append(dur)
                else:
                    # Cold start (also covers arrivals while others init/busy)
                    init = rng.lognormal(cold_mu, cold_sigma)
                    if len(containers) < MAX_POOL:
                        containers.append([tau + init, tau + init + dur])
                    busy_mem_s += (init + dur) * mem_gb
                    latencies.append(init + dur)
                    cold += 1
                    tick_cold += 1

                total_inv += 1

            if track_rolling:
                roll_cold[t] = tick_cold
                roll_total[t] = n

    # Close out idle periods at end of simulation
    final_ka = float(keepalive[T - 1]) * 60.0
    for c in containers:
        idle_start = c[0] if c[0] > c[1] else c[1]
        if idle_start < end_time:
            idle_mem_s += min(end_time - idle_start, final_ka) * mem_gb

    result = {
        "total_invocations": total_inv,
        "cold_starts": cold,
        "idle_mem_gb_s": idle_mem_s,
        "busy_mem_gb_s": busy_mem_s,
        "latencies": latencies,
    }
    if track_rolling:
        result["roll_cold"] = roll_cold
        result["roll_total"] = roll_total
    return result


def simulate_trace(counts, prewarm, keepalive, dur_means, dur_stds,
                   memory_mb=256.0, seed=0, tick_sec=60.0,
                   cold_mu=0.0, cold_sigma=0.3, track_rolling=False):
    """Simulate all functions independently and aggregate.

    Args:
        counts, prewarm, keepalive: [N, T] matrices
        dur_means, dur_stds: [N] arrays

    Returns:
        dict with aggregate metrics + per-function arrays.
    """
    N, T = counts.shape
    per_func = []
    all_lat = []

    for fi in range(N):
        r = simulate_function(
            counts[fi], prewarm[fi], keepalive[fi],
            float(dur_means[fi]), float(dur_stds[fi]),
            memory_mb=memory_mb, seed=seed * 100003 + fi,
            tick_sec=tick_sec, cold_mu=cold_mu, cold_sigma=cold_sigma,
            track_rolling=track_rolling,
        )
        all_lat.extend(r["latencies"])
        per_func.append(r)

    total_inv = sum(r["total_invocations"] for r in per_func)
    total_cold = sum(r["cold_starts"] for r in per_func)
    idle_mem = sum(r["idle_mem_gb_s"] for r in per_func)
    busy_mem = sum(r["busy_mem_gb_s"] for r in per_func)

    lats = np.array(all_lat) if all_lat else np.array([0.0])

    out = {
        "csr": total_cold / max(1, total_inv),
        "cold_starts": int(total_cold),
        "total_invocations": int(total_inv),
        "wm_total_gb_s": idle_mem,
        "wm_per_1k_inv": idle_mem / max(1e-10, total_inv / 1000.0),
        "wm_fraction": idle_mem / max(1e-10, idle_mem + busy_mem),
        "latency_p50": float(np.percentile(lats, 50)),
        "latency_p95": float(np.percentile(lats, 95)),
        "latency_p99": float(np.percentile(lats, 99)),
        # Per-function arrays for paired statistics & stratified tables
        "func_csr": [r["cold_starts"] / max(1, r["total_invocations"])
                     for r in per_func],
        "func_cold": [r["cold_starts"] for r in per_func],
        "func_total": [r["total_invocations"] for r in per_func],
        "func_wm": [r["idle_mem_gb_s"] for r in per_func],
    }
    if track_rolling:
        out["roll_cold"] = np.stack([r["roll_cold"] for r in per_func])
        out["roll_total"] = np.stack([r["roll_total"] for r in per_func])
    return out


def rolling_csr(roll_cold, roll_total, window=15):
    """Rolling CSR over a sliding window of ticks.

    Args:
        roll_cold, roll_total: [T] per-tick cold and total counts (can be
            summed across functions first)
        window: window size in ticks

    Returns:
        [T] rolling CSR (NaN where no invocations in window)
    """
    T = len(roll_cold)
    kernel = np.ones(window)
    c = np.convolve(roll_cold, kernel, mode="same")
    n = np.convolve(roll_total, kernel, mode="same")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(n > 0, c / n, np.nan)
    return r
