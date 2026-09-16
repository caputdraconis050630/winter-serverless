"""Numba-compiled port of src/sim/des.py simulate_function/simulate_trace.

Semantics are identical to the pure-python simulator except for two documented
deviations (validated statistically by scripts/validate_des_fast.py):
  1. RNG: legacy np.random (numba-supported) with per-function stream
     seed = seed + 9973*fi (the python sim reused one Generator stream with
     the same seed for every function). Stochastically equivalent; enables
     deterministic results under prange parallelism.
  2. Latencies: aggregated into a shared log-spaced histogram instead of an
     exact list (450M-event traces cannot hold per-event lists); percentiles
     are read from the histogram (relative bin width ~2.9%).

Runs only under the DES env (numpy<=2.4 + numba):
  PYTHONPATH=/data/260715/site-packages-des
"""

import numpy as np
from numba import njit, prange

MAX_POOL = 200  # keep in sync with src/sim/des.py

# latency histogram: 400 log-spaced bins over [1e-3, 1e4) seconds
HIST_BINS = 400
HIST_LO = -3.0
HIST_HI = 4.0


@njit(cache=True)
def _sim_one(counts, prewarm, keepalive, dur_mean, dur_std,
             memory_mb, seed, tick_sec, cold_mu, cold_sigma,
             hist):
    np.random.seed(seed)
    T = counts.shape[0]
    mem_gb = memory_mb / 1024.0
    end_time = T * tick_sec

    ready = np.empty(MAX_POOL, dtype=np.float64)
    free = np.empty(MAX_POOL, dtype=np.float64)
    ncon = 0

    total_inv = 0
    cold = 0
    idle_mem_s = 0.0
    busy_mem_s = 0.0
    dur_sigma = dur_std if dur_std > 0.01 else 0.01

    for t in range(T):
        now0 = t * tick_sec
        ka_sec = keepalive[t] * 60.0

        # 1. evict idle-expired before tick start
        i = 0
        while i < ncon:
            idle_start = ready[i] if ready[i] > free[i] else free[i]
            if idle_start <= now0 - ka_sec:
                idle_mem_s += ka_sec * mem_gb
                ncon -= 1
                ready[i] = ready[ncon]
                free[i] = free[ncon]
            else:
                i += 1

        # 2. prewarm to target
        target = prewarm[t]
        deficit = target - ncon
        if deficit > 0 and ncon < MAX_POOL:
            n_new = deficit
            if n_new > MAX_POOL - ncon:
                n_new = MAX_POOL - ncon
            for _ in range(n_new):
                init = np.exp(cold_mu + cold_sigma * np.random.randn())
                ready[ncon] = now0 + init
                free[ncon] = now0 + init
                ncon += 1
                busy_mem_s += init * mem_gb

        # 3. arrivals
        n = counts[t]
        if n > 0:
            offsets = np.sort(np.random.uniform(0.0, tick_sec, n))
            for k in range(n):
                tau = now0 + offsets[k]
                dur = dur_mean + dur_sigma * np.random.randn()
                if dur < 0.01:
                    dur = 0.01

                # lazy evict up to tau
                i = 0
                while i < ncon:
                    idle_start = ready[i] if ready[i] > free[i] else free[i]
                    if idle_start <= tau - ka_sec:
                        idle_mem_s += ka_sec * mem_gb
                        ncon -= 1
                        ready[i] = ready[ncon]
                        free[i] = free[ncon]
                    else:
                        i += 1

                # MRU warm-idle search
                best = -1
                best_idle = -1.0
                for i in range(ncon):
                    if ready[i] <= tau and free[i] <= tau:
                        idle_start = ready[i] if ready[i] > free[i] else free[i]
                        if idle_start > best_idle:
                            best_idle = idle_start
                            best = i

                if best >= 0:
                    idle_mem_s += (tau - best_idle) * mem_gb
                    free[best] = tau + dur
                    busy_mem_s += dur * mem_gb
                    lat = dur
                else:
                    init = np.exp(cold_mu + cold_sigma * np.random.randn())
                    if ncon < MAX_POOL:
                        ready[ncon] = tau + init
                        free[ncon] = tau + init + dur
                        ncon += 1
                    busy_mem_s += (init + dur) * mem_gb
                    lat = init + dur
                    cold += 1

                # histogram the latency
                b = int((np.log10(lat) - HIST_LO) / (HIST_HI - HIST_LO)
                        * HIST_BINS)
                if b < 0:
                    b = 0
                elif b >= HIST_BINS:
                    b = HIST_BINS - 1
                hist[b] += 1

                total_inv += 1

    # close out idle at end
    final_ka = keepalive[T - 1] * 60.0
    for i in range(ncon):
        idle_start = ready[i] if ready[i] > free[i] else free[i]
        if idle_start < end_time:
            rem = end_time - idle_start
            if rem > final_ka:
                rem = final_ka
            idle_mem_s += rem * mem_gb

    return total_inv, cold, idle_mem_s, busy_mem_s


@njit(parallel=True, cache=True)
def _sim_trace(counts, prewarm, keepalive, dur_means, dur_stds,
               seed, tick_sec, cold_mu, cold_sigma):
    N = counts.shape[0]
    f_total = np.zeros(N, dtype=np.int64)
    f_cold = np.zeros(N, dtype=np.int64)
    f_idle = np.zeros(N, dtype=np.float64)
    f_busy = np.zeros(N, dtype=np.float64)
    hists = np.zeros((N, HIST_BINS), dtype=np.int64)
    for fi in prange(N):
        ti, c, im, bm = _sim_one(
            counts[fi], prewarm[fi], keepalive[fi],
            dur_means[fi], dur_stds[fi], 256.0,
            seed + 9973 * fi, tick_sec, cold_mu, cold_sigma, hists[fi])
        f_total[fi] = ti
        f_cold[fi] = c
        f_idle[fi] = im
        f_busy[fi] = bm
    return f_total, f_cold, f_idle, f_busy, hists


def _hist_percentile(hist, q):
    edges = np.logspace(HIST_LO, HIST_HI, HIST_BINS + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    cum = np.cumsum(hist)
    if cum[-1] == 0:
        return 0.0
    target = q / 100.0 * cum[-1]
    idx = int(np.searchsorted(cum, target))
    return float(centers[min(idx, HIST_BINS - 1)])


def simulate_trace_fast(counts, prewarm, keepalive, dur_means, dur_stds,
                        seed=0, tick_sec=60.0, cold_mu=0.0, cold_sigma=0.3):
    """Drop-in equivalent of src.sim.des.simulate_trace (same output keys)."""
    f_total, f_cold, f_idle, f_busy, hists = _sim_trace(
        np.ascontiguousarray(counts.astype(np.int64)),
        np.ascontiguousarray(prewarm.astype(np.int64)),
        np.ascontiguousarray(keepalive.astype(np.float64)),
        np.ascontiguousarray(np.asarray(dur_means, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(dur_stds, dtype=np.float64)),
        seed, tick_sec, cold_mu, cold_sigma)

    total_inv = int(f_total.sum())
    total_cold = int(f_cold.sum())
    idle_mem = float(f_idle.sum())
    busy_mem = float(f_busy.sum())
    hist = hists.sum(axis=0)

    return {
        "csr": total_cold / max(total_inv, 1),
        "cold_starts": total_cold,
        "total_invocations": total_inv,
        "wm_total_gb_s": idle_mem,
        "wm_per_1k_inv": idle_mem / max(total_inv / 1000.0, 1e-9),
        "wm_fraction": idle_mem / max(idle_mem + busy_mem, 1e-9),
        "latency_p50": _hist_percentile(hist, 50),
        "latency_p95": _hist_percentile(hist, 95),
        "latency_p99": _hist_percentile(hist, 99),
        "func_csr": (f_cold / np.maximum(f_total, 1)).tolist(),
        "func_cold": f_cold.tolist(),
        "func_total": f_total.tolist(),
        "func_wm": f_idle.tolist(),
    }


@njit(cache=True)
def _bin_for_tick(t, bin_edges):
    for bi in range(bin_edges.shape[0] - 1):
        if t >= bin_edges[bi] and t < bin_edges[bi + 1]:
            return bi
    return bin_edges.shape[0] - 2


@njit(cache=True)
def _sim_one_binned(counts, prewarm, keepalive, dur_mean, dur_std,
                    memory_mb, seed, tick_sec, cold_mu, cold_sigma,
                    bin_edges, hist):
    np.random.seed(seed)
    T = counts.shape[0]
    mem_gb = memory_mb / 1024.0
    end_time = T * tick_sec

    ready = np.empty(MAX_POOL, dtype=np.float64)
    free = np.empty(MAX_POOL, dtype=np.float64)
    ncon = 0

    total_inv = 0
    cold = 0
    cold_bins = np.zeros(bin_edges.shape[0] - 1, dtype=np.int64)
    idle_mem_s = 0.0
    busy_mem_s = 0.0
    dur_sigma = dur_std if dur_std > 0.01 else 0.01

    for t in range(T):
        now0 = t * tick_sec
        ka_sec = keepalive[t] * 60.0

        i = 0
        while i < ncon:
            idle_start = ready[i] if ready[i] > free[i] else free[i]
            if idle_start <= now0 - ka_sec:
                idle_mem_s += ka_sec * mem_gb
                ncon -= 1
                ready[i] = ready[ncon]
                free[i] = free[ncon]
            else:
                i += 1

        target = prewarm[t]
        deficit = target - ncon
        if deficit > 0 and ncon < MAX_POOL:
            n_new = deficit
            if n_new > MAX_POOL - ncon:
                n_new = MAX_POOL - ncon
            for _ in range(n_new):
                init = np.exp(cold_mu + cold_sigma * np.random.randn())
                ready[ncon] = now0 + init
                free[ncon] = now0 + init
                ncon += 1
                busy_mem_s += init * mem_gb

        n = counts[t]
        if n > 0:
            offsets = np.sort(np.random.uniform(0.0, tick_sec, n))
            for k in range(n):
                tau = now0 + offsets[k]
                dur = dur_mean + dur_sigma * np.random.randn()
                if dur < 0.01:
                    dur = 0.01

                i = 0
                while i < ncon:
                    idle_start = ready[i] if ready[i] > free[i] else free[i]
                    if idle_start <= tau - ka_sec:
                        idle_mem_s += ka_sec * mem_gb
                        ncon -= 1
                        ready[i] = ready[ncon]
                        free[i] = free[ncon]
                    else:
                        i += 1

                best = -1
                best_idle = -1.0
                for i in range(ncon):
                    if ready[i] <= tau and free[i] <= tau:
                        idle_start = ready[i] if ready[i] > free[i] else free[i]
                        if idle_start > best_idle:
                            best_idle = idle_start
                            best = i

                if best >= 0:
                    idle_mem_s += (tau - best_idle) * mem_gb
                    free[best] = tau + dur
                    busy_mem_s += dur * mem_gb
                    lat = dur
                else:
                    init = np.exp(cold_mu + cold_sigma * np.random.randn())
                    if ncon < MAX_POOL:
                        ready[ncon] = tau + init
                        free[ncon] = tau + init + dur
                        ncon += 1
                    busy_mem_s += (init + dur) * mem_gb
                    lat = init + dur
                    cold += 1
                    cold_bins[_bin_for_tick(t, bin_edges)] += 1

                b = int((np.log10(lat) - HIST_LO) / (HIST_HI - HIST_LO)
                        * HIST_BINS)
                if b < 0:
                    b = 0
                elif b >= HIST_BINS:
                    b = HIST_BINS - 1
                hist[b] += 1

                total_inv += 1

    final_ka = keepalive[T - 1] * 60.0
    for i in range(ncon):
        idle_start = ready[i] if ready[i] > free[i] else free[i]
        if idle_start < end_time:
            rem = end_time - idle_start
            if rem > final_ka:
                rem = final_ka
            idle_mem_s += rem * mem_gb

    return total_inv, cold, idle_mem_s, busy_mem_s, cold_bins


@njit(parallel=True, cache=True)
def _sim_trace_binned(counts, prewarm, keepalive, dur_means, dur_stds,
                      seed, tick_sec, cold_mu, cold_sigma, bin_edges):
    N = counts.shape[0]
    n_bins = bin_edges.shape[0] - 1
    f_total = np.zeros(N, dtype=np.int64)
    f_cold = np.zeros(N, dtype=np.int64)
    f_idle = np.zeros(N, dtype=np.float64)
    f_busy = np.zeros(N, dtype=np.float64)
    f_cold_bins = np.zeros((N, n_bins), dtype=np.int64)
    hists = np.zeros((N, HIST_BINS), dtype=np.int64)
    for fi in prange(N):
        ti, c, im, bm, cb = _sim_one_binned(
            counts[fi], prewarm[fi], keepalive[fi],
            dur_means[fi], dur_stds[fi], 256.0,
            seed + 9973 * fi, tick_sec, cold_mu, cold_sigma,
            bin_edges, hists[fi])
        f_total[fi] = ti
        f_cold[fi] = c
        f_idle[fi] = im
        f_busy[fi] = bm
        f_cold_bins[fi] = cb
    return f_total, f_cold, f_idle, f_busy, f_cold_bins, hists


def simulate_trace_fast_binned(counts, prewarm, keepalive, dur_means, dur_stds,
                               bin_edges, seed=0, tick_sec=60.0,
                               cold_mu=0.0, cold_sigma=0.3):
    """Fast DES with per-function cold-start counts in tick bins."""
    f_total, f_cold, f_idle, f_busy, f_cold_bins, hists = _sim_trace_binned(
        np.ascontiguousarray(counts.astype(np.int64)),
        np.ascontiguousarray(prewarm.astype(np.int64)),
        np.ascontiguousarray(keepalive.astype(np.float64)),
        np.ascontiguousarray(np.asarray(dur_means, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(dur_stds, dtype=np.float64)),
        seed, tick_sec, cold_mu, cold_sigma,
        np.ascontiguousarray(np.asarray(bin_edges, dtype=np.int64)))

    total_inv = int(f_total.sum())
    total_cold = int(f_cold.sum())
    idle_mem = float(f_idle.sum())
    busy_mem = float(f_busy.sum())
    hist = hists.sum(axis=0)

    return {
        "csr": total_cold / max(total_inv, 1),
        "cold_starts": total_cold,
        "total_invocations": total_inv,
        "wm_total_gb_s": idle_mem,
        "wm_per_1k_inv": idle_mem / max(total_inv / 1000.0, 1e-9),
        "wm_fraction": idle_mem / max(idle_mem + busy_mem, 1e-9),
        "latency_p50": _hist_percentile(hist, 50),
        "latency_p95": _hist_percentile(hist, 95),
        "latency_p99": _hist_percentile(hist, 99),
        "func_csr": (f_cold / np.maximum(f_total, 1)).tolist(),
        "func_cold": f_cold.tolist(),
        "func_total": f_total.tolist(),
        "func_wm": f_idle.tolist(),
        "func_cold_bins": f_cold_bins.tolist(),
        "bin_edges": list(np.asarray(bin_edges, dtype=np.int64)),
    }
