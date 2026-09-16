"""Numba DES with per-sandbox concurrency C (containerConcurrency analogue).

Port of src/sim/des_fast.py with one semantic extension: a sandbox may hold
up to CC in-flight requests. At CC=1 the kernel reduces EXACTLY to
des_fast._sim_one — same RNG draw order, same eligibility, same accounting —
which is asserted against des_fast on real pools by
scripts/revision_r7a_concurrency.py --validate.

Semantics for CC>1 (pre-registered in revision_r7a_PREREG.md):
- Eligible sandbox: cold-init complete (ready <= tau) and in-flight < CC.
- Routing: pack — highest in-flight first (Knative routes to pods with
  spare capacity; packing maximizes idle-eviction), tie-break by most
  recent activity. At CC=1 every eligible sandbox has in-flight 0, so the
  tie-break IS the baseline MRU rule.
- No intra-sandbox queueing: concurrent requests serve in parallel
  (containerConcurrency semantics); warm latency = dur.
- Memory: a sandbox occupies memory once regardless of in-flight count.
  Busy time is the UNION of its service intervals (streamed via last_end;
  arrivals are time-ordered), idle time the gaps — at CC=1 this is
  byte-identical to the baseline per-request accounting.

DES env only: PYTHONPATH=/data/260715/site-packages-des
"""

import numpy as np
from numba import njit, prange

MAX_POOL = 200   # keep in sync with src/sim/des.py
CC_MAX = 64

HIST_BINS = 400
HIST_LO = -3.0
HIST_HI = 4.0


@njit(cache=True)
def _sim_one_cc(counts, prewarm, keepalive, dur_mean, dur_std,
                memory_mb, seed, tick_sec, cold_mu, cold_sigma,
                hist, cc):
    np.random.seed(seed)
    T = counts.shape[0]
    mem_gb = memory_mb / 1024.0
    end_time = T * tick_sec

    ready = np.empty(MAX_POOL, dtype=np.float64)
    last_end = np.empty(MAX_POOL, dtype=np.float64)   # == free at cc=1
    ends = np.zeros((MAX_POOL, CC_MAX), dtype=np.float64)
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
            idle_start = ready[i] if ready[i] > last_end[i] else last_end[i]
            if idle_start <= now0 - ka_sec:
                idle_mem_s += ka_sec * mem_gb
                ncon -= 1
                ready[i] = ready[ncon]
                last_end[i] = last_end[ncon]
                ends[i, :] = ends[ncon, :]
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
                last_end[ncon] = now0 + init
                ends[ncon, :] = 0.0
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
                    idle_start = (ready[i] if ready[i] > last_end[i]
                                  else last_end[i])
                    if idle_start <= tau - ka_sec:
                        idle_mem_s += ka_sec * mem_gb
                        ncon -= 1
                        ready[i] = ready[ncon]
                        last_end[i] = last_end[ncon]
                        ends[i, :] = ends[ncon, :]
                    else:
                        i += 1

                # eligible: ready and in-flight < cc; pack-first, then MRU
                best = -1
                best_inflight = -1
                best_act = -1.0
                for i in range(ncon):
                    if ready[i] > tau:
                        continue
                    infl = 0
                    for j in range(cc):
                        if ends[i, j] > tau:
                            infl += 1
                    if infl >= cc:
                        continue
                    act = ready[i] if ready[i] > last_end[i] else last_end[i]
                    if infl > best_inflight or (infl == best_inflight
                                                and act > best_act):
                        best_inflight = infl
                        best_act = act
                        best = i

                if best >= 0:
                    if tau >= last_end[best]:
                        # sandbox was idle: close the idle gap (baseline path)
                        idle_mem_s += (tau - best_act) * mem_gb
                        busy_mem_s += dur * mem_gb
                        last_end[best] = tau + dur
                    else:
                        # busy sandbox with spare capacity (cc>1 only):
                        # extend the busy-interval union
                        if tau + dur > last_end[best]:
                            busy_mem_s += (tau + dur - last_end[best]) * mem_gb
                            last_end[best] = tau + dur
                    # occupy a completed slot
                    for j in range(cc):
                        if ends[best, j] <= tau:
                            ends[best, j] = tau + dur
                            break
                    lat = dur
                else:
                    init = np.exp(cold_mu + cold_sigma * np.random.randn())
                    if ncon < MAX_POOL:
                        ready[ncon] = tau + init
                        last_end[ncon] = tau + init + dur
                        ends[ncon, :] = 0.0
                        ends[ncon, 0] = tau + init + dur
                        ncon += 1
                    busy_mem_s += (init + dur) * mem_gb
                    lat = init + dur
                    cold += 1

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
        idle_start = ready[i] if ready[i] > last_end[i] else last_end[i]
        if idle_start < end_time:
            rem = end_time - idle_start
            if rem > final_ka:
                rem = final_ka
            idle_mem_s += rem * mem_gb

    return total_inv, cold, idle_mem_s, busy_mem_s


@njit(parallel=True, cache=True)
def _sim_trace_cc(counts, prewarm, keepalive, dur_means, dur_stds,
                  seed, tick_sec, cold_mu, cold_sigma, cc):
    N = counts.shape[0]
    f_total = np.zeros(N, dtype=np.int64)
    f_cold = np.zeros(N, dtype=np.int64)
    f_idle = np.zeros(N, dtype=np.float64)
    f_busy = np.zeros(N, dtype=np.float64)
    hists = np.zeros((N, HIST_BINS), dtype=np.int64)
    for fi in prange(N):
        ti, c, im, bm = _sim_one_cc(
            counts[fi], prewarm[fi], keepalive[fi],
            dur_means[fi], dur_stds[fi], 256.0,
            seed + 9973 * fi, tick_sec, cold_mu, cold_sigma, hists[fi], cc)
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


def simulate_trace_cc(counts, prewarm, keepalive, dur_means, dur_stds,
                      seed=0, tick_sec=60.0, cold_mu=0.0, cold_sigma=0.3,
                      cc=1):
    """Drop-in analogue of des_fast.simulate_trace_fast with concurrency."""
    if not (1 <= cc <= CC_MAX):
        raise ValueError("cc out of range")
    f_total, f_cold, f_idle, f_busy, hists = _sim_trace_cc(
        np.ascontiguousarray(counts.astype(np.int64)),
        np.ascontiguousarray(prewarm.astype(np.int64)),
        np.ascontiguousarray(keepalive.astype(np.float64)),
        np.ascontiguousarray(np.asarray(dur_means, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(dur_stds, dtype=np.float64)),
        seed, tick_sec, cold_mu, cold_sigma, cc)

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
        "func_cold": f_cold.tolist(),
        "func_total": f_total.tolist(),
        "func_wm": f_idle.tolist(),
    }
