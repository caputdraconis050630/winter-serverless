"""Common-event DES with explicit prospective TTL and interval accounting.

Pool cap bounds retained containers, not request admission. Overflow requests
use ephemeral containers, charged through completion and never retained.
The fifth output interval is the drain after provisioning ends at minute 240.
"""
import hashlib

import numpy as np
from numba import njit

from common import BOUNDS


def event_tape(counts, duration_mean, duration_std, seed, function_key,
               cold_mu=0., cold_sigma=.3):
    key = np.frombuffer(hashlib.sha256(function_key.encode()).digest()[:16], dtype="<u4")
    children = np.random.SeedSequence([260907, int(seed), *map(int, key)]).spawn(4)
    arrival, duration, reactive, proactive = [np.random.default_rng(s) for s in children]
    ptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    offsets = arrival.uniform(0., 60., int(ptr[-1]))
    for t in range(len(counts)):
        offsets[ptr[t]:ptr[t + 1]].sort()
        offsets[ptr[t]:ptr[t + 1]] += t * 60.
    durations = np.maximum(.01, duration.normal(duration_mean, max(.01, duration_std), len(offsets)))
    cold = reactive.lognormal(cold_mu, cold_sigma, len(offsets))
    warm = proactive.lognormal(cold_mu, cold_sigma, (len(counts), 200))
    return ptr, offsets, durations, cold, warm


@njit(cache=True)
def _charge(out, a, b, col, memory):
    if b <= a:
        return
    for j in range(5):
        span = min(b, BOUNDS[j + 1]) - max(a, BOUNDS[j])
        if span > 0:
            out[j, col] += span * memory


@njit(cache=True)
def _close(out, birth, ready, free, stop, memory):
    _charge(out, birth, min(ready, stop), 3, memory)
    _charge(out, ready, min(free, stop), 4, memory)
    _charge(out, free, stop, 2, memory)


@njit(cache=True, nogil=True)
def _simulate(ptr, arrival, duration, cold_init, warm_init, targets, ttl,
              memory=.25, cap=200, legacy_ttl=False, certify=False):
    """Return [time-bin, metric]; legacy_ttl isolates old TTL/WM semantics."""
    out = np.zeros((5, 8), dtype=np.float64)
    active = np.zeros(cap, dtype=np.bool_)
    birth = np.zeros(cap)
    ready = np.zeros(cap)
    free = np.zeros(cap)
    expires = np.zeros(cap)
    occupied = 0
    extent = 0
    occupancy = np.zeros(len(targets), dtype=np.int64)
    no_expiry = True
    for t in range(len(targets)):
        now = t * 60.
        ka = ttl[t] * 60.
        b = 0 if t < 1 else 1 if t < 15 else 2 if t < 60 else 3
        for c in range(extent):
            if not active[c]:
                continue
            if legacy_ttl:
                expires[c] = free[c] + ka
            if expires[c] <= now:
                _close(out, birth[c], ready[c], free[c], expires[c], memory)
                active[c] = False
                occupied -= 1
                no_expiry = False
                if certify:
                    return out, occupancy, False, occupied
            elif not legacy_ttl:
                expires[c] = free[c] + ka
                if expires[c] <= now:
                    _close(out, birth[c], ready[c], free[c], now, memory)
                    active[c] = False
                    occupied -= 1
                    no_expiry = False
                    if certify:
                        return out, occupancy, False, occupied
        if certify:
            occupancy[t] = occupied
        deficit = min(cap, targets[t]) - occupied
        rank = 0
        for c in range(cap):
            if rank >= deficit:
                break
            if not active[c]:
                active[c] = True
                birth[c] = now
                ready[c] = now + warm_init[t, rank]
                free[c] = ready[c]
                expires[c] = free[c] + ka
                occupied += 1
                extent = max(extent, c + 1)
                rank += 1
                out[b, 5] += 1
        for i in range(ptr[t], ptr[t + 1]):
            at = arrival[i]
            best = -1
            best_idle = -np.inf
            vacant = -1
            for c in range(extent):
                if active[c] and expires[c] <= at:
                    _close(out, birth[c], ready[c], free[c], expires[c], memory)
                    active[c] = False
                    occupied -= 1
                    no_expiry = False
                    if certify:
                        return out, occupancy, False, occupied
                if not active[c]:
                    if vacant < 0:
                        vacant = c
                elif free[c] <= at and free[c] > best_idle:
                    best = c
                    best_idle = free[c]
            if vacant < 0 and extent < cap:
                vacant = extent
            out[b, 0] += 1
            if best >= 0:
                _close(out, birth[best], ready[best], free[best], at, memory)
                birth[best] = at
                ready[best] = at
                free[best] = at + duration[i]
                expires[best] = free[best] + ka
            else:
                out[b, 1] += 1
                out[b, 6] += 1
                r = at + cold_init[i]
                f = r + duration[i]
                if vacant >= 0:
                    active[vacant] = True
                    occupied += 1
                    extent = max(extent, vacant + 1)
                    birth[vacant], ready[vacant], free[vacant] = at, r, f
                    expires[vacant] = f + ka
                else:
                    out[b, 7] += 1
                    _close(out, at, r, f, f, memory)
    for c in range(extent):
        if active[c]:
            if expires[c] <= len(targets) * 60.:
                no_expiry = False
            _close(out, birth[c], ready[c], free[c], expires[c], memory)
    return out, occupancy, no_expiry, occupied


@njit(cache=True, nogil=True)
def simulate(ptr, arrival, duration, cold_init, warm_init, targets, ttl,
             memory=.25, cap=200, legacy_ttl=False):
    return _simulate(ptr, arrival, duration, cold_init, warm_init, targets, ttl,
                     memory, cap, legacy_ttl, False)[0]


def certificate(tape, initial_target, cap=200):
    """Exact reuse certificate, not an approximation or a traffic classifier.

    If a q(t>0)=0, TTL=5 replay never expires a container, any TTL>=5 and
    targets below its tick-boundary occupancy execute identical request paths.
    Only final drain idle memory changes, analytically by n * delta TTL.
    """
    w=len(tape[0])-1
    if w != 240:
        raise ValueError("Reuse certificate requires the registered 240-minute horizon")
    q=np.zeros(w,dtype=np.int64); q[0]=initial_target
    return _simulate(*tape,q,np.full(w,5.),.25,cap,False,True)


def reuse_certificate(cert, q, ttl):
    metrics,occupancy,clean,retained=cert
    if not clean or q[0]!=metrics[0,5] or np.min(ttl)<5. or np.any(q[1:]>occupancy[1:]):
        return None
    out=metrics.copy()
    out[4,2]+=retained*(float(ttl[-1])-5.)*60.*.25
    return out


def reference(tape, targets, ttl, memory=.25, cap=200):
    """Independent event-queue oracle, including a phase-interval ledger."""
    import heapq
    ptr, arrivals, durations, inits, prewarm = tape
    queue = []
    for t in range(len(targets)):
        heapq.heappush(queue, (t * 60., 1, t, -1))
    for i, at in enumerate(arrivals):
        heapq.heappush(queue, (float(at), 2, i, -1))
    containers = {}
    ledger = []
    out = np.zeros((5, 8))
    current_ttl = float(ttl[0]) * 60.

    def close(c, end, reason):
        for phase, lo, hi in (("init", c[0], min(c[1], end)),
                              ("execution", c[1], min(c[2], end)),
                              ("idle", c[2], end)):
            if hi > lo:
                ledger.append((phase, float(lo), float(hi), reason))

    def schedule(slot, c):
        heapq.heappush(queue, (c[3], 0, slot, c[4]))

    serial = 0
    while queue:
        at, kind, idx, version = heapq.heappop(queue)
        if kind == 0:
            if idx in containers and containers[idx][4] == version:
                close(containers.pop(idx), at, "expiry")
            continue
        b = min(3, int(np.searchsorted(BOUNDS, at, side="right") - 1))
        if kind == 1:
            current_ttl = float(ttl[idx]) * 60.
            for slot, c in list(containers.items()):
                c[3] = max(at, c[2] + current_ttl)
                serial += 1
                c[4] = serial
                if c[3] <= at:
                    close(containers.pop(slot), at, "ttl_decrease")
                else:
                    schedule(slot, c)
            deficit = min(cap, int(targets[idx])) - len(containers)
            for rank in range(max(0, deficit)):
                slot = next(s for s in range(cap) if s not in containers)
                r = at + float(prewarm[idx, rank])
                serial += 1
                c = [at, r, r, r + current_ttl, serial]
                containers[slot] = c
                schedule(slot, c)
                out[b, 5] += 1
        else:
            out[b, 0] += 1
            available = [(c[2], -s, s) for s, c in containers.items() if c[2] <= at]
            if available:
                slot = max(available)[2]
                close(containers[slot], at, "warm_hit")
                r = at
            else:
                out[b, 1] += 1
                out[b, 6] += 1
                r = at + float(inits[idx])
                if len(containers) >= cap:
                    out[b, 7] += 1
                    close([at, r, r + durations[idx]], r + durations[idx], "overflow")
                    continue
                slot = next(s for s in range(cap) if s not in containers)
            f = r + float(durations[idx])
            serial += 1
            c = [at, r, f, f + current_ttl, serial]
            containers[slot] = c
            schedule(slot, c)
    for phase, lo, hi, reason in ledger:
        col = {"idle": 2, "init": 3, "execution": 4}[phase]
        for j in range(5):
            out[j, col] += max(0., min(hi, BOUNDS[j + 1]) - max(lo, BOUNDS[j])) * memory
    return out, ledger
