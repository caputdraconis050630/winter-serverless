"""Bounded-memory, semantically indexed request streams shared by all policies."""
import hashlib

import numpy as np


def event_chunks(counts, duration_mean, duration_std, seed, function_key,
                 max_requests=1000000, max_minutes=1440, cold_mu=.25, cold_sigma=.31):
    counts = np.asarray(counts, np.int64)
    if np.any(counts < 0):
        raise ValueError("Negative invocation count")
    key = np.frombuffer(hashlib.sha256(function_key.encode()).digest()[:16], dtype="<u4")
    children = np.random.SeedSequence([260907, int(seed), *map(int, key)]).spawn(4)
    arrival_rng, duration_rng, cold_rng, warm_rng = [np.random.default_rng(s) for s in children]
    offset = 0
    while offset < len(counts):
        end = min(offset + max_minutes, len(counts))
        partial = np.cumsum(counts[offset:end])
        end = min(end, offset + max(1, int(np.searchsorted(partial, max_requests, side="right"))))
        segment = counts[offset:end]
        ptr = np.concatenate(([0], np.cumsum(segment, dtype=np.int64)))
        arrival = arrival_rng.uniform(0., 60., int(ptr[-1]))
        for t in range(len(segment)):
            arrival[ptr[t]:ptr[t + 1]].sort()
            arrival[ptr[t]:ptr[t + 1]] += (offset + t) * 60.
        dm = duration_mean if np.ndim(duration_mean) == 0 else np.repeat(duration_mean[offset:end], segment)
        ds = duration_std if np.ndim(duration_std) == 0 else np.repeat(duration_std[offset:end], segment)
        duration = np.maximum(.01, duration_rng.normal(dm, np.maximum(.01, ds), len(arrival)))
        cold = cold_rng.lognormal(cold_mu, cold_sigma, len(arrival))
        warm = warm_rng.lognormal(cold_mu, cold_sigma, (len(segment), 200))
        yield offset, end, (ptr, arrival, duration, cold, warm)
        offset = end
