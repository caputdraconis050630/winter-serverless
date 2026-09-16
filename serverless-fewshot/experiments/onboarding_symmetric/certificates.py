"""Exact no-expiration certificate generalized to the action's minimum TTL."""
import numpy as np
from simulator import _simulate


def make_certificate(tape, initial, minimum_ttl):
    w = len(tape[0])-1
    if w != 240 or minimum_ttl < 0:
        raise ValueError("Invalid certificate horizon or TTL")
    q = np.zeros(w, dtype=np.int64)
    q[0] = initial
    return _simulate(*tape, q, np.full(w, float(minimum_ttl)), .25, 200, False, True)


def reuse(cert, q, ttl, minimum_ttl):
    metrics, occupancy, clean, retained = cert
    if not clean or q[0] != metrics[0, 5] or np.min(ttl) < minimum_ttl or np.any(q[1:] > occupancy[1:]):
        return None
    # With no expiry and no extra provisioning, request paths are identical.
    # Every final resident expires after the horizon; only terminal idle differs.
    out = metrics.copy()
    out[4, 2] += retained*(float(ttl[-1])-minimum_ttl)*60.*.25
    return out
