"""Paths, immutable run configuration and atomic experiment outputs."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "lstm_comparison_v1"
RHOS = (1., 10., 100.)
METRICS = ("invocations", "cold", "idle", "init", "execution", "prewarm", "reactive", "overflow")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def freeze(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise RuntimeError(f"Frozen configuration differs: {path}")
    write_json(path, value)


def rolling_csr(cold, total, window=15):
    cold, total = np.asarray(cold, float), np.asarray(total, float)
    kernel = np.ones(window)
    denominator = np.convolve(total, kernel, mode="same")
    return np.divide(np.convolve(cold, kernel, mode="same"), denominator,
                     out=np.full_like(denominator, np.nan), where=denominator > 0)


def adaptation_lag(cold, total, epsilon=.1):
    """Existing retrospective AL: 31 inclusive ticks below own final-quarter band."""
    curve = rolling_csr(cold, total)
    tail = curve[3 * len(curve) // 4:]
    finite = tail[np.isfinite(tail)]
    if not len(finite):
        return None
    threshold = (1 + epsilon) * finite.mean()
    for t in range(len(curve) - 30):
        run = curve[t:t + 31]
        if np.isfinite(run).all() and np.all(run <= threshold + 1e-10):
            return t
    return None
