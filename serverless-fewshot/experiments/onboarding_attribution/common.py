"""Frozen inputs and output handling for the onboarding attribution study."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "onboarding_attribution_v1"
W = 240
RHOS = [1.0, 10.0, 100.0]
ALPHAS = [.01, .03, .05, .1, .15, .2, .3, .5, .8, 1.]
LAMBDAS = [10. ** i for i in range(-4, 5)]
BUDGETS = [.5, .75, 1., 1.25, 1.5, 2.]
METRICS = ["invocations", "cold", "idle", "init", "execution", "prewarm", "reactive", "overflow"]
BOUNDS = np.array([0., 60., 900., 3600., 14400., np.inf])


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_cohort(name):
    with np.load(OUT / "cohorts" / (name + ".npz"), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def require_tests():
    evidence = read_json(OUT / "tests.json")
    if not evidence.get("passed") or evidence["simulator_sha256"] != digest(HERE / "simulator.py"):
        raise RuntimeError("Simulator tests absent or stale; run test_study.py first")
    if evidence.get("heap_simulator_sha256") != digest(HERE / "heap_simulator.py"):
        raise RuntimeError("Heap simulator tests absent or stale")
    contract_path=OUT/"simulation_contract.json"
    contract={"simulator_sha256":evidence["simulator_sha256"]}
    if contract_path.exists():
        if read_json(contract_path)!=contract:
            raise RuntimeError("Simulator changed: use a new study output version instead of reusing old metrics")
    else:
        write_json(contract_path,contract)
