"""Export fixed cohorts and policy actions without altering archived campaigns."""
import argparse
import hashlib
import sys

import numpy as np
import pandas as pd

from common import OUT, ROOT, RHOS, digest, freeze, read_json, save_npz, write_json
from model import load_model, native_actions, predict

sys.path.insert(0, str(ROOT))
from scripts.phase6_des import decisions_from_rates as archived_decisions, rates_ewma, rates_fourier
from scripts.revision_v1_hybridfull import policy_b2_full
from src.decision.newsvendor import newsvendor_quantile

RUNS = ROOT / "results/runs"
V1 = ROOT / "results/onboarding_attribution_v1"


def decisions_from_rates(rates, tau):
    # At this rate the target exceeds cap 200 at every evaluated quantile.
    # Saturate before the archived int32 conversion to avoid integer overflow.
    rates = np.asarray(rates)
    if not np.isfinite(rates).all():
        raise ValueError("Non-finite forecast")
    return archived_decisions(np.clip(rates, 0., 1e6), tau)


def begin(name, data, metadata):
    directory = OUT / "cases" / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "data.npz"
    if not path.exists():
        save_npz(path, **data)
    else:
        with np.load(path) as old:
            for key, value in data.items():
                np.testing.assert_array_equal(old[key], value)
    metadata.update(name=name, n_functions=len(data["counts"]),
                    horizon_minutes=data["counts"].shape[1],
                    raw_invocations=int(data["counts"].sum()),
                    data_sha256=digest(path), actions={})
    return directory, metadata


def action(directory, meta, method, rho, q, ttl, source):
    q = np.clip(q, 0, 200).astype(np.int16)
    ttl = np.asarray(ttl, np.float32)
    if q.shape != ttl.shape or not np.isfinite(ttl).all() or np.any(ttl < 0):
        raise ValueError("Invalid action arrays")
    name = f"{method}__rho{rho:g}"
    path = directory / (name + ".npz")
    if not path.exists():
        save_npz(path, q=q, ttl=ttl)
    else:
        with np.load(path) as z:
            np.testing.assert_array_equal(q, z["q"])
            np.testing.assert_array_equal(ttl, z["ttl"])
    meta["actions"][name] = dict(method=method, rho=float(rho), file=path.name,
                                 sha256=digest(path), source=source)


def lstm_actions(directory, meta, data, split="s2", rhos=RHOS):
    path = directory / "lstm_forecasts.npz"
    if path.exists():
        rate = np.load(path)["rate"]
    else:
        rate = predict(load_model(split), data["counts"], data.get("prefix"))
        save_npz(path, rate=rate)
    source = dict(checkpoint=str(OUT / "models" / f"{split}_seed0/model.pt"),
                  forecast_sha256=digest(path), scope="Fifer-inspired predictor; trace-resolution adaptation")
    q, ttl = native_actions(rate)
    for rho in rhos:
        action(directory, meta, "LSTM_Fifer", rho, q, ttl, source)
        shared_q, shared_ttl = decisions_from_rates(rate, newsvendor_quantile(rho))
        action(directory, meta, "LSTM_shared", rho, shared_q, shared_ttl, source)


def rate_actions(directory, meta, method, rates, rhos, source):
    for rho in rhos:
        q, ttl = decisions_from_rates(rates, newsvendor_quantile(rho))
        action(directory, meta, method, rho, q, ttl, source)


def basic_actions(directory, meta, data, rhos=RHOS, include_hybrid=True, include_fourier=True):
    counts = data["counts"]
    for alpha in (.1, .3):
        rate_actions(directory, meta, f"EWMA_{alpha:g}", rates_ewma(counts, alpha), rhos,
                     "existing causal log-count EWMA")
    q = np.zeros_like(counts)
    for t in range(1, counts.shape[1]):
        q[:, t] = (counts[:, max(0, t-10):t] > 0).any(axis=1)
    for rho in rhos:
        action(directory, meta, "Keep_alive", rho, q, np.full(q.shape, 10.), "existing keep-alive comparator")
    if include_fourier:
        rate_actions(directory, meta, "Fourier", rates_fourier(counts), rhos,
                     "registered IceBreaker-style Fourier predictor")
    if include_hybrid:
        q, ttl = policy_b2_full(counts, verbose_every=100000)
        for rho in rhos:
            action(directory, meta, "Hybrid", rho, q, ttl, "registered Shahrad hybrid policy components")


def finish(directory, meta):
    freeze(directory / "case.json", meta)
    print("prepared", meta["name"], meta["n_functions"], "functions",
          meta["horizon_minutes"], "minutes", len(meta["actions"]), "conditions", flush=True)


def onboarding(name):
    path = V1 / "cohorts" / f"{name}.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in ("counts", "ids", "keys", "apps", "duration_mean", "duration_std")}
    data["weights"] = np.ones(len(data["counts"]))
    directory, meta = begin("initial_" + name, data, dict(
        surface="initial", provider="huawei" if name == "huawei" else "azure2019",
        source=str(path), event_key_prefix=name.split("_")[0] + ":",
        seeds=list(range(1000, 1020)), rhos=list(RHOS), start="first-arrival-minute boundary",
        model_source="Azure2021 S2", selection="fixed policies, no target tuning"))
    lstm_actions(directory, meta, data)
    basic_actions(directory, meta, data)
    cached = V1 / "models/trained0" / f"{name}_rates.npz"
    with np.load(cached) as z:
        for method, key in (("WINTER", "component"), ("WINTER_G", "gate"), ("WINTER_no_prior", "no_prior")):
            rate_actions(directory, meta, method, z[key], RHOS, dict(file=str(cached), sha256=digest(cached)))
    if name == "azure_primary":
        cache = RUNS / "revision_e5d_chronos_native_forecasts.npz"
        with np.load(cache) as z:
            levels, grid = z["q_levels"], z["cohort_q"]
        for rho in RHOS:
            tau = min(newsvendor_quantile(rho), float(levels[-1]))
            rate = np.empty(grid.shape[1:], np.float32)
            for f in range(rate.shape[0]):
                for t in range(rate.shape[1]):
                    rate[f,t] = np.interp(tau, levels, grid[:,f,t])
            _, ttl = decisions_from_rates(rate, newsvendor_quantile(rho))
            action(directory, meta, "Chronos", rho, np.ceil(rate - 1e-9), ttl,
                   dict(file=str(cache), sha256=digest(cache), quantile_read=float(tau)))
    finish(directory, meta)


def cached_action(directory, meta, method, rho, paths, archived_name=None):
    for base in paths:
        if not (base / "jobs.json").exists():
            continue
        for record in read_json(base / "jobs.json")["jobs"]:
            if record["method"] == (archived_name or method) and float(record["rho"]) == float(rho):
                path = base / record["file"]
                with np.load(path) as z:
                    q, ttl = z["prewarm"], z["keepalive"]
                action(directory, meta, method, rho, q, ttl, dict(file=str(path), sha256=digest(path)))
                return
    raise FileNotFoundError((method, rho, [str(p) for p in paths]))


def holdout():
    root = RUNS / "des_jobs_r20_2019_holdout/azure2019_holdout"
    with np.load(root / "shared.npz") as z:
        data = dict(counts=z["counts"], duration_mean=z["dur_means"], duration_std=z["dur_stds"])
    proc = ROOT / "data/processed_2019"
    full = np.load(proc / "counts.npy", mmap_mode="r")
    first, day1 = np.full(len(full), -1), np.zeros(len(full), np.int64)
    for f, row in enumerate(full):
        nz = np.flatnonzero(row)
        if len(nz):
            first[f] = nz[0]
            day1[f] = row[nz[0]:nz[0]+1440].sum()
    exclusion = set()
    for gap in (1,3,7):
        for threshold in (10,30,100):
            ids = np.flatnonzero((first >= gap*1440) & (first < full.shape[1]-1440) & (day1 >= threshold))
            if len(ids) > 100:
                ids = ids[np.random.default_rng(42).choice(len(ids), 100, replace=False)]
            exclusion.update(map(int, ids))
    ids = np.array([f for f in np.flatnonzero((first >= 4320) & (first < full.shape[1]-2880) & (day1 >= 30)) if f not in exclusion])
    assert len(exclusion) == 772 and len(ids) == 1755
    np.testing.assert_array_equal(data["counts"], np.stack([full[f,first[f]:first[f]+2880] for f in ids]))
    data.update(ids=ids, keys=np.load(proc / "func_ids.npy", allow_pickle=True)[ids].astype(str),
                apps=pd.read_csv(proc / "active_functions.csv").app_id.to_numpy()[ids].astype(str), weights=np.ones(len(ids)))
    directory, meta = begin("continuous_48h", data, dict(surface="continuous", provider="azure2019",
        source=str(root), seeds=[0,1,2], rhos=list(RHOS), event_key_prefix="continuous48:",
        model_source="Azure2021 S2", start="first-arrival-minute boundary"))
    lstm_actions(directory, meta, data)
    basic_actions(directory, meta, data)
    for rho in RHOS:
        for method, archived in (("WINTER", "A5_proto"), ("WINTER_G", "G_WE_A720"), ("No_handoff", "G_WE_Ainf")):
            cached_action(directory, meta, method, rho, [root], archived)
    finish(directory, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("surface", choices=["initial", "continuous"])
    ap.add_argument("--cohort", default="azure_primary", choices=["azure_primary", "huawei", "azure_evaluation"])
    args = ap.parse_args()
    if args.surface == "initial":
        onboarding(args.cohort)
    else:
        holdout()
