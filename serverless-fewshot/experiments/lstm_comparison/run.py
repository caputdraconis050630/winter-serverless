"""Resumable common-request replay and paired, clustered result aggregation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time

import numpy as np

from common import OUT, HERE, METRICS, adaptation_lag, digest, freeze, read_json, save_npz, write_json
from events import event_chunks
from simulator import advance, new_state


def windows_for(meta):
    horizon = meta["horizon_minutes"]
    windows = {"full": [0, horizon], "drain": [horizon, horizon + 1]}
    onset = meta.get("onset_minute", 0)
    for stop in (1, 15, 30, 60, 240):
        if onset + stop <= horizon:
            windows[f"post_{stop}"] = [onset, onset + stop]
    if horizon >= 2880 and onset == 0:
        for label, start, stop in (("young", 0, 720), ("mature", 720, horizon),
                                    ("pre_handoff", 600, 720), ("post_handoff", 720, 840)):
            windows[label] = [start, stop]
    return windows


def run(name, workers=4, limit=0):
    directory = OUT / "cases" / name
    meta = read_json(directory / "case.json")
    with np.load(directory / "data.npz") as z:
        data = {k: z[k] for k in z.files}
    if digest(directory / "data.npz") != meta["data_sha256"]:
        raise RuntimeError("Cohort data changed")
    names = sorted(meta["actions"])
    actions = []
    for key in names:
        record = meta["actions"][key]
        path = directory / record["file"]
        if digest(path) != record["sha256"]:
            raise RuntimeError(f"Actions changed: {path}")
        with np.load(path) as z:
            actions.append((z["q"], z["ttl"]))
    windows = windows_for(meta)
    contract = dict(case_sha256=digest(directory / "case.json"), actions=names,
        seeds=meta["seeds"], windows=windows, metrics=list(METRICS), memory_gb=.25, cap=200,
        simulator_sha256=digest(HERE / "simulator.py"), events_sha256=digest(HERE / "events.py"),
        runner_sha256=digest(HERE / "run.py"), phase_accounting="wall-clock minutes plus separate terminal drain",
        ttl="prospective", request_streams="policy independent; stable function and semantic RNG streams",
        curve_storage="seed-summed cold uint64 and seed-mean idle float32; endpoint totals float64")
    freeze(directory / "execution.json", contract)
    contract_hash = digest(directory / "execution.json")
    target = directory / "functions"
    target.mkdir(exist_ok=True)
    horizon = meta["horizon_minutes"]

    def function(f):
        path = target / f"{f:05d}.npz"
        if path.exists():
            with np.load(path) as old:
                if str(old["contract_sha256"]) != contract_hash:
                    raise RuntimeError(f"Stale result: {path}")
            return f, False
        # Identical actions (including native policies across rho) are replayed once.
        unique, mapping, seen = [], [], {}
        for q_all, ttl_all in actions:
            q, ttl = np.ascontiguousarray(q_all[f]), np.ascontiguousarray(ttl_all[f])
            key = hashlib.sha256(q.tobytes() + ttl.tobytes()).digest()
            if key not in seen:
                seen[key] = len(unique)
                unique.append((q, ttl))
            mapping.append(seen[key])
        u = len(unique)
        cold = np.zeros((u, horizon + 1), np.uint64)
        idle = np.zeros((u, horizon + 1), np.float64)
        stats = np.zeros((len(meta["seeds"]), u, len(windows), len(METRICS)))
        counts = data["counts"][f]
        for s, seed in enumerate(meta["seeds"]):
            states = [new_state(horizon) for _ in unique]
            expected_execution = 0.
            for start, end, tape in event_chunks(counts, data["duration_mean"][f], data["duration_std"][f],
                    seed, meta["event_key_prefix"] + str(data["keys"][f])):
                expected_execution += .25 * tape[2].sum()
                for j, (q, ttl) in enumerate(unique):
                    advance(*tape, q[start:end], ttl[start:end], *states[j], start, end == horizon)
            for j, state in enumerate(states):
                out = state[0]
                np.testing.assert_array_equal(out[:-1, 0], counts)
                if not np.isfinite(out).all() or np.any(out < -1e-6) or np.any(out[:, 1] > out[:, 0]):
                    raise RuntimeError("Invalid event or phase ledger")
                np.testing.assert_allclose(out[:,4].sum(), expected_execution, rtol=2e-9, atol=1e-5)
                cold[j] += np.rint(out[:,1]).astype(np.uint64)
                idle[j] += out[:,2] / len(meta["seeds"])
                for w, (a,b) in enumerate(windows.values()):
                    stats[s,j,w] = out[a:b].sum(axis=0)
        save_npz(path, mapping=np.array(mapping), cold=cold, idle=idle.astype(np.float32), stats=stats,
                 contract_sha256=np.array(contract_hash), function_index=np.array(f))
        return f, True

    start = time.monotonic()
    indices = range(min(limit, len(data["counts"])) if limit else len(data["counts"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(function, f) for f in indices]
        for done, future in enumerate(as_completed(futures), 1):
            f, fresh = future.result()
            if done % 25 == 0 or done == len(futures):
                print(f"{name}: {done}/{len(futures)} functions; {time.monotonic()-start:.1f}s; last={f}", flush=True)
    if not limit:
        aggregate(directory, meta, data, names, windows)


def aggregate(directory, meta, data, names, windows):
    n, a, s, w = len(data["counts"]), len(names), len(meta["seeds"]), len(windows)
    horizon = meta["horizon_minutes"]
    per_function = np.zeros((n, a, w, len(METRICS)))
    timeline = np.zeros((a, horizon + 1, 2))
    seed_totals = np.zeros((s, a, w, len(METRICS)))
    for f in range(n):
        with np.load(directory / "functions" / f"{f:05d}.npz") as z:
            mapping, weight = z["mapping"], data["weights"][f]
            per_function[f] = z["stats"][:,mapping].mean(axis=0)
            seed_totals += weight * z["stats"][:,mapping]
            timeline[:,:,0] += weight * z["cold"][mapping] / s
            timeline[:,:,1] += weight * z["idle"][mapping]
    total_curve = np.r_[np.einsum("f,ft->t", data["weights"], data["counts"]), 0.]
    save_npz(directory / "aggregate.npz", action_names=np.array(names), window_names=np.array(list(windows)),
             per_function=per_function, seed_totals=seed_totals, timeline=timeline, total_curve=total_curve,
             weights=data["weights"], apps=data["apps"], keys=data["keys"])
    summary = dict(case=meta["name"], surface=meta["surface"], provider=meta["provider"],
                   n_functions=n, seeds=meta["seeds"], windows=windows,
                   aggregate_sha256=digest(directory / "aggregate.npz"), by_action={})
    onset = meta.get("onset_minute", 0)
    for j, name in enumerate(names):
        record = meta["actions"][name]
        row = dict(method=record["method"], rho=record["rho"],
                   adaptation_lag=adaptation_lag(timeline[j,onset:horizon,0],total_curve[onset:horizon]), windows={})
        for k, label in enumerate(windows):
            values = seed_totals[:,j,k].mean(axis=0)
            d = dict(zip(METRICS, map(float, values)))
            inv = values[0]
            d.update(csr_pct=float(100 * values[1] / inv) if inv else None,
                     idle_per_1k=float(1000 * values[2] / inv) if inv else None,
                     cost_per_1k=float(1000 * (values[2] + 15 * record["rho"] * values[1]) / inv) if inv else None)
            row["windows"][label] = d
        summary["by_action"][name] = row
    # Cluster the seed-averaged paired outcomes by application; preserve weights.
    clusters, codes = np.unique(data["apps"], return_inverse=True)
    cluster = np.zeros((len(clusters), a, w, 3))
    for f in range(n):
        cluster[codes[f]] += data["weights"][f] * per_function[f,:,:,:3]
    rng = np.random.default_rng(260915)
    boot = np.zeros((2000, a, w, 3))
    for b in range(len(boot)):
        boot[b] = cluster[rng.integers(0, len(clusters), len(clusters))].sum(axis=0)
    summary["paired_bootstrap"] = dict(unit="application (all windows of a function grouped)",
        n_clusters=len(clusters), replicates=len(boot), seed=260915, contrasts={})
    for rho in meta["rhos"]:
        winter = f"WINTER__rho{rho:g}"
        if winter not in names:
            continue
        wi = names.index(winter)
        for comparator in ("LSTM_Fifer", "LSTM_shared", "EWMA_0.1", "EWMA_0.3", "Fourier", "Hybrid", "Chronos", "WINTER_G"):
            name = f"{comparator}__rho{rho:g}"
            if name not in names:
                continue
            ci = names.index(name)
            contrast = {}
            for k, label in enumerate(windows):
                if label == "drain":
                    continue
                denom = boot[:,wi,k,0]
                good = denom > 0
                delta = 100 * (boot[good,wi,k,1] - boot[good,ci,k,1]) / denom[good]
                contrast[label] = dict(winter_minus_comparator_csr_pp_ci95=np.percentile(delta, [2.5,97.5]).tolist()) if len(delta) else {}
            summary["paired_bootstrap"]["contrasts"][f"WINTER_vs_{comparator}__rho{rho:g}"] = contrast
    write_json(directory / "summary.json", summary)
    print("aggregated", meta["name"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(args.case, args.workers, args.limit)
