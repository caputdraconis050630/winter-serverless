"""Independently check frozen inputs, per-function ledgers and reported results.

Completed cases are cached by hashes. A partial report never declares the
whole suite validated. Final validation also requires the primary live run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from common import OUT, HERE, ROOT, METRICS, digest, read_json, write_json

EXPECTED = ["initial_azure_primary", "initial_huawei", "initial_azure_evaluation", "continuous_48h"]
EXPECTED += [f"steady_{p}_{s}" for p in ("azure2021", "azure2019", "huawei")
             for s in (["h_mixed", "h_sparse", "h_saturated"] if p == "huawei" else ["S1", "S2", "S3"])]
EXPECTED += [f"drift_{p}_{s}" for p in ("azure2021", "azure2019", "huawei") for s in ("natural", "synthetic")]
CODE = {digest(p): str(p.relative_to(HERE)) for p in HERE.rglob("*.py")}
DEST = OUT / "verification"


def equal(actual, expected, rtol=2e-9, atol=1e-6):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def lag_independent(cold, counts):
    # Explicit clipped neighborhoods give the same centered 15-minute readout.
    curve = []
    for t in range(len(counts)):
        a, b = max(0, t-7), min(len(counts), t+8)
        denominator = counts[a:b].sum()
        curve.append(cold[a:b].sum()/denominator if denominator else np.nan)
    curve = np.asarray(curve)
    tail = curve[len(curve)*3//4:]
    tail = tail[np.isfinite(tail)]
    if not len(tail):
        return None
    threshold = tail.mean()*1.1
    for t in range(len(curve)-30):
        block = curve[t:t+31]
        if np.isfinite(block).all() and block.max() <= threshold+1e-10:
            return t
    return None


def case(name):
    path = OUT / "cases" / name
    signature = {p: digest(path/p) for p in ("case.json", "execution.json", "aggregate.npz", "summary.json")}
    cache = DEST / (name+".json")
    if cache.exists():
        old = read_json(cache)
        if old["source_sha256"] == signature and old["verifier_sha256"] == digest(__file__):
            return old
    meta, contract, summary = [read_json(path/p) for p in ("case.json", "execution.json", "summary.json")]
    assert contract["case_sha256"] == signature["case.json"]
    assert summary["aggregate_sha256"] == signature["aggregate.npz"]
    assert digest(path/"data.npz") == meta["data_sha256"]
    provenance = {}
    for field in ("simulator_sha256", "events_sha256", "runner_sha256", "aggregate_sha256",
                  "residency_helpers_sha256", "reference_simulator_sha256"):
        if field in contract:
            assert contract[field] in CODE, (name, field, "source not retained")
            provenance[field] = CODE[contract[field]]
    for source in ("heap", "readylist"):
        key = f"migrated_{source}_contract_sha256"
        if contract.get(key):
            assert digest(path/f"execution_{source}.json") == contract[key]
    with np.load(path/"data.npz") as z:
        data = {k:z[k] for k in z.files}
    with np.load(path/"aggregate.npz") as z:
        aggregate = {k:z[k] for k in z.files}
    names = list(aggregate["action_names"])
    windows = list(aggregate["window_names"])
    assert names == sorted(meta["actions"]) == contract["actions"]
    assert set(windows) == set(summary["windows"]) == set(contract["windows"])
    assert tuple(contract["metrics"]) == METRICS
    assert contract["seeds"] == meta["seeds"] == summary["seeds"]
    count = data["counts"]
    n, horizon = count.shape
    seeds = len(meta["seeds"])
    assert n == meta["n_functions"] == summary["n_functions"]
    assert int(count.sum()) == meta["raw_invocations"]
    assert horizon == meta["horizon_minutes"]
    assert np.all(count >= 0) and np.all(data["weights"] > 0)
    for key in ("weights", "apps", "keys"):
        np.testing.assert_array_equal(data[key], aggregate[key])
    expected_curve = np.r_[np.einsum("f,ft->t", data["weights"], count), 0.]
    equal(aggregate["total_curve"], expected_curve)
    forecast_path = path/"lstm_forecasts.npz"
    forecast_hash = digest(forecast_path)
    with np.load(forecast_path) as z:
        lstm_rate = z["rate"]
    assert lstm_rate.shape == count.shape and np.isfinite(lstm_rate).all()
    for action, record in meta["actions"].items():
        file = path/record["file"]
        assert digest(file) == record["sha256"]
        with np.load(file) as z:
            q, ttl = z["q"], z["ttl"]
            assert q.shape == ttl.shape == count.shape
            assert np.isfinite(q).all() and np.isfinite(ttl).all()
            assert np.all((q >= 0) & (q <= contract["cap"])) and np.all(ttl >= 0)
            np.testing.assert_array_equal(q, np.rint(q))
            if record["method"].startswith("LSTM_"):
                assert record["source"]["forecast_sha256"] == forecast_hash
                checkpoint = Path(record["source"]["checkpoint"])
                training = read_json(checkpoint.parent/"training.json")
                assert digest(checkpoint) == training["checkpoint_sha256"]
            if record["method"] == "LSTM_Fifer":
                np.testing.assert_array_equal(q, np.ceil(np.clip(lstm_rate,0.,200.)))
                equal(ttl,10.,rtol=0.,atol=0.)
    summed = np.zeros_like(aggregate["seed_totals"])
    timeline = np.zeros_like(aggregate["timeline"])
    physical_jobs = 0
    physical_requests = 0
    function_hashes = hashlib.sha256()
    wi, di = windows.index("full"), windows.index("drain")
    for f in range(n):
        file = path/"functions"/f"{f:05d}.npz"
        function_hashes.update(bytes.fromhex(digest(file)))
        with np.load(file) as z:
            assert str(z["contract_sha256"]) == signature["execution.json"]
            assert int(z["function_index"]) == f
            mapping, stats, cold, idle = [z[k] for k in ("mapping", "stats", "cold", "idle")]
        assert stats.shape[0] == seeds and stats.shape[2:] == (len(windows), 8)
        assert np.isfinite(stats).all() and np.all(stats >= -1e-6)
        assert np.all(stats[:,:,:,1] <= stats[:,:,:,0]+1e-6)
        equal(stats[:,:,:,6], stats[:,:,:,1])
        assert np.all(stats[:,:,:,7] <= stats[:,:,:,1]+1e-6)
        # Total execution, including the terminal drain, is policy independent.
        execution = stats[:,:,wi,4]+stats[:,:,di,4]
        equal(execution, np.broadcast_to(execution[:,:1], execution.shape), rtol=3e-9, atol=1e-5)
        for w, label in enumerate(windows):
            lo, hi = contract["windows"][label]
            inv = count[f,lo:min(hi,horizon)].sum()
            equal(stats[:,:,w,0], inv)
            equal(cold[:,lo:hi].sum(1)/seeds, stats[:,:,w,1].mean(0))
            equal(idle[:,lo:hi].sum(1,dtype=np.float64), stats[:,:,w,2].mean(0), rtol=8e-7, atol=1e-4)
        equal(aggregate["per_function"][f], stats[:,mapping].mean(0))
        summed += data["weights"][f]*stats[:,mapping]
        timeline[:,:,0] += data["weights"][f]*cold[mapping]/seeds
        timeline[:,:,1] += data["weights"][f]*idle[mapping]
        physical_jobs += seeds*stats.shape[1]
        physical_requests += seeds*stats.shape[1]*int(count[f].sum())
    equal(summed, aggregate["seed_totals"])
    equal(timeline, aggregate["timeline"])
    weighted = np.einsum("f,fawm->awm", data["weights"], aggregate["per_function"])
    equal(weighted, summed.mean(0))
    for j, action in enumerate(names):
        row = summary["by_action"][action]
        onset = meta.get("onset_minute", 0)
        assert row["adaptation_lag"] == lag_independent(timeline[j,onset:horizon,0], expected_curve[onset:horizon])
        for w, label in enumerate(windows):
            reported = row["windows"][label]
            v = summed[:,j,w].mean(0)
            equal([reported[k] for k in METRICS], v)
            if v[0]:
                equal(reported["csr_pct"], 100*v[1]/v[0])
                equal(reported["idle_per_1k"], 1000*v[2]/v[0])
                equal(reported["cost_per_1k"], 1000*(v[2]+15*row["rho"]*v[1])/v[0])
            else:
                assert all(reported[k] is None for k in ("csr_pct", "idle_per_1k", "cost_per_1k"))
    # Native schedules must not change with the controller's cost ratio.
    for method in ("LSTM_Fifer", "Hybrid", "Keep_alive"):
        selected = [j for j, name in enumerate(names) if meta["actions"][name]["method"] == method]
        assert len({meta["actions"][names[j]]["sha256"] for j in selected}) == 1
        for j in selected[1:]:
            equal(summed[:,j], summed[:,selected[0]])
    if name == "continuous_48h":
        for rho in meta["rhos"]:
            g, baseline = [names.index(f"{m}__rho{rho:g}") for m in ("WINTER_G", "No_handoff")]
            equal(timeline[g,:720], timeline[baseline,:720], rtol=0., atol=0.)
    # Independently reproduce the principal paired bootstrap contrast.
    cluster_names, codes = np.unique(data["apps"], return_inverse=True)
    selected = [names.index(m+"__rho10") for m in ("WINTER", "LSTM_Fifer")]
    clusters = np.zeros((len(cluster_names),2,len(windows),2))
    for f in range(n):
        clusters[codes[f]] += data["weights"][f]*aggregate["per_function"][f,selected,:,:2]
    rng = np.random.default_rng(260915)
    differences = []
    for _ in range(2000):
        totals = clusters[rng.integers(0,len(clusters),len(clusters))].sum(0)
        denominator = totals[0,:,0]
        differences.append(np.divide(100*(totals[0,:,1]-totals[1,:,1]), denominator,
                                     out=np.full(len(windows),np.nan),where=denominator>0))
    differences = np.array(differences)
    contrast = summary["paired_bootstrap"]["contrasts"]["WINTER_vs_LSTM_Fifer__rho10"]
    for w, label in enumerate(windows):
        if label != "drain" and contrast[label]:
            equal(contrast[label]["winter_minus_comparator_csr_pp_ci95"],
                  np.nanpercentile(differences[:,w],[2.5,97.5]))
    report = dict(case=name,status="passed",source_sha256=signature,verifier_sha256=digest(__file__),
        source_files=provenance,functions=n,clusters=len(cluster_names),minutes=horizon,
        actions=len(names),seeds=seeds,logical_policy_function_seeds=n*len(names)*seeds,
        actual_policy_function_seeds=physical_jobs,actual_policy_request_events=physical_requests,
        raw_requests_per_seed=int(count.sum()),weighted_requests_per_seed=float(expected_curve.sum()),
        function_file_hash_chain=function_hashes.hexdigest(),
        checks=["frozen inputs and source hashes", "per-function and aggregate reconstruction",
                "common invocations and complete execution conservation", "nonnegative phase ledgers",
                "timeline/end-point agreement", "native policy rho invariance", "cost arithmetic",
                "independent centered-window AL", "independent principal paired cluster bootstrap"])
    write_json(cache, report)
    print("verified",name,flush=True)
    return report


def live():
    file = OUT / "strict_round/live/results.json"
    if not file.exists():
        return dict(status="pending")
    result = read_json(file)
    arms = result["arms"]
    assert len(arms) == 5
    with np.load(file.parent/"schedules.npz") as z:
        issued = z["counts"]
    pods = read_json(file.parent/"pod_timeline.json")["pods"]
    for name, arm in arms.items():
        records = arm["requests"]
        assert len(records) == arm["invocations"] == int(issued.sum()) == 686
        assert arm["truncated_by_cap"] == 0
        assert len(records) == sum(arm["http_status_counts"].values())
        observed = np.zeros_like(issued)
        for r in records:
            observed[r["func"],r["tick"]] += 1
            assert r["ok"] == (r["http_status"] is not None and 200 <= r["http_status"] < 300)
            assert r["latency"] >= 0
        def ready(record):
            for pod in pods[record["svc"]].values():
                begin = pod["ready_since"]
                ends = [pod[k] for k in ("drain_seen","gone") if pod[k] is not None]
                end = min(ends) if ends else float("inf")
                if begin is not None and begin <= record["t_start"] < end:
                    return True
            return False
        assert sum(not ready(r) for r in records) == arm["cold_event"]
        np.testing.assert_array_equal(observed,issued)
        assert sum(not r["ok"] for r in records) == arm["failed_requests"]
        equal(arm["csr_event"],arm["cold_event"]/686)
        equal(arm["csr_latency"],sum(r["latency"]>.7 for r in records)/686)
        assert arm["wall_sec"] >= 3600
    return dict(status="passed",source_sha256=digest(file),attempts=5*686,
                success="HTTP 2xx",arms={n:dict(http=a["http_status_counts"],failed=a["failed_requests"]) for n,a in arms.items()})


def models():
    import torch
    from model import CONFIG
    result = {}
    source = ROOT/"data/processed"
    source_hashes = {"source_counts_sha256":digest(source/"counts.npy"),
                     "source_splits_sha256":digest(source/"splits.npz")}
    with np.load(source/"splits.npz") as z:
        splits = {k:z[k] for k in z.files}
    for directory in sorted((OUT/"models").iterdir()):
        if not directory.is_dir():
            continue
        protocol, training = [read_json(directory/p) for p in ("protocol.json","training.json")]
        assert {k:protocol[k] for k in CONFIG} == CONFIG
        assert digest(directory/"model.pt") == training["checkpoint_sha256"]
        assert digest(directory/"protocol.json") == training["protocol_sha256"]
        for k,v in source_hashes.items():
            assert protocol[k] == v
        split = protocol["split"]
        np.testing.assert_array_equal(protocol["train_ids"],splits[split+"_train"])
        np.testing.assert_array_equal(protocol["validation_ids"],splits[split+"_val"])
        assert not set(protocol["train_ids"]) & set(protocol["validation_ids"])
        assert protocol["train_end"] == protocol["validation_start"] < protocol["validation_end"]
        losses = [r["validation_mse"] for r in training["history"]]
        assert int(np.argmin(losses))+1 == training["selected_epoch"]
        state = torch.load(directory/"model.pt",map_location="cpu",weights_only=True)
        assert state["config"] == CONFIG
        assert sum(p.numel() for p in state["state_dict"].values()) == training["parameters"] == 17217
        result[directory.name] = dict(checkpoint_sha256=training["checkpoint_sha256"],
            protocol_sha256=training["protocol_sha256"],training_sha256=digest(directory/"training.json"),
            selected_epoch=training["selected_epoch"],status="passed")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial",action="store_true")
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    rows,missing = {},[]
    for name in EXPECTED:
        if (OUT/"cases"/name/"summary.json").exists():
            rows[name] = case(name)
        else:
            missing.append(name)
    test_file = DEST/"tests.json"
    tests = read_json(test_file) if test_file.exists() else dict(status="pending")
    live_result = live()
    model_results = models()
    status = "passed" if not missing and tests["status"] == live_result["status"] == "passed" else "partial"
    report = dict(status=status,verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  verifier_sha256=digest(__file__),expected_cases=EXPECTED,missing_cases=missing,
                  cases=rows,tests=tests,live=live_result,models=model_results,
                  logical_policy_function_seeds=sum(v["logical_policy_function_seeds"] for v in rows.values()),
                  actual_policy_function_seeds=sum(v["actual_policy_function_seeds"] for v in rows.values()),
                  actual_policy_request_events=sum(v["actual_policy_request_events"] for v in rows.values()))
    write_json(DEST/"report.json",report)
    print({"status":status,"cases":len(rows),"missing":missing,"live":live_result["status"]},flush=True)
    if status != "passed" and not args.partial:
        raise SystemExit("Complete validation is still pending")


if __name__ == "__main__":
    main()
