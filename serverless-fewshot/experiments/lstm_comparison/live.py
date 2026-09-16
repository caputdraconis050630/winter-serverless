"""Isolated, one-hour Knative schedule replay with five concurrent arms.

This is a one-pod actuation check, not a live deployment of the predictors.
All schedules depend only on completed count bins. Five arms use separate
services on the same node, so shared-node interference remains a limitation.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import numpy as np
import torch

from common import ROOT, OUT, freeze, write_json, save_npz, digest, read_json
from model import load_model, predict, native_actions
sys.path.insert(0, str(ROOT))
from scripts import testbed_cohort as live
from scripts.phase63_onboarding_drift import build_prototypes
from src.data.features import features_from_counts


def causal_schedule(counts, q, ttl):
    schedule = (q > 0).astype(np.int64)
    for f in range(len(counts)):
        last = -100000
        for t in range(counts.shape[1]):
            if t and counts[f,t-1] > 0:
                last = t-1
            if t-last < ttl[f,t]:
                schedule[f,t] = 1
    return schedule


def main():
    directory = OUT / "live"
    directory.mkdir(exist_ok=True)
    if (directory / "results.json").exists():
        print("live replay already complete", flush=True)
        return
    archive_path = ROOT / "results/runs/testbed_cohort.json"
    archive = read_json(archive_path)
    pts = list(zip(archive["func_ids"], archive["onboard_t"]))
    proc = ROOT / "data/processed"
    counts = np.load(proc / "counts.npy").copy()
    feats = np.load(proc / "features.npy").copy()
    splits = np.load(proc / "splits.npz")
    issued = np.array(archive["issued_counts"], np.int64)
    # Keep the original source pool for prototypes; update target features to
    # the counts the load generator actually issues.
    trainer = live.ANILMetaTrainer(body_type="tcn", head_type="ridge", in_features=11,
        embedding_dim=64, n_quantiles=live.N_QUANTILES, n_horizons=1, device=live.DEVICE)
    checkpoint = ROOT / "results_azure2021/runs/best_anil_ridge_s1_s0.pt"
    trainer.load(checkpoint)
    pm = build_prototypes(trainer, feats, counts, splits["s1_train"])
    for f, (idx,start) in enumerate(pts):
        counts[idx,start:start+60] = issued[f]
        feats[idx] = features_from_counts(counts[idx], app_corate=feats[idx,:,10])
    seg, _, decisions = live.build_schedules(counts, feats, pts, trainer, pm)
    np.testing.assert_array_equal(seg, issued)
    rates = predict(load_model("s1"), issued)
    decisions["lstm_fifer"] = native_actions(rates)
    schedules = {k: causal_schedule(issued, *v) for k,v in decisions.items()}
    schedules["reactive"][:] = 0
    live.NS = "winter-lstm-eval-v2"
    arms = ["reactive", "keepalive10", "ewma", "protowarm", "lstm_fifer"]
    names = {arm: [f"a{a}-f{f:02d}" for f in range(10)] for a,arm in enumerate(arms)}
    all_names = [n for arm in arms for n in names[arm]]
    cfg = json.loads(subprocess.check_output(["kubectl","get","cm","config-autoscaler","-n","knative-serving","-o","json"]))["data"]
    if cfg.get("stable-window") != "6s" or cfg.get("scale-to-zero-grace-period") != "1s":
        raise RuntimeError("Testbed autoscaler differs from the recorded 6s/1s configuration")
    contract = dict(archive_sha256=digest(archive_path), checkpoint_sha256=digest(checkpoint),
        namespace=live.NS, arms=arms, minutes=60, invocations_per_arm=int(issued.sum()),
        source_split="Azure2021 S1", predictor_input="issued counts, strictly completed bins",
        live_mode="precomputed schedules; one pod per function; five concurrent isolated service groups",
        scheduling="q>0 or past invocation within current TTL; reactive remains min-scale zero",
        resource_requests="128Mi per function pod; original per-runtime memory limits; platform termination grace",
        autoscaler={k:cfg[k] for k in ("stable-window","scale-to-zero-grace-period")},
        duration="one trace minute per wall minute", cohort=pts, names=names)
    freeze(directory / "protocol.json", contract)
    save_npz(directory / "schedules.npz", counts=issued, **schedules)
    existing = subprocess.run(["kubectl","get","namespace",live.NS],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if existing.returncode:
        subprocess.run(["kubectl","create","namespace",live.NS], check=True)
    services = []
    for arm in arms:
        for f, name in enumerate(names[arm]):
            _, image, memory = live.IMAGES[f % len(live.IMAGES)]
            services.append(dict(apiVersion="serving.knative.dev/v1",kind="Service",
                metadata=dict(name=name,namespace=live.NS),spec=dict(template=dict(
                    metadata=dict(annotations={"autoscaling.knative.dev/min-scale":"0",
                        "autoscaling.knative.dev/max-scale":"1","autoscaling.knative.dev/target":"1"}),
                    spec=dict(containers=[dict(image=image,imagePullPolicy="IfNotPresent",
                        ports=[dict(containerPort=8080)],resources=dict(requests=dict(memory="128Mi"),limits=dict(memory=memory)))])))))
    manifest = directory / "services.json"
    write_json(manifest,dict(apiVersion="v1",kind="List",items=services))
    subprocess.run(["kubectl","apply","-f",str(manifest)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(["kubectl","wait","ksvc","--all","-n",live.NS,"--for=condition=Ready","--timeout=300s"],check=True)
    pas = {arm:[live.pa_name(n) for n in names[arm]] for arm in arms}
    if not all(p for a in arms for p in pas[a]):
        raise RuntimeError("A PodAutoscaler is missing")
    # Wait for deployment-created pods, including draining pods, to disappear.
    # This prevents the initial deployment from supplying warm capacity or
    # reserving the node's pod slots during the measured replay.
    print("live: waiting for deployment washout",flush=True)
    deadline = time.monotonic()+420
    while True:
        pods=json.loads(subprocess.check_output(["kubectl","get","pods","-n",live.NS,"-o","json"]))["items"]
        if not pods:break
        if time.monotonic()>deadline:raise RuntimeError("Deployment pods did not drain before measurement")
        time.sleep(2)
    watcher = live.PodWatcher(all_names)
    watcher.start()
    # Populate readiness intervals before run_arm tests for scale-to-zero.
    while not watcher.samples:
        time.sleep(.1)
    results = dict(protocol=contract, arms={}, start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    try:
        with ThreadPoolExecutor(max_workers=len(arms)) as pool:
            futures = {pool.submit(live.run_arm, arm, names[arm], pas[arm], issued, schedules[arm], watcher):arm for arm in arms}
            for future in as_completed(futures):
                arm = futures[future]
                results["arms"][arm] = future.result()
                write_json(directory / "progress.json", results)
                print("live completed", arm, results["arms"][arm]["csr_event"], flush=True)
    finally:
        watcher.stop()
        write_json(directory / "pod_timeline.json", dict(pods=watcher.pods, samples=watcher.samples))
        # Only services created by this experiment are removed.
        subprocess.run(["kubectl","delete","ksvc",*all_names,"-n",live.NS,"--wait=false"],check=False)
    if len(results["arms"]) == len(arms):
        write_json(directory / "results.json", results)
        print("live replay complete", flush=True)


if __name__ == "__main__":
    main()
