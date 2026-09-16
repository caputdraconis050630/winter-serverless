# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""E4: root-cause diagnosis of the archived end-to-end episode (paper Table 2).

The archived run (results/runs/testbed_e2e.json) reports, for the reactive
arm, 18/18 cold invocations (p50 1.430 s) while simultaneously integrating
720 pod-seconds over a 900 s episode -- i.e. a pod appeared to be Running at
36 of the 45 samples.  Those two statements are hard to hold together, and
the JSON ships in the public artifact.  Three hypotheses came out of a code
audit:

H1  scripts/testbed_measure.py patches the CLUSTER-WIDE configmap
    config-autoscaler to stable-window=30s, scale-to-zero-grace-period=15s
    (its lines 74-77).  That patch is persistent; testbed_e2e.py neither
    sets nor resets it.  The archives were written 35 minutes apart in the
    same session (testbed_cdf.json 10:39Z, testbed_e2e.json 11:14Z), so the
    tuned autoscaler was very likely still in force.  Effective scale-down
    then lands at ~45 s against a 40 s inter-arrival gap (45 trace-min at
    3x compression), which would make 18/18 cold *correct but produced under
    an unreported configuration* -- and would make the "reactive" arm a
    0.75-trace-minute keep-alive policy rather than the Knative default.

H2  testbed_e2e.py's pod_count() greps for "Running" without excluding
    pods that are terminating (testbed_measure.py's pods_for() does exclude
    them), which would inflate pod-seconds.

H3  set_min_scale() patches spec.template annotations every tick, and a
    template change mints a new Knative revision; revision churn would
    force cold starts structurally on the ProtoWarm arm.

Design: replay the archived episode (function 157, tick 18120, 45 ticks at
3x) as a 2x2 -- {tuned, default} autoscaler x {reactive, protowarm} -- with
instrumentation added and nothing else changed.  H1 is confirmed if the
reactive arm is ~18/18 cold under 'tuned' and materially warmer under
'default'.

Instrumentation added (the original measured none of it):
  * 2 s poll of every pod's phase / readiness / deletionTimestamp
  * 5 s poll of the revision list (H3)
  * pod-seconds computed three ways: the original Running-grep sampling,
    the same sampling with terminating pods excluded, and an exact integral
    over the 2 s timeline requiring all containers ready
  * cold classified two ways: the original latency threshold, and from the
    pod timeline (was a ready pod serving this revision at request start?)

Writes results/runs/testbed_e2e_diag.json.  Touches no archive.
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import rates_a5, decisions_from_rates  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile        # noqa: E402
from src.models.heads import N_QUANTILES                       # noqa: E402
from src.meta.trainer import ANILMetaTrainer                   # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NS = "serverless-test"
SVC = "python-ml"
PORT = 31118
CLUSTER = "serverless-test"

# archived episode constants (testbed_e2e.py + testbed_e2e.json)
FUNC_IDX = 157
SEG_START = 18120
SEG_LEN = 45
COMPRESS = 3
TICK_SEC = 60.0 / COMPRESS
COLD_THRESH = 0.7
RHO = 10.0

POD_POLL_SEC = 2.0
REV_POLL_SEC = 5.0

AUTOSCALER = {
    # what testbed_measure.py leaves behind
    "tuned": {"stable-window": "30s", "scale-to-zero-grace-period": "15s"},
    # Knative defaults (60 s / 30 s) -- merge-null removes the keys
    "default": {"stable-window": None, "scale-to-zero-grace-period": None},
}


def sh(cmd, timeout=60):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    return r.stdout.strip(), r.returncode


def node_ip():
    out, _ = sh("docker inspect -f "
                "'{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "
                f"{CLUSTER}-control-plane")
    return out.strip().strip("'")


NODE_IP = node_ip()
HOST = f"{SVC}.{NS}.127.0.0.1.sslip.io"


def invoke():
    t0 = time.time()
    out, rc = sh(f"curl -s -o /dev/null -w '%{{time_total}}' -H 'Host: {HOST}' "
                 f"http://{NODE_IP}:{PORT} --connect-timeout 20 --max-time 40",
                 timeout=50)
    t1 = time.time()
    try:
        lat = float(out) if rc == 0 else None
    except ValueError:
        lat = None
    return {"t_start": t0, "t_end": t1, "latency": lat}


def set_min_scale(n):
    """Identical to testbed_e2e.set_min_scale (this is part of what H3 tests)."""
    sh(f"kubectl patch ksvc {SVC} -n {NS} --type merge -p "
       f"'{{\"spec\":{{\"template\":{{\"metadata\":{{\"annotations\":{{"
       f"\"autoscaling.knative.dev/min-scale\":\"{int(n)}\"}}}}}}}}}}'",
       timeout=30)


def pod_count_original():
    """Verbatim copy of testbed_e2e.pod_count (H2's suspect)."""
    out, _ = sh(f"kubectl get pods -n {NS} -l serving.knative.dev/service={SVC} "
                f"--no-headers 2>/dev/null | grep -c Running || true")
    try:
        return int(out)
    except ValueError:
        return 0


def pod_count_excl_terminating():
    """testbed_measure.pods_for semantics: drop terminating pods."""
    out, _ = sh(f"kubectl get pods -n {NS} -l serving.knative.dev/service={SVC} "
                f"--no-headers 2>/dev/null | grep -v Terminating | grep -c Running "
                f"|| true")
    try:
        return int(out)
    except ValueError:
        return 0


class PodWatcher(threading.Thread):
    """Poll pod state; build a timeline of ready/serving pods."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []      # (t, n_ready, n_running_any, [names])
        self.revisions = []    # (t, [revision names])
        self._stop = threading.Event()

    def run(self):
        last_rev = 0.0
        while not self._stop.is_set():
            t = time.time()
            out, rc = sh(f"kubectl get pods -n {NS} "
                         f"-l serving.knative.dev/service={SVC} -o json", timeout=30)
            n_ready = n_running = 0
            names = []
            if rc == 0 and out:
                try:
                    for p in json.loads(out).get("items", []):
                        st = p.get("status", {})
                        meta = p.get("metadata", {})
                        phase = st.get("phase")
                        terminating = meta.get("deletionTimestamp") is not None
                        cs = st.get("containerStatuses", []) or []
                        all_ready = bool(cs) and all(c.get("ready") for c in cs)
                        if phase == "Running":
                            n_running += 1
                        if phase == "Running" and all_ready and not terminating:
                            n_ready += 1
                            names.append(meta.get("name"))
                except json.JSONDecodeError:
                    pass
            self.samples.append({"t": t, "ready": n_ready,
                                 "running_any": n_running, "names": names})
            if time.time() - last_rev > REV_POLL_SEC:
                rout, rrc = sh(f"kubectl get revisions -n {NS} "
                               f"-l serving.knative.dev/service={SVC} "
                               f"--no-headers -o custom-columns=NAME:.metadata.name",
                               timeout=30)
                if rrc == 0:
                    revs = [x for x in rout.split("\n") if x.strip()]
                    self.revisions.append({"t": time.time(), "n": len(revs),
                                           "names": revs})
                last_rev = time.time()
            self._stop.wait(POD_POLL_SEC)

    def stop(self):
        self._stop.set()

    def ready_at(self, t):
        """Was a ready pod present at time t (nearest sample at or before t)?"""
        prev = None
        for s in self.samples:
            if s["t"] <= t:
                prev = s
            else:
                break
        return prev["ready"] if prev else 0


def set_autoscaler(mode):
    data = AUTOSCALER[mode]
    payload = json.dumps({"data": data})
    sh(f"kubectl patch configmap config-autoscaler -n knative-serving "
       f"--type merge -p '{payload}'", timeout=30)
    time.sleep(10)      # let the autoscaler pick up the new config
    out, _ = sh("kubectl get cm config-autoscaler -n knative-serving "
                "-o jsonpath='{.data.stable-window}|{.data.scale-to-zero-grace-period}'")
    return out.strip().strip("'")


def wait_scale_to_zero(watcher, max_wait=300):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if pod_count_excl_terminating() == 0:
            return True
        time.sleep(5)
    return False


def replay(seg, prewarm, watcher):
    """Same loop shape as testbed_e2e.replay, with instrumentation added."""
    invocations = []
    pods_orig = 0.0
    pods_excl = 0.0
    last = time.time()
    tick_samples = []

    set_min_scale(0)
    wait_scale_to_zero(watcher)

    t_start = time.time()
    for t in range(SEG_LEN):
        tick_t0 = time.time()
        if prewarm is not None and t + 1 < SEG_LEN:
            set_min_scale(min(int(prewarm[t + 1]), 3))
        now = time.time()
        po, pe = pod_count_original(), pod_count_excl_terminating()
        pods_orig += po * (now - last)
        pods_excl += pe * (now - last)
        last = now
        tick_samples.append({"tick": t, "t": now, "orig": po, "excl": pe})

        n = int(seg[t])
        for _ in range(min(n, 5)):
            rec = invoke()
            rec["tick"] = t
            invocations.append(rec)
        rem = TICK_SEC - (time.time() - tick_t0)
        if rem > 0:
            time.sleep(rem)
    now = time.time()
    po, pe = pod_count_original(), pod_count_excl_terminating()
    pods_orig += po * (now - last)
    pods_excl += pe * (now - last)
    set_min_scale(0)

    # exact pod-seconds from the 2 s timeline (ready pods only)
    exact = 0.0
    samples = [s for s in watcher.samples if t_start <= s["t"] <= now]
    for a, b in zip(samples, samples[1:]):
        exact += a["ready"] * (b["t"] - a["t"])

    lat = np.array([r["latency"] for r in invocations
                    if r["latency"] is not None])
    cold_thresh = int((lat > COLD_THRESH).sum())
    cold_event = sum(1 for r in invocations
                     if watcher.ready_at(r["t_start"]) == 0)
    agree = sum(1 for r in invocations
                if (r["latency"] is not None)
                and ((r["latency"] > COLD_THRESH) ==
                     (watcher.ready_at(r["t_start"]) == 0)))
    return {
        "invocations": len(invocations),
        "cold_by_latency": cold_thresh,
        "cold_by_pod_event": cold_event,
        "classifier_agreement": agree / max(1, len(invocations)),
        "csr_latency": cold_thresh / max(1, len(invocations)),
        "csr_event": cold_event / max(1, len(invocations)),
        "pod_seconds_original_grep": pods_orig,
        "pod_seconds_excl_terminating": pods_excl,
        "pod_seconds_exact_ready": exact,
        "lat_p50": float(np.percentile(lat, 50)) if len(lat) else None,
        "lat_p95": float(np.percentile(lat, 95)) if len(lat) else None,
        "wall_sec": now - t_start,
        "per_invocation": invocations,
        "tick_samples": tick_samples,
    }


def main():
    counts = np.load(PROCESSED_DIR / "counts.npy")
    features = np.load(PROCESSED_DIR / "features.npy")
    seg = counts[FUNC_IDX, SEG_START:SEG_START + SEG_LEN]

    print(f"node ip {NODE_IP}; segment func {FUNC_IDX}@{SEG_START} "
          f"sum={seg.sum()}")

    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1,
                              device=DEVICE)
    trainer.load(RUNS_DIR / "best_anil_ridge_s1_s0.pt")
    hist_start = max(0, SEG_START - 2880)
    rates = rates_a5(counts[FUNC_IDX:FUNC_IDX + 1, hist_start:SEG_START + SEG_LEN],
                     features[FUNC_IDX:FUNC_IDX + 1, hist_start:SEG_START + SEG_LEN],
                     trainer, DEVICE)
    tau = newsvendor_quantile(RHO)
    pw, _ = decisions_from_rates(rates[:, -SEG_LEN:], tau)
    schedule = pw[0]

    arch_path = RUNS_DIR / "testbed_e2e.json"
    arch = json.load(open(arch_path)) if arch_path.exists() else None
    sched_match = None
    if arch is not None:
        sched_match = (list(map(int, schedule)) ==
                       list(map(int, arch["a5_schedule"])))
        print(f"schedule replication vs archive: {sched_match}")

    out = {
        "config": "E4 root-cause diagnosis of the archived Table 2 episode",
        "node_ip": NODE_IP,
        "episode": {"func_idx": FUNC_IDX, "segment_start": SEG_START,
                    "seg_len": SEG_LEN, "compress": COMPRESS,
                    "tick_sec": TICK_SEC, "rho": RHO,
                    "cold_thresh_s": COLD_THRESH,
                    "segment_counts": seg.tolist()},
        "schedule": list(map(int, schedule)),
        "schedule_matches_archive": sched_match,
        "hypotheses": {
            "H1": "persistent tuned autoscaler (30s/15s) from testbed_measure.py",
            "H2": "pod_count() counts terminating pods",
            "H3": "per-tick ksvc patch mints new revisions",
        },
        "arms": {},
    }

    watcher = PodWatcher()
    watcher.start()
    try:
        for mode in ("tuned", "default"):
            eff = set_autoscaler(mode)
            print(f"\n=== autoscaler {mode} -> '{eff}'")
            for arm, sched in (("reactive", None), ("protowarm", schedule)):
                print(f"--- arm {arm} ({mode}) ...", flush=True)
                rev_before = len(watcher.revisions[-1]["names"]) \
                    if watcher.revisions else 0
                res = replay(seg, sched, watcher)
                rev_after = len(watcher.revisions[-1]["names"]) \
                    if watcher.revisions else 0
                res["autoscaler_mode"] = mode
                res["autoscaler_effective"] = eff
                res["revisions_before"] = rev_before
                res["revisions_after"] = rev_after
                res["revisions_created"] = rev_after - rev_before
                out["arms"][f"{mode}/{arm}"] = res
                print(f"    cold(lat) {res['cold_by_latency']}/{res['invocations']} "
                      f"cold(event) {res['cold_by_pod_event']}/{res['invocations']} "
                      f"agree {res['classifier_agreement']:.2f} "
                      f"pod-s orig {res['pod_seconds_original_grep']:.0f} "
                      f"excl {res['pod_seconds_excl_terminating']:.0f} "
                      f"exact {res['pod_seconds_exact_ready']:.0f} "
                      f"revs +{res['revisions_created']}", flush=True)
    finally:
        watcher.stop()
        set_autoscaler("default")

    out["pod_timeline"] = watcher.samples
    out["revision_timeline"] = watcher.revisions
    if arch is not None:
        out["archived_reactive"] = arch["reactive"]
        out["archived_protowarm"] = arch["a5_prewarm"]

    path = RUNS_DIR / "testbed_e2e_diag.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0, "corrupt write (sdb guard)"
    json.load(open(path))
    print(f"\nSaved + verified {path}")


if __name__ == "__main__":
    main()
