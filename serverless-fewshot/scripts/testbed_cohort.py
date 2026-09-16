# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""E5: cohort-scale Knative measurement with the paper's own policy arms.

What the archived episode could not answer
------------------------------------------
results/runs/testbed_e2e.json replays ONE function for 45 trace-minutes at
3x compression and compares exactly two arms: Knative's reactive default and
a ProtoWarm schedule.  Neither an EWMA arm nor a fixed keep-alive arm was
ever run on hardware, so the paper's actual contested claim -- that a learned
predictor beats a cheap per-function estimator during onboarding, and ties it
once converged -- has no physical counterpart at all.  The segment was also
picked by maximising a burstiness score, and 3x compression shrinks the
inter-arrival gaps while leaving cold-start duration and the autoscaler's
timers uncompressed.

This run removes all four limitations:
  * 10 functions instead of 1, drawn by stratified sample (not by score)
  * onboarding segments -- each function's first 60 minutes of life -- which
    is the regime where the paper claims learning wins
  * four arms: reactive / keepalive10 (=B1) / ewma (=B4a) / protowarm
    (=gate v3), the last three matching DES arms one-for-one so
    testbed_cohort_desmirror.py can pair predicted against measured
  * real time, no compression: one trace-minute is one wall minute, so
    keep-alive windows, cold-start duration and autoscaler timers stay on a
    single consistent clock

Mechanism notes
---------------
min-scale is applied by patching the PodAutoscaler object, NOT the Service
template.  A template patch mints a new Knative revision on every tick
(hypothesis H3 in testbed_e2e_diag.py); patching the PA was verified to
leave the revision count unchanged while still starting a pod.

The autoscaler is pinned to stable-window=6s / scale-to-zero-grace-period=0s
for every arm so that the min-scale schedule, i.e. the policy, dominates pod
lifetime instead of Knative's default ~90 s of implicit keep-alive.  The
setting is recorded in the output; it is the same for all arms, so it cannot
favour one.

Cold starts are classified primarily from the pod timeline (was a ready pod
serving this function when the request started?) and secondarily by the
latency threshold the archived script used; the agreement rate between the
two classifiers is reported rather than assumed.

Writes results/runs/testbed_cohort.json.
"""

import argparse
import http.client
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

from scripts.phase63_onboarding_drift import (  # noqa: E402
    L, find_onboarding_points, PROCESSED_DIR, RUNS_DIR,
)
from scripts.phase6_des import decisions_from_rates             # noqa: E402
from src.decision.newsvendor import newsvendor_quantile         # noqa: E402
from src.models.heads import N_QUANTILES                        # noqa: E402
from src.meta.trainer import ANILMetaTrainer                    # noqa: E402
from src.models.prototypes import PrototypeManager              # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NS = "serverless-test"
CLUSTER = "serverless-test"
PORT = 31118

N_FUNCS = 10
SEG_MIN = 60                 # trace minutes == wall minutes (no compression)
TICK_SEC = 60.0
RHO = 10.0
# Per-tick request cap. The archived episode used 5, which truncates 38% of
# this cohort's invocations; 20 cuts that to 16% at a peak of ~320 req/min
# across the cohort, which this cluster absorbs easily. Truncation is not a
# confound either way: the issued counts are recorded and the DES mirror
# simulates exactly those, not the raw trace counts.
CAP_PER_TICK = 20
# Cohort band: functions whose first hour carries enough events to measure
# but not so many that the cap dominates.
VOL_MIN, VOL_MAX = 10, 400
COLD_THRESH = 0.7
GATE_THRESHOLD = 100         # registered constant
COHORT_SEED = 20260727
POD_POLL_SEC = 2.0
N_CLUSTERS = 16              # registered prototype count

# One service per cohort function; images cycle over the three real runtimes.
IMAGES = [("python-ml", "dev.local/testbed-python-ml:v1", "512Mi"),
          ("node-api", "dev.local/testbed-node-api:v1", "128Mi"),
          ("java-svc", "dev.local/testbed-java-svc:v1", "512Mi")]

ARMS = ["reactive", "keepalive10", "ewma", "protowarm"]


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


# ----------------------------------------------------------------- platform
def pin_autoscaler():
    """Make the min-scale schedule -- i.e. the policy -- dominate pod lifetime.

    Left at Knative's defaults every arm inherits ~90 s (60 s stable window +
    30 s grace) of implicit keep-alive, which is 1.5 ticks at this run's
    60 s tick and would blur the arms into each other.  The webhook enforces
    bounds that vary by release, so ask for the tightest setting and fall
    back until one is accepted; whichever lands is shared by all four arms
    and is recorded in the output, so it cannot favour any of them.
    """
    # The admission webhook rejects a zero grace period ("must be positive"),
    # so every candidate below uses a positive one; 6s/1s was verified to be
    # accepted on this cluster.
    for sw, gp in (("6s", "1s"), ("6s", "5s"), ("15s", "5s"), ("60s", "30s")):
        payload = json.dumps({"data": {"stable-window": sw,
                                       "scale-to-zero-grace-period": gp}})
        _, rc = sh(f"kubectl patch configmap config-autoscaler -n knative-serving "
                   f"--type merge -p '{payload}'", timeout=30)
        time.sleep(8)
        out, _ = sh("kubectl get cm config-autoscaler -n knative-serving -o "
                    "jsonpath='{.data.stable-window}|"
                    "{.data.scale-to-zero-grace-period}'")
        eff = out.strip().strip("'")
        if rc == 0 and eff == f"{sw}|{gp}":
            return eff
        print(f"  autoscaler {sw}/{gp} rejected (rc={rc}, effective '{eff}')")
    return eff


def deploy_services(names):
    docs = []
    for i, n in enumerate(names):
        _, image, mem = IMAGES[i % len(IMAGES)]
        docs.append(f"""apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: {n}
  namespace: {NS}
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/min-scale: "0"
        autoscaling.knative.dev/max-scale: "1"
        autoscaling.knative.dev/target: "1"
    spec:
      containers:
        - image: {image}
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8080
          resources:
            requests:
              memory: {mem}
            limits:
              memory: {mem}
""")
    manifest = "---\n".join(docs)
    p = Path("/tmp/testbed_cohort_ksvc.yaml")
    p.write_text(manifest)
    sh(f"kubectl apply -f {p}", timeout=180)
    for n in names:
        sh(f"kubectl wait --for=condition=Ready ksvc/{n} -n {NS} --timeout=300s",
           timeout=320)


def pa_name(svc):
    out, _ = sh(f"kubectl get pa -n {NS} -l serving.knative.dev/service={svc} "
                f"--no-headers -o custom-columns=NAME:.metadata.name")
    return out.split("\n")[0].strip() if out else None


def set_min_scale(pa, n):
    """Churn-free min-scale: patch the PodAutoscaler, never the Service."""
    sh(f"kubectl patch pa {pa} -n {NS} --type merge -p "
       f"'{{\"metadata\":{{\"annotations\":{{"
       f"\"autoscaling.knative.dev/min-scale\":\"{int(n)}\"}}}}}}'", timeout=30)


def _rfc3339(s):
    """Kubernetes RFC3339 timestamp -> epoch seconds."""
    from datetime import datetime, timezone
    if not s:
        return None
    # condition timestamps are second-precision, but some fields carry
    # fractional seconds -- accept both rather than crash on the rare one
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return (datetime.strptime(s, fmt)
                    .replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


class PodWatcher(threading.Thread):
    """Track each pod's readiness INTERVAL, not a sampled ready-count.

    Sampling a count every POD_POLL_SEC cannot classify a burst: up to
    CAP_PER_TICK requests complete between two polls, so if the pod becomes
    ready inside that gap the whole burst inherits the stale pre-burst
    sample and is scored cold.  Measured directly: an isolated request pair
    classified 6/6 correctly, while bursty arms agreed with the latency
    classifier only 60% of the time.

    So take the readiness instant from the API server instead of from the
    poll: status.conditions[Ready].lastTransitionTime is a server-side
    timestamp whose precision does not depend on how often we look.  Polling
    then only has to DISCOVER a pod, not time it.
    """

    def __init__(self, services):
        super().__init__(daemon=True)
        self.services = services
        # svc -> {pod_name: {"ready_since", "drain_seen", "delete_deadline",
        #                    "gone"}} epochs|None
        self.pods = {s: {} for s in services}
        self.samples = []          # coarse trace, kept for diagnostics only
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            t = time.time()
            out, rc = sh(f"kubectl get pods -n {NS} -o json", timeout=30)
            seen = {s: set() for s in self.services}
            ready_now = {s: 0 for s in self.services}
            if rc == 0 and out:
                try:
                    items = json.loads(out).get("items", [])
                except json.JSONDecodeError:
                    items = []
                for p in items:
                    meta, st = p.get("metadata", {}), p.get("status", {})
                    svc = (meta.get("labels", {})
                               .get("serving.knative.dev/service"))
                    if svc not in self.pods:
                        continue
                    name = meta.get("name")
                    seen[svc].add(name)
                    ready_since = None
                    for c in st.get("conditions", []) or []:
                        if c.get("type") == "Ready" and c.get("status") == "True":
                            ready_since = _rfc3339(c.get("lastTransitionTime"))
                    # A pod with a deletionTimestamp is draining: Knative has
                    # pulled it from the endpoints, so it serves no new request
                    # and must stop counting as warm capacity.
                    #
                    # Careful: deletionTimestamp is NOT when deletion started.
                    # Kubernetes sets it to now + terminationGracePeriodSeconds,
                    # and Knative's grace period is ~300 s, so the field points
                    # ~5 minutes into the future -- measured directly: pods that
                    # became ready at +0/+22/+44 s carried deletionTimestamps of
                    # +307/+329/+351 s. Using that value as the end truncated
                    # nothing, three successive pods overlapped across the whole
                    # window, and pod-seconds hit exactly the 2x ceiling.
                    # So take the instant the field is first OBSERVED instead.
                    draining = meta.get("deletionTimestamp") is not None
                    with self._lock:
                        rec = self.pods[svc].setdefault(
                            name, {"ready_since": None, "drain_seen": None,
                                   "delete_deadline": None, "gone": None})
                        if ready_since is not None and rec["ready_since"] is None:
                            rec["ready_since"] = ready_since
                        if draining and rec["drain_seen"] is None:
                            rec["drain_seen"] = t
                            rec["delete_deadline"] = _rfc3339(
                                meta.get("deletionTimestamp"))
                        if ready_since is not None and not draining:
                            ready_now[svc] += 1
                # pods that vanished since the previous poll are finished
                with self._lock:
                    for svc, known in self.pods.items():
                        for name, rec in known.items():
                            if name not in seen[svc] and rec["gone"] is None:
                                rec["gone"] = t
            self.samples.append({"t": t, "ready": ready_now})
            self._stop.wait(POD_POLL_SEC)

    def stop(self):
        self._stop.set()

    @staticmethod
    def _serving_end(rec, default):
        """When the pod stopped being able to serve a new request."""
        ends = [e for e in (rec["drain_seen"], rec["gone"]) if e is not None]
        return min(ends) if ends else default

    def ready_at(self, svc, t):
        """Number of pods serving `svc` that were already ready at time t."""
        n = 0
        with self._lock:
            for rec in self.pods.get(svc, {}).values():
                rs = rec["ready_since"]
                end = self._serving_end(rec, None)
                if rs is not None and rs <= t and (end is None or t < end):
                    n += 1
        return n

    def pod_seconds(self, svc, t0, t1):
        tot = 0.0
        with self._lock:
            for rec in self.pods.get(svc, {}).values():
                rs = rec["ready_since"]
                if rs is None:
                    continue
                end = self._serving_end(rec, t1)
                tot += max(0.0, min(end, t1) - max(rs, t0))
        return tot


def invoke(svc):
    host = f"{svc}.{NS}.127.0.0.1.sslip.io"
    t0 = time.time()
    ok = False
    try:
        conn = http.client.HTTPConnection(NODE_IP, PORT, timeout=60)
        conn.request("GET", "/", headers={"Host": host})
        resp = conn.getresponse()
        resp.read()
        ok = 200 <= resp.status < 500
        conn.close()
    except Exception:
        ok = False
    t1 = time.time()
    return {"svc": svc, "t_start": t0, "latency": t1 - t0, "ok": ok}


# ------------------------------------------------------------------ schedules
def warm_schedule(seg, pw, ka_min):
    """DES semantics -> min-scale signal.

    A pod is wanted at tick t when the policy prewarms t, or when the
    policy's keep-alive window still covers an invocation in (t-ka, t].
    """
    T = len(seg)
    warm = np.zeros(T, dtype=int)
    last = -10 ** 9
    for t in range(T):
        if seg[t] > 0:
            last = t
        if pw[t] > 0 or (t - last) < ka_min:
            warm[t] = 1
    return warm


def build_schedules(counts, features, pts, trainer, pm):
    """Per-arm min-scale schedules for the cohort, from the paper's policies."""
    F_n = len(pts)
    seg = np.stack([counts[fi, t0:t0 + SEG_MIN] for (fi, t0) in pts])
    tau = newsvendor_quantile(RHO)

    # --- embeddings for the learned arm (per-tick, online) ---
    feats_t = torch.from_numpy(features).float()
    trainer.body.eval()
    phi = torch.zeros(F_n, SEG_MIN, 64)
    with torch.no_grad():
        for w in range(SEG_MIN):
            batch = []
            for (fi, t0) in pts:
                t = t0 + w
                if t < L:
                    pad = torch.zeros(L - t, features.shape[2])
                    win = torch.cat([pad, feats_t[fi, :t]], dim=0)
                else:
                    win = feats_t[fi, t - L:t]
                batch.append(win)
            phi[:, w] = trainer.body(torch.stack(batch).to(DEVICE)).cpu()

    y = torch.from_numpy(np.stack(
        [np.log1p(counts[fi, t0:t0 + SEG_MIN]) for (fi, t0) in pts])).float()
    lam = trainer.head.ridge_lambda.item()
    eye = torch.eye(64)
    cen, pw_all = pm.centroids.cpu(), pm.proto_weights.cpu()
    n_out = N_QUANTILES
    pred_learned = np.zeros((F_n, SEG_MIN, n_out), dtype=np.float32)
    pred_zero = np.zeros((F_n, SEG_MIN, n_out), dtype=np.float32)
    for w in range(SEG_MIN):
        p_now = phi[:, w]
        sims = torch.nn.functional.cosine_similarity(
            p_now.unsqueeze(1), cen.unsqueeze(0), dim=2)
        Wp = pw_all[sims.argmax(dim=1)]
        if w == 0:
            W_head = Wp
        else:
            phi_s = phi[:, :w]
            y_s = y[:, :w].unsqueeze(-1).expand(F_n, w, n_out)
            PhiT = phi_s.transpose(1, 2)
            A = PhiT @ phi_s + lam * eye.unsqueeze(0)
            W_head = torch.linalg.solve(A, PhiT @ y_s + lam * Wp)
        pred_learned[:, w] = (p_now.unsqueeze(1) @ W_head).squeeze(1).numpy()
        pred_zero[:, w] = (p_now.unsqueeze(1) @ Wp).squeeze(1).numpy()

    ql = np.linspace(0.05, 0.95, n_out)

    def rate_of(pred):
        r = np.zeros((F_n, SEG_MIN), dtype=np.float32)
        for f in range(F_n):
            for w in range(SEG_MIN):
                r[f, w] = np.expm1(max(np.interp(tau, ql, pred[f, w]), 0.0))
        return r

    rates = {"learned": rate_of(pred_learned), "zero": rate_of(pred_zero)}

    ewma = np.zeros(F_n)
    rates["ewma"] = np.zeros((F_n, SEG_MIN), dtype=np.float32)
    for w in range(SEG_MIN):
        rates["ewma"][:, w] = np.expm1(np.maximum(ewma, 0))
        ewma = 0.1 * np.log1p(seg[:, w].astype(np.float64)) + 0.9 * ewma

    cum = np.zeros((F_n, SEG_MIN), dtype=np.int64)
    cum[:, 1:] = np.cumsum(seg[:, :-1], axis=1)
    zero_h, mid = cum == 0, (cum > 0) & (cum < GATE_THRESHOLD)
    rates["protowarm"] = np.where(zero_h, rates["zero"],
                          np.where(mid, rates["ewma"], rates["learned"]))

    sched, des_dec = {}, {}
    for arm, key in (("ewma", "ewma"), ("protowarm", "protowarm")):
        pw_a, ka_a = decisions_from_rates(rates[key], tau)
        des_dec[arm] = (pw_a, ka_a)
        sched[arm] = np.stack([warm_schedule(seg[f], pw_a[f], ka_a[f].mean())
                               for f in range(F_n)])
    # B1: fixed 10-minute keep-alive, no prediction
    pw_b1 = np.zeros((F_n, SEG_MIN), dtype=np.int32)
    ka_b1 = np.full((F_n, SEG_MIN), 10.0, np.float32)
    des_dec["keepalive10"] = (pw_b1, ka_b1)
    sched["keepalive10"] = np.stack([warm_schedule(seg[f], pw_b1[f], 10)
                                     for f in range(F_n)])
    sched["reactive"] = np.zeros((F_n, SEG_MIN), dtype=int)
    des_dec["reactive"] = (np.zeros((F_n, SEG_MIN), dtype=np.int32),
                           np.zeros((F_n, SEG_MIN), np.float32))
    return seg, sched, des_dec


# ---------------------------------------------------------------------- run
def run_arm(arm, names, pas, seg, sched, watcher):
    F_n = len(names)
    for pa in pas:
        set_min_scale(pa, 0)
    print(f"    [{arm}] waiting for scale-to-zero", flush=True)
    t_wait = time.time()
    while time.time() - t_wait < 180:
        if all(watcher.ready_at(n, time.time()) == 0 for n in names):
            break
        time.sleep(5)

    records = []
    truncated = 0
    t0 = time.time()
    for t in range(SEG_MIN):
        tick_t0 = time.time()
        # patch only on a change (and once at t=0): fewer API calls, and the
        # PA object keeps its value between ticks anyway
        for f in range(F_n):
            if t == 0 or sched[f, t] != sched[f, t - 1]:
                set_min_scale(pas[f], int(sched[f, t]))
        threads, out = [], []
        lock = threading.Lock()

        def fire(fi):
            n = int(seg[fi, t])
            truncate = max(0, n - CAP_PER_TICK)
            for _ in range(min(n, CAP_PER_TICK)):
                rec = invoke(names[fi])
                rec["tick"] = t
                rec["func"] = fi
                with lock:
                    out.append(rec)
            return truncate

        trunc_local = []
        for f in range(F_n):
            if int(seg[f, t]) > 0:
                th = threading.Thread(target=lambda i=f:
                                      trunc_local.append(fire(i)))
                th.start()
                threads.append(th)
        for th in threads:
            th.join()
        truncated += sum(trunc_local)
        records.extend(out)
        rem = TICK_SEC - (time.time() - tick_t0)
        if rem > 0:
            time.sleep(rem)
    t1 = time.time()
    for pa in pas:
        set_min_scale(pa, 0)

    per_func = []
    for f, n in enumerate(names):
        rs = [r for r in records if r["func"] == f]
        cold_ev = sum(1 for r in rs if watcher.ready_at(n, r["t_start"]) == 0)
        cold_lat = sum(1 for r in rs if r["latency"] > COLD_THRESH)
        agree = sum(1 for r in rs
                    if (r["latency"] > COLD_THRESH) ==
                       (watcher.ready_at(n, r["t_start"]) == 0))
        per_func.append({
            "func": f, "service": n, "invocations": len(rs),
            "cold_event": cold_ev, "cold_latency": cold_lat,
            "csr_event": cold_ev / max(1, len(rs)),
            "csr_latency": cold_lat / max(1, len(rs)),
            "agreement": agree / max(1, len(rs)),
            "pod_seconds": watcher.pod_seconds(n, t0, t1),
            "lat_p50": float(np.percentile([r["latency"] for r in rs], 50))
                       if rs else None,
        })
    tot_inv = sum(p["invocations"] for p in per_func)
    tot_cold_ev = sum(p["cold_event"] for p in per_func)
    tot_cold_lat = sum(p["cold_latency"] for p in per_func)
    return {
        "arm": arm, "wall_sec": t1 - t0,
        "invocations": tot_inv, "truncated_by_cap": truncated,
        "cold_event": tot_cold_ev, "cold_latency": tot_cold_lat,
        "csr_event": tot_cold_ev / max(1, tot_inv),
        "csr_latency": tot_cold_lat / max(1, tot_inv),
        "classifier_agreement": sum(p["agreement"] * p["invocations"]
                                    for p in per_func) / max(1, tot_inv),
        "pod_seconds": sum(p["pod_seconds"] for p in per_func),
        "per_function": per_func,
        "failed_requests": sum(1 for r in records if not r["ok"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=SEG_MIN)
    ap.add_argument("--funcs", type=int, default=N_FUNCS)
    args = ap.parse_args()
    globals()["SEG_MIN"] = args.minutes

    counts = np.load(PROCESSED_DIR / "counts.npy")
    features = np.load(PROCESSED_DIR / "features.npy")
    splits = np.load(PROCESSED_DIR / "splits.npz")

    # Stratified cohort: onboarding points of s1_test inside the volume band,
    # tiered by ACTIVE TICK COUNT rather than volume. Idle-gap structure, not
    # request count, is what creates cold-start opportunities, so stratifying
    # on it spans the regimes the arms are supposed to differ on.
    pts_all = find_onboarding_points(counts, splits["s1_test"])
    vol = np.array([counts[fi, t0:t0 + SEG_MIN].sum() for (fi, t0) in pts_all])
    act = np.array([(counts[fi, t0:t0 + SEG_MIN] > 0).sum() for (fi, t0) in pts_all])
    band = np.where((vol >= VOL_MIN) & (vol <= VOL_MAX))[0]
    order = band[np.argsort(act[band])]
    rng = np.random.default_rng(COHORT_SEED)
    tiers = np.array_split(order, 3)
    per = [args.funcs // 3] * 3
    for i in range(args.funcs - sum(per)):
        per[i] += 1
    sel = np.concatenate([rng.choice(tiers[i], size=min(per[i], len(tiers[i])),
                                     replace=False) for i in range(3)])
    pts = [pts_all[i] for i in sorted(sel.tolist())]
    print(f"cohort: {len(pts)} of {len(band)} in-band ({len(pts_all)} total) "
          f"onboarding functions")
    print(f"  volumes      {[int(vol[i]) for i in sorted(sel.tolist())]}")
    print(f"  active ticks {[int(act[i]) for i in sorted(sel.tolist())]}")

    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1,
                              device=DEVICE)
    trainer.load(RUNS_DIR / "best_anil_ridge_s1_s0.pt")
    pm = PrototypeManager(n_clusters=N_CLUSTERS, device=DEVICE)
    feats_t = torch.from_numpy(features).float()
    cnts_t = torch.from_numpy(counts).float()
    emb = pm.compute_embeddings(trainer.body, feats_t, splits["s1_train"])
    labels = pm.fit_clusters(emb)
    pm.compute_prototype_heads(trainer.body, trainer.head, feats_t, cnts_t,
                               splits["s1_train"], labels,
                               n_quantiles=N_QUANTILES, n_horizons=1)

    seg, sched, des_dec = build_schedules(counts, features, pts, trainer, pm)

    names = [f"fn{i:02d}" for i in range(len(pts))]
    autoscaler = pin_autoscaler()
    print(f"autoscaler pinned to '{autoscaler}'")
    deploy_services(names)
    pas = [pa_name(n) for n in names]
    assert all(pas), f"missing PodAutoscaler: {pas}"

    watcher = PodWatcher(names)
    watcher.start()
    out = {
        "config": "E5 cohort-scale Knative measurement, onboarding segments",
        "node_ip": NODE_IP, "n_functions": len(pts),
        "func_ids": [int(fi) for fi, _ in pts],
        "onboard_t": [int(t0) for _, t0 in pts],
        "services": names,
        "constants": {"seg_min": SEG_MIN, "tick_sec": TICK_SEC, "rho": RHO,
                      "cap_per_tick": CAP_PER_TICK, "cold_thresh_s": COLD_THRESH,
                      "compression": 1, "cohort_seed": COHORT_SEED,
                      "gate_threshold": GATE_THRESHOLD,
                      "n_clusters": N_CLUSTERS,
                      "autoscaler": autoscaler,
                      "min_scale_via": "PodAutoscaler patch (no revision churn)"},
        "segment_counts": seg.tolist(),
        # what the load generator actually issues after the per-tick cap;
        # testbed_cohort_desmirror.py simulates THESE, so the cap cannot
        # bias the predicted-vs-measured comparison
        "issued_counts": np.minimum(seg, CAP_PER_TICK).tolist(),
        "schedules": {a: sched[a].tolist() for a in ARMS},
        "arms": {},
    }
    try:
        for arm in ARMS:
            print(f"\n--- arm {arm} ({SEG_MIN} min) ---", flush=True)
            res = run_arm(arm, names, pas, seg, sched[arm], watcher)
            out["arms"][arm] = res
            print(f"    CSR(event) {res['csr_event']*100:.2f}% "
                  f"CSR(lat) {res['csr_latency']*100:.2f}% "
                  f"agree {res['classifier_agreement']:.2f} "
                  f"inv {res['invocations']} pod-s {res['pod_seconds']:.0f} "
                  f"failed {res['failed_requests']}", flush=True)
    finally:
        watcher.stop()

    np.save(RUNS_DIR / "testbed_cohort_des_decisions.npy",
            np.array([des_dec[a][0] for a in ARMS]))
    path = RUNS_DIR / "testbed_cohort.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0, "corrupt write (sdb guard)"
    json.load(open(path))
    print(f"\nSaved + verified {path}")


if __name__ == "__main__":
    main()
