# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP7 part 1: Cold/warm latency CDFs on real runtime images (Knative).

Protocol:
- Tune autoscaler for faster scale-to-zero (stable-window 30s, grace 15s).
- Per round: invoke all functions once (cold after scale-to-zero), then
  wait until every service's pods are gone before the next round.
- Warm phase: keep pods alive with continuous requests.
Outputs: results/runs/testbed_cdf.json, tables/T6_testbed_results.csv
"""

import json, subprocess, time, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

NS = "serverless-test"
NODE_IP = "172.18.0.4"
PORT = 31118
FUNCTIONS = ["python-ml", "node-api-real", "java-svc"]
N_COLD_ROUNDS = 20
N_WARM = 50


def sh(cmd, timeout=60):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    return r.stdout.strip(), r.returncode


def invoke(fn, timeout=60):
    host = f"{fn}.{NS}.127.0.0.1.sslip.io"
    out, rc = sh(
        f"curl -s -o /dev/null -w '%{{time_total}}' -H 'Host: {host}' "
        f"http://{NODE_IP}:{PORT} --connect-timeout 30 --max-time 55",
        timeout=timeout)
    if rc == 0 and out:
        try:
            return float(out)
        except ValueError:
            return None
    return None


def pods_for(fn):
    out, _ = sh(f"kubectl get pods -n {NS} -l serving.knative.dev/service={fn} "
                f"--no-headers 2>/dev/null | grep -v Terminating | wc -l")
    try:
        return int(out)
    except ValueError:
        return 0


def wait_scale_to_zero(max_wait=240):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if all(pods_for(fn) == 0 for fn in FUNCTIONS):
            return True
        time.sleep(10)
    return False


def main():
    print("=" * 60)
    print("WP7-1: Cold/Warm CDF Measurement (real runtimes)")
    print("=" * 60)

    # Faster scale-to-zero
    sh("kubectl patch configmap config-autoscaler -n knative-serving "
       "--type merge -p '{\"data\":{\"stable-window\":\"30s\","
       "\"scale-to-zero-grace-period\":\"15s\"}}'")
    print("Autoscaler tuned (stable 30s, grace 15s)")

    cold = {fn: [] for fn in FUNCTIONS}
    warm = {fn: [] for fn in FUNCTIONS}

    # --- Cold rounds ---
    for rnd in range(N_COLD_ROUNDS):
        ok = wait_scale_to_zero()
        if not ok:
            print(f"  round {rnd+1}: scale-to-zero timeout, forcing")
            sh(f"kubectl delete pods -n {NS} --all --force --grace-period=0")
            time.sleep(20)
        for fn in FUNCTIONS:
            lat = invoke(fn, timeout=70)
            if lat is not None:
                cold[fn].append(lat)
        done = {fn: len(v) for fn, v in cold.items()}
        print(f"  cold round {rnd+1}/{N_COLD_ROUNDS}: {done}", flush=True)

    # --- Warm phase ---
    print("Warm phase...")
    for fn in FUNCTIONS:
        invoke(fn)  # ensure warm
    time.sleep(2)
    for i in range(N_WARM):
        for fn in FUNCTIONS:
            lat = invoke(fn, timeout=30)
            if lat is not None:
                warm[fn].append(lat)
        time.sleep(0.5)  # keep pods warm, don't overload
        if (i + 1) % 10 == 0:
            print(f"  warm {i+1}/{N_WARM}", flush=True)

    # --- Save ---
    raw = {"cold": cold, "warm": warm,
           "protocol": {"cold_rounds": N_COLD_ROUNDS, "warm_reqs": N_WARM,
                        "autoscaler": "stable 30s, grace 15s",
                        "runtimes": {"python-ml": "python3.11+numpy/pandas",
                                     "node-api-real": "node20",
                                     "java-svc": "temurin17 JVM"}}}
    with open(RUNS_DIR / "testbed_cdf.json", "w") as f:
        json.dump(raw, f, indent=2)

    rows = []
    for fn in FUNCTIONS:
        c, w = np.array(cold[fn]), np.array(warm[fn])
        rows.append({
            "function": fn,
            "cold_n": len(c), "warm_n": len(w),
            "cold_p50_s": float(np.percentile(c, 50)) if len(c) else 0,
            "cold_p95_s": float(np.percentile(c, 95)) if len(c) else 0,
            "cold_p99_s": float(np.percentile(c, 99)) if len(c) else 0,
            "warm_p50_s": float(np.percentile(w, 50)) if len(w) else 0,
            "warm_p95_s": float(np.percentile(w, 95)) if len(w) else 0,
            "cold_overhead_p50_s": (float(np.percentile(c, 50) - np.percentile(w, 50))
                                    if len(c) and len(w) else 0),
            # lognormal fit for simulator calibration
            "lognorm_mu": float(np.mean(np.log(np.maximum(c, 1e-3)))) if len(c) else 0,
            "lognorm_sigma": float(np.std(np.log(np.maximum(c, 1e-3)))) if len(c) else 0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "T6_testbed_results.csv", index=False)
    print(df.to_string(index=False))
    print("\nWP7-1 COMPLETE")


if __name__ == "__main__":
    main()
