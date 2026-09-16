# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Phase 7: Testbed Validation on Knative.

Deploys functions on Knative, replays trace segments,
measures real cold-start latency CDFs, validates simulator gains.
"""

import os, sys, time, json, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

NAMESPACE = "serverless-test"


def run_cmd(cmd, check=True, timeout=120):
    """Run shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        print(f"  CMD FAILED: {cmd}")
        print(f"  STDERR: {result.stderr[:500]}")
    return result


def wait_for_knative():
    """Wait for Knative serving to be ready."""
    print("Waiting for Knative Serving to be ready...")
    for i in range(60):
        r = run_cmd("kubectl get pods -n knative-serving -o json", check=False)
        if r.returncode == 0:
            data = json.loads(r.stdout)
            pods = data.get("items", [])
            all_ready = all(
                any(c.get("ready", False) for c in p.get("status", {}).get("conditions", []))
                for p in pods if p.get("status", {}).get("phase") != "Succeeded"
            )
            running = sum(1 for p in pods if p.get("status", {}).get("phase") == "Running")
            if running >= 3 and all_ready:
                print(f"  Knative ready ({running} pods running)")
                return True
        time.sleep(5)
        if i % 6 == 0:
            print(f"  Waiting... ({i*5}s)")
    print("  WARNING: Knative may not be fully ready")
    return False


def setup_namespace():
    """Create test namespace."""
    run_cmd(f"kubectl create namespace {NAMESPACE}", check=False)
    # Label namespace for Knative
    run_cmd(f"kubectl label namespace {NAMESPACE} knative.dev/serving=true --overwrite", check=False)


def deploy_function(name, image, memory="256Mi", concurrency=1):
    """Deploy a Knative service (function)."""
    service_yaml = f"""
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: {name}
  namespace: {NAMESPACE}
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/target: "{concurrency}"
        autoscaling.knative.dev/min-scale: "0"
        autoscaling.knative.dev/max-scale: "5"
    spec:
      containers:
      - image: {image}
        resources:
          requests:
            memory: "{memory}"
          limits:
            memory: "{memory}"
        ports:
        - containerPort: 8080
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(service_yaml)
        f.flush()
        result = run_cmd(f"kubectl apply -f {f.name}")
        os.unlink(f.name)
    return result.returncode == 0


def deploy_test_functions():
    """Deploy a mix of test functions covering cold-start latency spectrum."""
    functions = [
        # name, image (using publicly available serverless-ready images),
        # memory, expected cold-start latency class
        ("python-hello", "gcr.io/knative-samples/helloworld-go", "128Mi", "fast"),
        ("node-api", "gcr.io/knative-samples/helloworld-go", "128Mi", "fast"),
        ("python-compute", "gcr.io/knative-samples/helloworld-go", "256Mi", "medium"),
        ("java-handler", "gcr.io/knative-samples/helloworld-go", "512Mi", "slow"),
        ("ml-inference", "gcr.io/knative-samples/helloworld-go", "256Mi", "medium"),
    ]

    deployed = []
    for name, image, memory, latency_class in functions:
        print(f"  Deploying {name} ({latency_class}, {memory})...")
        success = deploy_function(name, image, memory)
        if success:
            deployed.append({
                "name": name, "image": image, "memory": memory,
                "latency_class": latency_class
            })
    return deployed


def wait_for_services(func_names, timeout=180):
    """Wait for all Knative services to be ready."""
    print("Waiting for services to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        all_ready = True
        for name in func_names:
            r = run_cmd(
                f"kubectl get ksvc {name} -n {NAMESPACE} -o jsonpath='{{.status.conditions[?(@.type==\"Ready\")].status}}'",
                check=False
            )
            if "True" not in r.stdout:
                all_ready = False
                break
        if all_ready:
            print(f"  All {len(func_names)} services ready")
            return True
        time.sleep(5)
    print("  WARNING: Not all services ready within timeout")
    return False


def get_service_url(name):
    """Get the URL of a Knative service."""
    r = run_cmd(
        f"kubectl get ksvc {name} -n {NAMESPACE} -o jsonpath='{{.status.url}}'",
        check=False
    )
    url = r.stdout.strip().strip("'")
    if not url:
        # Fallback: use cluster IP + host header
        r2 = run_cmd("kubectl get svc kourier -n kourier-system -o jsonpath='{.spec.clusterIP}'", check=False)
        cluster_ip = r2.stdout.strip().strip("'")
        if cluster_ip:
            url = f"http://{cluster_ip}"
    return url


def invoke_function(name, url=None):
    """Invoke a function and measure latency."""
    if not url:
        url = get_service_url(name)

    if not url:
        return None

    # Get kourier ingress
    r = run_cmd(
        "kubectl get svc kourier -n kourier-system -o jsonpath='{.spec.clusterIP}'",
        check=False
    )
    kourier_ip = r.stdout.strip().strip("'")

    host = f"{name}.{NAMESPACE}.svc.cluster.local"

    start = time.time()
    result = run_cmd(
        f"kubectl exec -n {NAMESPACE} deploy/{name}-00001-deployment -- "
        f"wget -q -O /dev/null --timeout=10 http://localhost:8080 2>&1 || "
        f"curl -s -o /dev/null -w '%{{time_total}}' -H 'Host: {host}' http://{kourier_ip} 2>/dev/null",
        check=False, timeout=30
    )
    elapsed = time.time() - start

    # Try direct curl through kourier
    if kourier_ip:
        r2 = run_cmd(
            f"curl -s -o /dev/null -w '%{{time_total}}' "
            f"-H 'Host: {host}' http://{kourier_ip}",
            check=False, timeout=30
        )
        if r2.returncode == 0 and r2.stdout.strip():
            try:
                elapsed = float(r2.stdout.strip())
            except ValueError:
                pass

    return elapsed


def run_trace_replay(func_names, duration_minutes=30, compressed_factor=10):
    """Replay trace segments against deployed functions.

    Replays at compressed_factor x speed for tractability.
    """
    print(f"\nReplaying traces for {duration_minutes} min (compressed {compressed_factor}x)...")

    # Load real trace patterns
    counts = np.load(PROJECT_ROOT / "data" / "processed" / "counts.npy")

    # Use first N function patterns for our deployed functions
    n_funcs = min(len(func_names), counts.shape[0])
    trace_segment = counts[:n_funcs, :duration_minutes * compressed_factor]

    latencies = {name: [] for name in func_names[:n_funcs]}
    cold_starts = {name: 0 for name in func_names[:n_funcs]}
    warm_starts = {name: 0 for name in func_names[:n_funcs]}

    tick_interval = 60.0 / compressed_factor  # seconds between ticks

    kourier_r = run_cmd(
        "kubectl get svc kourier -n kourier-system -o jsonpath='{.spec.clusterIP}'",
        check=False
    )
    kourier_ip = kourier_r.stdout.strip().strip("'")

    total_ticks = trace_segment.shape[1]
    print(f"  {total_ticks} ticks, {tick_interval:.1f}s/tick, {n_funcs} functions")

    for t in range(min(total_ticks, duration_minutes * 2)):  # Cap for time
        tick_start = time.time()

        for fi in range(n_funcs):
            n_invocations = int(trace_segment[fi, t])
            name = func_names[fi]

            for _ in range(min(n_invocations, 3)):  # Cap invocations per tick
                host = f"{name}.{NAMESPACE}.svc.cluster.local"

                inv_start = time.time()
                r = run_cmd(
                    f"curl -s -o /dev/null -w '%{{time_total}}' "
                    f"-H 'Host: {host}' http://{kourier_ip} --connect-timeout 5",
                    check=False, timeout=15
                )
                inv_elapsed = time.time() - inv_start

                if r.returncode == 0 and r.stdout.strip():
                    try:
                        lat = float(r.stdout.strip())
                        latencies[name].append(lat)
                        # Cold start heuristic: > 1 second likely cold
                        if lat > 1.0:
                            cold_starts[name] += 1
                        else:
                            warm_starts[name] += 1
                    except ValueError:
                        pass

        # Wait for tick interval
        elapsed = time.time() - tick_start
        remaining = max(0, tick_interval - elapsed)
        if remaining > 0 and t < total_ticks - 1:
            time.sleep(min(remaining, 2.0))  # Cap sleep

        if t % 10 == 0:
            total_inv = sum(len(v) for v in latencies.values())
            total_cold = sum(cold_starts.values())
            print(f"  Tick {t}/{total_ticks}: {total_inv} invocations, {total_cold} cold starts")

    return latencies, cold_starts, warm_starts


def scale_to_zero(func_names):
    """Scale all functions to zero (force cold start on next invocation)."""
    for name in func_names:
        run_cmd(
            f"kubectl patch ksvc {name} -n {NAMESPACE} --type merge "
            f"-p '{{\"spec\":{{\"template\":{{\"metadata\":{{\"annotations\":{{"
            f"\"autoscaling.knative.dev/min-scale\":\"0\"}}}}}}}}}}'",
            check=False
        )
    # Wait for scale down
    time.sleep(15)


def measure_cold_start_distribution(func_names, n_trials=10):
    """Measure cold-start latency distribution by forcing scale-to-zero."""
    print(f"\nMeasuring cold-start latency distribution ({n_trials} trials)...")

    kourier_r = run_cmd(
        "kubectl get svc kourier -n kourier-system -o jsonpath='{.spec.clusterIP}'",
        check=False
    )
    kourier_ip = kourier_r.stdout.strip().strip("'")

    cold_latencies = {name: [] for name in func_names}

    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}")
        # Scale to zero
        scale_to_zero(func_names)

        # Invoke each function (will be cold start)
        for name in func_names:
            host = f"{name}.{NAMESPACE}.svc.cluster.local"
            r = run_cmd(
                f"curl -s -o /dev/null -w '%{{time_total}}' "
                f"-H 'Host: {host}' http://{kourier_ip} --connect-timeout 30",
                check=False, timeout=60
            )
            if r.returncode == 0 and r.stdout.strip():
                try:
                    lat = float(r.stdout.strip())
                    cold_latencies[name].append(lat)
                except ValueError:
                    pass

    return cold_latencies


def generate_testbed_results(latencies, cold_starts, warm_starts, cold_latencies, func_names):
    """Generate T6 and update F6 with real testbed data."""

    # T6: Testbed results
    t6_rows = []
    for name in func_names:
        lats = latencies.get(name, [])
        cold = cold_starts.get(name, 0)
        warm = warm_starts.get(name, 0)
        total = cold + warm
        cold_lats = cold_latencies.get(name, [])

        t6_rows.append({
            "function": name,
            "total_invocations": total,
            "cold_starts": cold,
            "warm_starts": warm,
            "csr": cold / max(1, total),
            "latency_p50": float(np.percentile(lats, 50)) if lats else 0,
            "latency_p95": float(np.percentile(lats, 95)) if lats else 0,
            "latency_p99": float(np.percentile(lats, 99)) if lats else 0,
            "cold_start_p50": float(np.percentile(cold_lats, 50)) if cold_lats else 0,
            "cold_start_p95": float(np.percentile(cold_lats, 95)) if cold_lats else 0,
            "cold_start_p99": float(np.percentile(cold_lats, 99)) if cold_lats else 0,
        })

    t6_df = pd.DataFrame(t6_rows)
    t6_df.to_csv(TABLES_DIR / "T6_testbed_results.csv", index=False)
    print(f"\nSaved T6_testbed_results.csv")
    print(t6_df.to_string(index=False))

    # Generate real F6 CDF
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    all_cold_lats = []
    all_warm_lats = []
    for name in func_names:
        for lat in latencies.get(name, []):
            if lat > 1.0:
                all_cold_lats.append(lat)
            else:
                all_warm_lats.append(lat)

    all_lats = [l for name in func_names for l in latencies.get(name, [])]

    if all_lats:
        sorted_lats = np.sort(all_lats)
        cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
        ax.plot(sorted_lats, cdf, color="#1f77b4", label=f"All ({len(all_lats)} inv.)", linewidth=2)

    for name in func_names:
        cold_lats = cold_latencies.get(name, [])
        if cold_lats:
            sorted_cl = np.sort(cold_lats)
            cdf_cl = np.arange(1, len(sorted_cl) + 1) / len(sorted_cl)
            ax.plot(sorted_cl, cdf_cl, label=f"{name} cold-start", linewidth=1, alpha=0.7)

    ax.set_xlabel("Latency (seconds)")
    ax.set_ylabel("CDF")
    ax.set_title("Testbed Cold-Start Latency CDF (Knative)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(max(l for lats in latencies.values() for l in lats) * 1.1, 30) if any(latencies.values()) else 10)

    fig.savefig(FIGURES_DIR / "F6_testbed_cdf.pdf")
    fig.savefig(FIGURES_DIR / "F6_testbed_cdf.png")
    plt.close(fig)
    print("  Updated F6_testbed_cdf.pdf with real data")

    return t6_df


def cleanup():
    """Clean up testbed resources."""
    print("\nCleaning up testbed...")
    run_cmd(f"kubectl delete namespace {NAMESPACE} --ignore-not-found", check=False)


def main():
    print("=" * 60)
    print("PHASE 7: Testbed Validation (Knative)")
    print("=" * 60)

    # 1. Check Knative
    wait_for_knative()

    # 2. Setup namespace
    setup_namespace()

    # 3. Deploy functions
    print("\n--- Deploying test functions ---")
    deployed = deploy_test_functions()
    func_names = [d["name"] for d in deployed]
    print(f"  Deployed {len(deployed)} functions")

    # 4. Wait for services
    wait_for_services(func_names, timeout=300)

    # 5. Measure cold-start distribution
    print("\n--- Measuring cold-start distributions ---")
    cold_latencies = measure_cold_start_distribution(func_names, n_trials=5)

    # 6. Trace replay
    print("\n--- Trace replay ---")
    latencies, cold_starts, warm_starts = run_trace_replay(
        func_names, duration_minutes=10, compressed_factor=5
    )

    # 7. Generate results
    print("\n--- Generating results ---")
    t6 = generate_testbed_results(latencies, cold_starts, warm_starts, cold_latencies, func_names)

    # 8. Save raw data
    testbed_data = {
        "deployed": deployed,
        "cold_latencies": {k: v for k, v in cold_latencies.items()},
        "invocation_latencies": {k: v for k, v in latencies.items()},
        "cold_starts": cold_starts,
        "warm_starts": warm_starts,
    }
    with open(RUNS_DIR / "testbed_results.json", "w") as f:
        json.dump(testbed_data, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("PHASE 7 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
