# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP7 part 2: End-to-end transfer validation on Knative.

Replays a bursty trace segment against the python-ml service under two arms:
  A) reactive  — Knative default scale-to-zero (min-scale 0 throughout)
  B) A5-prewarm — per-tick min-scale set to A5's prewarm decision (rho=10),
     applied one tick ahead (init lead time)
Measures per-invocation latency (cold = > COLD_THRESH) and integrates pod
count for container-seconds. Validates that the simulated CSR/memory
direction transfers to a real platform.
"""

import json, subprocess, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import rates_a5, decisions_from_rates
from src.decision.newsvendor import newsvendor_quantile
from src.models.heads import N_QUANTILES
from src.meta.trainer import ANILMetaTrainer

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NS = "serverless-test"
SVC = "python-ml"
NODE_IP = "172.18.0.4"
PORT = 31118
COMPRESS = 3            # 1 trace-min = 20 s wall clock
TICK_SEC = 60.0 / COMPRESS
SEG_LEN = 45            # trace minutes
COLD_THRESH = 0.7       # s; python-ml warm ~5-50 ms, cold ~2 s
RHO = 10.0


def sh(cmd, timeout=60):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    return r.stdout.strip(), r.returncode


def invoke():
    host = f"{SVC}.{NS}.127.0.0.1.sslip.io"
    out, rc = sh(f"curl -s -o /dev/null -w '%{{time_total}}' -H 'Host: {host}' "
                 f"http://{NODE_IP}:{PORT} --connect-timeout 20 --max-time 40",
                 timeout=50)
    try:
        return float(out) if rc == 0 else None
    except ValueError:
        return None


def set_min_scale(n):
    sh(f"kubectl patch ksvc {SVC} -n {NS} --type merge -p "
       f"'{{\"spec\":{{\"template\":{{\"metadata\":{{\"annotations\":{{"
       f"\"autoscaling.knative.dev/min-scale\":\"{int(n)}\"}}}}}}}}}}'",
       timeout=30)


def pod_count():
    out, _ = sh(f"kubectl get pods -n {NS} -l serving.knative.dev/service={SVC} "
                f"--no-headers 2>/dev/null | grep -c Running || true")
    try:
        return int(out)
    except ValueError:
        return 0


def pick_segment(counts, test_idx):
    """Bursty segment: activity episodes separated by >=5-min gaps."""
    best = None
    for fi in test_idx:
        row = counts[fi]
        for s in range(1440, len(row) - SEG_LEN, 60):
            seg = row[s:s + SEG_LEN]
            tot = seg.sum()
            if not (10 <= tot <= 80):
                continue
            act = seg > 0
            # count bursts (runs of activity) and max gap
            edges = np.flatnonzero(np.diff(np.concatenate(([0], act.view(np.int8), [0]))))
            n_bursts = len(edges) // 2
            if n_bursts >= 3:
                gaps = []
                for i in range(1, n_bursts):
                    gaps.append(edges[2 * i] - edges[2 * i - 1])
                if gaps and max(gaps) >= 5:
                    score = n_bursts + max(gaps) / 10
                    if best is None or score > best[0]:
                        best = (score, int(fi), int(s))
    return best[1], best[2]


def replay(seg, prewarm=None):
    """Replay one arm. prewarm: [SEG_LEN] min-scale schedule or None."""
    lat = []
    pods_integral = 0.0
    last_sample = time.time()

    set_min_scale(0)
    # ensure scaled to zero before starting
    print("    waiting for scale-to-zero...")
    for _ in range(30):
        if pod_count() == 0:
            break
        time.sleep(8)

    t_start = time.time()
    for t in range(SEG_LEN):
        tick_t0 = time.time()
        # apply next tick's prewarm one tick ahead
        if prewarm is not None and t + 1 < SEG_LEN:
            set_min_scale(min(int(prewarm[t + 1]), 3))
        # sample pods & integrate
        now = time.time()
        pods_integral += pod_count() * (now - last_sample)
        last_sample = now

        n = int(seg[t])
        for _ in range(min(n, 5)):        # cap per-tick invocations
            l = invoke()
            if l is not None:
                lat.append(l)
        # wait out the tick
        rem = TICK_SEC - (time.time() - tick_t0)
        if rem > 0:
            time.sleep(rem)
    now = time.time()
    pods_integral += pod_count() * (now - last_sample)
    set_min_scale(0)

    lat = np.array(lat)
    cold = int((lat > COLD_THRESH).sum())
    return {
        "invocations": len(lat),
        "cold": cold,
        "csr": cold / max(1, len(lat)),
        "pod_seconds": pods_integral,
        "lat_p50": float(np.percentile(lat, 50)) if len(lat) else 0,
        "lat_p95": float(np.percentile(lat, 95)) if len(lat) else 0,
        "wall_sec": time.time() - t_start,
    }


def main():
    print("=" * 60)
    print("WP7-2: End-to-End Transfer Validation (Knative)")
    print("=" * 60)

    counts = np.load(PROCESSED_DIR / "counts.npy")
    features = np.load(PROCESSED_DIR / "features.npy")
    splits_data = np.load(PROCESSED_DIR / "splits.npz")
    test_idx = splits_data["s1_test"]

    fi, s = pick_segment(counts, test_idx)
    seg = counts[fi, s:s + SEG_LEN]
    print(f"Segment: func {fi}, ticks [{s}, {s+SEG_LEN}), "
          f"{seg.sum()} invocations, pattern={seg.tolist()}")

    # A5 prewarm schedule (offline, honest online protocol up to segment)
    print("Computing A5 prewarm schedule...")
    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1,
                              device=DEVICE)
    trainer.load(RUNS_DIR / "best_anil_ridge_s1_s0.pt")
    hist_start = max(0, s - 2880)
    rates = rates_a5(counts[fi:fi + 1, hist_start:s + SEG_LEN],
                     features[fi:fi + 1, hist_start:s + SEG_LEN],
                     trainer, DEVICE)
    tau = newsvendor_quantile(RHO)
    pw, _ = decisions_from_rates(rates[:, -SEG_LEN:], tau)
    schedule = pw[0]
    print(f"A5 prewarm schedule: {schedule.tolist()}")

    print(f"\n--- Arm A: reactive (Knative default) [{SEG_LEN} ticks x "
          f"{TICK_SEC:.0f}s] ---")
    arm_a = replay(seg, prewarm=None)
    print(f"  {arm_a}")

    print(f"\n--- Arm B: A5 prewarm schedule ---")
    arm_b = replay(seg, prewarm=schedule)
    print(f"  {arm_b}")

    out = {
        "function_idx": int(fi), "segment_start": int(s),
        "segment_counts": seg.tolist(),
        "a5_schedule": schedule.tolist(),
        "compress": COMPRESS, "rho": RHO, "cold_thresh_s": COLD_THRESH,
        "reactive": arm_a, "a5_prewarm": arm_b,
    }
    with open(RUNS_DIR / "testbed_e2e.json", "w") as f:
        json.dump(out, f, indent=2)

    df = pd.DataFrame([
        {"arm": "reactive (default)", **{k: v for k, v in arm_a.items()}},
        {"arm": "A5 prewarm", **{k: v for k, v in arm_b.items()}},
    ])
    df.to_csv(TABLES_DIR / "T6b_testbed_e2e.csv", index=False)
    print("\n" + df.to_string(index=False))

    d_csr = arm_a["csr"] - arm_b["csr"]
    print(f"\nCSR reduction (reactive -> A5): {d_csr:+.4f} "
          f"({'direction matches simulation' if d_csr > 0 else 'no reduction'})")
    print("WP7-2 COMPLETE")


if __name__ == "__main__":
    main()
