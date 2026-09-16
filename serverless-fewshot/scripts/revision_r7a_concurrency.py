# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R7a: per-sandbox concurrency sensitivity (pre-registered in
revision_r7a_PREREG.md). DES env only:

  PYTHONPATH=/data/260715/site-packages-des:. python3.13 \
      scripts/revision_r7a_concurrency.py --validate
  PYTHONPATH=/data/260715/site-packages-des:. python3.13 \
      scripts/revision_r7a_concurrency.py --run

Reads the ARCHIVED decision surfaces; the only transform is
sandbox_target = ceil(prewarm / CC).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.des_fast import simulate_trace_fast  # noqa: E402
from src.sim.des_fast_cc import simulate_trace_cc  # noqa: E402

RUNS = PROJECT_ROOT / "results" / "runs"
POOLS = {
    "2019_S2": RUNS / "des_jobs_2019" / "S2",
    "2019_S3": RUNS / "des_jobs_2019" / "S3",
    "h_saturated": RUNS / "des_jobs_huawei" / "h_saturated",
}
ARMS = ["A5_full_system", "B4a_ewma", "Oracle"]
RHOS = [1.0, 10.0]
CCS = [1, 4, 10]
OUT = RUNS / "revision_r7a_concurrency.json"


def load_pool(jdir):
    shared = np.load(jdir / "shared.npz")
    cold_init = json.load(open(jdir / "jobs.json"))["cold_init"]
    return (shared["counts"], shared["dur_means"], shared["dur_stds"],
            cold_init)


def load_dec(jdir, method, rho):
    z = np.load(jdir / f"{method}__rho{rho}.npz")
    return z["prewarm"], z["keepalive"]


def validate():
    print("validation gate: des_fast_cc(cc=1) vs des_fast, B4a rho=10 seed 0",
          flush=True)
    gate = {}
    for pool, jdir in POOLS.items():
        counts, dm, ds, ci = load_pool(jdir)
        pw, ka = load_dec(jdir, "B4a_ewma", 10.0)
        t0 = time.time()
        a = simulate_trace_fast(counts, pw, ka, dm, ds, seed=0,
                                cold_mu=ci["mu"], cold_sigma=ci["sigma"])
        b = simulate_trace_cc(counts, pw, ka, dm, ds, seed=0,
                              cold_mu=ci["mu"], cold_sigma=ci["sigma"], cc=1)
        ok = (a["total_invocations"] == b["total_invocations"]
              and a["cold_starts"] == b["cold_starts"]
              and abs(a["wm_total_gb_s"] - b["wm_total_gb_s"])
              <= 1e-9 * max(1.0, a["wm_total_gb_s"])
              and abs(a["wm_fraction"] - b["wm_fraction"]) <= 1e-9)
        gate[pool] = {"ok": bool(ok),
                      "fast": [a["total_invocations"], a["cold_starts"],
                               a["wm_total_gb_s"]],
                      "cc1": [b["total_invocations"], b["cold_starts"],
                              b["wm_total_gb_s"]],
                      "sec": round(time.time() - t0, 1)}
        print(f"  {pool}: {'OK' if ok else 'FAIL'} "
              f"cold={a['cold_starts']} vs {b['cold_starts']} "
              f"({gate[pool]['sec']}s)", flush=True)
    out = json.load(open(OUT)) if OUT.exists() else {
        "prereg": "revision_r7a_PREREG.md"}
    out["validation_gate"] = gate
    json.dump(out, open(OUT, "w"), indent=1)
    if not all(g["ok"] for g in gate.values()):
        sys.exit("VALIDATION GATE FAILED — do not run the campaign")
    print("gate passed")


def run():
    out = json.load(open(OUT))
    assert all(g["ok"] for g in out.get("validation_gate", {}).values()), \
        "run --validate first (gate must pass)"
    out.setdefault("results", [])
    done = {(r["pool"], r["method"], r["rho"], r["cc"], r["seed"])
            for r in out["results"]}
    for pool, jdir in POOLS.items():
        counts, dm, ds, ci = load_pool(jdir)
        for method in ARMS:
            for rho in RHOS:
                pw, ka = load_dec(jdir, method, rho)
                for cc in CCS:
                    pw_cc = np.ceil(pw / cc).astype(np.int64) if cc > 1 else pw
                    seeds = [0, 1, 2] if cc == 1 else [0]
                    for seed in seeds:
                        key = (pool, method, rho, cc, seed)
                        if key in done:
                            continue
                        t0 = time.time()
                        r = simulate_trace_cc(
                            counts, pw_cc, ka, dm, ds, seed=seed,
                            cold_mu=ci["mu"], cold_sigma=ci["sigma"], cc=cc)
                        r.pop("func_cold"); r.pop("func_total")
                        r.pop("func_wm")
                        r.update({"pool": pool, "method": method, "rho": rho,
                                  "cc": cc, "seed": seed,
                                  "elapsed_sec": round(time.time() - t0, 1)})
                        out["results"].append(r)
                        json.dump(out, open(OUT, "w"), indent=1)
                        print(f"{pool} {method} rho={rho} cc={cc} seed={seed}: "
                              f"csr={r['csr']*100:.4f}% wm/1k={r['wm_per_1k_inv']:.0f} "
                              f"({r['elapsed_sec']}s)", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.validate:
        validate()
    elif a.run:
        run()
    else:
        ap.error("pass --validate or --run")
