#!/usr/bin/env python3
"""R3: natural deployment-rate estimate from onset timestamps
(pre-registered in revision_r3_PREREG.md). Counting only — no model.

Azure 2019: onset grid over the nine registered e2 definitions + a4 rule,
validated against revision_e2_cohort_sweep.json funnels.
Huawei 2023: deployment rate from first_present.npy.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
DAY = 1440

counts = np.load(ROOT / "data" / "processed_2019" / "counts.npy", mmap_mode="r")
N, T = counts.shape
print(f"2019 universe: {N} functions x {T} min", flush=True)

# one pass: first nonzero tick and day-1 activity per function
t0s = np.full(N, -1, dtype=np.int64)
day1 = np.zeros(N, dtype=np.int64)
CH = 2000
for s in range(0, N, CH):
    block = np.asarray(counts[s:s + CH])
    nz = block != 0
    has = nz.any(axis=1)
    first = np.where(has, nz.argmax(axis=1), -1)
    t0s[s:s + CH] = first
    for i in np.flatnonzero(has):
        t0 = first[i]
        day1[s + i] = block[i, t0:t0 + DAY].sum()
    if (s // CH) % 5 == 0:
        print(f"  scan {s}/{N}", flush=True)

valid = t0s >= 0
out = {"prereg": "revision_r3_PREREG.md",
       "azure2019": {"n_universe": int(N), "horizon_min": int(T),
                     "ever_invoked": int(valid.sum()), "definitions": {}},
       "gates": {}}

defs = [(g, m) for g in (1, 3, 7) for m in (10, 30, 100)]
for g, m in defs:
    sel = valid & (t0s >= g * DAY) & (t0s < T - DAY) & (day1 >= m)
    observable_days = (T - DAY - g * DAY) / DAY
    n = int(sel.sum())
    out["azure2019"]["definitions"][f"gap{g}d_thr{m}"] = {
        "n_onsets": n,
        "observable_days": observable_days,
        "onsets_per_day": n / observable_days,
        "onsets_per_day_per_1k_universe": n / observable_days / (N / 1000),
        "onset_day_histogram": np.bincount(
            (t0s[sel] // DAY).astype(int), minlength=14).tolist(),
    }

# validation gate vs e2 funnels (idle_prefix_ok / day1_threshold_ok)
e2 = json.load(open(RUNS / "revision_e2_cohort_sweep.json"))
gate_fail = 0
for g, m in defs:
    cell = e2["cells"][f"gap{g}d_thr{m}"]["funnel"]
    ours_idle = int((valid & (t0s >= g * DAY) & (t0s < T - DAY)).sum())
    ours_day1 = out["azure2019"]["definitions"][f"gap{g}d_thr{m}"]["n_onsets"]
    ok = (ours_idle == cell["idle_prefix_ok"]) and (ours_day1 == cell["day1_threshold_ok"])
    gate_fail += 0 if ok else 1
    out["gates"][f"gap{g}d_thr{m}"] = {
        "ours": [ours_idle, ours_day1],
        "e2_funnel": [cell["idle_prefix_ok"], cell["day1_threshold_ok"]],
        "match": bool(ok)}
print(f"e2 funnel gate: {9 - gate_fail}/9 match", flush=True)

# Huawei from first_present
fp = np.load(ROOT / "data" / "processed_huawei" / "first_present.npy")
meta = json.load(open(ROOT / "data" / "processed_huawei" / "meta.json"))
T_hw = int(meta.get("T_full", 37440))
dep = fp[(fp > 0) & (fp < T_hw - DAY)]
out["huawei"] = {
    "n_universe": int(len(fp)),
    "horizon_min": T_hw,
    "present_from_start": int((fp == 0).sum()),
    "n_deployments": int(len(dep)),
    "observable_days": (T_hw - DAY) / DAY,
    "deployments_per_day": float(len(dep) / ((T_hw - DAY) / DAY)),
    "deployments_per_day_per_1k_universe": float(
        len(dep) / ((T_hw - DAY) / DAY) / (len(fp) / 1000)),
    "deploy_day_histogram": np.bincount(
        (dep // DAY).astype(int), minlength=T_hw // DAY).tolist(),
}

path = RUNS / "revision_r3_deployrate.json"
json.dump(out, open(path, "w"), indent=1)
a4rule = out["azure2019"]["definitions"]["gap3d_thr30"]
print(f"a4 rule (3d,30): {a4rule['n_onsets']} onsets, "
      f"{a4rule['onsets_per_day']:.1f}/day, "
      f"{a4rule['onsets_per_day_per_1k_universe']:.2f}/day/1k")
print(f"huawei: {out['huawei']['n_deployments']} deployments, "
      f"{out['huawei']['deployments_per_day']:.1f}/day, "
      f"{out['huawei']['deployments_per_day_per_1k_universe']:.2f}/day/1k")
print(f"wrote {path}")
