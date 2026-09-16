#!/usr/bin/env python3
"""R6 stage 1: mine natural collapse events (pre-registered in
revision_r6_PREREG.md — detector unchanged, selection by the registered
post-filters). Torch env (ruptures lives there); CPU only.

Output: results/runs/revision_r6_catalog.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.drift.detector import detect_natural_drift  # noqa: E402

RUNS = PROJECT_ROOT / "results" / "runs"
DAY = 1440
W = 240
MAG_LO, MAG_HI = 0.05, 0.30
PRE_MIN, POST_MIN = 100, 10


def select_events(counts, rows, t_lo=None, t_hi=None):
    """Run the detector on counts[rows] and apply the registered filters.
    t_lo/t_hi: optional per-function cp bounds (arrays aligned with rows)."""
    N, T = len(rows), counts.shape[1]
    funnel = {"n_scanned": N, "n_cp_raw": 0, "n_level": 0, "n_filtered": 0}
    block = np.asarray(counts[rows])
    catalog = detect_natural_drift(block, np.asarray(rows).astype(str))
    funnel["n_cp_raw"] = len(catalog)
    best = {}
    for ev in catalog:
        i = ev["func_idx"]                 # index within block
        cp = ev["change_point_minute"]
        if ev["shift_type"] != "level":
            continue
        funnel["n_level"] += 1
        if not (MAG_LO <= ev["magnitude"] <= MAG_HI):
            continue
        lo = DAY if t_lo is None else max(DAY, int(t_lo[i]) + DAY)
        hi = T - DAY if t_hi is None else min(T - DAY, int(t_hi[i]) - DAY)
        if not (lo <= cp < hi):
            continue
        series = block[i]
        pre = float(series[cp - W:cp].sum())
        post = float(series[cp:cp + W].sum())
        if pre < PRE_MIN or post < POST_MIN:
            continue
        funnel["n_filtered"] += 1
        rec = {"row": int(rows[i]), "cp_minute": int(cp),
               "magnitude": ev["magnitude"], "pre_inv_240": pre,
               "post_inv_240": post, "pre_mean": ev["pre_mean"],
               "post_mean": ev["post_mean"]}
        if i not in best or pre > best[i]["pre_inv_240"]:
            best[i] = rec
    events = sorted(best.values(), key=lambda r: -r["pre_inv_240"])
    return events, funnel


def main():
    out = {"prereg": "revision_r6_PREREG.md",
           "filters": {"magnitude": [MAG_LO, MAG_HI], "cp_margin_min": DAY,
                       "pre_inv_240_min": PRE_MIN, "post_inv_240_min": POST_MIN,
                       "one_event_per_function": True},
           "traces": {}}

    # Azure 2021
    t0 = time.time()
    c21 = np.load(PROJECT_ROOT / "data" / "processed" / "counts.npy")
    ev, fn = select_events(c21, np.arange(c21.shape[0]))
    horizon_days = c21.shape[1] / DAY
    out["traces"]["azure2021"] = {
        "events": ev, "funnel": fn, "n_functions_scanned": fn["n_scanned"],
        "events_per_func_per_14d": len(ev) / fn["n_scanned"] / (horizon_days / 14)}
    print(f"azure2021: {fn} -> {len(ev)} events ({time.time()-t0:.0f}s)", flush=True)

    # Azure 2019 (DES sample rows)
    t0 = time.time()
    man = json.load(open(RUNS / "des_sample_manifest.json"))
    rows19 = sorted({r for s in man.values() for r in s["rows"]})
    c19 = np.load(PROJECT_ROOT / "data" / "processed_2019" / "counts.npy",
                  mmap_mode="r")
    ev, fn = select_events(c19, np.asarray(rows19))
    out["traces"]["azure2019"] = {
        "events": ev, "funnel": fn, "n_functions_scanned": fn["n_scanned"],
        "events_per_func_per_14d": len(ev) / fn["n_scanned"] / (c19.shape[1] / DAY / 14)}
    print(f"azure2019: {fn} -> {len(ev)} events ({time.time()-t0:.0f}s)", flush=True)

    # Huawei (active pool union, deployment-masked)
    t0 = time.time()
    P_HW = PROJECT_ROOT / "data" / "processed_huawei"
    spl = np.load(P_HW / "splits.npz")
    rows_hw = sorted(set(np.concatenate(
        [spl["h_mixed"], spl["h_sparse"], spl["h_saturated"]]).tolist()))
    rows_hw = np.asarray(rows_hw)
    chw = np.load(P_HW / "counts.npy", mmap_mode="r")
    fp = np.load(P_HW / "first_present.npy")[rows_hw]
    lp = np.load(P_HW / "last_present.npy")[rows_hw]
    ev, fn = select_events(chw, rows_hw, t_lo=fp, t_hi=lp)
    out["traces"]["huawei"] = {
        "events": ev, "funnel": fn, "n_functions_scanned": fn["n_scanned"],
        "events_per_func_per_14d": len(ev) / fn["n_scanned"] / (chw.shape[1] / DAY / 14)}
    print(f"huawei: {fn} -> {len(ev)} events ({time.time()-t0:.0f}s)", flush=True)

    path = RUNS / "revision_r6_catalog.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
