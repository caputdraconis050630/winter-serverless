#!/usr/bin/env python3
"""R8: statistics addendum for the revision — Holm-adjusted p-values for the
Huawei learned-vs-EWMA family, and the nuisance-variance components that
justify the +/-0.10 pp TOST margin.

Recomputation only: reads existing archives, writes
results/runs/revision_r8_stats_addendum.json. Family pre-registered in
revision_r8_PREREG.md (9 comparisons: 3 pools x 3 rho, A5 vs EWMA).
"""

import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent.parent / "results" / "runs"


def holm(pvals):
    """Holm step-down adjusted p-values, order-preserving."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


stats = json.load(open(RUNS / "revision_h2_huawei_stats.json"))
cells = stats["cells"]

POOLS = ["h_mixed", "h_sparse", "h_saturated"]
RHOS = ["1.0", "10.0", "100.0"]

family = []  # (key, p) for A5 vs EWMA
for pool in POOLS:
    for rho in RHOS:
        key = f"{pool}|{rho}|A5_full_system"
        family.append((f"{pool}|{rho}", cells[key]["wilcoxon_p_vs_ewma"]))

keys = [k for k, _ in family]
pvals = [p for _, p in family]
p9 = holm(pvals)

# secondary: within-pool m=3
p3 = {}
for pool in POOLS:
    idx = [i for i, k in enumerate(keys) if k.startswith(pool)]
    adj = holm([pvals[i] for i in idx])
    for j, i in enumerate(idx):
        p3[keys[i]] = adj[j]

# secondary: all vs-EWMA cells in the file (every arm, every pool/rho)
all_keys, all_p = [], []
for key, c in cells.items():
    p = c.get("wilcoxon_p_vs_ewma")
    if p is not None and not key.endswith("B4a_ewma"):
        all_keys.append(key)
        all_p.append(p)
p_all = dict(zip(all_keys, holm(all_p)))

primary = {
    k: {
        "wilcoxon_p_unadjusted": pvals[i],
        "p_holm_family9": float(p9[i]),
        "p_holm_within_pool3": float(p3[k]),
        "diff_vs_ewma_pp": cells[f"{k.split('|')[0]}|{k.split('|')[1]}|A5_full_system"]["diff_vs_ewma_pp"],
    }
    for i, k in enumerate(keys)
}

# --- TOST margin nuisance components -----------------------------------
modelseed = json.load(open(RUNS / "modelseed_robustness.json"))
sim = json.load(open(RUNS / "sim_results_des_revision.json"))

by_cell = {}
for r in sim["results"] if isinstance(sim, dict) and "results" in sim else sim:
    key = (r["method"], r["split"], r["cost_ratio"])
    by_cell.setdefault(key, []).append(r["csr"])
seed_sd_pp = sorted(np.std(v) * 100 for v in by_cell.values() if len(v) >= 5)
seed_sd_pp = np.array(seed_sd_pp)

multiseed = json.load(open(RUNS / "multiseed_training.json"))

margin = {
    "registered_margin_pp": 0.10,
    "across_training_seed_csr_sd_pp": modelseed["csr_across_models_std"] * 100,
    "within_ckpt_des_seed_sd_pp": {
        "n_cells": int(len(seed_sd_pp)),
        "median": float(np.median(seed_sd_pp)),
        "min": float(seed_sd_pp.min()),
        "max": float(seed_sd_pp.max()),
    },
    "forecaster_training_seed_crps_sd": multiseed["crps_std"],
    "note": (
        "margin / across-training-seed sd = "
        f"{0.10 / (modelseed['csr_across_models_std'] * 100):.2f}x (binding term); "
        f"margin / median DES-seed sd = {0.10 / float(np.median(seed_sd_pp)):.1f}x"
    ),
}

# --- power-note inputs -------------------------------------------------
onb = json.load(open(RUNS / "onboarding_drift_results.json"))
power = {
    "injection_cohort_n": onb["onboarding"]["n_functions"],
    "drift_trigger_n_events": onb["drift"]["trigger_bench"][0]["n_events"],
    "drift_event_composition": "4 drift configs x 4 qualifying functions",
    "seeds_small_campaigns": onb["config"]["des_seeds"],
}

out = {
    "prereg": "revision_r8_PREREG.md",
    "family_definition": "learned (A5_full_system) vs EWMA, 3 Huawei pools x 3 rho (m=9), Holm step-down",
    "primary_family9": primary,
    "secondary_all_vs_ewma_m": len(all_p),
    "secondary_all_vs_ewma_p_holm": {k: float(v) for k, v in p_all.items()},
    "tost_margin_evidence": margin,
    "power_note_inputs": power,
    "sources": [
        "revision_h2_huawei_stats.json",
        "modelseed_robustness.json",
        "sim_results_des_revision.json",
        "multiseed_training.json",
        "onboarding_drift_results.json",
    ],
}
path = RUNS / "revision_r8_stats_addendum.json"
json.dump(out, open(path, "w"), indent=1)
print(f"wrote {path}")
print("\nprimary family (m=9):")
for k, v in primary.items():
    flag = " <-- paper's exception" if k == "h_sparse|100.0" else ""
    print(f"  {k:22s} p={v['wilcoxon_p_unadjusted']:.4f}  p_holm9={v['p_holm_family9']:.4f}"
          f"  p_holm3={v['p_holm_within_pool3']:.4f}  diff={v['diff_vs_ewma_pp']:+.2f}pp{flag}")
print("\nmargin evidence:", margin["note"])
