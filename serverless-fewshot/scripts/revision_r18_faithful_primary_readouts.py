# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R18: readouts for the faithful-head primary steady-state revision.

Inputs:
  results/runs/revision_r16_faithful_primary_experiment.json
  results/runs/revision_r17_faithful_gates_2021.json
  existing archived comparator JSONs

Outputs:
  results/runs/revision_r18_faithful_primary_readouts.json
  results/tables/T_main_2021_faithful.csv
  results/tables/T_tost_azure_faithful.csv
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sstats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED = PROJECT_ROOT / "data" / "processed"
RUNS = PROJECT_ROOT / "results" / "runs"
TABLES = PROJECT_ROOT / "results" / "tables"

TOST_MARGIN_PP = 0.10
BOOT_N = 5000
BOOT_SEED = 11
RHO_MAIN = 10.0
KAPPA = 15.0
STATE_GBS = {
    "S1": 8e-6 * 20160 * 60,
    "S2": 8e-6 * 20160 * 60,
    "S3": 8e-6 * 10080 * 60,
}


def load_rows(path):
    data = json.load(open(path))
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data


def mean_cell(rows, method, split, rho):
    cell = [
        r for r in rows
        if r["method"] == method
        and r["split"] == split
        and abs(float(r["cost_ratio"]) - rho) < 1e-9
    ]
    if not cell:
        return None
    csr = np.array([r["csr"] for r in cell], dtype=float)
    return {
        "csr": float(csr.mean()),
        "csr_pct": float(csr.mean() * 100.0),
        "csr_std_pct": float(csr.std() * 100.0),
        "wm": float(np.mean([r["wm_per_1k_inv"] for r in cell])),
        "n": len(cell),
    }


def group_by_seed(rows):
    grouped = defaultdict(dict)
    for r in rows:
        grouped[(r["split"], float(r["cost_ratio"]), r["method"])][int(r["seed"])] = r
    return grouped


def seed_tost(diffs_pp):
    n = len(diffs_pp)
    mean = float(np.mean(diffs_pp))
    se = float(np.std(diffs_pp, ddof=1)) / np.sqrt(n)
    if se == 0:
        p = 0.0 if -TOST_MARGIN_PP < mean < TOST_MARGIN_PP else 1.0
        return mean, p, [mean, mean]
    t_lo = (mean + TOST_MARGIN_PP) / se
    t_hi = (mean - TOST_MARGIN_PP) / se
    p = max(1.0 - sstats.t.cdf(t_lo, n - 1), sstats.t.cdf(t_hi, n - 1))
    ci90 = sstats.t.interval(0.90, n - 1, loc=mean, scale=se)
    return mean, float(p), [float(ci90[0]), float(ci90[1])]


def aggregate_csr(cold, total):
    return float(np.sum(cold) / max(np.sum(total), 1e-9))


def paired_boot_ci_pp(rows_a, rows_b, rng):
    cold_a = np.mean([np.asarray(r["func_cold"], dtype=float) for r in rows_a], axis=0)
    cold_b = np.mean([np.asarray(r["func_cold"], dtype=float) for r in rows_b], axis=0)
    total = np.mean([np.asarray(r["func_total"], dtype=float) for r in rows_a], axis=0)
    n = len(total)
    diffs = np.empty(BOOT_N)
    for i in range(BOOT_N):
        idx = rng.integers(0, n, n)
        diffs[i] = aggregate_csr(cold_a[idx], total[idx]) - aggregate_csr(cold_b[idx], total[idx])
    diffs *= 100.0
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def tost_table(rows, methods, rhos):
    grouped = group_by_seed(rows)
    out = {}
    csv = [
        "arm,reference,split,rho,mean_diff_pp,ci90_lo_pp,ci90_hi_pp,"
        "tost_p,tost_equivalent,func_boot_ci_lo_pp,func_boot_ci_hi_pp"
    ]
    for arm in methods:
        for split in ["S1", "S2", "S3"]:
            for rho in rhos[split]:
                a = grouped.get((split, rho, arm), {})
                b = grouped.get((split, rho, "B4a_ewma"), {})
                seeds = sorted(set(a) & set(b))
                if len(seeds) < 10:
                    continue
                diffs = np.array([(a[s]["csr"] - b[s]["csr"]) * 100.0 for s in seeds])
                mean, p, ci90 = seed_tost(diffs)
                rows_a = [a[s] for s in seeds]
                rows_b = [b[s] for s in seeds]
                boot = paired_boot_ci_pp(rows_a, rows_b, np.random.default_rng(BOOT_SEED))
                rec = {
                    "n_seeds": len(seeds),
                    "mean_diff_pp": mean,
                    "ci90_pp": ci90,
                    "tost_p": p,
                    "tost_equivalent": bool(p < 0.05),
                    "func_boot_ci_pp": boot,
                }
                out[f"{arm}|vs|B4a_ewma|{split}|{rho:g}"] = rec
                csv.append(
                    f"{arm},B4a_ewma,{split},{rho:g},{mean:.5f},"
                    f"{ci90[0]:.5f},{ci90[1]:.5f},{p:.3e},"
                    f"{rec['tost_equivalent']},{boot[0]:.5f},{boot[1]:.5f}"
                )
    return out, csv


def cost_readout(rows, methods, rhos):
    grouped = group_by_seed(rows)
    out = {}
    for arm in methods:
        vals = []
        by_cell = {}
        for split in ["S1", "S2", "S3"]:
            for rho in rhos[split]:
                a = grouped.get((split, rho, arm), {})
                b = grouped.get((split, rho, "B4a_ewma"), {})
                seeds = sorted(set(a) & set(b))
                if len(seeds) < 10:
                    continue
                diffs = []
                for seed in seeds:
                    da = (
                        KAPPA * rho * a[seed]["cold_starts"]
                        + a[seed]["wm_total_gb_s"]
                        + STATE_GBS[split]
                    )
                    db = KAPPA * rho * b[seed]["cold_starts"] + b[seed]["wm_total_gb_s"]
                    diffs.append(da - db)
                mean = float(np.mean(diffs))
                vals.append(mean)
                by_cell[f"{split}|{rho:g}"] = {
                    "delta_cost_gbs": mean,
                    "delta_cost_million_gbs": mean / 1e6,
                }
        out[arm] = {
            "by_cell": by_cell,
            "min_delta_cost_million_gbs": float(np.min(vals) / 1e6) if vals else None,
            "max_delta_cost_million_gbs": float(np.max(vals) / 1e6) if vals else None,
            "n_positive_cost_cells": int(sum(v > 0 for v in vals)),
            "n_cells": len(vals),
        }
    return out


def ood_faithful(rows):
    import torch
    import torch.nn.functional as fun
    from scripts.phase63_onboarding_drift import build_prototypes, DEVICE
    from src.meta.trainer import ANILMetaTrainer
    from src.models.heads import N_QUANTILES

    torch.set_num_threads(1)
    features = np.load(PROCESSED / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED / "counts.npy", mmap_mode="r")
    splits = np.load(PROCESSED / "splits.npz")

    trainer = ANILMetaTrainer(
        body_type="tcn",
        head_type="ridge",
        in_features=features.shape[2],
        embedding_dim=64,
        n_quantiles=N_QUANTILES,
        n_horizons=1,
        device=DEVICE,
    )
    trainer.load(RUNS / "best_anil_ridge_s1_s0.pt")
    pm = build_prototypes(trainer, np.asarray(features), np.asarray(counts), splits["s1_train"])
    emb = pm.compute_embeddings(trainer.body, torch.from_numpy(np.asarray(features)).float(), splits["s2_test"])
    cen = pm.centroids.to(emb.device)
    sims = fun.cosine_similarity(emb.unsqueeze(1), cen.unsqueeze(0), dim=2)
    dist = (1.0 - sims.max(dim=1).values).cpu().numpy()
    out = {"n_s2": int(len(dist))}
    for rho in [0.1, 1.0, 10.0, 100.0]:
        def fc(method):
            cell = [
                r for r in rows
                if r["split"] == "S2"
                and r["method"] == method
                and abs(float(r["cost_ratio"]) - rho) < 1e-9
            ]
            return (
                np.mean([r["func_cold"] for r in cell], axis=0),
                np.mean([r["func_total"] for r in cell], axis=0),
            )
        cold_f, total = fc("A5_faithful")
        cold_e, _ = fc("B4a_ewma")
        delta = (cold_e - cold_f) / np.maximum(total, 1)  # >0: faithful wins
        rho_s, p_s = sstats.spearmanr(dist, delta)
        out[f"{rho:g}"] = {
            "spearman_r": float(rho_s),
            "p": float(p_s),
            "median_dist": float(np.median(dist)),
        }
    return out


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    r16 = load_rows(RUNS / "revision_r16_faithful_primary_experiment.json")
    r17 = load_rows(RUNS / "revision_r17_faithful_gates_2021.json")
    archived = load_rows(RUNS / "sim_results_des_revision.json")
    v4_old = load_rows(RUNS / "revision_e1_v4_2021.json")
    b2f = load_rows(RUNS / "revision_v1_hybridfull_2021.json")
    b3 = load_rows(RUNS / "revision_f1_fourier_steady.json")
    all_rows = r16 + r17 + archived + v4_old + b2f + b3

    rhos = {
        "S1": [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0],
        "S2": [0.01, 0.02, 0.05, 0.1, 1.0, 10.0, 100.0],
        "S3": [0.01, 0.02, 0.05, 0.1, 1.0, 10.0, 100.0],
    }
    table_methods = [
        ("A5_faithful", "WINTER faithful head"),
        ("A5_gated_v3_faithful", "WINTER-G v3 faithful"),
        ("A5_gated_v4_faithful", "WINTER-G v4 faithful"),
        ("A5_gated_aq_faithful", "WINTER-G faithful"),
        ("B4a_ewma", "EWMA"),
        ("B1_fixed_keepalive", "Keep-alive"),
        ("B2f_hybrid_full", "Hybrid histogram (full)"),
        ("B3_fourier", "Spectral (B3)"),
        ("Oracle", "Oracle"),
    ]

    table_csv = ["split,method,label,csr_pct,csr_std_pct,wm,nseeds"]
    table = {}
    print("% Faithful Table 1 rows")
    for split in ["S1", "S2", "S3"]:
        table[split] = {}
        print(f"% ---- {split} ----")
        for method, label in table_methods:
            rec = mean_cell(all_rows, method, split, RHO_MAIN)
            if not rec:
                print(f"% missing {split} {method}")
                continue
            table[split][method] = rec
            table_csv.append(
                f"{split},{method},{label},{rec['csr_pct']:.6f},"
                f"{rec['csr_std_pct']:.6f},{rec['wm']:.1f},{rec['n']}"
            )
            print(
                f" & {label:30s} & ${rec['csr_pct']:.2f}\\pm{rec['csr_std_pct']:.2f}$ "
                f"& {rec['wm']:,.0f} \\\\"
            )

    tost_methods = [
        "A5_faithful",
        "A5_gated_v3_faithful",
        "A5_gated_v4_faithful",
        "A5_gated_aq_faithful",
    ]
    tost, tost_csv = tost_table(r16 + r17 + archived, tost_methods, rhos)
    costs = cost_readout(r16 + r17 + archived, tost_methods, rhos)
    ood = ood_faithful(r16 + archived)

    out = {
        "sources": [
            "revision_r16_faithful_primary_experiment.json",
            "revision_r17_faithful_gates_2021.json",
            "sim_results_des_revision.json",
            "revision_e1_v4_2021.json",
            "revision_v1_hybridfull_2021.json",
            "revision_f1_fourier_steady.json",
        ],
        "rho_main": RHO_MAIN,
        "table1": table,
        "tost_vs_ewma": tost,
        "cost_vs_ewma": costs,
        "s2_ood_faithful": ood,
    }
    with open(RUNS / "revision_r18_faithful_primary_readouts.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    (TABLES / "T_main_2021_faithful.csv").write_text("\n".join(table_csv) + "\n")
    (TABLES / "T_tost_azure_faithful.csv").write_text("\n".join(tost_csv) + "\n")
    print(f"wrote {RUNS / 'revision_r18_faithful_primary_readouts.json'}")
    print(f"wrote {TABLES / 'T_main_2021_faithful.csv'}")
    print(f"wrote {TABLES / 'T_tost_azure_faithful.csv'}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", "/data/260715/site-packages:.")
    main()
