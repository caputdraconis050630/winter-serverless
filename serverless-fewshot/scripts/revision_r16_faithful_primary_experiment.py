# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R16: full-grid faithful-head Azure-2021 steady-state experiment.

This is an experiment-only extension of revision_r11_faithful.py. It does not
modify paper files or replace the archived blended serving arm. Existing R11
A5_faithful rows are reused for the reduced rho grid; this script computes only
missing primary-grid cells and writes a separate JSON.

Torch env:
  PYTHONPATH=/data/260715/site-packages:. python3.13 \
    scripts/revision_r16_faithful_primary_experiment.py

Output:
  results/runs/revision_r16_faithful_primary_experiment.json
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import (  # noqa: E402
    COLD_INIT,
    FULL_RHOS,
    REDUCED_RHOS,
    decisions_from_rates,
    run_config,
)
from scripts.phase63_onboarding_drift import build_prototypes  # noqa: E402
from scripts.revision_r11_faithful import rates_a5_faithful  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402

torch.set_num_threads(1)

P21 = PROJECT_ROOT / "data" / "processed"
RUNS = PROJECT_ROOT / "results" / "runs"
OUT = RUNS / "revision_r16_faithful_primary_experiment.json"
R11 = RUNS / "revision_r11_faithful.json"
ARCHIVED = RUNS / "sim_results_des_revision.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = list(range(10))
LOW_RHOS = [0.01, 0.02, 0.05]


def target_rhos(split):
    """Match the Azure-2021 primary steady-state grid."""
    core = FULL_RHOS if split == "S1" else REDUCED_RHOS
    return sorted({float(r) for r in list(core) + LOW_RHOS})


def result_key(row):
    return (
        row["method"],
        row["split"],
        float(row["cost_ratio"]),
        int(row["seed"]),
    )


def load_initial_output():
    if OUT.exists():
        out = json.load(open(OUT))
    else:
        out = {
            "description": "Experiment-only full-grid faithful WINTER head",
            "source": {
                "reused_reduced_grid": str(R11),
                "archived_comparators": str(ARCHIVED),
            },
            "method": "A5_faithful",
            "device": DEVICE,
            "seeds": SEEDS,
            "low_rhos": LOW_RHOS,
            "results": [],
        }

    if R11.exists():
        seen = {result_key(r) for r in out["results"]}
        for row in json.load(open(R11)).get("results", []):
            if row.get("method") != "A5_faithful":
                continue
            key = result_key(row)
            if key not in seen:
                out["results"].append(row)
                seen.add(key)

    out["results"] = dedupe_results(out["results"])
    return out


def dedupe_results(rows):
    by_key = {}
    for row in rows:
        by_key[result_key(row)] = row
    return [by_key[k] for k in sorted(by_key, key=lambda x: (x[1], x[2], x[3], x[0]))]


def save(out):
    out["results"] = dedupe_results(out["results"])
    with open(OUT, "w") as f:
        json.dump(out, f, default=float)
    blob = OUT.read_bytes()
    assert blob and blob.count(0) == 0
    json.load(open(OUT))


def split_arrays(features, counts_all, splits, dur_df, split):
    key = {"S1": "s1_test", "S2": "s2_test", "S3": "s3_test"}[split]
    test_idx = splits[key]
    counts = counts_all[test_idx]
    feats = features[test_idx]
    if split == "S3":
        t0 = int(splits.get("s3_test_t_start", [10080])[0])
        counts = counts[:, t0:]
        feats = feats[:, t0:, :]
    dm = np.nan_to_num(
        np.array(
            [
                dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
                for fi in test_idx
            ]
        ),
        nan=1.0,
    )
    ds = np.nan_to_num(
        np.array(
            [
                dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
                for fi in test_idx
            ]
        ),
        nan=0.5,
    )
    return test_idx, counts, feats, dm, ds


def add_archived_context(out):
    if not ARCHIVED.exists():
        return
    rows = json.load(open(ARCHIVED))["results"]
    ctx = {}
    for method in ["A5_full_system", "B4a_ewma"]:
        ctx[method] = {}
        for split in ["S1", "S2", "S3"]:
            for rho in target_rhos(split):
                cell = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["split"] == split
                    and abs(float(r["cost_ratio"]) - rho) < 1e-9
                ]
                if not cell:
                    continue
                ctx[method][f"{split}|{rho:g}"] = {
                    "csr": float(np.mean([r["csr"] for r in cell])),
                    "csr_std": float(np.std([r["csr"] for r in cell])),
                    "wm": float(np.mean([r["wm_per_1k_inv"] for r in cell])),
                    "n": len(cell),
                }
    out["archived_comparators"] = ctx


def add_readout(out):
    rows = out["results"]
    archived = out.get("archived_comparators", {})
    summary = {}
    for split in ["S1", "S2", "S3"]:
        summary[split] = {}
        for rho in target_rhos(split):
            cell = [
                r
                for r in rows
                if r["method"] == "A5_faithful"
                and r["split"] == split
                and abs(float(r["cost_ratio"]) - rho) < 1e-9
            ]
            if not cell:
                continue
            csr = float(np.mean([r["csr"] for r in cell]))
            csr_std = float(np.std([r["csr"] for r in cell]))
            wm = float(np.mean([r["wm_per_1k_inv"] for r in cell]))
            rec = {
                "csr": csr,
                "csr_pct": csr * 100.0,
                "csr_std_pct": csr_std * 100.0,
                "wm": wm,
                "n": len(cell),
            }
            key = f"{split}|{rho:g}"
            ewma = archived.get("B4a_ewma", {}).get(key)
            blend = archived.get("A5_full_system", {}).get(key)
            if ewma:
                rec["diff_vs_ewma_pp"] = (csr - ewma["csr"]) * 100.0
                rec["wm_ratio_vs_ewma"] = wm / ewma["wm"]
            if blend:
                rec["diff_vs_archived_blend_pp"] = (csr - blend["csr"]) * 100.0
                rec["wm_ratio_vs_archived_blend"] = wm / blend["wm"]
            summary[split][f"{rho:g}"] = rec
    out["readout"] = {
        "summary": summary,
        "note": (
            "Positive diff_vs_ewma_pp means the faithful head has a higher "
            "cold-start rate than EWMA."
        ),
    }


def print_readout(out):
    print("\n=== A5_faithful full-grid readout ===")
    better = worse = tied = 0
    for split in ["S1", "S2", "S3"]:
        print(f"\n{split}")
        for rho in target_rhos(split):
            rec = out["readout"]["summary"].get(split, {}).get(f"{rho:g}")
            if not rec:
                print(f"  rho={rho:g}: MISSING")
                continue
            diff = rec.get("diff_vs_ewma_pp")
            wm_ratio = rec.get("wm_ratio_vs_ewma")
            if diff is not None:
                if diff < -0.01:
                    better += 1
                elif diff > 0.01:
                    worse += 1
                else:
                    tied += 1
            print(
                f"  rho={rho:g}: csr={rec['csr_pct']:.4f}%"
                f"±{rec['csr_std_pct']:.4f} wm={rec['wm']:.1f}"
                f" diff_vs_ewma={diff:+.4f}pp"
                f" wm/ewma={wm_ratio:.3f} n={rec['n']}"
            )
    print(f"\nvs EWMA by CSR: better={better}, worse={worse}, tied(|diff|<=0.01pp)={tied}")
    print(f"wrote {OUT}")


def main():
    out = load_initial_output()
    save(out)

    features = np.load(P21 / "features.npy")
    counts_all = np.load(P21 / "counts.npy")
    splits = np.load(P21 / "splits.npz")
    dur_df = pd.read_csv(P21 / "duration_stats.csv")

    trainer = ANILMetaTrainer(
        body_type="tcn",
        head_type="ridge",
        in_features=features.shape[2],
        embedding_dim=64,
        n_quantiles=N_QUANTILES,
        n_horizons=1,
        device=DEVICE,
    )
    ckpt = RUNS / "best_anil_ridge_s1_s0.pt"
    trainer.load(ckpt)
    pm = build_prototypes(trainer, features, counts_all, splits["s1_train"])
    out["checkpoint"] = str(ckpt)
    out["meta_lambda"] = float(trainer.head.ridge_lambda.item())

    for split in ["S1", "S2", "S3"]:
        wanted = {
            ("A5_faithful", split, rho, seed)
            for rho in target_rhos(split)
            for seed in SEEDS
        }
        done = {result_key(r) for r in out["results"]}
        missing = sorted(wanted - done, key=lambda x: (x[2], x[3]))
        if not missing:
            print(f"### {split}: all faithful cells complete; skipping")
            continue

        test_idx, counts, feats, dm, ds = split_arrays(
            features, counts_all, splits, dur_df, split
        )
        missing_rhos = sorted({k[2] for k in missing})
        print(
            f"### {split}: {len(test_idx)} funcs; computing rates for "
            f"missing rhos {missing_rhos}",
            flush=True,
        )
        t_start = time.time()
        rates = rates_a5_faithful(counts.astype(np.float32), feats, trainer, pm)
        print(f"  rates done in {time.time() - t_start:.0f}s", flush=True)

        jobs = []
        for rho in missing_rhos:
            seeds = sorted({seed for _, _, r, seed in missing if r == rho})
            tau = newsvendor_quantile(rho)
            pw, ka = decisions_from_rates(rates, tau)
            jobs.append(("A5_faithful", rho, seeds, counts, pw, ka, dm, ds, COLD_INIT, split))

        with ProcessPoolExecutor(max_workers=4) as ex:
            for res in ex.map(run_config, jobs):
                for row in res:
                    row.pop("func_csr", None)
                out["results"].extend(res)
                r0 = res[0]
                print(
                    f"    {split} {r0['method']} rho={r0['cost_ratio']:<6g} "
                    f"csr={np.mean([x['csr'] for x in res]) * 100:.4f}% "
                    f"wm={np.mean([x['wm_per_1k_inv'] for x in res]):.1f}",
                    flush=True,
                )
                save(out)

    add_archived_context(out)
    add_readout(out)
    save(out)
    print_readout(out)


if __name__ == "__main__":
    main()
