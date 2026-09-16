# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R17: Azure-2021 faithful-branch gate experiment.

This script recomposes the v3, v4, and final age-qualified gates with the
Section-3-faithful WINTER head from revision_r11/revision_r16. It writes a
separate JSON and never mutates the archived blended serving campaign.

Torch env:
  PYTHONPATH=/data/260715/site-packages:. python3.13 \
    scripts/revision_r17_faithful_gates_2021.py

Output:
  results/runs/revision_r17_faithful_gates_2021.json
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
    rates_ewma,
    run_config,
)
from scripts.phase63_onboarding_drift import build_prototypes  # noqa: E402
from scripts.revision_a2_des import rates_proto_prefix  # noqa: E402
from scripts.revision_r11_faithful import rates_a5_faithful  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402

torch.set_num_threads(1)

P21 = PROJECT_ROOT / "data" / "processed"
RUNS = PROJECT_ROOT / "results" / "runs"
OUT = RUNS / "revision_r17_faithful_gates_2021.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = list(range(10))
LOW_RHOS = [0.01, 0.02, 0.05]
MIN_INV = 100
AGE_MIN = 720


def target_rhos(split):
    core = FULL_RHOS if split == "S1" else REDUCED_RHOS
    return sorted({float(r) for r in list(core) + LOW_RHOS})


def result_key(row):
    return (
        row["method"],
        row["split"],
        float(row["cost_ratio"]),
        int(row["seed"]),
    )


def dedupe(rows):
    by_key = {}
    for row in rows:
        by_key[result_key(row)] = row
    return [by_key[k] for k in sorted(by_key, key=lambda x: (x[1], x[2], x[3], x[0]))]


def save(out):
    out["results"] = dedupe(out["results"])
    with open(OUT, "w") as f:
        json.dump(out, f, default=float)
    blob = OUT.read_bytes()
    assert blob and blob.count(0) == 0
    json.load(open(OUT))


def load_output():
    if OUT.exists():
        out = json.load(open(OUT))
    else:
        out = {
            "description": "Faithful WINTER-head v3/v4/age-qualified gates, Azure 2021",
            "methods": [
                "A5_gated_v3_faithful",
                "A5_gated_v4_faithful",
                "A5_gated_aq_faithful",
            ],
            "seeds": SEEDS,
            "low_rhos": LOW_RHOS,
            "min_invocations": MIN_INV,
            "age_min": AGE_MIN,
            "results": [],
            "routing": {},
        }
    out["results"] = dedupe(out.get("results", []))
    out.setdefault("routing", {})
    return out


def age_matrix(counts):
    n, t_len = counts.shape
    age = np.zeros((n, t_len), dtype=np.int64)
    for f in range(n):
        nz = np.flatnonzero(counts[f])
        if len(nz):
            t0 = int(nz[0])
            age[f, t0:] = np.arange(t_len - t0)
    return age


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


def summarize(out):
    summary = {}
    rows = out["results"]
    for method in out["methods"]:
        summary[method] = {}
        for split in ["S1", "S2", "S3"]:
            summary[method][split] = {}
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
                summary[method][split][f"{rho:g}"] = {
                    "csr": float(np.mean([r["csr"] for r in cell])),
                    "csr_pct": float(np.mean([r["csr"] for r in cell]) * 100.0),
                    "csr_std_pct": float(np.std([r["csr"] for r in cell]) * 100.0),
                    "wm": float(np.mean([r["wm_per_1k_inv"] for r in cell])),
                    "n": len(cell),
                }
    out["readout"] = summary


def print_summary(out):
    print("\n=== R17 faithful-gate readout at rho=10 ===")
    for method in out["methods"]:
        print(f"\n{method}")
        for split in ["S1", "S2", "S3"]:
            rec = out["readout"][method][split].get("10")
            if not rec:
                print(f"  {split}: missing")
                continue
            print(
                f"  {split}: csr={rec['csr_pct']:.4f}%"
                f"±{rec['csr_std_pct']:.4f} wm={rec['wm']:.1f} n={rec['n']}"
            )


def main():
    out = load_output()
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

    methods = [
        "A5_gated_v3_faithful",
        "A5_gated_v4_faithful",
        "A5_gated_aq_faithful",
    ]
    for split in ["S1", "S2", "S3"]:
        wanted = {
            (method, split, rho, seed)
            for method in methods
            for rho in target_rhos(split)
            for seed in SEEDS
        }
        done = {result_key(r) for r in out["results"]}
        missing = sorted(wanted - done, key=lambda x: (x[2], x[0], x[3]))
        if not missing:
            print(f"### {split}: all faithful-gate cells complete; skipping")
            continue

        test_idx, counts, feats, dm, ds = split_arrays(
            features, counts_all, splits, dur_df, split
        )
        missing_rhos = sorted({k[2] for k in missing})
        print(
            f"### {split}: {len(test_idx)} funcs; composing faithful gates for "
            f"missing rhos {missing_rhos}",
            flush=True,
        )
        t0 = time.time()
        faithful = rates_a5_faithful(counts.astype(np.float32), feats, trainer, pm)
        ewma = rates_ewma(counts)
        proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
        age = age_matrix(counts)

        zero = cum_prev == 0
        mid = (cum_prev > 0) & (cum_prev < MIN_INV)
        conv_young = (cum_prev >= MIN_INV) & (age < AGE_MIN)
        conv_old = (cum_prev >= MIN_INV) & (age >= AGE_MIN)

        v3 = np.where(zero, proto, np.where(mid, ewma, faithful)).astype(np.float32)
        v4 = np.where(zero, proto, np.where(mid, faithful, ewma)).astype(np.float32)
        aq = np.where(
            zero,
            proto,
            np.where(mid, ewma, np.where(conv_young, faithful, ewma)),
        ).astype(np.float32)
        out["routing"][split] = {
            "frac_ticks_proto": float(zero.mean()),
            "frac_ticks_mid_ewma": float(mid.mean()),
            "frac_ticks_conv_young_faithful": float(conv_young.mean()),
            "frac_ticks_conv_old_ewma": float(conv_old.mean()),
            "frac_ticks_v3_faithful": float((cum_prev >= MIN_INV).mean()),
            "frac_ticks_v4_faithful": float(mid.mean()),
        }
        print(f"  rates/gates done in {time.time() - t0:.0f}s; routing {out['routing'][split]}")

        rate_mats = {
            "A5_gated_v3_faithful": v3,
            "A5_gated_v4_faithful": v4,
            "A5_gated_aq_faithful": aq,
        }
        jobs = []
        for rho in missing_rhos:
            tau = newsvendor_quantile(rho)
            for method, rates in rate_mats.items():
                seeds = sorted({seed for m, _, r, seed in missing if m == method and r == rho})
                if not seeds:
                    continue
                pw, ka = decisions_from_rates(rates, tau)
                jobs.append((method, rho, seeds, counts, pw, ka, dm, ds, COLD_INIT, split))

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

    summarize(out)
    save(out)
    print_summary(out)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
