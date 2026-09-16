#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R21: post-hoc cross-provider faithful steady-window audit.

This script exports DES jobs for the current Section-3 faithful WINTER head
and faithful gates on Azure 2019 and Huawei 2023 steady-window cohorts, then
summarizes the fast-DES outputs. It does not edit paper sources.

Typical use:
  PYTHONPATH=/data/260715/site-packages:. python3.13 \
    scripts/revision_r21_cross_provider_faithful.py --export --trace azure2019

  PYTHONPATH=/data/260715/site-packages-des:. python3.13 \
    scripts/des_runner_fast.py --jobs results/runs/des_jobs_r21_azure2019 \
    --out results/runs/revision_r21_azure2019_faithful_fast_runs.json

  PYTHONPATH=/data/260715/site-packages:. python3.13 \
    scripts/revision_r21_cross_provider_faithful.py --summarize \
    --inputs results/runs/revision_r21_azure2019_faithful_fast_runs.json
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RUNS = PROJECT_ROOT / "results" / "runs"
TABLES = PROJECT_ROOT / "results" / "tables"
P21 = PROJECT_ROOT / "data" / "processed"
P19 = PROJECT_ROOT / "data" / "processed_2019"
PHW = PROJECT_ROOT / "data" / "processed_huawei"

RHOS = [0.1, 1.0, 10.0, 100.0]
SEEDS = [0, 1, 2]
MIN_INV = 100
AGE_MIN = 720
KAPPA = 15.0
BOOT_N = 5000
BOOT_SEED = 260715


def parse_float_list(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def selected_rhos(args):
    return parse_float_list(args.rhos) if args.rhos else RHOS


def selected_seeds(args):
    return parse_int_list(args.seeds) if args.seeds else SEEDS


def selected_jobs_root(args, default):
    return Path(args.jobs_root) if args.jobs_root else default


def selected_manifest_out(args, default):
    return Path(args.manifest_out) if args.manifest_out else default


def age_matrix_from_counts(counts):
    n, t_len = counts.shape
    age = np.zeros((n, t_len), dtype=np.int64)
    for f in range(n):
        nz = np.flatnonzero(counts[f])
        if len(nz):
            t0 = int(nz[0])
            age[f, t0:] = np.arange(t_len - t0)
    return age


def method_meta(method):
    aliases = {
        "B4a_ewma": ("rate", "", "", "EWMA"),
        "A5_faithful": ("faithful", "", "", "faithful_head"),
        "G_WE_Ainf": ("faithful_gate", "WE", "inf", "v3"),
        "G_WL_A0": ("faithful_gate", "WL", "0", "v4"),
        "G_WE_A720": ("faithful_gate", "WE", "720", "WINTERG_720"),
    }
    return aliases.get(method, ("archived_or_context", "", "", method))


def compose_gate_rates(proto, ewma, faithful, cum_prev, age, method):
    zero = cum_prev == 0
    w_mask = (cum_prev > 0) & (cum_prev < MIN_INV)
    conv = cum_prev >= MIN_INV
    rates = np.empty_like(ewma, dtype=np.float32)
    rates[zero] = proto[zero]
    if method == "G_WE_Ainf":
        rates[w_mask] = ewma[w_mask]
        rates[conv] = faithful[conv]
    elif method == "G_WL_A0":
        rates[w_mask] = faithful[w_mask]
        rates[conv] = ewma[conv]
    elif method == "G_WE_A720":
        young = conv & (age < AGE_MIN)
        rates[w_mask] = ewma[w_mask]
        rates[young] = faithful[young]
        rates[conv & ~young] = ewma[conv & ~young]
    else:
        raise ValueError(method)
    return rates


def weighted_share(mask, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if mask.shape[0] != len(weights):
        raise ValueError("mask/weight length mismatch")
    return float((mask * weights[:, None]).sum() / (weights.sum() * mask.shape[1]))


def export_azure2019(args):
    import torch

    torch.set_num_threads(1)
    from scripts.phase6_des import COLD_INIT, decisions_from_rates, rates_ewma
    from scripts.phase63_onboarding_drift import build_prototypes
    from scripts.revision_a2_des import rates_proto_prefix
    from scripts.revision_r11_faithful import rates_a5_faithful
    from src.decision.newsvendor import newsvendor_quantile
    from src.meta.trainer import ANILMetaTrainer
    from src.models.heads import N_QUANTILES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_mm = np.load(P19 / "features.npy", mmap_mode="r")
    counts_mm = np.load(P19 / "counts.npy", mmap_mode="r")
    splits = np.load(P19 / "splits.npz")
    dur_df = pd.read_csv(P19 / "duration_stats.csv")
    sample_manifest = json.load(open(RUNS / "des_sample_manifest_2019.json"))

    trainer = ANILMetaTrainer(
        body_type="tcn",
        head_type="ridge",
        in_features=features_mm.shape[2],
        embedding_dim=64,
        n_quantiles=N_QUANTILES,
        n_horizons=1,
        device=device,
    )
    ckpt = RUNS / "best_anil_ridge_s1_s0.pt"
    trainer.load(ckpt)
    ckpt_md5 = hashlib.md5(open(ckpt, "rb").read()).hexdigest()

    feat21 = np.load(P21 / "features.npy")
    cnt21 = np.load(P21 / "counts.npy")
    spl21 = np.load(P21 / "splits.npz")
    pm = build_prototypes(trainer, feat21, cnt21, spl21["s1_train"])

    rhos = selected_rhos(args)
    seeds = selected_seeds(args)
    jobs_root = selected_jobs_root(
        args,
        RUNS / ("des_jobs_r21_azure2019_pilot" if args.pilot else "des_jobs_r21_azure2019"),
    )
    out = selected_manifest_out(
        args,
        RUNS / ("revision_r21_azure2019_manifest_pilot.json" if args.pilot else "revision_r21_azure2019_manifest.json"),
    )
    manifest = {
        "analysis": "R21 cross-provider faithful steady-window audit",
        "trace": "azure2019",
        "checkpoint": str(ckpt),
        "checkpoint_md5": ckpt_md5,
        "proto_source": "Azure 2021 s1_train",
        "sample_manifest": str(RUNS / "des_sample_manifest_2019.json"),
        "rhos": rhos,
        "seeds": seeds,
        "min_invocations": MIN_INV,
        "age_min": AGE_MIN,
        "splits": {},
    }
    if out.exists() and not args.pilot:
        prev = json.load(open(out))
        manifest["splits"].update(prev.get("splits", {}))

    split_names = args.splits.split(",")
    for split in split_names:
        key = {"S1": "s1_test", "S2": "s2_test", "S3": "s3_test"}[split]
        rows = np.asarray(sample_manifest[split]["rows"], dtype=np.int64)
        weights = np.asarray(sample_manifest[split]["weights"], dtype=np.float64)
        if args.pilot:
            # The manifest starts with the take-all high-volume stratum.
            # Pilot runs use the tail so DES validation remains lightweight.
            rows = rows[-args.pilot :]
            weights = weights[-args.pilot :]

        t0 = time.time()
        counts = np.asarray(counts_mm[rows]).astype(np.float32)
        feats = np.asarray(features_mm[rows], dtype=np.float32)
        if split == "S3":
            ts = int(splits.get("s3_test_t_start", [14400])[0])
            counts = counts[:, ts:]
            feats = feats[:, ts:]
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        print(
            f"### Azure 2019 {split}: {len(rows)} funcs, {counts.shape[1]} ticks, "
            f"{int(counts.sum()):,} invocations",
            flush=True,
        )

        ewma = rates_ewma(counts)
        faithful = rates_a5_faithful(counts.astype(np.float32), feats, trainer, pm)
        proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
        age = age_matrix_from_counts(counts)

        rate_mats = {
            "B4a_ewma": ewma,
            "A5_faithful": faithful,
        }
        for method in ["G_WE_Ainf", "G_WL_A0", "G_WE_A720"]:
            rate_mats[method] = compose_gate_rates(proto, ewma, faithful, cum_prev, age, method)

        jdir = jobs_root / split
        jdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            jdir / "shared.npz",
            counts=counts.astype(np.int64),
            dur_means=dm,
            dur_stds=ds,
        )
        job_meta = []
        for method, rates in rate_mats.items():
            for rho in rhos:
                tau = newsvendor_quantile(rho)
                pw, ka = decisions_from_rates(rates, tau)
                fname = f"{method}__rho{rho:g}.npz"
                np.savez_compressed(jdir / fname, prewarm=pw, keepalive=ka)
                job_meta.append({"method": method, "rho": float(rho), "file": fname})

        with open(jdir / "jobs.json", "w") as f:
            json.dump({"split": split, "seeds": seeds, "cold_init": COLD_INIT, "jobs": job_meta}, f, indent=1)

        zero = cum_prev == 0
        w_mask = (cum_prev > 0) & (cum_prev < MIN_INV)
        conv = cum_prev >= MIN_INV
        manifest["splits"][split] = {
            "rows": rows.tolist(),
            "weights": weights.tolist(),
            "pool_size": int(sample_manifest[split]["pool_size"]),
            "invocations": int(counts.sum()),
            "n_jobs": len(job_meta),
            "routing": {
                "zero_history": weighted_share(zero, weights),
                "pre_count_threshold": weighted_share(w_mask, weights),
                "count_qualified": weighted_share(conv, weights),
                "aq720_learned": weighted_share(conv & (age < AGE_MIN), weights),
            },
            "elapsed_sec": time.time() - t0,
        }
        print(f"    exported {len(job_meta)} jobs in {time.time() - t0:.0f}s", flush=True)
        with open(out, "w") as f:
            json.dump(manifest, f, indent=1)

    with open(out, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {out}")


def export_huawei(args):
    import torch

    torch.set_num_threads(1)
    from scripts.phase6_des import COLD_INIT, decisions_from_rates, rates_ewma
    from scripts.phase63_onboarding_drift import build_prototypes
    from scripts.revision_a2_des import rates_proto_prefix
    from scripts.revision_r11_faithful import rates_a5_faithful
    from src.decision.newsvendor import newsvendor_quantile
    from src.meta.trainer import ANILMetaTrainer
    from src.models.heads import N_QUANTILES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_mm = np.load(PHW / "features.npy", mmap_mode="r")
    counts_mm = np.load(PHW / "counts.npy", mmap_mode="r")
    splits = np.load(PHW / "splits.npz")
    first_present = np.load(PHW / "first_present.npy")
    steady_t = int(splits["steady_t"][0])
    dur_df = pd.read_csv(PHW / "duration_stats.csv")

    trainer = ANILMetaTrainer(
        body_type="tcn",
        head_type="ridge",
        in_features=features_mm.shape[2],
        embedding_dim=64,
        n_quantiles=N_QUANTILES,
        n_horizons=1,
        device=device,
    )
    ckpt = RUNS / "best_anil_ridge_s1_s0.pt"
    trainer.load(ckpt)
    ckpt_md5 = hashlib.md5(open(ckpt, "rb").read()).hexdigest()

    feat21 = np.load(P21 / "features.npy")
    cnt21 = np.load(P21 / "counts.npy")
    spl21 = np.load(P21 / "splits.npz")
    pm = build_prototypes(trainer, feat21, cnt21, spl21["s1_train"])

    rhos = selected_rhos(args)
    seeds = selected_seeds(args)
    jobs_root = selected_jobs_root(
        args,
        RUNS / ("des_jobs_r21_huawei_pilot" if args.pilot else "des_jobs_r21_huawei"),
    )
    out = selected_manifest_out(
        args,
        RUNS / ("revision_r21_huawei_manifest_pilot.json" if args.pilot else "revision_r21_huawei_manifest.json"),
    )
    manifest = {
        "analysis": "R21 cross-provider faithful steady-window audit",
        "trace": "huawei2023",
        "checkpoint": str(ckpt),
        "checkpoint_md5": ckpt_md5,
        "proto_source": "Azure 2021 s1_train",
        "age_source": "first_present.npy deployment-presence mask",
        "rhos": rhos,
        "seeds": seeds,
        "min_invocations": MIN_INV,
        "age_min": AGE_MIN,
        "splits": {},
    }
    if out.exists() and not args.pilot:
        prev = json.load(open(out))
        manifest["splits"].update(prev.get("splits", {}))

    for pool in args.pools.split(","):
        rows = np.asarray(splits[pool], dtype=np.int64)
        if args.pilot:
            rows = rows[-args.pilot :]
        t0 = time.time()
        counts = np.asarray(counts_mm[rows][:, :steady_t]).astype(np.float32)
        feats = np.asarray(features_mm[rows][:, :steady_t, :], dtype=np.float32)
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        fp = first_present[rows].astype(np.int64)
        print(
            f"### Huawei {pool}: {len(rows)} funcs, {counts.shape[1]} ticks, "
            f"{int(counts.sum()):,} invocations",
            flush=True,
        )

        ewma = rates_ewma(counts)
        faithful = rates_a5_faithful(counts.astype(np.float32), feats, trainer, pm)
        proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
        ticks = np.arange(counts.shape[1])[None, :]
        age = ticks - fp[:, None]

        rate_mats = {
            "B4a_ewma": ewma,
            "A5_faithful": faithful,
        }
        for method in ["G_WE_Ainf", "G_WL_A0", "G_WE_A720"]:
            rate_mats[method] = compose_gate_rates(proto, ewma, faithful, cum_prev, age, method)

        jdir = jobs_root / pool
        jdir.mkdir(parents=True, exist_ok=True)
        shared_src = RUNS / "des_jobs_huawei" / pool / "shared.npz"
        if shared_src.exists() and not args.pilot:
            shutil.copy(shared_src, jdir / "shared.npz")
        else:
            np.savez_compressed(
                jdir / "shared.npz",
                counts=counts.astype(np.int64),
                dur_means=dm,
                dur_stds=ds,
            )
        job_meta = []
        for method, rates in rate_mats.items():
            for rho in rhos:
                tau = newsvendor_quantile(rho)
                pw, ka = decisions_from_rates(rates, tau)
                fname = f"{method}__rho{rho:g}.npz"
                np.savez_compressed(jdir / fname, prewarm=pw, keepalive=ka)
                job_meta.append({"method": method, "rho": float(rho), "file": fname})

        with open(jdir / "jobs.json", "w") as f:
            json.dump({"split": pool, "seeds": seeds, "cold_init": COLD_INIT, "jobs": job_meta}, f, indent=1)

        zero = cum_prev == 0
        w_mask = (cum_prev > 0) & (cum_prev < MIN_INV)
        conv = cum_prev >= MIN_INV
        weights = np.ones(len(rows), dtype=np.float64)
        manifest["splits"][pool] = {
            "rows": rows.tolist(),
            "weights": weights.tolist(),
            "pool_size": int(len(splits[pool])),
            "invocations": int(counts.sum()),
            "n_jobs": len(job_meta),
            "routing": {
                "zero_history": weighted_share(zero, weights),
                "pre_count_threshold": weighted_share(w_mask, weights),
                "count_qualified": weighted_share(conv, weights),
                "aq720_learned": weighted_share(conv & (age < AGE_MIN), weights),
            },
            "elapsed_sec": time.time() - t0,
        }
        print(f"    exported {len(job_meta)} jobs in {time.time() - t0:.0f}s", flush=True)
        with open(out, "w") as f:
            json.dump(manifest, f, indent=1)

    with open(out, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {out}")


def load_rows(paths):
    rows = []
    for raw in paths:
        path = Path(raw)
        data = json.load(open(path))
        part = data.get("results", data) if isinstance(data, dict) else data
        for row in part:
            row = dict(row)
            row["_source_json"] = str(path)
            rows.append(row)
        print(f"loaded {len(part)} rows from {path}")
    return rows


def load_manifests(paths):
    out = {}
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        data = json.load(open(path))
        for split, rec in data.get("splits", {}).items():
            out[split] = rec
    return out


def summarize_cell(rows, weights):
    if not rows:
        return None
    total_by_seed = []
    cold_by_seed = []
    wm_by_seed = []
    for row in rows:
        total = np.asarray(row["func_total"], dtype=np.float64)
        cold = np.asarray(row["func_cold"], dtype=np.float64)
        wm = np.asarray(row["func_wm"], dtype=np.float64)
        total_by_seed.append(float((weights * total).sum()))
        cold_by_seed.append(float((weights * cold).sum()))
        wm_by_seed.append(float((weights * wm).sum()))
    total_mean = float(np.mean(total_by_seed))
    cold_mean = float(np.mean(cold_by_seed))
    wm_mean = float(np.mean(wm_by_seed))
    csr_seeds = np.asarray(cold_by_seed) / np.maximum(np.asarray(total_by_seed), 1e-9)
    rho = float(rows[0]["cost_ratio"])
    return {
        "n": len(rows),
        "n_functions": len(weights),
        "invocations": total_mean,
        "cold_starts": cold_mean,
        "csr_pct": 100.0 * cold_mean / max(total_mean, 1e-9),
        "csr_seed_std_pct": 100.0 * float(np.std(csr_seeds)),
        "wm_per_1k_inv": wm_mean / max(total_mean / 1000.0, 1e-9),
        "cost_per_1k_inv": (KAPPA * rho * cold_mean + wm_mean) / max(total_mean / 1000.0, 1e-9),
    }


def paired_boot_ci(rows_a, rows_b, weights):
    if not rows_a or not rows_b:
        return "", ""
    cold_a = np.mean([np.asarray(r["func_cold"], dtype=np.float64) for r in rows_a], axis=0)
    cold_b = np.mean([np.asarray(r["func_cold"], dtype=np.float64) for r in rows_b], axis=0)
    total = np.mean([np.asarray(r["func_total"], dtype=np.float64) for r in rows_a], axis=0)
    rng = np.random.default_rng(BOOT_SEED)
    n = len(total)
    diffs = np.empty(BOOT_N)
    for i in range(BOOT_N):
        idx = rng.integers(0, n, n)
        ww = weights[idx]
        diffs[i] = (
            (ww * cold_a[idx]).sum() / max((ww * total[idx]).sum(), 1e-9)
            - (ww * cold_b[idx]).sum() / max((ww * total[idx]).sum(), 1e-9)
        )
    diffs *= 100.0
    return f"{np.percentile(diffs, 2.5):.6f}", f"{np.percentile(diffs, 97.5):.6f}"


def summarize(args):
    rows = load_rows(args.inputs)
    manifests = load_manifests(args.manifests)
    TABLES.mkdir(parents=True, exist_ok=True)
    methods = ["B4a_ewma", "A5_faithful", "G_WE_Ainf", "G_WL_A0", "G_WE_A720"]
    splits = sorted({r["split"] for r in rows})
    rhos = sorted({float(r["cost_ratio"]) for r in rows})

    header = (
        "split,method,identity,family,age_min,alias,rho,n_seeds,n_functions,"
        "csr_pct,csr_seed_std_pct,wm_per_1k_inv,cost_per_1k_inv,"
        "delta_csr_pp_vs_ewma,delta_wm_per_1k_vs_ewma,"
        "delta_cost_per_1k_vs_ewma,boot_ci95_lo_pp,boot_ci95_hi_pp,"
        "route_learned_share,source_json"
    )
    lines = [header]
    rho10 = [header]
    readout = {"analysis": "R21 cross-provider faithful steady-window audit", "cells": {}}
    for split in splits:
        if split not in manifests:
            raise SystemExit(f"missing manifest metadata for split {split}")
        weights = np.asarray(manifests[split]["weights"], dtype=np.float64)
        route_share = manifests[split].get("routing", {}).get("aq720_learned", "")
        for rho in rhos:
            ref_rows = [
                r for r in rows
                if r["split"] == split and r["method"] == "B4a_ewma"
                and abs(float(r["cost_ratio"]) - rho) < 1e-9
            ]
            ref = summarize_cell(ref_rows, weights)
            for method in methods:
                cell_rows = [
                    r for r in rows
                    if r["split"] == split and r["method"] == method
                    and abs(float(r["cost_ratio"]) - rho) < 1e-9
                ]
                rec = summarize_cell(cell_rows, weights)
                if rec is None:
                    continue
                identity, family, age, alias = method_meta(method)
                dcsr = rec["csr_pct"] - ref["csr_pct"] if ref else np.nan
                dwm = rec["wm_per_1k_inv"] - ref["wm_per_1k_inv"] if ref else np.nan
                dcost = rec["cost_per_1k_inv"] - ref["cost_per_1k_inv"] if ref else np.nan
                lo, hi = ("", "")
                if method != "B4a_ewma" and ref_rows:
                    lo, hi = paired_boot_ci(cell_rows, ref_rows, weights)
                source = ";".join(sorted({r["_source_json"] for r in cell_rows}))
                learned = route_share if method == "G_WE_A720" else (
                    manifests[split].get("routing", {}).get("count_qualified", "")
                    if method == "G_WE_Ainf" else
                    manifests[split].get("routing", {}).get("pre_count_threshold", "")
                    if method == "G_WL_A0" else 1.0 if method == "A5_faithful" else 0.0
                )
                line = (
                    f"{split},{method},{identity},{family},{age},{alias},{rho:g},"
                    f"{rec['n']},{rec['n_functions']},{rec['csr_pct']:.6f},"
                    f"{rec['csr_seed_std_pct']:.6f},{rec['wm_per_1k_inv']:.3f},"
                    f"{rec['cost_per_1k_inv']:.3f},{dcsr:.6f},{dwm:.3f},"
                    f"{dcost:.3f},{lo},{hi},{learned},{source}"
                )
                lines.append(line)
                if abs(rho - 10.0) < 1e-9:
                    rho10.append(line)
                readout["cells"][f"{split}|{rho:g}|{method}"] = rec | {
                    "delta_csr_pp_vs_ewma": float(dcsr),
                    "delta_wm_per_1k_vs_ewma": float(dwm),
                    "delta_cost_per_1k_vs_ewma": float(dcost),
                    "boot_ci95_pp": [None if lo == "" else float(lo), None if hi == "" else float(hi)],
                    "route_learned_share": learned,
                    "source_json": source,
                }

    prefix = args.prefix
    (TABLES / f"{prefix}_fullrho.csv").write_text("\n".join(lines) + "\n")
    (TABLES / f"{prefix}_rho10.csv").write_text("\n".join(rho10) + "\n")
    out_json = RUNS / f"{prefix}.json"
    with open(out_json, "w") as f:
        json.dump(readout, f, indent=1, default=float)
    print(f"wrote {TABLES / f'{prefix}_fullrho.csv'}")
    print(f"wrote {TABLES / f'{prefix}_rho10.csv'}")
    print(f"wrote {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--trace", choices=["azure2019", "huawei"], default="azure2019")
    ap.add_argument("--splits", default="S1,S2,S3")
    ap.add_argument("--pools", default="h_mixed,h_sparse,h_saturated")
    ap.add_argument("--pilot", type=int, default=0, help="limit each split/pool to the first N functions")
    ap.add_argument("--rhos", default="", help="comma-separated rho override; defaults to the full R21 grid")
    ap.add_argument("--seeds", default="", help="comma-separated DES seed override; defaults to the full R21 seed set")
    ap.add_argument("--jobs-root", default="", help="override exported DES job root")
    ap.add_argument("--manifest-out", default="", help="override exported manifest JSON path")
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--manifests", nargs="*", default=[
        str(RUNS / "revision_r21_azure2019_manifest.json"),
        str(RUNS / "revision_r21_huawei_manifest.json"),
    ])
    ap.add_argument("--prefix", default="T_r21_cross_provider_faithful")
    args = ap.parse_args()
    if args.export == args.summarize:
        ap.error("choose exactly one of --export or --summarize")
    if args.export:
        if args.trace == "azure2019":
            export_azure2019(args)
        else:
            export_huawei(args)
    else:
        if not args.inputs:
            ap.error("--summarize requires --inputs")
        summarize(args)


if __name__ == "__main__":
    main()
