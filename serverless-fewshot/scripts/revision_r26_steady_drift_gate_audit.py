#!/usr/bin/env python3
"""R26 steady-window false-positive audit for drift-aware WINTER-G.

Exports fast-DES jobs for EWMA, current 720-minute WINTER-G, and the new
drift-aware gates on steady windows with no injected drift. This checks whether
the recovery transition creates false-positive cost/CSR regressions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import COLD_INIT, decisions_from_rates, rates_ewma  # noqa: E402
from scripts.phase63_onboarding_drift import build_prototypes  # noqa: E402
from scripts.revision_a2_des import rates_proto_prefix  # noqa: E402
from scripts.revision_r11_faithful import rates_a5_faithful  # noqa: E402
from scripts.revision_r21_cross_provider_faithful import (  # noqa: E402
    KAPPA,
    age_matrix_from_counts,
    paired_boot_ci,
    parse_float_list,
    parse_int_list,
    summarize_cell,
    weighted_share,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.drift.triggers import ConformalMonitor  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402


RUNS = PROJECT_ROOT / "results" / "runs"
TABLES = PROJECT_ROOT / "results" / "tables"
P21 = PROJECT_ROOT / "data" / "processed"
P19 = PROJECT_ROOT / "data" / "processed_2019"
PHW = PROJECT_ROOT / "data" / "processed_huawei"

RHOS = [0.1, 1.0, 10.0, 100.0]
SEEDS = [0, 1, 2]
MIN_INV = 100
AGE_MIN = 720
RECOVERY_TTL = 240
STABLE_EXIT = 120
EPS = 1e-9


def selected_rhos(args) -> List[float]:
    return parse_float_list(args.rhos) if args.rhos else RHOS


def selected_seeds(args) -> List[int]:
    return parse_int_list(args.seeds) if args.seeds else SEEDS


def _new_monitor() -> ConformalMonitor:
    return ConformalMonitor(
        alpha=0.1,
        delta=0.15,
        m_consecutive=3,
        calib_size=120,
        window=30,
        min_calib=30,
    )


def current_gate_rates(proto, ewma, faithful, cum_prev, age):
    zero = cum_prev == 0
    warm = (cum_prev > 0) & (cum_prev < MIN_INV)
    young = (cum_prev >= MIN_INV) & (age < AGE_MIN)
    mature = (cum_prev >= MIN_INV) & ~young
    rates = ewma.copy().astype(np.float32)
    rates[zero] = proto[zero]
    rates[warm] = ewma[warm]
    rates[young] = faithful[young]
    rates[mature] = ewma[mature]
    return rates, zero, warm, young, mature


def drift_recovery_matrix(counts, proto, ewma, faithful, cum_prev, age):
    n_funcs, t_len = counts.shape
    rates = np.empty_like(ewma, dtype=np.float32)
    route_recovery = np.zeros_like(counts, dtype=bool)
    route_young = np.zeros_like(counts, dtype=bool)
    fires = np.zeros(n_funcs, dtype=np.int64)
    recovery_entries = np.zeros(n_funcs, dtype=np.int64)
    recovery_ticks = np.zeros(n_funcs, dtype=np.int64)
    active_at_start = np.zeros(n_funcs, dtype=np.int64)

    for f in range(n_funcs):
        monitor = _new_monitor()
        state = "stable"
        recovery_start = 10**12
        last_fire = -10**9
        for t in range(t_len):
            if state == "recovery" and (t - recovery_start) >= RECOVERY_TTL and (t - last_fire) >= STABLE_EXIT:
                state = "stable"
                monitor = _new_monitor()
            if t == 0 and state == "recovery":
                active_at_start[f] = 1

            if cum_prev[f, t] == 0:
                rates[f, t] = proto[f, t]
                continue
            if cum_prev[f, t] < MIN_INV:
                rates[f, t] = ewma[f, t]
                continue
            if age[f, t] < AGE_MIN:
                rates[f, t] = faithful[f, t]
                route_young[f, t] = True
                continue

            if state == "recovery":
                rates[f, t] = faithful[f, t]
                route_recovery[f, t] = True
                recovery_ticks[f] += 1
                pred_log = np.log1p(max(float(faithful[f, t]), 0.0))
            else:
                rates[f, t] = ewma[f, t]
                pred_log = np.log1p(max(float(ewma[f, t]), 0.0))

            resid = abs(float(np.log1p(max(float(counts[f, t]), 0.0))) - pred_log)
            if monitor.update(resid):
                fires[f] += 1
                recovery_entries[f] += 1
                state = "recovery"
                recovery_start = t + 1
                last_fire = t
                monitor = _new_monitor()

    info = {
        "fires": fires,
        "recovery_entries": recovery_entries,
        "recovery_ticks": recovery_ticks,
        "route_recovery": route_recovery,
        "route_young": route_young,
        "active_at_start": active_at_start,
    }
    return rates, info


def guarded_decisions(current_rates, ewma, candidate_rates, route_recovery, rho, guard):
    tau = newsvendor_quantile(float(rho))
    cur_pw, cur_ka = decisions_from_rates(current_rates, tau)
    ewma_pw, ewma_ka = decisions_from_rates(ewma, tau)
    cand_pw, cand_ka = decisions_from_rates(candidate_rates, tau)
    use_recovery = route_recovery.copy()
    if guard:
        cand_proxy = cand_pw.astype(np.float64) * cand_ka.astype(np.float64)
        ewma_proxy = ewma_pw.astype(np.float64) * ewma_ka.astype(np.float64)
        use_recovery = route_recovery & (cand_proxy <= ewma_proxy + EPS)
    prewarm = cur_pw.copy()
    keepalive = cur_ka.copy()
    prewarm[use_recovery] = cand_pw[use_recovery]
    keepalive[use_recovery] = cand_ka[use_recovery]
    prewarm[route_recovery & ~use_recovery] = ewma_pw[route_recovery & ~use_recovery]
    keepalive[route_recovery & ~use_recovery] = ewma_ka[route_recovery & ~use_recovery]
    return prewarm.astype(np.int32), keepalive.astype(np.float32), use_recovery


def load_trainer(features_shape):
    import torch

    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = ANILMetaTrainer(
        body_type="tcn",
        head_type="ridge",
        in_features=features_shape[2],
        embedding_dim=64,
        n_quantiles=N_QUANTILES,
        n_horizons=1,
        device=device,
    )
    ckpt = RUNS / "best_anil_ridge_s1_s0.pt"
    trainer.load(ckpt)
    return trainer, ckpt


def build_rate_context(counts, feats, age, trainer, pm):
    ewma = rates_ewma(counts)
    faithful = rates_a5_faithful(counts.astype(np.float32), feats, trainer, pm)
    proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
    current, zero, warm, young, mature = current_gate_rates(proto, ewma, faithful, cum_prev, age)
    candidate, info = drift_recovery_matrix(counts, proto, ewma, faithful, cum_prev, age)
    route_current = young
    return {
        "ewma": ewma,
        "faithful": faithful,
        "proto": proto,
        "cum_prev": cum_prev,
        "current": current,
        "candidate": candidate,
        "route_recovery": info["route_recovery"],
        "route_young": info["route_young"],
        "fires": info["fires"],
        "recovery_entries": info["recovery_entries"],
        "recovery_ticks": info["recovery_ticks"],
        "route_current": route_current,
        "zero": zero,
        "warm": warm,
        "mature": mature,
    }


def export_jobs(split, counts, feats, dm, ds, age, weights, rhos, seeds, jobs_root, trainer, pm):
    t0 = time.time()
    ctx = build_rate_context(counts, feats, age, trainer, pm)
    jdir = jobs_root / split
    jdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        jdir / "shared.npz",
        counts=counts.astype(np.int64),
        dur_means=dm,
        dur_stds=ds,
    )
    job_meta = []
    route_meta = {
        "zero_history": weighted_share(ctx["zero"], weights),
        "pre_count_threshold": weighted_share(ctx["warm"], weights),
        "count_qualified": weighted_share(ctx["cum_prev"] >= MIN_INV, weights),
        "aq720_learned": weighted_share(ctx["route_current"], weights),
        "recovery_candidate": weighted_share(ctx["route_recovery"], weights),
        "mean_fires_per_function": float(np.mean(ctx["fires"])),
        "mean_recovery_entries_per_function": float(np.mean(ctx["recovery_entries"])),
        "mean_recovery_ticks_per_function": float(np.mean(ctx["recovery_ticks"])),
    }
    for rho in rhos:
        tau = newsvendor_quantile(float(rho))
        arms = {
            "B4a_ewma": decisions_from_rates(ctx["ewma"], tau),
            "G_WE_A720_current": decisions_from_rates(ctx["current"], tau),
        }
        for method, guard in [("G_WE_A720D", True), ("G_WE_A720D_NG", False)]:
            pw, ka, use = guarded_decisions(
                current_rates=ctx["current"],
                ewma=ctx["ewma"],
                candidate_rates=ctx["candidate"],
                route_recovery=ctx["route_recovery"],
                rho=float(rho),
                guard=guard,
            )
            arms[method] = (pw, ka)
            route_meta[f"{method}_served_recovery_rho{rho:g}"] = weighted_share(use, weights)
            route_meta[f"{method}_guard_blocked_rho{rho:g}"] = weighted_share(ctx["route_recovery"] & ~use, weights)
        for method, (pw, ka) in arms.items():
            fname = f"{method}__rho{rho:g}.npz"
            np.savez_compressed(jdir / fname, prewarm=pw.astype(np.int32), keepalive=ka.astype(np.float32))
            job_meta.append({"method": method, "rho": float(rho), "file": fname})
    with open(jdir / "jobs.json", "w") as f:
        json.dump({"split": split, "seeds": seeds, "cold_init": COLD_INIT, "jobs": job_meta}, f, indent=1)
    return {
        "weights": weights.tolist(),
        "pool_size": int(len(weights)),
        "invocations": int(counts.sum()),
        "n_jobs": len(job_meta),
        "routing": route_meta,
        "elapsed_sec": time.time() - t0,
    }


def export_azure2021(args):
    features = np.load(P21 / "features.npy")
    counts_all = np.load(P21 / "counts.npy")
    splits = np.load(P21 / "splits.npz")
    dur_df = pd.read_csv(P21 / "duration_stats.csv")
    trainer, ckpt = load_trainer(features.shape)
    pm = build_prototypes(trainer, features, counts_all, splits["s1_train"])
    rhos = selected_rhos(args)
    seeds = selected_seeds(args)
    jobs_root = Path(args.jobs_root) if args.jobs_root else RUNS / "des_jobs_r26_azure2021"
    out = Path(args.manifest_out) if args.manifest_out else RUNS / "revision_r26_azure2021_manifest.json"
    manifest = {
        "analysis": "R26 steady false-positive drift-gate audit",
        "trace": "azure2021",
        "checkpoint": str(ckpt),
        "checkpoint_md5": hashlib.md5(open(ckpt, "rb").read()).hexdigest(),
        "rhos": rhos,
        "seeds": seeds,
        "splits": {},
    }
    for split in args.splits.split(","):
        key = {"S1": "s1_test", "S2": "s2_test", "S3": "s3_test"}[split]
        rows = np.asarray(splits[key], dtype=np.int64)
        if args.pilot:
            rows = rows[: args.pilot]
        counts = counts_all[rows].astype(np.float32)
        feats = features[rows].astype(np.float32)
        if split == "S3":
            ts = int(splits.get("s3_test_t_start", [10080])[0])
            counts = counts[:, ts:]
            feats = feats[:, ts:, :]
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        age = age_matrix_from_counts(counts)
        weights = np.ones(len(rows), dtype=np.float64)
        print(f"### Azure2021 {split}: {len(rows)} funcs, {counts.shape[1]} ticks")
        manifest["splits"][split] = export_jobs(split, counts, feats, dm, ds, age, weights, rhos, seeds, jobs_root, trainer, pm)
        with open(out, "w") as f:
            json.dump(manifest, f, indent=1)
    print(f"wrote {out}")


def export_azure2019(args):
    features_mm = np.load(P19 / "features.npy", mmap_mode="r")
    counts_mm = np.load(P19 / "counts.npy", mmap_mode="r")
    splits = np.load(P19 / "splits.npz")
    dur_df = pd.read_csv(P19 / "duration_stats.csv")
    sample_manifest = json.load(open(RUNS / "des_sample_manifest_2019.json"))
    feat21 = np.load(P21 / "features.npy")
    cnt21 = np.load(P21 / "counts.npy")
    spl21 = np.load(P21 / "splits.npz")
    trainer, ckpt = load_trainer(features_mm.shape)
    pm = build_prototypes(trainer, feat21, cnt21, spl21["s1_train"])
    rhos = selected_rhos(args)
    seeds = selected_seeds(args)
    jobs_root = Path(args.jobs_root) if args.jobs_root else RUNS / "des_jobs_r26_azure2019"
    out = Path(args.manifest_out) if args.manifest_out else RUNS / "revision_r26_azure2019_manifest.json"
    manifest = {
        "analysis": "R26 steady false-positive drift-gate audit",
        "trace": "azure2019",
        "checkpoint": str(ckpt),
        "checkpoint_md5": hashlib.md5(open(ckpt, "rb").read()).hexdigest(),
        "rhos": rhos,
        "seeds": seeds,
        "splits": {},
    }
    for split in args.splits.split(","):
        rows = np.asarray(sample_manifest[split]["rows"], dtype=np.int64)
        weights = np.asarray(sample_manifest[split]["weights"], dtype=np.float64)
        if args.pilot:
            rows = rows[-args.pilot :]
            weights = weights[-args.pilot :]
        counts = np.asarray(counts_mm[rows]).astype(np.float32)
        feats = np.asarray(features_mm[rows], dtype=np.float32)
        if split == "S3":
            ts = int(splits.get("s3_test_t_start", [14400])[0])
            counts = counts[:, ts:]
            feats = feats[:, ts:, :]
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        age = age_matrix_from_counts(counts)
        print(f"### Azure2019 {split}: {len(rows)} funcs, {counts.shape[1]} ticks")
        manifest["splits"][split] = export_jobs(split, counts, feats, dm, ds, age, weights, rhos, seeds, jobs_root, trainer, pm)
        with open(out, "w") as f:
            json.dump(manifest, f, indent=1)
    print(f"wrote {out}")


def export_huawei(args):
    features_mm = np.load(PHW / "features.npy", mmap_mode="r")
    counts_mm = np.load(PHW / "counts.npy", mmap_mode="r")
    splits = np.load(PHW / "splits.npz")
    first_present = np.load(PHW / "first_present.npy")
    steady_t = int(splits["steady_t"][0])
    dur_df = pd.read_csv(PHW / "duration_stats.csv")
    feat21 = np.load(P21 / "features.npy")
    cnt21 = np.load(P21 / "counts.npy")
    spl21 = np.load(P21 / "splits.npz")
    trainer, ckpt = load_trainer(features_mm.shape)
    pm = build_prototypes(trainer, feat21, cnt21, spl21["s1_train"])
    rhos = selected_rhos(args)
    seeds = selected_seeds(args)
    jobs_root = Path(args.jobs_root) if args.jobs_root else RUNS / "des_jobs_r26_huawei"
    out = Path(args.manifest_out) if args.manifest_out else RUNS / "revision_r26_huawei_manifest.json"
    manifest = {
        "analysis": "R26 steady false-positive drift-gate audit",
        "trace": "huawei2023",
        "checkpoint": str(ckpt),
        "checkpoint_md5": hashlib.md5(open(ckpt, "rb").read()).hexdigest(),
        "rhos": rhos,
        "seeds": seeds,
        "splits": {},
    }
    for pool in args.pools.split(","):
        rows = np.asarray(splits[pool], dtype=np.int64)
        if args.pilot:
            rows = rows[-args.pilot :]
        counts = np.asarray(counts_mm[rows][:, :steady_t]).astype(np.float32)
        feats = np.asarray(features_mm[rows][:, :steady_t, :], dtype=np.float32)
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        ticks = np.arange(counts.shape[1])[None, :]
        age = ticks - first_present[rows].astype(np.int64)[:, None]
        weights = np.ones(len(rows), dtype=np.float64)
        print(f"### Huawei {pool}: {len(rows)} funcs, {counts.shape[1]} ticks")
        manifest["splits"][pool] = export_jobs(pool, counts, feats, dm, ds, age, weights, rhos, seeds, jobs_root, trainer, pm)
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


def summarize(args):
    rows = load_rows(args.inputs)
    manifests = load_manifests(args.manifests)
    methods = ["B4a_ewma", "G_WE_A720_current", "G_WE_A720D", "G_WE_A720D_NG"]
    splits = sorted({r["split"] for r in rows})
    rhos = sorted({float(r["cost_ratio"]) for r in rows})
    header = (
        "split,method,rho,n_seeds,n_functions,csr_pct,csr_seed_std_pct,"
        "wm_per_1k_inv,cost_per_1k_inv,delta_csr_pp_vs_ewma,"
        "delta_wm_per_1k_vs_ewma,delta_cost_per_1k_vs_ewma,"
        "boot_ci95_lo_pp,boot_ci95_hi_pp,route_recovery_share,"
        "served_recovery_share,guard_blocked_share,source_json"
    )
    lines = [header]
    rho10 = [header]
    readout = {"analysis": "R26 steady false-positive drift-gate audit", "cells": {}}
    for split in splits:
        if split not in manifests:
            raise SystemExit(f"missing manifest metadata for split {split}")
        weights = np.asarray(manifests[split]["weights"], dtype=np.float64)
        routing = manifests[split].get("routing", {})
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
                dcsr = rec["csr_pct"] - ref["csr_pct"] if ref else np.nan
                dwm = rec["wm_per_1k_inv"] - ref["wm_per_1k_inv"] if ref else np.nan
                dcost = rec["cost_per_1k_inv"] - ref["cost_per_1k_inv"] if ref else np.nan
                lo, hi = ("", "")
                if method != "B4a_ewma" and ref_rows:
                    lo, hi = paired_boot_ci(cell_rows, ref_rows, weights)
                source = ";".join(sorted({r["_source_json"] for r in cell_rows}))
                served = routing.get(f"{method}_served_recovery_rho{rho:g}", "")
                blocked = routing.get(f"{method}_guard_blocked_rho{rho:g}", "")
                line = (
                    f"{split},{method},{rho:g},{rec['n']},{rec['n_functions']},"
                    f"{rec['csr_pct']:.6f},{rec['csr_seed_std_pct']:.6f},"
                    f"{rec['wm_per_1k_inv']:.3f},{rec['cost_per_1k_inv']:.3f},"
                    f"{dcsr:.6f},{dwm:.3f},{dcost:.3f},{lo},{hi},"
                    f"{routing.get('recovery_candidate', '')},{served},{blocked},{source}"
                )
                lines.append(line)
                if abs(rho - 10.0) < 1e-9:
                    rho10.append(line)
                readout["cells"][f"{split}|{rho:g}|{method}"] = rec | {
                    "delta_csr_pp_vs_ewma": float(dcsr),
                    "delta_wm_per_1k_vs_ewma": float(dwm),
                    "delta_cost_per_1k_vs_ewma": float(dcost),
                    "route_recovery_share": routing.get("recovery_candidate", ""),
                    "served_recovery_share": served,
                    "guard_blocked_share": blocked,
                    "source_json": source,
                }
    TABLES.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    (TABLES / f"{prefix}_fullrho.csv").write_text("\n".join(lines) + "\n")
    (TABLES / f"{prefix}_rho10.csv").write_text("\n".join(rho10) + "\n")
    with open(RUNS / f"{prefix}.json", "w") as f:
        json.dump(readout, f, indent=1, default=float)
    print(f"wrote {TABLES / f'{prefix}_fullrho.csv'}")
    print(f"wrote {TABLES / f'{prefix}_rho10.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--trace", choices=["azure2021", "azure2019", "huawei"], default="azure2021")
    ap.add_argument("--splits", default="S1,S2,S3")
    ap.add_argument("--pools", default="h_mixed,h_sparse,h_saturated")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--rhos", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--jobs-root", default="")
    ap.add_argument("--manifest-out", default="")
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--manifests", nargs="*", default=[])
    ap.add_argument("--prefix", default="T_r26_steady_drift_gate_audit")
    args = ap.parse_args()
    if args.export:
        if args.trace == "azure2021":
            export_azure2021(args)
        elif args.trace == "azure2019":
            export_azure2019(args)
        else:
            export_huawei(args)
    if args.summarize:
        summarize(args)


if __name__ == "__main__":
    main()
