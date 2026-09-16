#!/usr/bin/env python3
"""R33: adaptive-EWMA drift recovery audit.

This experiment is intentionally separate from the WINTER reset-to-prior
pipeline.  It asks whether a mature-function drift alarm is better handled by
temporarily increasing the recency weight of EWMA, without invoking a learned
meta-prior.  The exporter is CPU-only and only reads count traces.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import COLD_INIT, decisions_from_rates, rates_ewma  # noqa: E402
from scripts.revision_r21_cross_provider_faithful import (  # noqa: E402
    KAPPA,
    paired_boot_ci,
    parse_float_list,
    parse_int_list,
    summarize_cell,
    weighted_share,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.drift.triggers import ConformalMonitor  # noqa: E402


RUNS = PROJECT_ROOT / "results" / "runs"
TABLES = PROJECT_ROOT / "results" / "tables"
P21 = PROJECT_ROOT / "data" / "processed"
P19 = PROJECT_ROOT / "data" / "processed_2019"
PHW = PROJECT_ROOT / "data" / "processed_huawei"

MIN_INV = 100
AGE_MIN = 720
DEFAULT_RHOS = [0.1, 1.0, 10.0, 100.0]
DEFAULT_SEEDS_AZ21 = list(range(10))
DEFAULT_SEEDS_XTRACE = [0, 1, 2]
MAX_RATE = 200.0
MAX_LOG_RATE = float(np.log1p(MAX_RATE))


@dataclass(frozen=True)
class EWMASpec:
    name: str
    mode: str = "adaptive"  # adaptive, decay, static
    trigger_mode: str = "signed_under"  # signed_under, abs
    alpha_base: float = 0.1
    alpha_fast: float = 0.3
    ttl: int = 60
    cooldown: int = 3600
    trigger_alpha: float = 0.1
    trigger_delta: float = 0.25
    trigger_m: int = 3
    trigger_calib_size: int = 120
    trigger_window: int = 30
    trigger_min_calib: int = 30
    min_under_log: float = 0.0


def default_candidates() -> List[EWMASpec]:
    return [
        EWMASpec(name="EWMA_a0p3_static", mode="static", alpha_fast=0.3),
        EWMASpec(name="EWMA_a0p5_static", mode="static", alpha_fast=0.5),
        EWMASpec(name="AE_U03_T60", trigger_mode="signed_under", alpha_fast=0.3, ttl=60),
        EWMASpec(name="AE_U05_T60", trigger_mode="signed_under", alpha_fast=0.5, ttl=60),
        EWMASpec(name="AE_U05_D120", mode="decay", trigger_mode="signed_under", alpha_fast=0.5, ttl=120),
        EWMASpec(name="AE_A03_T60", trigger_mode="abs", alpha_fast=0.3, ttl=60),
        EWMASpec(name="AE_A05_T60", trigger_mode="abs", alpha_fast=0.5, ttl=60),
        EWMASpec(name="AE_A05_D120", mode="decay", trigger_mode="abs", alpha_fast=0.5, ttl=120),
    ]


def parse_candidate(raw: str) -> EWMASpec:
    vals: Dict[str, object] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"candidate component lacks '=': {part}")
        key, value = [x.strip() for x in part.split("=", 1)]
        vals[key] = value
    if "name" not in vals:
        raise ValueError("candidate requires name=")

    int_keys = {"ttl", "cooldown", "trigger_m", "trigger_calib_size", "trigger_window", "trigger_min_calib"}
    float_keys = {"alpha_base", "alpha_fast", "trigger_alpha", "trigger_delta", "min_under_log"}
    for key in list(vals):
        if key in int_keys:
            vals[key] = int(vals[key])
        elif key in float_keys:
            vals[key] = float(vals[key])
    return EWMASpec(**vals)  # type: ignore[arg-type]


def selected_rhos(args) -> List[float]:
    return parse_float_list(args.rhos) if args.rhos else DEFAULT_RHOS


def selected_seeds(args) -> List[int]:
    if args.seeds:
        return parse_int_list(args.seeds)
    return DEFAULT_SEEDS_AZ21 if args.trace == "azure2021" else DEFAULT_SEEDS_XTRACE


def slug_float(x: float) -> str:
    return ("%g" % float(x)).replace(".", "p").replace("-", "m")


def cum_prev_matrix(counts: np.ndarray) -> np.ndarray:
    cum = np.cumsum(counts.astype(np.int64), axis=1)
    return np.concatenate([np.zeros((counts.shape[0], 1), dtype=np.int64), cum[:, :-1]], axis=1)


def age_matrix_from_counts(counts: np.ndarray) -> np.ndarray:
    n_funcs, t_len = counts.shape
    out = np.zeros((n_funcs, t_len), dtype=np.int64)
    for f in range(n_funcs):
        nz = np.flatnonzero(counts[f] > 0)
        if len(nz):
            t0 = int(nz[0])
            out[f, t0:] = np.arange(t_len - t0, dtype=np.int64)
    return out


def trigger_score(mode: str, y_log: float, pred_log: float, min_under_log: float) -> float:
    if mode == "abs":
        return abs(y_log - pred_log)
    if mode == "signed_under":
        return max(0.0, y_log - pred_log - float(min_under_log))
    raise ValueError(f"unknown trigger_mode: {mode}")


def new_monitor(spec: EWMASpec) -> ConformalMonitor:
    return ConformalMonitor(
        alpha=float(spec.trigger_alpha),
        delta=float(spec.trigger_delta),
        m_consecutive=int(spec.trigger_m),
        calib_size=int(spec.trigger_calib_size),
        window=int(spec.trigger_window),
        min_calib=int(spec.trigger_min_calib),
    )


def alpha_for_tick(spec: EWMASpec, t: int, active_start: Optional[int], active_until: int) -> float:
    if spec.mode == "static":
        return float(spec.alpha_fast)
    if active_start is None or t >= active_until:
        return float(spec.alpha_base)
    if spec.mode == "decay":
        span = max(1, int(spec.ttl))
        rel = min(max(0, t - int(active_start)), span)
        frac = 1.0 - rel / span
        return float(spec.alpha_base + (spec.alpha_fast - spec.alpha_base) * frac)
    if spec.mode == "adaptive":
        return float(spec.alpha_fast)
    raise ValueError(f"unknown mode: {spec.mode}")


def adaptive_ewma_rates(
    counts: np.ndarray,
    age: np.ndarray,
    spec: EWMASpec,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if spec.mode == "static":
        rates = rates_ewma(counts, alpha=float(spec.alpha_fast))
        z = np.zeros(counts.shape, dtype=bool)
        return rates, {
            "route_fast_ewma": z,
            "route_base_ewma": ~z,
            "fires": np.zeros(counts.shape[0], dtype=np.int64),
            "suppressed_by_cooldown": np.zeros(counts.shape[0], dtype=np.int64),
            "entries": np.zeros(counts.shape[0], dtype=np.int64),
        }

    n_funcs, t_len = counts.shape
    y = np.log1p(counts.astype(np.float64))
    cum_prev = cum_prev_matrix(counts)
    rates = np.zeros((n_funcs, t_len), dtype=np.float32)
    route_fast = np.zeros((n_funcs, t_len), dtype=bool)
    fires = np.zeros(n_funcs, dtype=np.int64)
    suppressed = np.zeros(n_funcs, dtype=np.int64)
    entries = np.zeros(n_funcs, dtype=np.int64)

    for f in range(n_funcs):
        state = 0.0
        monitor = new_monitor(spec)
        active_start: Optional[int] = None
        active_until = -1
        last_entry = -10**9
        for t in range(t_len):
            active = active_start is not None and t < active_until
            rates[f, t] = float(np.expm1(np.clip(state, 0.0, MAX_LOG_RATE)))
            if active:
                route_fast[f, t] = True

            mature = bool(cum_prev[f, t] >= MIN_INV and age[f, t] >= AGE_MIN)
            if mature:
                pred_log = math.log1p(max(float(rates[f, t]), 0.0))
                resid = trigger_score(
                    spec.trigger_mode,
                    y_log=float(y[f, t]),
                    pred_log=pred_log,
                    min_under_log=float(spec.min_under_log),
                )
                if monitor.update(resid):
                    fires[f] += 1
                    if (t + 1 - last_entry) >= int(spec.cooldown):
                        active_start = t + 1
                        active_until = min(t_len, t + 1 + int(spec.ttl))
                        last_entry = t + 1
                        entries[f] += 1
                    else:
                        suppressed[f] += 1
                    monitor = new_monitor(spec)

            update_active = active_start is not None and (t + 1) < active_until
            alpha = alpha_for_tick(spec, t + 1, active_start, active_until) if update_active else float(spec.alpha_base)
            state = alpha * float(y[f, t]) + (1.0 - alpha) * state

    return rates, {
        "route_fast_ewma": route_fast,
        "route_base_ewma": ~route_fast,
        "fires": fires,
        "suppressed_by_cooldown": suppressed,
        "entries": entries,
    }


def info_record(info: Dict[str, object], weights: np.ndarray) -> Dict[str, float]:
    return {
        "route_fast_ewma": weighted_share(info["route_fast_ewma"], weights),
        "route_base_ewma": weighted_share(info["route_base_ewma"], weights),
        "mean_fires_per_function": float(np.average(np.asarray(info["fires"], dtype=np.float64), weights=weights)),
        "mean_suppressed_by_cooldown_per_function": float(
            np.average(np.asarray(info["suppressed_by_cooldown"], dtype=np.float64), weights=weights)
        ),
        "mean_entries_per_function": float(np.average(np.asarray(info["entries"], dtype=np.float64), weights=weights)),
    }


def base_routing_record(counts: np.ndarray, age: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    cum_prev = cum_prev_matrix(counts)
    zero = cum_prev == 0
    count_insufficient = (cum_prev > 0) & (cum_prev < MIN_INV)
    mature = (cum_prev >= MIN_INV) & (age >= AGE_MIN)
    young_qualified = (cum_prev >= MIN_INV) & (age < AGE_MIN)
    return {
        "zero_history": weighted_share(zero, weights),
        "count_insufficient_ewma": weighted_share(count_insufficient, weights),
        "count_qualified_age_young": weighted_share(young_qualified, weights),
        "gate_mature_ewma": weighted_share(mature, weights),
    }


def export_split(
    *,
    trace: str,
    split: str,
    rows: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    dm: np.ndarray,
    ds: np.ndarray,
    age: np.ndarray,
    candidates: Sequence[EWMASpec],
    rhos: Sequence[float],
    seeds: Sequence[int],
    jobs_root: Path,
    manifest: Dict[str, object],
) -> None:
    print(
        f"### R33 {trace} {split}: {len(rows)} funcs, {counts.shape[1]} ticks, "
        f"{int(counts.sum()):,} invocations, candidates={len(candidates)}",
        flush=True,
    )
    t0 = time.time()
    rate_mats: Dict[str, np.ndarray] = {"B4a_ewma": rates_ewma(counts, alpha=0.1)}
    routing: Dict[str, object] = base_routing_record(counts, age, weights)
    for spec in candidates:
        print(f"    candidate {spec.name}", flush=True)
        rates, info = adaptive_ewma_rates(counts, age, spec)
        rate_mats[spec.name] = rates
        routing[spec.name] = info_record(info, weights)

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
            tau = newsvendor_quantile(float(rho))
            pw, ka = decisions_from_rates(rates, tau)
            fname = f"{method}__rho{slug_float(float(rho))}.npz"
            np.savez_compressed(jdir / fname, prewarm=pw.astype(np.int32), keepalive=ka.astype(np.float32))
            job_meta.append({"method": method, "rho": float(rho), "file": fname})
    with open(jdir / "jobs.json", "w") as f:
        json.dump({"split": split, "seeds": list(seeds), "cold_init": COLD_INIT, "jobs": job_meta}, f, indent=1)

    manifest["splits"][split] = {
        "rows": rows.tolist(),
        "weights": weights.tolist(),
        "pool_size": int(manifest.pop("_pool_size_override", len(rows))),
        "invocations": int(counts.sum()),
        "n_jobs": len(job_meta),
        "routing": routing,
        "elapsed_sec": time.time() - t0,
    }
    with open(manifest["_out_path"], "w") as f:
        json.dump({k: v for k, v in manifest.items() if not str(k).startswith("_")}, f, indent=1)
    print(f"    exported {len(job_meta)} jobs in {time.time() - t0:.1f}s -> {jdir}", flush=True)


def candidate_list(args) -> List[EWMASpec]:
    candidates = [] if args.no_defaults else default_candidates()
    candidates.extend(parse_candidate(raw) for raw in args.candidate)
    names = [c.name for c in candidates]
    if len(names) != len(set(names)):
        raise ValueError("duplicate candidate names")
    return candidates


def export_azure2021(args) -> None:
    counts_all = np.load(P21 / "counts.npy")
    splits = np.load(P21 / "splits.npz")
    dur_df = pd.read_csv(P21 / "duration_stats.csv")
    candidates = candidate_list(args)
    rhos = selected_rhos(args)
    seeds = selected_seeds(args)
    jobs_root = Path(args.jobs_root) if args.jobs_root else RUNS / "des_jobs_r33_adaptive_ewma_azure2021"
    out = Path(args.manifest_out) if args.manifest_out else RUNS / "revision_r33_adaptive_ewma_azure2021_manifest.json"
    manifest: Dict[str, object] = {
        "analysis": "R33 adaptive-EWMA drift recovery audit",
        "trace": "azure2021",
        "min_invocations": MIN_INV,
        "age_min": AGE_MIN,
        "candidates": [asdict(c) for c in candidates],
        "rhos": rhos,
        "seeds": seeds,
        "splits": {},
        "_out_path": str(out),
    }
    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        key = {"S1": "s1_test", "S2": "s2_test", "S3": "s3_test"}[split]
        rows = np.asarray(splits[key], dtype=np.int64)
        if args.pilot:
            rows = rows[: int(args.pilot)]
        counts = counts_all[rows].astype(np.float32)
        if split == "S3":
            ts = int(splits.get("s3_test_t_start", [10080])[0])
            counts = counts[:, ts:]
        weights = np.ones(len(rows), dtype=np.float64)
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        export_split(
            trace="azure2021",
            split=split,
            rows=rows,
            weights=weights,
            counts=counts,
            dm=dm,
            ds=ds,
            age=age_matrix_from_counts(counts),
            candidates=candidates,
            rhos=rhos,
            seeds=seeds,
            jobs_root=jobs_root,
            manifest=manifest,
        )
    with open(out, "w") as f:
        json.dump({k: v for k, v in manifest.items() if not str(k).startswith("_")}, f, indent=1)
    print(f"wrote {out}", flush=True)


def export_azure2019(args) -> None:
    counts_mm = np.load(P19 / "counts.npy", mmap_mode="r")
    splits = np.load(P19 / "splits.npz")
    dur_df = pd.read_csv(P19 / "duration_stats.csv")
    sample_manifest = json.load(open(RUNS / "des_sample_manifest_2019.json"))
    candidates = candidate_list(args)
    rhos = selected_rhos(args)
    seeds = selected_seeds(args)
    jobs_root = Path(args.jobs_root) if args.jobs_root else RUNS / "des_jobs_r33_adaptive_ewma_azure2019"
    out = Path(args.manifest_out) if args.manifest_out else RUNS / "revision_r33_adaptive_ewma_azure2019_manifest.json"
    manifest: Dict[str, object] = {
        "analysis": "R33 adaptive-EWMA drift recovery audit",
        "trace": "azure2019",
        "sample_manifest": str(RUNS / "des_sample_manifest_2019.json"),
        "min_invocations": MIN_INV,
        "age_min": AGE_MIN,
        "candidates": [asdict(c) for c in candidates],
        "rhos": rhos,
        "seeds": seeds,
        "splits": {},
        "_out_path": str(out),
    }
    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        rows = np.asarray(sample_manifest[split]["rows"], dtype=np.int64)
        weights = np.asarray(sample_manifest[split]["weights"], dtype=np.float64)
        if args.pilot:
            rows = rows[-int(args.pilot) :]
            weights = weights[-int(args.pilot) :]
        counts = np.asarray(counts_mm[rows]).astype(np.float32)
        if split == "S3":
            ts = int(splits.get("s3_test_t_start", [14400])[0])
            counts = counts[:, ts:]
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        manifest["_pool_size_override"] = int(sample_manifest[split]["pool_size"])
        export_split(
            trace="azure2019",
            split=split,
            rows=rows,
            weights=weights,
            counts=counts,
            dm=dm,
            ds=ds,
            age=age_matrix_from_counts(counts),
            candidates=candidates,
            rhos=rhos,
            seeds=seeds,
            jobs_root=jobs_root,
            manifest=manifest,
        )
    with open(out, "w") as f:
        json.dump({k: v for k, v in manifest.items() if not str(k).startswith("_")}, f, indent=1)
    print(f"wrote {out}", flush=True)


def export_huawei(args) -> None:
    counts_mm = np.load(PHW / "counts.npy", mmap_mode="r")
    splits = np.load(PHW / "splits.npz")
    first_present = np.load(PHW / "first_present.npy")
    steady_t = int(splits["steady_t"][0])
    dur_df = pd.read_csv(PHW / "duration_stats.csv")
    candidates = candidate_list(args)
    rhos = selected_rhos(args)
    seeds = selected_seeds(args)
    jobs_root = Path(args.jobs_root) if args.jobs_root else RUNS / "des_jobs_r33_adaptive_ewma_huawei"
    out = Path(args.manifest_out) if args.manifest_out else RUNS / "revision_r33_adaptive_ewma_huawei_manifest.json"
    manifest: Dict[str, object] = {
        "analysis": "R33 adaptive-EWMA drift recovery audit",
        "trace": "huawei",
        "age_source": "first_present.npy deployment-presence mask",
        "min_invocations": MIN_INV,
        "age_min": AGE_MIN,
        "candidates": [asdict(c) for c in candidates],
        "rhos": rhos,
        "seeds": seeds,
        "splits": {},
        "_out_path": str(out),
    }
    for pool in args.pools.split(","):
        pool = pool.strip()
        if not pool:
            continue
        rows = np.asarray(splits[pool], dtype=np.int64)
        if args.pilot:
            rows = rows[-int(args.pilot) :]
        counts = np.asarray(counts_mm[rows][:, :steady_t]).astype(np.float32)
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        fp = first_present[rows].astype(np.int64)
        ticks = np.arange(counts.shape[1], dtype=np.int64)[None, :]
        age = ticks - fp[:, None]
        weights = np.ones(len(rows), dtype=np.float64)
        manifest["_pool_size_override"] = int(len(splits[pool]))
        export_split(
            trace="huawei",
            split=pool,
            rows=rows,
            weights=weights,
            counts=counts,
            dm=dm,
            ds=ds,
            age=age,
            candidates=candidates,
            rhos=rhos,
            seeds=seeds,
            jobs_root=jobs_root,
            manifest=manifest,
        )
    with open(out, "w") as f:
        json.dump({k: v for k, v in manifest.items() if not str(k).startswith("_")}, f, indent=1)
    print(f"wrote {out}", flush=True)


def load_rows(paths: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for raw in paths:
        path = Path(raw)
        data = json.load(open(path))
        part = data.get("results", data) if isinstance(data, dict) else data
        for row in part:
            rec = dict(row)
            rec["_source_json"] = str(path)
            rows.append(rec)
        print(f"loaded {len(part)} rows from {path}", flush=True)
    return rows


def method_route(manifest: Dict[str, object], split: str, method: str) -> Dict[str, object]:
    routing = manifest["splits"][split].get("routing", {})
    out = {
        "route_zero_history": routing.get("zero_history", ""),
        "route_count_insufficient_ewma": routing.get("count_insufficient_ewma", ""),
        "route_count_qualified_age_young": routing.get("count_qualified_age_young", ""),
        "route_gate_mature_ewma": routing.get("gate_mature_ewma", ""),
        "route_fast_ewma": "",
        "route_base_ewma": "",
        "mean_fires_per_function": "",
        "mean_suppressed_by_cooldown_per_function": "",
        "mean_entries_per_function": "",
    }
    rec = routing.get(method, {})
    if isinstance(rec, dict):
        for key in rec:
            out[key] = rec[key]
    return out


def candidate_meta(manifest: Dict[str, object], method: str) -> Dict[str, object]:
    for rec in manifest.get("candidates", []):
        if rec.get("name") == method:
            return rec
    return {}


def summarize(args) -> None:
    rows = load_rows(args.inputs)
    manifest = json.load(open(args.manifest))
    methods = ["B4a_ewma"] + [str(c["name"]) for c in manifest.get("candidates", [])]
    splits = sorted({r["split"] for r in rows})
    rhos = sorted({float(r["cost_ratio"]) for r in rows})
    header = (
        "split,method,rho,n_seeds,n_functions,csr_pct,csr_seed_std_pct,"
        "wm_per_1k_inv,cost_per_1k_inv,delta_csr_pp_vs_ewma,"
        "delta_wm_per_1k_vs_ewma,delta_cost_per_1k_vs_ewma,"
        "delta_cost_pct_vs_ewma,boot_ci95_lo_pp_vs_ewma,"
        "boot_ci95_hi_pp_vs_ewma,mode,trigger_mode,alpha_base,"
        "alpha_fast,ttl,cooldown,route_zero_history,"
        "route_count_insufficient_ewma,route_count_qualified_age_young,"
        "route_gate_mature_ewma,route_fast_ewma,route_base_ewma,"
        "mean_fires_per_function,mean_suppressed_by_cooldown_per_function,"
        "mean_entries_per_function,source_json"
    )
    lines = [header]
    rho10 = [header]
    readout = {
        "analysis": manifest.get("analysis", "R33 adaptive-EWMA drift recovery audit"),
        "trace": manifest.get("trace", ""),
        "cells": {},
    }
    for split in splits:
        weights = np.asarray(manifest["splits"][split]["weights"], dtype=np.float64)
        for rho in rhos:
            ref_rows = [
                r for r in rows
                if r["split"] == split
                and r["method"] == "B4a_ewma"
                and abs(float(r["cost_ratio"]) - rho) < 1e-9
            ]
            ref = summarize_cell(ref_rows, weights)
            for method in methods:
                cell_rows = [
                    r for r in rows
                    if r["split"] == split
                    and r["method"] == method
                    and abs(float(r["cost_ratio"]) - rho) < 1e-9
                ]
                rec = summarize_cell(cell_rows, weights)
                if rec is None:
                    continue
                dcsr = rec["csr_pct"] - ref["csr_pct"] if ref else np.nan
                dwm = rec["wm_per_1k_inv"] - ref["wm_per_1k_inv"] if ref else np.nan
                dcost = rec["cost_per_1k_inv"] - ref["cost_per_1k_inv"] if ref else np.nan
                dcost_pct = 100.0 * dcost / max(ref["cost_per_1k_inv"], 1e-9) if ref else np.nan
                lo, hi = ("", "")
                if method != "B4a_ewma" and ref_rows:
                    lo, hi = paired_boot_ci(cell_rows, ref_rows, weights)
                meta = candidate_meta(manifest, method)
                route = method_route(manifest, split, method)
                source = ";".join(sorted({r["_source_json"] for r in cell_rows}))
                vals = [
                    split,
                    method,
                    "%g" % rho,
                    rec["n"],
                    rec["n_functions"],
                    f"{rec['csr_pct']:.6f}",
                    f"{rec['csr_seed_std_pct']:.6f}",
                    f"{rec['wm_per_1k_inv']:.3f}",
                    f"{rec['cost_per_1k_inv']:.3f}",
                    f"{dcsr:.6f}",
                    f"{dwm:.3f}",
                    f"{dcost:.3f}",
                    f"{dcost_pct:.6f}",
                    lo,
                    hi,
                    meta.get("mode", ""),
                    meta.get("trigger_mode", ""),
                    meta.get("alpha_base", ""),
                    meta.get("alpha_fast", ""),
                    meta.get("ttl", ""),
                    meta.get("cooldown", ""),
                    route["route_zero_history"],
                    route["route_count_insufficient_ewma"],
                    route["route_count_qualified_age_young"],
                    route["route_gate_mature_ewma"],
                    route["route_fast_ewma"],
                    route["route_base_ewma"],
                    route["mean_fires_per_function"],
                    route["mean_suppressed_by_cooldown_per_function"],
                    route["mean_entries_per_function"],
                    source,
                ]
                line = ",".join(str(v) for v in vals)
                lines.append(line)
                if abs(rho - 10.0) < 1e-9:
                    rho10.append(line)
                cell = dict(rec)
                cell.update(
                    {
                        "delta_csr_pp_vs_ewma": float(dcsr),
                        "delta_wm_per_1k_vs_ewma": float(dwm),
                        "delta_cost_per_1k_vs_ewma": float(dcost),
                        "delta_cost_pct_vs_ewma": float(dcost_pct),
                        "boot_ci95_pp_vs_ewma": [
                            None if lo == "" else float(lo),
                            None if hi == "" else float(hi),
                        ],
                        "candidate": meta,
                        "routing": route,
                        "source_json": source,
                    }
                )
                readout["cells"][f"{split}|{rho:g}|{method}"] = cell

    TABLES.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    (TABLES / f"{prefix}_fullrho.csv").write_text("\n".join(lines) + "\n")
    (TABLES / f"{prefix}_rho10.csv").write_text("\n".join(rho10) + "\n")
    with open(RUNS / f"{prefix}.json", "w") as f:
        json.dump(readout, f, indent=1, default=float)
    print(f"wrote {TABLES / f'{prefix}_fullrho.csv'}", flush=True)
    print(f"wrote {TABLES / f'{prefix}_rho10.csv'}", flush=True)
    print(f"wrote {RUNS / f'{prefix}.json'}", flush=True)


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--trace", choices=["azure2021", "azure2019", "huawei"], default="azure2021")
    ap.add_argument("--splits", default="S1,S2,S3")
    ap.add_argument("--pools", default="h_mixed,h_sparse,h_saturated")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--no-defaults", action="store_true")
    ap.add_argument("--candidate", action="append", default=[])
    ap.add_argument("--rhos", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--jobs-root", default="")
    ap.add_argument("--manifest-out", default="")
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--manifest", default="")
    ap.add_argument("--prefix", default="T_r33_adaptive_ewma")
    args = ap.parse_args(argv)

    if args.export == args.summarize:
        ap.error("choose exactly one of --export or --summarize")
    if args.export:
        if args.trace == "azure2021":
            export_azure2021(args)
        elif args.trace == "azure2019":
            export_azure2019(args)
        else:
            export_huawei(args)
    else:
        if not args.inputs or not args.manifest:
            ap.error("--summarize requires --inputs and --manifest")
        summarize(args)


if __name__ == "__main__":
    main()
