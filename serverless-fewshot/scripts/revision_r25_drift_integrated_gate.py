#!/usr/bin/env python3
"""R25 integrated drift-aware WINTER-G experiment.

R23 evaluated refitting a learned head after drift. R25 evaluates the
deployable gate transition that R23 left separate:

    mature EWMA -> drift trigger -> WINTER recovery -> mature EWMA

The primary gate, G_WE_A720D, uses the existing 720-minute handoff and adds
Stability+TTL recovery. A guarded version is primary: during recovery it serves
the learned decision only when a per-tick decision proxy is no larger than the
EWMA proxy at the same rho. G_WE_A720D_NG is the unguarded ablation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import revision_r23_drift_v2 as r23  # noqa: E402
from scripts.phase6_des import decisions_from_rates  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.drift.triggers import ConformalMonitor  # noqa: E402


AGE_MIN = 720
MIN_INV = 100
RECOVERY_TTL = 240
STABLE_EXIT = 120
EPS = 1e-9
WINDOWS_DEFAULT = (15, 30, 60, 240)
SYNTHETIC_KINDS = (
    "scale_down_0.5",
    "scale_down_0.2",
    "scale_up_2",
    "scale_up_5",
    "ramp_down_0.2_120",
    "ramp_up_5_120",
    "burst_compress_4",
    "phase_circular_120",
)


def output_paths(tag: str) -> Tuple[Path, Path, Path]:
    suffix = f"_{tag}" if tag else ""
    return (
        ROOT / "results" / "runs" / f"revision_r25_drift_integrated_gate{suffix}.json",
        ROOT / "results" / "tables" / f"T_r25_drift_integrated_gate_events{suffix}.csv",
        ROOT / "results" / "tables" / f"T_r25_drift_integrated_gate_summary{suffix}.csv",
    )


def first_seen(series: np.ndarray) -> int:
    nz = np.flatnonzero(np.asarray(series) > 0)
    return int(nz[0]) if len(nz) else 0


def event_state_arrays(
    counts_all: np.ndarray,
    func_id: int,
    start: int,
    end: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    full = r23.integer_base(np.asarray(counts_all[func_id]))
    first = first_seen(full)
    prior_total = int(full[:start].sum())
    segment = r23.integer_base(full[start:end])
    cum_prev = prior_total + np.concatenate([[0], np.cumsum(segment[:-1])]).astype(np.int64)
    ticks_abs = np.arange(start, end, dtype=np.int64)
    age = np.maximum(0, ticks_abs - int(first)).astype(np.int64)
    return cum_prev, age, prior_total, first


def mature_eligible(counts_all: np.ndarray, ev: r23.Event) -> Tuple[bool, int, int]:
    full = r23.integer_base(np.asarray(counts_all[ev.func_id]))
    first = first_seen(full)
    cum_before = int(full[: ev.onset].sum())
    age_at = max(0, int(ev.onset) - int(first))
    return bool(cum_before >= MIN_INV and age_at >= AGE_MIN), cum_before, age_at


def filter_mature_events(
    counts_all: np.ndarray,
    events: Sequence[r23.Event],
    mature_only: bool,
) -> Tuple[List[r23.Event], Dict[str, int]]:
    kept: List[r23.Event] = []
    mature = 0
    immature = 0
    for ev in events:
        ok, _, _ = mature_eligible(counts_all, ev)
        if ok:
            mature += 1
        else:
            immature += 1
        if ok or not mature_only:
            kept.append(ev)
    return kept, {"input": len(events), "mature": mature, "immature": immature, "kept": len(kept)}


def event_from_dict(raw: Dict[str, object]) -> r23.Event:
    return r23.Event(
        event_id=str(raw["event_id"]),
        source=str(raw["source"]),
        func_id=int(raw["func_id"]),
        onset=int(raw["onset"]),
        kind=str(raw["kind"]),
        pre_sum=int(raw["pre_sum"]),
        post_sum=int(raw["post_sum"]),
        bucket=str(raw["bucket"]),
        meta={str(k): float(v) for k, v in dict(raw.get("meta", {})).items()},
    )


def load_events_json(path: Path) -> Tuple[List[r23.Event], List[r23.Event], List[str]]:
    data = json.load(open(path))
    syn = [event_from_dict(ev) for ev in data.get("synthetic_events", [])]
    nat = [event_from_dict(ev) for ev in data.get("natural_events", [])]
    kinds = [str(k) for k in data.get("synthetic_kinds", SYNTHETIC_KINDS)]
    return syn, nat, kinds


def current_gate_rates(
    phi: np.ndarray,
    counts: np.ndarray,
    cum_prev: np.ndarray,
    age: np.ndarray,
    l2: float,
    refit_every: int,
    buffer_len: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    ewma = r23.ewma_rates(counts, 0.1)
    learned, winfo = r23.winter_rates(
        phi=phi,
        counts=counts,
        event_rel=len(counts) + 1,
        mode="scheduled",
        l2=l2,
        refit_every=refit_every,
        buffer_len=buffer_len,
        trigger_alpha=0.1,
        trigger_warm=30,
        trigger_window=120,
    )
    route = (cum_prev >= MIN_INV) & (age < AGE_MIN)
    rates = ewma.copy()
    rates[route] = learned[route]
    info = {
        "route_candidate_ticks": float(route.sum()),
        "route_learned_ticks": float(route.sum()),
        "recovery_entries_pre": 0.0,
        "recovery_entries_post": 0.0,
        "recovery_ticks": 0.0,
        "guard_blocked_ticks": 0.0,
        "recovery_active_at_onset": 0.0,
        "fit_count_pre": float(winfo.get("fit_count_pre", 0.0)),
        "fit_count_post": float(winfo.get("fit_count_post", 0.0)),
        "fires_pre": 0.0,
        "fires_post": 0.0,
    }
    return rates, info


def _new_monitor() -> ConformalMonitor:
    return ConformalMonitor(
        alpha=0.1,
        delta=0.15,
        m_consecutive=3,
        calib_size=120,
        window=30,
        min_calib=30,
    )


def drift_aware_candidate_rates(
    phi: np.ndarray,
    counts: np.ndarray,
    event_rel: int,
    cum_prev: np.ndarray,
    age: np.ndarray,
    l2: float,
    refit_every: int,
    buffer_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    T = len(counts)
    y = np.log1p(np.asarray(counts, dtype=np.float32))
    ewma = r23.ewma_rates(counts, 0.1)
    young_learned, _ = r23.winter_rates(
        phi=phi,
        counts=counts,
        event_rel=len(counts) + 1,
        mode="scheduled",
        l2=l2,
        refit_every=refit_every,
        buffer_len=buffer_len,
        trigger_alpha=0.1,
        trigger_warm=30,
        trigger_window=120,
    )

    rates = ewma.copy()
    route_candidate = np.zeros(T, dtype=bool)
    monitor = _new_monitor()
    wh = None
    buf_start = 0
    state = "stable"
    recovery_start = math.inf
    last_fire = -10**9

    fires_pre = fires_post = 0
    recovery_entries_pre = recovery_entries_post = 0
    fit_count_pre = fit_count_post = 0
    recovery_ticks = 0
    active_at_onset = 0

    for t in range(T):
        if state == "recovery" and (t - recovery_start) >= RECOVERY_TTL and (t - last_fire) >= STABLE_EXIT:
            state = "stable"
            wh = None
            monitor = _new_monitor()

        if t == event_rel and state == "recovery":
            active_at_onset = 1

        young = bool(cum_prev[t] >= MIN_INV and age[t] < AGE_MIN)
        mature = bool(cum_prev[t] >= MIN_INV and age[t] >= AGE_MIN)

        if young:
            rates[t] = young_learned[t]
            route_candidate[t] = True
            continue

        if not mature:
            rates[t] = ewma[t]
            continue

        if state == "recovery":
            recovery_ticks += 1
            due = wh is None or (t >= 20 and t % refit_every == 0)
            if due and t >= 20:
                s0 = max(int(buf_start), int(t - buffer_len))
                if t - s0 >= 20:
                    wh = r23.ridge_fit(phi[s0:t], y[s0:t], l2=l2)
                    if t < event_rel:
                        fit_count_pre += 1
                    else:
                        fit_count_post += 1
            if wh is not None:
                pred_q = phi[t] @ wh
                rates[t] = r23.pred_to_rate(pred_q)
                route_candidate[t] = True
                resid = abs(float(y[t]) - float(np.median(pred_q)))
            else:
                rates[t] = ewma[t]
                resid = abs(float(y[t]) - float(np.log1p(max(ewma[t], 0.0))))
            if monitor.update(resid):
                last_fire = t
                recovery_start = t + 1
                buf_start = max(0, t - 30)
                wh = None
                monitor = _new_monitor()
                if t < event_rel:
                    fires_pre += 1
                    recovery_entries_pre += 1
                else:
                    fires_post += 1
                    recovery_entries_post += 1
            continue

        rates[t] = ewma[t]
        resid = abs(float(y[t]) - float(np.log1p(max(ewma[t], 0.0))))
        if monitor.update(resid):
            last_fire = t
            recovery_start = t + 1
            buf_start = max(0, t - 30)
            wh = None
            monitor = _new_monitor()
            state = "recovery"
            if t < event_rel:
                fires_pre += 1
                recovery_entries_pre += 1
            else:
                fires_post += 1
                recovery_entries_post += 1

    info = {
        "fires_pre": float(fires_pre),
        "fires_post": float(fires_post),
        "fit_count_pre": float(fit_count_pre),
        "fit_count_post": float(fit_count_post),
        "recovery_entries_pre": float(recovery_entries_pre),
        "recovery_entries_post": float(recovery_entries_post),
        "recovery_ticks": float(recovery_ticks),
        "route_candidate_ticks": float(route_candidate.sum()),
        "recovery_active_at_onset": float(active_at_onset),
    }
    return rates.astype(np.float32), ewma.astype(np.float32), route_candidate, info


def trailing_mean_rates(counts: np.ndarray, window: int) -> np.ndarray:
    vals = np.asarray(counts, dtype=np.float64)
    out = np.zeros(len(vals), dtype=np.float32)
    if len(vals) == 0:
        return out
    csum = np.concatenate([[0.0], np.cumsum(vals)])
    init = float(vals[: min(60, len(vals))].mean())
    for t in range(len(vals)):
        s0 = max(0, t - int(window))
        n = t - s0
        out[t] = float((csum[t] - csum[s0]) / n) if n > 0 else init
    return out


def poisson_shortfall(lam: np.ndarray, q: np.ndarray) -> np.ndarray:
    lam64 = np.maximum(np.asarray(lam, dtype=np.float64), 0.0)
    q64 = np.maximum(np.asarray(q, dtype=np.float64), 0.0)
    return np.maximum(
        lam64 * sp_stats.poisson.sf(q64 - 1.0, lam64) - q64 * sp_stats.poisson.sf(q64, lam64),
        0.0,
    )


def expected_tick_cost_proxy(
    lam: np.ndarray,
    prewarm: np.ndarray,
    keepalive: np.ndarray,
    rho: float,
    include_keepalive: bool,
) -> np.ndarray:
    q = np.asarray(prewarm, dtype=np.float64)
    lam64 = np.maximum(np.asarray(lam, dtype=np.float64), 0.0)
    surplus = np.maximum(q - lam64, 0.0)
    idle_minutes = surplus * (np.asarray(keepalive, dtype=np.float64) if include_keepalive else 1.0)
    idle_gb_s = idle_minutes * r23.MEM_GB * r23.TICK_SECONDS
    cold = poisson_shortfall(lam64, q)
    return idle_gb_s + r23.KAPPA * float(rho) * cold


def guarded_gate_decisions(
    candidate_rates: np.ndarray,
    ewma_rates: np.ndarray,
    route_candidate: np.ndarray,
    rho: float,
    mode: str,
    observed_rates: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    tau = newsvendor_quantile(float(rho))
    cand_pw2, cand_ka2 = decisions_from_rates(candidate_rates[None, :], tau)
    ewma_pw2, ewma_ka2 = decisions_from_rates(ewma_rates[None, :], tau)
    cand_pw = cand_pw2[0].astype(np.int32)
    cand_ka = cand_ka2[0].astype(np.float32)
    ewma_pw = ewma_pw2[0].astype(np.int32)
    ewma_ka = ewma_ka2[0].astype(np.float32)

    use_candidate = route_candidate.copy()
    if mode == "idle":
        cand_proxy = cand_pw.astype(np.float64) * cand_ka.astype(np.float64)
        ewma_proxy = ewma_pw.astype(np.float64) * ewma_ka.astype(np.float64)
        use_candidate = route_candidate & (cand_proxy <= ewma_proxy + EPS)
    elif mode == "none":
        pass
    elif mode in {"cost30", "cost60", "cost120", "cost30ka", "cost60ka", "cost120ka"}:
        if observed_rates is None:
            raise ValueError(f"{mode} guard requires observed_rates")
        include_keepalive = mode.endswith("ka")
        cand_proxy = expected_tick_cost_proxy(observed_rates, cand_pw, cand_ka, rho, include_keepalive)
        ewma_proxy = expected_tick_cost_proxy(observed_rates, ewma_pw, ewma_ka, rho, include_keepalive)
        use_candidate = route_candidate & (cand_proxy <= ewma_proxy + EPS)
    elif mode.startswith("cost30b"):
        if observed_rates is None:
            raise ValueError(f"{mode} guard requires observed_rates")
        budget = float(mode[len("cost30b") :]) / 100.0
        cand_proxy = expected_tick_cost_proxy(observed_rates, cand_pw, cand_ka, rho, include_keepalive=False)
        ewma_proxy = expected_tick_cost_proxy(observed_rates, ewma_pw, ewma_ka, rho, include_keepalive=False)
        use_candidate = route_candidate & (cand_proxy <= ewma_proxy * budget + EPS)
    else:
        raise ValueError(f"unknown gate guard mode: {mode}")

    prewarm = np.where(use_candidate, cand_pw, ewma_pw).astype(np.int32)
    keepalive = np.where(use_candidate, cand_ka, ewma_ka).astype(np.float32)
    info = {
        "route_learned_ticks": float(use_candidate.sum()),
        "guard_blocked_ticks": float((route_candidate & ~use_candidate).sum()),
    }
    return prewarm, keepalive, info


def build_event_decisions(
    phi: np.ndarray,
    counts: np.ndarray,
    event_rel: int,
    cum_prev: np.ndarray,
    age: np.ndarray,
    rho: float,
    l2: float,
    refit_every: int,
    buffer_len: int,
    gate_variants: Sequence[str],
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Dict[str, float]]]:
    base_arms, infos = r23.build_arms(
        phi=phi,
        counts=counts,
        event_rel=event_rel,
        l2=l2,
        refit_every=refit_every,
        buffer_len=buffer_len,
    )
    arm_decisions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    out_infos: Dict[str, Dict[str, float]] = {}
    tau = newsvendor_quantile(float(rho))
    for arm_name, rates in base_arms.items():
        pw2, ka2 = decisions_from_rates(rates[None, :], tau)
        arm_decisions[arm_name] = (pw2[0].astype(np.int32), ka2[0].astype(np.float32))
        out_infos[arm_name] = dict(infos.get(arm_name, {}))

    current_rates, current_info = current_gate_rates(
        phi=phi,
        counts=counts,
        cum_prev=cum_prev,
        age=age,
        l2=l2,
        refit_every=refit_every,
        buffer_len=buffer_len,
    )
    pw2, ka2 = decisions_from_rates(current_rates[None, :], tau)
    arm_decisions["G_WE_A720_current"] = (pw2[0].astype(np.int32), ka2[0].astype(np.float32))
    out_infos["G_WE_A720_current"] = current_info

    cand_rates, ewma, route_candidate, gate_info = drift_aware_candidate_rates(
        phi=phi,
        counts=counts,
        event_rel=event_rel,
        cum_prev=cum_prev,
        age=age,
        l2=l2,
        refit_every=refit_every,
        buffer_len=buffer_len,
    )
    observed_by_mode = {
        "cost30": trailing_mean_rates(counts, 30),
        "cost30ka": trailing_mean_rates(counts, 30),
        "cost60": trailing_mean_rates(counts, 60),
        "cost60ka": trailing_mean_rates(counts, 60),
        "cost120": trailing_mean_rates(counts, 120),
        "cost120ka": trailing_mean_rates(counts, 120),
    }
    for variant in gate_variants:
        if variant == "idle":
            name = "G_WE_A720D"
            mode = "idle"
            observed = None
        elif variant == "none":
            name = "G_WE_A720D_NG"
            mode = "none"
            observed = None
        elif variant.startswith("cost30b"):
            name = f"G_WE_A720D_{variant.upper()}"
            mode = variant
            observed = observed_by_mode["cost30"]
        elif variant in observed_by_mode:
            name = f"G_WE_A720D_{variant.upper()}"
            mode = variant
            observed = observed_by_mode[variant]
        else:
            raise ValueError(f"unknown gate variant: {variant}")
        pw, ka, dinfo = guarded_gate_decisions(cand_rates, ewma, route_candidate, rho, mode, observed)
        arm_decisions[name] = (pw, ka)
        info = dict(gate_info)
        info.update(dinfo)
        out_infos[name] = info

    arm_decisions["keepalive_10"] = r23.fixed_keepalive_decisions(counts, 10)
    out_infos["keepalive_10"] = {}
    return arm_decisions, out_infos


def evaluate_event(
    ev: r23.Event,
    drift_kind: str,
    counts_all: np.ndarray,
    features_all: np.ndarray,
    duration_mean_all: np.ndarray,
    duration_std_all: np.ndarray,
    trainer,
    rng_seed: int,
    burnin: int,
    horizon: int,
    windows: Sequence[int],
    rhos: Sequence[float],
    seeds: Sequence[int],
    l2: float,
    refit_every: int,
    buffer_len: int,
    sim_mode: str,
    gate_variants: Sequence[str],
) -> List[Dict[str, float]]:
    start = ev.onset - burnin
    end = ev.onset + horizon
    event_rel = burnin
    base = r23.integer_base(counts_all[ev.func_id, start:end])
    cum_prev, age, prior_total, first_abs = event_state_arrays(counts_all, ev.func_id, start, end)

    if ev.source == "synthetic":
        rng = np.random.default_rng(rng_seed)
        series = r23.apply_synthetic_drift(base, event_rel, drift_kind, rng)
        app_corate = features_all[ev.func_id, start:end, 10].astype(np.float32)
        feats = r23.make_segment_features(series, app_corate, start)
        actual_kind = drift_kind
    else:
        series = base
        feats = features_all[ev.func_id, start:end].astype(np.float32)
        actual_kind = ev.kind

    phi = r23.embed_features(trainer, feats)
    dm = np.asarray(duration_mean_all[ev.func_id, start:end], dtype=np.float32)
    ds = np.asarray(duration_std_all[ev.func_id, start:end], dtype=np.float32)

    mature_ok, cum_before_onset, age_at_onset = mature_eligible(counts_all, ev)
    post_sum_actual = int(series[event_rel : event_rel + horizon].sum())
    pre_sum_actual = int(series[event_rel - 240 : event_rel].sum())
    active_actual = float(np.mean(series[event_rel : event_rel + horizon] > 0))

    rows: List[Dict[str, float]] = []
    for rho in rhos:
        arm_decisions, infos = build_event_decisions(
            phi=phi,
            counts=series,
            event_rel=event_rel,
            cum_prev=cum_prev,
            age=age,
            rho=float(rho),
            l2=l2,
            refit_every=refit_every,
            buffer_len=buffer_len,
            gate_variants=gate_variants,
        )
        for seed in seeds:
            for arm_name, (prewarm, ka) in arm_decisions.items():
                sim_fn = (
                    r23.simulate_roll_numba
                    if sim_mode == "numba" and r23._simulate_roll_jit is not None
                    else r23.simulate_roll
                )
                sim = sim_fn(
                    counts=series,
                    prewarm=prewarm,
                    keep_alive_min=ka,
                    durations_mean=dm,
                    durations_std=ds,
                    seed=seed,
                )
                metrics = r23.metrics_for_windows(sim, event_rel, windows, int(rho))
                info = infos.get(arm_name, {})
                row: Dict[str, float] = {
                    "event_id": ev.event_id,
                    "source": ev.source,
                    "func_id": float(ev.func_id),
                    "onset": float(ev.onset),
                    "kind": actual_kind,
                    "bucket": ev.bucket,
                    "arm": arm_name,
                    "rho": float(rho),
                    "seed": float(seed),
                    "pre_sum_original": float(ev.pre_sum),
                    "post_sum_original": float(ev.post_sum),
                    "pre_sum_actual": float(pre_sum_actual),
                    "post_sum_actual": float(post_sum_actual),
                    "post_active_frac": active_actual,
                    "mature_eligible": float(mature_ok),
                    "cum_inv_before_onset": float(cum_before_onset),
                    "age_at_onset": float(age_at_onset),
                    "prior_total_at_segment": float(prior_total),
                    "first_seen_abs": float(first_abs),
                    "fires_pre": float(info.get("fires_pre", 0.0)),
                    "fires_post": float(info.get("fires_post", 0.0)),
                    "fit_count_pre": float(info.get("fit_count_pre", 0.0)),
                    "fit_count_post": float(info.get("fit_count_post", 0.0)),
                    "recovery_entries_pre": float(info.get("recovery_entries_pre", 0.0)),
                    "recovery_entries_post": float(info.get("recovery_entries_post", 0.0)),
                    "recovery_ticks": float(info.get("recovery_ticks", 0.0)),
                    "route_candidate_ticks": float(info.get("route_candidate_ticks", 0.0)),
                    "route_learned_ticks": float(info.get("route_learned_ticks", 0.0)),
                    "guard_blocked_ticks": float(info.get("guard_blocked_ticks", 0.0)),
                    "recovery_active_at_onset": float(info.get("recovery_active_at_onset", 0.0)),
                }
                for k, v in ev.meta.items():
                    row[f"meta_{k}"] = float(v)
                row.update(metrics)
                rows.append(row)
    return rows


def summarize(rows: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    groups = ["source", "kind", "rho", "arm"]
    meta_means = [
        "fires_post",
        "fit_count_post",
        "recovery_entries_post",
        "recovery_ticks",
        "route_candidate_ticks",
        "route_learned_ticks",
        "guard_blocked_ticks",
        "recovery_active_at_onset",
    ]
    records = []
    for keys, g in rows.groupby(groups, sort=True):
        rec = dict(zip(groups, keys))
        rec["n_event_seed"] = int(len(g))
        rec["n_events"] = int(g["event_id"].nunique())
        rec["post_sum_actual_mean"] = float(g["post_sum_actual"].mean())
        rec["post_active_frac_mean"] = float(g["post_active_frac"].mean())
        rec["mature_eligible_frac"] = float(g["mature_eligible"].mean())
        for col in meta_means:
            rec[f"{col}_mean"] = float(g[col].mean()) if col in g else 0.0
        for w in windows:
            cold = float(g[f"cold_{w}"].sum())
            inv = float(g[f"inv_{w}"].sum())
            idle = float(g[f"idle_gb_s_{w}"].sum())
            cost = float(g[f"cost_{w}"].sum())
            rec[f"cold_{w}"] = cold
            rec[f"inv_{w}"] = inv
            rec[f"csr_{w}_pct"] = 100.0 * cold / inv if inv > 0 else 0.0
            rec[f"idle_gb_s_{w}"] = idle
            rec[f"busy_gb_s_{w}"] = float(g[f"busy_gb_s_{w}"].sum())
            rec[f"cost_{w}"] = cost
        rec["first10_active_csr_pct"] = float(
            100.0
            * g["first10_active_csr_pct"].mul(g["first10_active_inv"]).sum()
            / max(g["first10_active_inv"].sum(), 1.0)
        )
        records.append(rec)
    return add_deltas(pd.DataFrame(records), windows)


def add_deltas(summary: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    rows = []
    key_cols = ["source", "kind", "rho"]
    baselines = {
        "frozen": "winter_frozen",
        "ewma_0p1": "ewma_0.1",
        "current_gate": "G_WE_A720_current",
    }
    for _, group in summary.groupby(key_cols, sort=False):
        group = group.copy()
        for label, arm in baselines.items():
            base = group[group["arm"] == arm]
            if base.empty:
                continue
            base_row = base.iloc[0]
            for w in windows:
                group[f"delta_vs_{label}_csr_{w}_pp"] = (
                    group[f"csr_{w}_pct"] - float(base_row[f"csr_{w}_pct"])
                )
                group[f"delta_vs_{label}_cost_{w}"] = (
                    group[f"cost_{w}"] - float(base_row[f"cost_{w}"])
                )
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=r23.ROOT / "data" / "processed")
    p.add_argument("--model-path", type=Path, default=r23.DEFAULT_MODEL_PATH)
    p.add_argument("--tag", default="")
    p.add_argument("--synthetic-events", type=int, default=72)
    p.add_argument("--natural-per-kind", type=int, default=20)
    p.add_argument("--burnin", type=int, default=1440)
    p.add_argument("--horizon", type=int, default=240)
    p.add_argument("--step", type=int, default=60)
    p.add_argument("--min-pre", type=int, default=20)
    p.add_argument("--min-post", type=int, default=20)
    p.add_argument("--scan-functions", type=int, default=0)
    p.add_argument("--max-window-inv", type=int, default=0)
    p.add_argument("--seed", type=int, default=25025)
    p.add_argument("--rho", type=float, nargs="+", default=[1.0, 10.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--l2", type=float, default=1e-2)
    p.add_argument("--refit-every", type=int, default=10)
    p.add_argument("--buffer-len", type=int, default=120)
    p.add_argument(
        "--gate-variants",
        nargs="+",
        default=["idle", "none"],
        help=(
            "gate variants: idle=original guard, none=unguarded, "
            "cost{30,60,120}=recent-demand expected-cost guard, "
            "cost{30,60,120}ka includes keepalive in the idle proxy, "
            "cost30bNN allows an NN percent expected-cost budget"
        ),
    )
    p.add_argument("--sim-mode", choices=["numba", "exact"], default="numba")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--all-eligible", action="store_true")
    p.add_argument("--events-json", type=Path, default=None, help="reuse selected events from an R23/R25 JSON")
    args = p.parse_args(argv)

    t0 = time.time()
    args.data_dir = args.data_dir.resolve()
    args.model_path = args.model_path.resolve()
    out_json, out_events_csv, out_summary_csv = output_paths(args.tag)
    counts = np.load(args.data_dir / "counts.npy", mmap_mode="r")
    features = np.load(args.data_dir / "features.npy", mmap_mode="r")
    if counts.shape[:2] != features.shape[:2]:
        raise RuntimeError(f"counts/features shape mismatch: {counts.shape} vs {features.shape}")
    n_funcs, T = counts.shape
    duration_mean, duration_std = r23.load_duration_arrays(args.data_dir, n_funcs, T)
    trainer = r23.load_trainer(args.device, in_features=features.shape[2], model_path=args.model_path)
    windows = list(WINDOWS_DEFAULT[:-1]) + [int(args.horizon)]

    synthetic_kinds = list(SYNTHETIC_KINDS)
    if args.events_json is not None:
        syn_events_raw, nat_events_raw, synthetic_kinds = load_events_json(args.events_json)
    else:
        syn_events_raw = r23.choose_synthetic_events(
            counts=counts,
            max_events=args.synthetic_events,
            seed=args.seed,
            burnin=args.burnin,
            horizon=args.horizon,
            step=args.step,
            min_pre=args.min_pre,
            min_post=args.min_post,
            scan_functions=args.scan_functions,
            max_window_inv=args.max_window_inv,
        )
        nat_events_raw = r23.choose_natural_events(
            counts=counts,
            max_per_kind=args.natural_per_kind,
            seed=args.seed + 1,
            burnin=args.burnin,
            horizon=args.horizon,
            step=args.step,
            min_pre=args.min_pre,
            min_post=args.min_post,
            scan_functions=args.scan_functions,
            max_window_inv=args.max_window_inv,
        )
    mature_only = not args.all_eligible
    syn_events, syn_filter = filter_mature_events(counts, syn_events_raw, mature_only)
    nat_events, nat_filter = filter_mature_events(counts, nat_events_raw, mature_only)
    if not syn_events and not nat_events:
        raise RuntimeError("no events left after mature filter")

    all_rows: List[Dict[str, float]] = []
    total_jobs = len(syn_events) * len(synthetic_kinds) + len(nat_events)
    done = 0
    for ev_i, ev in enumerate(syn_events):
        for kind_i, kind in enumerate(synthetic_kinds):
            rows = evaluate_event(
                ev=ev,
                drift_kind=kind,
                counts_all=counts,
                features_all=features,
                duration_mean_all=duration_mean,
                duration_std_all=duration_std,
                trainer=trainer,
                rng_seed=args.seed + ev_i * 100 + kind_i,
                burnin=args.burnin,
                horizon=args.horizon,
                windows=windows,
                rhos=args.rho,
                seeds=args.seeds,
                l2=args.l2,
                refit_every=args.refit_every,
                buffer_len=args.buffer_len,
                sim_mode=args.sim_mode,
                gate_variants=args.gate_variants,
            )
            all_rows.extend(rows)
            done += 1
            if done % 10 == 0 or done == total_jobs:
                print(f"[progress] {done}/{total_jobs} event-configs complete", flush=True)

    for ev_i, ev in enumerate(nat_events):
        rows = evaluate_event(
            ev=ev,
            drift_kind=ev.kind,
            counts_all=counts,
            features_all=features,
            duration_mean_all=duration_mean,
            duration_std_all=duration_std,
            trainer=trainer,
            rng_seed=args.seed + 100000 + ev_i,
            burnin=args.burnin,
            horizon=args.horizon,
            windows=windows,
            rhos=args.rho,
            seeds=args.seeds,
            l2=args.l2,
            refit_every=args.refit_every,
            buffer_len=args.buffer_len,
            sim_mode=args.sim_mode,
            gate_variants=args.gate_variants,
        )
        all_rows.extend(rows)
        done += 1
        if done % 10 == 0 or done == total_jobs:
            print(f"[progress] {done}/{total_jobs} event-configs complete", flush=True)

    rows_df = pd.DataFrame(all_rows)
    summary = summarize(rows_df, windows)
    out_events_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(out_events_csv, index=False)
    summary.to_csv(out_summary_csv, index=False)
    payload = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset": {
            "data_dir": str(args.data_dir),
            "model_path": str(args.model_path),
            "n_funcs": int(n_funcs),
            "T": int(T),
        },
        "mature_only": mature_only,
        "event_filter": {"synthetic": syn_filter, "natural": nat_filter},
        "synthetic_events": [ev.__dict__ for ev in syn_events],
        "natural_events": [ev.__dict__ for ev in nat_events],
        "synthetic_kinds": list(synthetic_kinds),
        "windows": windows,
        "n_rows": int(len(rows_df)),
        "elapsed_sec": time.time() - t0,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=1, default=float)
    print(f"wrote {out_json}")
    print(f"wrote {out_events_csv}")
    print(f"wrote {out_summary_csv}")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
