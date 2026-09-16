#!/usr/bin/env python3
"""R28 sensitivity sweep for the integrated drift-aware WINTER-G gate.

R25 fixed Stability+TTL constants and a single guard. This script evaluates
whether the R25 dual-criterion failure is plausibly an artifact of those fixed
choices by sweeping recovery TTL/stability/monitor settings and guard rules on
the same mature drift events.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import revision_r23_drift_v2 as r23  # noqa: E402
from scripts import revision_r25_drift_integrated_gate as r25  # noqa: E402
from scripts.phase6_des import decisions_from_rates  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.drift.triggers import ConformalMonitor  # noqa: E402


WINDOWS = (15, 30, 60, 240)
DEFAULT_TRACES = ("azure2021", "azure2019", "huawei")


@dataclass(frozen=True)
class TraceSpec:
    name: str
    data_dir: Path
    events_json: Path


@dataclass(frozen=True)
class GateConfig:
    name: str
    ttl: int
    stable: int
    alpha: float
    m: int
    delta: float = 0.15
    calib_size: int = 120
    window: int = 30
    min_calib: int = 30


TRACE_SPECS = {
    "azure2021": TraceSpec(
        "azure2021",
        ROOT / "data" / "processed",
        ROOT / "results" / "runs" / "revision_r23_drift_v2.json",
    ),
    "azure2019": TraceSpec(
        "azure2019",
        ROOT / "data" / "processed_2019",
        ROOT / "results" / "runs" / "revision_r23_drift_v2_azure2019.json",
    ),
    "huawei": TraceSpec(
        "huawei",
        ROOT / "data" / "processed_huawei",
        ROOT / "results" / "runs" / "revision_r23_drift_v2_huawei.json",
    ),
}


def parse_config(spec: str) -> GateConfig:
    vals: Dict[str, str] = {}
    for part in spec.split(","):
        k, v = part.split("=", 1)
        vals[k.strip()] = v.strip()
    ttl = int(vals["ttl"])
    stable = int(vals["stable"])
    alpha = float(vals.get("alpha", 0.1))
    m = int(vals.get("m", 3))
    name = vals.get("name", f"ttl{ttl}_s{stable}_a{alpha:g}_m{m}")
    return GateConfig(name=name, ttl=ttl, stable=stable, alpha=alpha, m=m)


def install_config(cfg: GateConfig) -> None:
    r25.RECOVERY_TTL = int(cfg.ttl)
    r25.STABLE_EXIT = int(cfg.stable)

    def _new_monitor() -> ConformalMonitor:
        return ConformalMonitor(
            alpha=float(cfg.alpha),
            delta=float(cfg.delta),
            m_consecutive=int(cfg.m),
            calib_size=int(cfg.calib_size),
            window=int(cfg.window),
            min_calib=int(cfg.min_calib),
        )

    r25._new_monitor = _new_monitor


def guard_observed_rates(counts: np.ndarray, guard: str) -> np.ndarray | None:
    if guard in {"idle", "none"}:
        return None
    if guard.startswith("cost30"):
        return r25.trailing_mean_rates(counts, 30)
    if guard.startswith("cost60"):
        return r25.trailing_mean_rates(counts, 60)
    if guard.startswith("cost120"):
        return r25.trailing_mean_rates(counts, 120)
    raise ValueError(f"unknown guard: {guard}")


def choose_subset(
    events: Sequence[r23.Event],
    max_total: int,
    per_kind: int,
    seed: int,
) -> List[r23.Event]:
    rng = np.random.default_rng(seed)
    if per_kind > 0:
        selected: List[r23.Event] = []
        for _, group in pd.DataFrame(
            [{"i": i, "kind": ev.kind} for i, ev in enumerate(events)]
        ).groupby("kind", sort=True):
            idx = group["i"].to_numpy()
            rng.shuffle(idx)
            selected.extend(events[int(i)] for i in idx[:per_kind])
        return selected
    idx = np.arange(len(events))
    rng.shuffle(idx)
    return [events[int(i)] for i in idx[:max_total]]


def select_trace_events(
    counts: np.ndarray,
    events_json: Path,
    synthetic_bases: int,
    natural_per_kind: int,
    seed: int,
) -> Tuple[List[r23.Event], List[r23.Event], List[str], Dict[str, Dict[str, int]]]:
    syn_raw, nat_raw, kinds = r25.load_events_json(events_json)
    syn_mature, syn_filter = r25.filter_mature_events(counts, syn_raw, mature_only=True)
    nat_mature, nat_filter = r25.filter_mature_events(counts, nat_raw, mature_only=True)
    syn = choose_subset(syn_mature, synthetic_bases, per_kind=0, seed=seed)
    nat = choose_subset(nat_mature, max_total=0, per_kind=natural_per_kind, seed=seed + 1)
    filters = {
        "synthetic": dict(syn_filter, sampled=len(syn)),
        "natural": dict(nat_filter, sampled=len(nat)),
    }
    return syn, nat, kinds, filters


def simulate_decision(
    series: np.ndarray,
    dm: np.ndarray,
    ds: np.ndarray,
    event_rel: int,
    prewarm: np.ndarray,
    keepalive: np.ndarray,
    seed: int,
    sim_mode: str,
) -> Dict[str, float]:
    sim_fn = (
        r23.simulate_roll_numba
        if sim_mode == "numba" and r23._simulate_roll_jit is not None
        else r23.simulate_roll
    )
    sim = sim_fn(
        counts=series,
        prewarm=prewarm,
        keep_alive_min=keepalive,
        durations_mean=dm,
        durations_std=ds,
        seed=int(seed),
    )
    return r23.metrics_for_windows(sim, event_rel, WINDOWS, rho=1)


def evaluate_event(
    trace: str,
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
    rhos: Sequence[float],
    seeds: Sequence[int],
    l2: float,
    refit_every: int,
    buffer_len: int,
    configs: Sequence[GateConfig],
    guards: Sequence[str],
    sim_mode: str,
) -> List[Dict[str, float]]:
    start = ev.onset - burnin
    end = ev.onset + horizon
    event_rel = burnin
    base = r23.integer_base(counts_all[ev.func_id, start:end])
    cum_prev, age, _, _ = r25.event_state_arrays(counts_all, ev.func_id, start, end)

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
    mature_ok, cum_before_onset, age_at_onset = r25.mature_eligible(counts_all, ev)
    post_sum_actual = int(series[event_rel : event_rel + horizon].sum())
    post_active = float(np.mean(series[event_rel : event_rel + horizon] > 0))

    rows: List[Dict[str, float]] = []
    ewma_rates = r23.ewma_rates(series, 0.1)
    observed_cache = {guard: guard_observed_rates(series, guard) for guard in guards}

    for rho in rhos:
        tau = newsvendor_quantile(float(rho))
        ewma_pw2, ewma_ka2 = decisions_from_rates(ewma_rates[None, :], tau)
        baseline_decision = (ewma_pw2[0].astype(np.int32), ewma_ka2[0].astype(np.float32))

        candidates = []
        candidates.append(
            {
                "arm": "ewma_0.1",
                "config": "baseline",
                "guard": "baseline",
                "prewarm": baseline_decision[0],
                "keepalive": baseline_decision[1],
                "gate_info": {},
            }
        )

        for cfg in configs:
            install_config(cfg)
            cand_rates, ewma, route_candidate, gate_info = r25.drift_aware_candidate_rates(
                phi=phi,
                counts=series,
                event_rel=event_rel,
                cum_prev=cum_prev,
                age=age,
                l2=l2,
                refit_every=refit_every,
                buffer_len=buffer_len,
            )
            for guard in guards:
                pw, ka, dinfo = r25.guarded_gate_decisions(
                    candidate_rates=cand_rates,
                    ewma_rates=ewma,
                    route_candidate=route_candidate,
                    rho=float(rho),
                    mode=guard,
                    observed_rates=observed_cache[guard],
                )
                info = dict(gate_info)
                info.update(dinfo)
                arm_name = f"G_{cfg.name}_{guard}"
                candidates.append(
                    {
                        "arm": arm_name,
                        "config": cfg.name,
                        "guard": guard,
                        "prewarm": pw,
                        "keepalive": ka,
                        "gate_info": info,
                    }
                )

        for seed in seeds:
            for cand in candidates:
                metrics = simulate_decision(
                    series=series,
                    dm=dm,
                    ds=ds,
                    event_rel=event_rel,
                    prewarm=cand["prewarm"],
                    keepalive=cand["keepalive"],
                    seed=int(seed),
                    sim_mode=sim_mode,
                )
                info = cand["gate_info"]
                row = {
                    "trace": trace,
                    "event_id": ev.event_id,
                    "source": ev.source,
                    "kind": actual_kind,
                    "func_id": int(ev.func_id),
                    "onset": int(ev.onset),
                    "arm": cand["arm"],
                    "config": cand["config"],
                    "guard": cand["guard"],
                    "rho": float(rho),
                    "seed": int(seed),
                    "post_sum_actual": float(post_sum_actual),
                    "post_active_frac": post_active,
                    "mature_eligible": float(mature_ok),
                    "cum_inv_before_onset": float(cum_before_onset),
                    "age_at_onset": float(age_at_onset),
                    "fires_post": float(info.get("fires_post", 0.0)),
                    "recovery_entries_post": float(info.get("recovery_entries_post", 0.0)),
                    "recovery_ticks": float(info.get("recovery_ticks", 0.0)),
                    "route_candidate_ticks": float(info.get("route_candidate_ticks", 0.0)),
                    "route_learned_ticks": float(info.get("route_learned_ticks", 0.0)),
                    "guard_blocked_ticks": float(info.get("guard_blocked_ticks", 0.0)),
                }
                row.update(metrics)
                rows.append(row)
    return rows


def add_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in summary.groupby(["trace", "source", "kind", "rho"], sort=False):
        g = g.copy()
        base = g[g["arm"] == "ewma_0.1"]
        if not base.empty:
            b = base.iloc[0]
            for w in WINDOWS:
                g[f"delta_vs_ewma_csr_{w}_pp"] = g[f"csr_{w}_pct"] - float(b[f"csr_{w}_pct"])
                g[f"delta_vs_ewma_cost_{w}"] = g[f"cost_{w}"] - float(b[f"cost_{w}"])
        out.append(g)
    return pd.concat(out, ignore_index=True)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_cols = ["trace", "source", "kind", "rho", "arm", "config", "guard"]
    for keys, g in rows.groupby(group_cols, sort=True):
        rec = dict(zip(group_cols, keys))
        rec["n_event_seed"] = int(len(g))
        rec["n_events"] = int(g["event_id"].nunique())
        rec["post_sum_actual_mean"] = float(g["post_sum_actual"].mean())
        for c in [
            "fires_post",
            "recovery_entries_post",
            "recovery_ticks",
            "route_candidate_ticks",
            "route_learned_ticks",
            "guard_blocked_ticks",
        ]:
            rec[f"{c}_mean"] = float(g[c].mean()) if c in g else 0.0
        for w in WINDOWS:
            cold = float(g[f"cold_{w}"].sum())
            inv = float(g[f"inv_{w}"].sum())
            cost = float(g[f"cost_{w}"].sum())
            rec[f"cold_{w}"] = cold
            rec[f"inv_{w}"] = inv
            rec[f"csr_{w}_pct"] = 100.0 * cold / inv if inv else 0.0
            rec[f"cost_{w}"] = cost
        records.append(rec)
    return add_deltas(pd.DataFrame(records))


def dual_counts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in summary[summary["arm"] != "ewma_0.1"].groupby(["trace", "arm", "config", "guard"], sort=True):
        rec = dict(zip(["trace", "arm", "config", "guard"], keys))
        csr = g["delta_vs_ewma_csr_240_pp"] < 0
        cost = g["delta_vs_ewma_cost_240"] < 0
        rec["n_condition_cells"] = int(len(g))
        rec["csr_lower"] = int(csr.sum())
        rec["cost_lower"] = int(cost.sum())
        rec["dual_win"] = int((csr & cost).sum())
        rec["mean_delta_csr_pp"] = float(g["delta_vs_ewma_csr_240_pp"].mean())
        rec["mean_delta_cost"] = float(g["delta_vs_ewma_cost_240"].mean())
        rec["mean_route_learned_ticks"] = float(g["route_learned_ticks_mean"].mean())
        rec["mean_guard_blocked_ticks"] = float(g["guard_blocked_ticks_mean"].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def pooled_dual_counts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pooled = summary.copy()
    pooled["trace"] = "pooled"
    for keys, g in pooled[pooled["arm"] != "ewma_0.1"].groupby(["trace", "arm", "config", "guard"], sort=True):
        # Aggregate provider condition rows by arithmetic mean, matching R25's
        # cross-provider condition-cell readout rather than event weighting.
        cells = (
            g.groupby(["source", "kind", "rho"], sort=True)
            .agg(
                delta_csr=("delta_vs_ewma_csr_240_pp", "mean"),
                delta_cost=("delta_vs_ewma_cost_240", "mean"),
                route=("route_learned_ticks_mean", "mean"),
                blocked=("guard_blocked_ticks_mean", "mean"),
            )
            .reset_index()
        )
        csr = cells["delta_csr"] < 0
        cost = cells["delta_cost"] < 0
        rec = dict(zip(["trace", "arm", "config", "guard"], keys))
        rec["n_condition_cells"] = int(len(cells))
        rec["csr_lower"] = int(csr.sum())
        rec["cost_lower"] = int(cost.sum())
        rec["dual_win"] = int((csr & cost).sum())
        rec["mean_delta_csr_pp"] = float(cells["delta_csr"].mean())
        rec["mean_delta_cost"] = float(cells["delta_cost"].mean())
        rec["mean_route_learned_ticks"] = float(cells["route"].mean())
        rec["mean_guard_blocked_ticks"] = float(cells["blocked"].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--traces", nargs="+", default=list(DEFAULT_TRACES))
    ap.add_argument("--configs", nargs="+", default=[
        "name=ttl60_s30_a10_m3,ttl=60,stable=30,alpha=0.1,m=3",
        "name=ttl120_s60_a10_m3,ttl=120,stable=60,alpha=0.1,m=3",
        "name=ttl240_s120_a10_m3,ttl=240,stable=120,alpha=0.1,m=3",
        "name=ttl120_s60_a05_m5,ttl=120,stable=60,alpha=0.05,m=5",
        "name=ttl60_s30_a05_m5,ttl=60,stable=30,alpha=0.05,m=5",
    ])
    ap.add_argument("--guards", nargs="+", default=["none", "idle", "cost30", "cost30b110", "cost30b125"])
    ap.add_argument("--synthetic-bases", type=int, default=6)
    ap.add_argument("--natural-per-kind", type=int, default=2)
    ap.add_argument("--burnin", type=int, default=1440)
    ap.add_argument("--horizon", type=int, default=240)
    ap.add_argument("--rho", nargs="+", type=float, default=[1.0, 10.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--refit-every", type=int, default=10)
    ap.add_argument("--buffer-len", type=int, default=120)
    ap.add_argument("--seed", type=int, default=28028)
    ap.add_argument("--sim-mode", choices=["numba", "exact"], default="numba")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    configs = [parse_config(s) for s in args.configs]
    t0 = time.time()
    all_rows: List[Dict[str, float]] = []
    manifest = {
        "args": vars(args),
        "configs": [cfg.__dict__ for cfg in configs],
        "trace_filters": {},
    }

    for trace_i, trace_name in enumerate(args.traces):
        spec = TRACE_SPECS[trace_name]
        counts = np.load(spec.data_dir / "counts.npy", mmap_mode="r")
        features = np.load(spec.data_dir / "features.npy", mmap_mode="r")
        n_funcs, t_len = counts.shape
        dm, ds = r23.load_duration_arrays(spec.data_dir, n_funcs, t_len)
        trainer = r23.load_trainer(args.device, in_features=features.shape[2], model_path=r23.DEFAULT_MODEL_PATH)
        syn, nat, kinds, filters = select_trace_events(
            counts=counts,
            events_json=spec.events_json,
            synthetic_bases=args.synthetic_bases,
            natural_per_kind=args.natural_per_kind,
            seed=args.seed + trace_i * 1000,
        )
        manifest["trace_filters"][trace_name] = filters
        jobs = len(syn) * len(kinds) + len(nat)
        done = 0
        for ev_i, ev in enumerate(syn):
            for kind_i, kind in enumerate(kinds):
                rows = evaluate_event(
                    trace=trace_name,
                    ev=ev,
                    drift_kind=kind,
                    counts_all=counts,
                    features_all=features,
                    duration_mean_all=dm,
                    duration_std_all=ds,
                    trainer=trainer,
                    rng_seed=args.seed + trace_i * 100000 + ev_i * 100 + kind_i,
                    burnin=args.burnin,
                    horizon=args.horizon,
                    rhos=args.rho,
                    seeds=args.seeds,
                    l2=args.l2,
                    refit_every=args.refit_every,
                    buffer_len=args.buffer_len,
                    configs=configs,
                    guards=args.guards,
                    sim_mode=args.sim_mode,
                )
                all_rows.extend(rows)
                done += 1
                if done % 10 == 0 or done == jobs:
                    print(f"[{trace_name}] {done}/{jobs} event-configs complete", flush=True)
        for ev_i, ev in enumerate(nat):
            rows = evaluate_event(
                trace=trace_name,
                ev=ev,
                drift_kind=ev.kind,
                counts_all=counts,
                features_all=features,
                duration_mean_all=dm,
                duration_std_all=ds,
                trainer=trainer,
                rng_seed=args.seed + trace_i * 100000 + 50000 + ev_i,
                burnin=args.burnin,
                horizon=args.horizon,
                rhos=args.rho,
                seeds=args.seeds,
                l2=args.l2,
                refit_every=args.refit_every,
                buffer_len=args.buffer_len,
                configs=configs,
                guards=args.guards,
                sim_mode=args.sim_mode,
            )
            all_rows.extend(rows)
            done += 1
            if done % 10 == 0 or done == jobs:
                print(f"[{trace_name}] {done}/{jobs} event-configs complete", flush=True)

    rows_df = pd.DataFrame(all_rows)
    summary = summarize(rows_df)
    counts = pd.concat([dual_counts(summary), pooled_dual_counts(summary)], ignore_index=True)
    suffix = f"_{args.tag}" if args.tag else ""
    runs_path = ROOT / "results" / "runs" / f"revision_r28_gate_sensitivity{suffix}.json"
    events_path = ROOT / "results" / "tables" / f"T_r28_gate_sensitivity_events{suffix}.csv"
    summary_path = ROOT / "results" / "tables" / f"T_r28_gate_sensitivity_summary{suffix}.csv"
    counts_path = ROOT / "results" / "tables" / f"T_r28_gate_sensitivity_dual_counts{suffix}.csv"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)
    counts.to_csv(counts_path, index=False)
    manifest["elapsed_sec"] = time.time() - t0
    manifest["n_rows"] = int(len(rows_df))
    manifest["outputs"] = {
        "events": str(events_path),
        "summary": str(summary_path),
        "dual_counts": str(counts_path),
    }
    runs_path.write_text(json.dumps(manifest, indent=1, default=float))
    print(f"wrote {runs_path}")
    print(f"wrote {events_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {counts_path}")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
