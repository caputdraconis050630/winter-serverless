#!/usr/bin/env python3
"""R26 drift reset-to-prior audit for WINTER.

This experiment tests a hypothesis that is not covered by R23/R25:
after a drift onset or trigger, reset the per-function learned head to the
cross-function prototype prior and then refit only on post-reset history using
the prototype-biased ridge solve.

The script reuses R23 event selection, synthetic drift transforms, simulator,
and metrics. It writes a separate R26 artifact and does not overwrite R23/R25.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import revision_r23_drift_v2 as r23  # noqa: E402
from scripts.phase6_des import decisions_from_rates  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.drift.triggers import ConformalMonitor  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402


L = 60
D = 64
RESET_SUPPORT_TICKS = 20
EPS = 1e-12
RESET_ARMS = ("winter_prior_reset_oracle", "winter_prior_reset_trigger")
EWMA_ARMS = ("ewma_0.1", "ewma_0.3", "ewma_0.5")
PRACTICAL_WINTER = ("winter_scheduled", "winter_trigger", "winter_sched_trigger")
DEFAULT_SYNTHETIC_KINDS = (
    "scale_down_0.5",
    "scale_down_0.2",
    "scale_up_2",
    "scale_up_5",
    "ramp_down_0.2_120",
    "ramp_up_5_120",
    "burst_compress_4",
    "phase_circular_120",
)


def output_paths(tag: str) -> Tuple[Path, Path, Path, Path, Path]:
    suffix = f"_{tag}" if tag else ""
    return (
        ROOT / "results" / "runs" / f"revision_r26_drift_reset_prior{suffix}.json",
        ROOT / "results" / "tables" / f"T_r26_drift_reset_prior_events{suffix}.csv",
        ROOT / "results" / "tables" / f"T_r26_drift_reset_prior_summary{suffix}.csv",
        ROOT / "results" / "tables" / f"T_r26_drift_reset_prior_comparisons{suffix}.csv",
        ROOT / "results" / "tables" / f"T_r26_drift_reset_prior_counts{suffix}.csv",
    )


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
    payload = json.load(open(path))
    syn = [event_from_dict(ev) for ev in payload.get("synthetic_events", [])]
    nat = [event_from_dict(ev) for ev in payload.get("natural_events", [])]
    kinds = [str(k) for k in payload.get("synthetic_kinds", DEFAULT_SYNTHETIC_KINDS)]
    return syn, nat, kinds


@torch.no_grad()
def embed_functions(
    body,
    features: np.ndarray,
    func_indices: Sequence[int],
    device: str,
    n_windows: int = 50,
    batch_size: int = 256,
) -> torch.Tensor:
    body.eval()
    t_len = features.shape[1]
    starts = np.linspace(L, t_len - 1, n_windows, dtype=int)
    out: List[torch.Tensor] = []
    for start in range(0, len(func_indices), batch_size):
        idx = np.asarray(func_indices[start : start + batch_size], dtype=np.int64)
        accum: Optional[torch.Tensor] = None
        for t in starts:
            x = torch.from_numpy(
                np.array(features[idx, t - L : t, :], dtype=np.float32, copy=True)
            ).to(device)
            phi = body(x)
            accum = phi if accum is None else accum + phi
        if accum is None:
            continue
        out.append((accum / float(len(starts))).cpu())
    if not out:
        raise RuntimeError("cannot build prototypes with an empty training split")
    return torch.cat(out, dim=0)


@torch.no_grad()
def build_prototype_bank(
    trainer,
    prototype_data_dir: Path,
    split_key: str,
    n_clusters: int,
    func_cap: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    features = np.load(prototype_data_dir / "features.npy", mmap_mode="r")
    counts = np.load(prototype_data_dir / "counts.npy", mmap_mode="r")
    splits = np.load(prototype_data_dir / "splits.npz")
    if split_key not in splits.files:
        raise KeyError(f"prototype split {split_key!r} not found in {prototype_data_dir / 'splits.npz'}")
    train_idx = np.asarray(splits[split_key], dtype=np.int64)
    if len(train_idx) < n_clusters:
        raise RuntimeError(f"{len(train_idx)} prototype functions for {n_clusters} clusters")

    print(
        f"[proto] building C_proto={n_clusters} from {prototype_data_dir}::{split_key} "
        f"({len(train_idx)} functions, cap {func_cap}/cluster)",
        flush=True,
    )
    embeddings = embed_functions(trainer.body, features, train_idx, device=device)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings.numpy())
    centroids = torch.from_numpy(km.cluster_centers_).float()

    t_len = features.shape[1]
    n_outputs = N_QUANTILES
    proto_w = torch.zeros(n_clusters, D, n_outputs)
    starts = np.linspace(L, t_len - 2, 20, dtype=int)
    for c in range(n_clusters):
        cfuncs = train_idx[labels == c][:func_cap]
        if len(cfuncs) == 0:
            continue
        phis: List[torch.Tensor] = []
        ys: List[float] = []
        for fi in cfuncs:
            for t in starts:
                x = torch.from_numpy(
                    np.array(features[fi : fi + 1, t - L : t, :], dtype=np.float32, copy=True)
                ).to(device)
                phis.append(trainer.body(x).cpu())
                ys.append(float(np.log1p(counts[fi, min(t, t_len - 1)])))
        phi_pool = torch.cat(phis, dim=0).to(device)
        y_pool = torch.tensor(ys, dtype=torch.float32, device=device).unsqueeze(1)
        y_expanded = y_pool.expand(-1, n_outputs)
        w = trainer.head.adapt(phi_pool, y_expanded)
        proto_w[c] = w.detach().cpu()

    meta = {
        "prototype_data_dir": str(prototype_data_dir),
        "prototype_split": split_key,
        "prototype_clusters": int(n_clusters),
        "prototype_func_cap": int(func_cap),
        "cluster_sizes": np.bincount(labels, minlength=n_clusters).astype(int).tolist(),
    }
    return centroids, proto_w, meta


@torch.no_grad()
def assign_proto_heads(
    phi: np.ndarray,
    centroids: torch.Tensor,
    proto_w: torch.Tensor,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(np.asarray(phi, dtype=np.float32)).to(device)
    cen = centroids.to(device)
    sims = F.cosine_similarity(x.unsqueeze(1), cen.unsqueeze(0), dim=2)
    idx = sims.argmax(dim=1).cpu().numpy().astype(np.int64)
    heads = proto_w[idx].numpy().astype(np.float64)
    return heads, idx


def biased_ridge_fit(phi: np.ndarray, y: np.ndarray, w_proto: np.ndarray, l2: float) -> np.ndarray:
    yy = np.repeat(y[:, None], N_QUANTILES, axis=1)
    eye = np.eye(phi.shape[1], dtype=np.float64)
    lhs = phi.T @ phi + float(l2) * eye
    rhs = phi.T @ yy + float(l2) * w_proto
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def _new_monitor() -> ConformalMonitor:
    return ConformalMonitor(
        alpha=0.1,
        delta=0.15,
        m_consecutive=3,
        calib_size=120,
        window=30,
        min_calib=30,
    )


def prior_reset_rates(
    phi: np.ndarray,
    counts: np.ndarray,
    event_rel: int,
    centroids: torch.Tensor,
    proto_w: torch.Tensor,
    mode: str,
    l2: float,
    refit_every: int,
    buffer_len: int,
    device: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if mode not in {"oracle", "trigger"}:
        raise ValueError(f"unknown reset mode: {mode}")

    t_len = len(counts)
    y = np.log1p(np.asarray(counts, dtype=np.float32))
    proto_heads, proto_idx = assign_proto_heads(phi, centroids, proto_w, device=device)
    proto_pred = np.einsum("td,tdq->tq", phi.astype(np.float64), proto_heads)
    proto_rates = np.asarray([r23.pred_to_rate(row) for row in proto_pred], dtype=np.float32)

    rates = np.zeros(t_len, dtype=np.float32)
    wh: Optional[np.ndarray] = None
    reset_start: Optional[int] = None
    pending_reset: Optional[int] = None
    reset_done = False
    monitor = _new_monitor() if mode == "trigger" else None

    fires_pre = 0
    fires_post = 0
    fit_count_pre = 0
    fit_count_post = 0
    reset_count_post = 0
    first_reset_tick_post = math.nan
    prior_route_ticks_post = 0
    biased_route_ticks_post = 0
    pre_reset_learned_ticks_post = 0
    fallback_mean_ticks_post = 0
    proto_switches_post = 0
    last_proto_post: Optional[int] = None

    for t in range(t_len):
        if mode == "oracle" and t == event_rel and not reset_done:
            reset_start = t
            pending_reset = None
            wh = None
            reset_done = True
            reset_count_post += 1
            first_reset_tick_post = 0.0

        if pending_reset is not None and t >= pending_reset and not reset_done:
            reset_start = t
            pending_reset = None
            wh = None
            reset_done = True
            reset_count_post += 1
            first_reset_tick_post = float(t - event_rel)
            if monitor is not None:
                monitor.reset()

        in_reset_epoch = reset_start is not None and t >= reset_start
        pred_q: Optional[np.ndarray] = None
        used_prior = False
        used_biased = False
        used_pre_reset = False
        used_fallback = False

        if in_reset_epoch:
            if wh is None:
                pred_q = proto_pred[t]
                rates[t] = proto_rates[t]
                used_prior = True
            else:
                pred_q = phi[t] @ wh
                rates[t] = r23.pred_to_rate(pred_q)
                used_biased = True
        elif wh is not None:
            pred_q = phi[t] @ wh
            rates[t] = r23.pred_to_rate(pred_q)
            used_pre_reset = True
        else:
            if t > 0:
                rates[t] = float(np.mean(counts[max(0, t - L) : t]))
            used_fallback = True

        if t >= event_rel:
            if used_prior:
                prior_route_ticks_post += 1
            elif used_biased:
                biased_route_ticks_post += 1
            elif used_pre_reset:
                pre_reset_learned_ticks_post += 1
            elif used_fallback:
                fallback_mean_ticks_post += 1
            if in_reset_epoch:
                cur_proto = int(proto_idx[t])
                if last_proto_post is not None and cur_proto != last_proto_post:
                    proto_switches_post += 1
                last_proto_post = cur_proto

        if (
            mode == "trigger"
            and monitor is not None
            and pred_q is not None
            and not reset_done
        ):
            resid = abs(float(y[t]) - float(np.median(pred_q)))
            if monitor.update(resid):
                if t < event_rel:
                    fires_pre += 1
                else:
                    fires_post += 1
                    pending_reset = min(t + 1, t_len)

        if in_reset_epoch and reset_start is not None:
            support_ticks = t - reset_start
            reset_due = (
                support_ticks >= RESET_SUPPORT_TICKS
                and support_ticks % max(1, int(refit_every)) == 0
            )
            if reset_due:
                s0 = max(reset_start, t - int(buffer_len))
                if t - s0 >= RESET_SUPPORT_TICKS:
                    wh = biased_ridge_fit(phi[s0:t], y[s0:t], proto_heads[t], l2=l2)
                    if t < event_rel:
                        fit_count_pre += 1
                    else:
                        fit_count_post += 1
        else:
            scheduled_due = t >= RESET_SUPPORT_TICKS and (t % max(1, int(refit_every)) == 0)
            if scheduled_due and t < event_rel:
                s0 = max(0, t - int(buffer_len))
                if t - s0 >= RESET_SUPPORT_TICKS:
                    wh = r23.ridge_fit(phi[s0:t], y[s0:t], l2=l2)
                    fit_count_pre += 1

    info = {
        "fires_pre": float(fires_pre),
        "fires_post": float(fires_post),
        "fit_count_pre": float(fit_count_pre),
        "fit_count_post": float(fit_count_post),
        "reset_count_post": float(reset_count_post),
        "first_reset_tick_post": float(first_reset_tick_post),
        "prior_route_ticks_post": float(prior_route_ticks_post),
        "biased_route_ticks_post": float(biased_route_ticks_post),
        "pre_reset_learned_ticks_post": float(pre_reset_learned_ticks_post),
        "fallback_mean_ticks_post": float(fallback_mean_ticks_post),
        "proto_switches_post": float(proto_switches_post),
    }
    return rates, info


def build_arms_with_prior_reset(
    phi: np.ndarray,
    counts: np.ndarray,
    event_rel: int,
    centroids: torch.Tensor,
    proto_w: torch.Tensor,
    l2: float,
    refit_every: int,
    buffer_len: int,
    device: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    arms, infos = r23.build_arms(
        phi=phi,
        counts=counts,
        event_rel=event_rel,
        l2=l2,
        refit_every=refit_every,
        buffer_len=buffer_len,
    )
    for arm_name, mode in [
        ("winter_prior_reset_oracle", "oracle"),
        ("winter_prior_reset_trigger", "trigger"),
    ]:
        rates, info = prior_reset_rates(
            phi=phi,
            counts=counts,
            event_rel=event_rel,
            centroids=centroids,
            proto_w=proto_w,
            mode=mode,
            l2=l2,
            refit_every=refit_every,
            buffer_len=buffer_len,
            device=device,
        )
        arms[arm_name] = rates
        infos[arm_name] = info
    return arms, infos


def evaluate_event(
    ev: r23.Event,
    drift_kind: str,
    counts_all: np.ndarray,
    features_all: np.ndarray,
    duration_mean_all: np.ndarray,
    duration_std_all: np.ndarray,
    trainer,
    centroids: torch.Tensor,
    proto_w: torch.Tensor,
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
    device: str,
) -> List[Dict[str, float]]:
    start = ev.onset - burnin
    end = ev.onset + horizon
    event_rel = burnin
    base = r23.integer_base(counts_all[ev.func_id, start:end])

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
    arms, infos = build_arms_with_prior_reset(
        phi=phi,
        counts=series,
        event_rel=event_rel,
        centroids=centroids,
        proto_w=proto_w,
        l2=l2,
        refit_every=refit_every,
        buffer_len=buffer_len,
        device=device,
    )

    dm = np.asarray(duration_mean_all[ev.func_id, start:end], dtype=np.float32)
    ds = np.asarray(duration_std_all[ev.func_id, start:end], dtype=np.float32)
    post_sum_actual = int(series[event_rel : event_rel + horizon].sum())
    pre_sum_actual = int(series[event_rel - 240 : event_rel].sum())
    active_actual = float(np.mean(series[event_rel : event_rel + horizon] > 0))

    rows: List[Dict[str, float]] = []
    for rho in rhos:
        tau_star = newsvendor_quantile(float(rho))
        arm_decisions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for arm_name, rates in arms.items():
            prewarm2, ka2 = decisions_from_rates(rates[None, :], tau_star)
            arm_decisions[arm_name] = (
                prewarm2[0].astype(np.int32),
                ka2[0].astype(np.float32),
            )
        arm_decisions["keepalive_10"] = r23.fixed_keepalive_decisions(series, 10)

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
                metrics = r23.metrics_for_windows(sim, event_rel, windows, int(float(rho)))
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
                    "fires_pre": float(info.get("fires_pre", 0.0)),
                    "fires_post": float(info.get("fires_post", 0.0)),
                    "fit_count_pre": float(info.get("fit_count_pre", 0.0)),
                    "fit_count_post": float(info.get("fit_count_post", 0.0)),
                    "reset_count_post": float(info.get("reset_count_post", 0.0)),
                    "first_reset_tick_post": float(info.get("first_reset_tick_post", math.nan)),
                    "prior_route_ticks_post": float(info.get("prior_route_ticks_post", 0.0)),
                    "biased_route_ticks_post": float(info.get("biased_route_ticks_post", 0.0)),
                    "pre_reset_learned_ticks_post": float(
                        info.get("pre_reset_learned_ticks_post", 0.0)
                    ),
                    "fallback_mean_ticks_post": float(info.get("fallback_mean_ticks_post", 0.0)),
                    "proto_switches_post": float(info.get("proto_switches_post", 0.0)),
                }
                for k, v in ev.meta.items():
                    row[f"meta_{k}"] = float(v)
                row.update(metrics)
                rows.append(row)
    return rows


def summarize(rows: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    groups = ["source", "kind", "rho", "arm"]
    info_cols = [
        "fires_post",
        "fit_count_post",
        "reset_count_post",
        "first_reset_tick_post",
        "prior_route_ticks_post",
        "biased_route_ticks_post",
        "pre_reset_learned_ticks_post",
        "fallback_mean_ticks_post",
        "proto_switches_post",
    ]
    records: List[Dict[str, object]] = []
    for keys, g in rows.groupby(groups, sort=True):
        rec: Dict[str, object] = dict(zip(groups, keys))
        rec["n_event_seed"] = int(len(g))
        rec["n_events"] = int(g["event_id"].nunique())
        rec["post_sum_actual_mean"] = float(g["post_sum_actual"].mean())
        rec["post_active_frac_mean"] = float(g["post_active_frac"].mean())
        for col in info_cols:
            if col == "first_reset_tick_post":
                vals = g[col].replace([np.inf, -np.inf], np.nan).dropna()
                rec[f"{col}_mean"] = float(vals.mean()) if len(vals) else math.nan
            else:
                rec[f"{col}_mean"] = float(g[col].mean())
        for w in windows:
            cold = float(g[f"cold_{w}"].sum())
            inv = float(g[f"inv_{w}"].sum())
            idle = float(g[f"idle_gb_s_{w}"].sum())
            cost = float(g[f"cost_{w}"].sum())
            rec[f"csr_{w}_pct"] = 100.0 * cold / inv if inv > 0 else 0.0
            rec[f"cold_{w}"] = cold
            rec[f"inv_{w}"] = inv
            rec[f"idle_gb_s_{w}"] = idle
            rec[f"cost_{w}"] = cost
        rec["first10_active_csr_pct"] = float(
            100.0
            * g["first10_active_csr_pct"].mul(g["first10_active_inv"]).sum()
            / max(g["first10_active_inv"].sum(), 1.0)
        )
        records.append(rec)
    return pd.DataFrame(records)


def add_deltas(summary: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    key_cols = ["source", "kind", "rho"]
    for _, group in summary.groupby(key_cols, sort=False):
        group = group.copy()
        for base_name, suffix in [
            ("winter_frozen", "frozen"),
            ("ewma_0.1", "ewma_0p1"),
            ("winter_oracle_reset", "winter_oracle_reset"),
            ("winter_trigger", "winter_trigger"),
        ]:
            base = group[group["arm"] == base_name]
            if base.empty:
                continue
            base_row = base.iloc[0]
            for w in windows:
                group[f"delta_vs_{suffix}_csr_{w}_pp"] = (
                    group[f"csr_{w}_pct"] - float(base_row[f"csr_{w}_pct"])
                )
                group[f"delta_vs_{suffix}_cost_{w}"] = (
                    group[f"cost_{w}"] - float(base_row[f"cost_{w}"])
                )
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def _row_for_arm(group: pd.DataFrame, arm: str) -> pd.Series:
    sub = group[group["arm"] == arm]
    if sub.empty:
        raise ValueError(f"missing arm {arm}")
    return sub.iloc[0]


def _best_row(group: pd.DataFrame, arms: Sequence[str], metric: str) -> pd.Series:
    sub = group[group["arm"].isin(arms)].copy()
    if sub.empty:
        raise ValueError(f"missing arms {arms}")
    return sub.sort_values([metric, "arm"]).iloc[0]


def comparison_tables(summary: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    csr = f"csr_{window}_pct"
    cost = f"cost_{window}"
    detail: List[Dict[str, object]] = []
    for (source, kind, rho), group in summary.groupby(["source", "kind", "rho"], sort=True):
        ewma01 = _row_for_arm(group, "ewma_0.1")
        best_ewma = _best_row(group, EWMA_ARMS, csr)
        best_practical = _best_row(group, PRACTICAL_WINTER, csr)
        oracle_reset = _row_for_arm(group, "winter_oracle_reset")
        trigger = _row_for_arm(group, "winter_trigger")
        for arm in RESET_ARMS:
            row = _row_for_arm(group, arm)
            detail.append(
                {
                    "source": source,
                    "kind": kind,
                    "rho": float(rho),
                    "arm": arm,
                    "n_events": int(row["n_events"]),
                    "csr_pct": float(row[csr]),
                    "cost": float(row[cost]),
                    "ewma_0p1_csr_pct": float(ewma01[csr]),
                    "ewma_0p1_cost": float(ewma01[cost]),
                    "best_ewma_arm": str(best_ewma["arm"]),
                    "best_ewma_csr_pct": float(best_ewma[csr]),
                    "best_ewma_cost": float(best_ewma[cost]),
                    "best_practical_refit_arm": str(best_practical["arm"]),
                    "best_practical_refit_csr_pct": float(best_practical[csr]),
                    "best_practical_refit_cost": float(best_practical[cost]),
                    "winter_oracle_reset_csr_pct": float(oracle_reset[csr]),
                    "winter_oracle_reset_cost": float(oracle_reset[cost]),
                    "winter_trigger_csr_pct": float(trigger[csr]),
                    "winter_trigger_cost": float(trigger[cost]),
                    "delta_vs_ewma_0p1_csr_pp": float(row[csr] - ewma01[csr]),
                    "delta_vs_ewma_0p1_cost": float(row[cost] - ewma01[cost]),
                    "delta_vs_best_ewma_csr_pp": float(row[csr] - best_ewma[csr]),
                    "delta_vs_best_ewma_cost": float(row[cost] - best_ewma[cost]),
                    "delta_vs_best_practical_refit_csr_pp": float(row[csr] - best_practical[csr]),
                    "delta_vs_best_practical_refit_cost": float(row[cost] - best_practical[cost]),
                    "delta_vs_winter_oracle_reset_csr_pp": float(row[csr] - oracle_reset[csr]),
                    "delta_vs_winter_oracle_reset_cost": float(row[cost] - oracle_reset[cost]),
                    "delta_vs_winter_trigger_csr_pp": float(row[csr] - trigger[csr]),
                    "delta_vs_winter_trigger_cost": float(row[cost] - trigger[cost]),
                    "reset_count_post_mean": float(row["reset_count_post_mean"]),
                    "first_reset_tick_post_mean": float(row["first_reset_tick_post_mean"]),
                    "prior_route_ticks_post_mean": float(row["prior_route_ticks_post_mean"]),
                    "biased_route_ticks_post_mean": float(row["biased_route_ticks_post_mean"]),
                }
            )

    detail_df = pd.DataFrame(detail)
    count_rows: List[Dict[str, object]] = []
    for arm in RESET_ARMS:
        arm_df = detail_df[detail_df["arm"] == arm]
        for source in ["all"] + sorted(arm_df["source"].unique()):
            sdat = arm_df if source == "all" else arm_df[arm_df["source"] == source]
            count_rows.append(
                {
                    "source": source,
                    "arm": arm,
                    "n_condition_cells": int(len(sdat)),
                    "csr_lower_than_ewma_0p1": int(
                        (sdat["delta_vs_ewma_0p1_csr_pp"] < -EPS).sum()
                    ),
                    "csr_no_worse_than_ewma_0p1": int(
                        (sdat["delta_vs_ewma_0p1_csr_pp"] <= EPS).sum()
                    ),
                    "cost_lower_than_ewma_0p1": int(
                        (sdat["delta_vs_ewma_0p1_cost"] < -EPS).sum()
                    ),
                    "joint_lower_than_ewma_0p1": int(
                        (
                            (sdat["delta_vs_ewma_0p1_csr_pp"] < -EPS)
                            & (sdat["delta_vs_ewma_0p1_cost"] < -EPS)
                        ).sum()
                    ),
                    "csr_lower_than_best_ewma": int(
                        (sdat["delta_vs_best_ewma_csr_pp"] < -EPS).sum()
                    ),
                    "csr_no_worse_than_best_ewma": int(
                        (sdat["delta_vs_best_ewma_csr_pp"] <= EPS).sum()
                    ),
                    "cost_lower_than_best_ewma": int(
                        (sdat["delta_vs_best_ewma_cost"] < -EPS).sum()
                    ),
                    "joint_lower_than_best_ewma": int(
                        (
                            (sdat["delta_vs_best_ewma_csr_pp"] < -EPS)
                            & (sdat["delta_vs_best_ewma_cost"] < -EPS)
                        ).sum()
                    ),
                    "csr_lower_than_best_practical_refit": int(
                        (sdat["delta_vs_best_practical_refit_csr_pp"] < -EPS).sum()
                    ),
                    "cost_lower_than_best_practical_refit": int(
                        (sdat["delta_vs_best_practical_refit_cost"] < -EPS).sum()
                    ),
                    "mean_prior_route_ticks_post": float(sdat["prior_route_ticks_post_mean"].mean()),
                    "mean_biased_route_ticks_post": float(sdat["biased_route_ticks_post_mean"].mean()),
                }
            )
    return detail_df, pd.DataFrame(count_rows)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--model-path", type=Path, default=r23.DEFAULT_MODEL_PATH)
    parser.add_argument("--tag", default="")
    parser.add_argument("--events-json", type=Path, default=None)
    parser.add_argument("--prototype-data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--prototype-split", default="s1_train")
    parser.add_argument("--prototype-clusters", type=int, default=4)
    parser.add_argument("--prototype-func-cap", type=int, default=50)
    parser.add_argument("--synthetic-events", type=int, default=120)
    parser.add_argument("--natural-per-kind", type=int, default=30)
    parser.add_argument("--burnin", type=int, default=1440)
    parser.add_argument("--horizon", type=int, default=240)
    parser.add_argument("--step", type=int, default=60)
    parser.add_argument("--min-pre", type=int, default=20)
    parser.add_argument("--min-post", type=int, default=20)
    parser.add_argument("--scan-functions", type=int, default=0)
    parser.add_argument("--max-window-inv", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23023)
    parser.add_argument("--rho", type=float, nargs="+", default=[1.0, 10.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--l2", type=float, default=1e-2)
    parser.add_argument("--refit-every", type=int, default=10)
    parser.add_argument("--buffer-len", type=int, default=120)
    parser.add_argument("--sim-mode", choices=["numba", "exact"], default="numba")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    t0 = time.time()
    torch.set_num_threads(1)
    args.data_dir = args.data_dir.resolve()
    args.model_path = args.model_path.resolve()
    args.prototype_data_dir = args.prototype_data_dir.resolve()
    if args.events_json is not None:
        args.events_json = args.events_json.resolve()

    out_json, out_events_csv, out_summary_csv, out_compare_csv, out_counts_csv = output_paths(args.tag)
    counts = np.load(args.data_dir / "counts.npy", mmap_mode="r")
    features = np.load(args.data_dir / "features.npy", mmap_mode="r")
    if counts.shape[:2] != features.shape[:2]:
        raise RuntimeError(f"counts/features shape mismatch: {counts.shape} vs {features.shape}")
    n_funcs, t_len = counts.shape
    duration_mean, duration_std = r23.load_duration_arrays(args.data_dir, n_funcs, t_len)
    trainer = r23.load_trainer(args.device, in_features=features.shape[2], model_path=args.model_path)

    centroids, proto_w, proto_meta = build_prototype_bank(
        trainer=trainer,
        prototype_data_dir=args.prototype_data_dir,
        split_key=args.prototype_split,
        n_clusters=args.prototype_clusters,
        func_cap=args.prototype_func_cap,
        device=args.device,
    )

    windows = [15, 30, 60, int(args.horizon)]
    if args.events_json is not None:
        syn_events, nat_events, synthetic_kinds = load_events_json(args.events_json)
    else:
        synthetic_kinds = list(DEFAULT_SYNTHETIC_KINDS)
        syn_events = r23.choose_synthetic_events(
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
        nat_events = r23.choose_natural_events(
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
                centroids=centroids,
                proto_w=proto_w,
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
                device=args.device,
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
            centroids=centroids,
            proto_w=proto_w,
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
            device=args.device,
        )
        all_rows.extend(rows)
        done += 1
        if done % 10 == 0 or done == total_jobs:
            print(f"[progress] {done}/{total_jobs} event-configs complete", flush=True)

    rows_df = pd.DataFrame(all_rows)
    summary = add_deltas(summarize(rows_df, windows), windows)
    comparisons, counts_df = comparison_tables(summary, int(args.horizon))

    out_events_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(out_events_csv, index=False)
    summary.to_csv(out_summary_csv, index=False)
    comparisons.to_csv(out_compare_csv, index=False)
    counts_df.to_csv(out_counts_csv, index=False)

    payload = {
        "description": "R26 drift reset-to-prior audit",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset": {
            "data_dir": str(args.data_dir),
            "model_path": str(args.model_path),
            "n_funcs": int(n_funcs),
            "T": int(t_len),
            "counts_dtype": str(counts.dtype),
            "features_dtype": str(features.dtype),
        },
        "prototype": proto_meta,
        "synthetic_events": [ev.__dict__ for ev in syn_events],
        "natural_events": [ev.__dict__ for ev in nat_events],
        "synthetic_kinds": list(synthetic_kinds),
        "windows": windows,
        "reset_arms": list(RESET_ARMS),
        "n_rows": int(len(rows_df)),
        "elapsed_sec": time.time() - t0,
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2, default=float)

    print(f"[done] wrote {out_json}")
    print(f"[done] wrote {out_events_csv}")
    print(f"[done] wrote {out_summary_csv}")
    print(f"[done] wrote {out_compare_csv}")
    print(f"[done] wrote {out_counts_csv}")
    print(counts_df.to_string(index=False))


if __name__ == "__main__":
    main()
