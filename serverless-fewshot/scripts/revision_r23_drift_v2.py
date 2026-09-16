#!/usr/bin/env python3
"""R23 drift-v2 experiment for WINTER.

This script addresses the main weaknesses of the earlier drift runs:

* random drift onsets after a long warm-up, instead of one fixed midpoint;
* integer-consistent synthetic counts, instead of fractional scale-down counts
  that are later truncated by the simulator;
* several drift families beyond simple rate scaling;
* separated WINTER policies: frozen, scheduled, trigger-only, scheduled+trigger,
  and oracle-onset reset;
* short and long post-drift windows with cold-start and memory-cost summaries;
* mined natural shift candidates from the same trace.

Outputs are written under results/runs and results/tables.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

if hasattr(torch, "amp") and not hasattr(torch.amp, "GradScaler") and hasattr(torch, "cuda"):
    torch.amp.GradScaler = torch.cuda.amp.GradScaler

try:
    from numba import njit
except Exception:
    njit = None

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.features import features_from_counts
from src.decision.newsvendor import newsvendor_quantile
from src.meta.trainer import ANILMetaTrainer
from src.models.heads import N_QUANTILES
from src.drift.triggers import ConformalMonitor
from scripts.phase6_des import COLD_INIT, decisions_from_rates


DEFAULT_MODEL_PATH = ROOT / "results" / "runs" / "best_anil_ridge_s1_s0.pt"

L = 60
D = 64
KAPPA = 15.0
MEM_GB = 256.0 / 1024.0
TICK_SECONDS = 60.0
MAX_POOL = 200


@dataclass(frozen=True)
class Event:
    event_id: str
    source: str
    func_id: int
    onset: int
    kind: str
    pre_sum: int
    post_sum: int
    bucket: str
    meta: Dict[str, float]


def output_paths(tag: str) -> Tuple[Path, Path, Path]:
    suffix = f"_{tag}" if tag else ""
    return (
        ROOT / "results" / "runs" / f"revision_r23_drift_v2{suffix}.json",
        ROOT / "results" / "tables" / f"T_r23_drift_v2_events{suffix}.csv",
        ROOT / "results" / "tables" / f"T_r23_drift_v2_summary{suffix}.csv",
    )


def load_trainer(device: str, in_features: int, model_path: Path) -> ANILMetaTrainer:
    trainer = ANILMetaTrainer(
        body_type="tcn",
        head_type="ridge",
        in_features=in_features,
        embedding_dim=D,
        n_quantiles=N_QUANTILES,
        n_horizons=1,
        use_amp=False,
        device=device,
    )
    trainer.load(model_path)
    trainer.body.to(device)
    trainer.body.eval()
    return trainer


def make_segment_features(
    series_int: np.ndarray,
    app_corate: np.ndarray,
    start_abs: int,
) -> np.ndarray:
    return features_from_counts(series_int, t_offset=start_abs, app_corate=app_corate)


def embed_features(trainer: ANILMetaTrainer, feats: np.ndarray) -> np.ndarray:
    T = feats.shape[0]
    windows = np.zeros((T, L, feats.shape[1]), dtype=np.float32)
    for t in range(T):
        s0 = max(0, t - L + 1)
        block = feats[s0 : t + 1]
        windows[t, -len(block) :] = block
    xb = torch.from_numpy(windows).to(trainer.device)
    with torch.no_grad():
        phi = trainer.body(xb).detach().cpu().numpy().astype(np.float32)
    return phi


def pred_to_rate(pred_q: np.ndarray) -> float:
    # The ridge head is trained with repeated log1p targets across quantiles in
    # this codebase, so the median column is a stable point prediction.
    return float(np.expm1(np.maximum(np.median(pred_q), 0.0)))


def ridge_fit(phi: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    yy = np.repeat(y[:, None], N_QUANTILES, axis=1)
    eye = np.eye(phi.shape[1], dtype=np.float64)
    try:
        return np.linalg.solve(phi.T @ phi + float(l2) * eye, phi.T @ yy)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(phi.T @ phi + float(l2) * eye) @ (phi.T @ yy)


def winter_rates(
    phi: np.ndarray,
    counts: np.ndarray,
    event_rel: int,
    mode: str,
    l2: float,
    refit_every: int,
    buffer_len: int,
    trigger_alpha: float,
    trigger_warm: int,
    trigger_window: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    T = len(counts)
    y = np.log1p(np.asarray(counts, dtype=np.float32))
    rates = np.zeros(T, dtype=np.float32)
    wh: Optional[np.ndarray] = None
    buf_start = 0
    fires_pre = 0
    fires_post = 0
    fit_count_pre = 0
    fit_count_post = 0

    use_trigger = mode in {"trigger", "sched_trigger"}
    monitor = (
        ConformalMonitor(
            alpha=trigger_alpha,
            delta=0.15,
            m_consecutive=3,
            calib_size=trigger_window,
            window=30,
            min_calib=trigger_warm,
        )
        if use_trigger
        else None
    )

    for t in range(T):
        fired_now = False
        if wh is not None:
            pred_q = phi[t] @ wh
            rates[t] = pred_to_rate(pred_q)
            if monitor is not None:
                resid = abs(float(y[t]) - float(np.median(pred_q)))
                fired = monitor.update(resid)
                if fired:
                    fired_now = True
                    if t < event_rel:
                        fires_pre += 1
                    else:
                        fires_post += 1
                    buf_start = max(buf_start, t - 30)
        else:
            if t > 0:
                rates[t] = float(np.mean(counts[max(0, t - L) : t]))

        scheduled_due = t >= 20 and (t % refit_every == 0)
        trigger_due = fired_now
        oracle_due = mode == "oracle_reset" and t == event_rel + 20

        if mode == "frozen":
            allow_fit = scheduled_due and t < event_rel
        elif mode == "scheduled":
            allow_fit = scheduled_due
        elif mode == "trigger":
            allow_fit = scheduled_due if t < event_rel else trigger_due
        elif mode == "sched_trigger":
            allow_fit = scheduled_due or trigger_due
        elif mode == "oracle_reset":
            if t == event_rel:
                wh = None
                buf_start = event_rel
            allow_fit = (scheduled_due and t < event_rel) or oracle_due or (
                t > event_rel + 20 and scheduled_due
            )
        else:
            raise ValueError(f"unknown WINTER mode: {mode}")

        if allow_fit and t >= 20:
            s0 = max(buf_start, t - buffer_len)
            if t - s0 >= 20:
                wh = ridge_fit(phi[s0:t], y[s0:t], l2=l2)
                if t < event_rel:
                    fit_count_pre += 1
                else:
                    fit_count_post += 1
                trigger_due = False

    info = {
        "fires_pre": float(fires_pre),
        "fires_post": float(fires_post),
        "fit_count_pre": float(fit_count_pre),
        "fit_count_post": float(fit_count_post),
    }
    return rates, info


def ewma_rates(counts: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros(len(counts), dtype=np.float32)
    state = float(np.mean(counts[: min(60, len(counts))])) if len(counts) else 0.0
    for t, val in enumerate(counts):
        out[t] = state
        state = alpha * float(val) + (1.0 - alpha) * state
    return out


def fixed_keepalive_decisions(counts: np.ndarray, keep_alive_min: int) -> Tuple[np.ndarray, np.ndarray]:
    active = (counts > 0).astype(np.int32)
    conv = np.convolve(active, np.ones(keep_alive_min, dtype=np.int32), mode="full")[: len(counts)]
    prewarm = np.zeros(len(counts), dtype=bool)
    prewarm[1:] = conv[:-1] > 0
    ka = np.full(len(counts), keep_alive_min, dtype=np.float32)
    return prewarm, ka


def add_interval(arr: np.ndarray, start: float, end: float, value: float, end_time: float) -> None:
    if end <= start:
        return
    s = max(0.0, start)
    e = min(float(end_time), end)
    if e <= s:
        return
    while s < e:
        idx = int(s // TICK_SECONDS)
        if idx >= len(arr):
            break
        boundary = min(e, (idx + 1) * TICK_SECONDS)
        arr[idx] += (boundary - s) * value
        s = boundary


def simulate_roll(
    counts: np.ndarray,
    prewarm: np.ndarray,
    keep_alive_min: np.ndarray,
    durations_mean: np.ndarray,
    durations_std: np.ndarray,
    seed: int,
    cold_mu: float = COLD_INIT["mu"],
    cold_sigma: float = COLD_INIT["sigma"],
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    T = len(counts)
    end_time = float(T) * TICK_SECONDS
    roll_cold = np.zeros(T, dtype=np.float64)
    roll_total = np.zeros(T, dtype=np.float64)
    roll_idle = np.zeros(T, dtype=np.float64)
    roll_busy = np.zeros(T, dtype=np.float64)
    containers: List[List[float]] = []

    def evict_expired(now: float, ka_sec: float) -> None:
        nonlocal containers
        kept = []
        for c in containers:
            idle_start = max(c[0], c[1])
            if idle_start <= now - ka_sec:
                add_interval(roll_idle, idle_start, idle_start + ka_sec, MEM_GB, end_time)
            else:
                kept.append(c)
        containers = kept

    for t in range(T):
        n = int(counts[t])
        now0 = float(t) * TICK_SECONDS
        ka_sec = float(keep_alive_min[t]) * TICK_SECONDS
        evict_expired(now0, ka_sec)

        target = min(int(prewarm[t]), MAX_POOL)
        deficit = target - len(containers)
        if deficit > 0 and len(containers) < MAX_POOL:
            n_new = min(deficit, MAX_POOL - len(containers))
            inits = rng.lognormal(cold_mu, cold_sigma, n_new)
            for init in inits:
                init = float(init)
                add_interval(roll_busy, now0, now0 + init, MEM_GB, end_time)
                containers.append([now0 + init, now0 + init])

        if n <= 0:
            continue

        arrivals = now0 + np.sort(rng.random(n) * TICK_SECONDS)
        mean = max(float(durations_mean[t]), 1e-3)
        std = max(float(durations_std[t]), 0.01)
        service = np.maximum(rng.normal(mean, std, n), 0.01)

        for tau, dur in zip(arrivals, service):
            evict_expired(float(tau), ka_sec)
            roll_total[t] += 1.0
            best_idx = None
            best_idle = -1.0
            for ci, c in enumerate(containers):
                if c[0] <= tau and c[1] <= tau:
                    idle_start = max(c[0], c[1])
                    if idle_start > best_idle:
                        best_idle = idle_start
                        best_idx = ci
            if best_idx is not None:
                best = containers[best_idx]
                idle_start = max(best[0], best[1])
                add_interval(roll_idle, idle_start, float(tau), MEM_GB, end_time)
                add_interval(roll_busy, float(tau), float(tau + dur), MEM_GB, end_time)
                best[1] = float(tau + dur)
            else:
                roll_cold[t] += 1.0
                init = float(rng.lognormal(cold_mu, cold_sigma))
                add_interval(roll_busy, float(tau), float(tau + init + dur), MEM_GB, end_time)
                containers.append([float(tau + init), float(tau + init + dur)])

    final_ka = float(keep_alive_min[-1]) * TICK_SECONDS if T else 0.0
    for c in containers:
        idle_start = max(c[0], c[1])
        add_interval(roll_idle, idle_start, idle_start + final_ka, MEM_GB, end_time)

    return {
        "cold": roll_cold,
        "total": roll_total,
        "idle": roll_idle,
        "busy": roll_busy,
    }


if njit is not None:

    @njit(cache=True)
    def _add_interval_jit(arr, start, end, value, end_time):
        if end <= start:
            return
        s = start if start > 0.0 else 0.0
        e = end if end < end_time else end_time
        if e <= s:
            return
        while s < e:
            idx = int(s // TICK_SECONDS)
            if idx >= arr.shape[0]:
                break
            boundary = (idx + 1) * TICK_SECONDS
            if boundary > e:
                boundary = e
            arr[idx] += (boundary - s) * value
            s = boundary


    @njit(cache=True)
    def _simulate_roll_jit(
        counts,
        prewarm,
        keep_alive_min,
        durations_mean,
        durations_std,
        seed,
        cold_mu,
        cold_sigma,
    ):
        np.random.seed(seed)
        T = counts.shape[0]
        end_time = float(T) * TICK_SECONDS
        roll_cold = np.zeros(T, dtype=np.float64)
        roll_total = np.zeros(T, dtype=np.float64)
        roll_idle = np.zeros(T, dtype=np.float64)
        roll_busy = np.zeros(T, dtype=np.float64)
        ready = np.empty(MAX_POOL, dtype=np.float64)
        free = np.empty(MAX_POOL, dtype=np.float64)
        ncon = 0

        for t in range(T):
            n = int(counts[t])
            now0 = float(t) * TICK_SECONDS
            ka_sec = float(keep_alive_min[t]) * TICK_SECONDS

            i = 0
            while i < ncon:
                idle_start = ready[i] if ready[i] > free[i] else free[i]
                if idle_start <= now0 - ka_sec:
                    _add_interval_jit(roll_idle, idle_start, idle_start + ka_sec, MEM_GB, end_time)
                    ncon -= 1
                    ready[i] = ready[ncon]
                    free[i] = free[ncon]
                else:
                    i += 1

            target = int(prewarm[t])
            if target > MAX_POOL:
                target = MAX_POOL
            deficit = target - ncon
            if deficit > 0 and ncon < MAX_POOL:
                n_new = deficit
                if n_new > MAX_POOL - ncon:
                    n_new = MAX_POOL - ncon
                for _ in range(n_new):
                    init = np.exp(cold_mu + cold_sigma * np.random.randn())
                    _add_interval_jit(roll_busy, now0, now0 + init, MEM_GB, end_time)
                    ready[ncon] = now0 + init
                    free[ncon] = now0 + init
                    ncon += 1

            if n <= 0:
                continue

            offsets = np.sort(np.random.uniform(0.0, TICK_SECONDS, n))
            mean = float(durations_mean[t])
            if mean < 1e-3:
                mean = 1e-3
            std = float(durations_std[t])
            if std < 0.01:
                std = 0.01

            for k in range(n):
                tau = now0 + offsets[k]
                dur = mean + std * np.random.randn()
                if dur < 0.01:
                    dur = 0.01

                i = 0
                while i < ncon:
                    idle_start = ready[i] if ready[i] > free[i] else free[i]
                    if idle_start <= tau - ka_sec:
                        _add_interval_jit(roll_idle, idle_start, idle_start + ka_sec, MEM_GB, end_time)
                        ncon -= 1
                        ready[i] = ready[ncon]
                        free[i] = free[ncon]
                    else:
                        i += 1

                best = -1
                best_idle = -1.0
                for i in range(ncon):
                    if ready[i] <= tau and free[i] <= tau:
                        idle_start = ready[i] if ready[i] > free[i] else free[i]
                        if idle_start > best_idle:
                            best_idle = idle_start
                            best = i

                roll_total[t] += 1.0
                if best >= 0:
                    _add_interval_jit(roll_idle, best_idle, tau, MEM_GB, end_time)
                    _add_interval_jit(roll_busy, tau, tau + dur, MEM_GB, end_time)
                    free[best] = tau + dur
                else:
                    init = np.exp(cold_mu + cold_sigma * np.random.randn())
                    if ncon < MAX_POOL:
                        ready[ncon] = tau + init
                        free[ncon] = tau + init + dur
                        ncon += 1
                    _add_interval_jit(roll_busy, tau, tau + init + dur, MEM_GB, end_time)
                    roll_cold[t] += 1.0

        final_ka = float(keep_alive_min[T - 1]) * TICK_SECONDS if T else 0.0
        for i in range(ncon):
            idle_start = ready[i] if ready[i] > free[i] else free[i]
            _add_interval_jit(roll_idle, idle_start, idle_start + final_ka, MEM_GB, end_time)

        return roll_cold, roll_total, roll_idle, roll_busy

else:
    _simulate_roll_jit = None


def simulate_roll_numba(
    counts: np.ndarray,
    prewarm: np.ndarray,
    keep_alive_min: np.ndarray,
    durations_mean: np.ndarray,
    durations_std: np.ndarray,
    seed: int,
    cold_mu: float = COLD_INIT["mu"],
    cold_sigma: float = COLD_INIT["sigma"],
) -> Dict[str, np.ndarray]:
    if _simulate_roll_jit is None:
        raise RuntimeError("numba simulator requested but numba is unavailable")
    cold, total, idle, busy = _simulate_roll_jit(
        np.ascontiguousarray(counts.astype(np.int64)),
        np.ascontiguousarray(prewarm.astype(np.int64)),
        np.ascontiguousarray(keep_alive_min.astype(np.float64)),
        np.ascontiguousarray(durations_mean.astype(np.float64)),
        np.ascontiguousarray(durations_std.astype(np.float64)),
        int(seed),
        float(cold_mu),
        float(cold_sigma),
    )
    return {"cold": cold, "total": total, "idle": idle, "busy": busy}


def metrics_for_windows(
    sim: Dict[str, np.ndarray],
    event_rel: int,
    windows: Sequence[int],
    rho: int,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for w in windows:
        sl = slice(event_rel, min(event_rel + w, len(sim["cold"])))
        cold = float(np.sum(sim["cold"][sl]))
        total = float(np.sum(sim["total"][sl]))
        idle = float(np.sum(sim["idle"][sl]))
        busy = float(np.sum(sim["busy"][sl]))
        cost = idle + KAPPA * float(rho) * cold
        out[f"cold_{w}"] = cold
        out[f"inv_{w}"] = total
        out[f"csr_{w}_pct"] = 100.0 * cold / total if total > 0 else 0.0
        out[f"idle_gb_s_{w}"] = idle
        out[f"busy_gb_s_{w}"] = busy
        out[f"cost_{w}"] = cost
    active_idx = np.flatnonzero(sim["total"][event_rel:] > 0.0)[:10] + event_rel
    if len(active_idx):
        cold = float(np.sum(sim["cold"][active_idx]))
        total = float(np.sum(sim["total"][active_idx]))
        out["first10_active_csr_pct"] = 100.0 * cold / total if total > 0 else 0.0
        out["first10_active_inv"] = total
    else:
        out["first10_active_csr_pct"] = 0.0
        out["first10_active_inv"] = 0.0
    return out


def integer_base(series: np.ndarray) -> np.ndarray:
    return np.rint(np.maximum(series, 0.0)).astype(np.int64)


def scale_counts(post: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    src = post.astype(np.int64)
    if factor < 1.0:
        return rng.binomial(src, factor).astype(np.int64)
    extra = rng.poisson((factor - 1.0) * src).astype(np.int64)
    return src + extra


def ramp_counts(
    post: np.ndarray,
    target_factor: float,
    ramp_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.zeros_like(post, dtype=np.int64)
    for i, val in enumerate(post.astype(np.int64)):
        frac = min(1.0, float(i + 1) / float(max(1, ramp_len)))
        factor = 1.0 + frac * (target_factor - 1.0)
        out[i] = scale_counts(np.asarray([val], dtype=np.int64), factor, rng)[0]
    return out


def burst_compress(post: np.ndarray, compression: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(post, dtype=np.int64)
    block = 60
    keep = max(1, block // max(1, compression))
    for s in range(0, len(post), block):
        e = min(len(post), s + block)
        src = post[s:e].astype(np.int64)
        total = int(src.sum())
        if total <= 0:
            continue
        weights = src.astype(np.float64) + 0.25
        weights /= weights.sum()
        k = min(keep, e - s)
        slots = rng.choice(np.arange(e - s), size=k, replace=False, p=weights)
        slot_weights = weights[slots]
        slot_weights /= slot_weights.sum()
        out[s + slots] = rng.multinomial(total, slot_weights)
    return out


def phase_circular(post: np.ndarray, shift: int) -> np.ndarray:
    if len(post) == 0:
        return post.copy()
    return np.roll(post.astype(np.int64), int(shift))


def apply_synthetic_drift(
    base_segment: np.ndarray,
    event_rel: int,
    kind: str,
    rng: np.random.Generator,
) -> np.ndarray:
    out = base_segment.astype(np.int64).copy()
    post = out[event_rel:].copy()
    if kind == "scale_down_0.5":
        out[event_rel:] = scale_counts(post, 0.5, rng)
    elif kind == "scale_down_0.2":
        out[event_rel:] = scale_counts(post, 0.2, rng)
    elif kind == "scale_up_2":
        out[event_rel:] = scale_counts(post, 2.0, rng)
    elif kind == "scale_up_5":
        out[event_rel:] = scale_counts(post, 5.0, rng)
    elif kind == "ramp_down_0.2_120":
        out[event_rel:] = ramp_counts(post, 0.2, 120, rng)
    elif kind == "ramp_up_5_120":
        out[event_rel:] = ramp_counts(post, 5.0, 120, rng)
    elif kind == "burst_compress_4":
        out[event_rel:] = burst_compress(post, 4, rng)
    elif kind == "phase_circular_120":
        out[event_rel:] = phase_circular(post, 120)
    else:
        raise ValueError(f"unknown synthetic drift kind: {kind}")
    return out


def volume_bucket(pre_sum: int) -> str:
    if pre_sum < 100:
        return "low"
    if pre_sum < 1000:
        return "mid"
    return "high"


def choose_synthetic_events(
    counts: np.ndarray,
    max_events: int,
    seed: int,
    burnin: int,
    horizon: int,
    step: int,
    min_pre: int,
    min_post: int,
    scan_functions: int,
    max_window_inv: int,
) -> List[Event]:
    if max_events <= 0:
        return []
    rng = np.random.default_rng(seed)
    candidates: List[Event] = []
    n_funcs, T = counts.shape
    if scan_functions and scan_functions < n_funcs:
        func_iter = rng.choice(n_funcs, size=scan_functions, replace=False)
    else:
        func_iter = np.arange(n_funcs)
    for fi in func_iter:
        series = integer_base(np.asarray(counts[fi]))
        for onset in range(burnin, T - horizon + 1, step):
            pre = int(series[onset - 240 : onset].sum())
            post = int(series[onset : onset + 240].sum())
            if pre >= min_pre and post >= min_post:
                if max_window_inv and pre + 5 * post > max_window_inv:
                    continue
                active_frac = float(np.mean(series[onset - 240 : onset + 240] > 0))
                if active_frac <= 0.0:
                    continue
                candidates.append(
                    Event(
                        event_id=f"syn_f{fi}_t{onset}",
                        source="synthetic",
                        func_id=int(fi),
                        onset=int(onset),
                        kind="base",
                        pre_sum=pre,
                        post_sum=post,
                        bucket=volume_bucket(pre),
                        meta={"active_frac": active_frac},
                    )
                )

    if not candidates:
        raise RuntimeError("no synthetic event candidates found")

    selected: List[Event] = []
    for bucket in ["low", "mid", "high"]:
        bucket_events = [ev for ev in candidates if ev.bucket == bucket]
        rng.shuffle(bucket_events)
        take = min(len(bucket_events), max(1, max_events // 3))
        selected.extend(bucket_events[:take])
    if len(selected) < max_events:
        remaining = [ev for ev in candidates if ev.event_id not in {x.event_id for x in selected}]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_events - len(selected)])
    rng.shuffle(selected)
    return selected[:max_events]


def natural_event_score(pre: np.ndarray, post: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
    pre_sum = float(pre.sum())
    post_sum = float(post.sum())
    pre_active = float(np.mean(pre > 0))
    post_active = float(np.mean(post > 0))
    pre_mean = float(np.mean(pre))
    post_mean = float(np.mean(post))
    pre_var = float(np.var(pre))
    post_var = float(np.var(post))
    rate_ratio = (post_sum + 1.0) / (pre_sum + 1.0)
    duty_ratio = (post_active + 1e-3) / (pre_active + 1e-3)
    fano_pre = pre_var / max(pre_mean, 1e-6)
    fano_post = post_var / max(post_mean, 1e-6)
    fano_ratio = (fano_post + 1e-3) / (fano_pre + 1e-3)
    peak_ratio = (float(np.percentile(post, 95)) + 1.0) / (float(np.percentile(pre, 95)) + 1.0)
    meta = {
        "rate_ratio": rate_ratio,
        "duty_ratio": duty_ratio,
        "fano_ratio": fano_ratio,
        "peak_ratio": peak_ratio,
    }
    labels = [
        ("natural_growth", rate_ratio if rate_ratio >= 2.0 else 0.0),
        ("natural_collapse", 1.0 / rate_ratio if rate_ratio <= 0.5 else 0.0),
        ("natural_duty_up", duty_ratio if duty_ratio >= 2.0 else 0.0),
        ("natural_duty_down", 1.0 / duty_ratio if duty_ratio <= 0.5 else 0.0),
        ("natural_burst_up", fano_ratio if fano_ratio >= 2.0 and peak_ratio >= 1.5 else 0.0),
        ("natural_burst_down", 1.0 / fano_ratio if fano_ratio <= 0.5 else 0.0),
    ]
    label, score = max(labels, key=lambda x: x[1])
    return label, float(score), meta


def choose_natural_events(
    counts: np.ndarray,
    max_per_kind: int,
    seed: int,
    burnin: int,
    horizon: int,
    step: int,
    min_pre: int,
    min_post: int,
    scan_functions: int,
    max_window_inv: int,
) -> List[Event]:
    if max_per_kind <= 0:
        return []
    rng = np.random.default_rng(seed)
    pools: Dict[str, List[Tuple[float, Event]]] = {}
    n_funcs, T = counts.shape
    if scan_functions and scan_functions < n_funcs:
        func_iter = rng.choice(n_funcs, size=scan_functions, replace=False)
    else:
        func_iter = np.arange(n_funcs)
    for fi in func_iter:
        series = integer_base(np.asarray(counts[fi]))
        for onset in range(burnin, T - horizon + 1, step):
            pre = series[onset - 240 : onset]
            post = series[onset : onset + 240]
            pre_sum = int(pre.sum())
            post_sum = int(post.sum())
            if pre_sum < min_pre or post_sum < min_post:
                continue
            if max_window_inv and pre_sum + post_sum > max_window_inv:
                continue
            label, score, meta = natural_event_score(pre, post)
            if score <= 0.0:
                continue
            ev = Event(
                event_id=f"nat_f{fi}_t{onset}_{label}",
                source="natural",
                func_id=int(fi),
                onset=int(onset),
                kind=label,
                pre_sum=pre_sum,
                post_sum=post_sum,
                bucket=volume_bucket(pre_sum),
                meta=meta,
            )
            pools.setdefault(label, []).append((score, ev))

    selected: List[Event] = []
    for label, items in pools.items():
        # Keep strong shifts but randomize ties and near-ties.
        rng.shuffle(items)
        items.sort(key=lambda x: x[0], reverse=True)
        selected.extend(ev for _, ev in items[:max_per_kind])
    rng.shuffle(selected)
    return selected


def build_arms(
    phi: np.ndarray,
    counts: np.ndarray,
    event_rel: int,
    l2: float,
    refit_every: int,
    buffer_len: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    arms: Dict[str, np.ndarray] = {}
    infos: Dict[str, Dict[str, float]] = {}
    for mode in ["frozen", "scheduled", "trigger", "sched_trigger", "oracle_reset"]:
        rates, info = winter_rates(
            phi=phi,
            counts=counts,
            event_rel=event_rel,
            mode=mode,
            l2=l2,
            refit_every=refit_every,
            buffer_len=buffer_len,
            trigger_alpha=0.1,
            trigger_warm=30,
            trigger_window=120,
        )
        arms[f"winter_{mode}"] = rates
        infos[f"winter_{mode}"] = info

    for alpha in [0.1, 0.3, 0.5]:
        name = f"ewma_{alpha:.1f}"
        arms[name] = ewma_rates(counts, alpha)
        infos[name] = {}

    arms["oracle"] = np.asarray(counts, dtype=np.float32)
    infos["oracle"] = {}
    return arms, infos


def evaluate_event(
    ev: Event,
    drift_kind: str,
    counts_all: np.ndarray,
    features_all: np.ndarray,
    duration_mean_all: np.ndarray,
    duration_std_all: np.ndarray,
    trainer: ANILMetaTrainer,
    rng_seed: int,
    burnin: int,
    horizon: int,
    windows: Sequence[int],
    rhos: Sequence[int],
    seeds: Sequence[int],
    l2: float,
    refit_every: int,
    buffer_len: int,
    sim_mode: str,
) -> List[Dict[str, float]]:
    start = ev.onset - burnin
    end = ev.onset + horizon
    event_rel = burnin
    base = integer_base(counts_all[ev.func_id, start:end])

    if ev.source == "synthetic":
        rng = np.random.default_rng(rng_seed)
        series = apply_synthetic_drift(base, event_rel, drift_kind, rng)
        app_corate = features_all[ev.func_id, start:end, 10].astype(np.float32)
        feats = make_segment_features(series, app_corate, start)
        actual_kind = drift_kind
    else:
        series = base
        feats = features_all[ev.func_id, start:end].astype(np.float32)
        actual_kind = ev.kind

    phi = embed_features(trainer, feats)
    arms, infos = build_arms(
        phi=phi,
        counts=series,
        event_rel=event_rel,
        l2=l2,
        refit_every=refit_every,
        buffer_len=buffer_len,
    )

    dm = np.asarray(duration_mean_all[ev.func_id, start:end], dtype=np.float32)
    ds = np.asarray(duration_std_all[ev.func_id, start:end], dtype=np.float32)

    rows: List[Dict[str, float]] = []
    post_sum_actual = int(series[event_rel : event_rel + horizon].sum())
    pre_sum_actual = int(series[event_rel - 240 : event_rel].sum())
    active_actual = float(np.mean(series[event_rel : event_rel + horizon] > 0))
    for rho in rhos:
        tau_star = newsvendor_quantile(float(rho))
        arm_decisions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for arm_name, rates in arms.items():
            prewarm2, ka2 = decisions_from_rates(rates[None, :], tau_star)
            prewarm = prewarm2[0].astype(np.int32)
            ka = ka2[0].astype(np.float32)
            arm_decisions[arm_name] = (prewarm, ka)
        arm_decisions["keepalive_10"] = fixed_keepalive_decisions(series, 10)

        for seed in seeds:
            for arm_name, (prewarm, ka) in arm_decisions.items():
                sim_fn = simulate_roll_numba if sim_mode == "numba" and _simulate_roll_jit is not None else simulate_roll
                sim = sim_fn(
                    counts=series,
                    prewarm=prewarm,
                    keep_alive_min=ka,
                    durations_mean=dm,
                    durations_std=ds,
                    seed=seed,
                )
                metrics = metrics_for_windows(sim, event_rel, windows, int(rho))
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
                }
                for k, v in ev.meta.items():
                    row[f"meta_{k}"] = float(v)
                row.update(metrics)
                rows.append(row)
    return rows


def summarize(rows: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    groups = ["source", "kind", "rho", "arm"]
    records = []
    for keys, g in rows.groupby(groups, sort=True):
        rec = dict(zip(groups, keys))
        rec["n_event_seed"] = int(len(g))
        rec["n_events"] = int(g["event_id"].nunique())
        rec["post_sum_actual_mean"] = float(g["post_sum_actual"].mean())
        rec["post_active_frac_mean"] = float(g["post_active_frac"].mean())
        rec["fires_post_mean"] = float(g["fires_post"].mean())
        rec["fit_count_post_mean"] = float(g["fit_count_post"].mean())
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
    rows = []
    key_cols = ["source", "kind", "rho"]
    for _, group in summary.groupby(key_cols, sort=False):
        base = group[group["arm"] == "winter_frozen"]
        if base.empty:
            rows.append(group)
            continue
        base_row = base.iloc[0]
        group = group.copy()
        for w in windows:
            group[f"delta_vs_frozen_csr_{w}_pp"] = (
                group[f"csr_{w}_pct"] - float(base_row[f"csr_{w}_pct"])
            )
            group[f"delta_vs_frozen_cost_{w}"] = group[f"cost_{w}"] - float(base_row[f"cost_{w}"])
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def load_duration_arrays(data_dir: Path, n_funcs: int, T: int) -> Tuple[np.ndarray, np.ndarray]:
    stats_path = data_dir / "duration_stats.csv"
    stats = pd.read_csv(stats_path)
    mean_col = "dur_mean" if "dur_mean" in stats.columns else "mean_duration"
    std_col = "dur_std" if "dur_std" in stats.columns else "std_duration"
    mean = np.zeros((n_funcs, T), dtype=np.float32)
    std = np.zeros((n_funcs, T), dtype=np.float32)
    default_mean = float(stats[mean_col].median())
    default_std = float(stats[std_col].median())
    mean.fill(default_mean)
    std.fill(max(default_std, 1e-3))
    for row_idx, row in stats.iterrows():
        fi = row_idx
        if "func_id" in stats.columns:
            try:
                fi = int(row["func_id"])
            except (TypeError, ValueError):
                fi = row_idx
        if 0 <= int(fi) < n_funcs:
            mean[int(fi), :] = max(float(row[mean_col]), 1e-3)
            std[int(fi), :] = max(float(row[std_col]), 1e-3)
    return mean, std


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
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
    p.add_argument("--seed", type=int, default=23023)
    p.add_argument("--rho", type=int, nargs="+", default=[1, 10])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--l2", type=float, default=1e-2)
    p.add_argument("--refit-every", type=int, default=10)
    p.add_argument("--buffer-len", type=int, default=120)
    p.add_argument("--sim-mode", choices=["numba", "exact"], default="numba")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    t0 = time.time()
    args.data_dir = args.data_dir.resolve()
    args.model_path = args.model_path.resolve()
    out_json, out_events_csv, out_summary_csv = output_paths(args.tag)
    counts = np.load(args.data_dir / "counts.npy", mmap_mode="r")
    features = np.load(args.data_dir / "features.npy", mmap_mode="r")
    if counts.shape[:2] != features.shape[:2]:
        raise RuntimeError(f"counts/features shape mismatch: {counts.shape} vs {features.shape}")
    n_funcs, T = counts.shape
    duration_mean, duration_std = load_duration_arrays(args.data_dir, n_funcs, T)

    trainer = load_trainer(args.device, in_features=features.shape[2], model_path=args.model_path)
    windows = [15, 30, 60, args.horizon]
    synthetic_kinds = [
        "scale_down_0.5",
        "scale_down_0.2",
        "scale_up_2",
        "scale_up_5",
        "ramp_down_0.2_120",
        "ramp_up_5_120",
        "burst_compress_4",
        "phase_circular_120",
    ]

    syn_events = choose_synthetic_events(
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
    nat_events = choose_natural_events(
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
            drift_seed = args.seed + ev_i * 100 + kind_i
            rows = evaluate_event(
                ev=ev,
                drift_kind=kind,
                counts_all=counts,
                features_all=features,
                duration_mean_all=duration_mean,
                duration_std_all=duration_std,
                trainer=trainer,
                rng_seed=drift_seed,
                burnin=args.burnin,
                horizon=args.horizon,
                windows=windows,
                rhos=args.rho,
                seeds=args.seeds,
                l2=args.l2,
                refit_every=args.refit_every,
                buffer_len=args.buffer_len,
                sim_mode=args.sim_mode,
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
        )
        all_rows.extend(rows)
        done += 1
        if done % 10 == 0 or done == total_jobs:
            print(f"[progress] {done}/{total_jobs} event-configs complete", flush=True)

    rows_df = pd.DataFrame(all_rows)
    summary = add_deltas(summarize(rows_df, windows), windows)

    out_events_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(out_events_csv, index=False)
    summary.to_csv(out_summary_csv, index=False)

    payload = {
        "args": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        "dataset": {
            "data_dir": str(args.data_dir),
            "model_path": str(args.model_path),
            "n_funcs": int(n_funcs),
            "T": int(T),
            "counts_dtype": str(counts.dtype),
            "features_dtype": str(features.dtype),
        },
        "synthetic_events": [ev.__dict__ for ev in syn_events],
        "natural_events": [ev.__dict__ for ev in nat_events],
        "synthetic_kinds": synthetic_kinds,
        "windows": windows,
        "summary": summary.to_dict(orient="records"),
        "elapsed_sec": time.time() - t0,
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"[done] wrote {out_json}")
    print(f"[done] wrote {out_events_csv}")
    print(f"[done] wrote {out_summary_csv}")
    print(
        summary[
            [
                "source",
                "kind",
                "rho",
                "arm",
                "n_events",
                "csr_60_pct",
                "delta_vs_frozen_csr_60_pp",
                f"csr_{args.horizon}_pct",
                f"delta_vs_frozen_csr_{args.horizon}_pp",
                "fires_post_mean",
            ]
        ]
        .sort_values(["source", "kind", "rho", "csr_60_pct"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
