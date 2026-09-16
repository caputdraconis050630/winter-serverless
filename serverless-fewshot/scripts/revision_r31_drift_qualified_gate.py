#!/usr/bin/env python3
"""R31 drift-qualified WINTER-G full-trace design audit.

This script is a follow-up to R30.  R30 tested whether a detected mature
drift can be routed back to a reset-to-prior WINTER branch, but its learned
recovery branch still used the original full-trace feature context.  R31
separates three implementation choices that matter for the claim:

  * reset_mode=original    : R30-compatible head/support reset only;
  * reset_mode=context     : rebuild post-reset rolling features before TCN;
  * reset_mode=full_state  : context reset plus a post-reset EWMA state.

R31 also adds direction-aware triggers and a causal shadow-commit option.  It
exports DES jobs and summarizes their outputs without overwriting R30.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import COLD_INIT, decisions_from_rates, rates_ewma  # noqa: E402
from scripts.revision_r21_cross_provider_faithful import (  # noqa: E402
    paired_boot_ci,
    parse_float_list,
    parse_int_list,
    summarize_cell,
    weighted_share,
)
from src.data.features import features_from_counts  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.drift.triggers import ConformalMonitor  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402
from src.models.prototypes import PrototypeManager  # noqa: E402


RUNS = PROJECT_ROOT / "results" / "runs"
TABLES = PROJECT_ROOT / "results" / "tables"
P21 = PROJECT_ROOT / "data" / "processed"
P19 = PROJECT_ROOT / "data" / "processed_2019"
PHW = PROJECT_ROOT / "data" / "processed_huawei"

L = 60
D = 64
MED = N_QUANTILES // 2
MIN_INV = 100
AGE_MIN = 720
RESET_SUPPORT_TICKS = 20
MAX_RATE = 200.0
MAX_LOG_RATE = float(np.log1p(MAX_RATE))
FAITHFUL_BUFFER = 180
FAITHFUL_REFIT_EVERY = 60
FAITHFUL_ALPHA = 0.15
FULL_RHOS = [0.1, 1.0, 10.0, 100.0]
DEFAULT_SEEDS = list(range(10))


def _new_monitor(
    alpha: float = 0.1,
    delta: float = 0.15,
    m_consecutive: int = 3,
    calib_size: int = 120,
    window: int = 30,
    min_calib: int = 30,
) -> ConformalMonitor:
    return ConformalMonitor(
        alpha=float(alpha),
        delta=float(delta),
        m_consecutive=int(m_consecutive),
        calib_size=int(calib_size),
        window=int(window),
        min_calib=int(min_calib),
    )


def cum_prev_matrix(counts: np.ndarray) -> np.ndarray:
    cum = np.cumsum(counts.astype(np.int64), axis=1)
    return np.concatenate([np.zeros((counts.shape[0], 1), dtype=np.int64), cum[:, :-1]], axis=1)


def age_matrix(counts: np.ndarray) -> np.ndarray:
    n_funcs, t_len = counts.shape
    out = np.zeros((n_funcs, t_len), dtype=np.int64)
    for f in range(n_funcs):
        nz = np.flatnonzero(counts[f] > 0)
        if len(nz):
            t0 = int(nz[0])
            out[f, t0:] = np.arange(t_len - t0, dtype=np.int64)
    return out


def load_trainer(features_shape, model_path: Path, device: str):
    torch.set_num_threads(1)
    trainer = ANILMetaTrainer(
        body_type="tcn",
        head_type="ridge",
        in_features=features_shape[2],
        embedding_dim=D,
        n_quantiles=N_QUANTILES,
        n_horizons=1,
        device=device,
    )
    trainer.load(model_path)
    trainer.body.to(device)
    trainer.body.eval()
    return trainer


def build_prototypes_local(trainer, features, counts, train_idx, n_clusters: int, device: str):
    print(f"Building C_proto={n_clusters} prototypes from {len(train_idx)} train functions...")
    pm = PrototypeManager(n_clusters=n_clusters, device=device)
    feats_t = torch.from_numpy(features).float()
    counts_t = torch.from_numpy(counts).float()
    emb = pm.compute_embeddings(trainer.body, feats_t, train_idx)
    labels = pm.fit_clusters(emb)
    pm.compute_prototype_heads(
        trainer.body,
        trainer.head,
        feats_t,
        counts_t,
        train_idx,
        labels,
        n_quantiles=N_QUANTILES,
        n_horizons=1,
    )
    sizes = np.bincount(labels, minlength=n_clusters)
    print(f"  cluster sizes: {sizes.tolist()}", flush=True)
    return pm


@torch.no_grad()
def embed_all(features: np.ndarray, trainer, device: str, batch_size: int = 4096) -> np.ndarray:
    n_funcs, t_len, n_feat = features.shape
    feats_t = torch.from_numpy(np.ascontiguousarray(features)).float()
    out = np.empty((n_funcs, t_len, D), dtype=np.float32)
    trainer.body.eval()
    for t in range(t_len):
        if t < L:
            x_all = torch.zeros((n_funcs, L, n_feat), dtype=torch.float32)
            if t > 0:
                x_all[:, L - t :, :] = feats_t[:, :t, :]
        else:
            x_all = feats_t[:, t - L : t, :]
        chunks = []
        for s in range(0, n_funcs, batch_size):
            chunks.append(trainer.body(x_all[s : s + batch_size].to(device)).cpu())
        out[:, t, :] = torch.cat(chunks, dim=0).numpy()
        if t and t % 5000 == 0:
            print(f"      embeddings tick {t}/{t_len}", flush=True)
    return out


@torch.no_grad()
def prototype_rates_from_phi(
    phi: np.ndarray,
    pm,
    device: str,
    batch_size: int = 65536,
) -> Tuple[np.ndarray, np.ndarray]:
    n_funcs, t_len, _ = phi.shape
    flat = np.ascontiguousarray(phi.reshape(-1, D))
    rate_flat = np.empty(flat.shape[0], dtype=np.float32)
    idx_flat = np.empty(flat.shape[0], dtype=np.int64)
    cen = pm.centroids.to(device)
    proto_med = pm.proto_weights[:, :, MED].to(device)
    for s in range(0, flat.shape[0], batch_size):
        e = min(s + batch_size, flat.shape[0])
        x = torch.from_numpy(flat[s:e]).float().to(device)
        sims = torch.nn.functional.cosine_similarity(x.unsqueeze(1), cen.unsqueeze(0), dim=2)
        ids = sims.argmax(dim=1)
        pred = (x * proto_med[ids]).sum(dim=1)
        pred_np = pred.detach().cpu().numpy()
        rate_flat[s:e] = np.expm1(np.clip(pred_np, 0.0, MAX_LOG_RATE)).astype(np.float32)
        idx_flat[s:e] = ids.detach().cpu().numpy().astype(np.int64)
    return rate_flat.reshape(n_funcs, t_len), idx_flat.reshape(n_funcs, t_len)


def biased_ridge_fit(phi_s: np.ndarray, y_s: np.ndarray, w_proto: np.ndarray, l2: float) -> np.ndarray:
    yy = np.repeat(y_s[:, None], N_QUANTILES, axis=1)
    lhs = phi_s.T @ phi_s + float(l2) * np.eye(phi_s.shape[1], dtype=np.float64)
    rhs = phi_s.T @ yy + float(l2) * w_proto
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def pred_to_rate(log_pred: np.ndarray) -> float:
    return float(np.expm1(np.clip(float(np.median(log_pred)), 0.0, MAX_LOG_RATE)))


def faithful_rates_from_phi(phi: np.ndarray, counts: np.ndarray, trainer, pm, device: str) -> np.ndarray:
    n_funcs, t_len, _ = phi.shape
    lam = float(trainer.head.ridge_lambda.item())
    rates = np.zeros((n_funcs, t_len), dtype=np.float32)
    xbuf = torch.zeros(n_funcs, FAITHFUL_BUFFER, D, dtype=torch.float64)
    ybuf = torch.zeros(n_funcs, FAITHFUL_BUFFER, dtype=torch.float64)
    wcur = None
    eye = torch.eye(D, dtype=torch.float64, device=device)
    cen = pm.centroids.to(device)
    proto_med = pm.proto_weights[:, :, MED].double().to(device)

    ewma_state = np.zeros(n_funcs, dtype=np.float64)
    for t in range(min(L, t_len)):
        rates[:, t] = np.expm1(np.clip(ewma_state, 0.0, MAX_LOG_RATE)).astype(np.float32)
        prev = counts[:, max(t - 1, 0)].astype(np.float64)
        ewma_state = FAITHFUL_ALPHA * np.log1p(prev) + (1.0 - FAITHFUL_ALPHA) * ewma_state

    for t in range(L, t_len):
        phi_t = torch.from_numpy(np.ascontiguousarray(phi[:, t, :])).double()
        prev = counts[:, max(t - 1, 0)].astype(np.float64)
        ewma_state = FAITHFUL_ALPHA * np.log1p(prev) + (1.0 - FAITHFUL_ALPHA) * ewma_state
        slot = (t - L) % FAITHFUL_BUFFER
        xbuf[:, slot] = phi_t
        ybuf[:, slot] = torch.from_numpy(np.log1p(prev))
        n_filled = min(t - L + 1, FAITHFUL_BUFFER)
        if t % FAITHFUL_REFIT_EVERY == 0 and t > L + 30 and n_filled >= RESET_SUPPORT_TICKS:
            phi_g = phi_t.float().to(device)
            sims = torch.nn.functional.cosine_similarity(phi_g.unsqueeze(1), cen.unsqueeze(0), dim=2)
            wp = proto_med[sims.argmax(dim=1)]
            xv = xbuf[:, :n_filled].to(device)
            yv = ybuf[:, :n_filled].to(device)
            lhs = xv.transpose(1, 2) @ xv + lam * eye
            rhs = (xv.transpose(1, 2) @ yv.unsqueeze(-1)).squeeze(-1) + lam * wp
            wcur = torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1).cpu()
        if wcur is not None:
            pred = (phi_t * wcur).sum(dim=1).numpy()
            rates[:, t] = np.expm1(np.clip(pred, 0.0, MAX_LOG_RATE)).astype(np.float32)
        else:
            rates[:, t] = np.expm1(np.clip(ewma_state, 0.0, MAX_LOG_RATE)).astype(np.float32)
        if t and t % 5000 == 0:
            print(f"      faithful tick {t}/{t_len}", flush=True)
    return rates


def current_gate_rates(proto, ewma, faithful, cum_prev, age):
    zero = cum_prev == 0
    count_insufficient = (cum_prev > 0) & (cum_prev < MIN_INV)
    initial_learned = (cum_prev >= MIN_INV) & (age < AGE_MIN)
    mature = (cum_prev >= MIN_INV) & ~initial_learned
    rates = ewma.copy().astype(np.float32)
    rates[zero] = proto[zero]
    rates[initial_learned] = faithful[initial_learned]
    return rates, {
        "zero_history": zero,
        "count_insufficient_ewma": count_insufficient,
        "initial_learned": initial_learned,
        "gate_mature_ewma": mature,
    }


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    reset_mode: str = "context"
    trigger_mode: str = "signed_under"
    k_drift: int = 100
    recovery_ttl: int = 60
    stable_exit: int = 30
    cooldown: int = 720
    shadow_window: int = 0
    min_miss_gain: float = 1.0
    max_extra_prewarm_ratio: float = 2.0
    shadow_rho: float = 10.0
    trigger_alpha: float = 0.1
    trigger_delta: float = 0.35
    trigger_m: int = 8
    trigger_calib_size: int = 240
    trigger_window: int = 90
    trigger_min_calib: int = 120
    min_under_log: float = 0.0
    l2: float = 1e-2
    refit_every: int = 10
    buffer_len: int = 120


@dataclass
class EpochCache:
    start: int
    end: int
    phi: Optional[np.ndarray]
    proto_rate: Optional[np.ndarray]
    proto_idx: Optional[np.ndarray]
    ewma_rate: Optional[np.ndarray]


def selected_rhos(args) -> List[float]:
    return parse_float_list(args.rhos) if args.rhos else FULL_RHOS


def selected_seeds(args) -> List[int]:
    return parse_int_list(args.seeds) if args.seeds else DEFAULT_SEEDS


def _slug_float(x: float) -> str:
    return ("%g" % float(x)).replace(".", "p").replace("-", "m")


def _candidate_name(spec: CandidateSpec) -> str:
    if spec.name:
        return spec.name
    return (
        "G_WE_A720DQ_"
        f"{spec.reset_mode}_{spec.trigger_mode}_"
        f"S{spec.shadow_window}_K{spec.k_drift}_"
        f"T{spec.recovery_ttl}_CD{spec.cooldown}"
    )


def _base_candidate(**kwargs) -> CandidateSpec:
    spec = CandidateSpec(name="", **kwargs)
    return replace(spec, name=_candidate_name(spec))


def preset_candidates(preset: str) -> List[CandidateSpec]:
    """Return named candidate grids.

    The default uses the R30 very-strict detector constants to control cost,
    then varies only reset semantics and action selection.
    """
    if preset == "none":
        return []
    if preset == "smoke":
        return [
            _base_candidate(reset_mode="original", trigger_mode="abs", k_drift=100),
            _base_candidate(reset_mode="context", trigger_mode="abs", k_drift=100),
            _base_candidate(reset_mode="context", trigger_mode="signed_under", k_drift=100),
            _base_candidate(
                reset_mode="full_state",
                trigger_mode="signed_under",
                shadow_window=20,
                k_drift=20,
            ),
        ]
    if preset == "s1_design":
        specs: List[CandidateSpec] = []
        for reset_mode in ["original", "context", "full_state"]:
            for trigger_mode in ["abs", "signed_under"]:
                for shadow_window in [0, 20]:
                    for k_drift in [20, 100]:
                        specs.append(
                            _base_candidate(
                                reset_mode=reset_mode,
                                trigger_mode=trigger_mode,
                                shadow_window=shadow_window,
                                k_drift=k_drift,
                            )
                        )
        return specs
    if preset == "focused":
        return [
            _base_candidate(reset_mode="context", trigger_mode="signed_under", k_drift=20),
            _base_candidate(reset_mode="context", trigger_mode="signed_under", k_drift=100),
            _base_candidate(
                reset_mode="full_state",
                trigger_mode="signed_under",
                shadow_window=10,
                k_drift=20,
            ),
            _base_candidate(
                reset_mode="full_state",
                trigger_mode="signed_under",
                shadow_window=20,
                k_drift=20,
            ),
            _base_candidate(
                reset_mode="full_state",
                trigger_mode="abs",
                shadow_window=20,
                k_drift=20,
            ),
        ]
    if preset == "frozen_candidate":
        return [
            _base_candidate(
                reset_mode="full_state",
                trigger_mode="signed_under",
                shadow_window=20,
                k_drift=20,
            )
        ]
    if preset == "final_policy":
        return [
            CandidateSpec(
                name="G_WE_A720DQ_context_under_S0_K20_T60_CD3600_D025",
                reset_mode="context",
                trigger_mode="signed_under",
                shadow_window=0,
                k_drift=20,
                recovery_ttl=60,
                stable_exit=30,
                cooldown=3600,
                trigger_alpha=0.1,
                trigger_delta=0.25,
                trigger_m=3,
                trigger_calib_size=120,
                trigger_window=30,
                trigger_min_calib=30,
                min_under_log=0.0,
                l2=1e-2,
                refit_every=10,
                buffer_len=120,
            )
        ]
    raise ValueError(f"unknown preset: {preset}")


def parse_candidate_spec(raw: str) -> CandidateSpec:
    """Parse a comma-separated key=value candidate override."""
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
        vals["name"] = ""

    int_keys = {
        "k_drift",
        "recovery_ttl",
        "stable_exit",
        "cooldown",
        "shadow_window",
        "trigger_m",
        "trigger_calib_size",
        "trigger_window",
        "trigger_min_calib",
        "refit_every",
        "buffer_len",
    }
    float_keys = {
        "min_miss_gain",
        "max_extra_prewarm_ratio",
        "shadow_rho",
        "trigger_alpha",
        "trigger_delta",
        "min_under_log",
        "l2",
    }
    for key in list(vals):
        if key in int_keys:
            vals[key] = int(vals[key])
        elif key in float_keys:
            vals[key] = float(vals[key])

    spec = CandidateSpec(**vals)  # type: ignore[arg-type]
    if not spec.name:
        spec = replace(spec, name=_candidate_name(spec))
    return spec


@torch.no_grad()
def embed_segment_strict(
    features_seg: np.ndarray,
    trainer,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Embed a single-function segment with strictly previous-tick windows."""
    t_len, n_feat = features_seg.shape
    feats_t = torch.from_numpy(np.ascontiguousarray(features_seg)).float()
    out = np.empty((t_len, D), dtype=np.float32)
    trainer.body.eval()
    for start in range(0, t_len, batch_size):
        end = min(start + batch_size, t_len)
        wins = torch.zeros((end - start, L, n_feat), dtype=torch.float32)
        for i, t in enumerate(range(start, end)):
            s0 = max(0, t - L)
            block = feats_t[s0:t]
            if len(block):
                wins[i, L - len(block) :, :] = block
        out[start:end] = trainer.body(wins.to(device)).detach().cpu().numpy()
    return out


def build_epoch_cache(
    f: int,
    start: int,
    length: int,
    reset_mode: str,
    counts: np.ndarray,
    base_features: np.ndarray,
    trace_t_offset: int,
    trainer,
    pm,
    device: str,
) -> EpochCache:
    end = min(counts.shape[1], max(start + 1, start + int(length)))
    if reset_mode == "original":
        return EpochCache(start=start, end=end, phi=None, proto_rate=None, proto_idx=None, ewma_rate=None)

    seg_counts = counts[f, start:end].astype(np.float32)
    app_corate = base_features[f, start:end, 10].astype(np.float32)
    feats = features_from_counts(
        seg_counts[None, :],
        t_offset=int(trace_t_offset) + int(start),
        app_corate=app_corate[None, :],
    )[0]
    phi_seg = embed_segment_strict(feats, trainer, device=device)
    proto_rate, proto_idx = prototype_rates_from_phi(phi_seg[None, :, :], pm, device=device)
    ewma_rate = rates_ewma(seg_counts[None, :])[0].astype(np.float32)
    return EpochCache(
        start=start,
        end=end,
        phi=phi_seg,
        proto_rate=proto_rate[0].astype(np.float32),
        proto_idx=proto_idx[0].astype(np.int64),
        ewma_rate=ewma_rate,
    )


def trigger_score(
    mode: str,
    y_log: float,
    pred_log: float,
    min_under_log: float,
) -> float:
    if mode == "abs":
        return abs(y_log - pred_log)
    if mode == "signed_under":
        return max(0.0, y_log - pred_log - float(min_under_log))
    raise ValueError(f"unknown trigger_mode: {mode}")


def shadow_decision_score(
    counts: np.ndarray,
    base_rates: np.ndarray,
    candidate_rates: np.ndarray,
    rho: float,
) -> Dict[str, float]:
    tau = newsvendor_quantile(float(rho))
    base_pw, _ = decisions_from_rates(base_rates[None, :], tau)
    cand_pw, _ = decisions_from_rates(candidate_rates[None, :], tau)
    actual = counts.astype(np.float64)
    b = base_pw[0].astype(np.float64)
    c = cand_pw[0].astype(np.float64)
    base_miss = float(np.maximum(actual - b, 0.0).sum())
    cand_miss = float(np.maximum(actual - c, 0.0).sum())
    extra_prewarm = float(np.maximum(c - b, 0.0).sum())
    base_prewarm = float(np.maximum(b, 0.0).sum())
    return {
        "miss_gain": base_miss - cand_miss,
        "extra_prewarm": extra_prewarm,
        "extra_prewarm_ratio": extra_prewarm / max(base_prewarm, 1.0),
    }


def drift_qualified_rates(
    spec: CandidateSpec,
    phi: np.ndarray,
    counts: np.ndarray,
    base_features: np.ndarray,
    trace_t_offset: int,
    current: np.ndarray,
    ewma: np.ndarray,
    proto: np.ndarray,
    proto_idx: np.ndarray,
    pm,
    cum_prev: np.ndarray,
    age: np.ndarray,
    trainer,
    device: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if spec.reset_mode not in {"original", "context", "full_state"}:
        raise ValueError(f"unknown reset_mode: {spec.reset_mode}")
    if spec.trigger_mode not in {"abs", "signed_under"}:
        raise ValueError(f"unknown trigger_mode: {spec.trigger_mode}")

    n_funcs, t_len = counts.shape
    y = np.log1p(counts.astype(np.float64))
    cprefix = np.concatenate(
        [np.zeros((n_funcs, 1), dtype=np.int64), np.cumsum(counts.astype(np.int64), axis=1)],
        axis=1,
    )
    proto_w = pm.proto_weights.detach().cpu().numpy().astype(np.float64)
    rates = current.copy().astype(np.float32)

    route_recovery = np.zeros((n_funcs, t_len), dtype=bool)
    route_recovery_prior = np.zeros((n_funcs, t_len), dtype=bool)
    route_recovery_learned = np.zeros((n_funcs, t_len), dtype=bool)
    route_recovery_ewma = np.zeros((n_funcs, t_len), dtype=bool)
    route_shadow_candidate = np.zeros((n_funcs, t_len), dtype=bool)

    fires = np.zeros(n_funcs, dtype=np.int64)
    suppressed = np.zeros(n_funcs, dtype=np.int64)
    shadow_entries = np.zeros(n_funcs, dtype=np.int64)
    shadow_commits = np.zeros(n_funcs, dtype=np.int64)
    shadow_rejects = np.zeros(n_funcs, dtype=np.int64)
    recovery_entries = np.zeros(n_funcs, dtype=np.int64)
    fits = np.zeros(n_funcs, dtype=np.int64)
    shadow_miss_gain = np.zeros(n_funcs, dtype=np.float64)
    shadow_extra_prewarm = np.zeros(n_funcs, dtype=np.float64)

    epoch_len = (
        int(spec.shadow_window)
        + max(int(spec.recovery_ttl), int(spec.stable_exit))
        + int(spec.buffer_len)
        + int(spec.refit_every)
        + L
        + 5
    )

    for f in range(n_funcs):
        monitor = _new_monitor(
            alpha=spec.trigger_alpha,
            delta=spec.trigger_delta,
            m_consecutive=spec.trigger_m,
            calib_size=spec.trigger_calib_size,
            window=spec.trigger_window,
            min_calib=spec.trigger_min_calib,
        )
        state = "stable"
        support_start: Optional[int] = None
        recovery_start: Optional[int] = None
        shadow_start: Optional[int] = None
        shadow_end: Optional[int] = None
        last_alarm = -10**9
        last_entry = -10**9
        wh: Optional[np.ndarray] = None
        epoch: Optional[EpochCache] = None
        shadow_rates: Dict[int, float] = {}

        def ensure_epoch(start_tick: int, tick: int) -> EpochCache:
            nonlocal epoch
            if epoch is None or epoch.start != start_tick or tick >= epoch.end:
                length = max(epoch_len, tick - start_tick + 1)
                epoch = build_epoch_cache(
                    f=f,
                    start=start_tick,
                    length=length,
                    reset_mode=spec.reset_mode,
                    counts=counts,
                    base_features=base_features,
                    trace_t_offset=trace_t_offset,
                    trainer=trainer,
                    pm=pm,
                    device=device,
                )
            return epoch

        def phi_at(tick: int) -> np.ndarray:
            assert support_start is not None
            ep = ensure_epoch(support_start, tick)
            if ep.phi is None:
                return phi[f, tick, :].astype(np.float64)
            return ep.phi[tick - ep.start, :].astype(np.float64)

        def proto_at(tick: int) -> Tuple[float, int]:
            assert support_start is not None
            ep = ensure_epoch(support_start, tick)
            if ep.proto_rate is None or ep.proto_idx is None:
                return float(proto[f, tick]), int(proto_idx[f, tick])
            rel = tick - ep.start
            return float(ep.proto_rate[rel]), int(ep.proto_idx[rel])

        def ewma_at(tick: int) -> float:
            assert support_start is not None
            ep = ensure_epoch(support_start, tick)
            if spec.reset_mode == "full_state" and ep.ewma_rate is not None:
                return float(ep.ewma_rate[tick - ep.start])
            return float(ewma[f, tick])

        def fit_after_observe(tick: int) -> None:
            nonlocal wh
            assert support_start is not None
            support_ticks = tick - support_start
            if support_ticks < RESET_SUPPORT_TICKS:
                return
            if support_ticks % max(1, int(spec.refit_every)) != 0:
                return
            s0 = max(int(support_start), tick - int(spec.buffer_len))
            if tick - s0 < RESET_SUPPORT_TICKS:
                return
            xmat = np.stack([phi_at(j) for j in range(s0, tick)], axis=0)
            _, pidx = proto_at(tick)
            wh = biased_ridge_fit(
                xmat.astype(np.float64),
                y[f, s0:tick],
                proto_w[int(pidx)],
                l2=float(spec.l2),
            )
            fits[f] += 1

        def candidate_rate(tick: int) -> Tuple[float, str]:
            assert support_start is not None
            post_cum = int(cprefix[f, tick] - cprefix[f, int(support_start)])
            if post_cum == 0:
                pr, _ = proto_at(tick)
                return pr, "prior"
            if post_cum < int(spec.k_drift):
                return ewma_at(tick), "ewma"
            if wh is None:
                pr, _ = proto_at(tick)
                return pr, "prior"
            pred = phi_at(tick) @ wh
            return pred_to_rate(pred), "learned"

        for t in range(t_len):
            if state == "recovery" and recovery_start is not None:
                if (
                    (t - int(recovery_start)) >= int(spec.recovery_ttl)
                    and (t - int(last_alarm)) >= int(spec.stable_exit)
                ):
                    state = "stable"
                    support_start = None
                    recovery_start = None
                    shadow_start = None
                    shadow_end = None
                    wh = None
                    epoch = None
                    shadow_rates = {}
                    monitor = _new_monitor(
                        alpha=spec.trigger_alpha,
                        delta=spec.trigger_delta,
                        m_consecutive=spec.trigger_m,
                        calib_size=spec.trigger_calib_size,
                        window=spec.trigger_window,
                        min_calib=spec.trigger_min_calib,
                    )

            if state == "recovery" and support_start is not None:
                rate, route = candidate_rate(t)
                rates[f, t] = rate
                route_recovery[f, t] = True
                route_recovery_prior[f, t] = route == "prior"
                route_recovery_learned[f, t] = route == "learned"
                route_recovery_ewma[f, t] = route == "ewma"
                fit_after_observe(t)
                continue

            if state == "shadow" and support_start is not None and shadow_start is not None:
                rates[f, t] = current[f, t]
                cand_rate, _ = candidate_rate(t)
                shadow_rates[t] = cand_rate
                route_shadow_candidate[f, t] = True
                fit_after_observe(t)
                if shadow_end is not None and t >= shadow_end:
                    ticks = np.arange(int(shadow_start), t + 1, dtype=np.int64)
                    cand_vec = np.asarray([shadow_rates[int(j)] for j in ticks], dtype=np.float32)
                    score = shadow_decision_score(
                        counts=counts[f, ticks].astype(np.float32),
                        base_rates=current[f, ticks].astype(np.float32),
                        candidate_rates=cand_vec,
                        rho=float(spec.shadow_rho),
                    )
                    shadow_miss_gain[f] += score["miss_gain"]
                    shadow_extra_prewarm[f] += score["extra_prewarm"]
                    ok_gain = score["miss_gain"] >= float(spec.min_miss_gain)
                    ok_cost = score["extra_prewarm_ratio"] <= float(spec.max_extra_prewarm_ratio)
                    if ok_gain and ok_cost and t + 1 < t_len:
                        shadow_commits[f] += 1
                        recovery_entries[f] += 1
                        state = "recovery"
                        recovery_start = t + 1
                        last_entry = t + 1
                    else:
                        shadow_rejects[f] += 1
                        state = "stable"
                        support_start = None
                        recovery_start = None
                        shadow_start = None
                        shadow_end = None
                        wh = None
                        epoch = None
                        shadow_rates = {}
                    monitor = _new_monitor(
                        alpha=spec.trigger_alpha,
                        delta=spec.trigger_delta,
                        m_consecutive=spec.trigger_m,
                        calib_size=spec.trigger_calib_size,
                        window=spec.trigger_window,
                        min_calib=spec.trigger_min_calib,
                    )
                continue

            rates[f, t] = current[f, t]
            mature = bool(cum_prev[f, t] >= MIN_INV and age[f, t] >= AGE_MIN)
            if not mature:
                continue
            pred_log = math.log1p(max(float(ewma[f, t]), 0.0))
            resid = trigger_score(
                spec.trigger_mode,
                y_log=float(y[f, t]),
                pred_log=pred_log,
                min_under_log=float(spec.min_under_log),
            )
            if monitor.update(resid):
                fires[f] += 1
                last_alarm = t
                if (t - last_entry) < int(spec.cooldown):
                    suppressed[f] += 1
                    continue
                start = t + 1
                if start >= t_len:
                    continue
                support_start = start
                wh = None
                epoch = None
                shadow_rates = {}
                if int(spec.shadow_window) > 0:
                    state = "shadow"
                    shadow_entries[f] += 1
                    shadow_start = start
                    shadow_end = min(t_len - 1, start + int(spec.shadow_window) - 1)
                    ensure_epoch(start, int(shadow_end))
                else:
                    state = "recovery"
                    recovery_start = start
                    recovery_entries[f] += 1
                    last_entry = start
                    ensure_epoch(start, start)

    info = {
        "route_recovery": route_recovery,
        "route_recovery_prior": route_recovery_prior,
        "route_recovery_learned": route_recovery_learned,
        "route_recovery_ewma": route_recovery_ewma,
        "route_shadow_candidate": route_shadow_candidate,
        "fires": fires,
        "suppressed_by_cooldown": suppressed,
        "shadow_entries": shadow_entries,
        "shadow_commits": shadow_commits,
        "shadow_rejects": shadow_rejects,
        "recovery_entries": recovery_entries,
        "recovery_fits": fits,
        "shadow_miss_gain": shadow_miss_gain,
        "shadow_extra_prewarm": shadow_extra_prewarm,
    }
    return rates, info


def route_record(mask: np.ndarray, weights: np.ndarray) -> float:
    return weighted_share(mask, weights)


def candidate_info_record(info: Dict[str, object], weights: np.ndarray) -> Dict[str, float]:
    out = {
        "route_recovery": route_record(info["route_recovery"], weights),
        "route_recovery_prior": route_record(info["route_recovery_prior"], weights),
        "route_recovery_learned": route_record(info["route_recovery_learned"], weights),
        "route_recovery_ewma": route_record(info["route_recovery_ewma"], weights),
        "route_shadow_candidate": route_record(info["route_shadow_candidate"], weights),
    }
    for src, dst in [
        ("fires", "mean_fires_per_function"),
        ("suppressed_by_cooldown", "mean_suppressed_by_cooldown_per_function"),
        ("shadow_entries", "mean_shadow_entries_per_function"),
        ("shadow_commits", "mean_shadow_commits_per_function"),
        ("shadow_rejects", "mean_shadow_rejects_per_function"),
        ("recovery_entries", "mean_recovery_entries_per_function"),
        ("recovery_fits", "mean_recovery_fits_per_function"),
        ("shadow_miss_gain", "mean_shadow_miss_gain_per_function"),
        ("shadow_extra_prewarm", "mean_shadow_extra_prewarm_per_function"),
    ]:
        out[dst] = float(np.average(np.asarray(info[src], dtype=np.float64), weights=weights))
    return out


def export_azure2021(args) -> None:
    torch.set_num_threads(1)
    features = np.load(P21 / "features.npy")
    counts_all = np.load(P21 / "counts.npy")
    splits = np.load(P21 / "splits.npz")
    dur_df = pd.read_csv(P21 / "duration_stats.csv")
    device = args.device
    model_path = args.model_path.resolve()
    trainer = load_trainer(features.shape, model_path, device)
    pm = build_prototypes_local(
        trainer,
        features,
        counts_all,
        splits["s1_train"],
        n_clusters=int(args.prototype_clusters),
        device=device,
    )
    candidates = preset_candidates(args.preset)
    candidates.extend(parse_candidate_spec(raw) for raw in args.candidate)
    names = [c.name for c in candidates]
    if len(names) != len(set(names)):
        raise ValueError("duplicate candidate names in R31 export")

    rhos = selected_rhos(args)
    seeds = selected_seeds(args)
    jobs_root = Path(args.jobs_root) if args.jobs_root else RUNS / f"des_jobs_r31_azure2021_{args.preset}"
    out = (
        Path(args.manifest_out)
        if args.manifest_out
        else RUNS / f"revision_r31_drift_qualified_azure2021_{args.preset}_manifest.json"
    )
    manifest: Dict[str, object] = {
        "analysis": "R31 drift-qualified WINTER-G full-trace design audit",
        "trace": "azure2021",
        "checkpoint": str(model_path),
        "checkpoint_md5": hashlib.md5(open(model_path, "rb").read()).hexdigest(),
        "prototype_clusters": int(args.prototype_clusters),
        "prototype_source": "azure2021::s1_train",
        "min_invocations": MIN_INV,
        "age_min": AGE_MIN,
        "preset": args.preset,
        "candidates": [asdict(c) for c in candidates],
        "rhos": rhos,
        "seeds": seeds,
        "splits": {},
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
        feats = features[rows].astype(np.float32)
        trace_t_offset = 0
        if split == "S3":
            ts = int(splits.get("s3_test_t_start", [10080])[0])
            counts = counts[:, ts:]
            feats = feats[:, ts:, :]
            trace_t_offset = ts
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        weights = np.ones(len(rows), dtype=np.float64)
        print(
            f"### R31 Azure2021 {split}: {len(rows)} funcs, {counts.shape[1]} ticks, "
            f"{int(counts.sum()):,} invocations, candidates={len(candidates)}",
            flush=True,
        )
        t0 = time.time()
        phi = embed_all(feats, trainer, device=device, batch_size=int(args.embed_batch))
        ewma = rates_ewma(counts)
        proto, pidx = prototype_rates_from_phi(phi, pm, device=device)
        faithful = faithful_rates_from_phi(phi, counts, trainer, pm, device=device)
        cum_prev = cum_prev_matrix(counts)
        age = age_matrix(counts)
        current, masks = current_gate_rates(proto, ewma, faithful, cum_prev, age)

        rate_mats: Dict[str, np.ndarray] = {
            "B4a_ewma": ewma,
            "G_WE_A720_current": current,
        }
        routing_meta: Dict[str, object] = {
            "zero_history": route_record(masks["zero_history"], weights),
            "count_insufficient_ewma": route_record(masks["count_insufficient_ewma"], weights),
            "initial_learned": route_record(masks["initial_learned"], weights),
            "gate_mature_ewma": route_record(masks["gate_mature_ewma"], weights),
        }

        for spec in candidates:
            print(f"    candidate {spec.name}", flush=True)
            rates, info = drift_qualified_rates(
                spec=spec,
                phi=phi,
                counts=counts,
                base_features=feats,
                trace_t_offset=trace_t_offset,
                current=current,
                ewma=ewma,
                proto=proto,
                proto_idx=pidx,
                pm=pm,
                cum_prev=cum_prev,
                age=age,
                trainer=trainer,
                device=device,
            )
            rate_mats[spec.name] = rates
            routing_meta[spec.name] = candidate_info_record(info, weights)

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
                fname = f"{method}__rho{_slug_float(float(rho))}.npz"
                np.savez_compressed(
                    jdir / fname,
                    prewarm=pw.astype(np.int32),
                    keepalive=ka.astype(np.float32),
                )
                job_meta.append({"method": method, "rho": float(rho), "file": fname})
        with open(jdir / "jobs.json", "w") as f:
            json.dump({"split": split, "seeds": seeds, "cold_init": COLD_INIT, "jobs": job_meta}, f, indent=1)

        manifest["splits"][split] = {
            "rows": rows.tolist(),
            "weights": weights.tolist(),
            "pool_size": int(len(rows)),
            "invocations": int(counts.sum()),
            "n_jobs": len(job_meta),
            "routing": routing_meta,
            "elapsed_sec": time.time() - t0,
        }
        with open(out, "w") as f:
            json.dump(manifest, f, indent=1)
        print(
            f"    exported {len(job_meta)} jobs in {time.time() - t0:.1f}s -> {jdir}",
            flush=True,
        )
    print(f"wrote {out}")


def _load_cross_trace_components(args, feature_shape):
    device = args.device
    model_path = args.model_path.resolve()
    trainer = load_trainer(feature_shape, model_path, device)
    feat21 = np.load(P21 / "features.npy")
    cnt21 = np.load(P21 / "counts.npy")
    spl21 = np.load(P21 / "splits.npz")
    pm = build_prototypes_local(
        trainer,
        feat21,
        cnt21,
        spl21["s1_train"],
        n_clusters=int(args.prototype_clusters),
        device=device,
    )
    return trainer, pm, model_path


def _export_prepared_split(
    *,
    args,
    split: str,
    rows: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    feats: np.ndarray,
    dm: np.ndarray,
    ds: np.ndarray,
    trace_t_offset: int,
    age: np.ndarray,
    trainer,
    pm,
    candidates: Sequence[CandidateSpec],
    rhos: Sequence[float],
    seeds: Sequence[int],
    jobs_root: Path,
    manifest: Dict[str, object],
) -> None:
    from scripts.phase6_des import COLD_INIT, decisions_from_rates, rates_ewma

    print(
        f"### R31 {manifest['trace']} {split}: {len(rows)} funcs, "
        f"{counts.shape[1]} ticks, {int(counts.sum()):,} invocations, "
        f"candidates={len(candidates)}",
        flush=True,
    )
    t0 = time.time()
    phi = embed_all(feats, trainer, device=args.device, batch_size=int(args.embed_batch))
    ewma = rates_ewma(counts)
    proto, pidx = prototype_rates_from_phi(phi, pm, device=args.device)
    faithful = faithful_rates_from_phi(phi, counts, trainer, pm, device=args.device)
    cum_prev = cum_prev_matrix(counts)
    current, masks = current_gate_rates(proto, ewma, faithful, cum_prev, age)

    rate_mats: Dict[str, np.ndarray] = {
        "B4a_ewma": ewma,
        "G_WE_A720_current": current,
    }
    routing_meta: Dict[str, object] = {
        "zero_history": route_record(masks["zero_history"], weights),
        "count_insufficient_ewma": route_record(masks["count_insufficient_ewma"], weights),
        "initial_learned": route_record(masks["initial_learned"], weights),
        "gate_mature_ewma": route_record(masks["gate_mature_ewma"], weights),
    }

    for spec in candidates:
        print(f"    candidate {spec.name}", flush=True)
        rates, info = drift_qualified_rates(
            spec=spec,
            phi=phi,
            counts=counts,
            base_features=feats,
            trace_t_offset=trace_t_offset,
            current=current,
            ewma=ewma,
            proto=proto,
            proto_idx=pidx,
            pm=pm,
            cum_prev=cum_prev,
            age=age,
            trainer=trainer,
            device=args.device,
        )
        rate_mats[spec.name] = rates
        routing_meta[spec.name] = candidate_info_record(info, weights)

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
            fname = f"{method}__rho{_slug_float(float(rho))}.npz"
            np.savez_compressed(
                jdir / fname,
                prewarm=pw.astype(np.int32),
                keepalive=ka.astype(np.float32),
            )
            job_meta.append({"method": method, "rho": float(rho), "file": fname})
    with open(jdir / "jobs.json", "w") as f:
        json.dump({"split": split, "seeds": seeds, "cold_init": COLD_INIT, "jobs": job_meta}, f, indent=1)

    manifest["splits"][split] = {
        "rows": rows.tolist(),
        "weights": weights.tolist(),
        "pool_size": int(manifest.get("_pool_size_override", len(rows))),
        "invocations": int(counts.sum()),
        "n_jobs": len(job_meta),
        "routing": routing_meta,
        "elapsed_sec": time.time() - t0,
    }
    manifest.pop("_pool_size_override", None)
    with open(manifest["_out_path"], "w") as f:
        json.dump({k: v for k, v in manifest.items() if not str(k).startswith("_")}, f, indent=1)
    print(f"    exported {len(job_meta)} jobs in {time.time() - t0:.1f}s -> {jdir}", flush=True)


def export_azure2019(args) -> None:
    torch.set_num_threads(1)
    features_mm = np.load(P19 / "features.npy", mmap_mode="r")
    counts_mm = np.load(P19 / "counts.npy", mmap_mode="r")
    splits = np.load(P19 / "splits.npz")
    dur_df = pd.read_csv(P19 / "duration_stats.csv")
    sample_manifest = json.load(open(RUNS / "des_sample_manifest_2019.json"))

    trainer, pm, model_path = _load_cross_trace_components(args, features_mm.shape)
    candidates = preset_candidates(args.preset)
    candidates.extend(parse_candidate_spec(raw) for raw in args.candidate)
    names = [c.name for c in candidates]
    if len(names) != len(set(names)):
        raise ValueError("duplicate candidate names in R31 export")

    rhos = selected_rhos(args)
    seeds = selected_seeds(args)
    jobs_root = Path(args.jobs_root) if args.jobs_root else RUNS / f"des_jobs_r31_azure2019_{args.preset}"
    out = (
        Path(args.manifest_out)
        if args.manifest_out
        else RUNS / f"revision_r31_azure2019_{args.preset}_manifest.json"
    )
    manifest: Dict[str, object] = {
        "analysis": "R31 drift-qualified WINTER-G cross-trace final-policy audit",
        "trace": "azure2019",
        "checkpoint": str(model_path),
        "checkpoint_md5": hashlib.md5(open(model_path, "rb").read()).hexdigest(),
        "prototype_clusters": int(args.prototype_clusters),
        "prototype_source": "azure2021::s1_train",
        "sample_manifest": str(RUNS / "des_sample_manifest_2019.json"),
        "min_invocations": MIN_INV,
        "age_min": AGE_MIN,
        "preset": args.preset,
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
        feats = np.asarray(features_mm[rows], dtype=np.float32)
        trace_t_offset = 0
        if split == "S3":
            ts = int(splits.get("s3_test_t_start", [14400])[0])
            counts = counts[:, ts:]
            feats = feats[:, ts:, :]
            trace_t_offset = ts
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        manifest["_pool_size_override"] = int(sample_manifest[split]["pool_size"])
        _export_prepared_split(
            args=args,
            split=split,
            rows=rows,
            weights=weights,
            counts=counts,
            feats=feats,
            dm=dm,
            ds=ds,
            trace_t_offset=trace_t_offset,
            age=age_matrix(counts),
            trainer=trainer,
            pm=pm,
            candidates=candidates,
            rhos=rhos,
            seeds=seeds,
            jobs_root=jobs_root,
            manifest=manifest,
        )
    with open(out, "w") as f:
        json.dump({k: v for k, v in manifest.items() if not str(k).startswith("_")}, f, indent=1)
    print(f"wrote {out}")


def export_huawei(args) -> None:
    torch.set_num_threads(1)
    features_mm = np.load(PHW / "features.npy", mmap_mode="r")
    counts_mm = np.load(PHW / "counts.npy", mmap_mode="r")
    splits = np.load(PHW / "splits.npz")
    first_present = np.load(PHW / "first_present.npy")
    steady_t = int(splits["steady_t"][0])
    dur_df = pd.read_csv(PHW / "duration_stats.csv")

    trainer, pm, model_path = _load_cross_trace_components(args, features_mm.shape)
    candidates = preset_candidates(args.preset)
    candidates.extend(parse_candidate_spec(raw) for raw in args.candidate)
    names = [c.name for c in candidates]
    if len(names) != len(set(names)):
        raise ValueError("duplicate candidate names in R31 export")

    rhos = selected_rhos(args)
    seeds = selected_seeds(args)
    jobs_root = Path(args.jobs_root) if args.jobs_root else RUNS / f"des_jobs_r31_huawei_{args.preset}"
    out = (
        Path(args.manifest_out)
        if args.manifest_out
        else RUNS / f"revision_r31_huawei_{args.preset}_manifest.json"
    )
    manifest: Dict[str, object] = {
        "analysis": "R31 drift-qualified WINTER-G cross-provider final-policy audit",
        "trace": "huawei2023",
        "checkpoint": str(model_path),
        "checkpoint_md5": hashlib.md5(open(model_path, "rb").read()).hexdigest(),
        "prototype_clusters": int(args.prototype_clusters),
        "prototype_source": "azure2021::s1_train",
        "age_source": "first_present.npy deployment-presence mask",
        "min_invocations": MIN_INV,
        "age_min": AGE_MIN,
        "preset": args.preset,
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
        feats = np.asarray(features_mm[rows][:, :steady_t, :], dtype=np.float32)
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        fp = first_present[rows].astype(np.int64)
        ticks = np.arange(counts.shape[1], dtype=np.int64)[None, :]
        age = ticks - fp[:, None]
        weights = np.ones(len(rows), dtype=np.float64)
        manifest["_pool_size_override"] = int(len(splits[pool]))
        _export_prepared_split(
            args=args,
            split=pool,
            rows=rows,
            weights=weights,
            counts=counts,
            feats=feats,
            dm=dm,
            ds=ds,
            trace_t_offset=0,
            age=age,
            trainer=trainer,
            pm=pm,
            candidates=candidates,
            rhos=rhos,
            seeds=seeds,
            jobs_root=jobs_root,
            manifest=manifest,
        )
    with open(out, "w") as f:
        json.dump({k: v for k, v in manifest.items() if not str(k).startswith("_")}, f, indent=1)
    print(f"wrote {out}")


def load_rows(paths: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for raw in paths:
        path = Path(raw)
        data = json.load(open(path))
        part = data.get("results", data) if isinstance(data, dict) else data
        for row in part:
            rec = dict(row)
            if rec.get("method") == "G_WE_A720":
                rec["method"] = "G_WE_A720_current"
            rec["_source_json"] = str(path)
            rows.append(rec)
        print(f"loaded {len(part)} rows from {path}", flush=True)
    return rows


def load_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.load(open(path))


def method_routing(routing: Dict[str, object], method: str) -> Dict[str, object]:
    out = {
        "route_zero_history": routing.get("zero_history", ""),
        "route_count_insufficient_ewma": routing.get("count_insufficient_ewma", ""),
        "route_initial_learned": routing.get("initial_learned", ""),
        "route_gate_mature_ewma": routing.get("gate_mature_ewma", ""),
        "route_recovery": "",
        "route_recovery_prior": "",
        "route_recovery_learned": "",
        "route_recovery_ewma": "",
        "route_shadow_candidate": "",
        "mean_fires_per_function": "",
        "mean_suppressed_by_cooldown_per_function": "",
        "mean_shadow_entries_per_function": "",
        "mean_shadow_commits_per_function": "",
        "mean_shadow_rejects_per_function": "",
        "mean_recovery_entries_per_function": "",
        "mean_recovery_fits_per_function": "",
        "mean_shadow_miss_gain_per_function": "",
        "mean_shadow_extra_prewarm_per_function": "",
    }
    rec = routing.get(method, {})
    if isinstance(rec, dict):
        for key in list(out):
            if key in rec:
                out[key] = rec.get(key, "")
    return out


def candidate_meta(candidates: Sequence[Dict[str, object]], method: str) -> Dict[str, object]:
    for rec in candidates:
        if rec.get("name") == method:
            return rec
    return {}


def summarize(args) -> None:
    rows = load_rows(args.inputs)
    manifest = load_manifest(Path(args.manifest))
    candidates = manifest.get("candidates", [])
    methods = ["B4a_ewma", "G_WE_A720_current"] + [str(c["name"]) for c in candidates]
    splits = sorted({r["split"] for r in rows})
    rhos = sorted({float(r["cost_ratio"]) for r in rows})
    header = (
        "split,method,rho,n_seeds,n_functions,csr_pct,csr_seed_std_pct,"
        "wm_per_1k_inv,cost_per_1k_inv,delta_csr_pp_vs_ewma,"
        "delta_wm_per_1k_vs_ewma,delta_cost_per_1k_vs_ewma,"
        "delta_cost_pct_vs_ewma,delta_csr_pp_vs_current,"
        "delta_wm_per_1k_vs_current,delta_cost_per_1k_vs_current,"
        "delta_cost_pct_vs_current,boot_ci95_lo_pp_vs_ewma,"
        "boot_ci95_hi_pp_vs_ewma,reset_mode,trigger_mode,shadow_window,"
        "k_drift,recovery_ttl,cooldown,route_zero_history,"
        "route_count_insufficient_ewma,route_initial_learned,"
        "route_gate_mature_ewma,route_recovery,route_recovery_prior,"
        "route_recovery_learned,route_recovery_ewma,route_shadow_candidate,"
        "mean_fires_per_function,mean_suppressed_by_cooldown_per_function,"
        "mean_shadow_entries_per_function,mean_shadow_commits_per_function,"
        "mean_shadow_rejects_per_function,mean_recovery_entries_per_function,"
        "mean_recovery_fits_per_function,mean_shadow_miss_gain_per_function,"
        "mean_shadow_extra_prewarm_per_function,source_json"
    )
    lines = [header]
    rho10 = [header]
    readout = {
        "analysis": manifest.get("analysis", "R31"),
        "trace": manifest.get("trace", ""),
        "preset": manifest.get("preset", ""),
        "cells": {},
    }
    for split in splits:
        split_meta = manifest["splits"][split]
        weights = np.asarray(split_meta["weights"], dtype=np.float64)
        routing = split_meta.get("routing", {})
        for rho in rhos:
            ref_rows = [
                r for r in rows
                if r["split"] == split
                and r["method"] == "B4a_ewma"
                and abs(float(r["cost_ratio"]) - rho) < 1e-9
            ]
            cur_rows = [
                r for r in rows
                if r["split"] == split
                and r["method"] == "G_WE_A720_current"
                and abs(float(r["cost_ratio"]) - rho) < 1e-9
            ]
            ref = summarize_cell(ref_rows, weights)
            cur = summarize_cell(cur_rows, weights)
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
                dcsrc = rec["csr_pct"] - cur["csr_pct"] if cur else np.nan
                dwmc = rec["wm_per_1k_inv"] - cur["wm_per_1k_inv"] if cur else np.nan
                dcostc = rec["cost_per_1k_inv"] - cur["cost_per_1k_inv"] if cur else np.nan
                dcostc_pct = 100.0 * dcostc / max(cur["cost_per_1k_inv"], 1e-9) if cur else np.nan
                lo, hi = ("", "")
                if method != "B4a_ewma" and ref_rows:
                    lo, hi = paired_boot_ci(cell_rows, ref_rows, weights)
                route = method_routing(routing, method)
                meta = candidate_meta(candidates, method)
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
                    f"{dcsrc:.6f}",
                    f"{dwmc:.3f}",
                    f"{dcostc:.3f}",
                    f"{dcostc_pct:.6f}",
                    lo,
                    hi,
                    meta.get("reset_mode", ""),
                    meta.get("trigger_mode", ""),
                    meta.get("shadow_window", ""),
                    meta.get("k_drift", ""),
                    meta.get("recovery_ttl", ""),
                    meta.get("cooldown", ""),
                    route["route_zero_history"],
                    route["route_count_insufficient_ewma"],
                    route["route_initial_learned"],
                    route["route_gate_mature_ewma"],
                    route["route_recovery"],
                    route["route_recovery_prior"],
                    route["route_recovery_learned"],
                    route["route_recovery_ewma"],
                    route["route_shadow_candidate"],
                    route["mean_fires_per_function"],
                    route["mean_suppressed_by_cooldown_per_function"],
                    route["mean_shadow_entries_per_function"],
                    route["mean_shadow_commits_per_function"],
                    route["mean_shadow_rejects_per_function"],
                    route["mean_recovery_entries_per_function"],
                    route["mean_recovery_fits_per_function"],
                    route["mean_shadow_miss_gain_per_function"],
                    route["mean_shadow_extra_prewarm_per_function"],
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
                        "delta_csr_pp_vs_current": float(dcsrc),
                        "delta_wm_per_1k_vs_current": float(dwmc),
                        "delta_cost_per_1k_vs_current": float(dcostc),
                        "delta_cost_pct_vs_current": float(dcostc_pct),
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
    print(f"wrote {TABLES / f'{prefix}_fullrho.csv'}")
    print(f"wrote {TABLES / f'{prefix}_rho10.csv'}")
    print(f"wrote {RUNS / f'{prefix}.json'}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--trace", choices=["azure2021", "azure2019", "huawei"], default="azure2021")
    ap.add_argument("--splits", default="S1,S2,S3")
    ap.add_argument("--pools", default="h_mixed,h_sparse,h_saturated")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument(
        "--preset",
        choices=["none", "smoke", "s1_design", "focused", "frozen_candidate", "final_policy"],
        default="focused",
    )
    ap.add_argument("--candidate", action="append", default=[], help="extra comma-separated key=value candidate")
    ap.add_argument("--rhos", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--jobs-root", default="")
    ap.add_argument("--manifest-out", default="")
    ap.add_argument("--model-path", type=Path, default=RUNS / "best_anil_ridge_s1_s0.pt")
    ap.add_argument("--prototype-clusters", type=int, default=4)
    ap.add_argument("--embed-batch", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument(
        "--manifest",
        default=str(RUNS / "revision_r31_drift_qualified_azure2021_focused_manifest.json"),
    )
    ap.add_argument("--prefix", default="T_r31_drift_qualified_azure2021")
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
        if not args.inputs:
            ap.error("--summarize requires --inputs")
        summarize(args)


if __name__ == "__main__":
    main()
