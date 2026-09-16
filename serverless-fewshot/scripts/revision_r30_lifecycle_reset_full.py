#!/usr/bin/env python3
"""R30 full-trace lifecycle reset-to-prior WINTER-G audit.

This audit composes a single full-window routing policy:

  initial zero history                 -> prototype prior
  initial count-insufficient history   -> EWMA
  initial count-qualified, age-young   -> current Section-4 head
  mature stable traffic                -> EWMA
  detected mature drift                -> reset-to-prior recovery

Two recovery variants are exported.  G_WE_A720R_T20 follows the R26
event-window semantics: after a trigger, serve the prototype prior until 20
post-reset support ticks are available, then use a prototype-biased ridge head.
G_WE_A720R_C100 is the lifecycle-consistent variant: after a trigger, use the
same post-reset states as initial onboarding (zero-history prototype,
count-insufficient EWMA, count-qualified learned head).

The script exports fast-DES jobs and summarizes the resulting runs.  It does
not overwrite the R20/R21/R25/R26 artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import COLD_INIT, decisions_from_rates, rates_ewma  # noqa: E402
from scripts.phase63_onboarding_drift import build_prototypes  # noqa: E402
from scripts.revision_r21_cross_provider_faithful import (  # noqa: E402
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

L = 60
D = 64
MED = N_QUANTILES // 2
MIN_INV = 100
AGE_MIN = 720
RESET_SUPPORT_TICKS = 20
RESET_REFIT_EVERY = 10
RESET_BUFFER_LEN = 120
RECOVERY_TTL = 240
STABLE_EXIT = 120
MAX_RATE = 200.0
MAX_LOG_RATE = float(np.log1p(MAX_RATE))
FAITHFUL_BUFFER = 180
FAITHFUL_REFIT_EVERY = 60
FAITHFUL_ALPHA = 0.15
RHOS = [0.1, 1.0, 10.0, 100.0]
SEEDS = list(range(10))


def selected_rhos(args) -> List[float]:
    return parse_float_list(args.rhos) if args.rhos else RHOS


def selected_seeds(args) -> List[int]:
    return parse_int_list(args.seeds) if args.seeds else SEEDS


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
    return trainer


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


def lifecycle_reset_rates(
    variant: str,
    phi: np.ndarray,
    counts: np.ndarray,
    current: np.ndarray,
    ewma: np.ndarray,
    proto: np.ndarray,
    proto_idx: np.ndarray,
    pm,
    cum_prev: np.ndarray,
    age: np.ndarray,
    l2: float,
    refit_every: int,
    buffer_len: int,
    recovery_ttl: int,
    stable_exit: int,
    trigger_alpha: float,
    trigger_delta: float,
    trigger_m: int,
    trigger_calib_size: int,
    trigger_window: int,
    trigger_min_calib: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if variant not in {"t20", "c100"}:
        raise ValueError(f"unknown lifecycle reset variant: {variant}")

    n_funcs, t_len = counts.shape
    y = np.log1p(counts.astype(np.float64))
    cprefix = np.concatenate([np.zeros((n_funcs, 1), dtype=np.int64), np.cumsum(counts.astype(np.int64), axis=1)], axis=1)
    proto_w = pm.proto_weights.detach().cpu().numpy().astype(np.float64)
    rates = current.copy().astype(np.float32)

    route_recovery = np.zeros((n_funcs, t_len), dtype=bool)
    route_recovery_prior = np.zeros((n_funcs, t_len), dtype=bool)
    route_recovery_learned = np.zeros((n_funcs, t_len), dtype=bool)
    route_recovery_ewma = np.zeros((n_funcs, t_len), dtype=bool)
    fires = np.zeros(n_funcs, dtype=np.int64)
    entries = np.zeros(n_funcs, dtype=np.int64)
    fits = np.zeros(n_funcs, dtype=np.int64)

    for f in range(n_funcs):
        monitor = _new_monitor(
            alpha=trigger_alpha,
            delta=trigger_delta,
            m_consecutive=trigger_m,
            calib_size=trigger_calib_size,
            window=trigger_window,
            min_calib=trigger_min_calib,
        )
        state = "stable"
        recovery_start = math.inf
        last_fire = -10**9
        wh = None

        for t in range(t_len):
            if (
                state == "recovery"
                and (t - recovery_start) >= recovery_ttl
                and (t - last_fire) >= stable_exit
            ):
                state = "stable"
                recovery_start = math.inf
                wh = None
                monitor = _new_monitor(
                    alpha=trigger_alpha,
                    delta=trigger_delta,
                    m_consecutive=trigger_m,
                    calib_size=trigger_calib_size,
                    window=trigger_window,
                    min_calib=trigger_min_calib,
                )

            mature = bool(cum_prev[f, t] >= MIN_INV and age[f, t] >= AGE_MIN)
            pred_log = math.log1p(max(float(current[f, t]), 0.0))

            if state == "recovery" and t >= recovery_start:
                route_recovery[f, t] = True
                post_age = int(t - recovery_start)
                post_cum = int(cprefix[f, t] - cprefix[f, int(recovery_start)])

                use_prior = False
                use_learned = False
                use_ewma = False

                if variant == "c100" and 0 < post_cum < MIN_INV:
                    rates[f, t] = ewma[f, t]
                    pred_log = math.log1p(max(float(ewma[f, t]), 0.0))
                    use_ewma = True
                elif variant == "t20" and (post_age < RESET_SUPPORT_TICKS or wh is None):
                    rates[f, t] = proto[f, t]
                    pred_log = math.log1p(max(float(proto[f, t]), 0.0))
                    use_prior = True
                elif variant == "c100" and (post_cum == 0 or wh is None):
                    rates[f, t] = proto[f, t]
                    pred_log = math.log1p(max(float(proto[f, t]), 0.0))
                    use_prior = True
                else:
                    pred = phi[f, t].astype(np.float64) @ wh
                    rates[f, t] = pred_to_rate(pred)
                    pred_log = math.log1p(max(float(rates[f, t]), 0.0))
                    use_learned = True

                route_recovery_prior[f, t] = use_prior
                route_recovery_learned[f, t] = use_learned
                route_recovery_ewma[f, t] = use_ewma
                resid = abs(float(y[f, t]) - pred_log)
                if monitor.update(resid):
                    fires[f] += 1
                    entries[f] += 1
                    last_fire = t
                    recovery_start = min(t + 1, t_len)
                    wh = None
                    monitor = _new_monitor(
                        alpha=trigger_alpha,
                        delta=trigger_delta,
                        m_consecutive=trigger_m,
                        calib_size=trigger_calib_size,
                        window=trigger_window,
                        min_calib=trigger_min_calib,
                    )
                    continue

                support_ticks = int(t - recovery_start)
                if support_ticks >= RESET_SUPPORT_TICKS and support_ticks % max(1, refit_every) == 0:
                    s0 = max(int(recovery_start), t - int(buffer_len))
                    if t - s0 >= RESET_SUPPORT_TICKS:
                        wh = biased_ridge_fit(
                            phi[f, s0:t, :].astype(np.float64),
                            y[f, s0:t],
                            proto_w[int(proto_idx[f, t])],
                            l2=l2,
                        )
                        fits[f] += 1
                continue

            rates[f, t] = current[f, t]
            if mature:
                pred_log = math.log1p(max(float(ewma[f, t]), 0.0))
                resid = abs(float(y[f, t]) - pred_log)
                if monitor.update(resid):
                    fires[f] += 1
                    entries[f] += 1
                    state = "recovery"
                    last_fire = t
                    recovery_start = min(t + 1, t_len)
                    wh = None
                    monitor = _new_monitor(
                        alpha=trigger_alpha,
                        delta=trigger_delta,
                        m_consecutive=trigger_m,
                        calib_size=trigger_calib_size,
                        window=trigger_window,
                        min_calib=trigger_min_calib,
                    )

    info = {
        "route_recovery": route_recovery,
        "route_recovery_prior": route_recovery_prior,
        "route_recovery_learned": route_recovery_learned,
        "route_recovery_ewma": route_recovery_ewma,
        "fires": fires,
        "recovery_entries": entries,
        "recovery_fits": fits,
    }
    return rates, info


def route_record(mask: np.ndarray, weights: np.ndarray) -> float:
    return weighted_share(mask, weights)


def export_azure2021(args) -> None:
    features = np.load(P21 / "features.npy")
    counts_all = np.load(P21 / "counts.npy")
    splits = np.load(P21 / "splits.npz")
    dur_df = pd.read_csv(P21 / "duration_stats.csv")
    device = args.device
    model_path = args.model_path.resolve()
    trainer = load_trainer(features.shape, model_path, device)
    pm = build_prototypes(
        trainer,
        features,
        counts_all,
        splits["s1_train"],
        n_clusters=int(args.prototype_clusters),
    )
    rhos = selected_rhos(args)
    seeds = selected_seeds(args)
    jobs_root = Path(args.jobs_root) if args.jobs_root else RUNS / "des_jobs_r30_azure2021"
    out = Path(args.manifest_out) if args.manifest_out else RUNS / "revision_r30_lifecycle_reset_azure2021_manifest.json"
    manifest = {
        "analysis": "R30 full-trace lifecycle reset-to-prior WINTER-G audit",
        "trace": "azure2021",
        "checkpoint": str(model_path),
        "checkpoint_md5": hashlib.md5(open(model_path, "rb").read()).hexdigest(),
        "prototype_clusters": int(args.prototype_clusters),
        "prototype_source": "azure2021::s1_train",
        "min_invocations": MIN_INV,
        "age_min": AGE_MIN,
        "recovery_ttl": int(args.recovery_ttl),
        "stable_exit": int(args.stable_exit),
        "reset_support_ticks": RESET_SUPPORT_TICKS,
        "reset_refit_every": int(args.refit_every),
        "reset_buffer_len": int(args.buffer_len),
        "trigger": {
            "alpha": float(args.trigger_alpha),
            "delta": float(args.trigger_delta),
            "m_consecutive": int(args.trigger_m),
            "calib_size": int(args.trigger_calib_size),
            "window": int(args.trigger_window),
            "min_calib": int(args.trigger_min_calib),
        },
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
            rows = rows[: args.pilot]
        counts = counts_all[rows].astype(np.float32)
        feats = features[rows].astype(np.float32)
        if split == "S3":
            ts = int(splits.get("s3_test_t_start", [10080])[0])
            counts = counts[:, ts:]
            feats = feats[:, ts:, :]
        dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[rows], nan=1.0)
        ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[rows], nan=0.5)
        weights = np.ones(len(rows), dtype=np.float64)
        print(
            f"### Azure2021 {split}: {len(rows)} funcs, {counts.shape[1]} ticks, "
            f"{int(counts.sum()):,} invocations",
            flush=True,
        )
        t0 = time.time()
        phi = embed_all(feats, trainer, device=device, batch_size=int(args.embed_batch))
        ewma = rates_ewma(counts)
        proto, proto_idx = prototype_rates_from_phi(phi, pm, device=device)
        faithful = faithful_rates_from_phi(phi, counts, trainer, pm, device=device)
        cum_prev = cum_prev_matrix(counts)
        age = age_matrix(counts)
        current, masks = current_gate_rates(proto, ewma, faithful, cum_prev, age)
        t20_rates, t20_info = lifecycle_reset_rates(
            "t20",
            phi,
            counts,
            current,
            ewma,
            proto,
            proto_idx,
            pm,
            cum_prev,
            age,
            l2=float(args.l2),
            refit_every=int(args.refit_every),
            buffer_len=int(args.buffer_len),
            recovery_ttl=int(args.recovery_ttl),
            stable_exit=int(args.stable_exit),
            trigger_alpha=float(args.trigger_alpha),
            trigger_delta=float(args.trigger_delta),
            trigger_m=int(args.trigger_m),
            trigger_calib_size=int(args.trigger_calib_size),
            trigger_window=int(args.trigger_window),
            trigger_min_calib=int(args.trigger_min_calib),
        )
        c100_rates, c100_info = lifecycle_reset_rates(
            "c100",
            phi,
            counts,
            current,
            ewma,
            proto,
            proto_idx,
            pm,
            cum_prev,
            age,
            l2=float(args.l2),
            refit_every=int(args.refit_every),
            buffer_len=int(args.buffer_len),
            recovery_ttl=int(args.recovery_ttl),
            stable_exit=int(args.stable_exit),
            trigger_alpha=float(args.trigger_alpha),
            trigger_delta=float(args.trigger_delta),
            trigger_m=int(args.trigger_m),
            trigger_calib_size=int(args.trigger_calib_size),
            trigger_window=int(args.trigger_window),
            trigger_min_calib=int(args.trigger_min_calib),
        )

        rate_mats = {
            "B4a_ewma": ewma,
            "G_WE_A720_current": current,
            "G_WE_A720R_T20": t20_rates,
            "G_WE_A720R_C100": c100_rates,
        }
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
                fname = f"{method}__rho{rho:g}.npz"
                np.savez_compressed(jdir / fname, prewarm=pw.astype(np.int32), keepalive=ka.astype(np.float32))
                job_meta.append({"method": method, "rho": float(rho), "file": fname})
        with open(jdir / "jobs.json", "w") as f:
            json.dump({"split": split, "seeds": seeds, "cold_init": COLD_INIT, "jobs": job_meta}, f, indent=1)

        def rec_for(info: Dict[str, object]) -> Dict[str, float]:
            recovery = info["route_recovery"]
            prior = info["route_recovery_prior"]
            learned = info["route_recovery_learned"]
            ew = info["route_recovery_ewma"]
            return {
                "route_recovery": route_record(recovery, weights),
                "route_recovery_prior": route_record(prior, weights),
                "route_recovery_learned": route_record(learned, weights),
                "route_recovery_ewma": route_record(ew, weights),
                "mean_fires_per_function": float(np.mean(info["fires"])),
                "mean_recovery_entries_per_function": float(np.mean(info["recovery_entries"])),
                "mean_recovery_fits_per_function": float(np.mean(info["recovery_fits"])),
            }

        manifest["splits"][split] = {
            "rows": rows.tolist(),
            "weights": weights.tolist(),
            "pool_size": int(len(rows)),
            "invocations": int(counts.sum()),
            "n_jobs": len(job_meta),
            "routing": {
                "zero_history": route_record(masks["zero_history"], weights),
                "count_insufficient_ewma": route_record(masks["count_insufficient_ewma"], weights),
                "initial_learned": route_record(masks["initial_learned"], weights),
                "gate_mature_ewma": route_record(masks["gate_mature_ewma"], weights),
                "G_WE_A720R_T20": rec_for(t20_info),
                "G_WE_A720R_C100": rec_for(c100_info),
            },
            "elapsed_sec": time.time() - t0,
        }
        print(
            f"    exported {len(job_meta)} jobs in {time.time() - t0:.1f}s; "
            f"T20 recovery={manifest['splits'][split]['routing']['G_WE_A720R_T20']['route_recovery']:.4f}, "
            f"C100 recovery={manifest['splits'][split]['routing']['G_WE_A720R_C100']['route_recovery']:.4f}",
            flush=True,
        )
        with open(out, "w") as f:
            json.dump(manifest, f, indent=1)
    print(f"wrote {out}")


def load_rows(paths: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
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
        "mean_fires_per_function": "",
        "mean_recovery_entries_per_function": "",
        "mean_recovery_fits_per_function": "",
    }
    if method in {"G_WE_A720R_T20", "G_WE_A720R_C100"}:
        rec = routing.get(method, {})
        for key in [
            "route_recovery",
            "route_recovery_prior",
            "route_recovery_learned",
            "route_recovery_ewma",
            "mean_fires_per_function",
            "mean_recovery_entries_per_function",
            "mean_recovery_fits_per_function",
        ]:
            out[key] = rec.get(key, "")
    return out


def summarize(args) -> None:
    rows = load_rows(args.inputs)
    manifest = load_manifest(Path(args.manifest))
    methods = ["B4a_ewma", "G_WE_A720_current", "G_WE_A720R_T20", "G_WE_A720R_C100"]
    splits = sorted({r["split"] for r in rows})
    rhos = sorted({float(r["cost_ratio"]) for r in rows})
    header = (
        "split,method,rho,n_seeds,n_functions,csr_pct,csr_seed_std_pct,"
        "wm_per_1k_inv,cost_per_1k_inv,delta_csr_pp_vs_ewma,"
        "delta_wm_per_1k_vs_ewma,delta_cost_per_1k_vs_ewma,"
        "delta_csr_pp_vs_current,delta_wm_per_1k_vs_current,"
        "delta_cost_per_1k_vs_current,boot_ci95_lo_pp_vs_ewma,"
        "boot_ci95_hi_pp_vs_ewma,route_zero_history,"
        "route_count_insufficient_ewma,route_initial_learned,"
        "route_gate_mature_ewma,route_recovery,route_recovery_prior,"
        "route_recovery_learned,route_recovery_ewma,mean_fires_per_function,"
        "mean_recovery_entries_per_function,mean_recovery_fits_per_function,"
        "source_json"
    )
    lines = [header]
    rho10 = [header]
    readout = {"analysis": manifest.get("analysis", "R30"), "cells": {}}
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
                dcsrc = rec["csr_pct"] - cur["csr_pct"] if cur else np.nan
                dwmc = rec["wm_per_1k_inv"] - cur["wm_per_1k_inv"] if cur else np.nan
                dcostc = rec["cost_per_1k_inv"] - cur["cost_per_1k_inv"] if cur else np.nan
                lo, hi = ("", "")
                if method != "B4a_ewma" and ref_rows:
                    lo, hi = paired_boot_ci(cell_rows, ref_rows, weights)
                route = method_routing(routing, method)
                source = ";".join(sorted({r["_source_json"] for r in cell_rows}))
                line = (
                    f"{split},{method},{rho:g},{rec['n']},{rec['n_functions']},"
                    f"{rec['csr_pct']:.6f},{rec['csr_seed_std_pct']:.6f},"
                    f"{rec['wm_per_1k_inv']:.3f},{rec['cost_per_1k_inv']:.3f},"
                    f"{dcsr:.6f},{dwm:.3f},{dcost:.3f},"
                    f"{dcsrc:.6f},{dwmc:.3f},{dcostc:.3f},{lo},{hi},"
                    f"{route['route_zero_history']},{route['route_count_insufficient_ewma']},"
                    f"{route['route_initial_learned']},{route['route_gate_mature_ewma']},"
                    f"{route['route_recovery']},{route['route_recovery_prior']},"
                    f"{route['route_recovery_learned']},{route['route_recovery_ewma']},"
                    f"{route['mean_fires_per_function']},{route['mean_recovery_entries_per_function']},"
                    f"{route['mean_recovery_fits_per_function']},{source}"
                )
                lines.append(line)
                if abs(rho - 10.0) < 1e-9:
                    rho10.append(line)
                readout["cells"][f"{split}|{rho:g}|{method}"] = rec | {
                    "delta_csr_pp_vs_ewma": float(dcsr),
                    "delta_wm_per_1k_vs_ewma": float(dwm),
                    "delta_cost_per_1k_vs_ewma": float(dcost),
                    "delta_csr_pp_vs_current": float(dcsrc),
                    "delta_wm_per_1k_vs_current": float(dwmc),
                    "delta_cost_per_1k_vs_current": float(dcostc),
                    "boot_ci95_pp_vs_ewma": [None if lo == "" else float(lo), None if hi == "" else float(hi)],
                    **route,
                    "source_json": source,
                }

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--trace", choices=["azure2021"], default="azure2021")
    ap.add_argument("--splits", default="S1,S2,S3")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--rhos", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--jobs-root", default="")
    ap.add_argument("--manifest-out", default="")
    ap.add_argument("--model-path", type=Path, default=RUNS / "best_anil_ridge_s1_s0.pt")
    ap.add_argument("--prototype-clusters", type=int, default=4)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--refit-every", type=int, default=RESET_REFIT_EVERY)
    ap.add_argument("--buffer-len", type=int, default=RESET_BUFFER_LEN)
    ap.add_argument("--recovery-ttl", type=int, default=RECOVERY_TTL)
    ap.add_argument("--stable-exit", type=int, default=STABLE_EXIT)
    ap.add_argument("--trigger-alpha", type=float, default=0.1)
    ap.add_argument("--trigger-delta", type=float, default=0.15)
    ap.add_argument("--trigger-m", type=int, default=3)
    ap.add_argument("--trigger-calib-size", type=int, default=120)
    ap.add_argument("--trigger-window", type=int, default=30)
    ap.add_argument("--trigger-min-calib", type=int, default=30)
    ap.add_argument("--embed-batch", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--manifest", default=str(RUNS / "revision_r30_lifecycle_reset_azure2021_manifest.json"))
    ap.add_argument("--prefix", default="T_r30_lifecycle_reset_azure2021")
    args = ap.parse_args()

    if args.export:
        export_azure2021(args)
    if args.summarize:
        summarize(args)


if __name__ == "__main__":
    main()
