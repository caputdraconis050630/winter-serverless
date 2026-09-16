#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R19: operational-state diagnostics for the faithful gate family.

This is an experiment-only diagnostic. It does not edit manuscript files and
does not replace the primary active-pool steady-window endpoint. It answers two
questions raised by the faithful WINTER-G rerun:

1. Where do CSR, warm-memory, and registered cost accrue across the gate's
   in-window operational states?
2. Which learned/EWMA routing choices on those states form the gate-family
   frontier?

States are computed before serving tick t:

  Z0: K_t == 0
  W:  1 <= K_t < 100
  YC: K_t >= 100 and age_t < 720
  M:  K_t >= 100 and age_t >= 720

Z0 is fixed to the prototype zero-shot head in every factorial gate. The three
letters in G_<W><YC><M> choose E=EWMA or L=faithful learned head for W, YC, M.
Thus v3=G_ELL, v4=G_LEE, and the age-qualified gate is G_ELE.

Outputs:
  results/runs/revision_r19_gate_state_diagnostics.json
  results/tables/T_r19_gate_factorial_rho10.csv
  results/tables/T_r19_state_slices_rho10.csv
  results/tables/T_r19_mature_clean_rho10.csv
"""

import json
import os
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
)
from scripts.phase63_onboarding_drift import build_prototypes  # noqa: E402
from scripts.revision_a2_des import rates_proto_prefix  # noqa: E402
from scripts.revision_r11_faithful import rates_a5_faithful  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402

torch.set_num_threads(1)

P21 = PROJECT_ROOT / "data" / "processed"
RUNS = PROJECT_ROOT / "results" / "runs"
TABLES = PROJECT_ROOT / "results" / "tables"
OUT = RUNS / "revision_r19_gate_state_diagnostics.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = list(range(10))
LOW_RHOS = [0.01, 0.02, 0.05]
MIN_INV = 100
AGE_MIN = 720
KAPPA = 15.0
STATE_GB_PER_FN = 8e-6
TICK_SEC = 60.0
SLICE_NAMES = ("Z0", "W", "YC", "M")
MATURE_WASHOUT = 120


def target_rhos(split):
    core = FULL_RHOS if split == "S1" else REDUCED_RHOS
    return sorted({float(r) for r in list(core) + LOW_RHOS})


def result_key(row):
    return (
        row["experiment"],
        row["method"],
        row["split"],
        float(row["cost_ratio"]),
        int(row["seed"]),
    )


def dedupe(rows):
    by_key = {}
    for row in rows:
        by_key[result_key(row)] = row
    return [by_key[k] for k in sorted(by_key, key=lambda x: (x[2], x[3], x[4], x[1], x[0]))]


def load_output():
    if OUT.exists():
        out = json.load(open(OUT))
    else:
        out = {
            "description": "Operational-state diagnostics for faithful Azure-2021 gate family",
            "created_by": "scripts/revision_r19_gate_state_diagnostics.py",
            "seeds": SEEDS,
            "min_invocations": MIN_INV,
            "age_min": AGE_MIN,
            "kappa": KAPPA,
            "state_gb_per_function": STATE_GB_PER_FN,
            "mature_washout_min": MATURE_WASHOUT,
            "methods": {},
            "routing": {},
            "results": [],
            "reconciliation": {},
        }
    out["results"] = dedupe(out.get("results", []))
    out.setdefault("methods", {})
    out.setdefault("routing", {})
    out.setdefault("reconciliation", {})
    return out


def save(out):
    out["results"] = dedupe(out.get("results", []))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    json.load(open(OUT))


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


def age_matrix(counts):
    n, t_len = counts.shape
    age = np.zeros((n, t_len), dtype=np.int64)
    for f in range(n):
        nz = np.flatnonzero(counts[f])
        if len(nz):
            t0 = int(nz[0])
            age[f, t0:] = np.arange(t_len - t0)
    return age


def state_masks(counts, cum_prev):
    age = age_matrix(counts)
    masks = {
        "Z0": cum_prev == 0,
        "W": (cum_prev > 0) & (cum_prev < MIN_INV),
        "YC": (cum_prev >= MIN_INV) & (age < AGE_MIN),
        "M": (cum_prev >= MIN_INV) & (age >= AGE_MIN),
    }
    code = np.zeros(counts.shape, dtype=np.int8)
    for i, name in enumerate(SLICE_NAMES):
        code[masks[name]] = i
    assert np.all(sum(masks.values()))
    return masks, code, age


def method_uses_learned(method):
    if method == "A5_faithful":
        return True
    if method.startswith("G_"):
        return "L" in method[2:]
    return False


def compose_gate_rates(proto, ewma, faithful, masks, pattern):
    w_choice, yc_choice, m_choice = pattern
    rates = np.empty_like(ewma, dtype=np.float32)
    rates[masks["Z0"]] = proto[masks["Z0"]]
    for state, choice in [("W", w_choice), ("YC", yc_choice), ("M", m_choice)]:
        src = faithful if choice == "L" else ewma
        rates[masks[state]] = src[masks[state]]
    return rates


def add_interval_by_state(acc, state_code, start, end, mem_gb, tick_sec=TICK_SEC):
    if end <= start:
        return
    t_len = len(state_code)
    cur = max(0.0, start)
    end = min(end, t_len * tick_sec)
    while cur < end:
        tick = int(cur // tick_sec)
        if tick >= t_len:
            break
        next_edge = min(end, (tick + 1) * tick_sec)
        acc[int(state_code[tick])] += (next_edge - cur) * mem_gb
        cur = next_edge


def simulate_function_sliced(
    counts,
    prewarm,
    keepalive,
    state_code,
    dur_mean,
    dur_std,
    memory_mb=256.0,
    seed=0,
    tick_sec=TICK_SEC,
    cold_mu=0.0,
    cold_sigma=0.3,
):
    rng = np.random.default_rng(seed)
    t_len = len(counts)
    mem_gb = memory_mb / 1024.0
    end_time = t_len * tick_sec

    containers = []
    total_inv = 0
    cold = 0
    idle_mem_s = 0.0
    busy_mem_s = 0.0
    by_state = {
        name: {"total_invocations": 0, "cold_starts": 0, "wm_total_gb_s": 0.0}
        for name in SLICE_NAMES
    }
    idle_by_state = np.zeros(len(SLICE_NAMES), dtype=np.float64)
    cold_by_state = np.zeros(len(SLICE_NAMES), dtype=np.int64)
    total_by_state = np.zeros(len(SLICE_NAMES), dtype=np.int64)

    dur_sigma = max(float(dur_std), 0.01)

    def evict_expired(now, ka_sec):
        nonlocal idle_mem_s
        kept = []
        for c in containers:
            idle_start = c[0] if c[0] > c[1] else c[1]
            if idle_start <= now - ka_sec:
                idle_end = idle_start + ka_sec
                idle_mem_s += ka_sec * mem_gb
                add_interval_by_state(idle_by_state, state_code, idle_start, idle_end, mem_gb, tick_sec)
            else:
                kept.append(c)
        containers[:] = kept

    for t in range(t_len):
        now0 = t * tick_sec
        ka_sec = float(keepalive[t]) * 60.0
        sidx = int(state_code[t])

        evict_expired(now0, ka_sec)

        target = int(prewarm[t])
        deficit = target - len(containers)
        if deficit > 0 and len(containers) < 200:
            n_new = min(deficit, 200 - len(containers))
            inits = rng.lognormal(cold_mu, cold_sigma, n_new)
            for init in inits:
                containers.append([now0 + init, now0 + init])
                busy_mem_s += init * mem_gb

        n = int(counts[t])
        if n <= 0:
            continue

        offsets = np.sort(rng.uniform(0.0, tick_sec, n))
        durs = rng.normal(float(dur_mean), dur_sigma, n)

        for i in range(n):
            tau = now0 + offsets[i]
            dur = durs[i] if durs[i] > 0.01 else 0.01
            evict_expired(tau, ka_sec)

            best = None
            best_idle = -1.0
            for c in containers:
                if c[0] <= tau and c[1] <= tau:
                    idle_start = c[0] if c[0] > c[1] else c[1]
                    if idle_start > best_idle:
                        best_idle = idle_start
                        best = c

            if best is not None:
                idle = tau - best_idle
                idle_mem_s += idle * mem_gb
                add_interval_by_state(idle_by_state, state_code, best_idle, tau, mem_gb, tick_sec)
                best[1] = tau + dur
                busy_mem_s += dur * mem_gb
            else:
                init = rng.lognormal(cold_mu, cold_sigma)
                if len(containers) < 200:
                    containers.append([tau + init, tau + init + dur])
                busy_mem_s += (init + dur) * mem_gb
                cold += 1
                cold_by_state[sidx] += 1

            total_inv += 1
            total_by_state[sidx] += 1

    final_ka = float(keepalive[t_len - 1]) * 60.0
    for c in containers:
        idle_start = c[0] if c[0] > c[1] else c[1]
        if idle_start < end_time:
            idle_end = min(end_time, idle_start + final_ka)
            idle = idle_end - idle_start
            idle_mem_s += idle * mem_gb
            add_interval_by_state(idle_by_state, state_code, idle_start, idle_end, mem_gb, tick_sec)

    for i, name in enumerate(SLICE_NAMES):
        by_state[name]["total_invocations"] = int(total_by_state[i])
        by_state[name]["cold_starts"] = int(cold_by_state[i])
        by_state[name]["wm_total_gb_s"] = float(idle_by_state[i])

    return {
        "total_invocations": int(total_inv),
        "cold_starts": int(cold),
        "wm_total_gb_s": float(idle_mem_s),
        "busy_mem_gb_s": float(busy_mem_s),
        "by_state": by_state,
    }


def aggregate_sliced(per_func, rho, learned_state):
    total_inv = sum(r["total_invocations"] for r in per_func)
    cold = sum(r["cold_starts"] for r in per_func)
    wm = sum(r["wm_total_gb_s"] for r in per_func)
    out_state = {}
    state_gbs_total = 0.0
    for name in SLICE_NAMES:
        inv = sum(r["by_state"][name]["total_invocations"] for r in per_func)
        c = sum(r["by_state"][name]["cold_starts"] for r in per_func)
        w = sum(r["by_state"][name]["wm_total_gb_s"] for r in per_func)
        out_state[name] = {
            "total_invocations": int(inv),
            "cold_starts": int(c),
            "csr": float(c / max(inv, 1)),
            "csr_pct": float(100.0 * c / max(inv, 1)),
            "wm_total_gb_s": float(w),
            "wm_per_1k_inv": float(w / max(inv / 1000.0, 1e-10)),
            "registered_cost_gb_s_no_state": float(KAPPA * rho * c + w),
        }
    if learned_state:
        # Persistent 8 KB/function state, allocated by state exposure in minutes.
        # This matches the paper's tiny state-cost convention and keeps the
        # registered-cost decomposition additive across states.
        # Each function contributes one state label per tick.
        pass
    return {
        "csr": float(cold / max(total_inv, 1)),
        "csr_pct": float(100.0 * cold / max(total_inv, 1)),
        "cold_starts": int(cold),
        "total_invocations": int(total_inv),
        "wm_total_gb_s": float(wm),
        "wm_per_1k_inv": float(wm / max(total_inv / 1000.0, 1e-10)),
        "registered_cost_gb_s_no_state": float(KAPPA * rho * cold + wm),
        "state_gb_s": float(state_gbs_total),
        "registered_cost_gb_s": float(KAPPA * rho * cold + wm + state_gbs_total),
        "by_state": out_state,
    }


def state_exposure(state_code):
    out = {}
    total_ticks = state_code.size
    for i, name in enumerate(SLICE_NAMES):
        ticks = int(np.sum(state_code == i))
        out[name] = {
            "function_minutes": ticks,
            "share_function_minutes": float(ticks / max(total_ticks, 1)),
        }
    return out


def add_state_costs(row, state_code, learned_state):
    if not learned_state:
        return
    for i, name in enumerate(SLICE_NAMES):
        ticks = int(np.sum(state_code == i))
        gbs = STATE_GB_PER_FN * ticks * TICK_SEC
        row["by_state"][name]["state_gb_s"] = float(gbs)
        row["by_state"][name]["registered_cost_gb_s"] = (
            row["by_state"][name]["registered_cost_gb_s_no_state"] + gbs
        )
        row["state_gb_s"] += float(gbs)
    row["registered_cost_gb_s"] = row["registered_cost_gb_s_no_state"] + row["state_gb_s"]
    for name in SLICE_NAMES:
        row["by_state"][name].setdefault("state_gb_s", 0.0)
        row["by_state"][name].setdefault(
            "registered_cost_gb_s",
            row["by_state"][name]["registered_cost_gb_s_no_state"],
        )


def run_sliced_config(args):
    (
        experiment,
        method,
        rho,
        seeds,
        counts,
        prewarm,
        keepalive,
        state_code,
        dur_means,
        dur_stds,
        cold_init,
        split,
        learned_state,
    ) = args
    results = []
    for seed in seeds:
        t0 = time.time()
        per_func = []
        for fi in range(counts.shape[0]):
            per_func.append(
                simulate_function_sliced(
                    counts[fi],
                    prewarm[fi],
                    keepalive[fi],
                    state_code[fi],
                    float(dur_means[fi]),
                    float(dur_stds[fi]),
                    seed=seed * 100003 + fi,
                    cold_mu=cold_init["mu"],
                    cold_sigma=cold_init["sigma"],
                )
            )
        row = aggregate_sliced(per_func, rho, learned_state)
        add_state_costs(row, state_code, learned_state)
        row.update(
            {
                "experiment": experiment,
                "method": method,
                "cost_ratio": float(rho),
                "seed": int(seed),
                "split": split,
                "elapsed_sec": time.time() - t0,
            }
        )
        results.append(row)
    return results


def mature_segments(counts, rates_by_method, state_code, dur_means, dur_stds):
    segments = []
    method_segments = {m: [] for m in rates_by_method}
    state_segments = []
    dm = []
    ds = []
    for fi in range(counts.shape[0]):
        m_ticks = np.flatnonzero(state_code[fi] == SLICE_NAMES.index("M"))
        if len(m_ticks) == 0:
            continue
        start = int(m_ticks[0])
        if start >= counts.shape[1] - 1:
            continue
        segments.append(counts[fi, start:])
        for method, rates in rates_by_method.items():
            method_segments[method].append(rates[fi, start:])
        local_state = np.zeros(counts.shape[1] - start, dtype=np.int8)
        if len(local_state) > MATURE_WASHOUT:
            local_state[MATURE_WASHOUT:] = 1
        state_segments.append(local_state)
        dm.append(float(dur_means[fi]))
        ds.append(float(dur_stds[fi]))
    if not segments:
        return None
    max_len = max(len(x) for x in segments)
    cmat = np.zeros((len(segments), max_len), dtype=counts.dtype)
    smat = np.zeros((len(segments), max_len), dtype=np.int8)
    for i, (c, s) in enumerate(zip(segments, state_segments)):
        cmat[i, : len(c)] = c
        smat[i, : len(s)] = s
        if len(s) < max_len:
            smat[i, len(s):] = 0
    rseg = {}
    for method, mats in method_segments.items():
        rmat = np.zeros((len(segments), max_len), dtype=np.float32)
        for i, r in enumerate(mats):
            rmat[i, : len(r)] = r
        rseg[method] = rmat
    return cmat, rseg, smat, np.asarray(dm), np.asarray(ds)


def remap_mature_row(row):
    # The sliced simulator has only four labels globally. For the clean mature
    # diagnostic, Z0 is the common reset/washout prefix and W is post-washout.
    row["by_mature_phase"] = {
        "reset_washout": row["by_state"]["Z0"],
        f"post_washout_{MATURE_WASHOUT}min": row["by_state"]["W"],
    }
    return row


def mean_rows(rows):
    if not rows:
        return None
    return {
        "csr_pct": float(np.mean([r["csr_pct"] for r in rows])),
        "csr_std_pct": float(np.std([r["csr_pct"] for r in rows])),
        "wm_per_1k_inv": float(np.mean([r["wm_per_1k_inv"] for r in rows])),
        "registered_cost_gb_s": float(np.mean([r["registered_cost_gb_s"] for r in rows])),
        "registered_cost_per_1k_inv": float(
            np.mean([
                r["registered_cost_gb_s"] / max(r["total_invocations"] / 1000.0, 1e-10)
                for r in rows
            ])
        ),
        "n": len(rows),
    }


def add_readouts(out):
    rows = out["results"]
    summary = {}
    for experiment in sorted({r["experiment"] for r in rows}):
        summary[experiment] = {}
        for split in ["S1", "S2", "S3"]:
            summary[experiment][split] = {}
            for rho in target_rhos(split):
                summary[experiment][split][f"{rho:g}"] = {}
                for method in sorted({r["method"] for r in rows if r["experiment"] == experiment}):
                    cell = [
                        r
                        for r in rows
                        if r["experiment"] == experiment
                        and r["split"] == split
                        and r["method"] == method
                        and abs(float(r["cost_ratio"]) - rho) < 1e-9
                    ]
                    rec = mean_rows(cell)
                    if rec:
                        summary[experiment][split][f"{rho:g}"][method] = rec
    out["readout"] = summary


def write_tables(out):
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = out["results"]

    fact_lines = [
        "split,method,pattern,rho,csr_pct,csr_std_pct,wm_per_1k_inv,cost_per_1k_inv,delta_cost_per_1k_vs_ewma,n"
    ]
    for split in ["S1", "S2", "S3"]:
        refs = [
            r
            for r in rows
            if r["experiment"] == "factorial"
            and r["split"] == split
            and r["method"] == "B4a_ewma"
            and abs(float(r["cost_ratio"]) - 10.0) < 1e-9
        ]
        ref = mean_rows(refs)
        ref_cost = ref["registered_cost_per_1k_inv"] if ref else np.nan
        for method in [
            "B4a_ewma",
            "A5_faithful",
            "G_EEE",
            "G_EEL",
            "G_ELE",
            "G_ELL",
            "G_LEE",
            "G_LEL",
            "G_LLE",
            "G_LLL",
        ]:
            cell = [
                r
                for r in rows
                if r["experiment"] == "factorial"
                and r["split"] == split
                and r["method"] == method
                and abs(float(r["cost_ratio"]) - 10.0) < 1e-9
            ]
            rec = mean_rows(cell)
            if not rec:
                continue
            fact_lines.append(
                f"{split},{method},{out['methods'].get(method, {}).get('pattern', '')},10,"
                f"{rec['csr_pct']:.6f},{rec['csr_std_pct']:.6f},"
                f"{rec['wm_per_1k_inv']:.3f},{rec['registered_cost_per_1k_inv']:.3f},"
                f"{rec['registered_cost_per_1k_inv'] - ref_cost:.3f},{rec['n']}"
            )
    (TABLES / "T_r19_gate_factorial_rho10.csv").write_text("\n".join(fact_lines) + "\n")

    slice_lines = [
        "split,method,rho,state,share_function_minutes,total_invocations,cold_starts,csr_pct,wm_per_1k_inv,cost_per_1k_inv"
    ]
    for split in ["S1", "S2", "S3"]:
        for method in ["B4a_ewma", "A5_faithful", "G_ELL", "G_LEE", "G_ELE"]:
            cell = [
                r
                for r in rows
                if r["experiment"] == "factorial"
                and r["split"] == split
                and r["method"] == method
                and abs(float(r["cost_ratio"]) - 10.0) < 1e-9
            ]
            if not cell:
                continue
            for state in SLICE_NAMES:
                inv = np.mean([r["by_state"][state]["total_invocations"] for r in cell])
                cold = np.mean([r["by_state"][state]["cold_starts"] for r in cell])
                wm = np.mean([r["by_state"][state]["wm_total_gb_s"] for r in cell])
                cost = np.mean([r["by_state"][state].get("registered_cost_gb_s", r["by_state"][state]["registered_cost_gb_s_no_state"]) for r in cell])
                share = out["routing"][split]["state_exposure"][state]["share_function_minutes"]
                slice_lines.append(
                    f"{split},{method},10,{state},{share:.8f},"
                    f"{inv:.3f},{cold:.3f},{100.0 * cold / max(inv, 1):.6f},"
                    f"{wm / max(inv / 1000.0, 1e-10):.3f},"
                    f"{cost / max(inv / 1000.0, 1e-10):.3f}"
                )
    (TABLES / "T_r19_state_slices_rho10.csv").write_text("\n".join(slice_lines) + "\n")

    mature_lines = [
        "split,method,rho,phase,total_invocations,cold_starts,csr_pct,wm_per_1k_inv,cost_per_1k_inv"
    ]
    for split in ["S1", "S2", "S3"]:
        for method in ["B4a_ewma", "A5_faithful"]:
            cell = [
                r
                for r in rows
                if r["experiment"] == "mature_clean"
                and r["split"] == split
                and r["method"] == method
                and abs(float(r["cost_ratio"]) - 10.0) < 1e-9
            ]
            if not cell:
                continue
            for phase in ["reset_washout", f"post_washout_{MATURE_WASHOUT}min"]:
                inv = np.mean([r["by_mature_phase"][phase]["total_invocations"] for r in cell])
                cold = np.mean([r["by_mature_phase"][phase]["cold_starts"] for r in cell])
                wm = np.mean([r["by_mature_phase"][phase]["wm_total_gb_s"] for r in cell])
                cost = np.mean([r["by_mature_phase"][phase].get("registered_cost_gb_s", r["by_mature_phase"][phase]["registered_cost_gb_s_no_state"]) for r in cell])
                mature_lines.append(
                    f"{split},{method},10,{phase},{inv:.3f},{cold:.3f},"
                    f"{100.0 * cold / max(inv, 1):.6f},"
                    f"{wm / max(inv / 1000.0, 1e-10):.3f},"
                    f"{cost / max(inv / 1000.0, 1e-10):.3f}"
                )
    (TABLES / "T_r19_mature_clean_rho10.csv").write_text("\n".join(mature_lines) + "\n")


def load_expected_rows():
    rows = []
    for name in [
        "sim_results_des_revision.json",
        "revision_r16_faithful_primary_experiment.json",
        "revision_r17_faithful_gates_2021.json",
    ]:
        path = RUNS / name
        if not path.exists():
            continue
        data = json.load(open(path))
        rows.extend(data.get("results", data if isinstance(data, list) else []))
    return rows


def reconcile(out):
    expected = load_expected_rows()
    method_map = {
        "B4a_ewma": "B4a_ewma",
        "A5_faithful": "A5_faithful",
        "G_ELL": "A5_gated_v3_faithful",
        "G_LEE": "A5_gated_v4_faithful",
        "G_ELE": "A5_gated_aq_faithful",
    }
    exp_by = {
        (r["method"], r["split"], float(r["cost_ratio"]), int(r["seed"])): r
        for r in expected
    }
    checks = {}
    for ours in out["results"]:
        if ours["experiment"] != "factorial" or ours["method"] not in method_map:
            continue
        key = (
            method_map[ours["method"]],
            ours["split"],
            float(ours["cost_ratio"]),
            int(ours["seed"]),
        )
        exp = exp_by.get(key)
        if not exp:
            continue
        ckey = f"{ours['method']}|{ours['split']}|{ours['cost_ratio']:g}"
        rec = checks.setdefault(
            ckey,
            {
                "n": 0,
                "max_abs_cold_diff": 0,
                "max_abs_total_diff": 0,
                "max_abs_wm_diff_gbs": 0.0,
                "max_abs_csr_diff": 0.0,
            },
        )
        rec["n"] += 1
        rec["max_abs_cold_diff"] = max(
            rec["max_abs_cold_diff"], abs(int(ours["cold_starts"]) - int(exp["cold_starts"]))
        )
        rec["max_abs_total_diff"] = max(
            rec["max_abs_total_diff"],
            abs(int(ours["total_invocations"]) - int(exp["total_invocations"])),
        )
        rec["max_abs_wm_diff_gbs"] = max(
            rec["max_abs_wm_diff_gbs"],
            abs(float(ours["wm_total_gb_s"]) - float(exp["wm_total_gb_s"])),
        )
        rec["max_abs_csr_diff"] = max(
            rec["max_abs_csr_diff"], abs(float(ours["csr"]) - float(exp["csr"]))
        )
    out["reconciliation"] = checks


def print_rho10_summary(out):
    print("\n=== R19 rho=10 summary: factorial registered cost per 1k inv ===")
    rows = out["results"]
    for split in ["S1", "S2", "S3"]:
        print(f"\n{split}")
        ref = mean_rows([
            r for r in rows
            if r["experiment"] == "factorial" and r["split"] == split
            and r["method"] == "B4a_ewma" and abs(float(r["cost_ratio"]) - 10.0) < 1e-9
        ])
        ref_cost = ref["registered_cost_per_1k_inv"] if ref else np.nan
        for method in ["B4a_ewma", "A5_faithful", "G_EEE", "G_ELE", "G_ELL", "G_LEE", "G_LLL"]:
            rec = mean_rows([
                r for r in rows
                if r["experiment"] == "factorial" and r["split"] == split
                and r["method"] == method and abs(float(r["cost_ratio"]) - 10.0) < 1e-9
            ])
            if rec:
                print(
                    f"  {method:12s} CSR={rec['csr_pct']:.4f}% "
                    f"WM={rec['wm_per_1k_inv']:.1f} "
                    f"C/1k={rec['registered_cost_per_1k_inv']:.1f} "
                    f"dC={rec['registered_cost_per_1k_inv'] - ref_cost:+.1f}"
                )

    print("\n=== R19 rho=10 summary: clean gate-mature post-washout ===")
    for split in ["S1", "S2", "S3"]:
        print(f"\n{split}")
        for method in ["B4a_ewma", "A5_faithful"]:
            rec = mean_rows([
                r for r in rows
                if r["experiment"] == "mature_clean" and r["split"] == split
                and r["method"] == method and abs(float(r["cost_ratio"]) - 10.0) < 1e-9
            ])
            if not rec:
                continue
            phase = f"post_washout_{MATURE_WASHOUT}min"
            cell = [
                r for r in rows
                if r["experiment"] == "mature_clean" and r["split"] == split
                and r["method"] == method and abs(float(r["cost_ratio"]) - 10.0) < 1e-9
            ]
            inv = np.mean([r["by_mature_phase"][phase]["total_invocations"] for r in cell])
            cold = np.mean([r["by_mature_phase"][phase]["cold_starts"] for r in cell])
            wm = np.mean([r["by_mature_phase"][phase]["wm_total_gb_s"] for r in cell])
            cost = np.mean([r["by_mature_phase"][phase].get("registered_cost_gb_s", r["by_mature_phase"][phase]["registered_cost_gb_s_no_state"]) for r in cell])
            print(
                f"  {method:12s} CSR={100.0 * cold / max(inv, 1):.4f}% "
                f"WM={wm / max(inv / 1000.0, 1e-10):.1f} "
                f"C/1k={cost / max(inv / 1000.0, 1e-10):.1f}"
            )


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
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
    out["methods"] = {
        "B4a_ewma": {"description": "Official EWMA baseline"},
        "A5_faithful": {"description": "Faithful Section-3 WINTER head"},
    }
    for pattern in ["EEE", "EEL", "ELE", "ELL", "LEE", "LEL", "LLE", "LLL"]:
        out["methods"][f"G_{pattern}"] = {
            "description": "Prototype on Z0; E=EWMA/L=learned on W,YC,M",
            "pattern": pattern,
            "aliases": {
                "ELL": "v3",
                "LEE": "v4",
                "ELE": "age-qualified WINTER-G",
            }.get(pattern, ""),
        }
    save(out)

    factorial_methods = ["B4a_ewma", "A5_faithful"] + [f"G_{p}" for p in ["EEE", "EEL", "ELE", "ELL", "LEE", "LEL", "LLE", "LLL"]]
    mature_methods = ["B4a_ewma", "A5_faithful"]

    for split in ["S1", "S2", "S3"]:
        test_idx, counts, feats, dm, ds = split_arrays(features, counts_all, splits, dur_df, split)
        print(f"### {split}: {len(test_idx)} funcs, {counts.shape[1]} ticks", flush=True)
        t0 = time.time()
        faithful = rates_a5_faithful(counts.astype(np.float32), feats, trainer, pm)
        ewma = rates_ewma(counts)
        proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
        masks, state_code, age = state_masks(counts, cum_prev)
        out["routing"][split] = {
            "state_exposure": state_exposure(state_code),
            "invocation_exposure": {
                name: int(counts[masks[name]].sum()) for name in SLICE_NAMES
            },
            "n_functions": int(counts.shape[0]),
            "n_ticks": int(counts.shape[1]),
            "note": "States use evaluation-window-local K_t and age_t before tick t.",
        }
        print(f"  rate matrices and masks done in {time.time() - t0:.0f}s", flush=True)

        rate_mats = {
            "B4a_ewma": ewma,
            "A5_faithful": faithful,
        }
        for pattern in ["EEE", "EEL", "ELE", "ELL", "LEE", "LEL", "LLE", "LLL"]:
            rate_mats[f"G_{pattern}"] = compose_gate_rates(proto, ewma, faithful, masks, pattern)

        done = {result_key(r) for r in out["results"]}
        jobs = []
        for rho in target_rhos(split):
            tau = newsvendor_quantile(rho)
            for method in factorial_methods:
                seeds = [
                    s for s in SEEDS
                    if ("factorial", method, split, float(rho), s) not in done
                ]
                if not seeds:
                    continue
                pw, ka = decisions_from_rates(rate_mats[method], tau)
                jobs.append(
                    (
                        "factorial",
                        method,
                        float(rho),
                        seeds,
                        counts,
                        pw,
                        ka,
                        state_code,
                        dm,
                        ds,
                        COLD_INIT,
                        split,
                        method_uses_learned(method),
                    )
                )

        mature = mature_segments(
            counts,
            {m: rate_mats[m] for m in mature_methods},
            state_code,
            dm,
            ds,
        )
        if mature is not None:
            m_counts, m_rates, m_state, m_dm, m_ds = mature
            out["routing"][split]["mature_clean"] = {
                "n_functions_with_mature_segment": int(m_counts.shape[0]),
                "max_segment_ticks": int(m_counts.shape[1]),
                "washout_min": MATURE_WASHOUT,
                "note": "Each function is restarted from an empty sandbox at first gate-mature tick; post-washout rows omit the first 120 min.",
            }
            for rho in target_rhos(split):
                tau = newsvendor_quantile(rho)
                for method in mature_methods:
                    seeds = [
                        s for s in SEEDS
                        if ("mature_clean", method, split, float(rho), s) not in done
                    ]
                    if not seeds:
                        continue
                    pw, ka = decisions_from_rates(m_rates[method], tau)
                    jobs.append(
                        (
                            "mature_clean",
                            method,
                            float(rho),
                            seeds,
                            m_counts,
                            pw,
                            ka,
                            m_state,
                            m_dm,
                            m_ds,
                            COLD_INIT,
                            split,
                            method_uses_learned(method),
                        )
                    )

        if not jobs:
            print(f"  {split}: all requested jobs already complete", flush=True)
            continue

        print(f"  running {len(jobs)} configs on 4 workers", flush=True)
        with ProcessPoolExecutor(max_workers=4) as ex:
            for res in ex.map(run_sliced_config, jobs):
                for row in res:
                    if row["experiment"] == "mature_clean":
                        remap_mature_row(row)
                out["results"].extend(res)
                r0 = res[0]
                print(
                    f"    {r0['experiment']} {split} {r0['method']} "
                    f"rho={r0['cost_ratio']:g} "
                    f"CSR={np.mean([r['csr_pct'] for r in res]):.4f}% "
                    f"WM={np.mean([r['wm_per_1k_inv'] for r in res]):.1f}",
                    flush=True,
                )
                save(out)

        save(out)

    add_readouts(out)
    reconcile(out)
    save(out)
    write_tables(out)
    print_rho10_summary(out)
    print(f"wrote {OUT}")
    print(f"wrote {TABLES / 'T_r19_gate_factorial_rho10.csv'}")
    print(f"wrote {TABLES / 'T_r19_state_slices_rho10.csv'}")
    print(f"wrote {TABLES / 'T_r19_mature_clean_rho10.csv'}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", "/data/260715/site-packages:.")
    main()
