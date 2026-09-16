# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""M2-A: age-qualified gate on a held-out 2019 onboarding cohort, 48h horizon.

Pre-registration: results/runs/revision_m2_PREREG.md (written before this ran).

Machinery, constants and arm semantics are those of revision_a4_crosstrace.py
(2021-trained body + 2021 s2_train prototypes -> 2019 cohort). Three changes,
all declared in the pre-registration:

  1. Horizon 240 -> 2880 ticks, so the age qualifier can fire.
  2. Cohort = registered rule minus the 772-function union of the E2
     cohort-definition sweep (which contains the published n=100 cohort).
  3. Arms gain gated_aq_<A> for A in {180, 360, 720, 1440, 2880}.

Two implementation changes are forced by the horizon and change no result:

  * features are read as one contiguous slice per function instead of one
    memmap window per (function, tick) -- 5.05M random reads become 1,755
    sequential ones;
  * the prior-biased ridge accumulates A = Phi^T Phi + lam I and b = Phi^T Y
    incrementally instead of re-forming the prefix product each tick, which
    turns O(W^2) into O(W). Assertion 5 checks the two agree.

Run: SF_DATA_DIR=processed_2019 PYTHONPATH=/data/260715/site-packages:. \
     python3.13 scripts/revision_m2a_holdout_cohort.py [--pilot N]
Output: results/runs/revision_m2a_holdout.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)  # MKL batched-linalg thrashing on this VM

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("SF_DATA_DIR", "processed_2019")
os.environ["SF_RUNS_DIR"] = str(PROJECT_ROOT / "results_azure2021" / "runs")

from scripts.eval_adapt_biased import (  # noqa: E402
    load_trainer, build_prototypes, PROCESSED_DIR, CKPT_RUNS,
)
from scripts.phase63_onboarding_drift import DES_SEEDS, DEVICE  # noqa: E402
from scripts.phase6_des import decisions_from_rates, COLD_INIT  # noqa: E402
from src.sim.des import simulate_function  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402

L = 60
W = 2880                       # 48h horizon (pre-registered)
RHOS = [1.0, 10.0, 100.0]
MIN_IDLE_PREFIX = 3 * 1440
MIN_DAY1_INV = 30
GATE_THRESHOLD = 100
AGE_GRID = [180, 360, 720, 1440, 2880]
AGE_PRIMARY = 720
AGE_BINS = [0, 180, 360, 720, 1440, 2880]   # amendment K.2
QL = np.linspace(0.05, 0.95, N_QUANTILES)
TAU = newsvendor_quantile(10.0)
CKPT_MD5 = "a2aad046f16aa857d939dd6f2f9d7862"
PROC_2021 = PROJECT_ROOT / "data" / "processed"
BOOT_N, BOOT_SEED = 10000, 260715


# ---------------------------------------------------------------- cohort ----
def scan_trace(counts):
    """first arrival, day-one volume and time-to-100 for every function."""
    N, T = counts.shape
    first = np.full(N, -1, dtype=np.int64)
    day1 = np.zeros(N, dtype=np.int64)
    t100 = np.full(N, -1, dtype=np.int64)
    for f in range(N):
        row = np.asarray(counts[f])
        nz = np.flatnonzero(row)
        if len(nz) == 0:
            continue
        t0 = int(nz[0])
        first[f] = t0
        day1[f] = row[t0:t0 + 1440].sum()
        cs = np.cumsum(row[t0:])
        k = np.searchsorted(cs, GATE_THRESHOLD)
        if k < len(cs):
            t100[f] = int(k)
    return first, day1, t100


def e2_exclusion_union(first, day1, T):
    """Reproduce the 772-function union of revision_e2_cohort_sweep.py."""
    union = set()
    for gap in (1, 3, 7):
        for thr in (10, 30, 100):
            mask = (first >= gap * 1440) & (first < T - 1440) & (day1 >= thr)
            cand = np.flatnonzero(mask)
            rng = np.random.default_rng(42)
            if len(cand) > 100:
                idx = rng.choice(len(cand), 100, replace=False)
                picked = sorted(cand[i] for i in idx)
            else:
                picked = sorted(cand)
            union |= {int(x) for x in picked}
    return union


# ------------------------------------------------------------- embedding ----
def embed_cohort(features, pts, body, nF, chunk=96):
    """phi[F, W, 64] from one contiguous feature slice per function."""
    F_n = len(pts)
    phi = torch.zeros(F_n, W, 64)
    slabs = np.empty((F_n, L + W, nF), dtype=np.float32)
    for i, (fi, t0) in enumerate(pts):
        slabs[i] = features[fi, t0 - L:t0 + W]
    body.eval()
    with torch.no_grad():
        for w0 in range(0, W, chunk):
            w1 = min(w0 + chunk, W)
            wins = np.lib.stride_tricks.sliding_window_view(
                slabs[:, w0:w1 + L - 1], L, axis=1)     # [F, w1-w0, nF, L]
            x = torch.from_numpy(
                np.ascontiguousarray(wins.transpose(0, 1, 3, 2))
            ).reshape(-1, L, nF).to(DEVICE)
            phi[:, w0:w1] = body(x).cpu().reshape(F_n, w1 - w0, 64)
            print(f"  embed {w1}/{W}", flush=True)
    return phi


# ------------------------------------------------------------------ ridge ----
def rate_from_logq_rows(pred_q):
    val = np.array([np.interp(TAU, QL, row) for row in np.atleast_2d(pred_q)])
    return np.expm1(np.maximum(val, 0.0))


def ridge_rates(phi, y, centroids, proto_w, lam, F_n):
    """Incremental prior-biased ridge; returns learned / proto-only rates."""
    import torch.nn.functional as Fun
    n_out = N_QUANTILES
    # A/b accumulate in float64: the ridge solve is ill-conditioned enough
    # that float32 round-off moves the served rate (see check_incremental).
    phi_g, y_g = phi.to(DEVICE), y.to(DEVICE).double()
    cen_g, pw_g = centroids.to(DEVICE), proto_w.to(DEVICE).double()
    eye_g = torch.eye(64, device=DEVICE, dtype=torch.float64)
    A = lam * eye_g.unsqueeze(0).repeat(F_n, 1, 1)
    b = torch.zeros(F_n, 64, n_out, device=DEVICE, dtype=torch.float64)
    r_learn = np.zeros((F_n, W), dtype=np.float32)
    r_zero = np.zeros((F_n, W), dtype=np.float32)
    assign0 = None
    for w in range(W):
        phi_now = phi_g[:, w].double()
        ids = Fun.cosine_similarity(phi_now.unsqueeze(1),
                                    cen_g.unsqueeze(0), dim=2).argmax(dim=1)
        if w == 0:
            assign0 = ids.cpu().numpy().copy()
        Wp = pw_g[ids]
        W_p = Wp.clone() if w == 0 else torch.linalg.solve(A, b + lam * Wp)
        r_learn[:, w] = rate_from_logq_rows(
            (phi_now.unsqueeze(1) @ W_p).squeeze(1).cpu().numpy())
        r_zero[:, w] = rate_from_logq_rows(
            (phi_now.unsqueeze(1) @ Wp).squeeze(1).cpu().numpy())
        A = A + phi_now.unsqueeze(2) @ phi_now.unsqueeze(1)
        b = b + phi_now.unsqueeze(2) @ y_g[:, w].reshape(F_n, 1, 1).expand(
            F_n, 1, n_out)
        if (w + 1) % 480 == 0:
            print(f"  ridge {w+1}/{W}", flush=True)
    return r_learn, r_zero, assign0


def check_incremental(phi, y, lam, F_n, n_ticks=100):
    """Assertion 5: incremental accumulation == batch prefix solve."""
    n_out = N_QUANTILES
    phi_g = phi[:, :n_ticks].to(DEVICE).double()
    y_g = y[:, :n_ticks].to(DEVICE).double()
    eye_g = torch.eye(64, device=DEVICE, dtype=torch.float64)
    A = lam * eye_g.unsqueeze(0).repeat(F_n, 1, 1)
    b = torch.zeros(F_n, 64, n_out, device=DEVICE, dtype=torch.float64)
    for w in range(n_ticks):
        p = phi_g[:, w]
        A = A + p.unsqueeze(2) @ p.unsqueeze(1)
        b = b + p.unsqueeze(2) @ y_g[:, w].reshape(F_n, 1, 1).expand(
            F_n, 1, n_out)
    ps = phi_g
    ys = y_g.unsqueeze(-1).expand(F_n, n_ticks, n_out)
    A_ref = ps.transpose(1, 2) @ ps + lam * eye_g.unsqueeze(0)
    b_ref = ps.transpose(1, 2) @ ys
    da = float((A - A_ref).abs().max() / A_ref.abs().max())
    db = float((b - b_ref).abs().max() / b_ref.abs().max())
    # what actually matters downstream: the served rate
    last = phi_g[:, -1].unsqueeze(1)
    r_inc = rate_from_logq_rows((last @ torch.linalg.solve(A, b)
                                 ).squeeze(1).cpu().numpy())
    r_ref = rate_from_logq_rows((last @ torch.linalg.solve(A_ref, b_ref)
                                 ).squeeze(1).cpu().numpy())
    dr = float(np.abs(r_inc - r_ref).max())
    return da, db, dr


# -------------------------------------------------------------------- DES ----
def run_arm_seed(args):
    """One (arm, rho, seed) over the whole cohort; the parallel unit."""
    arm, rho, seed, seg_counts, pw, ka, dm, ds = args
    F_n = seg_counts.shape[0]
    tot_c = tot_n = 0.0
    cold_all = np.zeros(F_n)
    cold_pre = np.zeros(F_n)     # [age 600, 720)
    cold_post = np.zeros(F_n)    # [age 720, 840)
    wm = np.zeros(F_n)
    # pre-registration amendment K.2: age bins, so any prefix window is a sum
    bins = np.zeros((F_n, len(AGE_BINS) - 1))
    for f in range(F_n):
        r = simulate_function(seg_counts[f], pw[f], ka[f],
                              float(dm[f]), float(ds[f]),
                              seed=seed * 7919 + f,
                              cold_mu=COLD_INIT["mu"],
                              cold_sigma=COLD_INIT["sigma"],
                              track_rolling=True)
        rc = r["roll_cold"]
        tot_c += rc.sum()
        tot_n += r["roll_total"].sum()
        cold_all[f] = rc.sum()
        cold_pre[f] = rc[600:720].sum()
        cold_post[f] = rc[720:840].sum()
        for i in range(len(AGE_BINS) - 1):
            bins[f, i] = rc[AGE_BINS[i]:AGE_BINS[i + 1]].sum()
        wm[f] = r["idle_mem_gb_s"]
    return arm, rho, {"cold": tot_c, "inv": tot_n, "func_cold": cold_all,
                      "func_wm": wm, "pre": cold_pre.sum(),
                      "post": cold_post.sum(), "func_cold_bins": bins}


def boot_ci(diff, n=BOOT_N, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rho10only", action="store_true",
                    help="amendment K.2 binned pass: rho=10 arms only")
    args = ap.parse_args()

    ck = CKPT_RUNS / "best_anil_ridge_s2_s0.pt"
    md5 = hashlib.md5(open(ck, "rb").read()).hexdigest()
    assert md5 == CKPT_MD5, f"checkpoint md5 {md5} != registered {CKPT_MD5}"
    print(f"checkpoint OK ({ck}, md5 {md5})", flush=True)

    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    nF = features.shape[2]
    T = counts.shape[1]

    cache = Path("/tmp/claude-1000/-data-260715/"
                 "d735e123-b9b7-473b-8a34-6b51820575ec/scratchpad/c2019.npz")
    if cache.exists():
        z = np.load(cache)
        first, day1, t100 = z["first"], z["day1"], z["t100"]
    else:
        first, day1, t100 = scan_trace(counts)

    union = e2_exclusion_union(first, day1, T)
    assert len(union) == 772, f"exclusion union {len(union)} != 772"
    mask = (first >= MIN_IDLE_PREFIX) & (first < T - W) & (day1 >= MIN_DAY1_INV)
    cand = [int(f) for f in np.flatnonzero(mask)]
    holdout = [f for f in cand if f not in union]
    assert not (set(holdout) & union), "holdout intersects exclusion union"
    fires = [f for f in holdout if 0 <= t100[f] < AGE_PRIMARY]
    print(f"candidates {len(cand)} | holdout {len(holdout)} | "
          f"AND-fires {len(fires)}", flush=True)

    if args.pilot:
        rng = np.random.default_rng(42)
        holdout = sorted(rng.choice(holdout, args.pilot, replace=False).tolist())
    pts = [(f, int(first[f])) for f in holdout]
    F_n = len(pts)

    trainer, biased_head = load_trainer(nF, seed=0)
    feat21 = np.load(PROC_2021 / "features.npy", mmap_mode="r")
    cnt21 = np.load(PROC_2021 / "counts.npy", mmap_mode="r")
    spl21 = np.load(PROC_2021 / "splits.npz")
    centroids, proto_w = build_prototypes(
        trainer, biased_head, feat21, cnt21, spl21["s2_train"])

    t = time.time()
    phi = embed_cohort(features, pts, trainer.body, nF)
    print(f"embed stage {time.time()-t:.0f}s", flush=True)

    seg_counts = np.stack([np.asarray(counts[fi, t0:t0 + W], dtype=np.int64)
                           for (fi, t0) in pts])
    y = torch.from_numpy(np.log1p(seg_counts.astype(np.float32)))
    lam = trainer.head.ridge_lambda.item()

    da, db, dr = check_incremental(phi, y, lam, F_n)
    print(f"incremental-vs-batch rel|dA|={da:.2e} rel|db|={db:.2e} "
          f"max rate diff={dr:.2e}", flush=True)
    assert da < 1e-4 and db < 1e-4 and dr < 1e-4, \
        "incremental ridge disagrees with batch"

    t = time.time()
    r_learn, r_zero, assign0 = ridge_rates(
        phi, y, centroids, proto_w, lam, F_n)
    print(f"ridge stage {time.time()-t:.0f}s", flush=True)

    rates = {"A5_proto": r_learn, "A5_proto_zero": r_zero}
    ew = np.zeros(F_n)
    rates["B4a_ewma"] = np.zeros((F_n, W), dtype=np.float32)
    for w in range(W):
        rates["B4a_ewma"][:, w] = np.expm1(np.maximum(ew, 0))
        ew = 0.1 * np.log1p(seg_counts[:, w].astype(np.float64)) + 0.9 * ew
    rates["Oracle"] = seg_counts.astype(np.float32)

    cum = np.zeros((F_n, W), dtype=np.int64)
    cum[:, 1:] = np.cumsum(seg_counts[:, :-1], axis=1)
    age = np.tile(np.arange(W, dtype=np.int64), (F_n, 1))
    zero_h = cum == 0
    mid = (cum > 0) & (cum < GATE_THRESHOLD)
    conv = cum >= GATE_THRESHOLD
    rates["gated_v3"] = np.where(zero_h, r_zero,
                        np.where(mid, rates["B4a_ewma"], r_learn))
    rates["gated_v4"] = np.where(zero_h, r_zero,
                        np.where(mid, r_learn, rates["B4a_ewma"]))
    routing = {"zero_history": float(zero_h.mean()),
               "midband_lt100": float(mid.mean()),
               "converged_ge100": float(conv.mean())}
    for A in AGE_GRID:
        young = conv & (age < A)
        rates[f"gated_aq_{A}"] = np.where(
            zero_h, r_zero,
            np.where(mid, rates["B4a_ewma"],
                     np.where(young, r_learn, rates["B4a_ewma"])))
        routing[f"conv_young_learned_A{A}"] = float(young.mean())

    pw_b1 = np.zeros((F_n, W), dtype=np.int32)
    for f in range(F_n):
        act = (seg_counts[f] > 0).astype(np.int32)
        cv = np.convolve(act, np.ones(10, dtype=np.int32))[:W]
        pw_b1[f, 1:] = (cv[:-1] > 0).astype(np.int32)

    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"]
                                 for (fi, _) in pts]), nan=1.0)
    ds = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"]
                                 for (fi, _) in pts]), nan=0.5)

    out = {"prereg": "revision_m2_PREREG.md",
           "config": "cross-trace (2021 body+prototypes -> held-out 2019 cohort)",
           "checkpoint_md5": md5, "horizon_ticks": W,
           "gate_threshold": GATE_THRESHOLD, "age_grid": AGE_GRID,
           "n_candidates": len(cand), "n_exclusion_union": len(union),
           "n_functions": F_n, "n_and_fires": len(fires),
           "pilot": bool(args.pilot),
           "t100_median_min": float(np.median(
               [t100[f] for f in holdout if 0 <= t100[f] < AGE_PRIMARY])),
           "routing_fractions": routing,
           "assign_hist_k0": np.bincount(assign0, minlength=16).tolist(),
           "incremental_check": {"rel_dA": da, "rel_db": db, "max_rate_diff": dr},
           "by_rho": {}}

    # assertion 4 first: aq_2880 is decision-identical to v3 at this horizon,
    # so it is copied rather than simulated (pre-registration amendment 1.3).
    tau10 = newsvendor_quantile(10.0)
    p1, k1 = decisions_from_rates(rates["gated_aq_2880"], tau10)
    p2, k2 = decisions_from_rates(rates["gated_v3"], tau10)
    ident = bool((p1 == p2).all() and np.allclose(k1, k2))
    out["assert_aq2880_equals_v3"] = ident
    print(f"assert aq_2880 == v3: {ident}", flush=True)

    # coverage (pre-registration amendment 1): every arm at rho=10; the
    # secondary rho family runs only the arms the endpoints compare.
    del rates["Oracle"], rates["A5_proto_zero"], rates["gated_aq_2880"]
    RHO_ALL = 10.0
    RHO_SECONDARY = [1.0, 100.0]
    SECONDARY_ARMS = ["B4a_ewma", "gated_v3", "gated_v4",
                      f"gated_aq_{AGE_PRIMARY}"]
    out["coverage"] = {
        "rho10_arms": sorted(rates) + ["B1_fixed_keepalive"],
        "secondary_rhos": RHO_SECONDARY, "secondary_arms": SECONDARY_ARMS,
        "dropped_from_des": ["Oracle", "A5_proto_zero"],
        "gated_aq_2880": "copied from gated_v3 (assertion 4)"}

    del phi
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    from concurrent.futures import ProcessPoolExecutor
    from scipy import stats as sps
    seg_counts = seg_counts.astype(np.int32)
    tasks = []
    rho_list = [RHO_ALL] if args.rho10only else [RHO_ALL] + RHO_SECONDARY
    for rho in rho_list:
        tau = newsvendor_quantile(rho)
        arms = list(rates) if rho == RHO_ALL else SECONDARY_ARMS
        for arm in arms:
            pw, ka = decisions_from_rates(rates[arm], tau)
            for seed in DES_SEEDS:
                tasks.append((arm, rho, seed, seg_counts, pw, ka, dm, ds))
        if rho == RHO_ALL:
            ka1 = np.full((F_n, W), 10.0, np.float32)
            for seed in DES_SEEDS:
                tasks.append(("B1_fixed_keepalive", rho, seed, seg_counts,
                              pw_b1, ka1, dm, ds))

    acc = {}
    t = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for arm, rho, res in ex.map(run_arm_seed, tasks):
            k = (str(rho), arm)
            a = acc.setdefault(k, {"cold": 0.0, "inv": 0.0, "pre": 0.0,
                                   "post": 0.0,
                                   "func_cold": np.zeros(F_n),
                                   "func_wm": np.zeros(F_n),
                                   "bins": np.zeros((F_n, len(AGE_BINS) - 1)),
                                   "n": 0})
            for fld in ("cold", "inv", "pre", "post"):
                a[fld] += res[fld]
            a["func_cold"] += res["func_cold"]
            a["func_wm"] += res["func_wm"]
            a["bins"] += res["func_cold_bins"]
            a["n"] += 1
            done += 1
            print(f"  [{done}/{len(tasks)}] rho={rho} {arm} "
                  f"({time.time()-t:.0f}s)", flush=True)
    print(f"DES stage {time.time()-t:.0f}s", flush=True)

    for rho in rho_list:
        rho_out = {}
        for (rk, arm), a in acc.items():
            if rk != str(rho):
                continue
            n = a["n"]
            rho_out[arm] = {
                "overall_csr": float(a["cold"] / max(a["inv"], 1e-9)),
                "wm_total": float(a["func_wm"].sum() / n),
                "func_cold": (a["func_cold"] / n).tolist(),
                "func_wm": (a["func_wm"] / n).tolist(),
                "func_cold_bins": (a["bins"] / n).tolist(),
                "age_bins": AGE_BINS,
                "cold_pre_transition": float(a["pre"] / n),
                "cold_post_transition": float(a["post"] / n)}
        if ident and "gated_v3" in rho_out:
            rho_out["gated_aq_2880"] = dict(rho_out["gated_v3"],
                                            copied_from="gated_v3")
        aq = np.array(rho_out[f"gated_aq_{AGE_PRIMARY}"]["func_cold"])
        tests = {}
        for base in ["gated_v4", "gated_v3", "B4a_ewma"]:
            diff = np.array(rho_out[base]["func_cold"]) - aq
            nz = diff[diff != 0]
            tests[f"aq{AGE_PRIMARY}_vs_{base}"] = {
                "mean_diff_cold_per_fn": float(diff.mean()),
                "wilcoxon_p": float(sps.wilcoxon(nz)[1]) if len(nz) >= 6 else 1.0,
                "boot_ci95": boot_ci(diff),
                "wm_ratio": float(rho_out[f"gated_aq_{AGE_PRIMARY}"]["wm_total"]
                                  / max(rho_out[base]["wm_total"], 1e-9))}
        # S4 / S5 (amendment K.2): exposure window [0,720) and carry-over
        i720 = AGE_BINS.index(720)
        aqb = np.array(rho_out[f"gated_aq_{AGE_PRIMARY}"]["func_cold_bins"])
        for base in ["gated_v4", "gated_v3"]:
            bb = np.array(rho_out[base]["func_cold_bins"])
            for label, sl in [("S4_exposure_0_720", slice(0, i720)),
                              ("S5_carryover_720_2880", slice(i720, None))]:
                diff = bb[:, sl].sum(1) - aqb[:, sl].sum(1)
                nz = diff[diff != 0]
                tests[f"{label}_aq{AGE_PRIMARY}_vs_{base}"] = {
                    "mean_diff_cold_per_fn": float(diff.mean()),
                    "wilcoxon_p": (float(sps.wilcoxon(nz)[1])
                                   if len(nz) >= 6 else 1.0),
                    "boot_ci95": boot_ci(diff)}
        rho_out["_tests"] = tests
        out["by_rho"][str(rho)] = rho_out

    name = ("revision_m2a_holdout_pilot.json" if args.pilot else
            "revision_m2a_holdout_binned.json" if args.rho10only else
            "revision_m2a_holdout.json")
    path = PROJECT_ROOT / "results" / "runs" / name
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    json.load(open(path))
    print(f"Saved + verified {path}", flush=True)


if __name__ == "__main__":
    main()
