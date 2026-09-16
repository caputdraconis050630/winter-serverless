# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision E7: composite age-qualified gate, measured end-to-end.

The paper's deployed recommendation (Section Gate) routes a function-minute to
EWMA only when BOTH (cumulative invocations >= 100) AND (wall-clock age >= 720
minutes) hold; otherwise it follows v3's routing:
    cum == 0                     -> prototype zero-shot
    0 < cum < 100                -> EWMA
    cum >= 100 and age < 720     -> learned head (v3's converged arm)
    cum >= 100 and age >= 720    -> EWMA        (v4's converged arm)
Both constants are registered a priori (100 = dataset activity threshold,
720 min = trailing-window constant). No new constants are introduced.

Age is measured as minutes since the function's first observed arrival in the
evaluated series (segment-relative for S3's temporal split; conservative --
it can only delay the EWMA hand-off, never accelerate it).

Part 1 (steady state, 2021): composes A5_gated_aq rates for S1/S2/S3 and runs
the DES at rho in {1, 10, 100} with the campaign's 10 seeds. Comparators live
in sim_results_des_revision.json / revision_e1_v4_2021.json (identical
protocol).

Part 2 (cohort, 2019 cross-trace): recomposes the aq gate on the natural
onboarding cohort of revision_a4_crosstrace.py (identical machinery/constants,
2021-trained checkpoint + prototypes). Since every cohort tick is younger than
720 min, aq must be decision-identical to gated_v3; we assert that per rho and
then run the DES arm anyway so the number is measured, not argued.

Output: results/runs/revision_e7_agegate.json
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)  # MKL thread-thrashing fix (see memory note)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

AGE_MIN = 720           # registered trailing-window constant
GATE_THRESHOLD = 100    # registered activity threshold
STEADY_RHOS = [1.0, 10.0, 100.0]
STEADY_SEEDS = list(range(10))
L = 60

OUT = {"age_min": AGE_MIN, "gate_threshold": GATE_THRESHOLD}


def age_matrix(counts):
    """age[f, t] = t - first-arrival tick (0 before first arrival)."""
    N, T = counts.shape
    age = np.zeros((N, T), dtype=np.int64)
    for f in range(N):
        nz = np.flatnonzero(counts[f])
        if len(nz):
            t0 = nz[0]
            age[f, t0:] = np.arange(T - t0)
    return age


def steady_2021():
    from scripts.phase6_des import (rates_a5, rates_ewma, run_config,
                                    decisions_from_rates, COLD_INIT)
    from scripts.revision_a2_des import rates_proto_prefix
    from scripts.phase63_onboarding_drift import build_prototypes
    from src.decision.newsvendor import newsvendor_quantile
    from src.models.heads import N_QUANTILES
    from src.meta.trainer import ANILMetaTrainer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RUNS_DIR = PROJECT_ROOT / "results" / "runs"
    PROC = PROJECT_ROOT / "data" / "processed"
    features = np.load(PROC / "features.npy")
    counts_all = np.load(PROC / "counts.npy")
    splits_data = np.load(PROC / "splits.npz")
    dur_df = pd.read_csv(PROC / "duration_stats.csv")
    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE)
    trainer.load(RUNS_DIR / "best_anil_ridge_s1_s0.pt")
    pm = build_prototypes(trainer, features, counts_all,
                          splits_data["s1_train"])

    all_results, routing_all = [], {}
    for split in ["S1", "S2", "S3"]:
        key = {"S1": "s1_test", "S2": "s2_test", "S3": "s3_test"}[split]
        test_idx = splits_data[key]
        counts = counts_all[test_idx]
        feats = features[test_idx]
        if split == "S3":
            t0 = int(splits_data.get("s3_test_t_start", [10080])[0])
            counts = counts[:, t0:]
            feats = feats[:, t0:]
        dm = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
             for fi in test_idx]), nan=1.0)
        ds = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
             for fi in test_idx]), nan=0.5)
        print(f"### {split}: composing age-qualified rates...", flush=True)
        t0s = time.time()
        a5 = rates_a5(counts, feats, trainer, DEVICE)
        ew = rates_ewma(counts)
        proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
        age = age_matrix(counts)
        zero = cum_prev == 0
        mid = (cum_prev > 0) & (cum_prev < GATE_THRESHOLD)
        conv_young = (cum_prev >= GATE_THRESHOLD) & (age < AGE_MIN)
        conv_old = (cum_prev >= GATE_THRESHOLD) & (age >= AGE_MIN)
        aq = np.where(zero, proto,
             np.where(mid, ew,
             np.where(conv_young, a5, ew))).astype(np.float32)
        routing_all[split] = {
            "frac_ticks_proto": float(zero.mean()),
            "frac_ticks_mid_ewma": float(mid.mean()),
            "frac_ticks_conv_young_learned": float(conv_young.mean()),
            "frac_ticks_conv_old_ewma": float(conv_old.mean())}
        print(f"    rates in {time.time()-t0s:.0f}s; "
              f"routing {routing_all[split]}", flush=True)
        jobs = []
        for rho in STEADY_RHOS:
            tau = newsvendor_quantile(rho)
            pw, ka = decisions_from_rates(aq, tau)
            jobs.append(("A5_gated_aq", rho, STEADY_SEEDS, counts, pw, ka,
                         dm, ds, COLD_INIT, split))
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=3) as ex:
            for res_list in ex.map(run_config, jobs):
                all_results.extend(res_list)
                r = res_list[0]
                print(f"    aq rho={r['cost_ratio']:6.1f} "
                      f"CSR={np.mean([x['csr'] for x in res_list]):.4f} "
                      f"WM={np.mean([x['wm_per_1k_inv'] for x in res_list]):.0f}",
                      flush=True)
    OUT["steady_2021"] = {"results": all_results, "routing": routing_all,
                          "seeds": STEADY_SEEDS, "rhos": STEADY_RHOS}


def cohort_2019():
    os.environ.setdefault("SF_DATA_DIR", "processed_2019")
    os.environ["SF_RUNS_DIR"] = str(PROJECT_ROOT / "results_azure2021" / "runs")
    from scripts.eval_adapt_biased import (load_trainer, build_prototypes,
                                           PROCESSED_DIR)
    from scripts.phase63_onboarding_drift import DES_SEEDS, DEVICE
    from scripts.phase6_des import decisions_from_rates, COLD_INIT
    from scripts.revision_a4_crosstrace import (find_natural_cohort,
                                                rate_from_logq_rows, W)
    from src.sim.des import simulate_function, rolling_csr
    from src.decision.newsvendor import newsvendor_quantile
    from src.models.heads import N_QUANTILES
    import torch.nn.functional as Fun

    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    nF = features.shape[2]
    PROC_2021 = PROJECT_ROOT / "data" / "processed"
    feat21 = np.load(PROC_2021 / "features.npy", mmap_mode="r")
    cnt21 = np.load(PROC_2021 / "counts.npy", mmap_mode="r")
    spl21 = np.load(PROC_2021 / "splits.npz")

    pts = find_natural_cohort(counts)
    F_n = len(pts)
    print(f"cohort: {F_n} functions", flush=True)
    trainer, biased_head = load_trainer(nF, seed=0)
    centroids, proto_w = build_prototypes(
        trainer, biased_head, feat21, cnt21, spl21["s2_train"])

    trainer.body.eval()
    phi_all = torch.zeros(F_n, W, 64)
    with torch.no_grad():
        for w in range(W):
            batch = []
            for (fi, t0) in pts:
                t = t0 + w
                if t < L:
                    win = np.zeros((L, nF), dtype=np.float32)
                    win[L - t:] = features[fi, :t]
                else:
                    win = np.asarray(features[fi, t - L:t], dtype=np.float32)
                batch.append(torch.from_numpy(win))
            x = torch.stack(batch).to(DEVICE)
            phi_all[:, w] = trainer.body(x).cpu()
    y_all = np.stack([np.log1p(np.asarray(counts[fi, t0:t0 + W],
                                          dtype=np.float64))
                      for (fi, t0) in pts])
    y_t = torch.from_numpy(y_all).float()
    lam = trainer.head.ridge_lambda.item()
    n_out = N_QUANTILES
    pred_proto = np.zeros((F_n, W, n_out), dtype=np.float32)
    pred_zero = np.zeros((F_n, W, n_out), dtype=np.float32)
    phi_g = phi_all.to(DEVICE)
    y_g = y_t.to(DEVICE)
    cen_g = centroids.to(DEVICE)
    pw_g = proto_w.to(DEVICE)
    eye_g = torch.eye(64, device=DEVICE)
    for w in range(W):
        phi_now = phi_g[:, w]
        sims = Fun.cosine_similarity(phi_now.unsqueeze(1),
                                     cen_g.unsqueeze(0), dim=2)
        ids = sims.argmax(dim=1)
        Wp = pw_g[ids]
        if w == 0:
            W_p = Wp.clone()
        else:
            phi_s = phi_g[:, :w]
            y_s = y_g[:, :w].unsqueeze(-1).expand(F_n, w, n_out)
            PhiT = phi_s.transpose(1, 2)
            A = PhiT @ phi_s + lam * eye_g.unsqueeze(0)
            W_p = torch.linalg.solve(A, PhiT @ y_s + lam * Wp)
        pred_proto[:, w] = (phi_now.unsqueeze(1) @ W_p).squeeze(1).cpu().numpy()
        pred_zero[:, w] = (phi_now.unsqueeze(1) @ Wp).squeeze(1).cpu().numpy()

    seg_counts = np.stack([np.asarray(counts[fi, t0:t0 + W], dtype=np.int64)
                           for (fi, t0) in pts])
    r_proto = np.zeros((F_n, W), dtype=np.float32)
    r_zero = np.zeros((F_n, W), dtype=np.float32)
    for w in range(W):
        r_proto[:, w] = rate_from_logq_rows(pred_proto[:, w])
        r_zero[:, w] = rate_from_logq_rows(pred_zero[:, w])
    r_ewma = np.zeros((F_n, W), dtype=np.float32)
    ew = np.zeros(F_n)
    for w in range(W):
        r_ewma[:, w] = np.expm1(np.maximum(ew, 0))
        ew = 0.1 * np.log1p(seg_counts[:, w].astype(np.float64)) + 0.9 * ew

    cum = np.zeros((F_n, W), dtype=np.int64)
    cum[:, 1:] = np.cumsum(seg_counts[:, :-1], axis=1)
    age = np.tile(np.arange(W, dtype=np.int64), (F_n, 1))  # t0 = first arrival
    zero_h = cum == 0
    mid = (cum > 0) & (cum < GATE_THRESHOLD)
    conv_young = (cum >= GATE_THRESHOLD) & (age < AGE_MIN)
    r_v3 = np.where(zero_h, r_zero, np.where(mid, r_ewma, r_proto))
    r_aq = np.where(zero_h, r_zero,
           np.where(mid, r_ewma,
           np.where(conv_young, r_proto, r_ewma)))
    ident = bool(np.array_equal(r_v3, r_aq))
    print(f"aq == v3 on cohort rates: {ident} "
          f"(expected True: all ticks younger than {AGE_MIN} min)", flush=True)

    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"]
                                 for (fi, _) in pts]), nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"]
                                   for (fi, _) in pts]), nan=0.5)
    res = {"identity_with_v3": ident, "by_rho": {}}
    for rho in [1.0, 10.0, 100.0]:
        tau = newsvendor_quantile(rho)
        pw_aq, ka_aq = decisions_from_rates(r_aq, tau)
        pw_v3, ka_v3 = decisions_from_rates(r_v3, tau)
        assert np.array_equal(pw_aq, pw_v3) and np.array_equal(ka_aq, ka_v3)
        roll_c = np.zeros(W); roll_n = np.zeros(W)
        func_cold = np.zeros(F_n); func_wm = np.zeros(F_n)
        for f in range(F_n):
            for seed in DES_SEEDS:
                r = simulate_function(seg_counts[f], pw_aq[f], ka_aq[f],
                                      float(dm[f]), float(dstd[f]),
                                      seed=seed * 7919 + f,
                                      cold_mu=COLD_INIT["mu"],
                                      cold_sigma=COLD_INIT["sigma"],
                                      track_rolling=True)
                roll_c += r["roll_cold"]; roll_n += r["roll_total"]
                func_cold[f] += r["roll_cold"].sum()
                func_wm[f] += r["idle_mem_gb_s"]
        func_cold /= len(DES_SEEDS); func_wm /= len(DES_SEEDS)
        res["by_rho"][str(rho)] = {
            "overall_csr": float(roll_c.sum() / max(roll_n.sum(), 1e-9)),
            "func_cold_all": func_cold.tolist(),
            "wm_total": float(func_wm.sum()),
            "rolling_csr": rolling_csr(roll_c, roll_n, window=15).tolist()}
        print(f"  cohort aq rho={rho}: CSR="
              f"{res['by_rho'][str(rho)]['overall_csr']*100:.3f}%", flush=True)
    OUT["cohort_2019"] = res


def main():
    steady_2021()
    cohort_2019()
    path = PROJECT_ROOT / "results" / "runs" / "revision_e7_agegate.json"
    with open(path, "w") as f:
        json.dump(OUT, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(path))
    print(f"Saved + verified {path}")


if __name__ == "__main__":
    main()
