# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""M1: prototype granularity on the HEADLINE cohort (Azure 2019 deployments).

The manuscript's headline result -- the learned path cutting cold starts
32-46% below a per-function EWMA on 100 real new deployments -- is the one
place where the prototype mechanism named in the title is supposed to earn
its keep, and it is the one cohort that was never run with a control for
prototype granularity. revision_a4_crosstrace.json has A5_proto (k=16),
A5_noproto (no prior) and A5_proto_zero, but no single-template arm, so
"one of 16 learned templates" has never been tested against "one template".

Everything measured since says it should not matter: assignment collapses
to 79/100 functions on one cluster here, to 76/76 on Huawei and to 28/33 on
the 2021 injection, and k=16 has never significantly beaten k=1 at the
decision level on any cohort (it is significantly WORSE on the S2 cluster
hold-out). This run closes the gap on the cohort that carries the headline.

It also tests a variant the narrative implies but the code never built. The
registered builder caps a prototype's support at PROTO_FUNC_CAP=50
functions, so even the k=1 arm is a template fitted on 50 training
functions rather than on all of them; the "1_all" arm lifts that cap.

Protocol
--------
Identical to revision_a4_crosstrace.py in every registered respect
(cohort funnel, W=240, DES_SEEDS, RHOS, TAU semantics, gate threshold,
imputed duration model, 2021-trained body + 2021 s2_train prototypes),
with a single change: eval_adapt_biased.N_CLUSTERS is monkeypatched over
k in {1, 4, 16, 64} and one prototype-biased arm is built per k.  The
monkeypatch pattern is the one revision_e6_kmeans_ood.py already uses.

The k=16 arms are a replication anchor: they must reproduce the archived
A5_proto / A5_proto_zero / A5_noproto numbers exactly.

Writes results/runs/revision_m1_kproto_2019.json.  Touches no archive.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# MKL batched linalg on this VM thrashes threads; GPU does the heavy lifting.
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("SF_DATA_DIR", "processed_2019")
os.environ["SF_RUNS_DIR"] = str(PROJECT_ROOT / "results_azure2021" / "runs")

import scripts.eval_adapt_biased as eab                        # noqa: E402
from scripts.eval_adapt_biased import (                        # noqa: E402
    load_trainer, PROCESSED_DIR, RUNS_DIR,
)
from scripts.phase63_onboarding_drift import DES_SEEDS, DEVICE  # noqa: E402
from scripts.phase6_des import decisions_from_rates, COLD_INIT  # noqa: E402
from scripts.revision_a4_crosstrace import (                # noqa: E402
    find_natural_cohort, rate_from_logq_rows, PROC_2021,
    L, W, RHOS, GATE_THRESHOLD, QL,
)
from scripts.revision_m1_kproto import paired_stats, holm      # noqa: E402
from src.sim.des import simulate_function, rolling_csr         # noqa: E402
from src.decision.newsvendor import newsvendor_quantile        # noqa: E402
from src.models.heads import N_QUANTILES                       # noqa: E402

KS = [1, 4, 16, 64]
# arm keys: the k sweep plus the uncapped single-template variant
KEYS = [str(k) for k in KS] + ["1_all"]
NCLUST = {**{str(k): k for k in KS}, "1_all": 1}
ARMS_PRED = (["noproto"] + [f"k{key}" for key in KEYS]
             + [f"k{key}_zero" for key in KEYS])
BOOT_SEED = 0


def main():
    t_start = time.time()
    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    nF = features.shape[2]

    feat21 = np.load(PROC_2021 / "features.npy", mmap_mode="r")
    cnt21 = np.load(PROC_2021 / "counts.npy", mmap_mode="r")
    spl21 = np.load(PROC_2021 / "splits.npz")
    assert feat21.shape[2] == nF, "feature dim mismatch between traces"

    pts = find_natural_cohort(counts)          # a4 signature: counts only
    funnel = {"cohort_size": len(pts)}
    F_n = len(pts)
    print(f"natural cohort: {F_n} functions")

    trainer, biased_head = load_trainer(nF, seed=0)   # 2021 checkpoint

    # ---- one prototype set per k (monkeypatch the module global) ----
    protos = {}
    k_orig, cap_orig = eab.N_CLUSTERS, eab.PROTO_FUNC_CAP
    try:
        for k in KS:
            eab.N_CLUSTERS = k
            cen, pw = eab.build_prototypes(trainer, biased_head,
                                           feat21, cnt21, spl21["s2_train"])
            protos[str(k)] = (cen.to(DEVICE), pw.to(DEVICE))
            print(f"  built k={k}: centroids {tuple(cen.shape)} "
                  f"weights {tuple(pw.shape)}")
        # "one template from EVERY training function": the registered builder
        # caps a prototype's support at PROTO_FUNC_CAP=50 functions, so the
        # k=1 arm above is a single template fitted on 50 of the training
        # functions, not on all of them. Lift the cap to test the version the
        # narrative actually implies.
        eab.N_CLUSTERS = 1
        eab.PROTO_FUNC_CAP = 10 ** 9
        cen, pw = eab.build_prototypes(trainer, biased_head,
                                       feat21, cnt21, spl21["s2_train"])
        protos["1_all"] = (cen.to(DEVICE), pw.to(DEVICE))
        print(f"  built k=1 uncapped: support = all "
              f"{len(spl21['s2_train'])} train functions")
    finally:
        eab.N_CLUSTERS, eab.PROTO_FUNC_CAP = k_orig, cap_orig

    # ---- embeddings (identical to h1) ----
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
            if (w + 1) % 120 == 0:
                print(f"  embed {w+1}/{W}", flush=True)
    y_all = np.stack([np.log1p(np.asarray(counts[fi, t0:t0 + W], dtype=np.float64))
                      for (fi, t0) in pts])
    y_t = torch.from_numpy(y_all).float()

    # ---- online adaptation, all arms, GPU algebra (h1 code path) ----
    import torch.nn.functional as Fun
    lam = trainer.head.ridge_lambda.item()
    n_out = N_QUANTILES
    preds = {a: np.zeros((F_n, W, n_out), dtype=np.float32) for a in ARMS_PRED}
    assign0 = {}
    phi_g, y_g = phi_all.to(DEVICE), y_t.to(DEVICE)
    eye_g = torch.eye(64, device=DEVICE)
    for w in range(W):
        phi_now = phi_g[:, w]
        if w > 0:
            phi_s = phi_g[:, :w]
            y_s = y_g[:, :w].unsqueeze(-1).expand(F_n, w, n_out)
            PhiT = phi_s.transpose(1, 2)
            A = PhiT @ phi_s + lam * eye_g.unsqueeze(0)
            PhiTY = PhiT @ y_s
            W_n = torch.linalg.solve(A, PhiTY)
        else:
            W_n = torch.zeros(F_n, 64, n_out, device=DEVICE)
        preds["noproto"][:, w] = (phi_now.unsqueeze(1) @ W_n).squeeze(1).cpu().numpy()
        for key in KEYS:
            cen_g, pw_g = protos[key]
            sims = Fun.cosine_similarity(phi_now.unsqueeze(1),
                                         cen_g.unsqueeze(0), dim=2)
            ids = sims.argmax(dim=1)
            if w == 0:
                assign0[key] = ids.cpu().numpy().copy()
            Wp = pw_g[ids]
            W_p = Wp.clone() if w == 0 else torch.linalg.solve(A, PhiTY + lam * Wp)
            preds[f"k{key}"][:, w] = (phi_now.unsqueeze(1) @ W_p).squeeze(1).cpu().numpy()
            preds[f"k{key}_zero"][:, w] = (phi_now.unsqueeze(1) @ Wp).squeeze(1).cpu().numpy()
        if (w + 1) % 120 == 0:
            print(f"  adapt {w+1}/{W}", flush=True)
    print("ridge/pred stage done", flush=True)

    # ---- K=0 assignment structure per k ----
    assign_out = {}
    for key in KEYS:
        k = NCLUST[key]
        hist = np.bincount(assign0[key], minlength=k)
        p = hist / hist.sum()
        H = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        assign_out[f"k{key}"] = {
            "hist": hist.tolist(), "top_share": float(hist.max() / hist.sum()),
            "entropy_bits": H, "max_entropy_bits": float(np.log2(k)),
            "effective_clusters": float(2 ** H)}
        print(f"  {key:>5s}: top-share {hist.max()/hist.sum():.2f} "
              f"entropy {H:.2f} bits -> eff {2**H:.2f} clusters")

    def crps_curve(pred):
        err = y_all[:, :, None] - pred
        return 2 * np.maximum(QL * err, (QL - 1) * err).mean(axis=(0, 2))

    # ---- rates, baselines, gate (identical to h1) ----
    seg_counts = np.stack([np.asarray(counts[fi, t0:t0 + W], dtype=np.int64)
                           for (fi, t0) in pts])
    rates = {a: np.zeros((F_n, W), dtype=np.float32) for a in ARMS_PRED}
    for a in ARMS_PRED:
        for w in range(W):
            rates[a][:, w] = rate_from_logq_rows(preds[a][:, w])
    ewma = np.zeros(F_n)
    rates["B4a_ewma"] = np.zeros((F_n, W), dtype=np.float32)
    for w in range(W):
        rates["B4a_ewma"][:, w] = np.expm1(np.maximum(ewma, 0))
        ewma = 0.1 * np.log1p(seg_counts[:, w].astype(np.float64)) + 0.9 * ewma
    rates["Oracle"] = seg_counts.astype(np.float32)

    cum = np.zeros((F_n, W), dtype=np.int64)
    cum[:, 1:] = np.cumsum(seg_counts[:, :-1], axis=1)
    zero_h, mid = cum == 0, (cum > 0) & (cum < GATE_THRESHOLD)
    for key in KEYS:                   # gate v3 per granularity
        rates[f"gated_v3_k{key}"] = np.where(
            zero_h, rates[f"k{key}_zero"],
            np.where(mid, rates["B4a_ewma"], rates[f"k{key}"]))

    pw_b1 = np.zeros((F_n, W), dtype=np.int32)
    for f in range(F_n):
        act = (seg_counts[f] > 0).astype(np.int32)
        conv1 = np.convolve(act, np.ones(10, dtype=np.int32))[:W]
        pw_b1[f, 1:] = (conv1[:-1] > 0).astype(np.int32)
    b1_dec = (pw_b1, np.full((F_n, W), 10.0, np.float32))

    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] for (fi, _) in pts]), nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] for (fi, _) in pts]), nan=0.5)

    out = {"config": "M1/E1 Huawei prototype-granularity sweep",
           "trace": "azure_2019",
           "checkpoint": "results_azure2021/runs/best_anil_ridge_s2_s0.pt",
           "proto_source": "2021 s2_train",
           "constants": {"ks": KS, "rhos": RHOS, "des_seeds": DES_SEEDS,
                         "window_min": W, "gate_threshold": GATE_THRESHOLD},
           "cohort_funnel": funnel, "n_functions": F_n,
           "func_ids": [int(fi) for (fi, _) in pts],
           "assignment": assign_out,
           "crps_t1_10": {a: crps_curve(preds[a])[1:11].tolist() for a in ARMS_PRED},
           "by_rho": {}}

    for rho in RHOS:
        tau = newsvendor_quantile(rho)
        dec = {m: decisions_from_rates(r, tau) for m, r in rates.items()}
        dec["B1_fixed_keepalive"] = b1_dec
        rho_out = {}
        for m, (pw_m, ka_m) in dec.items():
            roll_c = np.zeros(W); roll_n = np.zeros(W)
            f_c60 = np.zeros(F_n); f_call = np.zeros(F_n); f_wm = np.zeros(F_n)
            for f in range(F_n):
                for seed in DES_SEEDS:
                    r = simulate_function(seg_counts[f], pw_m[f], ka_m[f],
                                          float(dm[f]), float(dstd[f]),
                                          seed=seed * 7919 + f,
                                          cold_mu=COLD_INIT["mu"],
                                          cold_sigma=COLD_INIT["sigma"],
                                          track_rolling=True)
                    roll_c += r["roll_cold"]; roll_n += r["roll_total"]
                    f_c60[f] += r["roll_cold"][:60].sum()
                    f_call[f] += r["roll_cold"].sum()
                    f_wm[f] += r["idle_mem_gb_s"]
            f_c60 /= len(DES_SEEDS); f_call /= len(DES_SEEDS); f_wm /= len(DES_SEEDS)
            rho_out[m] = {"overall_csr": float(roll_c.sum() / max(roll_n.sum(), 1e-9)),
                          "func_cold60": f_c60.tolist(),
                          "func_cold_all": f_call.tolist(),
                          "wm_total": float(f_wm.sum()),
                          "rolling_csr": rolling_csr(roll_c, roll_n, window=15).tolist()}
        out["by_rho"][str(rho)] = rho_out
        print(f"rho={rho}: " + " ".join(
            f"{m}={v['overall_csr']*100:.3f}%" for m, v in rho_out.items()), flush=True)

    # ---- paired contrasts (h1 convention: func_cold_all over the 4 h window) ----
    rng = np.random.default_rng(BOOT_SEED)
    contrasts = [("k1", "noproto", "prior_at_all"),
                 ("k1_all", "noproto", "prior_at_all_uncapped"),
                 ("k1_all", "k1", "uncapped_vs_capped_template"),
                 ("k16", "k1_all", "clustering_vs_single_template"),
                 ("k4", "k1", "clustering_k4_vs_k1"),
                 ("k16", "k1", "clustering_k16_vs_k1"),
                 ("k64", "k1", "clustering_k64_vs_k1"),
                 ("k16", "k4", "granularity_16_vs_4"),
                 ("k16", "noproto", "total_prototype_effect"),
                 ("k4", "noproto", "total_prototype_effect_k4"),
                 ("k4", "B4a_ewma", "k4_vs_ewma"),
                 ("k16", "B4a_ewma", "k16_vs_ewma")]
    stats_out = {}
    for rho in RHOS:
        rd = out["by_rho"][str(rho)]
        stats_out[str(rho)] = {
            name: paired_stats(rd[a]["func_cold_all"], rd[b]["func_cold_all"], rng)
            for a, b, name in contrasts}
    for _, _, name in contrasts:
        ps = [stats_out[str(r)][name]["wilcoxon_p"] for r in RHOS]
        for r, v in zip(RHOS, holm(ps)):
            stats_out[str(r)][name]["holm_p_across_rho"] = v
    out["stats"] = stats_out

    # ---- replication anchor vs archived h1 ----
    a_path = RUNS_DIR / "revision_a4_crosstrace.json"
    if a_path.exists():
        arch = json.load(open(a_path))
        rep = {}
        for rho in RHOS:
            new, old = out["by_rho"][str(rho)], arch["by_rho"][str(rho)]
            rep[str(rho)] = {
                "k16_vs_A5_proto": [new["k16"]["overall_csr"],
                                    old["A5_proto"]["overall_csr"]],
                "k16_zero_vs_A5_proto_zero": [new["k16_zero"]["overall_csr"],
                                              old["A5_proto_zero"]["overall_csr"]],
                "noproto_vs_A5_noproto": [new["noproto"]["overall_csr"],
                                          old["A5_noproto"]["overall_csr"]],
                "ewma": [new["B4a_ewma"]["overall_csr"],
                         old["B4a_ewma"]["overall_csr"]],
            }
        out["replication_check"] = rep
        out["archived_assign_hist_k0"] = arch["assign_hist_k0"]

    out["wall_sec"] = time.time() - t_start
    path = RUNS_DIR / "revision_m1_kproto_2019.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0, "corrupt write (sdb guard)"
    json.load(open(path))
    print(f"\nSaved + verified {path} ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
