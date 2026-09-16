# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""M1: decision-level (CSR) test of prototype granularity, powered up.

Question this answers
--------------------
The manuscript's only evidence that the k=16 *cluster structure* (as opposed
to a single meta-learned prior) matters is a first-ten-minute CRPS ladder
(nearest 0.486 < global-pooled 0.523) plus the k-sweep in
revision_e6_kmeans_ood.json -- both *forecast* metrics.  The paper's own
central methodological claim (an LSTM with 6.6x better CRPS has a worse CSR)
says forecast metrics need not transfer to the decision.  And the archived
decision-level contrast (revision_a1_onboarding.json, n=33) cannot resolve
the k=16-vs-k=1 sub-component: delta = -0.95 cold starts/function at rho=10
with only 8/33 non-zero pairs, Wilcoxon p=0.069.

So: rerun the zero-history injection with (a) a decision-level k-sweep
k in {1, 4, 16, 64} plus a no-prototype arm, and (b) ~3.5x the paired
observations, obtained *without* leakage by adding the S2 cluster-hold-out
cohort and repeating it over the three archived S2 model seeds.

Leakage control
---------------
Every evaluated function is held out of BOTH body meta-training AND
prototype construction for its own campaign:
  S1     body best_anil_ridge_s1_s0, prototypes from s1_train(120),
         cohort s1_test  -> 33 functions.  Exact replication anchor: the
         k16 / k1 / noproto arms must reproduce the archived a1
         nearest / global / noproto numbers.
  S2_m{0,1,2}
         body best_anil_ridge_s2_s{0,1,2}, prototypes from s2_train(76),
         cohort s2_test -> 80 functions.  This is the *cluster* hold-out:
         the cohort's behaviour clusters were withheld from prototype
         construction, so it is the adversarial direction for k>1.  A win
         for k=16 here is therefore conservative; a loss is expected and
         uninformative on its own.
S1 and S2 cohorts overlap by construction, so the pooled test is computed
over unique function ids (S2 instance preferred, S1-only functions added).

Everything else is inherited unchanged from the registered protocol:
ONBOARD_WINDOW=240, DES_SEEDS=[0,1,2], RHOS={1,10,100}, the QL quantile
grid, rate_from_logq / decisions_from_rates / simulate_function, and the
per-tick nearest-prototype reassignment used by the archived 'nearest' arm.

Writes results/runs/revision_m1_kproto.json.  Touches no existing archive.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# CPU batched linalg on this VM is ~300x slower multi-threaded (MKL thrash).
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase63_onboarding_drift import (  # noqa: E402
    L, QL, ONBOARD_WINDOW, DES_SEEDS, DEVICE,
    PROCESSED_DIR, RUNS_DIR,
    build_prototypes, assign_proto_W, rate_from_logq,
    find_onboarding_points,
)
from scripts.phase6_des import decisions_from_rates, COLD_INIT  # noqa: E402
from src.sim.des import simulate_function, rolling_csr  # noqa: E402
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402

RHOS = [1.0, 10.0, 100.0]
KS = [1, 4, 16, 64]                       # decision-level granularity sweep
ARMS = ["noproto"] + [f"k{k}" for k in KS]
BOOT = 20000
BOOT_SEED = 0

CAMPAIGNS = [
    # tag,      checkpoint,                  train split, cohort split(s)
    ("S1",    "best_anil_ridge_s1_s0.pt",    "s1_train",  ("s1_test",)),
    ("S2_m0", "best_anil_ridge_s2_s0.pt",    "s2_train",  ("s2_test",)),
    ("S2_m1", "best_anil_ridge_s2_s1.pt",    "s2_train",  ("s2_test",)),
    ("S2_m2", "best_anil_ridge_s2_s2.pt",    "s2_train",  ("s2_test",)),
]

# --with-val: power-sensitivity campaign only. s1_val was used for the
# early-stopping decision, so it is NOT held out as cleanly as s1_test and
# this campaign is reported as a sensitivity analysis, never as the primary
# result. Appending it also switches the output path, so the default run
# stays byte-identical to the archived revision_m1_kproto.json.
VAL_CAMPAIGN = ("S1_val", "best_anil_ridge_s1_s0.pt", "s1_train",
                ("s1_test", "s1_val"))


# ----------------------------------------------------------------------
def paired_stats(a, b, rng):
    """Paired comparison of per-function cold-start counts (a - b)."""
    from scipy import stats
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    nz = int((d != 0).sum())
    p = float(stats.wilcoxon(a, b).pvalue) if nz else float("nan")
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "nonzero_pairs": nz, "n": int(len(d)), "wilcoxon_p": p}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.argsort(p[ok])
    vals = p[ok][idx]
    m = len(vals)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(vals)])
    adj = np.minimum(adj, 1.0)
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out.tolist()


def run_campaign(tag, ckpt, train_key, cohort_keys, features, counts, splits,
                 dur_df):
    t_start = time.time()
    print("=" * 66)
    print(f"CAMPAIGN {tag}: ckpt={ckpt} protos<-{train_key} "
          f"cohort<-{'+'.join(cohort_keys)}")
    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE,
    )
    trainer.load(RUNS_DIR / ckpt)
    train_idx = splits[train_key]
    # Preserve each split's stored order and de-duplicate by first appearance.
    # The split arrays are shuffled, not sorted, and simulate_function seeds
    # off the cohort *position* (seed*7919 + f), so sorting the union would
    # silently re-roll every function's DES noise and break replication.
    seen, order = set(), []
    for key in cohort_keys:
        for i in splits[key].tolist():
            if i not in seen:
                seen.add(i)
                order.append(i)
    cohort_idx = np.array(order, dtype=int)
    assert len(set(train_idx.tolist()) & set(cohort_idx.tolist())) == 0, \
        "cohort leaks into prototype-training set"

    pms = {}
    sizes = {}
    for k in KS:
        pms[k] = build_prototypes(trainer, features, counts, train_idx,
                                  n_clusters=k)
        lab = pms[k].kmeans.labels_
        sizes[k] = np.bincount(lab, minlength=k).tolist()

    pts = find_onboarding_points(counts, cohort_idx)
    F_n, W = len(pts), ONBOARD_WINDOW
    n_out = N_QUANTILES
    print(f"  cohort: {F_n}/{len(cohort_idx)} functions with onboarding points")

    # ---- embeddings phi[F, W, d] (identical to phase63 / a1) ----
    feats_t = torch.from_numpy(features).float()
    trainer.body.eval()
    phi_all = torch.zeros(F_n, W, 64)
    with torch.no_grad():
        for w in range(W):
            batch = []
            for (fi, t0) in pts:
                t = t0 + w
                if t < L:
                    pad = torch.zeros(L - t, features.shape[2])
                    win = torch.cat([pad, feats_t[fi, :t]], dim=0)
                else:
                    win = feats_t[fi, t - L:t]
                batch.append(win)
            x = torch.stack(batch).to(DEVICE)
            phi_all[:, w] = trainer.body(x).cpu()
    y_all = np.stack([np.log1p(counts[fi, t0:t0 + W]) for (fi, t0) in pts])
    y_t = torch.from_numpy(y_all).float()

    # ---- K=0 assignment collapse per k ----
    assign = {}
    for k in KS:
        _, ids0 = assign_proto_W(pms[k], phi_all[:, 0])
        ids0 = ids0.cpu().numpy()
        hist = np.bincount(ids0, minlength=k)
        pp = hist / hist.sum()
        H = float(-(pp[pp > 0] * np.log2(pp[pp > 0])).sum())
        stab = np.zeros(F_n)
        for w in range(min(60, W)):
            _, idw = assign_proto_W(pms[k], phi_all[:, w])
            stab += (idw.cpu().numpy() == ids0)
        stab /= min(60, W)
        assign[f"k{k}"] = {
            "hist": hist.tolist(), "top_share": float(hist.max() / hist.sum()),
            "entropy_bits": H, "max_entropy_bits": float(np.log2(k)),
            "effective_clusters": float(2 ** H),
            "stability_first_hour_mean": float(stab.mean()),
            "cluster_sizes_train": sizes[k],
        }
        print(f"  k={k:2d}: K=0 top-share {hist.max()/hist.sum():.2f} "
              f"entropy {H:.2f} bits -> eff {2**H:.2f} clusters")

    # ---- online prototype-biased ridge per arm (a1 semantics) ----
    lam = trainer.head.ridge_lambda.item()
    eye = torch.eye(64)
    preds = {a: np.zeros((F_n, W, n_out), dtype=np.float32) for a in ARMS}
    for arm in ARMS:
        k = None if arm == "noproto" else int(arm[1:])
        for w in range(W):
            phi_now = phi_all[:, w]
            if k is None:
                Wp = None
            else:                        # per-tick reassignment, as archived
                Wp, _ = assign_proto_W(pms[k], phi_now)
                Wp = Wp.cpu()
            if w == 0:
                W_head = Wp if Wp is not None else torch.zeros(F_n, 64, n_out)
            else:
                phi_s = phi_all[:, :w]
                y_s = y_t[:, :w].unsqueeze(-1).expand(F_n, w, n_out)
                PhiT = phi_s.transpose(1, 2)
                A = PhiT @ phi_s + lam * eye.unsqueeze(0)
                rhs = PhiT @ y_s + (lam * Wp if Wp is not None else 0.0)
                W_head = torch.linalg.solve(A, rhs)
            preds[arm][:, w] = (phi_now.unsqueeze(1) @ W_head).squeeze(1).numpy()
        print(f"  arm {arm}: online loop done")

    def crps_curve(pred):
        err = y_all[:, :, None] - pred
        return 2 * np.maximum(QL * err, (QL - 1) * err).mean(axis=(0, 2))

    crps = {a: crps_curve(preds[a]) for a in ARMS}

    # ---- DES per arm x rho (a1 machinery, same seeds) ----
    seg_counts = np.stack([counts[fi, t0:t0 + W] for (fi, t0) in pts])
    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] for (fi, _) in pts]),
                       nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] for (fi, _) in pts]),
                         nan=0.5)

    rates = {a: np.zeros((F_n, W), dtype=np.float32) for a in ARMS}
    for a in ARMS:
        for w in range(W):
            rates[a][:, w] = rate_from_logq(preds[a][:, w])
    ewma = np.zeros(F_n)
    rates["B4a_ewma"] = np.zeros((F_n, W), dtype=np.float32)
    for w in range(W):
        rates["B4a_ewma"][:, w] = np.expm1(np.maximum(ewma, 0))
        ewma = 0.1 * np.log1p(seg_counts[:, w].astype(np.float64)) + 0.9 * ewma
    rates["Oracle"] = seg_counts.astype(np.float32)
    pw_b1 = np.zeros((F_n, W), dtype=np.int32)
    for f in range(F_n):
        act = (seg_counts[f] > 0).astype(np.int32)
        conv = np.convolve(act, np.ones(10, dtype=np.int32))[:W]
        pw_b1[f, 1:] = (conv[:-1] > 0).astype(np.int32)
    b1_dec = (pw_b1, np.full((F_n, W), 10.0, np.float32))

    by_rho = {}
    for rho in RHOS:
        tau = newsvendor_quantile(rho)
        decisions = {m: decisions_from_rates(r, tau) for m, r in rates.items()}
        decisions["B1_fixed_keepalive"] = b1_dec
        rho_out = {}
        for m, (pw_m, ka_m) in decisions.items():
            roll_c = np.zeros(W)
            roll_n = np.zeros(W)
            func_cold60 = np.zeros(F_n)
            func_wm = np.zeros(F_n)
            for f in range(F_n):
                for seed in DES_SEEDS:
                    r = simulate_function(seg_counts[f], pw_m[f], ka_m[f],
                                          float(dm[f]), float(dstd[f]),
                                          seed=seed * 7919 + f,
                                          cold_mu=COLD_INIT["mu"],
                                          cold_sigma=COLD_INIT["sigma"],
                                          track_rolling=True)
                    roll_c += r["roll_cold"]
                    roll_n += r["roll_total"]
                    func_cold60[f] += r["roll_cold"][:60].sum()
                    func_wm[f] += r["idle_mem_gb_s"]
            func_cold60 /= len(DES_SEEDS)
            func_wm /= len(DES_SEEDS)
            rho_out[m] = {
                "overall_csr": float(roll_c.sum() / max(roll_n.sum(), 1e-9)),
                "func_cold60": func_cold60.tolist(),
                "wm_total": float(func_wm.sum()),
            }
        by_rho[str(rho)] = rho_out
        print(f"  rho={rho}: " + " ".join(
            f"{m}={v['overall_csr']*100:.3f}%" for m, v in rho_out.items()))

    print(f"  campaign {tag} done in {time.time()-t_start:.0f}s")
    return {
        "tag": tag, "checkpoint": ckpt, "train_split": train_key,
        "cohort_split": "+".join(cohort_keys), "n_functions": F_n,
        "func_ids": [int(fi) for (fi, _) in pts],
        "onboard_t": [int(t0) for (_, t0) in pts],
        "assignment": assign,
        "crps_t1_10": {a: crps[a][1:11].tolist() for a in ARMS},
        "crps_vs_time": {a: crps[a].tolist() for a in ARMS},
        "by_rho": by_rho,
    }


# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    features = np.load(PROCESSED_DIR / "features.npy")
    counts = np.load(PROCESSED_DIR / "counts.npy")
    splits = np.load(PROCESSED_DIR / "splits.npz")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")

    out = {
        "config": "M1 decision-level prototype-granularity sweep",
        "constants": {"ks": KS, "rhos": RHOS, "des_seeds": DES_SEEDS,
                      "onboard_window_min": ONBOARD_WINDOW,
                      "bootstrap_resamples": BOOT, "bootstrap_seed": BOOT_SEED,
                      "reassignment": "per-tick nearest centroid (as archived)"},
        "campaigns": {},
    }
    campaigns = list(CAMPAIGNS)
    with_val = "--with-val" in sys.argv
    if with_val:
        campaigns.append(VAL_CAMPAIGN)
        out["constants"]["note_val"] = (
            "S1_val cohort = s1_test + s1_val; s1_val informed early stopping, "
            "so this campaign is a power sensitivity, not a primary result")
    for tag, ckpt, tr, co in campaigns:
        out["campaigns"][tag] = run_campaign(tag, ckpt, tr, co, features,
                                             counts, splits, dur_df)

    # ---- paired contrasts, per campaign ----
    rng = np.random.default_rng(BOOT_SEED)
    contrasts = [("k1", "noproto", "prior_at_all"),
                 ("k16", "k1", "clustering_k16_vs_k1"),
                 ("k4", "k1", "clustering_k4_vs_k1"),
                 ("k64", "k1", "clustering_k64_vs_k1"),
                 ("k16", "k4", "granularity_16_vs_4"),
                 ("k64", "k16", "granularity_64_vs_16"),
                 ("k16", "noproto", "total_prototype_effect")]
    out["stats"] = {}
    for tag, camp in out["campaigns"].items():
        st = {}
        for rho in RHOS:
            rd = camp["by_rho"][str(rho)]
            row = {}
            for a, b, name in contrasts:
                row[name] = paired_stats(rd[a]["func_cold60"],
                                         rd[b]["func_cold60"], rng)
            st[str(rho)] = row
        # Holm across the three rhos, per contrast
        for _, _, name in contrasts:
            ps = [st[str(r)][name]["wilcoxon_p"] for r in RHOS]
            adj = holm(ps)
            for r, v in zip(RHOS, adj):
                st[str(r)][name]["holm_p_across_rho"] = v
        out["stats"][tag] = st

    # ---- pooled over unique functions (S2_m0 preferred, S1-only added) ----
    s1, s2 = out["campaigns"]["S1"], out["campaigns"]["S2_m0"]
    s2_ids = set(s2["func_ids"])
    s1_only = [i for i, fi in enumerate(s1["func_ids"]) if fi not in s2_ids]
    pooled = {"n_s2": len(s2["func_ids"]), "n_s1_only": len(s1_only),
              "n_total": len(s2["func_ids"]) + len(s1_only),
              "note": "S1 functions already in the S2 cohort are taken from "
                      "the S2 campaign only, so no function is counted twice",
              "by_rho": {}}
    for rho in RHOS:
        r1, r2 = s1["by_rho"][str(rho)], s2["by_rho"][str(rho)]
        row = {}
        for a, b, name in contrasts:
            va = np.concatenate([np.array(r2[a]["func_cold60"]),
                                 np.array(r1[a]["func_cold60"])[s1_only]])
            vb = np.concatenate([np.array(r2[b]["func_cold60"]),
                                 np.array(r1[b]["func_cold60"])[s1_only]])
            row[name] = paired_stats(va, vb, rng)
        pooled["by_rho"][str(rho)] = row
    for _, _, name in contrasts:
        ps = [pooled["by_rho"][str(r)][name]["wilcoxon_p"] for r in RHOS]
        for r, v in zip(RHOS, holm(ps)):
            pooled["by_rho"][str(r)][name]["holm_p_across_rho"] = v
    out["pooled_unique"] = pooled

    # ---- replication check vs archived a1 (S1 / k16 == 'nearest') ----
    a1p = RUNS_DIR / "revision_a1_onboarding.json"
    if a1p.exists():
        a1 = json.load(open(a1p))
        rep = {}
        for rho in RHOS:
            new = out["campaigns"]["S1"]["by_rho"][str(rho)]
            old = a1["by_rho"][str(rho)]
            rep[str(rho)] = {
                "k16_vs_archived_nearest_csr": [
                    new["k16"]["overall_csr"], old["nearest"]["overall_csr"]],
                "k1_vs_archived_global_csr": [
                    new["k1"]["overall_csr"], old["global"]["overall_csr"]],
                "noproto_vs_archived_noproto_csr": [
                    new["noproto"]["overall_csr"], old["noproto"]["overall_csr"]],
            }
        out["replication_check"] = rep

    out["wall_sec"] = time.time() - t0
    path = RUNS_DIR / ("revision_m1_kproto_v2.json" if with_val
                       else "revision_m1_kproto.json")
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0, "corrupt write (sdb guard)"
    json.load(open(path))
    print(f"\nSaved + verified {path}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
