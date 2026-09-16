# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP3+WP4: Onboarding and drift evaluated as operational metrics (CSR over
time), with prototype-initialized biased ridge and measured drift triggers.

Part A: Build prototypes from meta-train functions (PrototypeManager).
Part B: Onboarding — each test function enters with zero history at its first
        sustained-activity tick; methods decide prewarm/keepalive online;
        the event-level DES measures rolling CSR over the first 240 min.
Part C: Drift — synthetic drift injected mid-trace; A5 with trigger-driven
        refit vs frozen head vs EWMA vs B1; rolling CSR around the event;
        trigger detection-delay / false-alarm benchmark (T5).
"""

import os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.des import simulate_function, rolling_csr
from src.sim.simulator import compute_adaptation_lag
from src.data.features import features_from_counts
from src.decision.newsvendor import newsvendor_quantile
from src.drift.detector import inject_synthetic_drift
from src.drift.triggers import build_triggers
from src.models.heads import N_QUANTILES, QUANTILES
from src.models.prototypes import PrototypeManager
from src.meta.trainer import ANILMetaTrainer
from scripts.phase6_des import decisions_from_rates, COLD_INIT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

L = 60
RHO_MAIN = 10.0
RHOS_EVAL = [1.0, 10.0]   # keepalive floor at rho=10 compresses differences;
                          # evaluate both operating points
TAU = newsvendor_quantile(RHO_MAIN)
QL = np.linspace(0.05, 0.95, N_QUANTILES)
ONBOARD_WINDOW = 240      # minutes evaluated after onboarding
DRIFT_WINDOW = 240        # minutes each side of the drift point
DES_SEEDS = [0, 1, 2]


# ====================================================================
# Part A: prototypes
# ====================================================================
def build_prototypes(trainer, features, counts, train_idx, n_clusters=16):
    print(f"Building {n_clusters} prototypes from {len(train_idx)} train functions...")
    pm = PrototypeManager(n_clusters=n_clusters, device=DEVICE)
    feats_t = torch.from_numpy(features).float()
    counts_t = torch.from_numpy(counts).float()
    emb = pm.compute_embeddings(trainer.body, feats_t, train_idx)
    labels = pm.fit_clusters(emb)
    pm.compute_prototype_heads(trainer.body, trainer.head, feats_t, counts_t,
                               train_idx, labels,
                               n_quantiles=N_QUANTILES, n_horizons=1)
    sizes = np.bincount(labels, minlength=n_clusters)
    print(f"  cluster sizes: {sizes.tolist()}")
    return pm


def assign_proto_W(pm, phi_batch):
    """Nearest-prototype head for a batch of embeddings [F, d] -> [F, d, O]."""
    cen = pm.centroids.to(phi_batch.device)          # [K, d]
    pw = pm.proto_weights.to(phi_batch.device)       # [K, d, O]
    sims = torch.nn.functional.cosine_similarity(
        phi_batch.unsqueeze(1), cen.unsqueeze(0), dim=2)  # [F, K]
    idx = sims.argmax(dim=1)
    return pw[idx], idx


def rate_from_logq(pred_q):
    """[.., Q] log1p-space quantiles -> rate at TAU (numpy)."""
    val = np.array([np.interp(TAU, QL, row) for row in np.atleast_2d(pred_q)])
    return np.expm1(np.maximum(val, 0.0))


# ====================================================================
# Part B: onboarding
# ====================================================================
def find_onboarding_points(counts, test_idx):
    """First tick with activity and >=30 invocations in the following 24h."""
    chosen = []
    T = counts.shape[1]
    for fi in test_idx:
        nz = np.nonzero(counts[fi])[0]
        onboard_t = None
        for t in nz:
            if t + ONBOARD_WINDOW >= T:
                break
            if counts[fi, t:t + 1440].sum() >= 30:
                onboard_t = int(t)
                break
        if onboard_t is not None:
            chosen.append((int(fi), onboard_t))
    return chosen


def onboarding_experiment(trainer, pm, features, counts, test_idx):
    print("\n" + "=" * 60)
    print("PART B: ONBOARDING")
    print("=" * 60)
    pts = find_onboarding_points(counts, test_idx)
    print(f"  {len(pts)}/{len(test_idx)} functions with valid onboarding points")
    F_n = len(pts)
    W = ONBOARD_WINDOW

    feats_t = torch.from_numpy(features).float()
    n_out = N_QUANTILES

    # ---- Batched embeddings phi[F, W, d] (windows may cross onboard_t;
    # pre-onboarding counts are zero by construction of the first-activity
    # onboarding point, so features encode "idle history" = standard proxy) ----
    print("  Embedding onboarding windows...")
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

    # Targets y[F, W] = log1p(count at onboard_t + w)
    y_all = np.stack([np.log1p(counts[fi, t0:t0 + W]) for (fi, t0) in pts])
    y_t = torch.from_numpy(y_all).float()

    lam = trainer.head.ridge_lambda.item()
    eye = torch.eye(64)

    # ---- Online prediction loops (batched over functions per tick) ----
    rates = {m: np.zeros((F_n, W), dtype=np.float32)
             for m in ["A5_proto", "A5_noproto"]}
    crps_w = {m: np.full(W, np.nan) for m in ["A5_proto", "A5_noproto"]}

    print("  Online adaptation loop...")
    for w in range(W):
        phi_now = phi_all[:, w]                       # [F, d]
        W_proto, _ = assign_proto_W(pm, phi_now)      # [F, d, O] (cpu ok)
        W_proto = W_proto.cpu()

        if w == 0:
            W_np = W_proto                             # zero-shot: prototype
            W_nn = torch.zeros(F_n, 64, n_out)
        else:
            phi_s = phi_all[:, :w]                     # [F, w, d]
            y_s = y_t[:, :w].unsqueeze(-1).expand(F_n, w, n_out)
            PhiT = phi_s.transpose(1, 2)
            A = PhiT @ phi_s + lam * eye.unsqueeze(0)
            PhiTY = PhiT @ y_s
            # biased ridge: (PhiTPhi + lam I)^-1 (PhiTY + lam W_proto)
            W_np = torch.linalg.solve(A, PhiTY + lam * W_proto)
            W_nn = torch.linalg.solve(A, PhiTY)

        pred_p = (phi_now.unsqueeze(1) @ W_np).squeeze(1).numpy()  # [F, O]
        pred_n = (phi_now.unsqueeze(1) @ W_nn).squeeze(1).numpy()

        rates["A5_proto"][:, w] = rate_from_logq(pred_p)
        rates["A5_noproto"][:, w] = rate_from_logq(pred_n)

        # CRPS (2x mean pinball) against realized target
        for name, pred in [("A5_proto", pred_p), ("A5_noproto", pred_n)]:
            err = y_all[:, w:w + 1] - pred             # [F, Q]
            pb = np.maximum(QL * err, (QL - 1) * err)
            crps_w[name][w] = 2 * pb.mean()

    # ---- Baseline decision matrices on the onboarding window ----
    seg_counts = np.stack([counts[fi, t0:t0 + W] for (fi, t0) in pts])

    ewma = np.zeros(F_n)
    rates["B4a_ewma"] = np.zeros((F_n, W), dtype=np.float32)
    for w in range(W):
        rates["B4a_ewma"][:, w] = np.expm1(np.maximum(ewma, 0))
        ewma = 0.1 * np.log1p(seg_counts[:, w].astype(np.float64)) + 0.9 * ewma

    rates["Oracle"] = seg_counts.astype(np.float32)

    # B1 policy within the onboarding window (no pre-onboarding knowledge)
    pw_b1 = np.zeros((F_n, W), dtype=np.int32)
    for f in range(F_n):
        act = (seg_counts[f] > 0).astype(np.int32)
        conv = np.convolve(act, np.ones(10, dtype=np.int32))[:W]
        pw_b1[f, 1:] = (conv[:-1] > 0).astype(np.int32)
    b1_decisions = (pw_b1, np.full((F_n, W), 10.0, np.float32))

    # ---- DES on the onboarding segment (per rho operating point) ----
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] for (fi, _) in pts]), nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] for (fi, _) in pts]), nan=0.5)

    print("  DES on onboarding segments...")
    out = {"by_rho": {}, "n_functions": F_n, "window": W,
           "crps_vs_time": {k: v.tolist() for k, v in crps_w.items()}}

    for rho in RHOS_EVAL:
        tau = newsvendor_quantile(rho)
        decisions = {m: decisions_from_rates(rmat, tau)
                     for m, rmat in rates.items()}
        decisions["B1_fixed_keepalive"] = b1_decisions

        rho_out = {}
        for m, (pw, ka) in decisions.items():
            roll_c = np.zeros(W); roll_n = np.zeros(W)
            func_cold60 = np.zeros(F_n)   # per-function cold in first 60 min
            func_wm = np.zeros(F_n)
            al_list = []
            for f in range(F_n):
                for seed in DES_SEEDS:
                    r = simulate_function(seg_counts[f], pw[f], ka[f],
                                          float(dm[f]), float(dstd[f]),
                                          seed=seed * 7919 + f,
                                          cold_mu=COLD_INIT["mu"],
                                          cold_sigma=COLD_INIT["sigma"],
                                          track_rolling=True)
                    roll_c += r["roll_cold"]; roll_n += r["roll_total"]
                    func_cold60[f] += r["roll_cold"][:60].sum()
                    func_wm[f] += r["idle_mem_gb_s"]
                rc = rolling_csr(r["roll_cold"], r["roll_total"], window=15)
                pairs = [(t, c) for t, c in enumerate(rc) if not np.isnan(c)]
                al = compute_adaptation_lag(pairs, event_tick=0)
                if al is not None:
                    al_list.append(float(al))
            func_cold60 /= len(DES_SEEDS)
            func_wm /= len(DES_SEEDS)
            curve = rolling_csr(roll_c, roll_n, window=15)
            cum_curve = np.cumsum(roll_c) / max(1e-9, len(DES_SEEDS))
            ckpts = {str(ck): float(cum_curve[min(ck, W) - 1])
                     for ck in [15, 30, 60, 120, 240]}
            rho_out[m] = {
                "rolling_csr": [None if np.isnan(x) else float(x) for x in curve],
                "cumulative_cold": cum_curve.tolist(),
                "cum_cold_at": ckpts,
                "overall_csr": float(roll_c.sum() / max(1, roll_n.sum())),
                "wm_total": float(func_wm.sum()),
                "func_cold60": func_cold60.tolist(),
                "func_wm": func_wm.tolist(),
                "al_median": float(np.median(al_list)) if al_list else None,
                "al_n": len(al_list),
            }
            print(f"    rho={rho:5.1f} {m:<16s} CSR={rho_out[m]['overall_csr']:.4f} "
                  f"cold@60m={ckpts['60']:.0f} WM={rho_out[m]['wm_total']:.0f} "
                  f"AL_med={rho_out[m]['al_median']}")

        # Paired tests on per-function cold@60 (A5_proto vs baselines)
        from scipy import stats as sps
        paired = {}
        ours60 = np.array(rho_out["A5_proto"]["func_cold60"])
        for b in ["B1_fixed_keepalive", "B4a_ewma", "A5_noproto"]:
            base60 = np.array(rho_out[b]["func_cold60"])
            diff = base60 - ours60
            nz = diff[diff != 0]
            p = float(sps.wilcoxon(nz)[1]) if len(nz) >= 6 else 1.0
            paired[b] = {"wilcoxon_p": p, "mean_diff": float(diff.mean()),
                         "n": int(len(diff))}
            print(f"      paired cold@60 A5_proto vs {b}: "
                  f"meanDiff={diff.mean():+.2f} p={p:.4f}")
        rho_out["_paired_cold60"] = paired
        out["by_rho"][str(rho)] = rho_out
    return out


# ====================================================================
# Part C: drift + trigger benchmark
# ====================================================================
def a5_online_rates(trainer, feats_seg, counts_seg, t_offset_rel,
                    refit_every=10, buffer_len=120, frozen_after=None,
                    trigger=None):
    """Online A5 rate prediction over a segment for ONE function.

    Args:
        feats_seg: [Tseg, C] features for the segment (already drift-consistent)
        counts_seg: [Tseg] counts
        frozen_after: if set, stop refitting after this relative tick
        trigger: optional drift trigger; on fire, buffer shrinks to last 30

    Returns:
        rates [Tseg], fire_ticks (list)
    """
    Tseg = len(counts_seg)
    n_out = N_QUANTILES
    lam = trainer.head.ridge_lambda.item()
    feats_t = torch.from_numpy(feats_seg).float()

    # Batch-embed all windows
    wins = []
    for t in range(Tseg):
        if t < L:
            pad = torch.zeros(L - t, feats_seg.shape[1])
            wins.append(torch.cat([pad, feats_t[:t]], dim=0))
        else:
            wins.append(feats_t[t - L:t])
    with torch.no_grad():
        phi = []
        X = torch.stack(wins)
        for s in range(0, Tseg, 512):
            phi.append(trainer.body(X[s:s + 512].to(DEVICE)).cpu().numpy())
    phi = np.concatenate(phi)                        # [Tseg, 64]

    y = np.log1p(counts_seg.astype(np.float64))
    rates = np.zeros(Tseg, dtype=np.float32)
    fires = []
    Wh = None
    buf_start = 0

    for t in range(Tseg):
        if Wh is not None:
            pred_q = phi[t] @ Wh                     # [Q]
            rates[t] = rate_from_logq(pred_q[None, :])[0]
            if trigger is not None:
                resid = abs(y[t] - float(np.median(pred_q)))
                if trigger.update(resid):
                    fires.append(t)
                    buf_start = max(buf_start, t - 30)
        # (re)fit
        allow_fit = frozen_after is None or t <= frozen_after
        if allow_fit and t >= 20 and (t % refit_every == 0 or
                                      (fires and fires[-1] == t)):
            s0 = max(buf_start, t - buffer_len)
            Xb = phi[s0:t]
            yb = np.repeat(y[s0:t, None], n_out, axis=1)
            try:
                Wh = np.linalg.solve(Xb.T @ Xb + lam * np.eye(64), Xb.T @ yb)
            except np.linalg.LinAlgError:
                pass
    return rates, fires


def drift_experiment(trainer, features, counts, test_idx, train_idx):
    print("\n" + "=" * 60)
    print("PART C: DRIFT + TRIGGERS")
    print("=" * 60)
    T = counts.shape[1]
    t_d = T // 2
    seg_lo, seg_hi = t_d - DRIFT_WINDOW, t_d + DRIFT_WINDOW
    rel_td = DRIFT_WINDOW
    Tseg = seg_hi - seg_lo

    active = [fi for fi in test_idx
              if counts[fi, seg_lo:t_d].sum() >= 100
              and counts[fi, t_d:seg_hi].sum() >= 100][:12]
    print(f"  {len(active)} active functions for drift injection")

    # Splice donor: busiest train function with a distinct pattern
    donor = int(train_idx[np.argmax(counts[train_idx].sum(axis=1))])
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    # scale(2.0) is a no-harm control; scale(0.5)/phase/splice stress
    # stale predictions in different ways
    drift_cfgs = [("scale", 2.0), ("scale", 0.5), ("phase", 120),
                  ("splice", donor)]
    out = {"drift_curves": {}, "trigger_bench": []}
    trig_rows = {}

    for dtype, param in drift_cfgs:
        key = f"{dtype}({param})" if dtype != "splice" else "splice(donor)"
        print(f"  --- drift {key} ---")
        # agg[(rho, method)] = [roll_cold, roll_total]
        agg = {}
        for fi in active:
            drifted, _ = inject_synthetic_drift(counts, fi, t_d, dtype, param)
            seg_counts = drifted[fi, seg_lo:seg_hi].astype(np.int64)
            feats_full = features_from_counts(
                drifted[fi], t_offset=0, app_corate=features[fi, :, 10])
            feats_seg = feats_full[seg_lo:seg_hi]

            dm = float(np.nan_to_num(dur_df.iloc[fi]["dur_mean"], nan=1.0))
            dstd = float(np.nan_to_num(dur_df.iloc[fi]["dur_std"], nan=0.5))

            variants = {}
            from src.drift.triggers import ConformalMonitor
            r_refit, fires = a5_online_rates(
                trainer, feats_seg, seg_counts, seg_lo,
                trigger=ConformalMonitor())
            variants["A5_refit"] = r_refit
            r_frozen, _ = a5_online_rates(
                trainer, feats_seg, seg_counts, seg_lo,
                frozen_after=rel_td - 1)
            variants["A5_frozen"] = r_frozen

            ew = np.zeros(Tseg, dtype=np.float32)
            st = 0.0
            for t in range(Tseg):
                ew[t] = np.expm1(max(st, 0))
                st = 0.1 * np.log1p(seg_counts[t]) + 0.9 * st
            variants["B4a_ewma"] = ew
            variants["Oracle"] = seg_counts.astype(np.float32)

            # B1 policy (rho-independent)
            act = (seg_counts > 0).astype(np.int32)
            conv = np.convolve(act, np.ones(10, dtype=np.int32))[:Tseg]
            pw_b1 = np.zeros(Tseg, dtype=np.int32)
            pw_b1[1:] = (conv[:-1] > 0).astype(np.int32)
            ka_b1 = np.full(Tseg, 10.0, np.float32)

            for rho in RHOS_EVAL:
                tau = newsvendor_quantile(rho)
                dec = {name: decisions_from_rates(r[None, :], tau)
                       for name, r in variants.items()}
                dec["B1_fixed_keepalive"] = (pw_b1[None, :], ka_b1[None, :])
                for name, (pw, ka) in dec.items():
                    rc_sum = np.zeros(Tseg); rn_sum = np.zeros(Tseg)
                    for seed in DES_SEEDS:
                        r = simulate_function(seg_counts, pw[0], ka[0], dm, dstd,
                                              seed=seed * 104729 + fi,
                                              cold_mu=COLD_INIT["mu"],
                                              cold_sigma=COLD_INIT["sigma"],
                                              track_rolling=True)
                        rc_sum += r["roll_cold"]; rn_sum += r["roll_total"]
                    a = agg.setdefault((rho, name),
                                       [np.zeros(Tseg), np.zeros(Tseg)])
                    a[0] += rc_sum; a[1] += rn_sum

            # ---- Trigger benchmark on frozen-head residual stream ----
            with np.errstate(all="ignore"):
                resid = np.abs(np.log1p(seg_counts) -
                               np.log1p(np.maximum(r_frozen, 0)))
            for tname, trig in build_triggers().items():
                first_post = None; false_pre = 0
                for t in range(len(resid)):
                    if trig.update(float(resid[t])):
                        if t < rel_td:
                            false_pre += 1
                        elif first_post is None:
                            first_post = t
                trig_rows.setdefault(tname, []).append({
                    "drift": key, "func": int(fi),
                    "delay": (first_post - rel_td) if first_post is not None else None,
                    "false_pre": false_pre,
                })

        curves = {}
        for (rho, name), (rc, rn) in agg.items():
            c = rolling_csr(rc, rn, window=15)
            pairs = [(t, v) for t, v in enumerate(c) if not np.isnan(v)]
            al = compute_adaptation_lag(pairs, event_tick=rel_td)
            entry = {
                "rolling_csr": [None if np.isnan(x) else float(x) for x in c],
                "al": float(al) if al is not None else None,
                "csr_pre": float(np.nansum(rc[:rel_td]) / max(1, np.nansum(rn[:rel_td]))),
                "csr_post60": float(np.nansum(rc[rel_td:rel_td + 60]) /
                                    max(1, np.nansum(rn[rel_td:rel_td + 60]))),
                "csr_post_all": float(np.nansum(rc[rel_td:]) / max(1, np.nansum(rn[rel_td:]))),
            }
            curves.setdefault(str(rho), {})[name] = entry
            print(f"    rho={rho:5.1f} {name:<16s} pre={entry['csr_pre']:.4f} "
                  f"post60={entry['csr_post60']:.4f} "
                  f"postAll={entry['csr_post_all']:.4f} AL={entry['al']}")
        out["drift_curves"][key] = curves

    # ---- False alarms on no-drift control ----
    print("  --- no-drift control (false alarms) ---")
    for fi in active[:5]:
        seg_counts = counts[fi, seg_lo:seg_hi].astype(np.int64)
        feats_full = features_from_counts(counts[fi], t_offset=0,
                                          app_corate=features[fi, :, 10])
        feats_seg = feats_full[seg_lo:seg_hi]
        r_frozen, _ = a5_online_rates(trainer, feats_seg, seg_counts, seg_lo,
                                      frozen_after=rel_td - 1)
        resid = np.abs(np.log1p(seg_counts) - np.log1p(np.maximum(r_frozen, 0)))
        for tname, trig in build_triggers().items():
            fires = sum(1 for t in range(len(resid))
                        if trig.update(float(resid[t])))
            trig_rows.setdefault(tname, []).append({
                "drift": "none", "func": int(fi),
                "delay": None, "false_pre": fires,
            })

    # ---- T5 aggregation ----
    t5 = []
    for tname, rows in trig_rows.items():
        delays = [r["delay"] for r in rows if r["delay"] is not None]
        drift_rows = [r for r in rows if r["drift"] != "none"]
        ctrl_rows = [r for r in rows if r["drift"] == "none"]
        detected = sum(1 for r in drift_rows if r["delay"] is not None)
        # false rate: pre-drift + control fires per evaluated tick
        pre_fires = sum(r["false_pre"] for r in rows)
        pre_ticks = (len(drift_rows) * rel_td
                     + len(ctrl_rows) * 2 * DRIFT_WINDOW)
        t5.append({
            "Trigger": tname,
            "AL_mean_min": float(np.mean(delays)) if delays else None,
            "AL_std_min": float(np.std(delays)) if delays else None,
            "Detection_Rate": detected / max(1, len(drift_rows)),
            "False_per_1k_ticks": 1000.0 * pre_fires / max(1, pre_ticks),
            "n_events": len(drift_rows),
        })
    t5_df = pd.DataFrame(t5)
    t5_df.to_csv(TABLES_DIR / "T5_drift_triggers.csv", index=False)
    print("\n  T5 (measured):")
    print(t5_df.to_string(index=False))
    out["trigger_bench"] = t5
    return out


# ====================================================================
def main():
    features = np.load(PROCESSED_DIR / "features.npy")
    counts = np.load(PROCESSED_DIR / "counts.npy")
    splits_data = np.load(PROCESSED_DIR / "splits.npz")

    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE,
    )
    trainer.load(RUNS_DIR / "best_anil_ridge_s1_s0.pt")

    train_idx = splits_data["s1_train"]
    test_idx = splits_data["s1_test"]

    pm = build_prototypes(trainer, features, counts, train_idx)

    onb = onboarding_experiment(trainer, pm, features, counts, test_idx)
    drf = drift_experiment(trainer, features, counts, test_idx, train_idx)

    out = {"onboarding": onb, "drift": drf,
           "config": {"rhos_eval": RHOS_EVAL, "L": L,
                      "onboard_window": ONBOARD_WINDOW,
                      "drift_window": DRIFT_WINDOW,
                      "des_seeds": DES_SEEDS}}
    path = RUNS_DIR / "onboarding_drift_results.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
