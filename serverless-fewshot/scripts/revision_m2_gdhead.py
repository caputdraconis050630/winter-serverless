# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""M2: does the gradient-ANIL head beat the closed-form head at the DECISION?

Why this run exists
-------------------
The prototype mechanism exists to patch two properties of the closed-form
ridge head, not of meta-learning in general:

  * at K=0 the R2D2-style head has NO meta-learned weight matrix at all --
    src/models/heads.py RidgeHead carries exactly one learnable parameter
    (log_lambda), so the head must be solved from data and there is nothing
    to solve it from before the first invocation;
  * plain ridge is non-monotonic at small K (CRPS 0.054 at K=1 against
    0.030 at K=0), the "few-shot overfitting hump" of main text 3.2.

ANILGDHead has neither problem: it keeps a meta-learned W_init that serves
K=0 directly. The archived head ablation (ablation_results.json
heatmap_k_head, produced by phase8_heads.py) already measured it as better
at EVERY K -- 0.0185 vs 0.0302 at K=0, 0.0222 vs 0.0542 at K=1, ~0.017 vs
~0.031 thereafter -- and did so on 4,800 training episodes against the
production ridge checkpoint's 160,000.

But that ablation is CRPS, and this paper's central methodological claim is
precisely that CRPS does not decide: an LSTM with 6.6x better CRPS has a
WORSE cold-start rate, because the decision layer reads one fractile rather
than the whole distribution. So the gradient head's forecast advantage is
not evidence that it would prewarm better. This run puts it through the
decision layer on the headline cohort and finds out.

Design
------
Train body+ANILGDHead on the 2021 s2_train split (the split the production
ridge checkpoint used), then replay the Azure-2019 natural new-deployment
cohort under the registered protocol -- same cohort finder, W=240,
DES_SEEDS, RHOS, TAU semantics -- and compare against the archived ridge
arms in revision_m1_kproto_2019.json, whose cohort is identical by
construction (find_natural_cohort is seeded).

Serving cost, measured separately on this hardware for context: batched on
GPU the gradient head costs 1.2-2.9x the closed-form solve at platform
scale (0.72 vs 0.59 us/function at B=5000/K=60, 2.23 vs 0.77 at K=240) and
is actually FASTER at small batch. The real price of the gradient head is
second-order meta-training and two extra hyperparameters, not serving.

Writes results/runs/revision_m2_gdhead.json. Touches no archive.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("SF_DATA_DIR", "processed_2019")
os.environ["SF_RUNS_DIR"] = str(PROJECT_ROOT / "results_azure2021" / "runs")

from scripts.eval_adapt_biased import PROCESSED_DIR, RUNS_DIR      # noqa: E402
from scripts.phase63_onboarding_drift import DES_SEEDS, DEVICE     # noqa: E402
from scripts.phase6_des import decisions_from_rates, COLD_INIT     # noqa: E402
from scripts.revision_a4_crosstrace import (                       # noqa: E402
    find_natural_cohort, rate_from_logq_rows, PROC_2021,
    L, W, RHOS, GATE_THRESHOLD, QL,
)
from scripts.revision_m1_kproto import paired_stats, holm          # noqa: E402
from src.models.bodies import build_body                           # noqa: E402
from src.models.heads import (                                     # noqa: E402
    ANILGDHead, N_QUANTILES, pinball_loss,
)
from src.sim.des import simulate_function, rolling_csr             # noqa: E402
from src.decision.newsvendor import newsvendor_quantile            # noqa: E402

INNER_STEPS = 5          # ANILGDHead defaults
INNER_LR = 0.01
BOOT_SEED = 0


# ------------------------------------------------------------------ training
def sample_episode(features_t, counts, func_indices, rng, k_support=10,
                   k_query=32):
    T = features_t.shape[1]
    fi = int(rng.choice(func_indices))
    times = rng.integers(L, T - 1, size=k_support + k_query)
    xs = torch.stack([features_t[fi, t - L:t] for t in times]).to(DEVICE)
    ys = torch.tensor([np.log1p(counts[fi, t]) for t in times],
                      dtype=torch.float32, device=DEVICE)
    return xs[:k_support], ys[:k_support], xs[k_support:], ys[k_support:]


def train_gd(features_t, counts, train_idx, n_steps, episodes_per_step, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    body = build_body("tcn", features_t.shape[2]).to(DEVICE)
    head = ANILGDHead(64, N_QUANTILES, 1, inner_steps=INNER_STEPS,
                      inner_lr=INNER_LR).to(DEVICE)
    params = list(body.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    ql = torch.tensor(QL, dtype=torch.float32, device=DEVICE)

    t0 = time.time()
    hist = []
    for step in range(n_steps):
        loss_acc = 0.0
        opt.zero_grad()
        for _ in range(episodes_per_step):
            sx, sy, qx, qy = sample_episode(features_t, counts, train_idx, rng)
            phi_s, phi_q = body(sx), body(qx)
            sy_exp = sy.unsqueeze(-1).expand(-1, N_QUANTILES)
            Wh, bh = head.adapt(phi_s, sy_exp)
            pred = head.predict(phi_q, Wh, bh)
            loss = pinball_loss(pred, qy, ql)
            (loss / episodes_per_step).backward()
            loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == n_steps - 1:
            hist.append({"step": step, "loss": loss_acc / episodes_per_step,
                         "sec": time.time() - t0})
            print(f"    gd step {step}: loss={loss_acc/episodes_per_step:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return body, head, hist


# --------------------------------------------------------------- online adapt
def gd_adapt_batched(phi_s, y_s, W_init, b_init, steps=INNER_STEPS,
                     lr=INNER_LR):
    """Inner-loop SGD for a whole cohort at once.

    Mirrors ANILGDHead.adapt exactly: MSE against the scalar target broadcast
    over all quantile outputs, plain SGD, no autograd graph (serving path).
    F.mse_loss averages over K*O elements, so the analytic gradient carries
    the same 2/(K*O) factor.
    """
    F_n, K, d = phi_s.shape
    O = W_init.shape[-1]
    Wm = W_init.unsqueeze(0).expand(F_n, d, O).clone()
    bm = b_init.unsqueeze(0).expand(F_n, O).clone()
    if K == 0:
        return Wm, bm
    Y = y_s.unsqueeze(-1).expand(F_n, K, O)
    scale = 2.0 / (K * O)
    for _ in range(steps):
        resid = torch.baddbmm(bm.unsqueeze(1), phi_s, Wm) - Y      # [F,K,O]
        Wm = Wm - lr * scale * (phi_s.transpose(1, 2) @ resid)
        bm = bm - lr * scale * resid.sum(dim=1)
    return Wm, bm




def des_arms(rates, extra_dec, seg_counts, dm, dstd, F_n):
    """Run the registered DES over a dict of per-tick rate matrices."""
    out = {}
    for rho in RHOS:
        tau = newsvendor_quantile(rho)
        dec = {m: decisions_from_rates(r, tau) for m, r in rates.items()}
        dec.update(extra_dec)
        rho_out = {}
        for m, (pw_m, ka_m) in dec.items():
            roll_c = np.zeros(W)
            roll_n = np.zeros(W)
            f_call = np.zeros(F_n)
            f_wm = np.zeros(F_n)
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
                    f_call[f] += r["roll_cold"].sum()
                    f_wm[f] += r["idle_mem_gb_s"]
            f_call /= len(DES_SEEDS)
            f_wm /= len(DES_SEEDS)
            rho_out[m] = {
                "overall_csr": float(roll_c.sum() / max(roll_n.sum(), 1e-9)),
                "func_cold_all": f_call.tolist(),
                "wm_total": float(f_wm.sum())}
        out[str(rho)] = rho_out
    return out


def matched_memory(arm_pts, ref_pts):
    """CSR difference against a reference operating curve at equal memory.

    Comparing two policies at the same rho is not a fair comparison: they sit
    at different points on their own CSR-memory curves, and buying a lower
    cold-start rate with more warm memory is not an improvement. The main
    text makes exactly this argument in its steady-state section, so apply
    it here: interpolate the reference curve to the arm's memory and report
    the gap there. Points outside the reference curve's memory range are
    reported as such rather than extrapolated.
    """
    xs = [p[0] for p in ref_pts]
    ys = [p[1] for p in ref_pts]
    wm, csr = arm_pts
    if wm < xs[0] or wm > xs[-1]:
        return {"wm": wm, "csr": csr, "ref_csr": None,
                "delta_pp": None, "note": "outside reference curve range"}
    ref = float(np.interp(wm, xs, ys))
    return {"wm": wm, "csr": csr, "ref_csr": ref,
            "delta_pp": csr - ref,
            "note": "negative = better than reference at equal memory"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    t_start = time.time()
    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")
    nF = features.shape[2]

    feat21 = np.asarray(np.load(PROC_2021 / "features.npy", mmap_mode="r"))
    cnt21 = np.asarray(np.load(PROC_2021 / "counts.npy", mmap_mode="r"))
    spl21 = np.load(PROC_2021 / "splits.npz")
    feats21_t = torch.from_numpy(feat21).float()

    pts = find_natural_cohort(counts)
    F_n = len(pts)
    print(f"natural cohort: {F_n} functions; seeds={seeds}")
    seg_counts = np.stack([np.asarray(counts[fi, t0:t0 + W], dtype=np.int64)
                           for (fi, t0) in pts])
    y_all = np.stack([np.log1p(np.asarray(counts[fi, t0:t0 + W],
                                          dtype=np.float64)) for (fi, t0) in pts])
    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"]
                                 for (fi, _) in pts]), nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"]
                                   for (fi, _) in pts]), nan=0.5)

    # seed-independent baselines
    ewma = np.zeros(F_n)
    base_rates = {"B4a_ewma": np.zeros((F_n, W), dtype=np.float32)}
    for w in range(W):
        base_rates["B4a_ewma"][:, w] = np.expm1(np.maximum(ewma, 0))
        ewma = 0.1 * np.log1p(seg_counts[:, w].astype(np.float64)) + 0.9 * ewma
    base_rates["Oracle"] = seg_counts.astype(np.float32)
    pw_b1 = np.zeros((F_n, W), dtype=np.int32)
    for f in range(F_n):
        act = (seg_counts[f] > 0).astype(np.int32)
        conv1 = np.convolve(act, np.ones(10, dtype=np.int32))[:W]
        pw_b1[f, 1:] = (conv1[:-1] > 0).astype(np.int32)
    extra_dec = {"B1_fixed_keepalive": (pw_b1,
                                        np.full((F_n, W), 10.0, np.float32))}
    cum = np.zeros((F_n, W), dtype=np.int64)
    cum[:, 1:] = np.cumsum(seg_counts[:, :-1], axis=1)
    zero_h, mid = cum == 0, (cum > 0) & (cum < GATE_THRESHOLD)

    out = {"config": "M2 gradient-ANIL head vs closed-form ridge, 2019 cohort",
           "train_split": "2021 s2_train",
           "train_budget": {"steps": args.steps,
                            "episodes_per_step": args.episodes,
                            "total_episodes": args.steps * args.episodes,
                            "production_ridge_episodes": 5000 * 32,
                            "phase8_ablation_episodes": 1200 * 4},
           "inner": {"steps": INNER_STEPS, "lr": INNER_LR},
           "model_seeds": seeds, "n_functions": F_n,
           "func_ids": [int(f) for f, _ in pts],
           "per_seed": {}}

    for seed in seeds:
        t_seed = time.time()
        print(f"\n===== SEED {seed}: {args.steps}x{args.episodes} = "
              f"{args.steps*args.episodes} episodes", flush=True)
        body, head, hist = train_gd(feats21_t, cnt21, spl21["s2_train"],
                                    args.steps, args.episodes, seed=seed)
        torch.save({"body": body.state_dict(), "head": head.state_dict(),
                    "steps": args.steps, "episodes_per_step": args.episodes,
                    "seed": seed}, RUNS_DIR / f"best_anil_gd_s2_s{seed}.pt")
        print(f"  trained in {time.time()-t_seed:.0f}s", flush=True)

        body.eval()
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
                phi_all[:, w] = body(torch.stack(batch).to(DEVICE)).cpu()

        W_init = head.W_init.detach().to(DEVICE)
        b_init = head.b_init.detach().to(DEVICE)
        phi_g = phi_all.to(DEVICE)
        y_g = torch.from_numpy(y_all).float().to(DEVICE)
        n_out = N_QUANTILES
        pred_gd = np.zeros((F_n, W, n_out), dtype=np.float32)
        pred_gd_zero = np.zeros((F_n, W, n_out), dtype=np.float32)
        with torch.no_grad():
            for w in range(W):
                phi_now = phi_g[:, w]
                Wm, bm = gd_adapt_batched(phi_g[:, :w], y_g[:, :w],
                                          W_init, b_init)
                pred_gd[:, w] = torch.baddbmm(
                    bm.unsqueeze(1), phi_now.unsqueeze(1), Wm
                ).squeeze(1).cpu().numpy()
                pred_gd_zero[:, w] = (phi_now @ W_init + b_init).cpu().numpy()

        rates = dict(base_rates)
        for name, pred in (("gd", pred_gd), ("gd_zero", pred_gd_zero)):
            r = np.zeros((F_n, W), dtype=np.float32)
            for w in range(W):
                r[:, w] = rate_from_logq_rows(pred[:, w])
            rates[name] = r
        rates["gated_v3_gd"] = np.where(
            zero_h, rates["gd_zero"],
            np.where(mid, rates["B4a_ewma"], rates["gd"]))

        by_rho = des_arms(rates, extra_dec, seg_counts, dm, dstd, F_n)
        out["per_seed"][str(seed)] = {"train_loss_history": hist,
                                      "train_sec": time.time() - t_seed,
                                      "by_rho": by_rho}
        for rho in RHOS:
            v = by_rho[str(rho)]
            print(f"  rho={rho}: " + " ".join(
                f"{m}={x['overall_csr']*100:.3f}%" for m, x in v.items()),
                flush=True)

    # ---------------- comparison against the archived ridge run ----------
    ref_path = RUNS_DIR / "revision_m1_kproto_2019.json"
    if ref_path.exists():
        ref = json.load(open(ref_path))
        assert ref["func_ids"] == out["func_ids"], "cohort mismatch vs ridge run"
        rng = np.random.default_rng(BOOT_SEED)
        out["ridge_reference"] = {
            str(rho): {k: {"csr": ref["by_rho"][str(rho)][k]["overall_csr"],
                           "wm": ref["by_rho"][str(rho)][k]["wm_total"]}
                       for k in ("noproto", "k1_all", "k16", "k16_zero")}
            for rho in RHOS}
        ridge_curve = sorted(
            (ref["by_rho"][str(rho)]["k16"]["wm_total"],
             ref["by_rho"][str(rho)]["k16"]["overall_csr"] * 100)
            for rho in RHOS)
        out["ridge_k16_curve"] = ridge_curve

        for seed in seeds:
            sd = out["per_seed"][str(seed)]
            stats_out, mm = {}, {}
            for rho in RHOS:
                a = sd["by_rho"][str(rho)]
                b = ref["by_rho"][str(rho)]
                row = {}
                for ours, base, name in [
                        ("gd", "k16", "gd_vs_ridge_proto"),
                        ("gd", "noproto", "gd_vs_ridge_noproto"),
                        ("gd", "k1_all", "gd_vs_single_template"),
                        ("gd_zero", "k16_zero", "gdzero_vs_protozero")]:
                    row[name] = paired_stats(a[ours]["func_cold_all"],
                                             b[base]["func_cold_all"], rng)
                row["gd_vs_ewma"] = paired_stats(a["gd"]["func_cold_all"],
                                                 a["B4a_ewma"]["func_cold_all"],
                                                 rng)
                stats_out[str(rho)] = row
                mm[str(rho)] = {
                    arm: matched_memory((a[arm]["wm_total"],
                                         a[arm]["overall_csr"] * 100),
                                        ridge_curve)
                    for arm in ("gd", "gd_zero", "gated_v3_gd")}
            for name in stats_out[str(RHOS[0])]:
                ps = [stats_out[str(r)][name]["wilcoxon_p"] for r in RHOS]
                for r, v in zip(RHOS, holm(ps)):
                    stats_out[str(r)][name]["holm_p_across_rho"] = v
            sd["stats"] = stats_out
            sd["matched_memory_vs_ridge_k16"] = mm

    # ---------------- across-seed summary --------------------------------
    summary = {}
    for rho in RHOS:
        row = {}
        for arm in ("gd", "gd_zero", "gated_v3_gd", "B4a_ewma"):
            csrs = [out["per_seed"][str(s)]["by_rho"][str(rho)][arm]["overall_csr"] * 100
                    for s in seeds]
            wms = [out["per_seed"][str(s)]["by_rho"][str(rho)][arm]["wm_total"]
                   for s in seeds]
            row[arm] = {"csr_mean": float(np.mean(csrs)),
                        "csr_min": float(np.min(csrs)),
                        "csr_max": float(np.max(csrs)),
                        "csr_std": float(np.std(csrs, ddof=1)) if len(csrs) > 1 else 0.0,
                        "wm_mean": float(np.mean(wms))}
        if "matched_memory_vs_ridge_k16" in out["per_seed"][str(seeds[0])]:
            for arm in ("gd", "gd_zero", "gated_v3_gd"):
                ds = [out["per_seed"][str(s)]["matched_memory_vs_ridge_k16"][str(rho)][arm]["delta_pp"]
                      for s in seeds]
                ds = [x for x in ds if x is not None]
                row[arm]["matched_memory_delta_pp"] = {
                    "n_in_range": len(ds),
                    "mean": float(np.mean(ds)) if ds else None,
                    "min": float(np.min(ds)) if ds else None,
                    "max": float(np.max(ds)) if ds else None}
        summary[str(rho)] = row
    out["across_seeds"] = summary

    out["wall_sec"] = time.time() - t_start
    path = RUNS_DIR / "revision_m2_gdhead.json"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0, "corrupt write (sdb guard)"
    json.load(open(path))
    print(f"\nSaved + verified {path} ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
