# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R2: direct-quantile-read decision ablation (pre-registered in
results/runs/revision_r2_PREREG.md — read that first).

Does the deployed reduction (quantile grid -> scalar rate -> Poisson
fractile) drive the "CRPS advantage does not reach the decision" finding,
or does the ranking survive a rule that reads the fractile directly from
the predictive quantile grid?

  --part steady   Azure 2021 S2+S3: B7 LSTM and B5 global head, each under
                  (median read -> Poisson layer) and (direct tau* grid read),
                  same trained weights per arm pair. B4a_ewma control.
                  rho in {1,10}, DES seeds 0..9.
  --part cohort   2019 natural cohort (a4 protocol): A5_proto fixed-read
                  (registered tau=0.909) vs direct tau*(rho) read.

Env: torch stage — PYTHONPATH=/data/260715/site-packages:. python3.13
Output: results/runs/revision_r2_directread.json (parts merged on rerun).
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_des import (  # noqa: E402
    rates_ewma, run_config, decisions_from_rates, COLD_INIT,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES, QUANTILES, pinball_loss  # noqa: E402

torch.set_num_threads(1)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QL = np.linspace(0.05, 0.95, N_QUANTILES)
L = 60
MED_IDX = N_QUANTILES // 2
TRAIN_SEED = 0
SEEDS = list(range(10))
RHOS = [1.0, 10.0]          # pre-registered; rho=100 excluded (clamp, see PREREG)
OUT = RUNS_DIR / "revision_r2_directread.json"


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:8]


# --------------------------------------------------------------------
# The new decision function (the only new decision logic in R2)
# --------------------------------------------------------------------
def decisions_from_quantiles(q_log, tau_star):
    """Direct fractile read from a log1p-space quantile grid [..., 19].

    Registered clamp: reads above the top grid level (0.95) use 0.95
    (revision_e4_fractile.py:175-181 rule). Pool size = ceil of the
    count-space tau* quantile. Keep-alive comes from the registered block
    unchanged, with the tau*-read as its scalar rate input.
    """
    tt = min(tau_star, QL[-1])
    w = np.interp(tt, QL, np.arange(N_QUANTILES))
    lo, hi = int(np.floor(w)), int(np.ceil(w))
    frac = w - lo
    qv = (1 - frac) * q_log[..., lo] + frac * q_log[..., hi]
    c = np.expm1(np.maximum(qv, 0.0)).astype(np.float32)
    pw = np.ceil(c - 1e-9).astype(np.int32)
    _, ka = decisions_from_rates(c, tau_star)
    return pw, ka


# --------------------------------------------------------------------
# B4 LSTM protocol, full quantile head (verbatim from
# revision_e4_fractile.py:94-122 / revision_b4_lstm_des.py training loop)
# --------------------------------------------------------------------
def train_predict_lstm_quantiles(feats_f, counts_f, train_end, pred_ticks):
    in_features = feats_f.shape[1]
    ft = torch.from_numpy(feats_f).float()
    lstm = torch.nn.LSTM(in_features, 64, num_layers=1, batch_first=True).to(DEVICE)
    head = torch.nn.Linear(64, N_QUANTILES).to(DEVICE)
    opt = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=1e-3)
    qdev = QUANTILES.to(DEVICE)
    for _ in range(50):
        t_samples = np.random.randint(L, train_end, size=16)
        x = torch.stack([ft[t - L:t] for t in t_samples]).to(DEVICE)
        y = torch.tensor([np.log1p(counts_f[t]) for t in t_samples]).float().to(DEVICE)
        out, _ = lstm(x)
        pred = head(out[:, -1, :])
        loss = pinball_loss(pred, y, qdev)
        opt.zero_grad(); loss.backward(); opt.step()
    lstm.eval()
    qs = np.zeros((len(pred_ticks), N_QUANTILES), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(pred_ticks), 2048):
            ticks = pred_ticks[s:s + 2048]
            x = torch.stack([
                torch.cat([torch.zeros(max(0, L - t), in_features), ft[max(0, t - L):t]])
                for t in ticks]).to(DEVICE)
            out, _ = lstm(x)
            qs[s:s + 2048] = head(out[:, -1, :]).cpu().numpy()
    return qs


# --------------------------------------------------------------------
# B5 global head, grid-preserving variant of phase6_des.rates_b5
# --------------------------------------------------------------------
def b5_quantile_grid(counts, features, trainer):
    """Same fit and forward as rates_b5, but keeps the full [N,T,19] grid
    (log1p space). Warmup ticks t<L carry the EWMA fallback as a rate; the
    grid is undefined there (handled by decision composition)."""
    N, T = counts.shape
    grid = np.zeros((N, T, N_QUANTILES), dtype=np.float32)
    warm_rates = np.zeros((N, T), dtype=np.float32)
    features_t = torch.from_numpy(features).float()
    trainer.body.eval()
    with torch.no_grad():
        all_phi, all_y = [], []
        for t in range(L, min(L + 1000, T)):
            bx = features_t[:, t - L:t, :].to(DEVICE)
            all_phi.append(trainer.body(bx).cpu())
            all_y.append(torch.from_numpy(np.log1p(counts[:, t])).float())
        Phi = torch.cat(all_phi, 0)
        Y = torch.cat(all_y, 0).unsqueeze(-1).expand(-1, N_QUANTILES)
        lam = trainer.head.ridge_lambda.item()
        Wg = torch.linalg.solve(Phi.T @ Phi + lam * torch.eye(64), Phi.T @ Y).to(DEVICE)

        ewma = np.zeros(N)
        for t in range(min(L, T)):
            warm_rates[:, t] = np.expm1(np.maximum(ewma, 0))
            prev = counts[:, max(t - 1, 0)].astype(np.float64)
            ewma = 0.1 * np.log1p(prev) + 0.9 * ewma

        for t in range(L, T):
            bx = features_t[:, t - L:t, :].to(DEVICE)
            grid[:, t] = (trainer.body(bx) @ Wg).cpu().numpy()
            if t % 5000 == 0:
                print(f"      B5 tick {t}/{T}", flush=True)
    return grid, warm_rates


def compose(dec_prefix, dec_suffix, t_switch):
    """Stitch (pw, ka) matrices: prefix ticks < t_switch, suffix after."""
    pw = dec_prefix[0].copy(); ka = dec_prefix[1].copy()
    pw[:, t_switch:] = dec_suffix[0][:, t_switch:]
    ka[:, t_switch:] = dec_suffix[1][:, t_switch:]
    return pw, ka


def steady():
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    features = np.load(PROCESSED_DIR / "features.npy")
    counts_all = np.load(PROCESSED_DIR / "counts.npy")
    splits = np.load(PROCESSED_DIR / "splits.npz")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")

    from src.meta.trainer import ANILMetaTrainer
    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE)
    ckpt = RUNS_DIR / "best_anil_ridge_s1_s0.pt"
    trainer.load(ckpt)
    print(f"checkpoint: {ckpt} md5={md5(ckpt)}", flush=True)

    part = {"checkpoint": str(ckpt), "checkpoint_md5": md5(ckpt),
            "train_seed": TRAIN_SEED, "rhos": RHOS, "seeds": SEEDS,
            "results": [], "gates": {}}

    # b4 ordering discipline: seed once, then S2 loop, then S3 loop
    for split in ["S2", "S3"]:
        key = {"S2": "s2_test", "S3": "s3_test"}[split]
        test_idx = splits[key]
        T_full = counts_all.shape[1]
        seg_counts = counts_all[test_idx]
        seg_feats = features[test_idx]
        n = len(test_idx)
        t0 = int(splits.get("s3_test_t_start", [10080])[0]) if split == "S3" else T_full // 2

        print(f"### {split}: {n} funcs, LSTM quantile heads (train_end={t0})", flush=True)
        t_start = time.time()
        pred_ticks = list(range(t0, T_full))
        lstm_q = np.zeros((n, len(pred_ticks), N_QUANTILES), dtype=np.float32)
        for i in range(n):
            lstm_q[i] = train_predict_lstm_quantiles(
                seg_feats[i], seg_counts[i], t0, pred_ticks)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{n} LSTMs ({time.time()-t_start:.0f}s)", flush=True)
        print(f"  LSTMs done in {time.time()-t_start:.0f}s", flush=True)

        if split == "S3":
            eval_counts = seg_counts[:, t0:]
            eval_feats = seg_feats[:, t0:]
        else:
            eval_counts = seg_counts
            eval_feats = seg_feats

        ew = rates_ewma(eval_counts)
        grid_b5, warm_b5 = b5_quantile_grid(eval_counts, eval_feats, trainer)
        rates_b5_med = warm_b5.copy()
        rates_b5_med[:, L:] = np.expm1(np.maximum(grid_b5[:, L:, MED_IDX], 0.0))

        # LSTM rate/grid aligned to the eval segment
        if split == "S3":
            lstm_grid_eval = lstm_q                      # [n, Te, 19]
            lstm_prefix_rates = None
            t_switch = 0
        else:
            lstm_grid_eval = np.zeros((n, T_full, N_QUANTILES), dtype=np.float32)
            lstm_grid_eval[:, t0:] = lstm_q
            lstm_prefix_rates = ew                        # EWMA before t0 (b4 rule)
            t_switch = t0
        lstm_med_rates = np.expm1(np.maximum(lstm_grid_eval[..., MED_IDX], 0.0))
        if split == "S2":
            lstm_med_rates[:, :t0] = ew[:, :t0]

        dm = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df) else 1.0
             for fi in test_idx]), nan=1.0)
        ds = np.nan_to_num(np.array(
            [dur_df.iloc[fi]["dur_std"] if fi < len(dur_df) else 0.5
             for fi in test_idx]), nan=0.5)

        jobs = []
        for rho in RHOS:
            tau = newsvendor_quantile(rho)
            dec_ew = decisions_from_rates(ew, tau)
            # B5 pair: shared EWMA-warmup decisions for t<L
            d_b5_med = decisions_from_rates(rates_b5_med, tau)
            d_b5_dir = compose(d_b5_med, decisions_from_quantiles(grid_b5, tau), L)
            # B7 pair: shared EWMA prefix for S2 (t<t0)
            d_b7_med = decisions_from_rates(lstm_med_rates, tau)
            d_b7_dir_core = decisions_from_quantiles(lstm_grid_eval, tau)
            d_b7_dir = compose(d_b7_med, d_b7_dir_core, t_switch) if t_switch else d_b7_dir_core
            for method, dec in [("B4a_ewma", dec_ew),
                                ("B5_median", d_b5_med), ("B5_direct", d_b5_dir),
                                ("B7_lstm_median", d_b7_med), ("B7_lstm_direct", d_b7_dir)]:
                jobs.append((method, rho, SEEDS, eval_counts, dec[0], dec[1],
                             dm, ds, COLD_INIT, split))

        print(f"  DES: {len(jobs)} configs x {len(SEEDS)} seeds", flush=True)
        with ProcessPoolExecutor(max_workers=4) as ex:
            for res in ex.map(run_config, jobs):
                for r in res:
                    r.pop("func_csr", None)  # keep func_cold for pairing, drop bulk
                part["results"].extend(res)
                r0 = res[0]
                print(f"    {r0['split']} {r0['method']:16s} rho={r0['cost_ratio']:<5} "
                      f"csr={np.mean([x['csr'] for x in res])*100:.3f}%", flush=True)

    # ---- reproduction gates ----
    camp = json.load(open(RUNS_DIR / "sim_results_des_revision.json"))
    camp_rows = camp["results"] if isinstance(camp, dict) and "results" in camp else camp
    gate = {}
    for split in ["S2", "S3"]:
        for rho in RHOS:
            ours = [r["csr"] for r in part["results"]
                    if r["method"] == "B4a_ewma" and r["split"] == split
                    and r["cost_ratio"] == rho]
            ref = [r["csr"] for r in camp_rows
                   if r["method"] == "B4a_ewma" and r["split"] == split
                   and r["cost_ratio"] == rho and r["seed"] in SEEDS]
            delta = abs(np.mean(ours) - np.mean(ref)) * 100 if ref else None
            gate[f"B4a|{split}|{rho}"] = {"ours_pct": np.mean(ours) * 100,
                                          "ref_pct": np.mean(ref) * 100 if ref else None,
                                          "delta_pp": delta}
    b4 = json.load(open(RUNS_DIR / "revision_b4_lstm_des.json"))
    b4_rows = b4["results"] if isinstance(b4, dict) and "results" in b4 else b4
    for split in ["S2", "S3"]:
        for rho in RHOS:
            ours = [r["csr"] for r in part["results"]
                    if r["method"] == "B7_lstm_median" and r["split"] == split
                    and r["cost_ratio"] == rho]
            ref = [r["csr"] for r in b4_rows
                   if r["split"] == split and r["cost_ratio"] == rho]
            gate[f"B7med|{split}|{rho}"] = {
                "ours_pct": np.mean(ours) * 100,
                "b4_ref_pct": np.mean(ref) * 100 if ref else None,
                "delta_pp": abs(np.mean(ours) - np.mean(ref)) * 100 if ref else None,
                "note": "best-effort (LSTM RNG stream), read comparison is within-run"}
    part["gates"] = gate
    return part


def cohort():
    """A5_proto fixed-read vs direct tau* read on the 2019 natural cohort.
    Mirrors revision_a4_crosstrace.py; reuses its cohort, prototypes and
    ridge/pred stage, swapping only the decision read."""
    from scripts.revision_a4_crosstrace import (
        find_natural_cohort, rate_from_logq_rows, load_trainer,
        build_prototypes, PROCESSED_DIR as P19, DES_SEEDS, W, GATE_THRESHOLD)
    from scripts.eval_adapt_biased import PROCESSED_DIR as _p  # noqa: F401
    from src.sim.des import simulate_function, rolling_csr

    features = np.load(P19 / "features.npy", mmap_mode="r")
    counts = np.load(P19 / "counts.npy", mmap_mode="r")
    dur_df = pd.read_csv(P19 / "duration_stats.csv")
    nF = features.shape[2]
    PROC_2021 = PROJECT_ROOT / "data" / "processed"
    feat21 = np.load(PROC_2021 / "features.npy", mmap_mode="r")
    cnt21 = np.load(PROC_2021 / "counts.npy", mmap_mode="r")
    spl21 = np.load(PROC_2021 / "splits.npz")

    pts = find_natural_cohort(counts)
    F_n = len(pts)
    print(f"cohort: {F_n} functions", flush=True)
    trainer, biased_head = load_trainer(nF, seed=0)
    centroids, proto_w = build_prototypes(trainer, biased_head, feat21, cnt21,
                                          spl21["s2_train"])

    import torch.nn.functional as Fun
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
    y_all = np.stack([np.log1p(np.asarray(counts[fi, t0:t0 + W], dtype=np.float64))
                      for (fi, t0) in pts])
    y_t = torch.from_numpy(y_all).float()
    lam = trainer.head.ridge_lambda.item()
    preds = np.zeros((F_n, W, N_QUANTILES), dtype=np.float32)   # A5_proto grid
    phi_g = phi_all.to(DEVICE); y_g = y_t.to(DEVICE)
    cen_g = centroids.to(DEVICE); pw_g = proto_w.to(DEVICE)
    eye_g = torch.eye(64, device=DEVICE)
    for w in range(W):
        phi_now = phi_g[:, w]
        sims = Fun.cosine_similarity(phi_now.unsqueeze(1), cen_g.unsqueeze(0), dim=2)
        Wp = pw_g[sims.argmax(dim=1)]
        if w == 0:
            W_p = Wp.clone()
        else:
            phi_s = phi_g[:, :w]
            y_s = y_g[:, :w].unsqueeze(-1).expand(F_n, w, N_QUANTILES)
            PhiT = phi_s.transpose(1, 2)
            A = PhiT @ phi_s + lam * eye_g.unsqueeze(0)
            W_p = torch.linalg.solve(A, PhiT @ y_s + lam * Wp)
        preds[:, w] = (phi_now.unsqueeze(1) @ W_p).squeeze(1).cpu().numpy()
    print("ridge/pred stage done", flush=True)

    seg_counts = np.stack([np.asarray(counts[fi, t0:t0 + W], dtype=np.int64)
                           for (fi, t0) in pts])
    rate_fixed = np.zeros((F_n, W), dtype=np.float32)
    for w in range(W):
        rate_fixed[:, w] = rate_from_logq_rows(preds[:, w])
    ewma = np.zeros(F_n)
    rates_ew = np.zeros((F_n, W), dtype=np.float32)
    for w in range(W):
        rates_ew[:, w] = np.expm1(np.maximum(ewma, 0))
        ewma = 0.1 * np.log1p(seg_counts[:, w].astype(np.float64)) + 0.9 * ewma

    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] for (fi, _) in pts]), nan=1.0)
    dstd = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] for (fi, _) in pts]), nan=0.5)

    out = {"n_functions": F_n, "des_seeds": DES_SEEDS, "rhos": RHOS, "by_rho": {}}
    for rho in RHOS:
        tau = newsvendor_quantile(rho)
        dec = {
            "A5_proto_fixedread": decisions_from_rates(rate_fixed, tau),
            "A5_proto_direct": decisions_from_quantiles(preds, tau),
            "B4a_ewma": decisions_from_rates(rates_ew, tau),
            "Oracle": decisions_from_rates(seg_counts.astype(np.float32), tau),
        }
        rho_out = {}
        for m, (pw_m, ka_m) in dec.items():
            roll_c = np.zeros(W); roll_n = np.zeros(W)
            func_cold_all = np.zeros(F_n); func_wm = np.zeros(F_n)
            for f in range(F_n):
                for seed in DES_SEEDS:
                    r = simulate_function(seg_counts[f], pw_m[f], ka_m[f],
                                          float(dm[f]), float(dstd[f]),
                                          seed=seed * 7919 + f,
                                          cold_mu=COLD_INIT["mu"],
                                          cold_sigma=COLD_INIT["sigma"],
                                          track_rolling=True)
                    roll_c += r["roll_cold"]; roll_n += r["roll_total"]
                    func_cold_all[f] += r["roll_cold"].sum()
                    func_wm[f] += r["idle_mem_gb_s"]
            func_cold_all /= len(DES_SEEDS); func_wm /= len(DES_SEEDS)
            rho_out[m] = {"overall_csr": float(roll_c.sum() / max(roll_n.sum(), 1e-9)),
                          "func_cold_all": func_cold_all.tolist(),
                          "wm_total": float(func_wm.sum()),
                          "rolling_csr": rolling_csr(roll_c, roll_n, window=15).tolist()}
            print(f"  rho={rho} {m}: csr={rho_out[m]['overall_csr']*100:.3f}%", flush=True)
        from scipy import stats as sps
        diff = (np.array(rho_out["A5_proto_fixedread"]["func_cold_all"])
                - np.array(rho_out["A5_proto_direct"]["func_cold_all"]))
        nz = diff[diff != 0]
        rho_out["_direct_vs_fixed"] = {
            "mean_diff_cold": float(diff.mean()),
            "wilcoxon_p": float(sps.wilcoxon(nz)[1]) if len(nz) >= 6 else 1.0}
        out["by_rho"][str(rho)] = rho_out

    # reproduction gate vs archived a4 (fixed read, rho in RHOS)
    a4 = json.load(open(RUNS_DIR / "revision_a4_crosstrace.json"))
    gate = {}
    for rho in RHOS:
        ref = a4["by_rho"][str(rho)]["A5_proto"]["overall_csr"]
        ours = out["by_rho"][str(rho)]["A5_proto_fixedread"]["overall_csr"]
        gate[f"a4|{rho}"] = {"ours_pct": ours * 100, "ref_pct": ref * 100,
                             "delta_pp": abs(ours - ref) * 100}
    out["gates"] = gate
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["steady", "cohort"], required=True)
    args = ap.parse_args()
    existing = json.load(open(OUT)) if OUT.exists() else {
        "prereg": "revision_r2_PREREG.md", "cold_init": COLD_INIT}
    if args.part == "steady":
        existing["steady"] = steady()
    else:
        existing["cohort"] = cohort()
    json.dump(existing, open(OUT, "w"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
