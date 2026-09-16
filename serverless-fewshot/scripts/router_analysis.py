#!/usr/bin/env python3
"""Complementarity analysis: oracle router and validation-tuned threshold router
over {ours (ANIL K=10), B5 (global, no adapt)}.

Protocol:
  - Per-function CRPS for both methods on s2_val (router tuning) and s2_test
    (reporting), using the exact gate/diag eval placement (same eval cap seed).
  - Routing feature is deployment-observable: mean invocation rate over the
    support half of the trace (counts[:, :T//2].mean).
  - Threshold tau chosen on val only (grid over val log-rate percentiles,
    including +/-inf so the router can degenerate to a single method);
    reported on test.
  - Oracle router = per-function min(ours, b5): upper bound on complementarity.

Env:
  SF_DATA_DIR   data subdir (processed | processed_2019), default "processed"
  SF_RUNS_DIR   dir with checkpoints, default results/runs
  SF_TAG        output tag, default = SF_DATA_DIR

Output: results/runs/router_analysis_<tag>.json (+ per-function npz)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from scipy import stats as sstats  # noqa: E402

from scripts.phase4_train import (  # noqa: E402
    PROCESSED_DIR, RUNS_DIR, DEVICE, N_QUANTILES, QUANTILES,
    cap_eval_indices, crps_from_quantiles,
)
from scripts.diag_gate_2019 import evaluate_k, evaluate_b5, HORIZONS  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402
from src.models.bodies import build_body  # noqa: E402

CKPT_RUNS = Path(os.environ.get("SF_RUNS_DIR", str(RUNS_DIR)))
TAG = os.environ.get("SF_TAG", os.environ.get("SF_DATA_DIR", "processed"))
K_SUPPORT = 10
OURS_SEEDS = (0, 1, 2)
N_BOOT = 10000


def load_ours(in_features, seed):
    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=in_features, embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=len(HORIZONS),
        use_amp=True, device=DEVICE,
    )
    trainer.load(CKPT_RUNS / f"best_anil_ridge_s2_s{seed}.pt")
    return trainer


def load_b5(in_features):
    ckpt = torch.load(CKPT_RUNS / "b5_global_s0.pt", map_location=DEVICE,
                      weights_only=False)
    body = build_body("tcn", in_features).to(DEVICE)
    head = torch.nn.Linear(64, N_QUANTILES * len(HORIZONS)).to(DEVICE)
    body.load_state_dict(ckpt["body"])
    head.load_state_dict(ckpt["head"])
    return body, head


def route_mask(log_rates, tau):
    """True -> ours, False -> b5. Low-traffic functions go to ours."""
    return log_rates < tau


def router_mean(ours, b5, log_rates, tau):
    m = route_mask(log_rates, tau)
    return float(np.where(m, ours, b5).mean())


def tune_tau(ours_val, b5_val, log_rates_val):
    grid = [-np.inf, np.inf] + list(
        np.percentile(log_rates_val, np.arange(5, 100, 5)))
    best_tau, best = None, np.inf
    for tau in grid:
        m = router_mean(ours_val, b5_val, log_rates_val, tau)
        if m < best - 1e-12:
            best, best_tau = m, tau
    return best_tau, best


def evaluate_selection(trainer, b5_body, b5_head, features, counts, func_indices,
                       k_support=K_SUPPORT):
    """Per-function backtest on the FIRST half only (temporally before the
    test-query period): adapt ours on early supports, score both methods on
    late-first-half selection windows. Deployment-honest: uses no test data.

    Returns (sel_ours, sel_b5): per-function CRPS arrays on selection windows.
    """
    quantiles = QUANTILES.to(DEVICE)
    n_outputs = N_QUANTILES * len(HORIZONS)
    trainer.body.eval()
    b5_body.eval()
    b5_head.eval()
    L = 60
    T = features.shape[1]

    def win(fi, t):
        return torch.from_numpy(np.asarray(features[fi, t - L:t], dtype=np.float32))

    sel_ours, sel_b5 = [], []
    with torch.no_grad():
        for fi in func_indices:
            # Emulate the deployed adaptation as closely as the timeline
            # allows: supports span up to 3T/8, selection windows fill the
            # remaining pre-test gap (3T/8, T/2).
            sup_times = np.linspace(L, 3 * T // 8, k_support, dtype=int)
            sel_times = np.linspace(3 * T // 8 + 1, T // 2 - 2, 48, dtype=int)

            sx = torch.stack([win(fi, t) for t in sup_times]).to(DEVICE)
            sy = torch.stack([
                torch.tensor([np.log1p(counts[fi, min(t + h - 1, T - 1)])
                              for h in HORIZONS])
                for t in sup_times]).float().to(DEVICE)
            qx = torch.stack([win(fi, t) for t in sel_times]).to(DEVICE)
            qy = torch.stack([
                torch.tensor([np.log1p(counts[fi, min(t + h - 1, T - 1)])
                              for h in HORIZONS])
                for t in sel_times]).float().to(DEVICE)

            phi_s = trainer.body(sx)
            phi_q = trainer.body(qx)
            sy_exp = sy.unsqueeze(-1).expand(k_support, len(HORIZONS), N_QUANTILES)
            sy_exp = sy_exp.reshape(k_support, n_outputs)
            W = trainer.head.adapt(phi_s, sy_exp)
            pred = trainer.head.predict(phi_q, W)
            sel_ours.append(crps_from_quantiles(
                pred[:, :N_QUANTILES], qy[:, 0], quantiles).item())

            phi_b = b5_body(qx)
            pred_b = b5_head(phi_b)
            sel_b5.append(crps_from_quantiles(
                pred_b[:, :N_QUANTILES], qy[:, 0], quantiles).item())
    return np.array(sel_ours), np.array(sel_b5)


ALPHA_GRID = (0.0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
              1.1, 1.25, 1.5, 2.0, np.inf)


def tune_alpha(sel_ours_val, sel_b5_val, ours_val, b5_val):
    """Margin rule: route to ours iff sel_ours < alpha * sel_b5.
    alpha=0 -> always B5, alpha=inf -> always ours (no-regret ends).
    Tuned on validation functions only."""
    best_a, best = None, np.inf
    for a in ALPHA_GRID:
        m = sel_ours_val < a * sel_b5_val
        v = float(np.where(m, ours_val, b5_val).mean())
        if v < best - 1e-12:
            best, best_a = v, a
    return best_a, best


def paired_bootstrap_ci(a, b, n_boot=N_BOOT, seed=7):
    """CI for mean(a) - mean(b) over paired per-function scores."""
    rng = np.random.default_rng(seed)
    diffs = a - b
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diffs[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize(scores):
    return {"mean": float(scores.mean()), "median": float(np.median(scores)),
            "p90": float(np.percentile(scores, 90))}


def main():
    features = np.load(PROCESSED_DIR / "features.npy", mmap_mode="r")
    counts = np.load(PROCESSED_DIR / "counts.npy", mmap_mode="r")
    splits = np.load(PROCESSED_DIR / "splits.npz")
    T = counts.shape[1]

    val_idx = cap_eval_indices(splits["s2_val"])
    test_idx = cap_eval_indices(splits["s2_test"])
    print(f"[{TAG}] val n={len(val_idx)}, test n={len(test_idx)}, "
          f"ckpts={CKPT_RUNS}")

    # Deployment-observable routing feature: support-half mean rate
    def obs_log_rate(idx):
        r = np.asarray(counts[idx, :T // 2]).mean(axis=1)
        return np.log10(r + 1e-6)

    lr_val, lr_test = obs_log_rate(val_idx), obs_log_rate(test_idx)

    b5_body, b5_head = load_b5(features.shape[2])
    b5_val = evaluate_b5(b5_body, b5_head, features, counts, val_idx)
    b5_test = evaluate_b5(b5_body, b5_head, features, counts, test_idx)

    out = {"tag": TAG, "k_support": K_SUPPORT,
           "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
           "b5_test": summarize(b5_test), "per_seed": {}}
    per_func = {"b5_val": b5_val, "b5_test": b5_test,
                "log_rate_val": lr_val, "log_rate_test": lr_test,
                "val_idx": val_idx, "test_idx": test_idx}

    for seed in OURS_SEEDS:
        ckpt = CKPT_RUNS / f"best_anil_ridge_s2_s{seed}.pt"
        if not ckpt.exists():
            print(f"  seed {seed}: checkpoint missing, skipping")
            continue
        trainer = load_ours(features.shape[2], seed)
        ours_val = evaluate_k(trainer, features, counts, val_idx, K_SUPPORT)
        ours_test = evaluate_k(trainer, features, counts, test_idx, K_SUPPORT)
        per_func[f"ours_val_s{seed}"] = ours_val
        per_func[f"ours_test_s{seed}"] = ours_test

        oracle_test = np.minimum(ours_test, b5_test)
        best_single = min(ours_test.mean(), b5_test.mean())
        gap = best_single - oracle_test.mean()

        # Router 1: rate threshold tuned on val
        tau, val_router = tune_tau(ours_val, b5_val, lr_val)
        mask_rate = route_mask(lr_test, tau)
        rate_router = np.where(mask_rate, ours_test, b5_test)

        # Router 2: per-function backtest selection (alpha=1, no tuning)
        sel_ours, sel_b5 = evaluate_selection(
            trainer, b5_body, b5_head, features, counts, test_idx)
        mask_bt = sel_ours < sel_b5
        bt_router = np.where(mask_bt, ours_test, b5_test)
        per_func[f"sel_ours_s{seed}"] = sel_ours
        per_func[f"sel_b5_s{seed}"] = sel_b5

        # Router 3: margin rule, alpha tuned on val functions only
        sel_ours_val, sel_b5_val = evaluate_selection(
            trainer, b5_body, b5_head, features, counts, val_idx)
        per_func[f"sel_ours_val_s{seed}"] = sel_ours_val
        per_func[f"sel_b5_val_s{seed}"] = sel_b5_val
        alpha, val_margin = tune_alpha(sel_ours_val, sel_b5_val,
                                       ours_val, b5_val)
        mask_mg = sel_ours < alpha * sel_b5
        mg_router = np.where(mask_mg, ours_test, b5_test)

        # How predictive is the backtest of the test-period winner?
        agree = float((mask_bt == (ours_test < b5_test)).mean())

        res = {
            "ours_test": summarize(ours_test),
            "oracle_test": summarize(oracle_test),
            "best_single_test_mean": float(best_single),
            "oracle_gap_vs_best_single": float(gap),
            "ours_win_rate_test": float((ours_test < b5_test).mean()),
        }
        for name, router, mask in (("rate_router", rate_router, mask_rate),
                                   ("backtest_router", bt_router, mask_bt),
                                   ("margin_router", mg_router, mask_mg)):
            ci_lo, ci_hi = paired_bootstrap_ci(router, b5_test)
            try:
                w_p = float(sstats.wilcoxon(router, b5_test).pvalue) \
                    if not np.allclose(router, b5_test) else 1.0
            except ValueError:
                w_p = 1.0
            rec = (best_single - router.mean()) / gap if gap > 0 else 0.0
            res[name] = {
                **summarize(router),
                "frac_routed_to_ours": float(mask.mean()),
                "vs_b5_mean_diff": float(router.mean() - b5_test.mean()),
                "vs_b5_diff_ci95": [ci_lo, ci_hi],
                "vs_b5_wilcoxon_p": w_p,
                "recovers_frac_of_oracle_gap": float(rec),
            }
        res["rate_router"]["tau_log10_rate"] = (
            float(tau) if np.isfinite(tau) else str(tau))
        res["backtest_router"]["selection_agreement_with_test_winner"] = agree
        res["margin_router"]["alpha"] = (
            float(alpha) if np.isfinite(alpha) else str(alpha))
        res["margin_router"]["val_mean"] = val_margin
        out["per_seed"][str(seed)] = res
        print(f"  seed {seed}: ours={ours_test.mean():.4f} b5={b5_test.mean():.4f} "
              f"oracle={oracle_test.mean():.4f} | backtest={bt_router.mean():.4f} "
              f"(agree={agree*100:.0f}%) | margin(a={alpha})={mg_router.mean():.4f} "
              f"(->ours {mask_mg.mean()*100:.0f}%, recovers "
              f"{res['margin_router']['recovers_frac_of_oracle_gap']*100:.0f}%)")

    out_path = RUNS_DIR / f"router_analysis_{TAG}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    np.savez(RUNS_DIR / f"router_perfunc_{TAG}.npz", **per_func)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
