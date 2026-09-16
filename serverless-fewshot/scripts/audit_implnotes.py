#!/usr/bin/env python3
"""Implementation-notes audit (2026-08-02): controlled measurements backing the
supplementary section "Implementation Notes: Deployed-Code Deviations".

Two questions, both about the steady-state serving variant `rates_a5`
(scripts/phase6_des.py):

  (a) Blend share: how much do the arm's prewarm decisions change when the
      0.6/0.4 EWMA blend is removed (pure ridge), and where does the blended
      arm sit between its two components?
  (b) Current-tick input: the design row concatenates the raw feature row of
      tick t (channel 0 = log1p of the current-tick count). How much of the
      served rate is attributable to it? (Zero the channel, keep the body
      input unchanged, compare decisions and lag correlations.)

Fixed subset for determinism: 40 functions of the 2021 S3 temporal hold-out
(rng seed 0), first 4,000 evaluation ticks, deployed checkpoint
best_anil_ridge_s1_s0.pt -- the same checkpoint every 2021 steady-state
campaign loads.

Output: results/runs/audit_implnotes.json
"""

import inspect
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

import scripts.phase6_des as p6  # noqa: E402
from scripts.phase6_des import (  # noqa: E402
    decisions_from_rates, rates_a5, rates_ewma,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
RUNS = PROJECT_ROOT / "results" / "runs"
CKPT = PROJECT_ROOT / "results_azure2021" / "runs" / "best_anil_ridge_s1_s0.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FUNCS, N_TICKS, RHOS = 40, 4000, [1.0, 10.0, 100.0]


def variant(replacement):
    """Return rates_a5 with one source line replaced (audit-only patching)."""
    src = inspect.getsource(rates_a5)
    old, new = replacement
    assert old in src, f"anchor gone: {old}"
    ns = {}
    exec(compile(src.replace(old, new), "<audit>", "exec"), p6.__dict__, ns)
    return ns["rates_a5"]


def corr(a, b):
    a, b = a[:, 61:].ravel(), b[:, 61:].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def main():
    features = np.load(PROCESSED / "features.npy")
    counts_all = np.load(PROCESSED / "counts.npy")
    splits = np.load(PROCESSED / "splits.npz")
    idx = splits["s3_test"]
    t0 = int(splits.get("s3_test_t_start", [10080])[0])
    counts = counts_all[idx][:, t0:]
    feats = features[idx][:, t0:]
    rng = np.random.default_rng(0)
    fsel = rng.choice(len(idx), size=min(N_FUNCS, len(idx)), replace=False)
    counts = np.ascontiguousarray(counts[fsel][:, :N_TICKS])
    feats = np.ascontiguousarray(feats[fsel][:, :N_TICKS])

    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge", in_features=features.shape[2],
        embedding_dim=64, n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE,
    )
    trainer.load(CKPT)

    rates = {
        "blend": rates_a5(counts, feats, trainer, DEVICE),
        "ridge_only": variant((
            "blended = 0.6 * ridge_pred + 0.4 * ewma",
            "blended = ridge_pred"))(counts, feats, trainer, DEVICE),
        "ch0_zeroed": variant((
            "raw = features_t[:, min(t, T - 1), :]",
            "raw = features_t[:, min(t, T - 1), :].clone(); raw[:, 0] = 0.0"
        ))(counts, feats, trainer, DEVICE),
        "ewma": rates_ewma(counts),
    }

    out = {
        "config": {"subset_seed": 0, "n_funcs": int(len(fsel)),
                   "n_ticks": N_TICKS, "split": "S3", "checkpoint": CKPT.name},
        "decision_agreement_vs_blend": {}, "mean_prewarm": {},
        "lag_correlation": {},
    }
    for rho in RHOS:
        tau = newsvendor_quantile(rho)
        pw = {k: decisions_from_rates(v, tau)[0] for k, v in rates.items()}
        out["decision_agreement_vs_blend"][str(rho)] = {
            k: float((pw["blend"] == pw[k]).mean())
            for k in ("ridge_only", "ch0_zeroed", "ewma")}
        out["mean_prewarm"][str(rho)] = {
            k: float(v.mean()) for k, v in pw.items()}
    c = counts.astype(float)
    c_prev = np.roll(c, 1, axis=1)
    for k, v in rates.items():
        out["lag_correlation"][k] = {
            "corr_rate_count_t": corr(v, c),
            "corr_rate_count_t_minus_1": corr(v, c_prev)}

    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / "audit_implnotes.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
