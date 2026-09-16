# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Revision A2(b) + B5 + B6(seeds): 2021 steady-state DES re-campaign.

A2(b): integrated three-way gate (A5_gated_v3):
         cum_prev == 0            -> prototype zero-shot rate (new-function branch)
         0 < cum_prev < 100       -> EWMA rate            (sparse-history branch)
         cum_prev >= 100          -> online-adapted A5    (learned branch)
       vs. the v1 gate evaluated in the paper (no prototype branch).
B5   : rho extended DOWN with LOW_RHOS so the gated curve reaches keep-alive's
       measured WM point (matched-memory comparison by measurement).
B6   : DES seeds 0..9 (10 seeds) for the headline methods.

New a-priori constants (Table 1): LOW_RHOS=[0.01,0.02,0.05]; SEEDS=range(10);
prototype point-rate = median predicted quantile (tau=0.5). Fixed before eval.

Methods re-run at 10 seeds: A5_gated_v3, A5_full_system, B4a_ewma,
B1_fixed_keepalive, Oracle. (B2/B5-global keep their published 3-seed rows.)
Output: results/runs/sim_results_des_revision.json (+ routing stats)
"""

import json
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
    rates_a5, rates_ewma, rates_oracle, policy_b1, run_config,
    decisions_from_rates, COLD_INIT, FULL_RHOS, REDUCED_RHOS,
)
from scripts.phase63_onboarding_drift import (  # noqa: E402
    build_prototypes, assign_proto_W,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402
from src.meta.trainer import ANILMetaTrainer  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L = 60
MIN_INV = 100
LOW_RHOS = [0.01, 0.02, 0.05]
SEEDS = list(range(10))
MED_IDX = N_QUANTILES // 2      # tau = 0.5


def rates_proto_prefix(counts, feats, trainer, pm, batch=1024):
    """Prototype zero-shot point rates, computed only on the zero-history
    prefix of each function (cum_prev == 0)."""
    N, T = counts.shape
    cum = np.cumsum(counts, axis=1)
    cum_prev = np.concatenate([np.zeros((N, 1)), cum[:, :-1]], axis=1)
    pairs = np.argwhere(cum_prev == 0)
    out = np.zeros((N, T), dtype=np.float32)
    feats_t = torch.from_numpy(np.ascontiguousarray(feats)).float()
    nF = feats.shape[2]
    trainer.body.eval()
    print(f"    proto prefix: {len(pairs)} (func,tick) pairs")
    with torch.no_grad():
        for s in range(0, len(pairs), batch):
            chunk = pairs[s:s + batch]
            wins = []
            for f, t in chunk:
                if t < L:
                    pad = torch.zeros(L - t, nF)
                    wins.append(torch.cat([pad, feats_t[f, :t]], dim=0))
                else:
                    wins.append(feats_t[f, t - L:t])
            x = torch.stack(wins).to(DEVICE)
            phi = trainer.body(x).cpu()
            Wp, _ = assign_proto_W(pm, phi)
            pred = (phi.unsqueeze(1) @ Wp.cpu()).squeeze(1)   # [B, O]
            med = pred[:, MED_IDX].numpy()
            out[chunk[:, 0], chunk[:, 1]] = np.expm1(np.maximum(med, 0.0))
    return out, cum_prev


def main():
    features = np.load(PROCESSED_DIR / "features.npy")
    counts_all = np.load(PROCESSED_DIR / "counts.npy")
    splits_data = np.load(PROCESSED_DIR / "splits.npz")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")

    trainer = ANILMetaTrainer(
        body_type="tcn", head_type="ridge",
        in_features=features.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE,
    )
    trainer.load(RUNS_DIR / "best_anil_ridge_s1_s0.pt")
    pm = build_prototypes(trainer, features, counts_all, splits_data["s1_train"])

    all_results, routing = [], {}
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

        print(f"\n### {split}: {len(test_idx)} funcs — computing rates...")
        t_start = time.time()
        a5 = rates_a5(counts, feats, trainer, DEVICE)
        ew = rates_ewma(counts)
        orc = rates_oracle(counts)
        proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
        print(f"    rates done in {time.time()-t_start:.0f}s")

        zero_mask = cum_prev == 0
        sparse_mask = (cum_prev > 0) & (cum_prev < MIN_INV)
        learned_mask = cum_prev >= MIN_INV
        gated = np.where(zero_mask, proto,
                         np.where(sparse_mask, ew, a5)).astype(np.float32)
        routing[split] = {
            "frac_ticks_proto": float(zero_mask.mean()),
            "frac_ticks_ewma": float(sparse_mask.mean()),
            "frac_ticks_learned": float(learned_mask.mean()),
            "min_invocations": MIN_INV,
        }
        print(f"    routing v3: proto={zero_mask.mean()*100:.1f}% "
              f"ewma={sparse_mask.mean()*100:.1f}% "
              f"learned={learned_mask.mean()*100:.1f}% of function-ticks")

        rhos = (FULL_RHOS if split == "S1" else REDUCED_RHOS) + LOW_RHOS
        rate_mats = {"A5_gated_v3": gated, "A5_full_system": a5,
                     "B4a_ewma": ew, "Oracle": orc}
        jobs = []
        for rho in rhos:
            tau = newsvendor_quantile(rho)
            for method, rmat in rate_mats.items():
                pw, ka = decisions_from_rates(rmat, tau)
                jobs.append((method, rho, SEEDS, counts, pw, ka,
                             dm, ds, COLD_INIT, split))
            pw_b1, ka_b1 = policy_b1(counts)
            jobs.append(("B1_fixed_keepalive", rho, SEEDS, counts, pw_b1, ka_b1,
                         dm, ds, COLD_INIT, split))

        with ProcessPoolExecutor(max_workers=5) as ex:
            for res_list in ex.map(run_config, jobs):
                all_results.extend(res_list)
                r = res_list[0]
                print(f"    {r['method']:16s} rho={r['cost_ratio']:6.2f} "
                      f"CSR={np.mean([x['csr'] for x in res_list]):.4f}")

    out_path = RUNS_DIR / "sim_results_des_revision.json"
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "routing_v3": routing,
                   "seeds": SEEDS, "low_rhos": LOW_RHOS}, f, default=float)
    blob = open(out_path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0
    json.load(open(out_path))
    print(f"\nSaved + verified {out_path} ({len(all_results)} rows)")


if __name__ == "__main__":
    main()
