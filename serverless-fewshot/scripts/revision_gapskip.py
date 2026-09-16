# -*- coding: utf-8 -*-
"""Gap-skipping decision layer: kill a sandbox immediately when the policy's
own forecast says it will not survive to the next arrival.

Rule (self-consistent, no tuned threshold): at tick t, if predicted
IAT = 1/rate(t) exceeds the adaptive keep-alive ka(t) the policy would set,
force ka(t) = KA_KILL = 1 min (one tick, the accounting floor). Prewarm
logic unchanged. Applied identically to every rate-based policy.

Sweep mirrors the revision campaign (same rho grid, 10 seeds, S1/S2/S3) so
curves are directly comparable. Output: results/runs/revision_gapskip.json
"""
import json, sys
import numpy as np, pandas as pd, torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.phase6_des import (rates_a5, rates_ewma, rates_oracle, run_config,
                                decisions_from_rates, COLD_INIT, FULL_RHOS, REDUCED_RHOS)
from scripts.phase63_onboarding_drift import build_prototypes, assign_proto_W
from scripts.revision_a2_des import rates_proto_prefix, LOW_RHOS, MIN_INV
from src.decision.newsvendor import newsvendor_quantile
from src.models.heads import N_QUANTILES
from src.meta.trainer import ANILMetaTrainer

PROCESSED = PROJECT_ROOT/"data"/"processed"; RUNS = PROJECT_ROOT/"results"/"runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KA_KILL = 1.0
SEEDS = list(range(10))

def decisions_gapskip(rates, tau):
    pw, ka = decisions_from_rates(rates, tau)
    with np.errstate(divide="ignore"):
        iat = np.where(rates > 1e-9, 1.0/np.maximum(rates, 1e-9), np.inf)
    kill = iat > ka
    ka = ka.copy(); ka[kill] = KA_KILL
    return pw, ka, float(kill.mean())

def main():
    features = np.load(PROCESSED/"features.npy"); counts_all = np.load(PROCESSED/"counts.npy")
    splits = np.load(PROCESSED/"splits.npz"); dur_df = pd.read_csv(PROCESSED/"duration_stats.csv")
    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
        in_features=features.shape[2], embedding_dim=64,
        n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE)
    trainer.load(RUNS/"best_anil_ridge_s1_s0.pt")
    pm = build_prototypes(trainer, features, counts_all, splits["s1_train"])

    all_results = []; killstats = {}
    for split in ["S1","S2","S3"]:
        key = {"S1":"s1_test","S2":"s2_test","S3":"s3_test"}[split]
        idx = splits[key]; counts = counts_all[idx]; feats = features[idx]
        if split=="S3":
            t0=int(splits.get("s3_test_t_start",[10080])[0]); counts=counts[:,t0:]; feats=feats[:,t0:]
        dm = np.nan_to_num(np.array([dur_df.iloc[i]["dur_mean"] if i < len(dur_df) else 1.0 for i in idx]), nan=1.0)
        ds = np.nan_to_num(np.array([dur_df.iloc[i]["dur_std"] if i < len(dur_df) else 0.5 for i in idx]), nan=0.5)
        print(f"### {split}: rates...")
        a5 = rates_a5(counts, feats, trainer, DEVICE)
        ew = rates_ewma(counts); orc = rates_oracle(counts)
        proto, cum_prev = rates_proto_prefix(counts, feats, trainer, pm)
        zero = cum_prev==0; sparse=(cum_prev>0)&(cum_prev<MIN_INV)
        gated = np.where(zero, proto, np.where(sparse, ew, a5)).astype(np.float32)
        rmats = {"A5_gated_v3": gated, "A5_full_system": a5, "B4a_ewma": ew, "Oracle": orc}
        rhos = (FULL_RHOS if split=="S1" else REDUCED_RHOS) + LOW_RHOS
        jobs = []
        for rho in rhos:
            tau = newsvendor_quantile(rho)
            for m, r in rmats.items():
                pw, ka, kf = decisions_gapskip(r, tau)
                killstats[f"{split}|{m}|{rho}"] = kf
                jobs.append((m+"_gapskip", rho, SEEDS, counts, pw, ka, dm, ds, COLD_INIT, split))
        with ProcessPoolExecutor(max_workers=5) as ex:
            for rl in ex.map(run_config, jobs):
                all_results.extend(rl); r = rl[0]
                print(f"  {r['method']:24s} rho={r['cost_ratio']:6.2f} "
                      f"CSR={np.mean([x['csr'] for x in rl]):.4f} "
                      f"WM={np.mean([x['wm_per_1k_inv'] for x in rl]):.0f}")
    path = RUNS/"revision_gapskip.json"
    json.dump({"results": all_results, "ka_kill": KA_KILL, "kill_frac": killstats},
              open(path,"w"), default=float)
    blob=open(path,"rb").read(); assert blob and blob.count(0)==0; json.load(open(path))
    print("Saved + verified", path)

if __name__ == "__main__":
    main()
