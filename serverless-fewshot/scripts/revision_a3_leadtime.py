# -*- coding: utf-8 -*-
"""Revision A3: provisioning lead-time sensitivity on S1.
Cold-init medians {~0s, 1.5s, 5s} (lognormal mu=ln(m), sigma kept at 0.31;
the ~0s arm uses mu=ln(1e-3), sigma=0.01 = effectively instant warmth).
Methods: A5_full_system, B4a_ewma, B1, Oracle; rhos {1,10}; seeds 0..4.
Output: results/runs/revision_a3_leadtime.json
"""
import json, sys
import numpy as np, pandas as pd, torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.phase6_des import (rates_a5, rates_ewma, rates_oracle, policy_b1,
                                run_config, decisions_from_rates)
from src.decision.newsvendor import newsvendor_quantile
from src.models.heads import N_QUANTILES
from src.meta.trainer import ANILMetaTrainer

PROCESSED = PROJECT_ROOT/"data"/"processed"; RUNS = PROJECT_ROOT/"results"/"runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INITS = {"instant": {"mu": float(np.log(1e-3)), "sigma": 0.01},
         "1.5s":    {"mu": float(np.log(1.5)),  "sigma": 0.31},
         "5s":      {"mu": float(np.log(5.0)),  "sigma": 0.31}}
SEEDS = list(range(5)); RHOS = [1.0, 10.0]

features = np.load(PROCESSED/"features.npy"); counts_all = np.load(PROCESSED/"counts.npy")
splits = np.load(PROCESSED/"splits.npz"); dur_df = pd.read_csv(PROCESSED/"duration_stats.csv")
trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
    in_features=features.shape[2], embedding_dim=64,
    n_quantiles=N_QUANTILES, n_horizons=1, device=DEVICE)
trainer.load(RUNS/"best_anil_ridge_s1_s0.pt")
idx = splits["s1_test"]; counts = counts_all[idx]; feats = features[idx]
dm = np.nan_to_num(np.array([dur_df.iloc[i]["dur_mean"] for i in idx]), nan=1.0)
ds = np.nan_to_num(np.array([dur_df.iloc[i]["dur_std"] for i in idx]), nan=0.5)
rmats = {"A5_full_system": rates_a5(counts, feats, trainer, DEVICE),
         "B4a_ewma": rates_ewma(counts), "Oracle": rates_oracle(counts)}
pw_b1, ka_b1 = policy_b1(counts)
jobs = []
for iname, ci in INITS.items():
    for rho in RHOS:
        tau = newsvendor_quantile(rho)
        for m, r in rmats.items():
            pw, ka = decisions_from_rates(r, tau)
            jobs.append((f"{m}|{iname}", rho, SEEDS, counts, pw, ka, dm, ds, ci, "S1"))
        jobs.append((f"B1_fixed_keepalive|{iname}", rho, SEEDS, counts, pw_b1, ka_b1, dm, ds, ci, "S1"))
res = []
with ProcessPoolExecutor(max_workers=5) as ex:
    for rl in ex.map(run_config, jobs):
        res.extend(rl); r = rl[0]
        print(f"{r['method']:28s} rho={r['cost_ratio']:5.1f} CSR={np.mean([x['csr'] for x in rl]):.4f}")
path = RUNS/"revision_a3_leadtime.json"
json.dump({"results": res, "inits": INITS}, open(path,"w"), default=float)
blob = open(path,"rb").read(); assert blob and blob.count(0)==0; json.load(open(path))
print("Saved + verified", path)
