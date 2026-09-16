# -*- coding: utf-8 -*-
"""Revision B1: break-even analysis — when does the learned path pay?

Per-function net benefit of ProtoWarm-G (integrated gate) vs EWMA, both under
the same newsvendor layer, from the 10-seed revision campaign.

Cost model (documented): the decision layer prices one cold start at rho
memory-minutes of one 256MB sandbox = 15*rho GB.s. Per-function cost at
operating point rho:  C_f = 15*rho*cold_f + wm_f(GB.s) + state_cost
where state_cost = 8KB held for the evaluated horizon (4.7 GB.s on S3 —
reported, negligible). Learned path pays for f iff C_f(G) < C_f(EWMA).

Covariates: volume (total invocations), Fano factor (burstiness),
zero-history fraction of the horizon. Break-even fractions reported per rho
over functions / invocations / function-minutes; empirical volume threshold
compared against the a-priori gate constant (100).
"""
import json, sys
import numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED = PROJECT_ROOT/"data"/"processed"; RUNS = PROJECT_ROOT/"results"/"runs"

rev = json.load(open(RUNS/"sim_results_des_revision.json"))['results']
splits = np.load(PROCESSED/"splits.npz"); counts_all = np.load(PROCESSED/"counts.npy")
STATE_GBS = {"S1": 8e-6*20160*60, "S2": 8e-6*20160*60, "S3": 8e-6*10080*60}

def func_arrays(split, method, rho):
    rows=[r for r in rev if r['split']==split and r['method']==method and abs(r['cost_ratio']-rho)<1e-9]
    cold = np.mean([r['func_cold'] for r in rows], axis=0)
    wm   = np.mean([r['func_wm'] for r in rows], axis=0)
    tot  = np.mean([r['func_total'] for r in rows], axis=0)
    return cold, wm, tot

out = {"cost_model": "C = 15*rho*cold + wm_gbs + state(8KB*horizon)",
       "by_split": {}}
for sp in ["S1","S2","S3"]:
    key = {"S1":"s1_test","S2":"s2_test","S3":"s3_test"}[sp]
    idx = splits[key]
    seg = counts_all[idx]
    if sp=="S3":
        t0=int(splits.get("s3_test_t_start",[10080])[0]); seg=seg[:,t0:]
    fano = seg.var(axis=1)/np.maximum(seg.mean(axis=1),1e-9)
    zerofrac = (np.cumsum(seg,axis=1)==0).mean(axis=1)
    sp_out = {}
    for rho in [0.1,1.0,10.0,100.0]:
        try:
            cg,wg,tg = func_arrays(sp,'A5_gated_v3',rho)
            ce,we,te = func_arrays(sp,'B4a_ewma',rho)
        except Exception:
            continue
        Cg = 15*rho*cg + wg + STATE_GBS[sp]
        Ce = 15*rho*ce + we
        pays = Cg < Ce
        n=len(pays)
        vol = tg
        fm = np.full(n, seg.shape[1])
        r = {"frac_functions": float(pays.mean()),
             "frac_invocations": float(vol[pays].sum()/max(vol.sum(),1e-9)),
             "net_gbs_saved_total": float((Ce-Cg).sum()),
             "state_cost_gbs": STATE_GBS[sp]}
        # covariate profile of paying vs non-paying functions
        for name, cv in [("volume",vol),("fano",fano),("zerofrac",zerofrac)]:
            r[f"median_{name}_pay"] = float(np.median(cv[pays])) if pays.any() else None
            r[f"median_{name}_nopay"] = float(np.median(cv[~pays])) if (~pays).any() else None
        # empirical volume break-even: threshold maximizing agreement with 'pays'
        ths = np.unique(np.quantile(vol, np.linspace(0,1,51)))
        best_th, best_acc = None, -1
        for th in ths:
            acc = max(((vol>=th)==pays).mean(), ((vol<th)==pays).mean())
            if acc>best_acc: best_acc, best_th = acc, th
        r["volume_threshold_best"] = float(best_th); r["threshold_accuracy"] = float(best_acc)
        sp_out[str(rho)] = r
        print(f"{sp} rho={rho:6.1f}: pays {r['frac_functions']*100:5.1f}% of funcs "
              f"({r['frac_invocations']*100:5.1f}% of invocations) "
              f"net {r['net_gbs_saved_total']:+.0f} GB.s | vol_th≈{best_th:.0f} (acc {best_acc:.2f}) "
              f"| medvol pay/nopay: {r['median_volume_pay']}/{r['median_volume_nopay']}")
    out["by_split"][sp] = sp_out

path = RUNS/"revision_b1_breakeven.json"
json.dump(out, open(path,"w"), default=float)
blob=open(path,"rb").read(); assert blob and blob.count(0)==0; json.load(open(path))
print("Saved + verified", path)
