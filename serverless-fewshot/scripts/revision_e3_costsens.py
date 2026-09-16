# -*- coding: utf-8 -*-
"""Revision E3: cost-model sensitivity of the break-even conclusions.

The B1 break-even analysis prices one cold start at 15*rho GB.s (kappa=15,
the registered constant: memory-minutes of one 256MB sandbox). The decision
layer consumes only the cost *ratio* rho (tau* = rho/(1+rho)); the kappa
constant appears solely in the post-hoc accounting C = kappa*rho*cold + wm +
state, which is linear in kappa. Reweighting archived per-function arrays is
therefore exact — no re-simulation (linearity verified by inspection of
revision_b1_breakeven.py; logged in REVISION_LOG.md).

For kappa in {5, 15, 50} GB.s (10x range): recompute, per split x rho,
  - sign and magnitude of aggregate net benefit of gated-v3 vs EWMA
  - fraction of functions for which learning pays
Additionally: critical rho* per split and kappa (zero crossing of net(rho),
log-interpolated) with a 1000-resample function-level bootstrap CI.

Source: results/runs/sim_results_des_revision.json (2021, 10 seeds).
Output: results/runs/revision_e3_costsens.json
"""
import json, sys
import numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED = PROJECT_ROOT/"data"/"processed"; RUNS = PROJECT_ROOT/"results"/"runs"

KAPPAS = [5.0, 15.0, 50.0]
RHOS = [0.1, 1.0, 10.0, 100.0]
N_BOOT = 1000
BOOT_SEED = 123  # registered RANDOM_ARM_SEED reused; no new constants

rev = json.load(open(RUNS/"sim_results_des_revision.json"))['results']
STATE_GBS = {"S1": 8e-6*20160*60, "S2": 8e-6*20160*60, "S3": 8e-6*10080*60}

def func_arrays(split, method, rho):
    rows=[r for r in rev if r['split']==split and r['method']==method
          and abs(r['cost_ratio']-rho)<1e-9]
    if not rows:
        raise KeyError((split, method, rho))
    cold = np.mean([r['func_cold'] for r in rows], axis=0)
    wm   = np.mean([r['func_wm'] for r in rows], axis=0)
    return cold, wm

out = {"kappas": KAPPAS, "n_boot": N_BOOT,
       "note": "net = C(EWMA) - C(gated_v3); positive => learning pays",
       "by_split": {}}
for sp in ["S1","S2","S3"]:
    sp_out = {"by_kappa": {}}
    percol = {}
    for rho in RHOS:
        try:
            cg,wg = func_arrays(sp,'A5_gated_v3',rho)
            ce,we = func_arrays(sp,'B4a_ewma',rho)
        except KeyError:
            continue
        percol[rho] = (cg,wg,ce,we)
    for kappa in KAPPAS:
        krec = {}
        nets = []
        for rho,(cg,wg,ce,we) in percol.items():
            net_f = (kappa*rho*ce + we) - (kappa*rho*cg + wg + STATE_GBS[sp]/len(cg))
            krec[str(rho)] = {
                "net_gbs_total": float(net_f.sum()),
                "sign": "pays" if net_f.sum()>0 else "does_not_pay",
                "frac_functions_pay": float((net_f>0).mean()),
            }
            nets.append((rho, net_f))
        # critical rho*: zero crossing of total net over log-rho grid
        rhos_g = np.array([r for r,_ in nets]); tot = np.array([nf.sum() for _,nf in nets])
        def crossing(totv):
            s = np.sign(totv)
            for i in range(len(s)-1):
                if s[i] != s[i+1] and s[i] != 0:
                    lr = np.log10(rhos_g)
                    x = lr[i] + (lr[i+1]-lr[i])*(0-totv[i])/(totv[i+1]-totv[i])
                    return 10**x
            return None
        rho_star = crossing(tot)
        rng = np.random.default_rng(BOOT_SEED)
        n = len(nets[0][1])
        stars = []
        for _ in range(N_BOOT):
            bidx = rng.integers(0, n, n)
            c = crossing(np.array([nf[bidx].sum() for _,nf in nets]))
            stars.append(c)
        stars_v = np.array([s for s in stars if s is not None])
        krec["rho_star"] = {
            "point": rho_star,
            "boot_frac_no_crossing": float(np.mean([s is None for s in stars])),
            "ci95": ([float(np.percentile(stars_v,2.5)),
                      float(np.percentile(stars_v,97.5))]
                     if len(stars_v)>=50 else None),
        }
        sp_out["by_kappa"][str(kappa)] = krec
        signs = {r: krec[str(r)]["sign"] for r in percol}
        print(f"{sp} kappa={kappa:4.0f}: " +
              " ".join(f"rho={r}:{s}({krec[str(r)]['net_gbs_total']:+.2e})"
                       for r,s in signs.items()) +
              f" | rho*={rho_star}")
    out["by_split"][sp] = sp_out

path = RUNS/"revision_e3_costsens.json"
json.dump(out, open(path,"w"), default=float)
blob=open(path,"rb").read(); assert blob and blob.count(0)==0; json.load(open(path))
print("Saved + verified", path)
