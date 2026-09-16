"""Exact finite-grid selection using nonnegative partial-sum bounds.

Every candidate is either fully evaluated or proved unable to improve ANY
registered budget. No test data or approximated simulation enters selection.
"""
import argparse
import time

import numpy as np

from common import BUDGETS,OUT,RHOS,load_cohort,read_json,require_tests,save_npz,write_json
from policies import actions,native_baselines
from simulator import certificate,reuse_certificate,simulate
from study import action_hash,get_tape,priors


def can_improve(cold_bound,wm_bound,budgets,best):
    # Strict comparisons retain ties for the registered WM/ID tie break.
    return bool(np.any((wm_bound<=budgets+1e-7)&(cold_bound<=best+1e-7)))


def main(requested_rho=None):
    require_tests()
    z=load_cohort("azure_calibration")
    configs=read_json(OUT/"candidate_configs.json")
    fleet,profile=priors()
    lower=np.zeros((3,len(configs),5,8))
    remaining=[]
    for f in range(len(z["counts"])):
        path=OUT/"screen"/f"f{f:04d}.npz"
        if path.exists():
            lower+=np.load(path)["metrics"]
        else:
            if z["counts"][f].sum()<=10000:
                raise RuntimeError("Cheap screening incomplete")
            remaining.append(f)
    reference=np.load(OUT/"evaluation/azure_calibration/native_rho10/component.npz")["metrics"]
    cold_order=reference[:,:,:4,1].sum(2).mean(1)
    remaining.sort(key=lambda f:(-cold_order[f],f))
    print("exact screening deferred",len(remaining),"functions",flush=True)
    tapes={}
    memo={}
    certs={}
    unavoidable={}
    for f in remaining:
        tapes[f]=[get_tape("azure_calibration",z,f,s) for s in range(5)]
        memo[f]={}; certs[f]={}
        unavoidable[f]=np.mean([np.count_nonzero(t[1]<min(float(np.min(t[4]+np.arange(240)[:,None]*60.)),float(np.min(t[1]+t[3]))))
                                for t in tapes[f]])
        cache=OUT/"screen_cache"/f"f{f:04d}.npz"
        if cache.exists():
            data=np.load(cache)
            memo[f]=dict(zip(data["keys"].tolist(),data["values"]))
    lower_cold_tail=sum(unavoidable.values())
    all_scores=np.full_like(lower,np.nan)
    proofs=[]
    for ri,rho in enumerate(RHOS):
        if requested_rho is not None and rho!=requested_rho:
            continue
        score_path=OUT/f"calibration_scores_rho{rho:g}.npz"
        if score_path.exists():
            all_scores[ri]=np.load(score_path)["metrics"]
            continue
        cache_dir=OUT/"screen_cache"/f"rho{rho:g}"
        for f in remaining:
            cache=cache_dir/f"f{f:04d}.npz"
            if cache.exists():
                data=np.load(cache)
                memo[f].update(zip(data["keys"].tolist(),data["values"]))
        budgets=[]
        for arm in ("component","gate"):
            p=OUT/"evaluation/azure_calibration"/f"native_rho{rho:g}"/(arm+".npz")
            while not p.exists():
                print("waiting for frozen native calibration",p.name,rho,flush=True)
                time.sleep(10)
            m=np.load(p)["metrics"]
            wm=float(m[:,:,:4,2].sum(axis=2).mean(axis=1).sum())
            budgets.extend(wm*np.array(BUDGETS))
        budgets=np.array(budgets)
        best=np.full(len(budgets),np.inf)
        base=native_baselines(z["counts"],rho,fleet,profile)
        action_cache={}
        full={}
        completed=OUT/"screen_complete"/f"rho{rho:g}"
        completed.mkdir(parents=True,exist_ok=True)

        def update(ci,m):
            cold=round(5*m[:4,1].sum())/5; wm=m[:4,2].sum()
            best[wm<=budgets]=np.minimum(best[wm<=budgets],cold)
            full[ci]=m
            all_scores[ri,ci]=m

        def evaluate(ci,prunable=False):
            config=configs[ci]
            dest=completed/(config["id"]+".npz")
            if dest.exists():
                update(ci,np.load(dest)["metrics"])
                return True
            m=lower[ri,ci].copy()
            q,ttl=actions(config,z["counts"],rho,fleet,base,action_cache)
            missing_cold=lower_cold_tail
            for f in remaining:
                if prunable and not can_improve(m[:4,1].sum()+missing_cold,m[:4,2].sum(),budgets,best):
                    proofs.append({"rho":rho,"config":config["id"],"cold_lower_bound":float(m[:4,1].sum()+missing_cold),
                                   "wm_lower_bound":float(m[:4,2].sum())})
                    return False
                key=action_hash(q[f],ttl[f])
                if key not in memo[f]:
                    initial=int(q[f,0])
                    if initial not in certs[f]:
                        certs[f][initial]=[certificate(t,initial) for t in tapes[f]]
                    values=[]
                    for si,t in enumerate(tapes[f]):
                        value=reuse_certificate(certs[f][initial][si],q[f],ttl[f])
                        if value is None:
                            value=simulate(*t,q[f].astype(np.int64),ttl[f].astype(float))
                        values.append(value)
                    memo[f][key]=np.mean(values,axis=0)
                    if memo[f][key][:4,1].sum()+1e-8<unavoidable[f]:
                        raise AssertionError("First-readiness lower bound violated")
                m+=memo[f][key]
                missing_cold-=unavoidable[f]
            update(ci,m)
            save_npz(dest,metrics=m)
            return True

        # Obtain valid upper bounds from real, completely executed candidates.
        totals=lower[ri,:,:4].sum(axis=1)
        anchors=set(range(6))
        for budget in budgets:
            feasible=np.flatnonzero(totals[:,2]<=budget)
            anchors.update(sorted(feasible,key=lambda i:(totals[i,1],totals[i,2],configs[i]["id"]))[:2])
        for ci in sorted(anchors):
            evaluate(int(ci))
            print("anchor",rho,configs[ci]["id"],flush=True)
        order=sorted(range(len(configs)),key=lambda i:(totals[i,1],totals[i,2],configs[i]["id"]))
        for n,ci in enumerate(order):
            if ci not in full:
                evaluate(ci,True)
            if (n+1)%100==0:
                print("bounded grid",rho,n+1,"/",len(order),"full",len(full),"pruned",len(proofs),flush=True)
                for f in remaining:
                    save_npz(cache_dir/f"f{f:04d}.npz",keys=np.array(list(memo[f])),values=np.stack(list(memo[f].values())))
        for proof in (p for p in proofs if p["rho"]==rho):
            if can_improve(proof["cold_lower_bound"],proof["wm_lower_bound"],budgets,best):
                raise AssertionError("Invalid pruning certificate")
        write_json(OUT/f"screen_proofs_rho{rho:g}.json",{"budgets":budgets.tolist(),"best_cold":[float(x) if np.isfinite(x) else None for x in best],
                   "fully_evaluated":len(full),"total_configs":len(configs),
                   "proofs":[p for p in proofs if p["rho"]==rho]})
        save_npz(score_path,metrics=all_scores[ri])
    if requested_rho is not None:
        print("completed rho",requested_rho,flush=True)
        return
    save_npz(OUT/"calibration_scores.npz",metrics=all_scores)
    write_json(OUT/"screen_optimization.json",{"method":"exact nonnegative lower bounds, not subsampling",
               "deferred_functions":remaining,"shared_first_readiness_cold_bound":lower_cold_tail,
               "deferred_order":"descending native calibration cold contribution; exact bounds unchanged",
               "all_pruned_certificates_verified":True,"evaluated_mask":np.isfinite(all_scores[:,:,0,0]).tolist()})


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--rho",type=float,choices=RHOS)
    main(ap.parse_args().rho)
