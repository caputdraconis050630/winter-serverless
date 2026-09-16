"""Resumable exhaustive calibration and frozen-policy paired evaluation."""
import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from common import (BUDGETS,LAMBDAS,OUT,RHOS,load_cohort,read_json,require_tests,
                    save_npz,write_json)
from policies import actions,decisions,native_baselines,simple_configs
from simulator import certificate,event_tape,reuse_certificate,simulate


def priors():
    z=read_json(OUT/"simple_priors.json")
    return z["fleet_rate"],np.array(z["age_profile_rates"])


def get_tape(cohort,z,f,seed):
    path=OUT/"events"/cohort/f"f{f:04d}_s{seed}.npz"
    if path.exists():
        with np.load(path) as data:
            return tuple(data[k] for k in ("ptr","arrival","duration","cold","warm"))
    tape=event_tape(z["counts"][f],float(z["duration_mean"][f]),float(z["duration_std"][f]),
                    seed,cohort.split("_")[0]+":"+z["keys"][f],.25,.31)
    save_npz(path,**dict(zip(("ptr","arrival","duration","cold","warm"),tape)))
    return tape


def action_hash(q,ttl):
    h=hashlib.sha256(np.asarray(q,dtype="<i8").tobytes())
    h.update(np.asarray(ttl,dtype="<f8").tobytes())
    return h.hexdigest()


def run_actions(cohort,policies,seeds,tag,workers=3,legacy=False):
    """Persist actual actions and per-function, per-seed, per-window endpoints."""
    z=load_cohort(cohort)
    n=len(z["counts"])
    directory=OUT/"evaluation"/cohort/tag
    directory.mkdir(parents=True,exist_ok=True)
    names=sorted(policies)
    for name in names:
        path=directory/(name+".npz")
        if path.exists():
            with np.load(path) as cached:
                q,ttl=policies[name]
                np.testing.assert_array_equal(cached["q"],q,err_msg=str(path))
                np.testing.assert_array_equal(cached["ttl"],ttl,err_msg=str(path))
                np.testing.assert_array_equal(cached["seeds"],seeds,err_msg=str(path))
    todo=[name for name in names if not (directory/(name+".npz")).exists()]
    outputs={name:np.empty((n,len(seeds),5,8)) for name in todo}
    def worker(f):
        tapes=[get_tape(cohort,z,f,s) for s in seeds]
        memo={}
        certificates={}
        row={}
        for name in todo:
            q,ttl=policies[name]
            key=action_hash(q[f],ttl[f])
            if key not in memo:
                initial=int(q[f,0])
                if not legacy and initial not in certificates and z["counts"][f].sum()>10000:
                    certificates[initial]=[certificate(t,initial) for t in tapes]
                values=[]
                for si,t in enumerate(tapes):
                    value=reuse_certificate(certificates[initial][si],q[f],ttl[f]) if initial in certificates else None
                    if value is None:
                        value=simulate(*t,q[f].astype(np.int64),ttl[f].astype(np.float64),legacy_ttl=legacy)
                    values.append(value)
                memo[key]=np.stack(values)
            row[name]=memo[key]
        return f,row
    if todo:
        print("evaluate",cohort,tag,len(todo),"arms",len(seeds),"seeds",flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(worker,f) for f in range(n)]
            for k,job in enumerate(as_completed(jobs)):
                f,row=job.result()
                for name,value in row.items():
                    outputs[name][f]=value
                if (k+1)%100==0:
                    print("eval",cohort,tag,k+1,"/",n,flush=True)
        for name in todo:
            q,ttl=policies[name]
            save_npz(directory/(name+".npz"),metrics=outputs[name],q=q,ttl=ttl,seeds=np.array(seeds))
    return {name:np.load(directory/(name+".npz"))["metrics"] for name in names}


def screen(workers=3,limit=None,cheap_only=False):
    require_tests()
    cohort="azure_calibration"
    z=load_cohort(cohort)
    fleet,profile=priors()
    configs=simple_configs()
    write_json(OUT/"candidate_configs.json",configs)
    native={rho:native_baselines(z["counts"],rho,fleet,profile) for rho in RHOS}
    directory=OUT/"screen"
    directory.mkdir(parents=True,exist_ok=True)
    n=len(z["counts"])
    def worker(f):
        start=time.monotonic()
        path=directory/f"f{f:04d}.npz"
        if path.exists():
            return f,0.,0
        tapes=[get_tape(cohort,z,f,s) for s in range(5)]
        out=np.empty((len(RHOS),len(configs),5,8))
        memo={}
        certificates={}
        for ri,rho in enumerate(RHOS):
            base={name:(q[f:f+1],ttl[f:f+1]) for name,(q,ttl) in native[rho].items()}
            cache={}
            for ci,config in enumerate(configs):
                q,ttl=actions(config,z["counts"][f:f+1],rho,fleet,base,cache)
                key=action_hash(q[0],ttl[0])
                if key not in memo:
                    initial=int(q[0,0])
                    if initial not in certificates and z["counts"][f].sum()>10000:
                        certificates[initial]=[certificate(t,initial) for t in tapes]
                    values=[]
                    for si,t in enumerate(tapes):
                        value=reuse_certificate(certificates[initial][si],q[0],ttl[0]) if initial in certificates else None
                        if value is None:
                            value=simulate(*t,q[0].astype(np.int64),ttl[0].astype(np.float64))
                        values.append(value)
                    memo[key]=np.mean(values,axis=0)
                out[ri,ci]=memo[key]
        save_npz(path,metrics=out,unique_actions=np.array(len(memo)))
        return f,time.monotonic()-start,len(memo)
    order=list(range(n))
    if cheap_only:
        order=[f for f in order if z["counts"][f].sum()<=10000]
    if limit is not None:
        order=order[:limit]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(worker,f) for f in order]
        for k,job in enumerate(as_completed(jobs)):
            f,elapsed,unique=job.result()
            print("screen",k+1,"/",len(order),"f",f,"seconds",round(elapsed,2),"unique",unique,flush=True)
    if len(order)!=n:
        return
    total=np.zeros((3,len(configs),5,8))
    for f in range(n):
        total+=np.load(directory/f"f{f:04d}.npz")["metrics"]
    save_npz(OUT/"calibration_scores.npz",metrics=total)


def model_actions(cohort,rho,model="trained0"):
    z=np.load(OUT/"models"/model/(cohort+"_rates.npz"))
    return {name:decisions(z[name],rho) for name in ("component","gate","no_prior","prototype_only") if name in z}


def calibrate_models(workers=3):
    require_tests()
    for rho in (10.,1.,100.):
        run_actions("azure_calibration",model_actions("azure_calibration",rho),list(range(5)),f"native_rho{rho:g}",workers)
        for model in ("trained0","random0","random1","random2","random3","random4"):
            rates=np.load(OUT/"models"/model/"azure_calibration_rates.npz")["lambda_rates"]
            policies={f"lambda{i}":decisions(rates[i],rho) for i in range(len(LAMBDAS))}
            run_actions("azure_calibration",policies,list(range(5)),f"repr_{model}_rho{rho:g}",workers)


def select(workers=3,only_rho=None):
    require_tests()
    configs=read_json(OUT/"candidate_configs.json")
    selection={}
    for ri,rho in enumerate(RHOS):
        if only_rho is not None and rho!=only_rho:
            continue
        shard=OUT/f"calibration_scores_rho{rho:g}.npz"
        raw=np.load(shard)["metrics"] if shard.exists() else np.load(OUT/"calibration_scores.npz")["metrics"][ri]
        scores=raw[:,:4].sum(axis=1)
        models=model_actions("azure_calibration",rho)
        result=run_actions("azure_calibration",models,list(range(5)),f"native_rho{rho:g}",workers)
        row={}
        for arm in ("component","gate"):
            reference=float(result[arm][:,:,:4,2].sum(axis=2).mean(axis=1).sum())
            choices=[]
            for multiplier in BUDGETS:
                limit=reference*multiplier
                eligible=[i for i in range(len(configs)) if scores[i,2]<=limit]
                best=min(eligible,key=lambda i:(round(5*scores[i,1]),scores[i,2],configs[i]["id"])) if eligible else None
                choices.append({"multiplier":multiplier,"wm_limit":limit,
                                "config":configs[best] if best is not None else None,
                                "metrics":scores[best].tolist() if best is not None else None})
            row[arm]={"wm_reference":reference,"choices":choices}
        selection[str(rho)]=row
    destination=OUT/("selection.json" if only_rho is None else f"selection_rho{only_rho:g}.json")
    if only_rho is None:
        for rho in RHOS:
            previous=OUT/f"selection_rho{rho:g}.json"
            if previous.exists() and read_json(previous)!={str(rho):selection[str(rho)]}:
                raise RuntimeError("Full selection differs from frozen rho shard")
    if destination.exists() and read_json(destination)!=selection:
        raise RuntimeError("Refusing to change frozen selected policies")
    write_json(destination,selection)


def evaluate(workers=3,core_only=False,only_rho=None):
    require_tests()
    selection_file="selection.json" if only_rho is None else f"selection_rho{only_rho:g}.json"
    selected=read_json(OUT/selection_file) if not core_only else None
    fleet,profile=priors()
    for cohort in ("azure_primary","azure_evaluation","huawei"):
        z=load_cohort(cohort)
        for rho in (10.,1.,100.):
            if only_rho is not None and rho!=only_rho:
                continue
            native=native_baselines(z["counts"],rho,fleet,profile)
            policies=model_actions(cohort,rho)
            policies.update(native)
            cache={}
            for ref in ("component","gate"):
                for choice in selected[str(rho)][ref]["choices"] if selected else []:
                    config=choice["config"]
                    if config is not None:
                        policies[config["id"]]=actions(config,z["counts"],rho,fleet,native,cache)
            qe,te=native["ewma"]
            for arm in ("component","gate"):
                qw,tw=policies[arm]
                policies[arm+"_q_ewma_ttl"]=(qw,te)
                policies[arm+"_ttl_ewma_q"]=(qe,tw)
            run_actions(cohort,policies,list(range(1000,1020)),f"main_rho{rho:g}",workers)
            strict={}
            for name,(q,ttl) in policies.items():
                q=q.copy(); ttl=ttl.copy()
                q[:,0]=0; ttl[:,0]=10.
                strict[name]=(q,ttl)
            run_actions(cohort,strict,list(range(1000,1020)),f"strict_rho{rho:g}",workers)
            for seed in (1,2):
                extra=model_actions(cohort,rho,f"trained{seed}")
                run_actions(cohort,extra,list(range(1000,1020)),f"model{seed}_rho{rho:g}",workers)


def representations(workers=3):
    """Select each representation's lambda only on calibration endpoints."""
    output={}
    for rho in (10.,1.,100.):
        reference=np.load(OUT/"evaluation/azure_calibration"/f"native_rho{rho:g}"/"component.npz")["metrics"]
        reference_wm=float(reference[:,:,:4,2].sum(axis=2).mean(axis=1).sum())
        row={}
        for model in ("trained0","random0","random1","random2","random3","random4"):
            ratefile=OUT/"models"/model/"azure_calibration_rates.npz"
            rates=np.load(ratefile)["lambda_rates"]
            policies={f"lambda{i}":decisions(rates[i],rho) for i in range(len(LAMBDAS))}
            result=run_actions("azure_calibration",policies,list(range(5)),f"repr_{model}_rho{rho:g}",workers)
            totals={name:metrics[:,:,:4].sum(axis=2).mean(axis=1).sum(axis=0) for name,metrics in result.items()}
            choices=[]
            for multiplier in BUDGETS:
                budget=reference_wm*multiplier
                eligible=[name for name,m in totals.items() if m[2]<=budget]
                best=min(eligible,key=lambda name:(round(5*totals[name][1]),totals[name][2],name)) if eligible else None
                choices.append({"multiplier":multiplier,"lambda_index":int(best[6:]) if best else None,
                                "metrics":totals[best].tolist() if best else None})
            row[model]=choices
        output[str(rho)]=row
    destination=OUT/"representation_selection.json"
    if destination.exists() and read_json(destination)!=output:
        raise RuntimeError("Refusing to change frozen representation selection")
    write_json(destination,output)
    for rho in (10.,1.,100.):
        for model,choices in output[str(rho)].items():
            indices=sorted({c["lambda_index"] for c in choices if c["lambda_index"] is not None})
            for cohort in ("azure_primary","azure_evaluation","huawei"):
                rates=np.load(OUT/"models"/model/(cohort+"_rates.npz"))["lambda_rates"]
                policies={f"lambda{i}":decisions(rates[i],rho) for i in indices}
                if policies:
                    run_actions(cohort,policies,list(range(1000,1020)),f"repr_{model}_rho{rho:g}",workers)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=("screen","calibrate_models","select","evaluate","representations"))
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--limit",type=int)
    ap.add_argument("--cheap-only",action="store_true")
    ap.add_argument("--core-only",action="store_true")
    ap.add_argument("--rho",type=float,choices=RHOS)
    args=ap.parse_args()
    if args.stage=="screen":
        screen(args.workers,args.limit,args.cheap_only)
    elif args.stage=="evaluate":
        evaluate(args.workers,args.core_only,args.rho)
    elif args.stage=="select":
        select(args.workers,args.rho)
    else:
        globals()[args.stage](args.workers)
