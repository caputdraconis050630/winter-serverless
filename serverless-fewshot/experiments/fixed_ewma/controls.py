"""Recalibrate the original simple-controller family at fixed alpha=0.3.

Reuse is allowed only for byte-identical q/TTL arrays and identical seeds.
Existing representation-only results remain unchanged. No target selection.
"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/onboarding_attribution_v1'
OUT=ROOT/'results/fixed_ewma_v1/controls'
sys.path.insert(0,str(ROOT/'experiments/onboarding_attribution'))
import numpy as np
import common
common.OUT=OUT
common.ALPHAS=[.3]
import policies
original_ewma=policies.ewma


def fixed_ewma(counts,alpha=.3,initial=0.):
    assert alpha==.3, alpha
    return original_ewma(counts,alpha=alpha,initial=initial)


policies.ewma=fixed_ewma
import study
import prune_screen
import analyze
original_run=study.run_actions
original_tape=study.get_tape


def link(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.symlink_to(source.resolve())
    else:assert common.digest(source)==common.digest(dest),dest


def tape(cohort,z,f,seed):
    path=OLD/'events'/cohort/f'f{f:04d}_s{seed}.npz'
    if path.exists():
        with np.load(path) as d:return tuple(d[k] for k in ('ptr','arrival','duration','cold','warm'))
    return original_tape(cohort,z,f,seed)


def run_actions(cohort,actions,seeds,tag,workers=3,legacy=False):
    reused=[]
    for name,(q,ttl) in actions.items():
        src=OLD/'evaluation'/cohort/tag/(name+'.npz')
        dest=OUT/'evaluation'/cohort/tag/(name+'.npz')
        if dest.exists() or not src.exists():continue
        with np.load(src) as z:
            same=(np.array_equal(q,z['q']) and np.array_equal(ttl,z['ttl'])
                  and np.array_equal(seeds,z['seeds']))
        if same:
            link(src,dest);reused.append(name)
    if reused:print('verified action reuse',cohort,tag,len(reused),'arms',flush=True)
    return original_run(cohort,actions,seeds,tag,workers,legacy)


study.get_tape=prune_screen.get_tape=tape
study.run_actions=run_actions


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('cohorts','simple_priors.json','tests.json','simulation_contract.json',
                 'representation_selection.json'):
        source=OLD/name
        if source.is_dir():
            for f in source.glob('*'):link(f,OUT/name/f.name)
        else:link(source,OUT/name)
    # A changed native gate changes its validation memory budget.
    for model in sorted((OLD/'models').iterdir()):
        if not model.is_dir():continue
        for src in model.iterdir():
            dest=OUT/'models'/model.name/src.name
            if model.name.startswith('trained') and src.name.endswith('_rates.npz'):
                cohort=src.name.removesuffix('_rates.npz')
                with np.load(src) as z:d={k:z[k] for k in z.files}
                if 'gate' not in d:
                    link(src,dest);continue
                counts=common.load_cohort(cohort)['counts']
                history=np.zeros_like(counts,dtype=np.int64)
                history[:,1:]=np.cumsum(counts[:,:-1],axis=1,dtype=np.int64)
                mask=(history>0)&(history<100)
                before=original_ewma(counts,.1)
                np.testing.assert_array_equal(d['gate'][mask],before[mask])
                after=fixed_ewma(counts)
                d['gate']=np.where(mask,after,d['gate'])
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():
                    with np.load(dest) as z:
                        for key,value in d.items():np.testing.assert_array_equal(z[key],value)
                else:common.save_npz(dest,**d)
            else:link(src,dest)
    protocol=common.read_json(OLD/'protocol.json')
    protocol.update(version='fixed-ewma-v1-controls',ewma_alphas=[.3],
        freeze_date_utc='2026-09-16',source_protocol_sha256=common.digest(OLD/'protocol.json'),
        revision='Fixed .3 in all EWMA families, native fallback and all model gates; reselect budgets on validation only',
        wrapper_sha256=common.digest(__file__))
    if (OUT/'protocol.json').exists():assert common.read_json(OUT/'protocol.json')==protocol
    else:common.write_json(OUT/'protocol.json',protocol)
    common.write_json(OUT/'candidate_configs.json',policies.simple_configs())
    print('fixed-alpha control candidates',len(policies.simple_configs()),flush=True)
    # Reuse representation measurements, which have no EWMA dependency.
    for src in (OLD/'evaluation').glob('*/repr_*/*.npz'):
        link(src,OUT/src.relative_to(OLD))


def calibration(workers):
    for rho in (10.,1.,100.):
        run_actions('azure_calibration',study.model_actions('azure_calibration',rho),
                    list(range(5)),f'native_rho{rho:g}',workers)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','calibrate','screen','select','evaluate','analyze'])
    ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='calibrate':calibration(args.workers)
    elif args.stage=='screen':
        study.screen(args.workers,cheap_only=True)
        prune_screen.main()
    elif args.stage=='select':study.select(args.workers)
    elif args.stage=='evaluate':study.evaluate(args.workers)
    elif args.stage=='analyze':
        analyze.summaries();analyze.comparisons()


if __name__=='__main__':main()
