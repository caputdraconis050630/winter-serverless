"""Compare saved WINTER-G/LSTM CSR and modeled cost across all existing cases.

Read-only with respect to manuscript and experiment data. No new policy,
training, simulation, parameter selection, or change to existing windows.
2,000 exploratory paired cluster-bootstrap replicates, conditional on retained
models/cohorts; marginal intervals are not adjusted across this full grid.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT.parent/'serverless-fewshot/results/lstm_comparison_v1/cases'
OUT = Path(__file__).with_suffix('.json')
SEED = 260915
REPS = 2000


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def divide(a,b):
    return np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=np.asarray(b)>0)

def metrics(t, rho):
    inv,cold,idle=np.moveaxis(t,-1,0)
    return np.stack([100*divide(cold,inv),1000*divide(idle,inv),1000*divide(idle+15*rho*cold,inv)],axis=-1)

def ci(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return np.quantile(a,[.025,.975]).tolist() if len(a) else None

def main():
    records=[];manifests={};checks=[]
    for folder in sorted(CASES.iterdir()):
        if not (folder/'aggregate.npz').exists():continue
        case=folder.name
        meta=json.loads((folder/'case.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        sourcehash=sha(folder/'aggregate.npz')
        assert sourcehash==summary['aggregate_sha256']
        with np.load(folder/'aggregate.npz') as z:
            names=z['action_names'].tolist();wins=z['window_names'].tolist()
            original=z['per_function'][...,:3]
            weights=z['weights'];apps=z['apps'];seeds=z['seed_totals'][...,:3]
            pf=original*weights[:,None,None,None]
        np.testing.assert_allclose(pf.sum(axis=0),seeds.mean(axis=0),rtol=2e-12,atol=1e-6)
        is_drift=case.startswith('drift_')
        arm='WINTER' if is_drift else 'WINTER_G'
        assert is_drift==not_any(names,'WINTER_G__rho')
        comps=['LSTM_Fifer','LSTM_shared'] if is_drift else ['LSTM_Fifer','LSTM_shared','EWMA_0.1','WINTER']
        if case=='continuous_48h':comps+=['No_handoff']
        # The main post-shift evidence excludes pre-shift burn-in.
        selected=[w for w in wins if w!='drain' and (not is_drift or w.startswith('post_'))]
        arrays=[pf[:,:,wins.index(w)] for w in selected]
        bounds={w:summary['windows'][w] for w in selected}
        # Disjoint intervals derive exactly from already stored cumulative totals.
        # Report the entire preset set; do not select intervals by observed advantage.
        for a,b in [(1,15),(15,30),(30,60),(60,240)]:
            lo,hi='post_'+str(a),'post_'+str(b)
            if lo in wins and hi in wins:
                name=f'interval_{a}_{b}'
                arrays.append(pf[:,:,wins.index(hi)]-pf[:,:,wins.index(lo)])
                selected.append(name)
                bounds[name]=[summary['windows'][lo][1],summary['windows'][hi][1]]
        cube=np.stack(arrays,axis=2)
        assert cube.min()>-1e-5
        _,codes=np.unique(apps,return_inverse=True);ng=int(codes.max())+1
        cluster=np.zeros((ng,*cube.shape[1:]))
        np.add.at(cluster,codes,cube)
        rng=np.random.default_rng(SEED)
        draws=np.array([np.bincount(rng.integers(ng,size=ng),minlength=ng) for _ in range(REPS)],dtype=float)
        totals=cube.sum(axis=0)
        boot=(draws@cluster.reshape(ng,-1)).reshape(REPS,*cube.shape[1:])
        for rho in meta['rhos']:
            an=f'{arm}__rho{rho:g}';ai=names.index(an)
            for comp in comps:
                cn=f'{comp}__rho{rho:g}'
                if cn not in names:continue
                bi=names.index(cn)
                np.testing.assert_allclose(totals[ai,:,0],totals[bi,:,0],atol=1e-6)
                for wi,win in enumerate(selected):
                    ta=totals[ai,wi];tb=totals[bi,wi]
                    if ta[0]<=0:continue
                    point=metrics(np.stack([ta,tb]),rho)
                    sample=metrics(boot[:,[ai,bi],wi,:],rho)
                    for idx,name in [(ai,an),(bi,cn)]:
                        if win in summary['windows']:
                            v=summary['by_action'][name]['windows'][win]
                            actual=metrics(totals[idx,wi],rho)
                            np.testing.assert_allclose(actual,[v['csr_pct'],v['idle_per_1k'],v['cost_per_1k']],rtol=2e-12,atol=1e-7)
                    row={'case':case,'surface':meta['surface'],'provider':meta['provider'],
                        'arm':arm,'comparator':comp,'rho':rho,'window':win,
                        'absolute_minute_bounds':bounds[win],
                        'n_functions_or_windows':len(weights),'n_clusters':ng,'n_des_seeds':len(meta['seeds']),
                        'invocations':float(ta[0]),'evidence_for_g':not is_drift}
                    for mi,key in enumerate(['csr_pct','wm_per_1k','cost_per_1k']):
                        pa,pb=point[:,mi];sa,sb=sample[:,:,mi].T
                        row[key]={'arm':float(pa),'comparator':float(pb),'difference':float(pa-pb),
                            'difference_ci95':ci(sa-sb),'ratio':float(pa/pb) if pb else None,
                            'ratio_ci95':ci(divide(sa,sb))}
                    row['both_point_metrics_lower']=bool(point[0,0]<point[1,0] and point[0,2]<point[1,2])
                    records.append(row)
        manifests[case]={'aggregate_sha256':sourcehash,'summary_sha256':sha(folder/'summary.json'),
            'case_sha256':sha(folder/'case.json'),'arm_for_analysis':arm,
            'g_present':not is_drift,'windows':bounds,'cluster_unit':'Huawei function' if meta['provider']=='huawei' else 'Azure application'}
        checks.append({'case':case,'weighted_per_function_equals_seed_mean':True,'summary_metrics_match':True,'common_request_denominators':True})
        print(case,len(selected),'windows;',ng,'clusters',flush=True)
    # Verify output-switch actions rather than infer G's implementation from names.
    folder=CASES/'continuous_48h';switch=[]
    for rho in [1,10,100]:
        with np.load(folder/f'WINTER_G__rho{rho}.npz') as g, np.load(folder/f'No_handoff__rho{rho}.npz') as n, np.load(folder/f'EWMA_0.1__rho{rho}.npz') as e:
            for key in ['q','ttl']:
                np.testing.assert_array_equal(g[key][:,:720],n[key][:,:720])
                np.testing.assert_array_equal(g[key][:,720:],e[key][:,720:])
        switch.append({'rho':rho,'before_720_identical_to_no_handoff':True,'from_720_identical_actions_to_ewma01':True})
    report={'description':__doc__,'date':'2026-09-16','bootstrap':{'replicates':REPS,'seed':SEED,
        'unit':'application in Azure; function in Huawei; saved sampling weights retained',
        'interpretation':'Exploratory marginal percentile intervals; simulation seeds averaged before resampling; no model-training uncertainty or multiplicity adjustment'},
        'cost_definition':'C_1k = WM_1k + 150*rho*CSR_percent; window-local invocation denominator; excludes compute/refit/control-plane cost and terminal drain',
        'window_definition':'post_n is cumulative [onset,onset+n); interval_a_b is disjoint [onset+a,onset+b); continuous young=[0,720), mature=[720,2880), pre_handoff=[600,720), post_handoff=[720,840). In steady pools labels are replay-clock windows, not individual function ages.',
        'drift_warning':'No WINTER_G arm exists in the six saved shift cases. Scheduled WINTER outcomes are component diagnostics, not direct G evidence.',
        'sources':manifests,'verification':checks,'continuous_action_switch':switch,'comparisons':records}
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    flat=[]
    for r in records:
        d={k:r[k] for k in ['case','arm','comparator','rho','window','invocations','n_clusters','evidence_for_g','both_point_metrics_lower']}
        for metric in ['csr_pct','wm_per_1k','cost_per_1k']:
            for k in ['arm','comparator','difference','ratio']:d[metric+'_'+k]=r[metric][k]
            for k in ['difference_ci95','ratio_ci95']:
                v=r[metric][k]
                d[metric+'_'+k+'_low']=v[0] if v else None
                d[metric+'_'+k+'_high']=v[1] if v else None
        flat.append(d)
    with OUT.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(len(manifests),'cases;',len(records),'comparisons;',OUT,flush=True)


def not_any(names,prefix):return not any(n.startswith(prefix) for n in names)

if __name__=='__main__':main()
