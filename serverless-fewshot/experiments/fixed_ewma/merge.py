"""Compose verified new and invariant policy outcomes without rerunning them."""
from copy import deepcopy
from pathlib import Path
import argparse
import sys
import numpy as np
from campaign import ROOT, OUT, OLD, GATE, WINTER, ALL_CASES, DRIFT, link
from common import digest, freeze, read_json, save_npz, write_json, adaptation_lag


def merge(name):
    current=OUT/'cases'/name
    current.mkdir(parents=True,exist_ok=True)
    base=OLD/'cases'/name
    original=read_json(base/'case.json')
    replay=OUT/'replay/cases'/name
    assert (replay/'summary.json').exists(),name
    sources=[base,replay]
    if name in DRIFT:sources.append(WINTER/'cases'/name)
    arrays={};metadata={};summaries={}
    for source in sources:
        d=read_json(source/'case.json');s=read_json(source/'summary.json')
        assert s['aggregate_sha256']==digest(source/'aggregate.npz')
        assert d['data_sha256']==original['data_sha256']
        assert d['seeds']==original['seeds'] and d['event_key_prefix']==original['event_key_prefix']
        with np.load(source/'aggregate.npz') as z:arrays[str(source)]={k:z[k] for k in z.files}
        metadata[str(source)]=d;summaries[str(source)]=s
    ref=arrays[str(base)]
    for a in arrays.values():
        for key in ('window_names','weights','apps','keys','total_curve'):
            np.testing.assert_array_equal(a[key],ref[key])
    origins={}
    for source in sources:
        for key in metadata[str(source)]['actions']:
            method=key.rsplit('__',1)[0]
            if method.startswith('EWMA_') and method!='EWMA_0.3':continue
            if name in DRIFT and method in ('WINTER','WINTER_frozen') and source==base:continue
            if name in DRIFT and source==base and method not in ('LSTM_Fifer','LSTM_shared','EWMA_0.3'):continue
            origins[key]=source
    names=sorted(origins)
    meta=deepcopy(original)
    meta.update(actions={},ewma_alpha=.3,revision='fixed_ewma_v1',
                measurement='per-action composition of same-request source and revised replay outcomes')
    summary=deepcopy(summaries[str(base)])
    summary['by_action']={}
    combined={k:ref[k].copy() for k in ('window_names','weights','apps','keys','total_curve')}
    combined['action_names']=np.array(names)
    for key,axis in (('per_function',1),('seed_totals',1),('timeline',0)):
        parts=[]
        for action in names:
            source=str(origins[action]);a=arrays[source]
            index=a['action_names'].tolist().index(action)
            parts.append(np.take(a[key],[index],axis=axis))
        combined[key]=np.concatenate(parts,axis=axis)
    link(base/'data.npz',current/'data.npz')
    if (base/'lstm_forecasts.npz').exists():link(base/'lstm_forecasts.npz',current/'lstm_forecasts.npz')
    for action in names:
        source=origins[action];record=deepcopy(metadata[str(source)]['actions'][action])
        link(source/record['file'],current/record['file'])
        record['measurement_source']=str(source)
        meta['actions'][action]=record
        summary['by_action'][action]=deepcopy(summaries[str(source)]['by_action'][action])
    contract=dict(aggregation='exact action-axis composition; underlying function/seed records retained at sources',
        ewma_alpha=.3,actions=names,seeds=meta['seeds'],windows=summary['windows'],
        merger_sha256=digest(__file__),
        source_files={str(p):digest(p) for source in sources
                      for p in [source/n for n in ('case.json','execution.json','aggregate.npz','summary.json')]},
        action_sources={key:str(path) for key,path in origins.items()})
    freeze(current/'case.json',meta)
    contract['case_sha256']=digest(current/'case.json')
    freeze(current/'execution.json',contract)
    save_npz(current/'aggregate.npz',**combined)
    summary['aggregate_sha256']=digest(current/'aggregate.npz')
    # Same bootstrap draws and grouping as the retained comparison protocol.
    _,codes=np.unique(combined['apps'],return_inverse=True)
    ng=int(codes.max())+1
    cube=combined['per_function'][...,:3]*combined['weights'][:,None,None,None]
    cluster=np.zeros((ng,*cube.shape[1:]));np.add.at(cluster,codes,cube)
    rng=np.random.default_rng(260915)
    draws=np.array([np.bincount(rng.integers(ng,size=ng),minlength=ng) for _ in range(2000)],float)
    boot=(draws@cluster.reshape(ng,-1)).reshape(2000,*cube.shape[1:])
    contrasts={};windows=combined['window_names'].tolist()
    for rho in meta['rhos']:
        wi=names.index(f'WINTER__rho{rho:g}')
        for method in ('LSTM_Fifer','LSTM_shared','EWMA_0.3','Fourier','Hybrid','Chronos','WINTER_G'):
            key=f'{method}__rho{rho:g}'
            if key not in names:continue
            ci=names.index(key);result={}
            for k,window in enumerate(windows):
                if window=='drain':continue
                inv=boot[:,wi,k,0];good=inv>0
                delta=100*(boot[good,wi,k,1]-boot[good,ci,k,1])/inv[good]
                result[window]=dict(winter_minus_comparator_csr_pp_ci95=np.percentile(delta,[2.5,97.5]).tolist()) if len(delta) else {}
            contrasts[f'WINTER_vs_{method}__rho{rho:g}']=result
    summary['paired_bootstrap']=dict(unit='application (Huawei function)',n_clusters=ng,
        replicates=2000,seed=260915,contrasts=contrasts)
    write_json(current/'summary.json',summary)
    print('merged',name,len(names),'actions',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=ALL_CASES);args=ap.parse_args()
    for name in ([args.case] if args.case else ALL_CASES):merge(name)
