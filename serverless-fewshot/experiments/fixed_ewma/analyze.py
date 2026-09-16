"""Paired fixed-alpha outcomes, all retained windows and cumulative curves."""
import csv
import argparse
import numpy as np
from campaign import OUT,GATE,ALL_CASES,DRIFT
from common import read_json,write_json,digest,rolling_csr,adaptation_lag

METHODS=('WINTER','WINTER_G','LSTM_Fifer','LSTM_shared','EWMA_0.3','No_handoff')
METRICS=('csr_pct','wm_per_1k','cost_per_1k')


def values(t,rho):
    inv,cold,idle=np.moveaxis(t,-1,0)
    def div(x):return np.divide(x,inv,out=np.full_like(inv,np.nan,dtype=float),where=inv>0)
    return np.stack((100*div(cold),1000*div(idle),1000*div(idle+15*rho*cold)),axis=-1)


def interval(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return np.quantile(x,[.025,.975]).tolist() if len(x) else None


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def flatten(records):
    out=[]
    for record in records:
        row={k:v for k,v in record.items() if not isinstance(v,dict)}
        for metric in METRICS:
            for key,value in record[metric].items():
                if key.endswith('ci95'):
                    row[metric+'_'+key+'_low'],row[metric+'_'+key+'_high']=value if value is not None else (None,None)
                else:row[metric+'_'+key]=value
        out.append(row)
    return out


def main(preview=False):
    drift=[];phase=[];points=[];curves=[];checks=[];routes={}
    for name in ALL_CASES:
        directory=OUT/'cases'/name
        if preview and not (directory/'summary.json').exists():continue
        meta=read_json(directory/'case.json');summary=read_json(directory/'summary.json')
        assert digest(directory/'aggregate.npz')==summary['aggregate_sha256']
        with np.load(directory/'aggregate.npz') as z:d={k:z[k] for k in z.files}
        actions=d['action_names'].tolist();wins=d['window_names'].tolist()
        if name in DRIFT:windows=[f'post_{t}' for t in (1,15,30,60,240)]
        else:windows=[w for w in wins if w!='drain']
        parts=[d['per_function'][:,:,wins.index(w),:3] for w in windows]
        for a,b in ((1,15),(15,30),(30,60),(60,240)):
            if f'post_{b}' in wins:
                windows.append(f'interval_{a}_{b}')
                parts.append(d['per_function'][:,:,wins.index(f'post_{b}'),:3]-
                             d['per_function'][:,:,wins.index(f'post_{a}'),:3])
        cube=np.stack(parts,axis=2)*d['weights'][:,None,None,None]
        _,codes=np.unique(d['apps'],return_inverse=True);ng=int(codes.max())+1
        cluster=np.zeros((ng,*cube.shape[1:]));np.add.at(cluster,codes,cube)
        rng=np.random.default_rng(260915)
        draws=np.array([np.bincount(rng.integers(ng,size=ng),minlength=ng) for _ in range(2000)],float)
        boot=(draws@cluster.reshape(ng,-1)).reshape(2000,*cube.shape[1:]);totals=cube.sum(0)
        for rho in meta['rhos']:
            available=[m for m in METHODS if f'{m}__rho{rho:g}' in actions]
            for method in available:
                ai=actions.index(f'{method}__rho{rho:g}')
                onset=meta.get('onset_minute',0);horizon=meta['horizon_minutes']
                cold=d['timeline'][ai,onset:horizon,0];inv=d['total_curve'][onset:horizon]
                curve=rolling_csr(cold,inv);finite=curve[3*len(curve)//4:];finite=finite[np.isfinite(finite)]
                ref=float(100*finite.mean()) if len(finite) else None
                for wi,window in enumerate(windows):
                    v=values(totals[ai,wi],rho)
                    if name in DRIFT:
                        points.append(dict(case=name,method=method,rho=rho,window=window,
                            csr_pct=float(v[0]),wm_per_1k=float(v[1]),cost_per_1k=float(v[2]),
                            adaptation_lag=adaptation_lag(cold,inv),reference_csr_pct=ref))
                for arm in (('WINTER','WINTER_G') if name in DRIFT else ('WINTER_G',)):
                    if method==arm:continue
                    bi=actions.index(f'{arm}__rho{rho:g}')
                    np.testing.assert_array_equal(totals[ai,:,0],totals[bi,:,0])
                    for wi,window in enumerate(windows):
                        pt=values(totals[[bi,ai],wi],rho);bs=values(boot[:,[bi,ai],wi],rho)
                        record=dict(case=name,provider=meta['provider'],source=meta.get('drift_source',meta['surface']),
                            rho=rho,window=window,arm=arm,comparator=method,
                            n_event_windows=len(codes),n_clusters=ng,invocations=float(totals[bi,wi,0]))
                        if not np.isfinite(pt).all():continue
                        for mi,metric in enumerate(METRICS):
                            a,b=pt[:,mi];sa,sb=bs[:,:,mi].T
                            ratio=np.divide(sa,sb,out=np.full_like(sa,np.nan),where=sb>0)
                            record[metric]=dict(arm=float(a),comparator=float(b),difference=float(a-b),
                                difference_ci95=interval(sa-sb),ratio=float(a/b) if b>0 else None,ratio_ci95=interval(ratio))
                        (drift if name in DRIFT else phase).append(record)
                if name in DRIFT and rho==10:
                    cumulative=np.cumsum(d['timeline'][ai,1440:1680],axis=0);den=np.cumsum(inv)
                    for minute in range(240):
                        curves.append(dict(case=name,method=method,minute=minute+1,
                            invocations=float(den[minute]),cold=float(cumulative[minute,0]),idle=float(cumulative[minute,1])))
        checks.append(dict(case=name,aggregate_sha256=digest(directory/'aggregate.npz'),
            ewma_alpha=.3,shared_requests_verified=True,weighted_totals_verified=True))
        if name in DRIFT:routes[name]=read_json(GATE/'cases'/name/'routing_summary.json')
        print('paired analysis',name,flush=True)
    assert len(drift)==1080 and len(points)==648 and len(curves)==8640
    destination=OUT/('analysis_preview' if preview else 'analysis');destination.mkdir(exist_ok=True)
    common=dict(status='preview' if preview else 'complete',ewma_alpha=.3,protocol_sha256=digest(OUT/'protocol.json'),
        analyzer_sha256=digest(__file__),uncertainty='2000 paired application/function-cluster resamples; seed 260915; marginal 95% intervals')
    write_json(destination/'drift.json',dict(**common,cases=[c for c in checks if c['case'] in DRIFT],
        comparisons=drift,operating_points=points,routing=routes))
    write_json(destination/'nonshift.json',dict(**common,cases=[c for c in checks if c['case'] not in DRIFT],comparisons=phase))
    csv_write(destination/'comparisons.csv',flatten(drift));csv_write(destination/'operating_points.csv',points)
    csv_write(destination/'figure_curves.csv',curves);csv_write(destination/'nonshift_comparisons.csv',flatten(phase))
    print('Analysis complete',len(drift),'drift contrasts',len(phase),'other contrasts',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true')
    main(ap.parse_args().preview)
