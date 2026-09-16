"""Paired analysis of all registered added WINTER-G drift cases."""
import csv
from pathlib import Path
import numpy as np
from campaign import ROOT, OLD, OUT, CASE_NAMES
from common import read_json, write_json, digest, save_npz


def values(t,rho):
    inv,cold,idle=np.moveaxis(t,-1,0)
    def div(x):return np.divide(x,inv,out=np.full_like(inv,np.nan,dtype=float),where=inv>0)
    return np.stack((100*div(cold),1000*div(idle),1000*div(idle+15*rho*cold)),axis=-1)


def interval(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return np.quantile(x,[.025,.975]).tolist() if len(x) else None


def analyze():
    plan=read_json(OUT/'protocol.json');records=[];checks=[];routes={};curves={}
    for name in CASE_NAMES:
        new=OUT/'cases'/name;old=OLD/'cases'/name
        meta=read_json(new/'case.json');base=read_json(old/'case.json')
        assert digest(old/'case.json')==plan['source_cases'][name]['case_sha256']
        assert digest(old/'aggregate.npz')==plan['source_cases'][name]['aggregate_sha256']
        assert digest(new/'aggregate.npz')==read_json(new/'summary.json')['aggregate_sha256']
        with np.load(new/'data.npz') as a,np.load(old/'data.npz') as b:
            for k in a.files:np.testing.assert_array_equal(a[k],b[k])
        with np.load(new/'routing.npz') as z: states=z['states']
        for rho in meta['rhos']:
            with np.load(new/f'WINTER_G__rho{rho:g}.npz') as g, np.load(new/f'No_handoff__rho{rho:g}.npz') as n, np.load(old/f'EWMA_0.1__rho{rho:g}.npz') as e:
                for key in ('q','ttl'):
                    np.testing.assert_array_equal(g[key][states!=3],n[key][states!=3])
                    mask=(states==1)|(states==3)
                    np.testing.assert_array_equal(g[key][mask],e[key][mask])
        arrays=[];actions=[];timelines=[]
        for d in (new,old):
            with np.load(d/'aggregate.npz') as z:
                ns=z['action_names'].tolist();ws=z['window_names'].tolist()
                arrays.append(z['per_function'][...,:3]);actions.extend(ns);timelines.append(z['timeline'])
                np.testing.assert_allclose(np.einsum('f,fawm->awm',z['weights'],z['per_function']),z['seed_totals'].mean(0),rtol=2e-12,atol=1e-6)
                if d==new:
                    wins=ws;weights=z['weights'];apps=z['apps'];total_curve=z['total_curve'];keys=z['keys']
                else:
                    assert ws==wins
                    for k,v in [('weights',weights),('apps',apps),('total_curve',total_curve),('keys',keys)]:np.testing.assert_array_equal(z[k],v)
        arrays=np.concatenate(arrays,axis=1);timeline=np.concatenate(timelines)
        windows=[f'post_{t}' for t in (1,15,30,60,240)]
        parts=[arrays[:,:,wins.index(w)] for w in windows]
        for a,b in ((1,15),(15,30),(30,60),(60,240)):
            windows.append(f'interval_{a}_{b}')
            parts.append(arrays[:,:,wins.index(f'post_{b}')]-arrays[:,:,wins.index(f'post_{a}')])
        cube=np.stack(parts,axis=2)*weights[:,None,None,None]
        _,codes=np.unique(apps,return_inverse=True);ng=codes.max()+1
        clusters=np.zeros((ng,*cube.shape[1:]));np.add.at(clusters,codes,cube)
        rng=np.random.default_rng(260915)
        draws=np.array([np.bincount(rng.integers(ng,size=ng),minlength=ng) for _ in range(2000)],float)
        boot=(draws@clusters.reshape(ng,-1)).reshape(2000,*cube.shape[1:]);totals=cube.sum(0)
        for rho in meta['rhos']:
            ai=actions.index(f'WINTER_G__rho{rho:g}')
            for other in ('LSTM_Fifer','LSTM_shared','EWMA_0.1','No_handoff','WINTER','WINTER_frozen'):
                bi=actions.index(f'{other}__rho{rho:g}')
                np.testing.assert_array_equal(totals[ai,:,0],totals[bi,:,0])
                for wi,window in enumerate(windows):
                    pt=values(totals[[ai,bi],wi],rho);bs=values(boot[:,[ai,bi],wi],rho)
                    r=dict(case=name,provider=meta['provider'],source=meta['drift_source'],rho=rho,window=window,
                           comparator=other,n_event_windows=len(weights),n_clusters=int(ng),invocations=float(totals[ai,wi,0]))
                    for mi,m in enumerate(('csr_pct','wm_per_1k','cost_per_1k')):
                        a,b=pt[:,mi];sa,sb=bs[:,:,mi].T
                        ratio=np.divide(sa,sb,out=np.full_like(sa,np.nan),where=sb>0)
                        r[m]=dict(g=float(a),comparator=float(b),difference=float(a-b),difference_ci95=interval(sa-sb),
                                  ratio=float(a/b) if b>0 else None,ratio_ci95=interval(ratio))
                    records.append(r)
        routes[name]=read_json(new/'routing_summary.json')
        checks.append(dict(case=name,source_hashes_unchanged=True,exact_same_input_arrays=True,
                           original_and_new_weighted_seed_totals_match=True,common_request_denominators=True,
                           max_direct_log_forecast_error=read_json(new/'forecast_checks.json')['max_direct_log_forecast_error'],
                           aggregate_sha256=digest(new/'aggregate.npz')))
        curves[name]=(actions,timeline,total_curve)
    report=dict(status='complete',protocol_sha256=digest(OUT/'protocol.json'),analysis_sha256=digest(__file__),
                cases=checks,routing=routes,comparisons=records,
                baseline_replay_checks=read_json(OUT/'baseline_replay_checks.json'),
                uncertainty=plan['uncertainty'],
                cost='WM/1k + 150*rho*CSR_percent; window-local invocation denominator; excludes terminal drain and predictor/control-plane compute')
    write_json(OUT/'analysis.json',report)
    flat=[]
    for r in records:
        row={k:v for k,v in r.items() if not isinstance(v,dict)}
        for m in ('csr_pct','wm_per_1k','cost_per_1k'):
            for k,v in r[m].items():
                if k.endswith('ci95'):row[m+'_'+k+'_low'],row[m+'_'+k+'_high']=v if v is not None else (None,None)
                else:row[m+'_'+k]=v
        flat.append(row)
    with (OUT/'comparisons.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    figures(curves)
    print('verified',len(checks),'cases;',len(records),'paired comparisons',flush=True)
    for r in records:
        if r['rho']==10 and r['window']=='post_240' and r['comparator']=='LSTM_Fifer':
            print(r['case'],'CSR',r['csr_pct'],'COST',r['cost_per_1k'],flush=True)


def figures(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    labels={'WINTER_G':'WINTER-G','LSTM_Fifer':'LSTM native','LSTM_shared':'LSTM shared','EWMA_0.1':'EWMA (0.1)','No_handoff':'No hand-off'}
    styles={'WINTER_G':('#b2182b','-'),'LSTM_Fifer':('#2166ac','-'),'LSTM_shared':('#67a9cf','--'),'EWMA_0.1':('#444444',':'),'No_handoff':('#4d9221','--')}
    for metric in ('csr','cost'):
        fig,axes=plt.subplots(3,2,figsize=(10,8),sharex=True,layout='constrained')
        for ax,name in zip(axes.flat,CASE_NAMES):
            names,timeline,total=curves[name];denom=np.cumsum(total[1440:1680]);t=np.arange(1,241)
            for m,label in labels.items():
                i=names.index(m+'__rho10');cold=np.cumsum(timeline[i,1440:1680,0]);idle=np.cumsum(timeline[i,1440:1680,1])
                top=100*cold if metric=='csr' else 1000*(idle+150*cold)
                value=np.divide(top,denom,out=np.full_like(top,np.nan),where=denom>0)
                color,style=styles[m];ax.plot(t,value,label=label,color=color,ls=style,lw=1.6)
            title=name.replace('drift_','').replace('azure2019','Azure 2019').replace('azure2021','Azure 2021').replace('huawei','Huawei').replace('_natural',' / Natural shift').replace('_synthetic',' / Injected shift')
            ax.set_title(title,fontsize=10)
            ax.grid(alpha=.2);ax.set_xlim(1,240)
        for ax in axes[-1]:ax.set_xlabel('Minutes after shift')
        fig.supylabel('Cumulative CSR (%)' if metric=='csr' else 'Cumulative modeled cost (GB s / 1k calls)',fontsize=11)
        handles,legend=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend,loc='outside lower center',ncol=5,fontsize=9)
        fig.suptitle('Post-shift outcomes at $\\rho=10$; continuous state through 24 h burn-in',fontsize=12)
        fig.savefig(directory/f'post_shift_{metric}.pdf');fig.savefig(directory/f'post_shift_{metric}.png',dpi=160)
        plt.close(fig)

if __name__=='__main__':analyze()
