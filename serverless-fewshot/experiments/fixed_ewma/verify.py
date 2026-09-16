"""Independently reconstruct revised ledgers and validate fixed-alpha reuse."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from campaign import ROOT,OUT,OLD,WINTER,ALL_CASES,DRIFT,ewma_masks,startup_rates,startup_actions
from common import digest,read_json,write_json,METRICS,adaptation_lag
from cases import rates_ewma,decisions_from_rates,newsvendor_quantile
from events import event_chunks
from simulator import advance,new_state
from run import windows_for
import verify_results as original_verifier


def equal(a,b):np.testing.assert_allclose(a,b,rtol=3e-9,atol=1e-5)


def verify_case(name):
    p=OUT/'replay/cases'/name;current=OUT/'cases'/name
    meta=read_json(p/'case.json');contract=read_json(p/'execution.json');ch=digest(p/'execution.json')
    assert contract['case_sha256']==digest(p/'case.json')
    with np.load(p/'data.npz') as z:data={k:z[k] for k in z.files}
    with np.load(p/'aggregate.npz') as z:agg={k:z[k] for k in z.files}
    assert meta['data_sha256']==digest(p/'data.npz')
    assert read_json(p/'summary.json')['aggregate_sha256']==digest(p/'aggregate.npz')
    names=sorted(meta['actions']);assert names==agg['action_names'].tolist()
    windows=windows_for(meta);assert list(windows)==agg['window_names'].tolist()
    n,horizon=data['counts'].shape;s=len(meta['seeds'])
    seed_totals=np.zeros_like(agg['seed_totals']);timeline=np.zeros_like(agg['timeline'])
    chain=hashlib.sha256();logical=n*len(names)*s;actual=0
    for f in range(n):
        path=p/'functions'/f'{f:05d}.npz';chain.update(bytes.fromhex(digest(path)))
        with np.load(path) as z:
            assert str(z['contract_sha256'])==ch
            m=z['mapping'];stats=z['stats'][:,m]
            equal(stats.mean(0),agg['per_function'][f])
            assert np.isfinite(stats).all() and (stats>=-1e-6).all()
            assert (stats[...,1]<=stats[...,0]).all()
            for wi,(lo,hi) in enumerate(windows.values()):
                expected=data['counts'][f,lo:min(hi,horizon)].sum()
                np.testing.assert_array_equal(stats[:,:,wi,0],np.full((s,len(names)),expected))
                equal(z['cold'][m,lo:hi].sum(1)/s,stats[:,:,wi,1].mean(0))
                np.testing.assert_allclose(z['idle'][m,lo:hi].sum(1),stats[:,:,wi,2].mean(0),rtol=4e-6,atol=.01)
            full=list(windows).index('full');drain=list(windows).index('drain')
            total_exec=stats[:,:,full,4]+stats[:,:,drain,4]
            equal(total_exec,np.broadcast_to(total_exec[:,:1],total_exec.shape))
            seed_totals+=data['weights'][f]*stats
            timeline[:,:,0]+=data['weights'][f]*z['cold'][m]/s
            timeline[:,:,1]+=data['weights'][f]*z['idle'][m]
        for si in range(s):
            path=p/'seed_functions'/f'{f:05d}_{si:02d}.npz'
            with np.load(path) as z:
                assert str(z['contract_sha256'])==ch and int(z['seed'])==meta['seeds'][si]
                equal(z['stats'][z['mapping']],stats[si])
                actual+=len(z['stats'])-len(z['reused_actions'])
    equal(seed_totals,agg['seed_totals']);equal(timeline,agg['timeline'])
    assert not list((p/'partial_states').glob('*.npz'))
    # Recompute the fixed baseline, not merely its method label.
    rate=rates_ewma(data['counts'],.3)
    for rho in meta['rhos']:
        q,t=decisions_from_rates(rate,newsvendor_quantile(rho))
        with np.load(current/f'EWMA_0.3__rho{rho:g}.npz') as z:
            np.testing.assert_array_equal(np.clip(q,0,200).astype(np.int16),z['q'])
            np.testing.assert_array_equal(t.astype(np.float32),z['ttl'])
    del rate
    # The separate reference engine executes an actual changed full-horizon row.
    volumes=data['counts'].sum(1)
    f=int(np.argmin(abs(volumes-2500)))
    actions=[]
    for key in names:
        rec=meta['actions'][key];path=p/rec['file'];assert digest(path)==rec['sha256']
        with np.load(path) as z:actions.append((z['q'][f],z['ttl'][f]))
    states=[new_state(horizon) for _ in actions]
    for lo,hi,tape in event_chunks(data['counts'][f],data['duration_mean'][f],data['duration_std'][f],
                                   meta['seeds'][0],meta['event_key_prefix']+str(data['keys'][f])):
        for (q,t),st in zip(actions,states):advance(*tape,q[lo:hi],t[lo:hi],*st,lo,hi==horizon)
    actual_ref=np.array([[st[0][lo:hi].sum(0) for lo,hi in windows.values()] for st in states])
    with np.load(p/'functions'/f'{f:05d}.npz') as z:expected=z['stats'][0,z['mapping']]
    equal(actual_ref,expected)
    # Check every composed action plane against its recorded source.
    composition=read_json(current/'execution.json');cm=read_json(current/'case.json')
    assert composition['case_sha256']==digest(current/'case.json')
    for path,expected_hash in composition['source_files'].items():assert digest(path)==expected_hash,path
    with np.load(current/'aggregate.npz') as z:merged={k:z[k] for k in z.files}
    summary=read_json(current/'summary.json');assert summary['aggregate_sha256']==digest(current/'aggregate.npz')
    c_names=merged['action_names'].tolist()
    for src in set(composition['action_sources'].values()):
        with np.load(Path(src)/'aggregate.npz') as z:
            sn=z['action_names'].tolist()
            for key,source in composition['action_sources'].items():
                if source!=src:continue
                i,j=c_names.index(key),sn.index(key)
                for field,axis in (('per_function',1),('seed_totals',1),('timeline',0)):
                    np.testing.assert_array_equal(np.take(merged[field],i,axis=axis),np.take(z[field],j,axis=axis))
    assert not any(k.startswith(('EWMA_0.1','EWMA_0.5','WINTER_frozen')) for k in c_names)
    for key,row in summary['by_action'].items():
        i=c_names.index(key);onset=meta.get('onset_minute',0)
        assert row['adaptation_lag']==original_verifier.lag_independent(merged['timeline'][i,onset:horizon,0],merged['total_curve'][onset:horizon])
        for wi,window in enumerate(merged['window_names']):
            v=merged['seed_totals'][:,i,wi].mean(0);r=row['windows'][str(window)]
            for mi,metric in enumerate(METRICS):equal(v[mi],r[metric])
            if v[0]:
                equal(100*v[1]/v[0],r['csr_pct']);equal(1000*v[2]/v[0],r['idle_per_1k'])
                equal(1000*(v[2]+15*row['rho']*v[1])/v[0],r['cost_per_1k'])
    print('verified',name,flush=True)
    return dict(status='passed',source_sha256={f:digest(current/f) for f in ('case.json','execution.json','aggregate.npz','summary.json')},
        ewma_alpha=.3,functions=n,seeds=s,actions=len(c_names),logical_policy_function_seeds=n*s*len(c_names),
        revised_logical_policy_function_seeds=logical,actual_revised_policy_function_seeds=actual,
        function_file_hash_chain=chain.hexdigest(),independent_reference_function=f,
        checks=['new per-function/seed ledgers','full-horizon reference replay','causal .3 EWMA action reconstruction',
                'exact composed source planes','phase conservation','cost arithmetic','independent AL'])


def main():
    dest=OUT/'verification';dest.mkdir(exist_ok=True)
    source=read_json(OLD/'verification/report.json');assert source['status']=='passed'
    protocol=read_json(OUT/'protocol.json')
    for file,expected in protocol['source_sha256'].items():
        assert digest(ROOT/file)==expected,file
    matched=read_json(WINTER/'verification.json')
    complete=read_json(WINTER/'completion.json')
    assert matched['status']=='passed' and complete['status']=='complete'
    assert complete['verification_sha256']==digest(WINTER/'verification.json')
    assert matched['protocol_sha256']==complete['protocol_sha256']==digest(WINTER/'protocol.json')
    assert {r['case'] for r in matched['cases']}==set(DRIFT)
    for record in matched['cases']:
        assert record['status']=='passed'
        assert digest(WINTER/'cases'/record['case']/'aggregate.npz')==record['aggregate_sha256']
    for name in ALL_CASES:
        for file,h in source['cases'][name]['source_sha256'].items():assert digest(OLD/'cases'/name/file)==h
    reports={}
    for name in ALL_CASES:
        path=dest/(name+'.json')
        if path.exists():
            old=read_json(path)
            if old.get('verifier_sha256')==digest(__file__) and all(digest(OUT/'cases'/name/k)==v for k,v in old['source_sha256'].items()):
                reports[name]=old;continue
        report=verify_case(name);report['verifier_sha256']=digest(__file__)
        write_json(path,report);reports[name]=report
    original_verifier.OUT=OUT
    live=original_verifier.live();assert live['status']=='passed'
    from verify_live import main as verify_schedules
    schedule_report=verify_schedules()
    live['schedule_verification_sha256']=digest(OUT/'strict_round/live/schedule_verification.json')
    assert read_json(OUT/'controls/verification.json')['status']=='passed'
    text=(OUT/'logs/replay_tests.log').read_text();assert 'PASS: exact full-schedule reuse' in text
    result=dict(status='passed',ewma_alpha=.3,expected_cases=ALL_CASES,missing_cases=[],cases=reports,live=live,
        tests=dict(status='passed',source_tests=source['tests'],reuse_test_log_sha256=digest(OUT/'logs/replay_tests.log')),
        logical_policy_function_seeds=sum(r['logical_policy_function_seeds'] for r in reports.values()),
        protocol_sha256=digest(OUT/'protocol.json'),verifier_sha256=digest(__file__),
        reused_matched_winter_verification_sha256=digest(WINTER/'verification.json'),
        controls_verification_sha256=digest(OUT/'controls/verification.json'))
    write_json(dest/'report.json',result);print('All fixed-alpha measurements verified',flush=True)


if __name__=='__main__':main()
