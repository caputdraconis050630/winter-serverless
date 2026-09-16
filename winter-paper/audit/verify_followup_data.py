"""Read all v1/v2 result arrays and recheck endpoints and primary v2 CIs."""
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

from verify_submission_data import SF, OUT, read, write, csv_write, close, functions


def v1():
    root = SF / 'results/onboarding_attribution_v1'
    old = pd.read_csv(root / 'endpoints.csv')
    lookup = {(r.cohort,r.tag,r.arm,r.interval): r for r in old.itertuples()}
    intervals = dict(minute0=[0],minute1_15=[1],minute15_60=[2],minute60_240=[3],first60=[0,1,2],first240=[0,1,2,3])
    checked = 0
    refs = {}
    paths = sorted((root / 'evaluation').rglob('*.npz'))
    for path in paths:
        cohort, tag, name = path.relative_to(root / 'evaluation').parts
        with np.load(path) as z:
            m = z['metrics']
        assert np.isfinite(m).all() and m.min() >= -1e-7
        assert np.all(m[:,:,:,1] <= m[:,:,:,0])
        key = (cohort, m.shape[1])
        requests = m[:,:,:4,0].sum(2)
        execution = m[:,:,:,4].sum(2)
        if key not in refs:
            refs[key] = requests, execution
        else:
            close(requests, refs[key][0])
            close(execution, refs[key][1], rtol=1e-8, atol=1e-7)
        if cohort == 'azure_calibration':
            continue
        rho = float(tag.split('rho')[-1])
        for interval, bins in intervals.items():
            a = m[:,:,bins,:].sum(2).mean(1).sum(0)
            if not a[0]:
                continue
            row = lookup[(cohort, tag, path.stem, interval)]
            actual = dict(invocations=a[0],cold=a[1],csr_pct=100*a[1]/a[0],wm_per_1k=1000*a[2]/a[0],
                          init_per_1k=1000*a[3]/a[0],execution_per_1k=1000*a[4]/a[0],
                          allocated_per_1k=1000*a[2:5].sum()/a[0],modeled_cost_per_1k=1000*(a[2]+15*rho*a[1])/a[0])
            for k,v in actual.items():
                close(v,getattr(row,k),atol=1e-7,rtol=1e-10)
            checked += 1
    assert checked == len(old)
    return dict(result_npz_files=len(paths), endpoint_rows=checked, all_endpoints_match=True,
                nonnegative_finite_metrics=True, request_totals_and_execution_conserved_across_policies=True,
                note='Cross-policy execution conservation does not independently re-integrate every container ledger.')


def v2():
    root = SF / 'results/onboarding_symmetric_v2'
    old = pd.read_csv(root / 'endpoints.csv')
    contrasts = pd.read_csv(root / 'contrasts.csv')
    lookup = {(r.cohort,float(r.rho),r.condition,r.arm,r.interval): r for r in old.itertuples()}
    ns = dict(np=np)
    functions(SF / 'experiments/onboarding_attribution/analyze.py', {'PairedBootstrap','window_contrast','holm'}, ns)
    functions(SF / 'experiments/onboarding_symmetric/report.py', {'cost_contrast'}, ns)
    checked, files, primary_records, mechanism_records, raw_policies = 0, 0, [], [], 0
    for directory in sorted((root / 'evaluation').glob('*/*')):
        if not directory.is_dir():
            continue
        cohort = directory.parent.name
        condition, rho_text = directory.name.split('_rho')
        rho = float(rho_text)
        meta = read(directory / 'complete.json')
        matrices, names = [], None
        for f in range(meta['n']):
            with np.load(directory / f'f{f:04d}.npz') as z:
                m = z['metrics']
                local_names = z['names'].tolist()
                assert z['seeds'].tolist() == list(range(1000,1020))
            assert names is None or names == local_names
            names = local_names
            assert np.isfinite(m).all() and m.min() >= -1e-7
            assert np.all(m[:,:,:,1] <= m[:,:,:,0])
            req, exe = m[:,:,:4,0].sum(2), m[:,:,:,4].sum(2)
            close(req, np.broadcast_to(req[0], req.shape))
            close(exe, np.broadcast_to(exe[0], exe.shape), rtol=1e-8, atol=1e-7)
            matrices.append(m)
            files += 1
            raw_policies += len(names)
        all_m = np.stack(matrices)
        del matrices
        mm = {name: all_m[:,i] for i,name in enumerate(names)}
        for suffix in ['native']+[f'b{i}' for i in range(6)]+['trained_lambda_wrapper']:
            rnd = [f'random{i}__{suffix}' for i in range(5)]
            if all(n in mm for n in rnd):
                mm['random_mean__'+suffix] = np.mean([mm[n] for n in rnd], axis=0)
        for arm, m in mm.items():
            drain = m[:,:,4,2].mean(1).sum()
            fullinv = m[:,:,:4,0].sum(2).mean(1).sum()
            for interval,bins in [('first15',[0,1]),('first60',[0,1,2]),('first240',[0,1,2,3])]:
                a = m[:,:,bins].sum(2).mean(1).sum(0)
                row = lookup[(cohort,rho,condition,arm,interval)]
                actual = dict(invocations=a[0],cold=a[1],csr_pct=100*a[1]/a[0],wm_per1k=1000*a[2]/a[0],
                              allocated_per1k=1000*a[2:5].sum()/a[0],modeled_cost_per1k=1000*(a[2]+15*rho*a[1])/a[0],
                              drain_idle_per1k_4h=1000*drain/fullinv)
                if interval=='first240':
                    actual['cost_plus_drain_per1k']=1000*(a[2]+15*rho*a[1]+drain)/a[0]
                for k,v in actual.items():
                    close(v,getattr(row,k),atol=1e-7,rtol=1e-10)
                checked += 1
        if rho == 10:
            with np.load(SF / 'results/onboarding_attribution_v1/cohorts' / f'{cohort}.npz') as z:
                bs = ns['PairedBootstrap'](z['apps'],z['counts'].sum(1))
            subset = contrasts[(contrasts.cohort==cohort)&(contrasts.rho==rho)&(contrasts.condition==condition)]
            subset = subset[(subset.primary==True)|(subset.mechanistic_family==True)]
            for row in subset.itertuples():
                actual = ns['window_contrast'](bs,mm[row.arm],mm[row.comparator])
                actual.update(ns['cost_contrast'](bs,mm[row.arm],mm[row.comparator],rho))
                for k,v in actual.items():
                    if isinstance(v,bool):
                        assert v==getattr(row,k)
                    else:
                        close(v,getattr(row,k),atol=1e-7,rtol=1e-10)
                rec = dict(cohort=cohort,condition=condition,arm=row.arm,comparator=row.comparator,**actual)
                if row.primary:
                    primary_records.append(rec)
                if row.mechanistic_family:
                    mechanism_records.append(rec)
        print('PASS v2',cohort,condition,rho,'files',files,flush=True)
        del mm, all_m
    assert checked==len(old)
    for records, column in [(primary_records,'holm6_p'),(mechanism_records,'holm9_p')]:
        adjusted=ns['holm']([x['bootstrap_t_p'] for x in records])
        for r,p in zip(records,adjusted):
            oldrow=contrasts[(contrasts.cohort==r['cohort'])&(contrasts.rho==10)&(contrasts.condition==r['condition'])&(contrasts.arm==r['arm'])&(contrasts.comparator==r['comparator'])].iloc[0]
            close(p,oldrow[column])
            r[column]=float(p)
    csv_write('v2_primary_recomputed.csv',primary_records)
    csv_write('v2_mechanisms_recomputed.csv',mechanism_records)
    return dict(result_npz_files=files,function_policy_records=raw_policies,endpoint_rows=checked,
                all_endpoints_match=True,primary_contrasts_recomputed=len(primary_records),
                mechanistic_contrasts_recomputed=len(mechanism_records),all_checked_CIs_and_Holm_p_match=True,
                nonnegative_finite_metrics=True,request_totals_and_execution_conserved_across_policies=True,
                note='Existing function/seed/bin arrays reaggregated; no new DES runs, calibration, or tuning.')


def main():
    result={}
    for name,call in [('v1',v1),('v2',v2)]:
        print('START',name,flush=True)
        result[name]=call()
        write('followup_verification.json',result)
        print('PASS',name,flush=True)


if __name__=='__main__':
    main()
