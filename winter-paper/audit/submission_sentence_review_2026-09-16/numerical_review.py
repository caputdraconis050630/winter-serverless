"""Independently reaggregate saved outcomes; never fit or rerun a policy."""
from pathlib import Path
import csv
import hashlib
import json
import re
import numpy as np

PAPER = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = PAPER.parent / 'serverless-fewshot/results/fixed_ewma_v1'
records, failures, sources = [], [], {}

def read(p):
    sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())

def check(label, actual, expected, atol=1e-8, rtol=2e-11):
    if actual is None or expected is None:
        ok = actual is expected
        error = None
    else:
        a, b = np.asarray(actual), np.asarray(expected)
        ok = bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
        error = float(np.nanmax(np.abs(a-b))) if a.size else 0.
    row = dict(check=label, passed=ok, max_absolute_error=error)
    records.append(row)
    if not ok:
        row.update(actual=np.asarray(actual).tolist(), expected=np.asarray(expected).tolist())
        failures.append(row)

def interval(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return np.percentile(values, [2.5, 97.5]) if len(values) else None

def al_and_reference(cold, inv):
    numerator = np.convolve(cold, np.ones(15), mode='same')
    denominator = np.convolve(inv, np.ones(15), mode='same')
    c = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator>0)
    tail = c[3*len(c)//4:]
    ref = float(np.nanmean(tail)) if np.isfinite(tail).any() else None
    lag = None
    if ref is not None:
        for i in range(len(c)-30):
            if np.isfinite(c[i:i+31]).all() and (c[i:i+31] <= ref*1.1+1e-10).all():
                lag = i
                break
    return lag, None if ref is None else 100*ref

phase = read(PAPER/'audit/fixed_ewma_2026-09-16/nonshift.json')
drift = read(PAPER/'audit/fixed_ewma_2026-09-16/drift/analysis.json')
all_contrasts = phase['comparisons'] + drift['comparisons']
names = sorted({r['case'] for r in all_contrasts})
case_summaries = {}
for case in names:
    directory = RUN/'cases'/case
    metadata = read(directory/'case.json')
    summary = read(directory/'summary.json')
    case_summaries[case] = summary
    aggregate = directory/'aggregate.npz'
    sources[str(aggregate)] = hashlib.sha256(aggregate.read_bytes()).hexdigest()
    assert sources[str(aggregate)] == summary['aggregate_sha256']
    with np.load(aggregate) as f:
        z = {k:f[k] for k in f.files}
    actions, windows = z['action_names'].tolist(), z['window_names'].tolist()
    weighted = z['per_function'] * z['weights'][:,None,None,None]
    totals = weighted.sum(0)
    check(case+'/seed-average totals', totals, z['seed_totals'].mean(0), atol=1e-5)
    for ai, action in enumerate(actions):
        result = summary['by_action'][action]
        rho = result['rho']
        for wi, window in enumerate(windows):
            v = totals[ai,wi]
            saved = result['windows'][window]
            check(f'{case}/{action}/{window}/raw',v,[saved[k] for k in
                ['invocations','cold','idle','init','execution','prewarm','reactive','overflow']],atol=1e-5)
            if v[0]>0:
                check(f'{case}/{action}/{window}/CSR,WM,COST',
                      [100*v[1]/v[0],1000*v[2]/v[0],1000*(v[2]+15*rho*v[1])/v[0]],
                      [saved['csr_pct'],saved['idle_per_1k'],saved['cost_per_1k']])
        start = metadata.get('onset_minute',0)
        end = metadata['horizon_minutes']
        al, ref = al_and_reference(z['timeline'][ai,start:end,0],z['total_curve'][start:end])
        check(f'{case}/{action}/AL',al,result['adaptation_lag'])
        for p in drift['operating_points']:
            if p['case']==case and p['method']==result['method'] and p['rho']==rho:
                check(f'{case}/{action}/{p["window"]}/reference',ref,p['reference_csr_pct'])

    # Form cluster totals independently from the saved contrast records.
    wn = [w for w in windows if w!='drain']
    values = [weighted[:,:,windows.index(w),:3] for w in wn]
    for a,b in [(1,15),(15,30),(30,60),(60,240)]:
        if f'post_{b}' in windows:
            wn.append(f'interval_{a}_{b}')
            values.append(weighted[:,:,windows.index(f'post_{b}'),:3]-weighted[:,:,windows.index(f'post_{a}'),:3])
    cube = np.stack(values,2)
    groups, inverse = np.unique(z['apps'],return_inverse=True)
    cluster = np.zeros((len(groups),*cube.shape[1:]))
    for k in range(len(groups)):
        cluster[k] = cube[inverse==k].sum(0)
    rng = np.random.default_rng(260915)
    draws = np.zeros((2000,len(groups)))
    for i in range(2000):
        draws[i] = np.bincount(rng.choice(len(groups),size=len(groups),replace=True),minlength=len(groups))
    boot = (draws@cluster.reshape(len(groups),-1)).reshape(2000,*cube.shape[1:])
    total = cube.sum(0)
    for r in [x for x in all_contrasts if x['case']==case]:
        ia=actions.index(f'{r["arm"]}__rho{r["rho"]:g}')
        ib=actions.index(f'{r["comparator"]}__rho{r["rho"]:g}')
        wi=wn.index(r['window']);rho=r['rho']
        outcomes=[];resamples=[]
        for idx in [ia,ib]:
            inv,cold,wm=total[idx,wi];sample=boot[:,idx,wi]
            outcomes.append([100*cold/inv,1000*wm/inv,1000*(wm+15*rho*cold)/inv])
            with np.errstate(divide='ignore',invalid='ignore'):
                resamples.append(np.array([100*sample[:,1]/sample[:,0],
                    1000*sample[:,2]/sample[:,0],1000*(sample[:,2]+15*rho*sample[:,1])/sample[:,0]]))
        for mi,metric in enumerate(['csr_pct','wm_per_1k','cost_per_1k']):
            av,bv=outcomes[0][mi],outcomes[1][mi]
            saved=r[metric]
            key=f'{case}/{r["arm"]}-{r["comparator"]}/{rho}/{r["window"]}/{metric}'
            check(key+'/points',[av,bv,av-bv],[saved['arm'],saved['comparator'],saved['difference']])
            check(key+'/difference CI',interval(resamples[0][mi]-resamples[1][mi]),saved['difference_ci95'])
            with np.errstate(divide='ignore',invalid='ignore'):
                ratio=resamples[0][mi]/resamples[1][mi]
            check(key+'/ratio',av/bv if bv>0 else None,saved['ratio'])
            check(key+'/ratio CI',interval(ratio),saved['ratio_ci95'])
    # Initial-window table CIs also use this paired resampling protocol.
    if case.startswith('initial_'):
        for rho in [1,10,100]:
            a=actions.index(f'WINTER__rho{rho}');b=actions.index(f'LSTM_Fifer__rho{rho}');w=wn.index('full')
            d=100*(boot[:,a,w,1]/boot[:,a,w,0]-boot[:,b,w,1]/boot[:,b,w,0])
            saved=summary['paired_bootstrap']['contrasts'][f'WINTER_vs_LSTM_Fifer__rho{rho}']['full']['winter_minus_comparator_csr_pp_ci95']
            check(f'{case}/{rho}/initial W-L CI',interval(d),saved)
    print(case, 'checked', flush=True)

# Verify every displayed control contrast against its raw per-function files.
controls = RUN/'controls'
comparisons = list(csv.DictReader((controls/'comparisons.csv').open()))
selection=read(controls/'selection.json')
representation=read(controls/'representation_selection.json')
control_details = []
for cohort in ['azure_primary','azure_evaluation','huawei']:
    with np.load(controls/'cohorts'/f'{cohort}.npz',allow_pickle=True) as z:
        apps=z['apps'];inv=z['counts'].sum(1)
    ids,codes=np.unique(apps,return_inverse=True)
    n=len(ids)
    samples=np.random.default_rng(260907).multinomial(n,np.full(n,1/n),10000)
    bi=samples@np.bincount(codes,weights=inv,minlength=n)
    for arm in ['component','gate']:
        chosen=next(c for c in selection['10.0'][arm]['choices'] if c['multiplier']==1)['config']['id']
        for condition,tag in [('main','main_rho10'),('strict','strict_rho10')]:
            paths=[controls/'evaluation'/cohort/tag/f'{name}.npz' for name in [arm,chosen]]
            if not all(p.exists() for p in paths):
                failures.append(dict(check=f'{cohort}/{condition}/{arm}/raw files',paths=list(map(str,paths))))
                continue
            arrays=[]
            for p in paths:
                with np.load(p) as z:
                    arrays.append(z['metrics'][:,:,:4].sum(2).mean(1))
            a,b=arrays
            cold=np.bincount(codes,weights=a[:,1]-b[:,1],minlength=n)
            am=np.bincount(codes,weights=a[:,2],minlength=n);bm=np.bincount(codes,weights=b[:,2],minlength=n)
            r=next(r for r in comparisons if r['cohort']==cohort and r['condition']==condition
                   and float(r['rho'])==10 and r['arm']==arm and float(r['budget_multiplier'])==1)
            check(f'{cohort}/{condition}/{arm}/CSR CI',interval(100*(samples@cold)/bi),[float(r['csr_ci_low_pp']),float(r['csr_ci_high_pp'])])
            check(f'{cohort}/{condition}/{arm}/WM CI',interval((samples@am)/(samples@bm)),[float(r['wm_ratio_ci_low']),float(r['wm_ratio_ci_high'])])
            control_details.append(dict(cohort=cohort,condition=condition,arm=arm,cold_per_function=float((a[:,1]-b[:,1]).mean())))

live=read(RUN/'strict_round/live/results.json')
for name,v in live['arms'].items():
    check('live/'+name+'/CSR',[100*v['csr_event'],100*v['csr_latency']],
          [100*v['cold_event']/v['invocations'],100*v['cold_latency']/v['invocations']])
    check('live/'+name+'/HTTP 200',v['invocations'],v['http_status_counts']['200'])
    check('live/'+name+'/failed',v['failed_requests'],0)
bench=read(PAPER.parent/'serverless-fewshot/results/lstm_comparison_v1/benchmark.json')
for v in bench['rows']:
    check(f'benchmark/{v["model"]}/{v["device"]}/{v["batch"]}',
          1000*np.median(v['samples_ms'])/v['batch'],v['median_us_per_function'])
cdf=read(PAPER.parent/'serverless-fewshot/results/runs/testbed_cdf.json')
cdf_statistics={kind:{k:dict(n=len(v),median=float(np.median(v))) for k,v in cdf[kind].items()} for kind in ['cold','warm']}
log=np.log(np.concatenate(list(cdf['cold'].values())))
cdf_statistics['lognormal_fit']={'mu':float(log.mean()),'sigma':float(log.std())}

result=dict(status='failed' if failures else 'passed',cases=len(names),paired_contrasts=len(all_contrasts),
    checks=len(records),failures=failures,records=records,sources=sources,controls=control_details,
    latency_calibration=cdf_statistics,experiments_rerun=0)
(OUT/'numerical_checks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ['status','cases','paired_contrasts','checks','failures']},indent=2))
