"""Check every generated table included in the submitted documents."""
from pathlib import Path
import csv, json, re

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=ROOT.parent/'serverless-fewshot/results/fixed_ewma_v1'
read=lambda p:json.loads(p.read_text())
summaries={p.parent.name:read(p) for p in (RUN/'cases').glob('*/summary.json')}
phase=read(ROOT/'audit/fixed_ewma_2026-09-16/nonshift.json')
drift=read(ROOT/'audit/fixed_ewma_2026-09-16/drift/analysis.json')
controls=list(csv.DictReader((RUN/'controls/comparisons.csv').open()))
endpoints=list(csv.DictReader((RUN/'controls/endpoints.csv').open()))
selections=read(RUN/'controls/selection.json')
bench=read(ROOT.parent/'serverless-fewshot/results/lstm_comparison_v1/benchmark.json')
live=read(RUN/'strict_round/live/results.json')
names={'\\sys{}':'WINTER','\\sysG{}':'WINTER_G','WINTER':'WINTER','WINTER-G':'WINTER_G',
       'LSTM--Fifer':'LSTM_Fifer','LSTM--shared':'LSTM_shared','EWMA':'EWMA_0.3',
       'Hybrid histogram':'Hybrid','Spectral':'Fourier','Chronos-Bolt':'Chronos','No age hand-off':'No_handoff'}
cohorts={'Azure, 100':'azure_primary','Azure, 970':'azure_evaluation','Huawei, 76':'huawei',
         'Initial A19, 100':'azure_primary','Initial A19, 970':'azure_evaluation','Initial Huawei, 76':'huawei'}
fmt=lambda x,d=3:'--' if x is None else f'{float(x):.{d}f}'
interval=lambda x,d=4,signed=False:'['+', '.join(format(float(v),('+' if signed else '')+f'.{d}f') for v in x)+']'
def one(rows):
    assert len(rows)==1,len(rows)
    return rows[0]
def point(case,m,rho=10,window='full'):
    return summaries[case]['by_action'][f'{m}__rho{rho:g}']['windows'][window]
def lag(case,m,rho=10):
    a=summaries[case]['by_action'][f'{m}__rho{rho:g}']['adaptation_lag']
    return '--' if a is None else str(a)
def contrast(data,case,arm,comp,rho=10,window='full'):
    return one([x for x in data['comparisons'] if x['case']==case and x['arm']==arm
        and x['comparator']==comp and x['rho']==rho and x['window']==window])
def ccase(row):return 'drift_'+{'Azure 2019':'azure2019','Azure 2021':'azure2021','Huawei':'huawei'}[row[0]]+'_'+{'Natural':'natural','Injected':'synthetic'}[row[1]]
def control(cohort,arm):return one([r for r in endpoints if r['cohort']==cohort and r['arm']==arm and r['tag']=='main_rho10' and r['interval']=='first240'])
def expected(file,row,case):
    if file=='lstm_initial_rows.tex':
        m=names[row[0]];v=point(case,m)
        return row[:1]+[fmt(point(case,m,window=w)['csr_pct']) for w in ['post_15','post_60','full']]+[lag(case,m),f'{v["idle_per_1k"]:,.0f}',f'{v["cost_per_1k"]:,.0f}']
    if file=='lstm_holdout_rows.tex':
        v=point('continuous_48h',names[row[0]])
        return row[:1]+[fmt(v['csr_pct'],4),f'{v["idle_per_1k"]:,.0f}',f'{v["cost_per_1k"]:,.0f}']
    if file=='lstm_initial_grid_rows.tex':
        c='initial_'+cohorts[row[0]];r=float(row[1]);w=point(c,'WINTER',r);l=point(c,'LSTM_Fifer',r)
        ci=summaries[c]['paired_bootstrap']['contrasts'][f'WINTER_vs_LSTM_Fifer__rho{r:g}']['full']['winter_minus_comparator_csr_pp_ci95']
        return row[:2]+[fmt(w['csr_pct']),fmt(l['csr_pct']),fmt(w['idle_per_1k']/l['idle_per_1k'],2),interval(ci,3,True),lag(c,'WINTER',r),lag(c,'LSTM_Fifer',r)]
    if file=='lstm_reference_rows.tex':
        refs=read(ROOT/'audit/lstm_comparison/timing_references.json');c='initial_'+cohorts[row[0]];cells=[]
        for m in ['WINTER','LSTM_Fifer']:
            p=one([x for x in refs if x['case']==c and x['method']==m]);cells += [fmt(p['reference_csr_pct']),lag(c,m)]
        return row[:1]+cells
    if file=='final_timing_rows.tex':
        c='initial_'+cohorts[row[0]]
        return row[:1]+[fmt(point(c,m,window=t)['csr_pct']) for t in ['post_1','post_15','full'] for m in ['WINTER','LSTM_Fifer']]
    if file=='final_strong_controls_rows.tex':
        c=cohorts[row[0]];choice=next(x for x in selections['10.0']['component']['choices'] if x['multiplier']==1)['config']['id']
        w,e=control(c,'component'),control(c,choice)
        return row[:1]+[fmt(w['csr_pct']),fmt(e['csr_pct']),f'{float(w["wm_per_1k"]):,.0f}',f'{float(e["wm_per_1k"]):,.0f}']
    if file in ['final_control_intervals_rows.tex','final_representation_rows.tex']:
        representation=file=='final_representation_rows.tex';c=cohorts[row[0]]
        rs=[r for r in controls if r['cohort']==c and float(r['rho'])==10 and r['condition']==('representation' if representation else 'main')]
        if not representation:rs=[r for r in rs if r['arm']==('component' if row[1]=='WINTER' else 'gate') and float(r['budget_multiplier'])==1]
        r=one(rs);cells=[f'{float(r["csr_difference_pp"]):+.4f}',interval([r['csr_ci_low_pp'],r['csr_ci_high_pp']],4,True),fmt(r['wm_ratio'])]
        if not representation:cells.append(interval([r['wm_ratio_ci_low'],r['wm_ratio_ci_high']],3))
        return row[:1 if representation else 2]+cells
    if file=='final_handoff_intervals_rows.tex':
        r=contrast(phase,'continuous_48h','WINTER_G',names[row[0]])
        return row[:1]+[f'{r["csr_pct"]["difference"]:+.4f}',interval(r['csr_pct']['difference_ci95'],4,True),fmt(r['wm_per_1k']['ratio']),interval(r['wm_per_1k']['ratio_ci95'],3)]
    if file=='winter_g_continuous_rows.tex':
        w={'0--12':'young','12--48':'mature','0--48':'full'}[row[1]]
        r=contrast(phase,'continuous_48h','WINTER_G',names[row[2]],float(row[0]),w)
        return row[:3]+[f'{r["csr_pct"]["difference"]:+.4f}',interval(r['csr_pct']['difference_ci95']),fmt(r['cost_per_1k']['ratio']),interval(r['cost_per_1k']['ratio_ci95'],3)]
    if file.startswith('winter_g_'):
        c=ccase(row)
        if file in ['winter_g_shift_contrasts_rows.tex','winter_g_winter_contrasts_rows.tex','winter_g_shift_timing_rows.tex']:
            early=file=='winter_g_shift_timing_rows.tex';arm=names[row[2]] if early else ('WINTER' if file=='winter_g_winter_contrasts_rows.tex' else 'WINTER_G')
            r=contrast(drift,c,arm,'LSTM_Fifer' if early else names[row[2]],window='post_15' if early else 'post_240')
            head=[fmt(r['csr_pct'][k],4) for k in ['arm','comparator']] if early else [f'{r["csr_pct"]["difference"]:+.4f}']
            return row[:3]+head+[interval(r['csr_pct']['difference_ci95']),fmt(r['cost_per_1k']['ratio']),interval(r['cost_per_1k']['ratio_ci95'],3)]
        if file=='winter_g_routing_rows.tex':
            r=drift['routing'][c];m=r['post_function_minutes'];v=r['post_invocations_by_route'];x=contrast(drift,c,'WINTER_G','LSTM_Fifer',window='post_240')
            return row[:2]+[str(x['n_clusters']),f'{r["onset_functions"]["2"]}/{x["n_event_windows"]}',fmt(100*m['2']/sum(m.values())),fmt(100*v['2']/sum(v.values()))]
        rho=float(row[2]) if file=='winter_g_shift_endpoints_rows.tex' else 10
        methods=['WINTER','WINTER_G','LSTM_Fifer']+(['LSTM_shared'] if file=='winter_g_shift_endpoints_rows.tex' else [])
        ps=[one([r for r in drift['operating_points'] if r['case']==c and r['method']==m and r['rho']==rho and r['window']=='post_240']) for m in methods]
        if file=='winter_g_matched_reference_rows.tex':return row[:2]+[fmt(r['reference_csr_pct'],4) for r in ps]
        if file=='winter_g_shift_endpoints_rows.tex':return row[:3]+[fmt(r['csr_pct'],4) for r in ps]+[f'{r["cost_per_1k"]:,.0f}' for r in ps]
        if file=='winter_g_matched_main_rows.tex':return row[:2]+[fmt(r['csr_pct'],4) for r in ps]+[fmt(r['cost_per_1k']/ps[2]['cost_per_1k']) for r in ps[:2]]+['--' if r['adaptation_lag'] is None else str(r['adaptation_lag']) for r in ps]
    if file=='lstm_steady_grid_rows.tex':
        label,split=row[0].split();p={'A21':'azure2021','A19':'azure2019','Huawei':'huawei'}[label]
        c='steady_'+p+'_'+('h_'+split if p=='huawei' else split);rho=float(row[1]);ms=['EWMA_0.3','LSTM_Fifer','WINTER','WINTER_G'];vs=[point(c,m,rho) for m in ms]
        return row[:2]+[fmt(v['csr_pct']) for v in vs]+[fmt(v['idle_per_1k']/vs[0]['idle_per_1k'],2) for v in vs[1:]]
    if file=='lstm_live_rows.tex':
        arm={'Reactive':'reactive','Keep-alive':'keepalive10','EWMA':'ewma','Learned schedule':'protowarm','LSTM--Fifer schedule':'lstm_fifer'}[row[0]]
        v=live['arms'][arm]
        return row[:1]+[fmt(100*v['cold_event']/v['invocations']),fmt(100*v['cold_latency']/v['invocations']),f'{v["pod_seconds"]:,.0f}',str(v['failed_requests'])]
    if file=='lstm_overhead_rows.tex':
        model={'WINTER body/read':'WINTER_body_scalar_read','LSTM peak':'LSTM_peak'}[row[1]]
        return row[:2]+[fmt(one([r for r in bench['rows'] if r['batch']==int(row[0]) and r['model']==model and r['device']==device])['median_us_per_function'],2) for device in ['cpu','cuda']]
    raise ValueError(file)

files=set()
for p in [ROOT/'main.tex',ROOT/'supplementary.tex',*ROOT.glob('sections/*.tex'),*ROOT.glob('supplement/*.tex')]:
    files.update(re.findall(r'\\tableinput\s+(generated/\S+)',p.read_text()))
report=[];failed=[];total=0
for f in sorted(files):
    rows=[];case=None
    for i,line in enumerate((ROOT/f).read_text().splitlines(),1):
        if '\\multicolumn' in line:
            case='initial_huawei' if 'Huawei' in line else 'initial_azure_primary'
        if ' & ' not in line:continue
        row=[x.strip() for x in line.removesuffix(' \\\\').split(' & ')]
        exp=expected(Path(f).name,row,case);ok=row==exp
        rows.append(dict(line=i,passed=ok,cells=len(row)));total+=len(row)
        if not ok:failed.append(dict(file=f,line=i,actual=row,expected=exp))
    report.append(dict(file=f,rows=len(rows),cells=sum(r['cells'] for r in rows),passed=all(r['passed'] for r in rows)))
result=dict(status='failed' if failed else 'passed',files=len(files),rows=sum(r['rows'] for r in report),cells=total,tables=report,failures=failed)
(OUT/'table_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
