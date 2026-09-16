"""Compact audit of changed operating points and claims used in the manuscript."""
import csv
from campaign import OUT,OLD,GATE,WINTER,ALL_CASES,DRIFT
from common import read_json,write_json,digest


def main():
    rows=[]
    for name in ALL_CASES:
        current=read_json(OUT/'cases'/name/'summary.json')['by_action']
        original=read_json(OLD/'cases'/name/'summary.json')['by_action']
        if name in DRIFT:
            original.update(read_json(GATE/'cases'/name/'summary.json')['by_action'])
            original.update(read_json(WINTER/'cases'/name/'summary.json')['by_action'])
        window='post_240' if name in DRIFT else 'full'
        for method in ('WINTER','WINTER_G','No_handoff','EWMA_0.3','LSTM_Fifer','LSTM_shared'):
            key=method+'__rho10'
            if key not in current:continue
            previous=('EWMA_0.1__rho10' if method=='EWMA_0.3' and not name.startswith('initial_') else key)
            before=original[previous];after=current[key]
            row=dict(case=name,method=method,window=window,rho=10)
            for k in ('csr_pct','idle_per_1k','cost_per_1k'):
                row[k+'_before']=before['windows'][window][k]
                row[k+'_after']=after['windows'][window][k]
            row.update(al_before=before['adaptation_lag'],al_after=after['adaptation_lag'])
            rows.append(row)
    with (OUT/'analysis/revision_changes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(ewma_alpha=.3,operating_points=rows,primary_drift=[],continuous=[],controls=[],live={})
    for r in read_json(OUT/'analysis/drift.json')['comparisons']:
        if r['rho']==10 and r['window'] in ('post_15','post_240') and r['arm']=='WINTER_G':
            report['primary_drift'].append(r)
    for r in read_json(OUT/'analysis/nonshift.json')['comparisons']:
        if r['case']=='continuous_48h' and r['window'] in ('young','mature','full'):
            report['continuous'].append(r)
    selected=read_json(OUT/'controls/selection.json')
    report['control_configurations']={str(rho):{arm:next(c for c in selected[str(float(rho))][arm]['choices'] if c['multiplier']==1.)
        for arm in ('component','gate')} for rho in (1,10,100)}
    with (OUT/'controls/comparisons.csv').open() as f:
        for r in csv.DictReader(f):
            if r['rho']=='10.0' and r['budget_multiplier']=='1.0' and r['condition'] in ('main','strict'):
                report['controls'].append(r)
    live=read_json(OUT/'strict_round/live/results.json')
    report['live']={arm:{k:v for k,v in d.items() if k not in ('requests','records','invocations_detail')}
                    for arm,d in live['arms'].items()}
    report['sources']={str(p):digest(p) for p in (OUT/'analysis/drift.json',OUT/'analysis/nonshift.json',
                       OUT/'controls/selection.json',OUT/'controls/comparisons.csv',OUT/'strict_round/live/results.json')}
    write_json(OUT/'analysis/manuscript_audit.json',report)
    for r in rows:
        if r['method'] in ('WINTER_G','No_handoff') or (r['method']=='WINTER' and r['case'].startswith('steady_')):
            print(r['case'],r['method'],'CSR',round(r['csr_pct_before'],5),'->',round(r['csr_pct_after'],5),
                  'WM',round(r['idle_per_1k_before'],2),'->',round(r['idle_per_1k_after'],2),
                  'Cost',round(r['cost_per_1k_before'],2),'->',round(r['cost_per_1k_after'],2),
                  'AL',r['al_before'],'->',r['al_after'])
    print('Primary control configurations:',report['control_configurations']['10'])


if __name__=='__main__':main()
