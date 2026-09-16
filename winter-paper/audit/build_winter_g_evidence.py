"""Generate matched WINTER/WINTER-G evidence from paper-local records.

--export-curves imports the verified matched campaign's curve CSV; generation
and verification use only the paper's retained CSV/JSON records. No fitting,
simulation, confidence-interval recomputation or policy selection occurs here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'audit/fixed_ewma_2026-09-16'
MATCHED = DATA/'drift'
PHASE = DATA/'nonshift.json'
CASES = [f'drift_{p}_{s}' for p in ('azure2019','azure2021','huawei') for s in ('natural','synthetic')]
METHODS = ('WINTER','WINTER_G','LSTM_Fifer','LSTM_shared','EWMA_0.3','No_handoff')
PROVIDERS = {'azure2019':'Azure 2019','azure2021':'Azure 2021','huawei':'Huawei'}
NAMES = {'WINTER':'WINTER','WINTER_G':'WINTER-G','LSTM_Fifer':'LSTM--Fifer','LSTM_shared':'LSTM--shared','EWMA_0.3':'EWMA','No_handoff':'No age hand-off'}
COLORS = {'WINTER':'#222222','WINTER_G':'#0077BB','LSTM_Fifer':'#AA3377','LSTM_shared':'#EE7733','EWMA_0.3':'#009988','No_handoff':'#444444'}
MANIFEST = DATA/'paper_evidence_manifest.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def label(case):
    _,p,s=case.split('_')
    return PROVIDERS[p], 'Natural' if s=='natural' else 'Injected'
def only(rows,case,window,comp,rho,arm=None):
    found=[r for r in rows if r['case']==case and r['window']==window and r['comparator']==comp and r['rho']==rho and (arm is None or r['arm']==arm)]
    assert len(found)==1,(case,window,comp,rho,len(found))
    return found[0]
def interval(v,d=4):return '['+', '.join(f'{x:.{d}f}' for x in v)+']' if v else '--'


def load():
    p=read(PHASE);t=read(MATCHED/'analysis.json');done=read(DATA/'completion.json')
    assert t['status']==p['status']==done['status']=='complete'
    assert t['ewma_alpha']==p['ewma_alpha']==done['ewma_alpha']==.3
    assert len(t['comparisons'])==1080 and len(t['operating_points'])==648
    assert {r['case'] for r in t['cases']}==set(CASES)
    assert read(DATA/'verification.json')['status']=='passed'
    for filename,expected in done['paper_sources'].items():assert sha(DATA/filename)==expected,filename
    assert t['protocol_sha256']==p['protocol_sha256']==sha(DATA/'protocol.json')
    for rows,key in ((p['comparisons'],'arm'),(t['comparisons'],'arm')):
        for r in rows:
            for side in (key,'comparator'):
                np.testing.assert_allclose(r['cost_per_1k'][side],r['wm_per_1k'][side]+150*r['rho']*r['csr_pct'][side],rtol=2e-12,atol=1e-7)
            for metric in ('csr_pct','wm_per_1k','cost_per_1k'):
                v=r[metric]
                np.testing.assert_allclose(v['difference'],v[key]-v['comparator'],rtol=2e-12,atol=1e-8)
                if v['comparator']>0:np.testing.assert_allclose(v['ratio'],v[key]/v['comparator'],rtol=2e-12)
    return t,p,t


def export_curves():
    """Import the verified matched campaign's already aggregated curve data."""
    import shutil
    source=ROOT.parent/'serverless-fewshot/results/fixed_ewma_v1/analysis/figure_curves.csv'
    assert sha(source)==read(DATA/'completion.json')['paper_sources']['drift/figure_curves.csv']
    shutil.copy2(source,MATCHED/'figure_curves.csv')


def operating_point(data,case,method,rho=10,window='post_240'):
    rows=[r for r in data['operating_points'] if r['case']==case and r['method']==method and r['rho']==rho and r['window']==window]
    assert len(rows)==1
    return rows[0]


def curves_and_checks(d):
    assert sha(MATCHED/'figure_curves.csv')==read(DATA/'completion.json')['paper_sources']['drift/figure_curves.csv']
    with (MATCHED/'figure_curves.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==8640
    curves={};errors=[]
    for case in CASES:
        for m in METHODS:
            rs=[r for r in rows if r['case']==case and r['method']==m]
            assert [int(r['minute']) for r in rs]==list(range(1,241))
            arr=np.array([[float(r[k]) for k in ('invocations','cold','idle')] for r in rs])
            assert (arr>=0).all() and (arr[:,1]<=arr[:,0]).all()
            inv,cold,idle=arr.T
            csr=np.divide(100*cold,inv,out=np.full(240,np.nan),where=inv>0)
            cost=np.divide(1000*(idle+150*cold),inv,out=np.full(240,np.nan),where=inv>0)
            curves[case,m]=(csr,cost)
            for minute in (1,15,30,60,240):
                r=operating_point(d,case,m,10,f'post_{minute}')
                np.testing.assert_allclose(csr[minute-1],r['csr_pct'],rtol=0,atol=1e-10)
                np.testing.assert_allclose(cost[minute-1],r['cost_per_1k'],rtol=1e-6,atol=0)
                errors.append(abs(cost[minute-1]/r['cost_per_1k']-1))
    return curves,max(errors)


def generate():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
                         'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42})
    d,p,t=load();curves,error=curves_and_checks(t);outputs={};figures={}
    preview=DATA/'paper_visual';preview.mkdir(exist_ok=True)
    def table(name,rows):
        path=ROOT/'generated'/f'winter_g_{name}_rows.tex'
        path.write_text('\n'.join(' & '.join(r)+r' \\' for r in rows)+'\n');outputs[str(path.relative_to(ROOT))]=sha(path)
    def save(fig,name):
        path=ROOT/(name+'.pdf');fig.savefig(path,bbox_inches='tight',pad_inches=.06)
        fig.savefig(preview/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.06);plt.close(fig)
        outputs[path.name]=figures[path.name]=sha(path)
    # The continuous table includes primary intervals and the full existing rho grid.
    continuous=[]
    conditions=[(10,w,c) for w in ('young','mature','full') for c in ('LSTM_Fifer','LSTM_shared','No_handoff')]
    conditions += [(rho,'full',c) for rho in (1,100) for c in ('LSTM_Fifer','LSTM_shared')]
    for rho,w,c in conditions:
        r=only(p['comparisons'],'continuous_48h',w,c,rho);s=r['csr_pct'];v=r['cost_per_1k']
        continuous.append([str(rho),{'young':'0--12','mature':'12--48','full':'0--48'}[w],NAMES[c],f"{s['difference']:+.4f}",interval(s['difference_ci95']),f"{v['ratio']:.3f}",interval(v['ratio_ci95'],3)])
    table('continuous',continuous)
    endpoints=[];contrasts=[];wcontrasts=[];timing=[];routing=[];main=[];reference=[]
    for case in CASES:
        provider,kind=label(case)
        for rho in (1,10):
            ps=[operating_point(t,case,m,rho) for m in ('WINTER','WINTER_G','LSTM_Fifer','LSTM_shared')]
            endpoints.append([provider,kind,str(rho)]+[f"{r['csr_pct']:.4f}" for r in ps]+[f"{r['cost_per_1k']:,.0f}" for r in ps])
        for arm,comparators,target in (('WINTER_G',('WINTER','LSTM_Fifer','LSTM_shared','No_handoff','EWMA_0.3'),contrasts),('WINTER',('LSTM_Fifer','LSTM_shared','EWMA_0.3'),wcontrasts)):
            for c in comparators:
                r=only(t['comparisons'],case,'post_240',c,10,arm);v=r['cost_per_1k'];cs=r['csr_pct']
                digits=3
                target.append([provider,kind,NAMES[c],f"{cs['difference']:+.4f}",interval(cs['difference_ci95']),f"{v['ratio']:.3f}",interval(v['ratio_ci95'],digits)])
            r=only(t['comparisons'],case,'post_15','LSTM_Fifer',10,arm);cs=r['csr_pct'];v=r['cost_per_1k']
            timing.append([provider,kind,NAMES[arm],f"{cs['arm']:.4f}",f"{cs['comparator']:.4f}",interval(cs['difference_ci95']),f"{v['ratio']:.3f}",interval(v['ratio_ci95'],3)])
        rt=d['routing'][case];m=rt['post_function_minutes'];iv=rt['post_invocations_by_route']
        routing.append([provider,kind,str(r['n_clusters']),f"{rt['onset_functions']['2']}/{r['n_event_windows']}",f"{100*m['2']/sum(m.values()):.3f}",f"{100*iv['2']/sum(iv.values()):.3f}"])
        ps=[operating_point(t,case,m) for m in ('WINTER','WINTER_G','LSTM_Fifer')]
        main.append([provider,kind]+[f"{r['csr_pct']:.4f}" for r in ps]+[f"{r['cost_per_1k']/ps[2]['cost_per_1k']:.3f}" for r in ps[:2]]+['--' if r['adaptation_lag'] is None else str(r['adaptation_lag']) for r in ps])
        reference.append([provider,kind]+[f"{r['reference_csr_pct']:.4f}" for r in ps])
    for name,rows in (('shift_endpoints',endpoints),('shift_contrasts',contrasts),('winter_contrasts',wcontrasts),('shift_timing',timing),('routing',routing),('matched_main',main),('matched_reference',reference)):table(name,rows)
    # Primary paired effects: all six conditions, three specified comparisons.
    fig,axes=plt.subplots(1,2,figsize=(7.16,3.3),sharey=True)
    comps=('LSTM_Fifer','LSTM_shared','WINTER')
    for ax,metric,ref in zip(axes,('csr_pct','cost_per_1k'),(0,1)):
        for j,c in enumerate(comps):
            for i,case in enumerate(CASES):
                v=only(t['comparisons'],case,'post_240',c,10,'WINTER_G')[metric]
                key='difference' if metric=='csr_pct' else 'ratio';point=v[key];lo,hi=v[key+'_ci95'];y=i+(j-1)*.23
                ax.plot([lo,hi],[y,y],color=COLORS[c],lw=1.1)
                ax.plot(point,y,('o','s','^')[j],color=COLORS[c],ms=3.8,label=NAMES[c].replace('--','–') if i==0 else None)
        ax.axvline(ref,c='#666666',ls=':',lw=.8);ax.grid(axis='x',alpha=.2);ax.set_ylim(5.55,-.55)
    axes[0].set_yticks(range(6),[a+' / '+b for a,b in map(label,CASES)])
    axes[0].set_xlabel('CSR difference (pp)');axes[1].set_xlabel('Modeled cost ratio')
    axes[0].set_title('(a) WINTER-G minus comparator');axes[1].set_title('(b) WINTER-G / comparator')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1),w_pad=1.6);save(fig,'fig21_winter_g_shift_effects')
    for mi,metric in enumerate(('csr','cost')):
        fig,axes=plt.subplots(3,2,figsize=(7.16,6.3),sharex=True)
        for ax,case in zip(axes.flat,CASES):
            for m in METHODS:
                ax.plot(np.arange(1,241),curves[case,m][mi],color=COLORS[m],lw=1.15,ls={'LSTM_shared':'--','EWMA_0.3':':','No_handoff':'-.'}.get(m,'-'),label=NAMES[m].replace('--','–'))
            a,b=label(case);ax.set_title(a+' / '+b,fontsize=10);ax.set_xlim(1,240);ax.grid(alpha=.2)
            ax.tick_params(labelsize=8.5)
        for ax in axes[-1]:ax.set_xlabel('Minutes after shift',fontsize=9)
        fig.supylabel('Cumulative CSR (%)' if mi==0 else 'Cumulative modeled cost (GB·s / 1,000 invocations)',fontsize=9)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=8.5,frameon=False)
        fig.tight_layout(rect=(.015,.065,1,1),h_pad=1.5,w_pad=1.5)
        save(fig,f'fig{22+mi}_winter_g_shift_{metric}')
    sources=[DATA/n for n in ('protocol.json','completion.json','verification.json')]+[PHASE,Path(__file__)]
    sources += [MATCHED/n for n in ('analysis.json','comparisons.csv','operating_points.csv','figure_curves.csv')]
    result=dict(status='passed',sources={str(s.relative_to(ROOT)):sha(s) for s in sources},outputs=outputs,figures=figures,
                additional_replay_cases=6,matched_winter_replay_cases=6,matched_winter_validation='passed',original_replay_cases=19,paired_comparisons=1080,prior_gate_comparisons=0,continuous_table_rows=len(continuous),
                ewma_alpha=.3,affected_replay_cases=19,matched_winter_reused_cases=6,
                checks={'cost_identity':'passed','difference_and_ratio_direction':'passed','curve_csr_endpoint_absolute_tolerance':1e-10,
                        'curve_cost_endpoint_relative_tolerance':1e-6,'max_curve_cost_endpoint_relative_error':error,
                        'all_six_conditions_and_two_rhos_present':True},
                uncertainty='Saved 2000-replicate exploratory marginal paired cluster CIs; no multiplicity or source-training adjustment')
    write(MANIFEST,result)
    print('Generated WINTER-G tables and figures; maximum cost-curve endpoint relative error',error)
    return result


def verify():
    m=read(MANIFEST);assert m['status']=='passed' and m['additional_replay_cases']==6
    for path,digest in {**m['sources'],**m['outputs']}.items():assert sha(ROOT/path)==digest,path
    d,p,t=load();curves_and_checks(t)
    return m


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export-curves',action='store_true');parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.export_curves:export_curves()
    if args.verify:verify();print('WINTER-G evidence verified')
    else:generate()
