"""Read-only independent audit; requires the full measurement-data archive and NumPy.
Only the adjacent audit JSON is written. Run from the release or workspace tree.
"""
import json, hashlib, re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]/'serverless-fewshot/results/lstm_comparison_v1/cases'
OUT=Path(__file__).with_name('four_issue_verification.json')
report={'scope':'Independent read-only recalculation from all saved function statistics, plus all published paired bootstrap contrasts; no replay or measurement mutation.','cases':[]}
for d in sorted(ROOT.iterdir()):
 if not (d/'summary.json').exists(): continue
 s=json.loads((d/'summary.json').read_text()); meta=json.loads((d/'case.json').read_text())
 with np.load(d/'aggregate.npz') as z:
  names=z['action_names'].tolist(); windows=z['window_names'].tolist(); pf=z['per_function']; weights=z['weights']; apps=z['apps']; seed_totals=z['seed_totals']
 rawtot=np.zeros_like(seed_totals); raw_max=0.
 for f in range(len(pf)):
  with np.load(d/'functions'/f'{f:05d}.npz') as z:
   raw=z['stats'][:,z['mapping']]; direct=raw.mean(axis=0)
   np.testing.assert_allclose(direct,pf[f],rtol=1e-12,atol=1e-8)
   raw_max=max(raw_max,float(np.max(np.abs(direct-pf[f]))))
   rawtot+=weights[f]*raw
 np.testing.assert_allclose(rawtot,seed_totals,rtol=1e-12,atol=1e-7)
 totals=np.einsum('f,fawm->awm',weights,pf)
 points=0
 for a,name in enumerate(names):
  for w,win in enumerate(windows):
   r=s['by_action'][name]['windows'][win]; inv,cold,idle=totals[a,w,:3]
   if inv<=0: continue
   for k,v in {'csr_pct':100*cold/inv,'idle_per_1k':1000*idle/inv,'cost_per_1k':1000*(idle+15*meta['actions'][name]['rho']*cold)/inv}.items():
    np.testing.assert_allclose(v,r[k],rtol=1e-11,atol=1e-7);points+=1
 clusters,codes=np.unique(apps,return_inverse=True); cl=np.zeros((len(clusters),len(names),len(windows),2))
 np.add.at(cl,codes,pf[:,:,:,:2]*weights[:,None,None,None])
 # Same published bootstrap sample, independent multiplicity-matrix computation.
 rng=np.random.default_rng(260915); multiplicities=np.array([np.bincount(rng.integers(0,len(clusters),len(clusters)),minlength=len(clusters)) for _ in range(2000)])
 boot=(multiplicities @ cl.reshape(len(clusters),-1)).reshape(2000,len(names),len(windows),2)
 cis=0;err=0.
 for key,contrast in s['paired_bootstrap']['contrasts'].items():
  comp,rho=key.removeprefix('WINTER_vs_').split('__rho'); wi=names.index('WINTER__rho'+rho); ci=names.index(comp+'__rho'+rho)
  for win,row in contrast.items():
   k=windows.index(win); den=boot[:,wi,k,0]; good=den>0
   if not good.any(): assert not row;continue
   ci95=np.quantile(100*(boot[good,wi,k,1]-boot[good,ci,k,1])/den[good],[.025,.975])
   expected=row['winter_minus_comparator_csr_pp_ci95']; np.testing.assert_allclose(ci95,expected,rtol=1e-10,atol=1e-10)
   err=max(err,float(np.max(np.abs(ci95-expected))));cis+=1
 row={'case':d.name,'n_functions':len(pf),'n_clusters':len(clusters),'n_seeds':len(meta['seeds']),'point_values_checked':points,'intervals_checked':cis,'max_interval_error_pp':err,'raw_statistics_max_error':raw_max,'summary_sha256':hashlib.sha256((d/'summary.json').read_bytes()).hexdigest(),'aggregate_sha256':hashlib.sha256((d/'aggregate.npz').read_bytes()).hexdigest(),'model_source':meta['model_source']}
 if d.name.startswith('initial'):
  w=s['by_action']['WINTER__rho10']['windows']['full'];l=s['by_action']['LSTM_Fifer__rho10']['windows']['full']
  row['primary_comparison']={'winter_csr_pct':w['csr_pct'],'lstm_csr_pct':l['csr_pct'],'delta_pp':w['csr_pct']-l['csr_pct'],'idle_ratio':w['idle_per_1k']/l['idle_per_1k'],'ci95_pp':s['paired_bootstrap']['contrasts']['WINTER_vs_LSTM_Fifer__rho10']['full']['winter_minus_comparator_csr_pp_ci95']}
 if d.name=='continuous_48h':
  gate=s['by_action']['WINTER_G__rho10']['windows']['full'];keep=s['by_action']['No_handoff__rho10']['windows']['full'];ls=s['by_action']['LSTM_shared__rho10']['windows']['full']
  row['handoff']={'idle_reduction_pct':100*(1-gate['idle_per_1k']/keep['idle_per_1k']),'csr_increase_pp':gate['csr_pct']-keep['csr_pct'],'gate_csr':gate['csr_pct'],'gate_idle':gate['idle_per_1k'],'lstm_shared_csr':ls['csr_pct'],'lstm_shared_idle':ls['idle_per_1k']}
 report['cases'].append(row);print(d.name, 'PASS',points,'values',cis,'intervals',flush=True)
report['status']='passed';report['totals']={k:sum(x[k] for x in report['cases']) for k in ['n_functions','point_values_checked','intervals_checked']};report['totals']['cases']=len(report['cases']);OUT.write_text(json.dumps(report,indent=2)+'\n');print(report['totals'],flush=True)
