"""Read-only active-pool, live replay and predictor-record audit."""
from collections import defaultdict
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from verify_submission_data import SF, PAPER, RUNS, OUT, read, write, close, csv_write


def main():
    result, allrows = {}, []
    for name in ('revision_r16_faithful_primary_experiment.json','revision_r17_faithful_gates_2021.json'):
        d=read(RUNS/name)
        for r in d['results']:
            for vector,total in [('func_cold','cold_starts'),('func_total','total_invocations'),('func_wm','wm_total_gb_s')]:
                close(sum(r[vector]),r[total])
            close(r['cold_starts']/r['total_invocations'],r['csr'])
            close(1000*r['wm_total_gb_s']/r['total_invocations'],r['wm_per_1k_inv'])
        allrows.extend(d['results'])
        result[name]=dict(seed_records=len(d['results']),arithmetic_matches=True)
    r18=read(RUNS/'revision_r18_faithful_primary_readouts.json')
    checked=0
    for split,arms in r18['table1'].items():
        for arm,rec in arms.items():
            rr=[r for r in allrows if r['split']==split and r['method']==arm and float(r['cost_ratio'])==10]
            if not rr:
                continue
            close(np.mean([r['csr'] for r in rr])*100,rec['csr_pct'])
            close(np.std([r['csr'] for r in rr])*100,rec['csr_std_pct'])
            close(np.mean([r['wm_per_1k_inv'] for r in rr]),rec['wm'])
            checked+=1
    result['r18_current_head_rows']=checked
    r19=read(RUNS/'revision_r19_gate_state_diagnostics.json')
    for r in r19['results']:
        close(r['cold_starts']/r['total_invocations'],r['csr'])
        close(1000*r['wm_total_gb_s']/r['total_invocations'],r['wm_per_1k_inv'])
        close(r['wm_total_gb_s']+15*r['cost_ratio']*r['cold_starts'],r['registered_cost_gb_s_no_state'])
    saved=pd.read_csv(SF/'results/tables/T_r19_mature_clean_rho10.csv')
    for row in saved.itertuples():
        rr=[r['by_state']['Z0' if row.phase=='reset_washout' else 'W'] for r in r19['results']
            if r['experiment']=='mature_clean' and r['method']==row.method and r['split']==row.split and float(r['cost_ratio'])==10]
        inv=np.mean([r['total_invocations'] for r in rr]);cold=np.mean([r['cold_starts'] for r in rr]);wm=np.mean([r['wm_total_gb_s'] for r in rr])
        close(100*cold/inv,row.csr_pct,atol=0.00000051,rtol=0)
        close(1000*wm/inv,row.wm_per_1k_inv,atol=.00051,rtol=0)
    result['r19']=dict(seed_records=len(r19['results']),arithmetic_matches=True,mature_phase_rows_checked=len(saved),
                      note='Mature comparison uses post_washout_120min; whole-window readout is a different estimand.')
    live=read(RUNS/'testbed_cohort.json'); mirror=read(RUNS/'testbed_cohort_desmirror.json')
    out=[]
    for arm,r in live['arms'].items():
        close(sum(x['invocations'] for x in r['per_function']),r['invocations'])
        close(sum(x['cold_event'] for x in r['per_function']),r['cold_event'])
        close(sum(x['pod_seconds'] for x in r['per_function']),r['pod_seconds'])
        close(r['cold_event']/r['invocations'],r['csr_event'])
        p=mirror['predicted']['registered'][arm]
        close(sum(p['func_total']),686)
        close(sum(p['func_cold'])/sum(p['func_total']),p['csr'])
        out.append(dict(arm=arm,invocations=r['invocations'],cold_event=r['cold_event'],
                        live_csr_pct=100*r['csr_event'],pod_seconds=r['pod_seconds'],
                        des_csr_pct=100*p['csr'],cold_latency_classifier=r['cold_latency']))
    csv_write('live_rechecked.csv',out)
    cdf=read(RUNS/'testbed_cdf.json'); cdf_rows=[]
    for condition in ('cold','warm'):
        for runtime,values in cdf[condition].items():
            a=np.asarray(values)
            assert np.isfinite(a).all() and np.min(a)>=0
            cdf_rows.append(dict(condition=condition,runtime=runtime,n=len(a),median_seconds=float(np.median(a))))
    csv_write('latency_samples_rechecked.csv',cdf_rows)
    result['live']=dict(arms=len(out),functions=live['n_functions'],invocations_per_arm=686,raw_latency_samples=sum(r['n'] for r in cdf_rows),all_stored_arithmetic_matches=True)
    r9=read(RUNS/'revision_r9_forecast_cost.json')
    close(r9['comparison']['chronos_small_us_per_forecast']/r9['results']['gpu_bf16']['10000'],r9['comparison']['honest_compute_multiplier'])
    result['overhead']=dict(record='revision_r9_forecast_cost.json',recorded_timing_cells=sum(len(v) for v in r9['results'].values()),
                            medians=r9['results'],individual_repetition_samples_available=False,
                            interpretation='Body forward plus head read; different batches and operations cannot establish an end-to-end speedup.')
    # Keep the manuscript checker's normal report in this audit folder.
    p=PAPER/'audit/check_manuscript.py'
    code=p.read_text().replace('(ROOT / "audit/source_checks.json").write_text','(OUTPUT_DIR / "manuscript_checks.json").write_text')
    exec(compile(code,str(p),'exec'),{'__file__':str(p),'OUTPUT_DIR':OUT})
    result['manuscript_source_checks']='passed; no PDF rebuild and no scientific validity certification'
    write('active_live_verification.json',result)
    print('PASS active/live/predictor/source checks',flush=True)


if __name__=='__main__':
    main()
