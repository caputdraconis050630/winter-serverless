"""Recheck archived experiment arithmetic without policy replay or mutation.

All output is confined to data_verification_2026-09-15. Scientific limitations
of historical DES and A1 scoring remain even when these checks pass.
"""
import ast
import bisect
import csv
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'winter-paper'
SF = ROOT / 'serverless-fewshot'
RUNS = SF / 'results/runs'
OUT = PAPER / 'audit/data_verification_2026-09-15'


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_write(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


def close(a, b, atol=1e-9, rtol=1e-10):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def compare(a, b, path=''):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            compare(a[k], b[k], path + '/' + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + '/' + str(i))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        close(a, b)
    else:
        assert a == b, (path, a, b)


def provenance_and_onboarding():
    expected = read(PAPER / 'audit/verified_provenance.json')
    hash_rows = []
    for name, sha in expected['source_sha256'].items():
        p = SF / 'scripts' / name
        hash_rows.append(dict(path=str(p.relative_to(ROOT)), expected=sha, actual=digest(p)))
    for name, rec in expected['checkpoints'].items():
        p = SF / 'results_azure2021/runs' / f'best_anil_ridge_{name}_s0.pt'
        hash_rows.append(dict(path=str(p.relative_to(ROOT)), expected=rec['sha256'], actual=digest(p)))
    for name, sha in read(PAPER / 'audit/nondrift_sources.json')['input_sha256'].items():
        hash_rows.append(dict(path=name, expected=sha, actual=digest(ROOT / name)))
    assert all(r['expected'] == r['actual'] for r in hash_rows)
    csv_write('manuscript_source_hashes.csv', hash_rows)
    # The archived checker writes one JSON. Redirect only that destination.
    p = PAPER / 'audit/verify_provenance.py'
    code = p.read_text().replace('(HERE / "verified_provenance.json").write_text',
                                '(OUTPUT_DIR / "verified_provenance.json").write_text')
    exec(compile(code, str(p), 'exec'), {'__file__': str(p), 'OUTPUT_DIR': OUT})
    compare(expected, read(OUT / 'verified_provenance.json'))
    p = PAPER / 'audit/build_nondrift_evidence.py'
    spec = importlib.util.spec_from_file_location('archive_aggregation', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dest = OUT / 'reaggregation'
    (dest / 'generated').mkdir(parents=True, exist_ok=True)
    module.HERE, module.PAPER = dest, dest
    shutil.copyfile(PAPER / 'audit/onboarding_b2f.json', dest / 'onboarding_b2f.json')
    data = module.onboarding()
    compare(data, read(PAPER / 'audit/onboarding_metrics.json'))
    module.timing_and_memory_tables(data)
    compare(read(dest / 'matched_memory.json'), read(PAPER / 'audit/matched_memory.json'))
    table_match = {}
    for name, rhos in [('onboarding_main_rows.tex', [10]), ('onboarding_full_rows.tex', [1, 10, 100])]:
        text = module.table_rows(data, rhos)
        (dest / 'generated' / name).write_text(text)
        table_match[name] = text == (PAPER / 'generated' / name).read_text()
    assert all(table_match.values())
    return dict(source_hashes=len(hash_rows), cohorts_and_checkpoint_values_match=True,
                onboarding_cells=sum(len(rows) for d in data.values() for rows in d['by_rho'].values()),
                all_onboarding_values_match=True, generated_table_match=table_match,
                matched_memory_unavailable=sum(x['interpolated_delta_csr_pp'] is None for x in read(dest / 'matched_memory.json')))


def r22():
    d = read(RUNS / 'revision_r22_ewma_alpha_frontier.json')
    availability = []
    for cohort, cd in d['cohorts'].items():
        for rho, rd in cd['by_rho'].items():
            for alpha, a in rd['ewma_alpha'].items():
                cold = np.asarray(a['func_cold_all'])
                assert len(cold) == cd['n_functions']
                close(cold.sum(), a['cold_starts'])
                close(100*cold.sum()/cd['invocations'], a['csr_pct'])
                close(a['wm_total']*1000/cd['invocations'], a['wm_per_1k_inv'])
                close(a['wm_per_1k_inv'] + 150*float(rho)*a['csr_pct'], a['cost_per_1k_inv'])
                availability.append(dict(cohort=cohort, rho=rho, alpha=alpha, n_functions=len(cold),
                                         first_hour_length=len(a['func_cold60']), rolling_length=len(a['rolling_csr']),
                                         fast_no_rolling=a.get('fast_des_no_rolling', False)))
    for rho, rd in d['cohorts']['azure2019_validation']['by_rho'].items():
        cells = list(rd['ewma_alpha'].values())
        csr = min(cells, key=lambda x: (x['csr_pct'], x['cost_per_1k_inv'], x['alpha']))['alpha']
        cost = min(cells, key=lambda x: (x['cost_per_1k_inv'], x['csr_pct'], x['alpha']))['alpha']
        close(csr, d['selected_alpha']['by_csr'][rho])
        close(cost, d['selected_alpha']['by_cost'][rho])
    csv_write('r22_availability.csv', availability)
    return dict(cells=len(availability), function_records=sum(x['n_functions'] for x in availability),
                rolling_records=sum(x['rolling_length'] for x in availability),
                missing_validation_rolling_cells=sum(x['fast_no_rolling'] for x in availability),
                selected_alpha=d['selected_alpha'], all_arithmetic_matches=True)


def holdout48():
    d = read(RUNS / 'revision_r20_2019_holdout_fast_runs.json')
    groups = defaultdict(list)
    for r in d:
        close(sum(r['func_cold']), r['cold_starts'])
        close(sum(r['func_total']), r['total_invocations'])
        close(sum(r['func_wm']), r['wm_total_gb_s'])
        close(r['cold_starts']/r['total_invocations'], r['csr'])
        close(r['wm_total_gb_s']*1000/r['total_invocations'], r['wm_per_1k_inv'])
        groups[(r['method'], r['cost_ratio'])].append(r)
    rows = []
    for (arm, rho), rr in sorted(groups.items()):
        assert sorted(x['seed'] for x in rr) == [0, 1, 2]
        cold = np.mean([x['cold_starts'] for x in rr])
        inv = np.mean([x['total_invocations'] for x in rr])
        wm = np.mean([x['wm_total_gb_s'] for x in rr])
        rows.append(dict(arm=arm, rho=rho, invocations=float(inv), cold=float(cold),
                         csr_pct=100*cold/inv, wm_per1k=1000*wm/inv,
                         cost_per1k=1000*(wm+15*rho*cold)/inv))
    pairs = []
    a = np.mean([x['func_cold'] for x in groups['G_WE_A720', 10.0]], axis=0)
    for other in ('B4a_ewma', 'G_WE_Ainf'):
        b = np.mean([x['func_cold'] for x in groups[other, 10.0]], axis=0)
        delta = a-b
        rng = np.random.default_rng(260907)
        boot = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(10000)])
        pairs.append(dict(arm='G_WE_A720', comparator=other, mean=float(delta.mean()),
                          ci=np.quantile(boot, [.025, .975]).tolist(), bootstrap_seed=260907))
    base = RUNS / 'des_jobs_r20_2019_holdout/azure2019_holdout'
    z = np.load(base / 'shared.npz')
    assert z['counts'].shape == (1755, 2880)
    assert z['counts'].sum() == 40703980
    identities = []
    for rho in (1, 10, 100):
        gate = np.load(base / f'G_WE_A720__rho{rho}.npz')
        ewma = np.load(base / f'B4a_ewma__rho{rho}.npz')
        cont = np.load(base / f'G_WE_Ainf__rho{rho}.npz')
        for field in ('prewarm', 'keepalive'):
            assert np.array_equal(gate[field][:, :720], cont[field][:, :720])
            assert np.array_equal(gate[field][:, 720:], ewma[field][:, 720:])
        identities.append(dict(rho=rho, before720_equals_no_handoff=True, from720_equals_ewma=True))
    csv_write('48h_reaggregated.csv', rows)
    write('48h_paired_recheck.json', pairs)
    return dict(seed_records=len(d), cells=len(groups), functions=1755, invocations=40703980,
                action_identities=identities, arithmetic_matches=True, corrected_48h_policy_replays=0,
                note='Bootstrap is an audit recomputation at the stated seed; action identity is not outcome identity.')


def r21():
    rows = list(csv.DictReader((SF / 'results/tables/T_r21_reviewer_full_faithful_fullrho.csv').open()))
    source_names = sorted({p for r in rows for p in r['source_json'].split(';')})
    raw = [r for p in source_names for r in read(SF / p)]
    manifests = {}
    for name in ('revision_r21_azure2019_remaining_manifest.json', 'revision_r21_huawei_manifest.json',
                 'revision_r21_huawei_hsaturated_remaining_manifest.json'):
        manifests.update(read(RUNS / name)['splits'])
    ns = dict(np=np, KAPPA=15.0, BOOT_N=5000, BOOT_SEED=260715)
    functions(SF / 'scripts/revision_r21_cross_provider_faithful.py', {'summarize_cell', 'paired_boot_ci'}, ns)
    out = []
    for row in rows:
        selected = lambda arm: [r for r in raw if r['split'] == row['split'] and r['method'] == arm and float(r['cost_ratio']) == float(row['rho'])]
        rr, ref = selected(row['method']), selected('B4a_ewma')
        assert sorted(x['seed'] for x in rr) == [0, 1, 2]
        weights = np.asarray(manifests[row['split']]['weights'])
        a, b = ns['summarize_cell'](rr, weights), ns['summarize_cell'](ref, weights)
        for key in ('csr_pct', 'csr_seed_std_pct', 'wm_per_1k_inv', 'cost_per_1k_inv'):
            tol = 0.00000051 if 'pct' in key else 0.00051
            close(a[key], float(row[key]), atol=tol, rtol=0)
        for key, target in [('csr_pct', 'delta_csr_pp_vs_ewma'), ('wm_per_1k_inv', 'delta_wm_per_1k_vs_ewma'), ('cost_per_1k_inv', 'delta_cost_per_1k_vs_ewma')]:
            close(a[key]-b[key], float(row[target]), atol=0.00000051 if key=='csr_pct' else .00051, rtol=0)
        if row['method'] != 'B4a_ewma':
            lo, hi = ns['paired_boot_ci'](rr, ref, weights)
            assert lo == row['boot_ci95_lo_pp'] and hi == row['boot_ci95_hi_pp']
        out.append(dict(split=row['split'], method=row['method'], rho=row['rho'], **a))
    csv_write('r21_reaggregated.csv', out)
    return dict(cells=len(rows), seed_records=len(raw), source_files=source_names,
                weighted_estimates_match=True, bootstrap_ci_cells=96, all_bootstrap_endpoints_match=True)


def a1():
    rv = ROOT / 'anil-fresh-start/research-v2'
    ns = dict(np=np, math=math, bisect=bisect, TTL_GRID=(5.,15.,30.,60.,120.,300.,600.), TTL=(5.,15.,30.,60.,120.,300.,600.))
    functions(rv / 'unified_window_experiment.py', {'merge'}, ns)
    functions(rv / 'gamma_a1_cost_matched.py', {'replay'}, ns)
    functions(rv / 'final_age_judgment.py', {'run'}, ns)
    cases = []
    for mult in (1, 3):
        ev = [(0., 1., mult), (1000., 1., mult), (2000., 1., mult)]
        for name, result in [('gamma', ns['replay'](ev, 'fixed_ttl', 0, [1000.,1000.], ttl_fixed=60.)[0]),
                             ('final_age', ns['run'](ev, 'rate', [30.,30.])[0])]:
            assert result['cold'] == mult
            cases.append(dict(producer=name, multiplicity=mult, observed_cold=result['cold'],
                              expected_cold_no_prearrival_warming=3*mult))
    write('a1_scoring_diagnostics.json', cases)
    p = ROOT / 'results/research-v2-final-age-judgment-v1'
    scores, means = pd.read_csv(p / 'scores.csv'), pd.read_csv(p / 'means.csv')
    rec = scores.groupby(['dataset','model']).mean(numeric_only=True).reset_index()
    for col in means.columns:
        if col not in ('dataset','model'):
            close(rec[col], means[col])
    summary = []
    pairs = pd.read_csv(p / 'saved_harmed.csv')
    for ds, group in scores.groupby('dataset'):
        pivot = group.pivot(index='function', columns='model', values=['csr_pct','cost_gbs'])
        delta = pivot['cost_gbs']['age'] - pivot['cost_gbs']['rate']
        ps = pairs[(pairs.dataset == ds) & (pairs.comparison == 'age')]
        summary.append(dict(dataset=ds, n_functions=len(pivot),
                            csr_equal_per_function=bool((pivot['csr_pct']['age']==pivot['csr_pct']['rate']).all()),
                            age_minus_rate_cost_mean=float(delta.mean()),
                            cost_relative_change_pct=float(100*delta.sum()/pivot['cost_gbs']['rate'].sum()),
                            saved_event_rows=int(ps.saved.sum()), harmed_event_rows=int(ps.harmed.sum()),
                            note='saved/harmed count event rows, not multiplicity-weighted individual requests'))
    csv_write('a1_final_age_reaggregated.csv', summary)
    # Reproduce the archived function bootstrap, preserving original group/RNG order.
    d = pd.read_csv(ROOT / 'results/research-v2-independent-a1-confirmation-v1/scores.csv')
    d = d[d.fold == 'evaluation']
    rng, bs = np.random.default_rng(0), []
    for (ds,k), group in d.groupby(['dataset','K']):
        pvt = group.pivot(index='function',columns='model',values=['csr_pct','cost_gbs'])
        for metric in ('csr_pct','cost_gbs'):
            delta = (pvt[metric]['gamma_predictive']-pvt[metric]['ewma']).dropna().to_numpy()
            boots = [np.mean(rng.choice(delta,len(delta),replace=True)) for _ in range(2000)]
            bs.append(dict(dataset=ds,K=int(k),metric=metric,n_functions=len(delta),mean_delta=float(delta.mean()),
                           ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975))))
    csv_write('a1_bootstrap_recomputed.csv', bs)
    old = pd.read_csv(ROOT / 'results/research-v2-cluster-bootstrap-v1/bootstrap.csv')
    new = pd.DataFrame(bs)
    for c in ('mean_delta','ci_low','ci_high'):
        close(old[c],new[c],atol=1e-12,rtol=1e-12)
    campaign_means = []
    for means_path in sorted((ROOT / 'results').glob('*a1*/means.csv')):
        scores_path = means_path.with_name('scores.csv')
        frame, saved = pd.read_csv(scores_path), pd.read_csv(means_path)
        keys = [k for k in ('dataset','fold','K','model','ttl') if k in saved.columns]
        calculated = frame.groupby(keys, dropna=False).mean(numeric_only=True).reset_index()
        saved = saved.sort_values(keys).reset_index(drop=True)
        calculated = calculated.sort_values(keys).reset_index(drop=True)
        assert len(saved) == len(calculated)
        for col in saved.columns:
            if col in keys:
                assert saved[col].equals(calculated[col])
            else:
                close(saved[col].to_numpy(dtype=float), calculated[col].to_numpy(dtype=float))
        campaign_means.append(dict(campaign=means_path.parent.name,score_rows=len(frame),
                                   mean_rows=len(saved),function_column_present='function' in frame,
                                   all_stored_means_match=True))
    csv_write('a1_campaign_means_checked.csv', campaign_means)
    return dict(final_age_score_rows=len(scores), final_age_mean_rows=len(means),
                bootstrap_rows=len(bs), bootstrap_matches=True, diagnostics=cases,
                scientific_status='self-hit scoring invalidates deployment-performance interpretation',
                datasets=summary, other_campaign_means=campaign_means)


def main():
    OUT.mkdir(exist_ok=True)
    result = dict(scope='Archived hashes, raw-count arithmetic, schedule identity and deterministic scoring diagnosis; no full policy replays')
    for name, call in [('provenance_onboarding',provenance_and_onboarding),('r22',r22),('48h',holdout48),('r21',r21),('a1',a1)]:
        print('START', name, flush=True)
        result[name] = call()
        write('numeric_verification.json', result)
        print('PASS', name, flush=True)


if __name__ == '__main__':
    main()
