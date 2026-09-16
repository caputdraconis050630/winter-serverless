"""Read-only analysis of saved policy outcomes for the FGCS reviewer assessment.

No simulation, policy selection, model fitting, or manuscript modification.
Intervals below are exploratory, marginal paired cluster bootstrap intervals,
conditional on the saved source models and cohorts. They are not new primary
tests or multiplicity-adjusted findings.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
CASES = WORKSPACE / 'serverless-fewshot/results/fixed_ewma_v1/cases'
ATTR = WORKSPACE / 'serverless-fewshot/results/fixed_ewma_v1/controls'
OUT = Path(__file__).with_suffix('.json')


def compare(case, arm, comparator, window='full', rho=10):
    path = CASES / case / 'aggregate.npz'
    with np.load(path) as z:
        names, windows = z['action_names'].tolist(), z['window_names'].tolist()
        ids = [names.index(x + '__rho' + str(rho)) for x in (arm, comparator)]
        wi = windows.index(window)
        pf = z['per_function'][:, ids, wi, :3].copy()
        pf *= z['weights'][:, None, None]
        _, labels = np.unique(z['apps'], return_inverse=True)
        drain = z['per_function'][:, ids, windows.index('drain'), 2]
        drain = (drain * z['weights'][:, None]).sum(axis=0)
    groups = int(labels.max()) + 1
    totals = pf.sum(axis=0)
    np.testing.assert_allclose(pf[:, 0, 0], pf[:, 1, 0])
    cluster = np.zeros((groups, 2, 3))
    np.add.at(cluster, labels, pf)
    rng = np.random.default_rng(260915)
    draws = np.array([np.bincount(rng.integers(groups, size=groups), minlength=groups)
                      for _ in range(2000)])
    boot = (draws @ cluster.reshape(groups, 6)).reshape(2000, 2, 3)

    def metrics(t):
        inv, cold, idle = t[..., 0], t[..., 1], t[..., 2]
        return {'csr_pct': 100 * cold / inv,
                'wm_per_1k': 1000 * idle / inv,
                'cost_per_1k': 1000 * (idle + 15 * rho * cold) / inv}

    point, b = metrics(totals), metrics(boot)
    row = {'case': case, 'arm': arm, 'comparator': comparator, 'window': window,
           'rho': rho, 'n_functions_or_windows': len(pf), 'n_clusters': groups,
           'aggregate_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for k, values in point.items():
        row[k] = {'arm': float(values[0]), 'comparator': float(values[1]),
                  'difference': float(values[0] - values[1]),
                  'difference_ci95': np.quantile(b[k][:, 0] - b[k][:, 1], [.025, .975]).tolist()}
    ratio = point['wm_per_1k'][0] / point['wm_per_1k'][1]
    row['wm_ratio'] = float(ratio)
    row['wm_ratio_ci95'] = np.quantile(b['wm_per_1k'][:, 0] / b['wm_per_1k'][:, 1], [.025, .975]).tolist()
    row['fewer_cold_per_1000'] = float(-10 * row['csr_pct']['difference'])
    row['drain_idle_per_1k'] = (1000 * drain / totals[:, 0]).tolist()
    row['drain_inclusive_wm_ratio'] = float((totals[0, 2] + drain[0]) / (totals[1, 2] + drain[1]))
    return row


report = {'description': __doc__, 'bootstrap_replicates': 2000, 'seed': 260915,
          'comparisons': [], 'attribution_point_agreement': [], 'initial_timing': {}}
for c in ('initial_azure_primary', 'initial_azure_evaluation', 'initial_huawei'):
    report['comparisons'].append(compare(c, 'WINTER', 'LSTM_Fifer'))
    s = json.loads((CASES / c / 'summary.json').read_text())
    report['initial_timing'][c] = {
        arm: {win: s['by_action'][arm + '__rho10']['windows'][win]['csr_pct']
              for win in ('post_1', 'post_15', 'post_30', 'post_60', 'full')}
        for arm in ('WINTER', 'LSTM_Fifer')}
for c in ('No_handoff', 'LSTM_shared', 'LSTM_Fifer', 'EWMA_0.3'):
    report['comparisons'].append(compare('continuous_48h', 'WINTER_G', c))
for provider in ('azure2019', 'azure2021', 'huawei'):
    for kind in ('natural', 'synthetic'):
        case = 'drift_' + provider + '_' + kind
        for c in ('LSTM_Fifer',):
            report['comparisons'].append(compare(case, 'WINTER', c, 'post_240'))

with (ATTR / 'endpoints.csv').open() as f:
    endpoints = list(csv.DictReader(f))
for cohort in ('azure_primary', 'azure_evaluation', 'huawei'):
    s = json.loads((CASES / ('initial_' + cohort) / 'summary.json').read_text())
    for arm, name in [('component', 'WINTER'), ('gate', 'WINTER_G')]:
        rows = [r for r in endpoints if r['cohort'] == cohort and r['tag'] == 'main_rho10'
                and r['arm'] == arm and r['interval'] == 'first240']
        if len(rows) != 1:
            raise ValueError((cohort, arm, len(rows), sorted(set(r['interval'] for r in endpoints))))
        old = rows[0]
        new = s['by_action'][name + '__rho10']['windows']['full']
        delta = {k: float(old[a]) - new[b] for k, a, b in
                 [('csr', 'csr_pct', 'csr_pct'), ('wm', 'wm_per_1k', 'idle_per_1k')]}
        report['attribution_point_agreement'].append({'cohort': cohort, 'arm': arm, 'differences': delta})
        np.testing.assert_allclose([float(old['csr_pct']), float(old['wm_per_1k'])],
                                   [new['csr_pct'], new['idle_per_1k']], rtol=1e-8, atol=1e-7)

OUT.write_text(json.dumps(report, indent=2) + '\n')
for r in report['comparisons']:
    print(r['case'], r['arm'], '-', r['comparator'], 'CSR pp',
          round(r['csr_pct']['difference'], 6),
          [round(x, 6) for x in r['csr_pct']['difference_ci95']],
          'WM ratio', round(r['wm_ratio'], 4),
          [round(x, 4) for x in r['wm_ratio_ci95']],
          'Cost ratio', round(r['cost_per_1k']['arm']/r['cost_per_1k']['comparator'], 4),
          'drain WM ratio', round(r['drain_inclusive_wm_ratio'], 4))
print('Attribution point agreement:', report['attribution_point_agreement'])
print(OUT)
