"""Generate compact final-revision tables from saved, verified experiments.

No fitting, simulation, selection, or new statistical testing is performed.
The two source campaigns retain their own bootstrap protocols.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTR = ROOT.parent / 'serverless-fewshot/results/fixed_ewma_v1/controls'
OUT = ROOT / 'audit/final_revision_2026-09-16'
LABELS = {'azure_primary': 'Azure, 100', 'azure_evaluation': 'Azure, 970',
          'huawei': 'Huawei, 76'}


def read_csv(name):
    with (ATTR / name).open() as file:
        return list(csv.DictReader(file))


def only(rows):
    assert len(rows) == 1, len(rows)
    return rows[0]


def endpoint(cohort, arm):
    return only([r for r in read_csv('endpoints.csv') if r['cohort'] == cohort
                 and r['tag'] == 'main_rho10' and r['interval'] == 'first240'
                 and r['arm'] == arm])


def conservative(cohort, arm='component'):
    selections = json.loads((ATTR / 'selection.json').read_text())
    choice = only([c for c in selections['10.0'][arm]['choices'] if c['multiplier'] == 1.])
    assert choice['config'] is not None
    return endpoint(cohort, choice['config']['id'])


def generate():
    files = {}

    def table(name, rows):
        path = ROOT / 'generated' / (name + '.tex')
        path.write_text('\n'.join(' & '.join(r) + r' \\' for r in rows) + '\n')
        files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    comparisons = read_csv('comparisons.csv')
    main, controls, representations = [], [], []
    for cohort, label in LABELS.items():
        w, e = endpoint(cohort, 'component'), conservative(cohort)
        main.append([label, f"{float(w['csr_pct']):.3f}", f"{float(e['csr_pct']):.3f}",
                     f"{float(w['wm_per_1k']):,.0f}", f"{float(e['wm_per_1k']):,.0f}"])
        for arm in ('component', 'gate'):
            r = only([r for r in comparisons if r['cohort'] == cohort
                      and r['condition'] == 'main' and float(r['rho']) == 10
                      and r['arm'] == arm and float(r['budget_multiplier']) == 1])
            controls.append([label, 'WINTER' if arm == 'component' else 'WINTER-G',
                             f"{float(r['csr_difference_pp']):+.4f}",
                             f"[{float(r['csr_ci_low_pp']):+.4f}, {float(r['csr_ci_high_pp']):+.4f}]",
                             f"{float(r['wm_ratio']):.3f}",
                             f"[{float(r['wm_ratio_ci_low']):.3f}, {float(r['wm_ratio_ci_high']):.3f}]"])
        r = only([r for r in comparisons if r['cohort'] == cohort
                  and r['condition'] == 'representation' and float(r['rho']) == 10])
        representations.append([label, f"{float(r['csr_difference_pp']):+.4f}",
                                f"[{float(r['csr_ci_low_pp']):+.4f}, {float(r['csr_ci_high_pp']):+.4f}]",
                                f"{float(r['wm_ratio']):.3f}"])
    table('final_strong_controls_rows', main)
    table('final_control_intervals_rows', controls)
    table('final_representation_rows', representations)

    evidence_path = ROOT / 'audit/reviewer_evidence_2026_09_16.json'
    evidence = json.loads(evidence_path.read_text())
    timing, handoff = [], []
    for cohort, label in LABELS.items():
        d = evidence['initial_timing']['initial_' + cohort]
        timing.append([label] + [f"{d[a][t]:.3f}" for t in ('post_1', 'post_15', 'full')
                                for a in ('WINTER', 'LSTM_Fifer')])
    for r in evidence['comparisons']:
        c, wm = r['csr_pct'], r['wm_ratio']
        cells = [f"{c['difference']:+.4f}",
                 f"[{c['difference_ci95'][0]:+.4f}, {c['difference_ci95'][1]:+.4f}]",
                 f'{wm:.3f}',
                 f"[{r['wm_ratio_ci95'][0]:.3f}, {r['wm_ratio_ci95'][1]:.3f}]"]
        if r['case'] == 'continuous_48h':
            label = {'No_handoff':'No age hand-off', 'LSTM_shared':'LSTM--shared',
                     'LSTM_Fifer':'LSTM--Fifer', 'EWMA_0.3':'EWMA'}[r['comparator']]
            handoff.append([label] + cells)
    table('final_timing_rows', timing)
    table('final_handoff_intervals_rows', handoff)
    source_paths = [ATTR / x for x in ('endpoints.csv', 'comparisons.csv', 'selection.json',
                    'representation_selection.json', 'protocol.json', 'verification.json')]
    source_paths += [evidence_path, Path(__file__)]
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (OUT / 'evidence_manifest.json').write_text(json.dumps({
        'sources': sources, 'outputs': files,
        'protocols': {'controls':'10,000 paired cluster bootstrap replicates, saved campaign',
                      'handoff':'2,000 exploratory marginal paired cluster bootstrap replicates'},
        'attribution_point_agreement': evidence['attribution_point_agreement']}, indent=2) + '\n')
    print('Generated', len(files), 'tables from saved evidence.')


if __name__ == '__main__':
    generate()
