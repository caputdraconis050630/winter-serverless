"""Check the compact bundle's published representation and selection evidence."""
from pathlib import Path
import csv
import json
import os
import sys
import tempfile

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'serverless-fewshot'
OUT = EXP / 'results/fixed_ewma_v1/controls'
sys.path.insert(0, str(EXP / 'experiments/onboarding_attribution'))
import analyze
import prune_screen


def read(path):
    return json.loads(path.read_text())


def metrics(path):
    with np.load(path) as z:
        return z['metrics']


def main():
    selected = read(OUT / 'selection.json')
    configs = read(OUT / 'candidate_configs.json')
    with np.load(OUT / 'calibration_scores.npz') as z:
        scores = z['metrics']
    assert len(configs) == 497 and all(c.get('alpha', .3) == .3 for c in configs)
    controller_choices = 0
    for ri, rho in enumerate([1, 10, 100]):
        proof = read(OUT / f'screen_proofs_rho{rho}.json')
        pruned = {p['config'] for p in proof['proofs']}
        assert len(pruned) + proof['fully_evaluated'] == 497
        best = np.array([np.inf if x is None else x for x in proof['best_cold']])
        for p in proof['proofs']:
            assert not prune_screen.can_improve(p['cold_lower_bound'], p['wm_lower_bound'], np.array(proof['budgets']), best)
        for i, c in enumerate(configs):
            assert np.isnan(scores[ri, i]).all() == (c['id'] in pruned)
        totals = scores[ri, :, :4].sum(1)
        for arm, row in selected[str(float(rho))].items():
            raw = metrics(OUT / f'evaluation/azure_calibration/native_rho{rho}/{arm}.npz')
            reference = float(raw[:, :, :4, 2].sum(2).mean(1).sum())
            assert row['wm_reference'] == reference
            for choice in row['choices']:
                assert choice['wm_limit'] == reference * choice['multiplier']
                eligible = [i for i, m in enumerate(totals) if m[2] <= choice['wm_limit']]
                i = min(eligible, key=lambda i: (round(5 * totals[i, 1]), totals[i, 2], configs[i]['id'])) if eligible else None
                assert choice['config'] == (configs[i] if i is not None else None)
                if i is not None:
                    np.testing.assert_array_equal(choice['metrics'], totals[i])
                controller_choices += 1

    # Representation choices were frozen before the fixed-alpha control update;
    # use their original source budget, not the revised native gate budget.
    reference = metrics(EXP / 'results/onboarding_attribution_v1/evaluation/azure_calibration/native_rho10/component.npz')
    reference_wm = float(reference[:, :, :4, 2].sum(2).mean(1).sum())
    reps = read(OUT / 'representation_selection.json')['10.0']
    representation_choices = 0
    for model, choices in reps.items():
        totals = {i: metrics(OUT / f'evaluation/azure_calibration/repr_{model}_rho10/lambda{i}.npz')[:, :, :4].sum(2).mean(1).sum(0) for i in range(9)}
        for choice in choices:
            eligible = [i for i, m in totals.items() if m[2] <= reference_wm * choice['multiplier']]
            best = min(eligible, key=lambda i: (round(5 * totals[i][1]), totals[i][2], f'lambda{i}')) if eligible else None
            assert choice['lambda_index'] == best
            if best is not None:
                np.testing.assert_array_equal(choice['metrics'], totals[best])
            representation_choices += 1

    rows = list(csv.DictReader((OUT / 'comparisons.csv').open()))
    contrasts = []
    for cohort in ['azure_primary', 'azure_evaluation', 'huawei']:
        with np.load(OUT / f'cohorts/{cohort}.npz', allow_pickle=True) as z:
            bootstrap = analyze.PairedBootstrap(z['apps'], z['counts'].sum(1))
        arrays = {}
        for model in ['trained0'] + [f'random{i}' for i in range(5)]:
            choice = next(c for c in reps[model] if c['multiplier'] == 1.)
            arrays[model] = metrics(OUT / f'evaluation/{cohort}/repr_{model}_rho10/lambda{choice["lambda_index"]}.npz')
        result = analyze.window_contrast(bootstrap, arrays['trained0'], np.mean([arrays[f'random{i}'] for i in range(5)], axis=0))
        saved = next(r for r in rows if r['cohort'] == cohort and r['condition'] == 'representation' and float(r['rho']) == 10.)
        for key, value in result.items():
            if isinstance(value, bool):
                assert str(value) == saved[key], (cohort, key)
            else:
                np.testing.assert_allclose(value, float(saved[key]), rtol=2e-11, atol=1e-8, err_msg=f'{cohort}/{key}')
        contrasts.append(dict(cohort=cohort, numerical_fields=len(result), all_match=True))
    result = dict(status='passed', controller_candidates_per_rho=497, cost_ratios=[1,10,100],
                  controller_choices=controller_choices, representation_rho=10,
                  representation_choices=representation_choices, representation_contrasts=contrasts,
                  fresh_simulations=False)
    output = ROOT / '.repro-output'
    output.mkdir(exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix='core-evidence-', dir=output))
    (destination / 'controls-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Report:', destination / 'controls-validation.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
