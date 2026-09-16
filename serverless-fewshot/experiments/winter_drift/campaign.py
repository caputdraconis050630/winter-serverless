"""Replay standalone WINTER with the exact learned branch retained for WINTER-G.

This additive campaign leaves all earlier results intact. It reuses the frozen
onboarding forecasts, cohorts, controller and common-request streams. Only the
output selection differs from the previously measured WINTER-G policy.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/lstm_comparison'))
from common import digest, freeze, read_json
import cases
import run_seeds

OLD = ROOT / 'results/lstm_comparison_v1'
GATE = ROOT / 'results/winter_g_drift_v1'
OUT = ROOT / 'results/winter_drift_v1'
CASE_NAMES = [f'drift_{p}_{s}' for p in ('azure2019', 'azure2021', 'huawei')
              for s in ('natural', 'synthetic')]


def source_hashes(name):
    result = {}
    for campaign, files in ((OLD, ('case.json', 'aggregate.npz', 'execution.json')),
                            (GATE, ('case.json', 'aggregate.npz', 'execution.json',
                                    'onboarding_rates.npz', 'routing.npz'))):
        for filename in files:
            path = campaign / 'cases' / name / filename
            result[str(path.relative_to(ROOT))] = digest(path)
    return result


def prepare(name, expected):
    for path, hash_value in expected.items():
        assert digest(ROOT / path) == hash_value, path
    base_dir = OLD / 'cases' / name
    gate_dir = GATE / 'cases' / name
    base = read_json(base_dir / 'case.json')
    gate = read_json(gate_dir / 'case.json')
    assert digest(base_dir / 'data.npz') == base['data_sha256']
    with np.load(base_dir / 'data.npz') as z:
        data = {k: z[k] for k in z.files}
    with np.load(gate_dir / 'data.npz') as z:
        for k, value in data.items():
            np.testing.assert_array_equal(value, z[k])
    cases.OUT = OUT
    directory, meta = cases.begin(name, data, dict(
        surface='drift', provider=base['provider'], drift_source=base['drift_source'],
        seeds=base['seeds'], rhos=[1., 10.], onset_minute=1440,
        event_key_prefix=base['event_key_prefix'], events=base['events'],
        model_source=base['model_source'], source_hashes=expected,
        learned_loop=gate['learned_loop'], support_start=gate['support_start'],
        output_selection='WINTER: learned forecast at every minute; prototype initialization at t=0; no count/age gate',
        drift_reset=False, drift_onset_available_to_policy=False,
        comparison='WINTER and retained WINTER-G share the exact learned branch; only output selection differs'))
    rate_path = gate_dir / 'onboarding_rates.npz'
    with np.load(rate_path) as z:
        learned = z['learned']
        np.testing.assert_array_equal(learned[:, 0], z['proto'][:, 0])
    assert learned.shape == data['counts'].shape and np.isfinite(learned).all()
    provenance = dict(file=str(rate_path), sha256=digest(rate_path), key='learned',
                      model=gate['actions']['WINTER_G__rho10']['source']['model'])
    cases.rate_actions(directory, meta, 'WINTER', learned, meta['rhos'], provenance)
    with np.load(gate_dir / 'routing.npz') as z:
        states = z['states']
    for rho in meta['rhos']:
        with np.load(directory / f'WINTER__rho{rho:g}.npz') as w, \
             np.load(gate_dir / f'WINTER_G__rho{rho:g}.npz') as g:
            for key in ('q', 'ttl'):
                np.testing.assert_array_equal(w[key][states == 2], g[key][states == 2])
                np.testing.assert_array_equal(w[key][:, 0], g[key][:, 0])
    cases.finish(directory, meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASE_NAMES)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    plan = dict(version=1, purpose='Compare WINTER and WINTER-G under one onboarding update procedure',
                cases=CASE_NAMES, seeds=[0, 1, 2], rhos=[1., 10.],
                horizon_minutes=1680, onset_minute=1440,
                windows_minutes=[1, 15, 30, 60, 240], primary_window=240, primary_rho=10.,
                new_arms=['WINTER'], retained_comparators=['WINTER_G', 'LSTM_Fifer', 'LSTM_shared', 'EWMA_0.1', 'No_handoff'],
                output='Retained onboarding learned array at every minute, including below count 100',
                uncertainty='2000 paired cluster bootstrap replicates, seed 260915; marginal exploratory 95% intervals',
                no_target_tuning=True, no_drift_signal=True, no_state_reset=True,
                source_protocol_sha256=digest(GATE / 'protocol.json'),
                source_cases={name: source_hashes(name) for name in CASE_NAMES},
                script_sha256=digest(__file__))
    freeze(OUT / 'protocol.json', plan)
    run_seeds.OUT = OUT
    for name in ([args.case] if args.case else CASE_NAMES):
        prepare(name, plan['source_cases'][name])
        if not args.prepare_only:
            run_seeds.run(name, args.workers)


if __name__ == '__main__':
    main()
