"""Verify complete standalone replay and independently reproduce selected rows."""
import hashlib
import numpy as np
from campaign import ROOT, OLD, GATE, OUT, CASE_NAMES
from common import read_json, write_json, digest
from cases import decisions_from_rates
from events import event_chunks
from simulator import advance, new_state
from run import windows_for


def main():
    plan = read_json(OUT / 'protocol.json')
    assert digest(ROOT / 'experiments/winter_drift/campaign.py') == plan['script_sha256']
    reports = []
    for name in CASE_NAMES:
        directory = OUT / 'cases' / name
        meta = read_json(directory / 'case.json')
        contract = read_json(directory / 'execution.json')
        contract_hash = digest(directory / 'execution.json')
        assert contract['case_sha256'] == digest(directory / 'case.json')
        for path, expected in plan['source_cases'][name].items():
            assert digest(ROOT / path) == expected, path
        assert digest(directory / 'data.npz') == meta['data_sha256']
        with np.load(directory / 'data.npz') as z:
            data = {k: z[k] for k in z.files}
        with np.load(GATE / 'cases' / name / 'onboarding_rates.npz') as z:
            rate = z['learned']
        actions = []
        for key in sorted(meta['actions']):
            record = meta['actions'][key]
            path = directory / record['file']
            assert digest(path) == record['sha256']
            q, ttl = decisions_from_rates(rate, record['rho'] / (1 + record['rho']))
            with np.load(path) as z:
                np.testing.assert_array_equal(np.clip(q, 0, 200).astype(np.int16), z['q'])
                np.testing.assert_array_equal(ttl.astype(np.float32), z['ttl'])
                actions.append((z['q'].copy(), z['ttl'].copy()))
        summary = read_json(directory / 'summary.json')
        assert digest(directory / 'aggregate.npz') == summary['aggregate_sha256']
        with np.load(directory / 'aggregate.npz') as z:
            per_function = z['per_function']
        chain = hashlib.sha256()
        for f in range(len(data['counts'])):
            saved = []
            for si in range(len(meta['seeds'])):
                path = directory / 'seed_functions' / f'{f:05d}_{si:02d}.npz'
                chain.update(digest(path).encode())
                with np.load(path) as z:
                    assert str(z['contract_sha256']) == contract_hash
                    assert int(z['function_index']) == f and int(z['seed']) == meta['seeds'][si]
                    saved.append(z['stats'][z['mapping']])
            np.testing.assert_allclose(np.mean(saved, axis=0), per_function[f], rtol=0, atol=0)
        assert not list((directory / 'partial_states').glob('*.npz'))
        # Compare the optimized production runner with the separately maintained
        # reference engine at both median and maximum workload volume.
        volumes = data['counts'].sum(1)
        sample = sorted({int(np.argsort(volumes, kind='stable')[len(volumes) // 2]), int(volumes.argmax())})
        max_error = 0.
        windows = windows_for(meta)
        for f in sample:
            with np.load(directory / 'functions' / f'{f:05d}.npz') as z:
                expected = z['stats'][:, z['mapping']]
            for si, seed in enumerate(meta['seeds']):
                states = [new_state(1680) for _ in actions]
                for lo, hi, tape in event_chunks(data['counts'][f], data['duration_mean'][f], data['duration_std'][f],
                                                seed, meta['event_key_prefix'] + str(data['keys'][f])):
                    for (qa, ta), state in zip(actions, states):
                        advance(*tape, qa[f, lo:hi], ta[f, lo:hi], *state, lo, hi == 1680)
                actual = np.array([[state[0][lo:hi].sum(0) for lo, hi in windows.values()] for state in states])
                np.testing.assert_array_equal(actual[..., [0, 1, 5, 6, 7]], expected[si, ..., [0, 1, 5, 6, 7]].transpose(1, 2, 0))
                np.testing.assert_allclose(actual, expected[si], rtol=5e-8, atol=1e-5)
                max_error = max(max_error, float(np.max(np.abs(actual - expected[si]))))
        reports.append(dict(case=name, status='passed', event_windows=len(data['counts']),
                            function_seed_jobs=len(data['counts']) * len(meta['seeds']),
                            seed_files_hash_chain=chain.hexdigest(), actions_match_cached_learned=True,
                            per_function_seed_reconstruction=True, independent_reference_rows=sample,
                            reference_max_absolute_ledger_error=max_error,
                            source_hashes_unchanged=True, aggregate_sha256=digest(directory / 'aggregate.npz')))
        print('verified', name, flush=True)
    write_json(OUT / 'verification.json', dict(status='passed', cases=reports,
                protocol_sha256=digest(OUT / 'protocol.json'), verifier_sha256=digest(__file__),
                event_windows=sum(x['event_windows'] for x in reports),
                function_seed_jobs=sum(x['function_seed_jobs'] for x in reports)))


if __name__ == '__main__':
    main()
