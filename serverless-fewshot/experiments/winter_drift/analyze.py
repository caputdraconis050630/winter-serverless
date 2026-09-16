"""Matched WINTER/WINTER-G outcomes and paired uncertainty on retained shifts."""
import csv
import numpy as np
from campaign import ROOT, OLD, GATE, OUT, CASE_NAMES
from common import read_json, write_json, digest, rolling_csr, adaptation_lag

METHODS = ('WINTER', 'WINTER_G', 'LSTM_Fifer', 'LSTM_shared', 'EWMA_0.1', 'No_handoff')


def values(t, rho):
    inv, cold, idle = np.moveaxis(t, -1, 0)
    def div(x):
        return np.divide(x, inv, out=np.full_like(inv, np.nan, dtype=float), where=inv > 0)
    return np.stack((100 * div(cold), 1000 * div(idle), 1000 * div(idle + 15 * rho * cold)), axis=-1)


def interval(x):
    finite = np.asarray(x)[np.isfinite(x)]
    return np.quantile(finite, [.025, .975]).tolist() if len(finite) else None


def analyze():
    plan = read_json(OUT / 'protocol.json')
    records, points, checks, curves = [], [], [], []
    old_g = read_json(GATE / 'analysis.json')
    old_lookup = {(r['case'], r['rho'], r['window'], r['comparator']): r for r in old_g['comparisons']}
    for name in CASE_NAMES:
        for path, expected in plan['source_cases'][name].items():
            assert digest(ROOT / path) == expected, path
        current = OUT / 'cases' / name
        meta = read_json(current / 'case.json')
        arrays, timelines, actions = [], [], []
        for base, wanted in ((OUT, ('WINTER',)), (GATE, ('WINTER_G', 'No_handoff')),
                             (OLD, ('LSTM_Fifer', 'LSTM_shared', 'EWMA_0.1'))):
            directory = base / 'cases' / name
            summary = read_json(directory / 'summary.json')
            assert digest(directory / 'aggregate.npz') == summary['aggregate_sha256']
            with np.load(directory / 'aggregate.npz') as z:
                names = z['action_names'].tolist()
                selected = [i for i, action in enumerate(names) if action.rsplit('__', 1)[0] in wanted]
                arrays.append(z['per_function'][:, selected, :, :3])
                timelines.append(z['timeline'][selected])
                actions.extend(names[i] for i in selected)
                np.testing.assert_allclose(np.einsum('f,fawm->awm', z['weights'], z['per_function']),
                                           z['seed_totals'].mean(0), rtol=2e-12, atol=1e-6)
                if base == OUT:
                    wins = z['window_names'].tolist()
                    shared = {k: z[k].copy() for k in ('weights', 'apps', 'keys', 'total_curve')}
                else:
                    assert z['window_names'].tolist() == wins
                    for k, value in shared.items():
                        np.testing.assert_array_equal(value, z[k])
        arrays = np.concatenate(arrays, axis=1)
        timeline = np.concatenate(timelines)
        windows = [f'post_{t}' for t in (1, 15, 30, 60, 240)]
        parts = [arrays[:, :, wins.index(w)] for w in windows]
        for a, b in ((1, 15), (15, 30), (30, 60), (60, 240)):
            windows.append(f'interval_{a}_{b}')
            parts.append(arrays[:, :, wins.index(f'post_{b}')] - arrays[:, :, wins.index(f'post_{a}')])
        cube = np.stack(parts, axis=2) * shared['weights'][:, None, None, None]
        _, codes = np.unique(shared['apps'], return_inverse=True)
        ng = int(codes.max()) + 1
        cluster = np.zeros((ng, *cube.shape[1:]))
        np.add.at(cluster, codes, cube)
        rng = np.random.default_rng(260915)
        draws = np.array([np.bincount(rng.integers(ng, size=ng), minlength=ng) for _ in range(2000)], float)
        boot = (draws @ cluster.reshape(ng, -1)).reshape(2000, *cube.shape[1:])
        totals = cube.sum(0)
        for rho in plan['rhos']:
            for method in METHODS:
                ai = actions.index(f'{method}__rho{rho:g}')
                cold = timeline[ai, 1440:1680, 0]
                inv = shared['total_curve'][1440:1680]
                tail = rolling_csr(cold, inv)[180:]
                ref = float(100 * np.nanmean(tail))
                for wi, window in enumerate(windows):
                    csr, wm, cost = values(totals[ai, wi], rho)
                    points.append(dict(case=name, method=method, rho=rho, window=window,
                                       csr_pct=float(csr), wm_per_1k=float(wm), cost_per_1k=float(cost),
                                       adaptation_lag=adaptation_lag(cold, inv), reference_csr_pct=ref))
                for arm in ('WINTER', 'WINTER_G'):
                    if method == arm:
                        continue
                    bi = actions.index(f'{arm}__rho{rho:g}')
                    np.testing.assert_array_equal(totals[ai, :, 0], totals[bi, :, 0])
                    for wi, window in enumerate(windows):
                        pt = values(totals[[bi, ai], wi], rho)
                        bs = values(boot[:, [bi, ai], wi], rho)
                        record = dict(case=name, provider=meta['provider'], source=meta['drift_source'],
                                      rho=rho, window=window, arm=arm, comparator=method,
                                      n_event_windows=len(codes), n_clusters=ng, invocations=float(totals[bi, wi, 0]))
                        for mi, metric in enumerate(('csr_pct', 'wm_per_1k', 'cost_per_1k')):
                            a, b = pt[:, mi]
                            sa, sb = bs[:, :, mi].T
                            ratio = np.divide(sa, sb, out=np.full_like(sa, np.nan), where=sb > 0)
                            record[metric] = dict(arm=float(a), comparator=float(b), difference=float(a - b),
                                                  difference_ci95=interval(sa - sb), ratio=float(a / b) if b > 0 else None,
                                                  ratio_ci95=interval(ratio))
                        if arm == 'WINTER_G' and method != 'WINTER':
                            prior = old_lookup[name, rho, window, method]
                            for metric in ('csr_pct', 'wm_per_1k', 'cost_per_1k'):
                                for key, value in record[metric].items():
                                    expected = prior[metric]['g' if key == 'arm' else key]
                                    if value is None or expected is None:
                                        assert value is expected
                                    else:
                                        np.testing.assert_allclose(value, expected, rtol=2e-10, atol=1e-8)
                        records.append(record)
                if rho == 10:
                    cumulative = np.cumsum(timeline[ai, 1440:1680], axis=0)
                    den = np.cumsum(inv)
                    for minute in range(240):
                        curves.append(dict(case=name, method=method, rho=rho, minute=minute + 1,
                                           invocations=float(den[minute]), cold=float(cumulative[minute, 0]),
                                           idle=float(cumulative[minute, 1])))
        checks.append(dict(case=name, aggregate_sha256=digest(current / 'aggregate.npz'),
                           source_hashes_unchanged=True, weighted_totals_verified=True,
                           shared_requests_verified=True, previous_g_estimates_and_cis_preserved=True))
    assert len(records) == 1080 and len(points) == 648 and len(curves) == 8640
    report = dict(status='complete', protocol_sha256=digest(OUT / 'protocol.json'), analyzer_sha256=digest(__file__),
                  cases=checks, comparisons=records, operating_points=points, uncertainty=plan['uncertainty'])
    write_json(OUT / 'analysis.json', report)
    flattened = []
    for record in records:
        row = {k: v for k, v in record.items() if not isinstance(v, dict)}
        for metric in ('csr_pct', 'wm_per_1k', 'cost_per_1k'):
            for key, value in record[metric].items():
                if key.endswith('ci95'):
                    row[metric + '_' + key + '_low'], row[metric + '_' + key + '_high'] = value if value is not None else (None, None)
                else:
                    row[metric + '_' + key] = value
        flattened.append(row)
    for filename, rows in (('comparisons.csv', flattened), ('operating_points.csv', points), ('figure_curves.csv', curves)):
        with (OUT / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    print('Complete:', len(checks), 'cases,', len(records), 'paired contrasts', flush=True)
    for row in records:
        if row['rho'] == 10 and row['window'] == 'post_240' and row['comparator'] in ('WINTER', 'LSTM_Fifer'):
            print(row['case'], row['arm'], '/', row['comparator'], 'CSR', row['csr_pct'], 'COST', row['cost_per_1k'], flush=True)


if __name__ == '__main__':
    analyze()
