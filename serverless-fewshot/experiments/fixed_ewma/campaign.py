"""Additive alpha=0.3 replay; retain immutable source experiments and requests.

Only EWMA-selected actions change. Learned/prototype actions are copied exactly,
after verifying the old gate against the old EWMA on every replaced minute.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/lstm_comparison'))
import numpy as np
import cases
import run_seeds
from common import digest, freeze, read_json, save_npz, write_json

OUT = ROOT / 'results/fixed_ewma_v1'
OLD = ROOT / 'results/lstm_comparison_v1'
GATE = ROOT / 'results/winter_g_drift_v1'
WINTER = ROOT / 'results/winter_drift_v1'
ALPHA = .3
INITIAL = ['initial_azure_primary', 'initial_huawei', 'initial_azure_evaluation']
DRIFT = [f'drift_{p}_{s}' for p in ('azure2021','azure2019','huawei')
         for s in ('natural','synthetic')]
STEADY = [f'steady_{p}_{s}' for p in ('azure2021','azure2019','huawei')
          for s in (('h_mixed','h_sparse','h_saturated') if p=='huawei' else ('S1','S2','S3'))]
ALL_CASES = INITIAL + ['continuous_48h'] + DRIFT + STEADY


def ewma_masks(meta, data):
    counts = data['counts']
    history = np.zeros_like(counts, dtype=np.int64)
    history[:, 1:] = np.cumsum(counts[:, :-1], axis=1, dtype=np.int64)
    below = (history > 0) & (history < 100)
    ticks = np.arange(counts.shape[1])[None, :]
    if meta['surface'] in ('initial', 'continuous'):
        age = ticks
    elif meta['provider'] == 'huawei' and meta['surface'] == 'steady':
        first = np.load(ROOT / 'data/processed_huawei/first_present.npy')[data['ids']]
        age = ticks - first[:, None]
    elif meta['provider'] == 'huawei':
        age = ticks
    else:
        first = np.where(counts.any(axis=1), (counts > 0).argmax(axis=1), counts.shape[1])
        age = np.maximum(ticks-first[:, None], 0)
    gate = below | ((history >= 100) & (age >= 720))
    if meta['surface'] == 'drift':
        with np.load(GATE / 'cases' / meta['name'] / 'routing.npz') as z:
            states = z['states']
            np.testing.assert_array_equal(gate, (states == 1) | (states == 3))
            np.testing.assert_array_equal(below, states == 1)
    return {'WINTER_G': gate, 'No_handoff': below}


def link(source, dest):
    if dest.exists():
        assert digest(dest) == digest(source), dest
    else:
        dest.symlink_to(source.resolve())


def startup_rates(counts, alpha):
    """Retain the documented active-pool emission order; change only alpha."""
    out = np.empty((len(counts), min(120, counts.shape[1])), np.float32)
    state = np.zeros(len(counts), np.float64)
    for t in range(out.shape[1]):
        prev = counts[:, max(t-1, 0)].astype(np.float32).astype(np.float64)
        if t < 60:
            out[:,t] = np.expm1(np.maximum(state, 0))
            state = alpha*np.log1p(prev)+(1-alpha)*state
        else:
            state = alpha*np.log1p(prev)+(1-alpha)*state
            out[:,t] = np.expm1(np.maximum(state, 0))
    return out


def startup_actions(rates, rho):
    q,ttl=cases.decisions_from_rates(rates,cases.newsvendor_quantile(rho))
    return np.clip(q,0,200).astype(np.int16),np.asarray(ttl,np.float32)


def prepare(name):
    base = OLD / 'cases' / name
    original = read_json(base / 'case.json')
    gate_base = GATE / 'cases' / name if name in DRIFT else base
    original_gate = read_json(gate_base / 'case.json')
    directory = OUT / 'replay/cases' / name
    directory.mkdir(parents=True, exist_ok=True)
    link(base / 'data.npz', directory / 'data.npz')
    with np.load(base / 'data.npz') as z:
        data = {k:z[k] for k in z.files}
    assert digest(base / 'data.npz') == original['data_sha256']
    masks = ewma_masks(original, data)
    meta = deepcopy(original)
    meta.update(actions={}, ewma_alpha=ALPHA,
        revision='EWMA output substitution only; same learned/prototype actions, gate, counts and event keys',
        source_case_sha256=digest(base/'case.json'),
        source_gate_case_sha256=digest(gate_base/'case.json'),
        preparation_sha256=digest(__file__))
    checks = []
    startup = None
    if original['surface']=='steady':
        startup = (startup_rates(data['counts'], .15), startup_rates(data['counts'], ALPHA))
        history = np.zeros_like(data['counts'], dtype=np.int64)
        history[:,1:] = np.cumsum(data['counts'][:,:-1],axis=1,dtype=np.int64)
        learned_mask = (history>=100) & ~masks['WINTER_G']
        learned_mask[:,120:] = False
    for rho in original['rhos']:
        with np.load(base/f'EWMA_0.1__rho{rho:g}.npz') as z:
            before = (z['q'], z['ttl'])
        with np.load(base/f'EWMA_0.3__rho{rho:g}.npz') as z:
            after = (z['q'], z['ttl'])
        for method, mask in masks.items():
            key = f'{method}__rho{rho:g}'
            if key not in original_gate['actions']:
                continue
            source = gate_base / original_gate['actions'][key]['file']
            assert digest(source) == original_gate['actions'][key]['sha256']
            with np.load(source) as z:
                old = (z['q'], z['ttl'])
            for j in range(2):
                np.testing.assert_array_equal(old[j][mask], before[j][mask],
                    err_msg=f'{name} {method} rho={rho} old EWMA branch {j}')
            new = tuple(np.where(mask, after[j], old[j]) for j in range(2))
            if startup is not None and method=='WINTER_G':
                old_start = startup_actions(startup[0], rho)
                new_start = startup_actions(startup[1], rho)
                m = learned_mask[:,:120]
                for j in range(2):
                    np.testing.assert_array_equal(old[j][:,:120][m],old_start[j][m])
                    new[j][:,:120][m] = new_start[j][m]
            for j in range(2):
                kept = ~mask if startup is None else (~mask & ~learned_mask)
                np.testing.assert_array_equal(new[j][kept], old[j][kept])
                np.testing.assert_array_equal(new[j][mask], after[j][mask])
            cases.action(directory, meta, method, rho, *new,
                dict(file=str(source), sha256=digest(source), ewma_alpha=ALPHA,
                     replacement_file=str(base/f'EWMA_0.3__rho{rho:g}.npz'),
                     replacement_sha256=digest(base/f'EWMA_0.3__rho{rho:g}.npz')))
            changed = (new[0]!=old[0]) | (new[1]!=old[1])
            checks.append(dict(method=method,rho=rho,ewma_minutes=int(mask.sum()),
                changed_minutes=int(changed.sum()),changed_functions=int(changed.any(axis=1).sum()),
                unchanged_outside_ewma=True,old_branch_matches_ewma=True))
        if startup is not None:
            source=base/f'WINTER__rho{rho:g}.npz'
            with np.load(source) as z: old=(z['q'],z['ttl'])
            old_start=startup_actions(startup[0],rho)
            new_start=startup_actions(startup[1],rho)
            new=tuple(x.copy() for x in old)
            for j in range(2):
                np.testing.assert_array_equal(old[j][:,:120],old_start[j],
                    err_msg=f'{name} standalone WINTER startup {j}')
                new[j][:,:120]=new_start[j]
                np.testing.assert_array_equal(new[j][:,120:],old[j][:,120:])
            cases.action(directory,meta,'WINTER',rho,*new,dict(file=str(source),
                sha256=digest(source),ewma_alpha=ALPHA,old_startup_alpha=.15,
                changed_interval=[0,120],learned_forecasts_unchanged=True))
            changed=(new[0]!=old[0])|(new[1]!=old[1])
            checks.append(dict(method='WINTER',rho=rho,changed_minutes=int(changed.sum()),
                changed_functions=int(changed.any(axis=1).sum()),
                startup_alpha=ALPHA,unchanged_after_minute120=True))
    freeze(directory / 'case.json', meta)
    freeze(directory / 'action_checks.json', dict(ewma_alpha=ALPHA, checks=checks))
    print('prepared', name, len(meta['actions']), 'actions',
          'changed function counts', [c['changed_functions'] for c in checks], flush=True)


def freeze_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in ALL_CASES:
        for root in (OLD, GATE, WINTER) if name in DRIFT else (OLD,):
            for f in ('case.json','execution.json','aggregate.npz','summary.json'):
                path = root/'cases'/name/f
                sources[str(path.relative_to(ROOT))] = digest(path)
    freeze(OUT/'protocol.json', dict(version=1, ewma_alpha=ALPHA,
        selection='Existing source-validation minimum 4h CSR; fixed in this revision before new replay',
        non_blind='Earlier results inspected; no claim of original preregistration or independent 48h selection',
        cases=ALL_CASES, changed_methods=['WINTER_G','No_handoff','WINTER active-pool startup'],
        baseline='Reuse exact current EWMA_0.3 actions and common-request outcomes',
        controller='unchanged; same rho grid and prospective retention',
        initialization='EWMA log state zero; prototype branch retained; active-pool startup alpha .15 to .3 with emission order retained',
        histories='full original horizon including 24h drift burn-in; no reset',
        source_sha256=sources, preparation_sha256=digest(__file__),
        live='Repeat five-arm one-hour common-node actuation with alpha=.3',
        controls='Freeze alpha=.3 and reselect remaining conservative-controller parameters on source validation'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run'])
    ap.add_argument('--case',choices=ALL_CASES);ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args()
    if args.mode=='prepare':
        freeze_protocol()
        for name in ([args.case] if args.case else ALL_CASES):prepare(name)
    else:
        assert read_json(OUT/'protocol.json')['ewma_alpha']==ALPHA
        run_seeds.OUT=OUT/'replay'
        for name in ([args.case] if args.case else ALL_CASES):run_seeds.run(name,args.workers)


if __name__=='__main__':
    main()
