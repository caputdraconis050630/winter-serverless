"""Add the existing age/count router and its no-handoff control to saved shifts.

No drift oracle, detector, reset, target tuning or replacement of old results.
The added learned branch uses the paper's all-past-row onboarding ridge loop;
provider-specific checkpoints match the retained drift/LSTM source split.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/lstm_comparison'))
sys.path.insert(0, str(ROOT))
from common import digest, freeze, read_json, save_npz, write_json
import cases
import run_seeds
from drift import causal_embeddings, make_segment_features, load_trainer
from scripts.eval_adapt_biased import build_prototypes

OLD = ROOT / 'results/lstm_comparison_v1'
OUT = ROOT / 'results/winter_g_drift_v1'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(1)
CASE_NAMES = [f'drift_{p}_{s}' for p in ('azure2019','azure2021','huawei') for s in ('natural','synthetic')]


def routing(counts, provider):
    history = np.zeros_like(counts, dtype=np.int64)
    history[:,1:] = np.cumsum(counts[:,:-1], axis=1, dtype=np.int64)
    first = (counts > 0).argmax(1)
    if provider == 'huawei':
        age = np.broadcast_to(np.arange(counts.shape[1]), counts.shape)
    else:
        age = np.maximum(np.arange(counts.shape[1])[None,:] - first[:,None], 0)
    # First-arrival time affects no output before its count has been observed.
    return np.where(history == 0, 0, np.where(history < 100, 1, np.where(age < 720, 2, 3))).astype(np.int8)


@torch.no_grad()
def fit_rates(phi, counts, lam, centroids, prior, device=DEVICE):
    """Scalar-equivalent, double precision incremental sufficient statistics."""
    n, horizon, dim = phi.shape
    prior = np.asarray(prior)
    if prior.ndim == 3:
        np.testing.assert_allclose(prior, np.broadcast_to(prior[:,:,:1], prior.shape), atol=0, rtol=0)
        prior = prior[:,:,0]
    p = torch.as_tensor(phi, device=device, dtype=torch.float64)
    y = torch.as_tensor(np.log1p(counts.astype(np.float64)), device=device)
    cen = torch.as_tensor(centroids, device=device, dtype=torch.float64)
    wp = torch.as_tensor(prior, device=device, dtype=torch.float64)
    eye = torch.eye(dim, device=device, dtype=torch.float64)
    a = lam * eye.expand(n, dim, dim).clone()
    b = torch.zeros((n,dim), device=device, dtype=torch.float64)
    learned = np.empty((n,horizon), np.float32)
    proto = np.empty_like(learned)
    direct_errors = []
    for t in range(horizon):
        x = p[:,t]
        ids = torch.nn.functional.cosine_similarity(x[:,None,:],cen[None,:,:],dim=2).argmax(1)
        w0 = wp[ids]
        w = w0 if t == 0 else torch.linalg.solve(a, (b + lam*w0).unsqueeze(-1)).squeeze(-1)
        lograte = (x*w).sum(1)
        learned[:,t] = torch.expm1(lograte.clamp(0,20)).cpu().numpy()
        proto[:,t] = torch.expm1((x*w0).sum(1).clamp(0,20)).cpu().numpy()
        if t in (1,15,60,240,719,1440,horizon-1):
            direct = torch.linalg.solve(p[0,:t].T@p[0,:t] + lam*eye,
                                        p[0,:t].T@y[0,:t] + lam*w0[0])
            err = float(abs(x[0]@direct - lograte[0]))
            direct_errors.append(err)
            if err > 1e-6:
                raise RuntimeError(f'incremental/direct log forecast disagreement: {err}')
        # Counts at t become support only after forecasting t.
        a += x[:,:,None] * x[:,None,:]
        b += x * y[:,t,None]
    return learned, proto, max(direct_errors, default=0.)


def source_model(split):
    directory = OUT / 'models' / split
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / f'results_azure2021/runs/best_anil_ridge_{split}_s0.pt'
    trainer = load_trainer(DEVICE, 11, checkpoint)
    path = directory / 'prototypes.npz'
    if not path.exists():
        if split == 's2':
            old = ROOT / 'results/onboarding_attribution_v1/models/trained0'
            assert read_json(old/'model.json')['checkpoint_sha256'] == digest(checkpoint)
            shutil.copyfile(old/'prototypes.npz', path)
        else:
            source = ROOT/'data/processed'
            features = np.load(source/'features.npy', mmap_mode='r')
            counts = np.load(source/'counts.npy', mmap_mode='r')
            ids = np.load(source/'splits.npz')[split+'_train']
            c,w = build_prototypes(trainer, None, features, counts, ids)
            save_npz(path, centroids=c.numpy(), prior=w.numpy())
    with np.load(path) as z:
        centroids, prior = z['centroids'], z['prior']
    lam = float(trainer.head.ridge_lambda.item())
    freeze(directory/'model.json', dict(split=split, checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint), prototypes_sha256=digest(path),
        prototype_builder_sha256=digest(ROOT/'scripts/eval_adapt_biased.py'),
        lambda_value=lam, prototype_count=len(centroids), device=DEVICE))
    return trainer, lam, centroids, prior


def prepare(name, models):
    old = OLD/'cases'/name
    original = read_json(old/'case.json')
    assert digest(old/'data.npz') == original['data_sha256']
    with np.load(old/'data.npz') as z:
        data = {k:z[k] for k in z.files}
    provider = original['provider']
    split = 's1' if provider == 'azure2021' else 's2'
    if split not in models:
        models[split] = source_model(split)
    trainer, lam, centroids, prior = models[split]
    cases.OUT = OUT
    directory, meta = cases.begin(name, data, dict(
        surface='drift',provider=provider,drift_source=original['drift_source'],
        seeds=original['seeds'],rhos=original['rhos'],onset_minute=1440,
        event_key_prefix=original['event_key_prefix'],events=original['events'],
        source_case=str(old),source_case_sha256=digest(old/'case.json'),
        source_aggregate_sha256=digest(old/'aggregate.npz'),
        model_source=original['model_source'],
        learned_loop='causal next-minute all-past-row prior-biased ridge, each tick; meta-learned lambda; C=16',
        support_start='replay observation minute 0; preceding observed zero-count rows included',
        age_clock='Huawei: recorded presence at replay start; Azure: first observed in-window arrival',
        gate='N=0 prototype; 0<N<100 EWMA0.1; N>=100 age<720 learned; otherwise EWMA0.1',
        no_handoff='same policy with no age boundary; zero-count and below-count routing unchanged',
        drift_reset=False, drift_onset_available_to_policy=False,
        comparison='Added router experiment; scheduled WINTER retained separately with its original update loop'))
    rates_path = directory/'onboarding_rates.npz'
    if not rates_path.exists():
        pnames=dict(azure2019='processed_2019',azure2021='processed',huawei='processed_huawei')
        features=np.load(ROOT/'data'/pnames[provider]/'features.npy',mmap_mode='r')
        learned=np.empty(data['counts'].shape,np.float32);proto=np.empty_like(learned)
        errors=[];started=time.monotonic()
        cache=directory/'forecast_chunks';cache.mkdir(exist_ok=True)
        for start in range(0,len(learned),64):
            stop=min(start+64,len(learned));path=cache/f'{start:05d}.npz'
            if path.exists():
                with np.load(path) as z:
                    learned[start:stop]=z['learned'];proto[start:stop]=z['proto'];errors.append(float(z['direct_error']))
                continue
            phi=[]
            for i in range(start,stop):
                event=original['events'][i];offset=event['onset']-1440;fi=int(data['ids'][i])
                feat=np.array(features[fi,offset:offset+1680],dtype=np.float32,copy=True)
                if original['drift_source']=='synthetic':
                    feat=make_segment_features(data['counts'][i],feat[:,10],offset)
                phi.append(causal_embeddings(trainer,feat))
            a,b,err=fit_rates(np.stack(phi),data['counts'][start:stop],lam,centroids,prior)
            save_npz(path,learned=a,proto=b,direct_error=np.array(err))
            learned[start:stop]=a;proto[start:stop]=b;errors.append(err)
            print(name,'forecasts',stop,'/',len(learned),'elapsed',round(time.monotonic()-started,1),flush=True)
        save_npz(rates_path,learned=learned,proto=proto)
        write_json(directory/'forecast_checks.json',dict(max_direct_log_forecast_error=max(errors),device=DEVICE))
    with np.load(rates_path) as z:learned,proto=z['learned'],z['proto']
    states=routing(data['counts'],provider)
    ewma=cases.rates_ewma(data['counts'],.1)
    gate=np.where(states==0,proto,np.where(states==2,learned,ewma))
    no_handoff=np.where(states==0,proto,np.where(states>=2,learned,ewma))
    np.testing.assert_array_equal(gate[states!=3],no_handoff[states!=3])
    np.testing.assert_array_equal(gate[(states==1)|(states==3)],ewma[(states==1)|(states==3)])
    provenance=dict(file=str(rates_path),sha256=digest(rates_path),model=read_json(OUT/'models'/split/'model.json'),
                    adapter='onboarding procedure; separate from retained scheduled shift-response adapter')
    cases.rate_actions(directory,meta,'WINTER_G',gate,meta['rhos'],provenance)
    cases.rate_actions(directory,meta,'No_handoff',no_handoff,meta['rhos'],provenance)
    save_npz(directory/'routing.npz',states=states)
    meta['routing_sha256']=digest(directory/'routing.npz')
    meta['source_comparators']={k:v for k,v in original['actions'].items()
                              if v['method'] in ('LSTM_Fifer','LSTM_shared','EWMA_0.1','WINTER','WINTER_frozen')}
    cases.finish(directory,meta)
    write_json(directory/'routing_summary.json',dict(
        post_function_minutes={str(s):int((states[:,1440:]==s).sum()) for s in range(4)},
        onset_functions={str(s):int((states[:,1440]==s).sum()) for s in range(4)},
        post_invocations_by_route={str(s):int(data['counts'][:,1440:][states[:,1440:]==s].sum()) for s in range(4)}))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=CASE_NAMES)
    ap.add_argument('--workers',type=int,default=5);ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(version=1,purpose='Close the direct WINTER-G post-drift evidence gap',
        cases=CASE_NAMES,seeds=[0,1,2],rhos=[1.,10.],horizon_minutes=1680,onset_minute=1440,
        windows_minutes=[1,15,30,60,240],primary_window=240,primary_rho=10.,
        new_arms=['WINTER_G','No_handoff'],
        retained_comparators=['LSTM_Fifer','LSTM_shared','EWMA_0.1','WINTER','WINTER_frozen'],
        source_splits={'azure2021':'s1','azure2019':'s2','huawei':'s2'},
        rules='fixed age720/count100; no detector or reset; no target selection; reuse verified common-event baselines',
        uncertainty='2000 paired cluster resamples, Azure app / Huawei function, DES seeds averaged; exploratory marginal 95% intervals',
        created_by=str(Path(__file__).relative_to(ROOT)),script_sha256=digest(__file__),
        source_cases={n:dict(case_sha256=digest(OLD/'cases'/n/'case.json'),
                            aggregate_sha256=digest(OLD/'cases'/n/'aggregate.npz')) for n in CASE_NAMES})
    freeze(OUT/'protocol.json',plan)
    models={};run_seeds.OUT=OUT
    for name in ([args.case] if args.case else CASE_NAMES):
        if (OUT/'cases'/name/'summary.json').exists():
            print('already complete',name,flush=True);continue
        prepare(name,models)
        if not args.prepare_only:run_seeds.run(name,args.workers)
    print('campaign stage complete',datetime.now(timezone.utc).isoformat(),flush=True)


if __name__=='__main__':main()
