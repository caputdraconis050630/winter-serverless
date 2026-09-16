"""Reproduce retained controls on a deterministic median-volume row per case."""
from pathlib import Path
import sys
import numpy as np
from campaign import ROOT, OLD, OUT, CASE_NAMES
from common import read_json, write_json, digest
from events import event_chunks
from simulator_events import advance, new_state, execution_curve
from run import windows_for


def main():
    records=[]
    for name in CASE_NAMES:
        directory=OLD/'cases'/name;meta=read_json(directory/'case.json')
        with np.load(directory/'data.npz') as z:data={k:z[k] for k in z.files}
        f=int(np.argsort(data['counts'].sum(1),kind='stable')[len(data['counts'])//2])
        names=sorted(meta['actions']);selected=[m+'__rho10' for m in ('LSTM_Fifer','LSTM_shared','EWMA_0.1')]
        with np.load(directory/'functions'/f'{f:05d}.npz') as z:
            expected=z['stats'][:,z['mapping'][[names.index(m) for m in selected]]]
        windows=windows_for(meta);horizon=meta['horizon_minutes'];actions=[]
        for m in selected:
            path=directory/meta['actions'][m]['file'];assert digest(path)==meta['actions'][m]['sha256']
            with np.load(path) as z:actions.append((z['q'][f],z['ttl'][f]))
        max_error=0.
        for si,seed in enumerate(meta['seeds']):
            states=[new_state(horizon) for _ in actions]
            for lo,hi,tape in event_chunks(data['counts'][f],data['duration_mean'][f],data['duration_std'][f],seed,
                                           meta['event_key_prefix']+str(data['keys'][f])):
                baseline=execution_curve(tape[1],tape[2],horizon);completions=tape[1]+tape[2]
                order=np.argsort(completions,kind='stable')
                for (q,ttl),state in zip(actions,states):
                    advance(*tape,q[lo:hi],ttl[lo:hi],*state,lo,hi==horizon,.25,200,baseline,order,completions)
            actual=np.stack([[st[0][a:b].sum(0) for a,b in windows.values()] for st in states])
            np.testing.assert_allclose(actual,expected[si],rtol=2e-9,atol=1e-5)
            np.testing.assert_array_equal(actual[:,:,[0,1,5,6,7]],expected[si][:,:,[0,1,5,6,7]])
            max_error=max(max_error,float(abs(actual-expected[si]).max()))
        records.append(dict(case=name,row=f,row_selection='median request volume, stable tie order; no outcome selection',
            seeds=meta['seeds'],policies=selected,all_windows_match=True,max_absolute_ledger_error=max_error))
        print('verified retained controls',name,'row',f,flush=True)
    write_json(OUT/'baseline_replay_checks.json',dict(status='passed',checks=records,script_sha256=digest(__file__)))

if __name__=='__main__':main()
