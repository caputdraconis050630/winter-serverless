"""Independently verify live EWMA schedules and unchanged comparison schedules."""
import numpy as np
from campaign import OUT,OLD,ROOT
from common import read_json,write_json,digest
from cases import rates_ewma,decisions_from_rates,newsvendor_quantile


def schedule(counts,q,ttl):
    result=np.zeros_like(q,dtype=np.int64)
    for f,row in enumerate(counts):
        last=-100000
        for t in range(len(row)):
            if t and row[t-1]>0:last=t-1
            result[f,t]=int(q[f,t]>0 or t-last<ttl[f,t])
    return result


def main():
    current=OUT/'strict_round/live';old=OLD/'strict_round/live'
    protocol=read_json(current/'protocol.json');original=read_json(old/'protocol.json')
    assert protocol['ewma_alpha']==.3 and protocol['ewma_weight']=='new observation'
    assert protocol['checkpoint_sha256']==original['checkpoint_sha256']
    lstm=OLD/'models/s1_seed0/model.pt'
    assert digest(lstm)==original['lstm_checkpoint_sha256']
    # The archived live procedure uses rho=10, unchanged in this revision.
    from scripts import testbed_cohort
    assert testbed_cohort.RHO==10 and testbed_cohort.GATE_THRESHOLD==100
    with np.load(current/'schedules.npz') as z:a={k:z[k] for k in z.files}
    with np.load(old/'schedules.npz') as z:b={k:z[k] for k in z.files}
    counts=a['counts'];np.testing.assert_array_equal(counts,b['counts'])
    for arm in ('reactive','keepalive10','lstm_fifer'):
        np.testing.assert_array_equal(a[arm],b[arm])
    q,t=decisions_from_rates(rates_ewma(counts,.3),newsvendor_quantile(10))
    expected=schedule(counts,q,t)
    np.testing.assert_array_equal(a['ewma'],expected)
    h=np.zeros_like(counts,dtype=np.int64);h[:,1:]=np.cumsum(counts[:,:-1],axis=1)
    mask=(h>0)&(h<100)
    np.testing.assert_array_equal(a['protowarm'],np.where(mask,expected,b['protowarm']))
    report=dict(status='passed',ewma_alpha=.3,
        source_sha256={str(p):digest(p) for p in (current/'protocol.json',current/'schedules.npz',
            old/'protocol.json',old/'schedules.npz',lstm)},
        verifier_sha256=digest(__file__),
        checks=['same request cohort','same LSTM source weights','three invariant live schedules',
                'independent .3 EWMA schedule','gate output substitution on exact low-count minutes'])
    write_json(current/'schedule_verification.json',report)
    print('Verified all fixed-alpha live schedules',flush=True)
    return report


if __name__=='__main__':main()
