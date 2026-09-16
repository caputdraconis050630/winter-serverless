"""Check exact reuse against a fresh replay on changing capacity and TTL."""
from pathlib import Path
import tempfile
import sys
import numpy as np
import campaign
import runner
from common import save_npz,write_json,digest
import run_seeds


def fixture(root,counts,actions):
    p=root/'cases/test';p.mkdir(parents=True)
    save_npz(p/'data.npz',counts=counts,duration_mean=np.array([.8,2.]),
             duration_std=np.array([.3,.5]),keys=np.array(['a','b']),
             apps=np.array(['x','y']),weights=np.ones(2))
    meta=dict(name='test',surface='initial',provider='test',horizon_minutes=counts.shape[1],
        n_functions=2,raw_invocations=int(counts.sum()),seeds=[7,8],rhos=[1.],
        event_key_prefix='alpha-reuse-test:',data_sha256=digest(p/'data.npz'),actions={})
    for method,(q,t) in actions.items():
        key=method+'__rho1';save_npz(p/(key+'.npz'),q=q.astype(np.int16),ttl=t.astype(np.float32))
        meta['actions'][key]=dict(method=method,rho=1.,file=key+'.npz',sha256=digest(p/(key+'.npz')))
    write_json(p/'case.json',meta)
    return p


def main():
    rng=np.random.default_rng(44);counts=rng.poisson(.8,(2,80))
    counts[:,20:30]+=7
    q=(counts>1).astype(np.int16);t=np.full(counts.shape,3.,np.float32)
    changed=q.copy();changed[:,-1]=7  # A prefix match must not permit reuse.
    with tempfile.TemporaryDirectory(prefix='winter-alpha-replay-') as tmp:
        root=Path(tmp);old=root/'old';new=root/'new';fresh=root/'fresh'
        fixture(old,counts,{'WINTER':(q,t),'EWMA_0.3':(q*2,t/2)})
        run_seeds.OUT=old;run_seeds.run('test',2)
        for out in (new,fresh):fixture(out,counts,{'WINTER':(q,t),'WINTER_G':(changed,t)})
        runner.OUT=new;runner.SOURCE=old;runner.run('test',2)
        run_seeds.OUT=fresh;run_seeds.run('test',2)
        with np.load(new/'cases/test/aggregate.npz') as a,np.load(fresh/'cases/test/aggregate.npz') as b:
            for k in a.files:np.testing.assert_array_equal(a[k],b[k])
        for path in (new/'cases/test/seed_functions').glob('*.npz'):
            with np.load(path) as z:assert len(z['reused_actions'])==1
    x=np.array([[1,0,3,0,1]*30]);a=campaign.startup_rates(x,.3)
    x[:,90:]=999;b=campaign.startup_rates(x,.3)
    np.testing.assert_array_equal(a[:,:91],b[:,:91])
    print('PASS: exact full-schedule reuse equals fresh replay; suffix changes reject reuse; startup is causal.')


if __name__=='__main__':main()
