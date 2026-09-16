"""Check the fixed-alpha controller selection, every evaluated arm and reuse."""
import hashlib
import numpy as np
import controls
from controls import OUT,OLD,common,policies,prune_screen


def main():
    assert common.read_json(OUT/'protocol.json')['ewma_alphas']==[.3]
    configs=common.read_json(OUT/'candidate_configs.json')
    assert configs==policies.simple_configs() and len(configs)==497
    assert all(c.get('alpha',.3)==.3 for c in configs)
    selected=common.read_json(OUT/'selection.json')
    scores=np.load(OUT/'calibration_scores.npz')['metrics']
    for ri,rho in enumerate((1,10,100)):
        proof=common.read_json(OUT/f'screen_proofs_rho{rho}.json')
        pruned={p['config'] for p in proof['proofs']}
        assert len(pruned)+proof['fully_evaluated']==497
        best=np.array([np.inf if x is None else x for x in proof['best_cold']])
        for p in proof['proofs']:
            assert not prune_screen.can_improve(p['cold_lower_bound'],p['wm_lower_bound'],np.array(proof['budgets']),best)
        for i,c in enumerate(configs):assert np.isnan(scores[ri,i]).all()==(c['id'] in pruned)
        totals=scores[ri,:,:4].sum(1)
        for arm,row in selected[str(float(rho))].items():
            with np.load(OUT/'evaluation/azure_calibration'/f'native_rho{rho}'/(arm+'.npz')) as z:m=z['metrics']
            reference=float(m[:,:,:4,2].sum(2).mean(1).sum())
            assert row['wm_reference']==reference
            for choice in row['choices']:
                assert choice['wm_limit']==reference*choice['multiplier']
                eligible=[i for i,m in enumerate(totals) if m[2]<=choice['wm_limit']]
                i=min(eligible,key=lambda i:(round(5*totals[i,1]),totals[i,2],configs[i]['id'])) if eligible else None
                assert choice['config']==(configs[i] if i is not None else None)
                if i is not None:np.testing.assert_array_equal(choice['metrics'],totals[i])
    hashes={};checked=0;reused=0
    for cohort in ('azure_calibration','azure_primary','azure_evaluation','huawei'):
        data=common.load_cohort(cohort);counts=data['counts'];n=len(counts)
        expected_seeds=np.arange(5) if cohort=='azure_calibration' else np.arange(1000,1020)
        with np.load(OLD/'evaluation'/cohort/'native_rho10/component.npz' if cohort=='azure_calibration'
                     else OLD/'evaluation'/cohort/'main_rho10/component.npz') as z:
            execution=z['metrics'][:,:,:,4].sum(2)
        count_bins=np.stack([counts[:,lo:hi].sum(1) for lo,hi in ((0,1),(1,15),(15,60),(60,240))]+[np.zeros(n)],axis=-1)
        for path in sorted((OUT/'evaluation'/cohort).glob('*/*.npz')):
            with np.load(path) as z:
                m=z['metrics'];q=z['q'];t=z['ttl']
                assert m.shape==(n,len(expected_seeds),5,8)
                np.testing.assert_array_equal(z['seeds'],expected_seeds)
                assert np.isfinite(m).all() and (m>=-1e-6).all()
                assert (m[...,1]<=m[...,0]).all() and (m[...,7]<=m[...,1]).all()
                np.testing.assert_array_equal(m[...,1],m[...,6])
                np.testing.assert_array_equal(m[...,0],np.broadcast_to(count_bins[:,None,:],m[...,0].shape))
                np.testing.assert_allclose(m[:,:,:,4].sum(2),execution,rtol=3e-9,atol=1e-5)
                assert q.shape==t.shape==counts.shape
                assert (q>=0).all() and (q<=200).all() and np.isfinite(t).all() and (t>=0).all()
                if path.parent.name.startswith('strict'):
                    assert (q[:,0]==0).all() and (t[:,0]==10).all()
                if path.stem=='ewma':
                    rho=float(path.parent.name.split('rho')[-1])
                    eq,et=policies.decisions(policies.ewma(counts),rho)
                    if path.parent.name.startswith('strict'):eq[:,0]=0;et[:,0]=10
                    np.testing.assert_array_equal(eq,q);np.testing.assert_array_equal(et,t)
            hashes[str(path.relative_to(OUT))]=common.digest(path);checked+=1;reused+=int(path.is_symlink())
        print('verified controls',cohort,flush=True)
    # Reused representation choices refer to unchanged rate matrices and budgets.
    assert common.digest(OUT/'representation_selection.json')==common.digest(OLD/'representation_selection.json')
    for model in (OUT/'models').iterdir():
        for path in model.glob('*_rates.npz'):
            source=OLD/'models'/model.name/path.name
            with np.load(source) as before,np.load(path) as after:
                for key in before.files:
                    if key!='gate':np.testing.assert_array_equal(before[key],after[key])
            hashes[str(path.relative_to(OUT))]=common.digest(path)
    common.write_json(OUT/'artifact_hashes.json',hashes)
    common.write_json(OUT/'verification.json',dict(status='passed',passed=True,ewma_alpha=.3,
        evaluation_policy_files=checked,reused_identical_action_files=reused,
        grid_candidates=497,selection_sha256=common.digest(OUT/'selection.json'),
        protocol_sha256=common.digest(OUT/'protocol.json'),verifier_sha256=common.digest(__file__),
        artifact_hashes_sha256=common.digest(OUT/'artifact_hashes.json'),
        checks=['exact bounded-grid selection','new gate validation budgets','all requests and phase ledgers',
                'same function/seed streams','strict starts','native alpha reconstruction','representation invariance']))
    print('Verified',checked,'control policy files;',reused,'reused with identical actions',flush=True)


if __name__=='__main__':main()
