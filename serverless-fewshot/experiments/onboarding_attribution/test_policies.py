"""Causality, archived-controller parity, and model-cache checks."""
import sys
import unittest

import numpy as np

from common import OUT,ROOT,read_json,write_json
from policies import actions,b1,decisions,ewma,simple_configs


class PolicyTests(unittest.TestCase):
    def test_future_counts_do_not_change_past(self):
        rng=np.random.default_rng(42)
        c=rng.poisson(2.,(3,240))
        changed=c.copy(); changed[:,60:]=10000
        for alpha in (.01,.1,.3,1.):
            np.testing.assert_array_equal(ewma(c,alpha)[:,:61],ewma(changed,alpha)[:,:61])
        np.testing.assert_array_equal(b1(c)[0][:,:61],b1(changed)[0][:,:61])
        sys.path.insert(0,str(ROOT))
        from src.data.features import features_from_counts
        np.testing.assert_array_equal(features_from_counts(c)[:,:60],features_from_counts(changed)[:,:60])

    def test_controller_matches_executed_quantities(self):
        sys.path.insert(0,str(ROOT))
        from scripts.phase6_des import decisions_from_rates
        rates=np.array([0.,.0009,.001,.01,.1,1.,2.,2.01,10.,100.,1000.],dtype=np.float32)[None,:]
        for rho in (1.,10.,100.):
            q,t=decisions(rates,rho)
            oldq,oldt=decisions_from_rates(rates,rho/(1+rho))
            np.testing.assert_array_equal(q,np.minimum(oldq,200))
            np.testing.assert_allclose(t,oldt,atol=2e-6,rtol=0)

    def test_b2f_does_not_use_future_activity(self):
        sys.path.insert(0,str(ROOT))
        from scripts.revision_v1_hybridfull import policy_b2_full
        counts=np.zeros((2,240),dtype=np.int64)
        counts[0,::7]=3
        counts[1,:60:3]=2
        changed=counts.copy(); changed[:,60:]=100
        original=policy_b2_full(counts,verbose_every=100000)
        future_changed=policy_b2_full(changed,verbose_every=100000)
        for a,b in zip(original,future_changed):
            np.testing.assert_array_equal(a[:,:61],b[:,:61])

    def test_configs_unique_and_native_anchor(self):
        configs=simple_configs()
        self.assertEqual(len(configs),len({c["id"] for c in configs}))
        self.assertTrue(any(c["id"]=="ageka60" for c in configs))

    def test_ridge_checks(self):
        files=list((OUT/"models").glob("*/*_ridge_check.json"))
        self.assertEqual(len(files),24)
        self.assertLess(max(read_json(p)["max_log_prediction_error"] for p in files),1e-4)

    def test_gate_causal_routing(self):
        for cohort in ("azure_primary","azure_calibration","azure_evaluation","huawei"):
            c=np.load(OUT/"cohorts"/(cohort+".npz"))["counts"]
            z=np.load(OUT/"models/trained0"/(cohort+"_rates.npz"))
            n=np.zeros_like(c); n[:,1:]=np.cumsum(c[:,:-1],axis=1)
            expected=np.where(n==0,z["prototype_only"],np.where(n<100,ewma(c),z["component"]))
            np.testing.assert_array_equal(z["gate"],expected)


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PolicyTests))
    write_json(OUT/"policy_tests.json",{"passed":result.wasSuccessful(),"tests":result.testsRun})
    raise SystemExit(0 if result.wasSuccessful() else 1)
