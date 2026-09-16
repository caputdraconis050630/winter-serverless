"""Run independent-oracle and boundary tests before accepting any results."""
import unittest

import numpy as np

from common import HERE, OUT, digest, write_json
from simulator import certificate,event_tape,reference,reuse_certificate,simulate
from heap_simulator import simulate_fast


class SimulatorTests(unittest.TestCase):
    def compare(self, counts, q, ttl, seed=0, cap=200, mean=1., std=.5):
        tape = event_tape(np.asarray(counts), mean, std, seed, "test")
        actual = simulate(*tape, np.asarray(q, dtype=np.int64), np.asarray(ttl, dtype=float), cap=cap)
        fast = simulate_fast(*tape,np.asarray(q,dtype=np.int64),np.asarray(ttl,dtype=float),cap=cap)
        np.testing.assert_allclose(fast,actual,atol=1e-8,rtol=1e-12)
        expected, ledger = reference(tape, q, ttl, cap=cap)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-12)
        self.assertEqual(int(actual[:, 0].sum()), sum(counts))
        self.assertTrue((actual >= 0).all())
        self.assertAlmostEqual(actual[:, 2:5].sum(), sum(hi-lo for _,lo,hi,_ in ledger)*.25, places=7)
        return actual

    def test_identity_and_tape(self):
        counts = np.array([0,5,1,0,20,1])
        a = event_tape(counts, 1., .5, 3, "a")
        b = event_tape(counts, 1., .5, 3, "a")
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x,y)
        q,ttl = np.ones(6,dtype=np.int64), np.full(6,10.)
        np.testing.assert_array_equal(simulate(*a,q,ttl),simulate(*b,q,ttl))
        self.compare(counts,q,ttl)

    def test_ttl_step(self):
        counts=np.zeros(4,dtype=np.int64)
        tape=list(event_tape(counts,1.,.5,1,"ttl",cold_sigma=0.))
        q=np.array([1,0,0,0]); ttl=np.array([60.,60.,1.,1.])
        r=simulate(*tape,q,ttl,memory=1.)
        self.assertAlmostEqual(r[:,2].sum(),119.)
        self.assertAlmostEqual(simulate(*tape,q,ttl,memory=1.,legacy_ttl=True)[:,2].sum(),60.)
        self.compare(counts,q,ttl)

    def test_no_resurrection(self):
        self.compare([0,0,1,0],[1,0,0,0],[.1,60,60,60])

    def test_busy_and_overflow(self):
        self.compare([20,0,5,0],[0,0,1,0],[1,30,1,240],cap=2,mean=150,std=20)

    def test_zero_ttl_finishes_busy_work(self):
        result=self.compare([10,0,4,0],[0,0,0,0],[0,0,0,0],cap=2,mean=150,std=0.)
        self.assertEqual(result[:,2].sum(),0.)
        self.assertGreater(result[:,4].sum(),0.)

    def test_no_requests(self):
        self.compare([0]*240,[0]*240,[10]*240)
        self.compare([0]*240,[1]*240,[10]*240)

    def test_horizon_and_drain(self):
        c=np.zeros(240,dtype=int); c[-1]=20
        r=self.compare(c,np.zeros(240,dtype=int),np.full(240,240.),cap=3,mean=100.)
        self.assertGreater(r[4,2:5].sum(),0.)
        self.assertEqual(r[4,:2].sum(),0.)

    def test_randomized_oracle(self):
        rng=np.random.default_rng(260907)
        for s in range(80):
            n=int(rng.integers(4,30))
            self.compare(rng.integers(0,20,n),rng.integers(0,9,n),
                         rng.choice([0.,.1,1.,5.,10.,30.,60.],n),seed=s,cap=8,
                         mean=float(rng.choice([.1,5.,100.])))

    def test_exact_reuse_certificate(self):
        counts=np.full(240,300,dtype=int)
        tape=event_tape(counts,10.,2.,0,"certificate")
        cert=certificate(tape,2,cap=8)
        q=np.full(240,3,dtype=np.int64); q[0]=2
        ttl=np.full(240,30.)
        reused=reuse_certificate(cert,q,ttl)
        self.assertIsNotNone(reused)
        np.testing.assert_allclose(reused,simulate(*tape,q,ttl,cap=8),atol=1e-7,rtol=1e-12)
        q[1]=9
        self.assertIsNone(reuse_certificate(cert,q,ttl))
        q[1]=3; ttl[1]=1
        self.assertIsNone(reuse_certificate(cert,q,ttl))


if __name__ == "__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SimulatorTests))
    write_json(OUT/"tests.json",{"passed":result.wasSuccessful(),"tests":result.testsRun,
                                "heap_simulator_sha256":digest(HERE/"heap_simulator.py"),
                                "simulator_sha256":digest(HERE/"simulator.py")})
    raise SystemExit(0 if result.wasSuccessful() else 1)
