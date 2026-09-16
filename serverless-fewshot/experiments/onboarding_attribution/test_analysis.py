"""Statistical identity and exact-pruning checks, with no workload tuning."""
import unittest
import numpy as np

from analyze import PairedBootstrap,holm
from prune_screen import can_improve
from common import OUT,write_json


class AnalysisTests(unittest.TestCase):
    def test_holm(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

    def test_identity(self):
        a=np.zeros((8,20,8)); a[:,:,0]=10; a[:,:,1]=1; a[:,:,2:5]=2
        bs=PairedBootstrap(np.array(["a","a","b","c","d","e","f","g"]),np.full(8,10))
        r=bs.contrast(a,a)
        self.assertEqual(r["cold_ci_low"],0.)
        self.assertEqual(r["cold_ci_high"],0.)
        self.assertEqual(r["wm_ratio"],1.)
        self.assertEqual(r["bootstrap_t_p"],1.)

    def test_paired_direction(self):
        a=np.zeros((8,20,8)); a[:,:,0]=10; a[:,:,1]=1; a[:,:,2:5]=2
        b=a.copy(); b[:,:,1]=3; b[:,:,2]=4
        bs=PairedBootstrap(np.array(list("abcdefgh")),np.full(8,10))
        r=bs.contrast(a,b)
        self.assertEqual(r["cold_per_fn_difference"],-2.)
        self.assertEqual(r["wm_ratio"],.5)
        self.assertAlmostEqual(r["csr_difference_pp"],-20.)

    def test_exact_pruning_preserves_minima(self):
        rng=np.random.default_rng(42)
        cells=rng.uniform(0,20,(7,100,2))
        budgets=np.array([40.,60.,80.,100.])
        best=np.full(4,np.inf)
        complete={}
        pruned=[]
        for c in range(100):
            partial=np.zeros(2)
            for f in range(7):
                if not can_improve(partial[0],partial[1],budgets,best):
                    pruned.append((c,partial.copy()))
                    break
                partial+=cells[f,c]
            else:
                complete[c]=partial
                best[partial[1]<=budgets]=np.minimum(best[partial[1]<=budgets],partial[0])
        total=cells.sum(0)
        expected=np.array([min([r[0] for r in total if r[1]<=b],default=np.inf) for b in budgets])
        np.testing.assert_array_equal(best,expected)
        for _,bound in pruned:
            self.assertFalse(can_improve(bound[0],bound[1],budgets,best))

    def test_pruning_retains_float_rounding_ties(self):
        self.assertTrue(can_improve(100.+1e-9,100.+1e-9,np.array([100.]),np.array([100.])))


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalysisTests))
    write_json(OUT/"analysis_tests.json",{"passed":result.wasSuccessful(),"tests":result.testsRun})
    raise SystemExit(0 if result.wasSuccessful() else 1)
