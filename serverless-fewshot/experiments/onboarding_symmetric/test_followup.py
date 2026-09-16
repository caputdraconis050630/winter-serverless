"""Frozen-grid, causal transformation, statistics and pruning checks."""
import unittest
import numpy as np

from experiment import (GRID, OUT, OLD, RHOS, budgets, can_improve, digest, frozen_json,
                        transform, write_json)
from test_study import SimulatorTests
from test_policies import PolicyTests
from test_analysis import AnalysisTests


class FollowupTests(unittest.TestCase):
    def test_ordered_parallel_seeds_match_reference(self):
        from experiment import values_for
        from simulator import event_tape, simulate
        tapes = [event_tape(np.ones(240, dtype=np.int64), .3, .1, s, "parallel-test") for s in range(5)]
        q = np.full(240, 2, dtype=np.int64)
        ttl = np.full(240, 10.)
        expected = np.mean([simulate(*t, q, ttl) for t in tapes], axis=0)
        actual = values_for(tapes, q, ttl, {}, {}, True)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_grid(self):
        self.assertEqual(len(GRID), 180)
        self.assertEqual(len({x["id"] for x in GRID}), 180)

    def test_identity(self):
        q = np.arange(240, dtype=np.int64)[None].clip(0, 200)
        ttl = np.full(q.shape, 10., dtype=np.float32)
        a, b = transform(q, ttl, dict(scale=1., floor=0, ttl_factor=1., horizon=240))
        np.testing.assert_array_equal(a, q)
        np.testing.assert_array_equal(b, ttl)

    def test_horizon_and_input_unchanged(self):
        q = np.zeros((3, 240), dtype=np.int64)
        ttl = np.full(q.shape, 20., dtype=np.float32)
        a, b = transform(q, ttl, dict(scale=2., floor=2, ttl_factor=4., horizon=15))
        self.assertTrue((a[:, :15] == 2).all())
        self.assertTrue((a[:, 15:] == 0).all())
        self.assertTrue((b[:, :15] == 80).all())
        self.assertTrue((q == 0).all())
        self.assertTrue((ttl == 20).all())

    def test_caps(self):
        a, b = transform(np.full(240, 199), np.full(240, 200.),
                         dict(scale=4., floor=8, ttl_factor=4., horizon=240))
        self.assertTrue((a == 200).all())
        self.assertTrue((b == 240).all())

    def test_common_absolute_budgets(self):
        from experiment import read_json
        for rho in RHOS:
            original = read_json(OLD / "selection.json")[str(rho)]["component"]["choices"]
            np.testing.assert_array_equal(budgets(rho), [x["wm_limit"] for x in original])

    def test_exact_bound_ties_retained(self):
        self.assertTrue(can_improve(1.+1e-9, 2.+1e-9, np.array([2.]), np.array([1.])))
        self.assertFalse(can_improve(2., 3., np.array([2.]), np.array([1.])))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    write_json(OUT / "tests.json", dict(passed=result.wasSuccessful(), tests=result.testsRun))
    raise SystemExit(0 if result.wasSuccessful() else 1)
