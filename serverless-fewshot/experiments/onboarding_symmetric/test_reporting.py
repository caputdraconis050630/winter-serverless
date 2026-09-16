"""Analytic fixtures for reporting units, windows, and paired cost deltas."""
import unittest
import numpy as np

from report import endpoint, cost_contrast
from analyze import PairedBootstrap, window_contrast, holm
from experiment import OUT, write_json


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.a = np.zeros((3, 2, 5, 8))
        self.a[:, :, :4, 0] = 10
        self.a[:, :, :4, 1] = 1
        self.a[:, :, :4, 2] = 20
        self.a[:, :, :4, 3] = 3
        self.a[:, :, :4, 4] = 5
        self.a[:, :, 4, 2] = 7

    def test_full_window_units(self):
        row = endpoint("toy", 10., "main", "a", self.a, [0, 1, 2, 3], "first240")
        self.assertEqual(row["invocations"], 120.)
        self.assertEqual(row["cold"], 12.)
        self.assertEqual(row["csr_pct"], 10.)
        self.assertEqual(row["wm_per1k"], 2000.)
        self.assertEqual(row["allocated_per1k"], 2800.)
        self.assertEqual(row["modeled_cost_per1k"], 17000.)
        self.assertEqual(row["drain_idle_per1k_4h"], 175.)
        self.assertEqual(row["cost_plus_drain_per1k"], 17175.)

    def test_partial_window_does_not_import_drain(self):
        row = endpoint("toy", 1., "main", "a", self.a, [0, 1], "first15")
        self.assertEqual(row["invocations"], 60.)
        self.assertEqual(row["modeled_cost_per1k"], 3500.)
        self.assertIsNone(row["cost_plus_drain_per1k"])

    def test_seed_average_is_not_a_sum(self):
        self.a[:, 1, :4, 1] = 3
        row = endpoint("toy", 10., "main", "a", self.a, [0, 1, 2, 3], "first240")
        self.assertEqual(row["cold"], 24.)
        self.assertEqual(row["csr_pct"], 20.)

    def test_paired_cost_and_cold_units(self):
        b = self.a.copy()
        b[:, :, :4, 1] = 0
        b[:, :, :4, 2] = 10
        b[:, :, 4, 2] = 0
        bs = PairedBootstrap(np.arange(3), np.full(3, 40))
        row = cost_contrast(bs, self.a, b, 10.)
        self.assertEqual(row["cost_delta_per1k"], 16000.)
        self.assertEqual(row["drain_cost_delta_per1k"], 16175.)
        self.assertEqual(row["cost_delta_per1k_ci_low"], 16000.)
        self.assertEqual(row["cost_delta_per1k_ci_high"], 16000.)
        cold = window_contrast(bs, self.a, b)
        self.assertEqual(cold["cold_per_fn_difference"], 4.)
        self.assertEqual(cold["csr_difference_pp"], 10.)
        self.assertEqual(cold["wm_ratio"], 2.)

    def test_holm_preserves_original_order(self):
        np.testing.assert_allclose(holm([.03, .001, .02]), [.04, .003, .04])


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReportingTests))
    write_json(OUT / "reporting_tests.json", dict(passed=result.wasSuccessful(), tests=result.testsRun))
    raise SystemExit(not result.wasSuccessful())
