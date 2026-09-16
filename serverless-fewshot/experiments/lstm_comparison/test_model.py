"""Forecast causality, source split and controller invariants."""
import unittest

import numpy as np
import torch

from model import PeakLSTM, native_actions, predict, samples


class CausalityTest(unittest.TestCase):
    def test_future_counts_cannot_change_previous_forecasts(self):
        torch.manual_seed(7)
        model = PeakLSTM().eval()
        rng = np.random.default_rng(5)
        counts = rng.poisson(2., (3, 80)).astype(float)
        before = predict(model, counts)
        counts[:, 40:] = 100000
        after = predict(model, counts)
        np.testing.assert_array_equal(before[:, :41], after[:, :41])

    def test_training_targets_stay_inside_training_period(self):
        counts = np.ones((4, 200))
        counts[2:] = 100000
        counts[:2, 120:] = 100000
        x, y = samples(counts, np.array([0, 1]), 20, 120, 5000,
                       np.random.default_rng(0))
        np.testing.assert_allclose(x, np.log(2))
        np.testing.assert_allclose(y, np.log(2))

    def test_native_policy_uses_peak_deficit_and_ten_minute_timeout(self):
        q, ttl = native_actions(np.array([[0., .01, 1., 1.001, 10000.]]))
        np.testing.assert_array_equal(q, [[0, 1, 1, 2, 200]])
        np.testing.assert_array_equal(ttl, np.full((1, 5), 10.))


if __name__ == "__main__":
    unittest.main()
