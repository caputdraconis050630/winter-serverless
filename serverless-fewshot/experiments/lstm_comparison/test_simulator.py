"""Independent checks against the existing event-queue and interval evaluators."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "onboarding_attribution"))
import simulator as reference_sim

# Use the production module name so Numba's disk cache is reusable by run.py.
sys.path.pop(0)
del sys.modules["simulator"]
import simulator as minute_sim


class MinuteAccountingTest(unittest.TestCase):
    def test_streamed_events_and_state_equal_single_tape(self):
        from events import event_chunks
        rng = np.random.default_rng(132)
        counts = rng.poisson(4, 300)
        dm = np.linspace(.1, 60., len(counts))
        ds = dm / 3
        whole = list(event_chunks(counts, dm, ds, 1000, "stream-test"))[0][2]
        chunks = list(event_chunks(counts, dm, ds, 1000, "stream-test", max_requests=83, max_minutes=19))
        for column in range(1, 5):
            np.testing.assert_array_equal(np.concatenate([v[column] for _, _, v in chunks]), whole[column])
        q, ttl = rng.integers(0, 8, len(counts)), rng.uniform(.05, 30., len(counts))
        expected = minute_sim.simulate_fast(*whole, q, ttl)
        state = minute_sim.new_state(len(counts))
        for start, end, tape in chunks:
            minute_sim.advance(*tape, q[start:end], ttl[start:end], *state, start, end == len(counts))
        np.testing.assert_array_equal(state[0], expected)

    def fixture(self, length, seed):
        rng = np.random.default_rng(seed)
        counts = rng.poisson(.4, length)
        counts[length // 3:length // 3 + 3] = 35
        q = rng.integers(0, 6, length)
        ttl = rng.choice([.05, .5, 5., 30.], length)
        tape = reference_sim.event_tape(counts, 30., 15., seed, "unit-test", .25, .31)
        return tape, q, ttl

    def test_matches_independent_event_queue(self):
        for seed in range(3):
            tape, q, ttl = self.fixture(240, seed)
            got = minute_sim.simulate_fast(*tape, q, ttl, .25, 7)
            expected, ledger = reference_sim.reference(tape, q, ttl, .25, 7)
            grouped = np.stack([got[:1].sum(0), got[1:15].sum(0),
                                got[15:60].sum(0), got[60:240].sum(0), got[-1]])
            np.testing.assert_allclose(grouped, expected, rtol=1e-11, atol=1e-7)
            self.assertTrue(ledger)

    def test_long_replays_conserve_requests_and_phases(self):
        for length in (1680, 2880):
            tape, q, ttl = self.fixture(length, length)
            got = minute_sim.simulate_fast(*tape, q, ttl, .25, 7)
            expected, _ = reference_sim.reference(tape, q, ttl, .25, 7)
            np.testing.assert_allclose(got.sum(0), expected.sum(0), rtol=1e-11, atol=1e-6)
            np.testing.assert_array_equal(got[:-1, 0], np.diff(tape[0]))
            self.assertTrue(np.all(got[:, 1] <= got[:, 0]))
            self.assertTrue(np.all(got >= -1e-9))
            self.assertEqual(got[-1, 0], 0)
            np.testing.assert_allclose(got[:, 4].sum(), .25 * tape[2].sum(), rtol=1e-11)

    def test_future_actions_do_not_rewrite_past_memory(self):
        tape, q, ttl = self.fixture(240, 19)
        first = minute_sim.simulate_fast(*tape, q, ttl, .25, 7)
        q[60:] = 0
        ttl[60:] = .01
        second = minute_sim.simulate_fast(*tape, q, ttl, .25, 7)
        np.testing.assert_allclose(first[:60], second[:60], rtol=1e-11, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
