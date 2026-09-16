import unittest
import numpy as np
from experiment import OUT, write_json
from simulator import simulate, reference
from certificates import make_certificate, reuse


class CertificateTests(unittest.TestCase):
    def test_deterministic_positive_and_drain(self):
        counts = np.ones(240, dtype=np.int64)
        tape = (np.arange(241), np.arange(240)*60.+5., np.full(240, 61.),
                np.ones(240), np.ones((240, 200)))
        for floor in (2., 6.136363506317139, 12., 30., 60.):
            q = np.full(240, 2, dtype=np.int64)
            ttl = np.full(240, floor)
            ttl[30:70] *= 4.
            ttl[-1] *= 2.
            cert = make_certificate(tape, 2, floor)
            fast = reuse(cert, q, ttl, floor)
            self.assertIsNotNone(fast)
            np.testing.assert_allclose(fast, simulate(*tape, q, ttl), rtol=1e-10, atol=1e-7)
            exact, _ = reference(tape, q, ttl)
            np.testing.assert_allclose(fast, exact, rtol=1e-10, atol=1e-7)

    def test_unsafe_actions_rejected(self):
        tape = (np.arange(241), np.arange(240)*60.+5., np.full(240, 61.),
                np.ones(240), np.ones((240, 200)))
        q = np.full(240, 2, dtype=np.int64)
        ttl = np.full(240, 12.)
        cert = make_certificate(tape, 2, 12.)
        q[30] = 200
        self.assertIsNone(reuse(cert, q, ttl, 12.))
        q[30] = 2
        ttl[80] = 11.
        self.assertIsNone(reuse(cert, q, ttl, 12.))

    def test_expiring_reference_rejected(self):
        tape = (np.arange(241), np.arange(240)*60.+5., np.full(240, 1.),
                np.ones(240), np.ones((240, 200)))
        q = np.full(240, 2, dtype=np.int64)
        ttl = np.full(240, 5.)
        self.assertIsNone(reuse(make_certificate(tape, 2, 5.), q, ttl, 5.))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CertificateTests))
    write_json(OUT / "certificate_tests.json", dict(passed=result.wasSuccessful(), tests=result.testsRun))
    raise SystemExit(0 if result.wasSuccessful() else 1)
