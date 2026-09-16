"""Critical causal and saturation boundaries for the additional experiments."""
import unittest
import numpy as np
from cases import decisions_from_rates
from live import causal_schedule
from drift import causal_embeddings, make_segment_features


class CausalityTest(unittest.TestCase):
    def test_poisson_target_saturates_before_integer_overflow(self):
        for tau in (.1/1.1, .5, 10/11, 100/101):
            q, ttl = decisions_from_rates(np.array([[1e6,1e12,5e26]]),tau)
            np.testing.assert_array_equal(np.minimum(q,200), [[200,200,200]])
            self.assertTrue(np.isfinite(ttl).all())

    def test_live_schedule_excludes_current_and_future_counts(self):
        counts = np.zeros((1,60),int)
        q,ttl = np.zeros_like(counts),np.full(counts.shape,10.)
        before = causal_schedule(counts,q,ttl)
        counts[:,30:] = 8
        after = causal_schedule(counts,q,ttl)
        np.testing.assert_array_equal(before[:,:31],after[:,:31])
        self.assertEqual(after[0,31],1)

    def test_drift_encoder_reads_only_past_bins(self):
        import torch
        class LastCount(torch.nn.Module):
            def forward(self,x):return x[:,-1,:]
        class Trainer:
            body=LastCount()
            device="cpu"
        counts = np.arange(100)
        feats = make_segment_features(counts,np.zeros(100),0)
        before = causal_embeddings(Trainer(),feats)
        counts[40:]=10000
        after = causal_embeddings(Trainer(),make_segment_features(counts,np.zeros(100),0))
        np.testing.assert_array_equal(before[:41],after[:41])
        self.assertNotEqual(before[41,0],after[41,0])


if __name__ == "__main__":unittest.main()
