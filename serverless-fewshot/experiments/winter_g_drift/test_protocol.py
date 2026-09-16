"""Scientific checks for timing, routing and the additional adapter."""
import unittest
import numpy as np
from campaign import routing, fit_rates, causal_embeddings, make_segment_features

class ProtocolTest(unittest.TestCase):
    def test_count_and_age_boundaries_and_no_shift_reset(self):
        c=np.zeros((1,1680),np.int64);c[0,100]=99;c[0,102]=1;c[0,1440:]=20
        s=routing(c,'azure2019')[0]
        self.assertEqual(s[100],0);self.assertEqual(s[101],1);self.assertEqual(s[102],1)
        self.assertEqual(s[103],2);self.assertEqual(s[819],2);self.assertEqual(s[820],3)
        self.assertEqual(s[1440],3);self.assertEqual(s[1441],3)
        self.assertEqual(routing(c,'huawei')[0,720],3)

    def test_unobserved_arrival_does_not_change_route_prefix(self):
        a=np.zeros((1,1700),int);b=a.copy();a[0,1200]=200;b[0,1300]=200
        np.testing.assert_array_equal(routing(a,'azure2019')[:,:1201],routing(b,'azure2019')[:,:1201])

    def test_current_future_targets_excluded_and_direct_ridge(self):
        rng=np.random.default_rng(42);p=rng.normal(size=(2,85,5)).astype(np.float32)
        c=rng.poisson(3,size=(2,85));cent=rng.normal(size=(4,5));prior=rng.normal(size=(4,5))
        a,proto,error=fit_rates(p,c,.75,cent,prior,device='cpu')
        altered=c.copy();altered[:,40:]=900
        b,_,_=fit_rates(p,altered,.75,cent,prior,device='cpu')
        np.testing.assert_array_equal(a[:,:41],b[:,:41]);self.assertLess(error,1e-9)
        for t in (0,1,20,84):
            x=p[0,t].astype(float);idx=np.argmax(cent@x/np.maximum(np.linalg.norm(cent,axis=1)*np.linalg.norm(x),1e-8))
            w=np.linalg.solve(p[0,:t].astype(float).T@p[0,:t]+.75*np.eye(5),p[0,:t].astype(float).T@np.log1p(c[0,:t])+.75*prior[idx])
            np.testing.assert_allclose(a[0,t],np.expm1(np.clip(x@w,0,20)),rtol=1e-6,atol=1e-7)

    def test_encoder_excludes_current_and_future_features(self):
        import torch
        class Body(torch.nn.Module):
            def forward(self,x):return x.mean(1)
        class Trainer:
            body=Body();device='cpu'
        c=np.arange(100);f=make_segment_features(c,np.zeros(100),0)
        a=causal_embeddings(Trainer(),f);c[40:]=10000
        b=causal_embeddings(Trainer(),make_segment_features(c,np.zeros(100),0))
        np.testing.assert_array_equal(a[:41],b[:41]);self.assertFalse(np.array_equal(a[41],b[41]))

if __name__=='__main__':unittest.main()
