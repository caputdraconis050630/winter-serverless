import unittest
import time
import numpy as np
import simulator
import simulator_fast
from events import event_chunks


class EquivalentEngineTest(unittest.TestCase):
    def test_random_sparse_dense_and_ttl_transitions(self):
        for length,rate,cap in ((240,.2,7),(1680,4,200),(2880,20,200),(120,10000,200)):
            rng=np.random.default_rng(length)
            counts=rng.poisson(rate,length)
            q=rng.integers(0,cap+1,length).astype(np.int16)
            ttl=rng.choice([.01,.1,5.,10.,30.],length).astype(np.float32)
            tapes=list(event_chunks(counts,.2,.1,19,"engine-check",max_requests=10000000))
            old=simulator.new_state(length,cap);new=simulator_fast.new_state(length,cap)
            shared=simulator_fast.new_state(length,cap)
            for lo,hi,tape in tapes:
                simulator.advance(*tape,q[lo:hi],ttl[lo:hi],*old,lo,hi==length,.25,cap)
                simulator_fast.advance(*tape,q[lo:hi],ttl[lo:hi],*new,lo,hi==length,.25,cap)
                baseline=simulator_fast.execution_curve(tape[1],tape[2],length)
                simulator_fast.advance(*tape,q[lo:hi],ttl[lo:hi],*shared,lo,hi==length,.25,cap,baseline)
            np.testing.assert_array_equal(new[0][:,[0,1,5,6,7]],old[0][:,[0,1,5,6,7]])
            np.testing.assert_allclose(new[0],old[0],rtol=1e-9,atol=2e-6)
            np.testing.assert_array_equal(shared[0][:,[0,1,5,6,7]],old[0][:,[0,1,5,6,7]])
            np.testing.assert_allclose(shared[0],old[0],rtol=1e-9,atol=2e-6)

    def test_exact_completion_ties(self):
        counts=np.array([20,0,20,0,20])
        _,_,tape=next(event_chunks(counts,1.,.01,2,"ties"))
        ptr,arr,dur,cold,warm=tape
        for t in range(5):arr[ptr[t]:ptr[t+1]]=t*60+5
        dur[:]=1;cold[:]=1;warm[:]=1
        q=np.full(5,10,np.int16);ttl=np.array([1,0,1,0,1],np.float32)
        a=simulator.simulate_fast(*tape,q,ttl,cap=20)
        b=simulator_fast.simulate_fast(*tape,q,ttl,cap=20)
        np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-12)

    def test_unfinished_heap_checkpoint_migration(self):
        rng=np.random.default_rng(341)
        counts=rng.poisson(8,360)
        q=rng.integers(0,30,len(counts)).astype(np.int16)
        ttl=rng.choice([.01,.1,10,30],len(counts)).astype(np.float32)
        old=simulator.new_state(len(counts));new=None
        for lo,hi,tape in event_chunks(counts,25.,10.,3,"migration",max_minutes=120):
            simulator.advance(*tape,q[lo:hi],ttl[lo:hi],*old,lo,hi==len(counts))
            if new is None:new=simulator_fast.convert_heap_checkpoint(old)
            else:
                baseline=simulator_fast.execution_curve(tape[1],tape[2],len(counts))
                simulator_fast.advance(*tape,q[lo:hi],ttl[lo:hi],*new,lo,hi==len(counts),.25,200,baseline)
        np.testing.assert_array_equal(new[0][:,[0,1,5,6,7]],old[0][:,[0,1,5,6,7]])
        np.testing.assert_allclose(new[0],old[0],rtol=1e-9,atol=2e-6)


if __name__=="__main__":unittest.main()
