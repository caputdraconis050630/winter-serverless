import unittest
import numpy as np
import simulator
import simulator_fast
import simulator_events as shared
from events import event_chunks


class SharedCompletionTest(unittest.TestCase):
    def compare(self,counts,mean,std,cap,seed,ties=False,migrate=False):
        n=len(counts);rng=np.random.default_rng(seed)
        q=rng.integers(0,cap+1,n).astype(np.int16)
        ttl=rng.choice([0.,.01,.1,5.,30.],n).astype(np.float32)
        old=simulator.new_state(n,cap);new=shared.new_state(n,cap)
        for lo,hi,tape in event_chunks(counts,mean,std,seed,"shared-completions",max_requests=40000,max_minutes=73):
            if ties:
                ptr,arr,dur,cold,warm=tape
                for t in range(hi-lo):arr[ptr[t]:ptr[t+1]]=(lo+t)*60+5
                dur[:]=1.;cold[:]=1.;warm[:]=1.
            simulator.advance(*tape,q[lo:hi],ttl[lo:hi],*old,lo,hi==n,.25,cap)
            if migrate and lo==0:new=shared.convert_heap_checkpoint(old);continue
            baseline=shared.execution_curve(tape[1],tape[2],n)
            times=tape[1]+tape[2];order=np.argsort(times,kind="stable")
            shared.advance(*tape,q[lo:hi],ttl[lo:hi],*new,lo,hi==n,.25,cap,baseline,order,times)
        np.testing.assert_array_equal(new[0][:,[0,1,5,6,7]],old[0][:,[0,1,5,6,7]])
        np.testing.assert_allclose(new[0],old[0],rtol=1e-9,atol=3e-6)

    def test_random_density_duration_and_capacity(self):
        for rate,mean,cap in [(.2,.01,7),(8,25.,20),(10000,.01,200),(300,120.,200)]:
            self.compare(np.random.default_rng(19).poisson(rate,180),mean,mean/2,cap,19)

    def test_equal_timestamps_and_chunk_carryover(self):
        self.compare(np.full(240,20),1.,.1,20,41,ties=True)

    def test_checkpoint_migration(self):
        self.compare(np.full(240,20),50.,30.,20,91,migrate=True)


if __name__=="__main__":unittest.main()
