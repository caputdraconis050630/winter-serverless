"""Equivalent DES with an ordered ready list and deferred residency integration.

Completed containers arrive in timestamp order, so MRU insertion is constant
time (except exact-time ties). Allocation residency is integrated at removal;
initialization and execution are charged when started. Idle = allocation -
initialization - execution is finalized only at terminal drain. This avoids
re-integrating each retained container's past at every warm request.
"""
import numpy as np
from numba import njit
from simulator import _set,_remove,_charge


@njit(cache=True,nogil=True)
def execution_curve(arrival,duration,length,memory=.25):
    out=np.zeros((length+1,8))
    for i in range(len(arrival)):
        _charge(out,arrival[i],arrival[i]+duration[i],4,memory)
    return out[:,4].copy()


@njit(cache=True,inline="always")
def ready_remove(c,heap,pos,size):
    previous=pos[2,c]
    if previous == -2:return
    following=heap[2,c]
    if previous == -1:size[2]=following
    else:heap[2,previous]=following
    if following >= 0:pos[2,following]=previous
    pos[2,c]=-2;heap[2,c]=-1


@njit(cache=True,inline="always")
def ready_add(c,at,heap,pos,size,key):
    head=size[2];key[2,c]=at
    if head < 0 or key[2,head] < at or (key[2,head] == at and c < head):
        heap[2,c]=head;pos[2,c]=-1
        if head >= 0:pos[2,head]=c
        size[2]=c
    else:
        # Exact completion ties retain the original lowest-slot tie break.
        previous=head;following=heap[2,previous]
        while following >= 0 and key[2,following] == at and following < c:
            previous=following;following=heap[2,previous]
        heap[2,previous]=c;pos[2,c]=previous;heap[2,c]=following
        if following >= 0:pos[2,following]=c


@njit(cache=True,inline="always")
def retire(c,at,out,birth,heap,pos,size,key,memory):
    _charge(out,birth[c],at,2,memory)
    _remove(0,c,heap,pos,size,key);_remove(1,c,heap,pos,size,key)
    ready_remove(c,heap,pos,size)


@njit(cache=True,inline="always")
def expire(at,out,birth,heap,pos,size,key,memory):
    while size[0] > 0 and key[0,heap[0,0]] <= at:
        c=heap[0,0];retire(c,key[0,c],out,birth,heap,pos,size,key,memory)


@njit(cache=True,inline="always")
def complete(at,heap,pos,size,key):
    while size[1] > 0 and key[1,heap[1,0]] <= at:
        c=heap[1,0];value=key[1,c]
        _remove(1,c,heap,pos,size,key)
        ready_add(c,value,heap,pos,size,key)


@njit(cache=True,nogil=True)
def advance(ptr,arrival,duration,cold_init,warm_init,targets,ttl,
            out,heap,pos,size,key,birth,ready,free,tick_offset=0,
            final=True,memory=.25,cap=200,execution_baseline=None):
    if execution_baseline is not None:
        out[:,4]+=execution_baseline
    for t in range(len(targets)):
        at=(t+tick_offset)*60.;ka=ttl[t]*60.;b=t+tick_offset
        out[b,0]+=ptr[t+1]-ptr[t]
        expire(at,out,birth,heap,pos,size,key,memory)
        complete(at,heap,pos,size,key)
        for c in range(cap):
            if pos[0,c]<0:continue
            expiry=free[c]+ka
            if expiry<=at:retire(c,at,out,birth,heap,pos,size,key,memory)
            else:_set(0,c,expiry,heap,pos,size,key)
        deficit=min(cap,targets[t])-size[0];rank=0
        for c in range(cap):
            if rank>=deficit:break
            if pos[0,c]<0:
                birth[c]=at;ready[c]=at+warm_init[t,rank];free[c]=ready[c]
                _charge(out,at,ready[c],3,memory)
                _set(0,c,free[c]+ka,heap,pos,size,key)
                _set(1,c,free[c],heap,pos,size,key)
                rank+=1;out[b,5]+=1
        for i in range(ptr[t],ptr[t+1]):
            at=arrival[i]
            expire(at,out,birth,heap,pos,size,key,memory)
            complete(at,heap,pos,size,key)
            if size[2]>=0:
                c=size[2];ready_remove(c,heap,pos,size)
                free[c]=at+duration[i]
                if execution_baseline is None:_charge(out,at,free[c],4,memory)
                _set(0,c,free[c]+ka,heap,pos,size,key)
                _set(1,c,free[c],heap,pos,size,key)
            else:
                out[b,1]+=1;out[b,6]+=1
                r=at+cold_init[i];f=r+duration[i]
                _charge(out,at,r,3,memory)
                if execution_baseline is not None:_charge(out,at,at+duration[i],4,-memory)
                _charge(out,r,f,4,memory)
                if size[0]>=cap:
                    out[b,7]+=1;_charge(out,at,f,2,memory)
                else:
                    c=0
                    while pos[0,c]>=0:c+=1
                    birth[c]=at;ready[c]=r;free[c]=f
                    _set(0,c,f+ka,heap,pos,size,key)
                    _set(1,c,f,heap,pos,size,key)
    if final:
        for c in range(cap):
            if pos[0,c]>=0:_charge(out,birth[c],key[0,c],2,memory)
        for t in range(len(out)):
            out[t,2]-=out[t,3]+out[t,4]
            if out[t,2]<0 and out[t,2]>-1e-6:out[t,2]=0.
    return out


def new_state(length,cap=200):
    heap=np.zeros((3,cap),np.int64);heap[2]=-1
    pos=np.full((3,cap),-1,np.int64);pos[2]=-2
    size=np.zeros(3,np.int64);size[2]=-1
    return (np.zeros((length+1,8)),heap,pos,size,np.zeros((3,cap)),np.zeros(cap),np.zeros(cap),np.zeros(cap))


def convert_heap_checkpoint(state,memory=.25):
    """Convert an unfinished original ledger without changing realized history."""
    out,heap,pos,size,key,birth,ready,free=(x.copy() for x in state)
    # Previously closed segments already include their full allocated memory.
    out[:,2]+=out[:,3]+out[:,4]
    # The old ledger deferred these last, still-open segments until reuse/expiry.
    for c in np.flatnonzero(pos[0]>=0):
        _charge(out,birth[c],ready[c],3,memory)
        _charge(out,ready[c],free[c],4,memory)
    order=sorted(np.flatnonzero(pos[2]>=0),key=lambda c:(key[2,c],c))
    heap[2]=-1;pos[2]=-2;key[2]=0
    size[2]=order[0] if order else -1
    for i,c in enumerate(order):
        pos[2,c]=order[i-1] if i else -1
        heap[2,c]=order[i+1] if i+1<len(order) else -1
        key[2,c]=free[c]
    return out,heap,pos,size,key,birth,ready,free


def simulate_fast(ptr,arrival,duration,cold_init,warm_init,targets,ttl,memory=.25,cap=200):
    return advance(ptr,arrival,duration,cold_init,warm_init,targets,ttl,*new_state(len(targets),cap),0,True,memory,cap)
