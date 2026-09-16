"""Equivalent replay reusing exogenous warm-completion order across policies.

The common request tape determines warm service completion times. A policy
only maps each warm request to its chosen container; cold/proactive completion
events remain in its private heap. At chunk boundaries pending warm events
are transferred to that heap, preserving the ordinary checkpoint format.
"""
import numpy as np
from numba import njit
from simulator import _set,_remove,_charge
from simulator_fast import (new_state,execution_curve,ready_add,ready_remove,
                            retire,expire,convert_heap_checkpoint)


@njit(cache=True,inline="always")
def complete(at,order,times,mapping,pointer,heap,pos,size,key,free):
    while True:
        warm_time=times[order[pointer]] if pointer<len(order) else np.inf
        other_time=key[1,heap[1,0]] if size[1]>0 else np.inf
        if min(warm_time,other_time)>at:break
        if warm_time<=other_time:
            c=mapping[order[pointer]];pointer+=1
            if c>=0 and pos[0,c]>=0 and free[c]==warm_time:
                ready_add(c,warm_time,heap,pos,size,key)
        else:
            c=heap[1,0];_remove(1,c,heap,pos,size,key)
            ready_add(c,other_time,heap,pos,size,key)
    return pointer


@njit(cache=True,nogil=True)
def advance(ptr,arrival,duration,cold_init,warm_init,targets,ttl,
            out,heap,pos,size,key,birth,ready,free,tick_offset,final,memory,cap,
            execution_baseline,completion_order,completion_times):
    out[:,4]+=execution_baseline
    mapping=np.full(len(arrival),-1,np.int16);pointer=0
    for t in range(len(targets)):
        at=(t+tick_offset)*60.;ka=ttl[t]*60.;b=t+tick_offset
        out[b,0]+=ptr[t+1]-ptr[t]
        expire(at,out,birth,heap,pos,size,key,memory)
        pointer=complete(at,completion_order,completion_times,mapping,pointer,heap,pos,size,key,free)
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
                _set(0,c,free[c]+ka,heap,pos,size,key);_set(1,c,free[c],heap,pos,size,key)
                rank+=1;out[b,5]+=1
        for i in range(ptr[t],ptr[t+1]):
            at=arrival[i]
            expire(at,out,birth,heap,pos,size,key,memory)
            pointer=complete(at,completion_order,completion_times,mapping,pointer,heap,pos,size,key,free)
            if size[2]>=0:
                c=size[2];ready_remove(c,heap,pos,size)
                free[c]=completion_times[i];mapping[i]=c
                _set(0,c,free[c]+ka,heap,pos,size,key)
            else:
                out[b,1]+=1;out[b,6]+=1
                r=at+cold_init[i];f=r+duration[i]
                _charge(out,at,r,3,memory);_charge(out,at,completion_times[i],4,-memory);_charge(out,r,f,4,memory)
                if size[0]>=cap:out[b,7]+=1;_charge(out,at,f,2,memory)
                else:
                    c=0
                    while pos[0,c]>=0:c+=1
                    birth[c]=at;ready[c]=r;free[c]=f
                    _set(0,c,f+ka,heap,pos,size,key);_set(1,c,f,heap,pos,size,key)
    if final:
        for c in range(cap):
            if pos[0,c]>=0:_charge(out,birth[c],key[0,c],2,memory)
        for t in range(len(out)):
            out[t,2]-=out[t,3]+out[t,4]
            if out[t,2]<0 and out[t,2]>-1e-6:out[t,2]=0.
    else:
        for c in range(cap):
            if pos[0,c]>=0 and pos[2,c]==-2 and pos[1,c]<0:
                _set(1,c,free[c],heap,pos,size,key)
    return out
