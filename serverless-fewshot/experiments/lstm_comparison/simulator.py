"""Common-event, prospective-TTL DES with minute-level phase accounting.

Derived from onboarding_attribution/heap_simulator.py. The allocation and
request rules are preserved; every minute and a separate terminal drain are
accounted explicitly. The original four-hour evaluator is not modified.
"""
import numpy as np
from numba import njit

@njit(cache=True, inline="always")
def _charge(out, start, end, col, memory):
    if end <= start:
        return
    horizon = (out.shape[0] - 1) * 60.
    lo = max(0., start)
    hi = min(end, horizon)
    if hi > lo:
        first = int(lo // 60.)
        last = min(out.shape[0] - 2, int(hi // 60.))
        for minute in range(first, last + 1):
            overlap = min(hi, (minute + 1) * 60.) - max(lo, minute * 60.)
            if overlap > 0.:
                out[minute, col] += overlap * memory
    if end > horizon:
        out[-1, col] += max(0., end - max(start, horizon)) * memory


@njit(cache=True, inline="always")
def _close(out, birth, ready, free, stop, memory):
    _charge(out, birth, min(ready, stop), 3, memory)
    _charge(out, ready, min(free, stop), 4, memory)
    _charge(out, free, stop, 2, memory)



@njit(cache=True,inline="always")
def _less(k,a,b,key):
    return key[k,a]<key[k,b] or (key[k,a]==key[k,b] and a<b)


@njit(cache=True)
def _repair(k,p,heap,pos,size,key):
    while p>0:
        parent=(p-1)//2
        if not _less(k,heap[k,p],heap[k,parent],key):
            break
        a,b=heap[k,p],heap[k,parent]
        heap[k,p],heap[k,parent]=b,a
        pos[k,a],pos[k,b]=parent,p
        p=parent
    while p*2+1<size[k]:
        child=p*2+1
        if child+1<size[k] and _less(k,heap[k,child+1],heap[k,child],key):
            child+=1
        if not _less(k,heap[k,child],heap[k,p],key):
            break
        a,b=heap[k,p],heap[k,child]
        heap[k,p],heap[k,child]=b,a
        pos[k,a],pos[k,b]=child,p
        p=child


@njit(cache=True)
def _set(k,c,value,heap,pos,size,key):
    key[k,c]=value
    if pos[k,c]<0:
        p=size[k]; size[k]+=1
        heap[k,p]=c; pos[k,c]=p
    _repair(k,pos[k,c],heap,pos,size,key)


@njit(cache=True)
def _remove(k,c,heap,pos,size,key):
    p=pos[k,c]
    if p<0:
        return
    size[k]-=1
    last=heap[k,size[k]]
    pos[k,c]=-1
    if p<size[k]:
        heap[k,p]=last; pos[k,last]=p
        _repair(k,p,heap,pos,size,key)


@njit(cache=True)
def _expire(at,out,birth,ready,free,heap,pos,size,key,memory):
    while size[0]>0 and key[0,heap[0,0]]<=at:
        c=heap[0,0]
        _close(out,birth[c],ready[c],free[c],key[0,c],memory)
        for k in range(3):
            _remove(k,c,heap,pos,size,key)


@njit(cache=True)
def _complete(at,heap,pos,size,key):
    while size[1]>0 and key[1,heap[1,0]]<=at:
        c=heap[1,0]
        value=key[1,c]
        _remove(1,c,heap,pos,size,key)
        _set(2,c,-value,heap,pos,size,key)


@njit(cache=True,nogil=True)
def advance(ptr,arrival,duration,cold_init,warm_init,targets,ttl,
            out,heap,pos,size,key,birth,ready,free,tick_offset=0,
            final=True,memory=.25,cap=200):
    for t in range(len(targets)):
        at=(t+tick_offset)*60.; ka=ttl[t]*60.
        b=t+tick_offset
        _expire(at,out,birth,ready,free,heap,pos,size,key,memory)
        _complete(at,heap,pos,size,key)
        for c in range(cap):
            if pos[0,c]<0:
                continue
            expiry=free[c]+ka
            if expiry<=at:
                _close(out,birth[c],ready[c],free[c],at,memory)
                for k in range(3):
                    _remove(k,c,heap,pos,size,key)
            else:
                _set(0,c,expiry,heap,pos,size,key)
        deficit=min(cap,targets[t])-size[0]
        rank=0
        for c in range(cap):
            if rank>=deficit:
                break
            if pos[0,c]<0:
                birth[c]=at; ready[c]=at+warm_init[t,rank]; free[c]=ready[c]
                _set(0,c,free[c]+ka,heap,pos,size,key)
                _set(1,c,free[c],heap,pos,size,key)
                rank+=1; out[b,5]+=1
        for i in range(ptr[t],ptr[t+1]):
            at=arrival[i]
            _expire(at,out,birth,ready,free,heap,pos,size,key,memory)
            _complete(at,heap,pos,size,key)
            out[b,0]+=1
            if size[2]>0:
                c=heap[2,0]
                _close(out,birth[c],ready[c],free[c],at,memory)
                _remove(2,c,heap,pos,size,key)
                birth[c]=at; ready[c]=at; free[c]=at+duration[i]
                _set(0,c,free[c]+ka,heap,pos,size,key)
                _set(1,c,free[c],heap,pos,size,key)
            else:
                out[b,1]+=1; out[b,6]+=1
                r=at+cold_init[i]; f=r+duration[i]
                if size[0]>=cap:
                    out[b,7]+=1
                    _close(out,at,r,f,f,memory)
                else:
                    c=0
                    while pos[0,c]>=0:
                        c+=1
                    birth[c]=at; ready[c]=r; free[c]=f
                    _set(0,c,f+ka,heap,pos,size,key)
                    _set(1,c,f,heap,pos,size,key)
    if final:
        for c in range(cap):
            if pos[0,c]>=0:
                _close(out,birth[c],ready[c],free[c],key[0,c],memory)
    return out


def new_state(length, cap=200):
    return (np.zeros((length+1,8)), np.zeros((3,cap),dtype=np.int64),
            np.full((3,cap),-1,dtype=np.int64), np.zeros(3,dtype=np.int64),
            np.zeros((3,cap)), np.zeros(cap), np.zeros(cap), np.zeros(cap))


def simulate_fast(ptr,arrival,duration,cold_init,warm_init,targets,ttl,
                  memory=.25,cap=200,legacy_ttl=False):
    if legacy_ttl:
        raise ValueError("This evaluator only supports prospective TTL")
    state = new_state(len(targets),cap)
    return advance(ptr,arrival,duration,cold_init,warm_init,targets,ttl,
                   *state,0,True,memory,cap)
