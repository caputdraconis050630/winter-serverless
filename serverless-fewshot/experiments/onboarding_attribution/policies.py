"""Causal policies and the predeclared finite conservative-warming family."""
import itertools
import sys

import numpy as np
from scipy.stats import poisson

from common import ALPHAS, ROOT


def ewma(counts, alpha=.1, initial=0.):
    state=np.full(len(counts),float(initial))
    rates=np.empty(counts.shape,dtype=np.float32)
    for t in range(counts.shape[1]):
        rates[:,t]=np.expm1(np.maximum(state,0.))
        state=(1.-alpha)*state+alpha*np.log1p(counts[:,t])
    return rates


def decisions(rates,rho):
    # Above this ceiling the executed controller already saturates at cap 200.
    r=np.clip(rates,0.,1e6)
    tau=rho/(1.+rho)
    q=(1.-np.exp(-r)>1.-tau).astype(np.int64)
    hi=r>2.
    q[hi]=np.minimum(200,poisson.ppf(tau,r[hi])).astype(np.int64)
    minimum=5.+8.*tau
    ttl=np.clip((.3+1.2*tau)/np.maximum(r,1e-9),minimum,30.)
    ttl=np.where(r<.001,max(2.,minimum*.5),ttl).astype(np.float32)
    return q,ttl


def b1(counts):
    q=np.zeros(counts.shape,dtype=np.int64)
    for t in range(1,counts.shape[1]):
        q[:,t]=(counts[:,max(0,t-10):t]>0).any(axis=1)
    return q,np.full(counts.shape,10.,dtype=np.float32)


def native_baselines(counts,rho,fleet,ageprofile):
    result={"ewma":decisions(ewma(counts),rho),"fixed_keepalive":b1(counts)}
    q,ttl=b1(counts)
    ttl[:,:60]=60.
    result["ageka60"]=(q,ttl)
    result["fleetprior"]=decisions(ewma(counts,initial=np.log1p(fleet)),rho)
    result["ageprofile"]=decisions(np.broadcast_to(ageprofile,counts.shape),rho)
    sys.path.insert(0,str(ROOT))
    from scripts.revision_v1_hybridfull import policy_b2_full
    result["b2f"]=policy_b2_full(counts,verbose_every=100000)
    return result


def simple_configs():
    configs=[]
    for name in ("ewma","fixed_keepalive","ageka60","fleetprior","ageprofile","b2f"):
        configs.append({"id":name,"family":"native","name":name})
    for alpha,prior in itertools.product(ALPHAS,(False,True)):
        prefix=f"ewma_a{alpha:g}_p{int(prior)}"
        configs.append({"id":prefix,"family":"ewma","alpha":alpha,"prior":prior,
                        "scale":1.,"floor":0,"ttl_factor":1.,"horizon":240})
        for s,k,h,H in itertools.product((.5,1.,2.,4.),(0,1,2,4,8),(1.,2.,4.),(15,60,240)):
            if s==1 and k==0 and h==1:
                continue
            configs.append({"id":f"{prefix}_s{s:g}_k{k}_h{h:g}_H{H}","family":"ewma",
                            "alpha":alpha,"prior":prior,"scale":s,"floor":k,
                            "ttl_factor":h,"horizon":H})
    for k,ttl,H in itertools.product((1,2,4,8,16,32,64,128,200),(5,10,30,60,240),(15,60,240)):
        configs.append({"id":f"fixed_k{k}_ttl{ttl}_H{H}","family":"fixed","target":k,"ttl":ttl,"horizon":H})
    return configs


def actions(config,counts,rho,fleet,native,cache):
    if config["family"]=="native":
        return native[config["name"]]
    if config["family"]=="fixed":
        q,ttl=[a.copy() for a in native["ewma"]]
        q[:,:config["horizon"]]=config["target"]
        ttl[:,:config["horizon"]]=config["ttl"]
        return q,ttl
    key=(config["alpha"],config["prior"])
    if key not in cache:
        r=ewma(counts,key[0],np.log1p(fleet) if key[1] else 0.)
        cache[key]=decisions(r,rho)
    q,ttl=[a.copy() for a in cache[key]]
    H=config["horizon"]
    q[:,:H]=np.minimum(200,np.ceil(q[:,:H]*config["scale"])+config["floor"])
    ttl[:,:H]=np.minimum(240,ttl[:,:H]*config["ttl_factor"])
    return q,ttl
