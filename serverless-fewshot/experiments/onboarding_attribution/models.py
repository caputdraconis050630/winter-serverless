"""Cache source-model and random-representation forecasts, no target training."""
import argparse
import os
import sys

import numpy as np
import torch

from common import HERE,LAMBDAS,OUT,ROOT,digest,load_cohort,read_json,save_npz,write_json

sys.path.insert(0,str(ROOT))
os.environ["SF_RUNS_DIR"]=str(ROOT/"results_azure2021/runs")
os.environ.setdefault("SF_DATA_DIR","processed_2019")
torch.set_num_threads(1)
from scripts.eval_adapt_biased import load_trainer,build_prototypes
from src.models.bodies import build_body

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
if DEVICE=="cuda":
    torch.cuda.set_per_process_memory_fraction(.20)


@torch.no_grad()
def embed(body,context):
    out=np.empty((len(context),240,64),dtype=np.float32)
    for f in range(len(context)):
        x=torch.from_numpy(context[f]).unfold(0,60,1).permute(0,2,1).contiguous()
        out[f]=body(x.to(DEVICE)).cpu().numpy()
        if (f+1)%200==0:
            print("embedded",f+1,"/",len(context),flush=True)
    return out


@torch.no_grad()
def canonical(phi,counts,lam,centroids,prior):
    result={k:np.empty(counts.shape,dtype=np.float32) for k in ("component","no_prior","prototype_only")}
    centroid=torch.tensor(centroids,device=DEVICE)
    wp=torch.tensor(prior,device=DEVICE)
    eye=torch.eye(64,device=DEVICE)
    for start in range(0,len(phi),100):
        p=torch.tensor(phi[start:start+100],device=DEVICE)
        y=torch.tensor(np.log1p(counts[start:start+100]),device=DEVICE,dtype=torch.float32)
        n=len(p)
        for t in range(240):
            now=p[:,t]
            ids=torch.nn.functional.cosine_similarity(now[:,None,:],centroid[None,:,:],dim=2).argmax(1)
            w0=wp[ids]
            if t:
                pt=p[:,:t].transpose(1,2)
                a=pt@p[:,:t]+lam*eye
                b=pt@y[:,:t,None].expand(n,t,19)
                wn=torch.linalg.solve(a,b)
                w=torch.linalg.solve(a,b+lam*w0)
            else:
                wn=torch.zeros_like(w0)
                w=w0
            for key,weights in (("component",w),("no_prior",wn),("prototype_only",w0)):
                pred=(now[:,None,:]@weights).squeeze(1)[:,0]
                result[key][start:start+n,t]=torch.expm1(pred.clamp(0.,20.)).cpu().numpy()
    from policies import ewma
    history=np.zeros_like(counts)
    history[:,1:]=np.cumsum(counts[:,:-1],axis=1)
    result["gate"]=np.where(history==0,result["prototype_only"],
                            np.where(history<100,ewma(counts),result["component"]))
    return result


@torch.no_grad()
def ridge_grid(phi,counts):
    """Float64 RLS, algebraically the all-past-row zero-prior ridge solution."""
    results=np.empty((len(LAMBDAS),*counts.shape),dtype=np.float32)
    checks=[]
    for li,lam in enumerate(LAMBDAS):
        for start in range(0,len(phi),128):
            p=torch.tensor(phi[start:start+128],device=DEVICE,dtype=torch.float64)
            y=torch.tensor(np.log1p(counts[start:start+128]),device=DEVICE,dtype=torch.float64)
            n=len(p)
            inv=torch.eye(64,device=DEVICE,dtype=torch.float64).expand(n,64,64).clone()/lam
            coef=torch.zeros((n,64),device=DEVICE,dtype=torch.float64)
            for t in range(240):
                x=p[:,t]
                pred=(x*coef).sum(1)
                results[li,start:start+n,t]=torch.expm1(pred.clamp(0.,20.)).cpu().numpy()
                if start==0 and t in (1,15,60,239):
                    a=p[0,:t].T@p[0,:t]+lam*torch.eye(64,device=DEVICE,dtype=torch.float64)
                    direct=torch.linalg.solve(a,p[0,:t].T@y[0,:t])
                    error=float(abs(x[0]@direct-pred[0]))
                    checks.append(error)
                    if error>1e-4:
                        raise RuntimeError(f"RLS/direct disagreement {error}, lambda {lam}")
                u=(inv@x[:,:,None]).squeeze(2)
                denom=1.+(x*u).sum(1)
                gain=u/denom[:,None]
                coef+=gain*(y[:,t]-pred)[:,None]
                inv-=gain[:,:,None]*u[:,None,:]
    return results,max(checks)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models",default="trained0,trained1,trained2,random0,random1,random2,random3,random4")
    args=ap.parse_args()
    names=["azure_calibration","azure_primary","azure_evaluation","huawei"]
    source=ROOT/"data/processed"
    counts21=np.load(source/"counts.npy",mmap_mode="r")
    features21=np.load(source/"features.npy",mmap_mode="r")
    splits=np.load(source/"splits.npz")
    fleet=float(np.median(np.asarray(counts21[splits["s2_train"]]).mean(axis=1)))
    cal=load_cohort("azure_calibration")["counts"]
    profile=np.repeat(np.log1p(cal).reshape(len(cal),48,5).mean(axis=(0,2)),5)
    write_json(OUT/"simple_priors.json",{"fleet_rate":fleet,"age_profile_rates":np.expm1(profile).tolist(),
                                       "fleet_source":"Azure2021 s2_train median function mean rate",
                                       "age_profile_source":"Azure calibration only, five-minute bins"})
    for model in args.models.split(","):
        seed=int(model[-1])
        trained=model.startswith("trained")
        directory=OUT/"models"/model
        directory.mkdir(parents=True,exist_ok=True)
        if trained:
            trainer,biased=load_trainer(11,seed)
            body=trainer.body.eval()
            lam=trainer.head.ridge_lambda.item()
            proto_path=directory/"prototypes.npz"
            if proto_path.exists():
                z=np.load(proto_path); centroid,prior=z["centroids"],z["prior"]
            else:
                c,p=build_prototypes(trainer,biased,features21,counts21,splits["s2_train"])
                centroid,prior=c.numpy(),p.numpy()
                save_npz(proto_path,centroids=centroid,prior=prior)
            write_json(directory/"model.json",{"source":"Azure2021 s2","seed":seed,"lambda":lam,
                       "checkpoint_sha256":digest(ROOT/f"results_azure2021/runs/best_anil_ridge_s2_s{seed}.pt")})
        else:
            torch.manual_seed(seed)
            body=build_body("tcn",11).to(DEVICE).eval()
            torch.save(body.state_dict(),directory/"body.pt")
            write_json(directory/"model.json",{"source":"random frozen initialization; no source-trained weights","seed":seed,
                                               "body_sha256":digest(directory/"body.pt")})
        for cohort in names:
            output=directory/(cohort+"_rates.npz")
            if output.exists():
                continue
            z=load_cohort(cohort)
            cache=directory/(cohort+"_phi.npz")
            if cache.exists():
                phi=np.load(cache)["phi"]
            else:
                print(model,cohort,"embedding",flush=True)
                phi=embed(body,z["context"])
                save_npz(cache,phi=phi)
            arrays={}
            if trained:
                arrays.update(canonical(phi,z["counts"],lam,centroid,prior))
            if not trained or seed==0:
                arrays["lambda_rates"],error=ridge_grid(phi,z["counts"])
                write_json(directory/(cohort+"_ridge_check.json"),{"max_log_prediction_error":error})
            save_npz(output,**arrays)
            print(model,cohort,"rates saved",flush=True)


if __name__=="__main__":
    main()
