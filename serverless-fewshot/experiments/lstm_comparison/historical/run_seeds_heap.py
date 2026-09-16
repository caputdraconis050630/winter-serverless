"""Same exact DES, scheduled by function/seed with resumable long-function state."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time
import numpy as np
from common import OUT,HERE,METRICS,digest,freeze,read_json,save_npz,write_json
from events import event_chunks
from simulator import advance,new_state
from run import windows_for,aggregate


def run(name,workers=6):
    directory=OUT/"cases"/name
    meta=read_json(directory/"case.json")
    with np.load(directory/"data.npz") as z:data={k:z[k] for k in z.files}
    assert digest(directory/"data.npz")==meta["data_sha256"]
    names=sorted(meta["actions"]);actions=[]
    for name in names:
        rec=meta["actions"][name];path=directory/rec["file"]
        assert digest(path)==rec["sha256"]
        with np.load(path) as z:actions.append((z["q"],z["ttl"]))
    horizon=meta["horizon_minutes"];windows=windows_for(meta)
    contract=dict(case_sha256=digest(directory/"case.json"),actions=names,seeds=meta["seeds"],windows=windows,
        metrics=list(METRICS),memory_gb=.25,cap=200,simulator_sha256=digest(HERE/"simulator.py"),
        events_sha256=digest(HERE/"events.py"),runner_sha256=digest(__file__),aggregate_sha256=digest(HERE/"run.py"),
        phase_accounting="wall-clock minutes plus separate terminal drain",ttl="prospective",
        request_streams="policy independent; stable function and semantic RNG streams",
        scheduling="function/seed jobs; state checkpoint every 300 seconds; same event loop and aggregate as run.py")
    freeze(directory/"execution.json",contract);contract_hash=digest(directory/"execution.json")
    destination=directory/"seed_functions";destination.mkdir(exist_ok=True)
    partial=directory/"partial_states";partial.mkdir(exist_ok=True)
    (directory/"functions").mkdir(exist_ok=True)

    def job(f,s):
        seed=meta["seeds"][s];path=destination/f"{f:05d}_{s:02d}.npz"
        if path.exists():
            with np.load(path) as z:assert str(z["contract_sha256"])==contract_hash
            return f,s
        unique=[];mapping=[];seen={}
        for qa,ta in actions:
            q,ttl=np.ascontiguousarray(qa[f]),np.ascontiguousarray(ta[f])
            key=hashlib.sha256(q.tobytes()+ttl.tobytes()).digest()
            if key not in seen:seen[key]=len(unique);unique.append((q,ttl))
            mapping.append(seen[key])
        states=[new_state(horizon) for _ in unique]
        resume=partial/f"{f:05d}_{s:02d}.npz"
        offset=0;execution=0.
        if resume.exists():
            with np.load(resume) as z:
                assert str(z["contract_sha256"])==contract_hash
                offset=int(z["offset"]);execution=float(z["execution"])
                states=[tuple(z[f"state{k}"][j].copy() for k in range(8)) for j in range(len(unique))]
        start=last_report=last_save=time.monotonic()
        counts=data["counts"][f]
        for lo,hi,tape in event_chunks(counts,data["duration_mean"][f],data["duration_std"][f],seed,
                                       meta["event_key_prefix"]+str(data["keys"][f])):
            if hi<=offset:continue
            assert lo>=offset
            execution+=.25*tape[2].sum()
            for j,(q,ttl) in enumerate(unique):advance(*tape,q[lo:hi],ttl[lo:hi],*states[j],lo,hi==horizon)
            now=time.monotonic()
            if now-last_report>60:
                progress=dict(case=meta["name"],function=f,seed=seed,minute=hi,horizon=horizon,
                              processed_requests=int(counts[:hi].sum()),total_requests=int(counts.sum()),elapsed_seconds=now-start)
                write_json(partial/f"{f:05d}_{s:02d}.json",progress)
                print(f'{meta["name"]} f={f} seed={seed}: {hi}/{horizon} minutes ({progress["processed_requests"]:,} requests)',flush=True)
                last_report=now
            if now-last_save>300 and hi<horizon:
                save_npz(resume,**{f"state{k}":np.stack([st[k] for st in states]) for k in range(8)},
                         offset=np.array(hi),execution=np.array(execution),contract_sha256=np.array(contract_hash))
                last_save=now
        stats=np.zeros((len(unique),len(windows),8));cold=[];idle=[]
        for j,state in enumerate(states):
            out=state[0]
            np.testing.assert_array_equal(out[:-1,0],counts)
            assert np.isfinite(out).all() and np.all(out>=-1e-6) and np.all(out[:,1]<=out[:,0])
            np.testing.assert_allclose(out[:,4].sum(),execution,rtol=2e-9,atol=1e-5)
            cold.append(np.rint(out[:,1]).astype(np.uint64));idle.append(out[:,2])
            for w,(lo,hi) in enumerate(windows.values()):stats[j,w]=out[lo:hi].sum(0)
        save_npz(path,mapping=np.array(mapping),cold=np.stack(cold),idle=np.stack(idle),stats=stats,
                 contract_sha256=np.array(contract_hash),function_index=np.array(f),seed=np.array(seed))
        if resume.exists():resume.unlink()
        return f,s

    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks=[pool.submit(job,f,s) for f in range(len(data["counts"])) for s in range(len(meta["seeds"]))]
        for done,future in enumerate(as_completed(tasks),1):
            f,s=future.result()
            if done%25==0 or done==len(tasks):print(f'{meta["name"]}: {done}/{len(tasks)} function-seeds; {time.monotonic()-start:.1f}s',flush=True)
    for f in range(len(data["counts"])):
        rows=[]
        for s in range(len(meta["seeds"])):
            with np.load(destination/f"{f:05d}_{s:02d}.npz") as z:rows.append({k:z[k] for k in z.files})
        for row in rows:np.testing.assert_array_equal(row["mapping"],rows[0]["mapping"])
        save_npz(directory/"functions"/f"{f:05d}.npz",mapping=rows[0]["mapping"],
            cold=np.stack([r["cold"] for r in rows]).sum(0),idle=np.stack([r["idle"] for r in rows]).mean(0).astype(np.float32),
            stats=np.stack([r["stats"] for r in rows]),contract_sha256=np.array(contract_hash),function_index=np.array(f))
    aggregate(directory,meta,data,names,windows)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("case");ap.add_argument("--workers",type=int,default=6)
    args=ap.parse_args();run(args.case,args.workers)
