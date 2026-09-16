"""Freeze the protocol, application-disjoint cohorts, and input provenance."""
import hashlib
import platform
import sys

import numpy as np
import pandas as pd

from common import (ALPHAS, BUDGETS, HERE, LAMBDAS, OUT, RHOS, ROOT,
                    digest, read_json, save_npz, write_json)


def scan(counts):
    first = np.full(len(counts), -1, dtype=np.int64)
    day1 = np.zeros(len(counts), dtype=np.int64)
    for f, row in enumerate(counts):
        nz = np.flatnonzero(row)
        if len(nz):
            first[f] = nz[0]
            day1[f] = row[nz[0]:nz[0] + 1440].sum()
    return first, day1


def candidates(first, day1, horizon, total):
    return np.flatnonzero((first >= 4320) & (first < total - horizon) & (day1 >= 30))


def capped(ids):
    if len(ids) > 100:
        pick = np.random.default_rng(42).choice(len(ids), 100, replace=False)
        ids = ids[np.sort(pick)]
    return ids


def export(name, proc, ids, first, apps):
    path = OUT / "cohorts" / (name + ".npz")
    if path.exists():
        return
    counts = np.load(proc / "counts.npy", mmap_mode="r")
    feats = np.load(proc / "features.npy", mmap_mode="r")
    durations = pd.read_csv(proc / "duration_stats.csv")
    rawkeys = np.load(proc / "func_ids.npy", allow_pickle=True)
    seg = np.stack([counts[f, first[f]:first[f] + 240] for f in ids])
    context = np.stack([feats[f, first[f] - 60:first[f] + 239] for f in ids])
    keys = np.array([str(rawkeys[f]) for f in ids], dtype=str)
    save_npz(path, ids=np.asarray(ids), t0=first[ids], counts=seg,
             context=context, keys=keys, apps=np.asarray(apps[ids], dtype=str),
             duration_mean=np.nan_to_num(durations.dur_mean.to_numpy()[ids], nan=1.),
             duration_std=np.nan_to_num(durations.dur_std.to_numpy()[ids], nan=.5))
    print(name, len(ids), "functions", len(set(apps[ids])), "groups", int(seg.sum()), "requests", flush=True)


def main():
    protocol = {
        "version": "onboarding-attribution-v1", "freeze_date_utc": "2026-09-07",
        "question": "Information value beyond more conservative warming; not meta-learning versus supervised superiority",
        "rhos": RHOS, "primary_rho": 10., "window_minutes": 240,
        "validation_seeds": list(range(5)), "evaluation_seeds": list(range(1000, 1020)),
        "model_seeds": [0, 1, 2], "primary_model_seed": 0,
        "random_representation_seeds": list(range(5)), "lambdas": LAMBDAS,
        "bootstrap_seed": 260907, "bootstrap_replicates": 10000,
        "ewma_alphas": ALPHAS, "initial_states": ["zero", "source_fleet_median"],
        "budget_multipliers": BUDGETS, "matched_wm_ratio_ci_band": [.99, 1.01],
        "boost_scale": [.5, 1., 2., 4.], "boost_floor": [0, 1, 2, 4, 8],
        "boost_ttl_factor": [1., 2., 4.], "boost_horizon": [15, 60, 240],
        "fixed_targets": [1, 2, 4, 8, 16, 32, 64, 128, 200],
        "fixed_ttls": [5, 10, 30, 60, 240], "pool_cap": 200,
        "memory_gb": .25, "cold_init": {"mu": .25, "sigma": .31},
        "application_partition": "SHA256('winter-attribution-v1|' + app_id) integer mod 5: 0,1 calibration; 2,3,4 evaluation",
        "data_scope": "Previously examined traces; fresh comparison settings, not pristine unseen datasets",
        "selection": "minimum validation cold subject to idle WM budget; tie lower WM then lexicographic arm ID",
        "representation_selection": "identical lambda grid and native controller for learned and five random bodies; report resource-incomparable cells as inconclusive",
        "strict_start": "minute 0: q=0, TTL=10 for every arm; ordinary causal online schedule from minute 1",
        "primary_contrasts": ["component vs selected simple baseline", "gate vs selected simple baseline", "learned no-prior vs random no-prior"],
        "accounting": "disjoint init, execution, idle intervals; main horizon [0,14400); drain after disabling proactive creation",
        "ttl_order": "expire under old deadlines, update current TTL prospectively, prewarm, arrivals; MRU tie lowest free slot",
        "cost": "idle GBs + 15*rho*cold; predictor execution costs excluded",
    }
    dest = OUT / "protocol.json"
    if dest.exists() and read_json(dest) != protocol:
        raise RuntimeError("Refusing to change frozen protocol")
    write_json(dest, protocol)
    p19, ph = ROOT / "data/processed_2019", ROOT / "data/processed_huawei"
    c19 = np.load(p19 / "counts.npy", mmap_mode="r")
    first, day1 = scan(c19)
    primary = capped(candidates(first, day1, 1440, c19.shape[1]))
    apps = pd.read_csv(p19 / "active_functions.csv").app_id.to_numpy()
    union = set()
    for gap in (1,3,7):
        for threshold in (10,30,100):
            ids=np.flatnonzero((first>=gap*1440)&(first<c19.shape[1]-1440)&(day1>=threshold))
            union.update(map(int,capped(ids)))
    hold=[int(f) for f in candidates(first,day1,2880,c19.shape[1]) if f not in union]
    mainapps=set(apps[primary])
    clean=[f for f in hold if apps[f] not in mainapps]
    def fold(f):
        return int(hashlib.sha256(("winter-attribution-v1|"+apps[f]).encode()).hexdigest(),16)%5
    calibration=np.array([f for f in clean if fold(f)<2])
    evaluation=np.array([f for f in clean if fold(f)>=2])
    assert len(union)==772 and len(hold)==1755 and len(clean)==1595
    assert not set(apps[calibration]) & set(apps[evaluation])
    export("azure_primary",p19,primary,first,apps)
    export("azure_calibration",p19,calibration,first,apps)
    export("azure_evaluation",p19,evaluation,first,apps)
    ch=np.load(ph/"counts.npy",mmap_mode="r")
    hf,hd=scan(ch)
    hi=capped(candidates(hf,hd,1440,ch.shape[1]))
    happs=np.array(["huawei_function_"+str(i) for i in range(len(ch))])
    export("huawei",ph,hi,hf,happs)
    inputs=[HERE/"prepare.py",ROOT/"scripts/phase6_des.py",ROOT/"src/sim/des.py",
            ROOT/"scripts/eval_adapt_biased.py",ROOT/"src/data/features.py",
            ROOT/"scripts/revision_v1_hybridfull.py"]
    for p in (p19,ph,ROOT/"data/processed"):
        for filename in ("counts.npy","features.npy","func_ids.npy","splits.npz","duration_stats.csv"):
            inputs.append(p/filename)
    inputs.extend((ROOT/"results_azure2021/runs").glob("best_anil_ridge_s2_s*.pt"))
    inputs.extend(ROOT/"results/runs"/f for f in ["revision_a4_crosstrace.json","revision_h1_huawei_cohort.json","revision_r10_cohort_controls.json"])
    manifest=read_json(OUT/"manifest.json") if (OUT/"manifest.json").exists() else {"inputs":{},"python":sys.version,"platform":platform.platform()}
    for path in inputs:
        key=str(path.relative_to(ROOT))
        if key not in manifest["inputs"]:
            print("hash",key,flush=True)
            manifest["inputs"][key]={"sha256":digest(path),"bytes":path.stat().st_size}
            write_json(OUT/"manifest.json",manifest)
    manifest["protocol_sha256"]=digest(dest)
    manifest["cohorts"]={p.stem:digest(p) for p in (OUT/"cohorts").glob("*.npz")}
    write_json(OUT/"manifest.json",manifest)


if __name__=="__main__":
    main()
