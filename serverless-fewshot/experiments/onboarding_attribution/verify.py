"""End-to-end artifact and input-conservation checks; never alters originals."""
import argparse
import importlib.metadata

import numpy as np

from common import HERE,OUT,ROOT,digest,load_cohort,read_json,require_tests,write_json
from prune_screen import can_improve
from policies import simple_configs


def main(full_hash=True):
    require_tests()
    assert read_json(OUT/"policy_tests.json")["passed"]
    assert read_json(OUT/"analysis_tests.json")["passed"]
    manifest=read_json(OUT/"manifest.json")
    assert digest(OUT/"protocol.json")==manifest["protocol_sha256"]
    changed=[]
    if full_hash:
        for name,expected in manifest["inputs"].items():
            if digest(ROOT/name)!=expected["sha256"]:
                changed.append(name)
    assert not changed,changed
    for cohort,expected in manifest["cohorts"].items():
        assert digest(OUT/"cohorts"/(cohort+".npz"))==expected,cohort
    for model in ("trained0","trained1","trained2"):
        with np.load(OUT/"models"/model/"prototypes.npz") as proto:
            assert proto["centroids"].shape==(16,64)
            assert proto["prior"].shape==(16,64,19)
    for seed in range(5):
        directory=OUT/"models"/f"random{seed}"
        assert digest(directory/"body.pt")==read_json(directory/"model.json")["body_sha256"]
    checked=0
    for cohort in ("azure_primary","azure_evaluation","huawei"):
        z=load_cohort(cohort)
        execution=np.empty((len(z["counts"]),20))
        for f in range(len(z["counts"])):
            for si,seed in enumerate(range(1000,1020)):
                with np.load(OUT/"events"/cohort/f"f{f:04d}_s{seed}.npz") as tape:
                    execution[f,si]=.25*tape["duration"].sum()
        for rho in (1,10,100):
            tags=[f"main_rho{rho}",f"strict_rho{rho}",f"model1_rho{rho}",f"model2_rho{rho}"]
            reps=read_json(OUT/"representation_selection.json")[str(float(rho))]
            for model,choices in reps.items():
                indices={c["lambda_index"] for c in choices if c["lambda_index"] is not None}
                if indices:
                    tag=f"repr_{model}_rho{rho}"
                    tags.append(tag)
                    for i in indices:
                        assert (OUT/"evaluation"/cohort/tag/f"lambda{i}.npz").exists()
            for tag in tags:
                directory=OUT/"evaluation"/cohort/tag
                for name in (() if tag.startswith("repr") else ("component","gate","no_prior","prototype_only")):
                    assert (directory/(name+".npz")).exists(),(cohort,tag,name)
                if tag.startswith(("main","strict")):
                    assert (directory/"ewma.npz").exists()
                for p in directory.glob("*.npz"):
                    arrays=np.load(p)
                    m=arrays["metrics"]
                    assert m.shape==(len(z["counts"]),20,5,8),(p,m.shape)
                    assert np.isfinite(m).all() and (m>=-1e-8).all(),p
                    assert arrays["q"].shape==arrays["ttl"].shape==z["counts"].shape
                    assert ((arrays["q"]>=0)&(arrays["q"]<=200)).all()
                    assert np.isfinite(arrays["ttl"]).all() and (arrays["ttl"]>=0).all()
                    np.testing.assert_array_equal(m[:,:,:4,0].sum(2),np.repeat(z["counts"].sum(1)[:,None],20,axis=1))
                    assert (m[:,:,:,1]<=m[:,:,:,0]).all()
                    np.testing.assert_allclose(m[:,:,:,4].sum(2),execution,rtol=1e-9,atol=1e-5,err_msg=str(p))
                    np.testing.assert_array_equal(m[:,:,:,1],m[:,:,:,6])
                    assert (m[:,:,:,7]<=m[:,:,:,1]).all()
                    assert m[:,:,4,:2].sum()==0
                    np.testing.assert_array_equal(arrays["seeds"],np.arange(1000,1020))
                    if tag.startswith("strict"):
                        assert not arrays["q"][:,0].any()
                        assert (arrays["ttl"][:,0]==10.).all()
                    checked+=1
    configs=read_json(OUT/"candidate_configs.json")
    assert configs==simple_configs()
    selection=read_json(OUT/"selection.json")
    scores=np.load(OUT/"calibration_scores.npz")["metrics"]
    for ri,rho in enumerate((1,10,100)):
        proof=read_json(OUT/f"screen_proofs_rho{rho}.json")
        ids={p["config"] for p in proof["proofs"]}
        assert len(ids)==len(proof["proofs"])
        assert proof["fully_evaluated"]+len(ids)==len(configs)==proof["total_configs"]
        for ci,c in enumerate(configs):
            assert bool(np.isnan(scores[ri,ci]).all())==(c["id"] in ids)
        best=np.array([np.inf if v is None else v for v in proof["best_cold"]])
        for p in proof["proofs"]:
            assert not can_improve(p["cold_lower_bound"],p["wm_lower_bound"],np.array(proof["budgets"]),best)
        totals=scores[ri,:,:4].sum(1)
        for arm,row in selection[str(float(rho))].items():
            native=np.load(OUT/"evaluation/azure_calibration"/f"native_rho{rho}"/(arm+".npz"))["metrics"]
            reference=float(native[:,:,:4,2].sum(2).mean(1).sum())
            assert reference==row["wm_reference"]
            for choice in row["choices"]:
                assert choice["wm_limit"]==reference*choice["multiplier"]
                eligible=[i for i,m in enumerate(totals) if m[2]<=choice["wm_limit"]]
                expected=min(eligible,key=lambda i:(round(5*totals[i,1]),totals[i,2],configs[i]["id"])) if eligible else None
                assert choice["config"]==(configs[expected] if expected is not None else None)
                if expected is not None:
                    np.testing.assert_array_equal(choice["metrics"],totals[expected])
        for model,choices in read_json(OUT/"representation_selection.json")[str(float(rho))].items():
            totals={}
            for i in range(9):
                metrics=np.load(OUT/"evaluation/azure_calibration"/f"repr_{model}_rho{rho}"/f"lambda{i}.npz")["metrics"]
                totals[i]=metrics[:,:,:4].sum(2).mean(1).sum(0)
            for choice in choices:
                budget=selection[str(float(rho))]["component"]["wm_reference"]*choice["multiplier"]
                eligible=[i for i,m in totals.items() if m[2]<=budget]
                expected=min(eligible,key=lambda i:(round(5*totals[i][1]),totals[i][2],f"lambda{i}")) if eligible else None
                assert choice["lambda_index"]==expected
    for name in ("bridge.json","representation_selection.json","comparisons.json","endpoints.csv","time_and_interaction.csv","RESULTS.md"):
        assert (OUT/name).exists(),name
    for p in (OUT/"events").glob("*/*.npz"):
        with np.load(p) as a:
            assert int(a["ptr"][-1])==len(a["arrival"])==len(a["duration"])==len(a["cold"])
            assert (np.diff(a["arrival"])>=0).all()
    artifacts={str(p.relative_to(OUT)):digest(p)
               for folder in ("cohorts","models","evaluation","screen_complete")
               for p in (OUT/folder).rglob("*") if p.is_file()}
    write_json(OUT/"artifact_hashes.json",artifacts)
    write_json(OUT/"verification.json",{"passed":True,"evaluation_policy_files":checked,
               "original_inputs_rehashed":full_hash,"original_input_changes":changed,
               "source_code_sha256":{p.name:digest(p) for p in HERE.glob("*.py")},
               "package_versions":{name:importlib.metadata.version(name) for name in ("numpy","scipy","torch","numba","matplotlib")},
               "artifact_hashes_sha256":digest(OUT/"artifact_hashes.json"),
               "application_mapping_sha256":digest(ROOT/"data/processed_2019/active_functions.csv"),
               "selection_sha256":digest(OUT/"selection.json"),
               "comparisons_sha256":digest(OUT/"comparisons.json")})
    print("verified",checked,"evaluation policy files; original inputs unchanged",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--skip-input-hashes",action="store_true")
    args=ap.parse_args(); main(not args.skip_input_hashes)
