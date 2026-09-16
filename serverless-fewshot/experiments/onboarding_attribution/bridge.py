"""Bridge archived campaigns to common events and prospective TTL accounting."""
import sys

import numpy as np

from common import OUT,ROOT,load_cohort,read_json,require_tests,save_npz,write_json
from policies import native_baselines
from study import get_tape,model_actions,priors
from simulator import reference,simulate

sys.path.insert(0,str(ROOT))
from src.sim.des import simulate_function as original


def main():
    require_tests()
    fleet,profile=priors()
    report={"scope":"Three bridge seeds only; separate from final evaluation seeds 1000..1019", "cohorts":{}}
    for cohort,filename in (("azure_primary","revision_a4_crosstrace.json"),("huawei","revision_h1_huawei_cohort.json")):
        z=load_cohort(cohort)
        archive=read_json(ROOT/"results/runs"/filename)
        age=read_json(ROOT/"results/runs/revision_r10_cohort_controls.json")["cohorts"]["azure2019" if cohort=="azure_primary" else "huawei"]
        rows={}
        for rho in (10.,1.,100.):
            policies=model_actions(cohort,rho)
            policies.update(native_baselines(z["counts"],rho,fleet,profile))
            summary={}
            for name in ("ewma","component","gate","ageka60","fleetprior"):
                q,ttl=policies[name]
                old=np.zeros((len(q),3,3))
                legacy=np.zeros((len(q),3,5,8))
                corrected=np.zeros_like(legacy)
                for f in range(len(q)):
                    for seed in range(3):
                        r=original(z["counts"][f],q[f],ttl[f],float(z["duration_mean"][f]),float(z["duration_std"][f]),
                                   seed=seed*7919+f,cold_mu=.25,cold_sigma=.31)
                        old[f,seed]=[r["cold_starts"],r["idle_mem_gb_s"],r["busy_mem_gb_s"]]
                        tape=get_tape(cohort,z,f,seed)
                        legacy[f,seed]=simulate(*tape,q[f].astype(np.int64),ttl[f].astype(float),legacy_ttl=True)
                        corrected[f,seed]=simulate(*tape,q[f].astype(np.int64),ttl[f].astype(float))
                key={"ewma":"B4a_ewma","component":"A5_proto","gate":"gated_v3"}.get(name)
                ar=archive["by_rho"][str(rho)][key] if key else age["by_rho"][str(rho)]["AgeKA60" if name=="ageka60" else "EWMA_fleetprior"]
                inv=float(z["counts"].sum())
                native=old.mean(axis=1).sum(axis=0)
                le=legacy[:,:,:4].sum(axis=2).mean(axis=1).sum(axis=0)
                co=corrected[:,:,:4].sum(axis=2).mean(axis=1).sum(axis=0)
                summary[name]={"archived_csr_pct":ar["overall_csr"]*100,
                    "archived_wm":ar["wm_total"],
                    "original_des_replay_csr_pct":native[0]/inv*100,
                    "original_des_replay_wm":native[1],
                    "common_events_legacy_ttl_csr_pct":le[1]/inv*100,"common_events_legacy_ttl_wm":le[2],
                    "corrected_csr_pct":co[1]/inv*100,"corrected_wm":co[2],
                    "archive_cold_residual":native[0]-ar["overall_csr"]*inv,
                    "archive_wm_relative_residual":native[1]/ar["wm_total"]-1}
                save_npz(OUT/"bridge"/f"{cohort}_{name}_rho{rho:g}.npz",original=old,legacy=legacy,corrected=corrected,q=q,ttl=ttl)
            rows[str(rho)]=summary
            report["cohorts"][cohort]=rows
            write_json(OUT/"bridge.json",report)
            print("bridge",cohort,rho,summary,flush=True)
        # Full phase ledgers for a deterministic, prespecified small fixture set.
        for f in (0,int(np.argmax(z["counts"].sum(axis=1)))):
            for name,(q,ttl) in policies.items():
                if name not in ("ewma","component","gate","ageka60"):
                    continue
                tape=get_tape(cohort,z,f,0)
                oracle,ledger=reference(tape,q[f],ttl[f])
                fast=simulate(*tape,q[f].astype(np.int64),ttl[f].astype(float))
                np.testing.assert_allclose(fast,oracle,atol=1e-6,rtol=1e-11)
                write_json(OUT/"ledgers"/f"{cohort}_f{f}_{name}_rho100_seed0.json",
                           {"columns":["phase","start_sec","end_sec","close_reason"],"intervals":ledger,
                            "metrics":oracle.tolist(),"cohort_function_index":f})


if __name__=="__main__":
    main()
