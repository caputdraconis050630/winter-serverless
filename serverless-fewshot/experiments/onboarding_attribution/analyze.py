"""Effect sizes, application-cluster bootstrap, and publication-ready figures."""
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from common import METRICS,OUT,load_cohort,read_json,write_json

INTERVALS={"minute0":[0],"minute1_15":[1],"minute15_60":[2],"minute60_240":[3],"first60":[0,1,2],"first240":[0,1,2,3]}


def holm(values):
    p=np.asarray(values)
    order=np.argsort(p,kind="stable")
    adj=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order])
    result=np.empty(len(p)); result[order]=np.minimum(1.,adj)
    return result


def csv_write(path,rows):
    if not rows:
        return
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,"w",newline="") as stream:
        writer=csv.DictWriter(stream,columns)
        writer.writeheader(); writer.writerows(rows)


class PairedBootstrap:
    def __init__(self,apps,invocations):
        self.groups,self.idx=np.unique(apps,return_inverse=True)
        self.g=len(self.groups)
        self.n=np.bincount(self.idx).astype(float)
        self.inv=self.group_sum(invocations)
        self.weights=np.random.default_rng(260907).multinomial(self.g,np.full(self.g,1/self.g),10000).astype(float)
        self.bn=self.weights@self.n
        self.bi=self.weights@self.inv

    def group_sum(self,value):
        return np.bincount(self.idx,weights=value,minlength=self.g)

    def contrast(self,a,b):
        # a and b are [function, DES seed, metric], on identical event tapes.
        av=a.mean(axis=1); bv=b.mean(axis=1)
        d=self.group_sum(av[:,1]-bv[:,1])
        point=d.sum()/self.n.sum()
        boot_d=self.weights@d
        boot_mean=boot_d/self.bn
        ci=np.quantile(boot_mean,[.025,.975])
        csr=100*boot_d/self.bi
        wm_a=self.group_sum(av[:,2]); wm_b=self.group_sum(bv[:,2])
        allocated_a=self.group_sum(av[:,2:5].sum(1)); allocated_b=self.group_sum(bv[:,2:5].sum(1))
        wm_ratio=np.quantile((self.weights@wm_a)/(self.weights@wm_b),[.025,.975])
        allocated_ratio=np.quantile((self.weights@allocated_a)/(self.weights@allocated_b),[.025,.975])
        residual=d-point*self.n
        se=np.sqrt(self.g/(self.g-1)*np.sum(residual**2))/self.n.sum()
        if se>1e-14:
            v=self.weights@(d*d)-2*boot_mean*(self.weights@(d*self.n))+boot_mean**2*(self.weights@(self.n*self.n))
            bse=np.sqrt(np.maximum(0.,self.g/(self.g-1)*v))/self.bn
            tboot=(boot_mean-point)/np.maximum(bse,1e-14)
            p=(1+np.count_nonzero(np.abs(tboot)>=abs(point/se)))/10001
        else:
            p=1. if abs(point)<1e-12 else 1/10001
        seed_diff=(a[:,:,1]-b[:,:,1]).sum(axis=0)/len(a)
        largest=int(np.argmax(self.inv))
        leave=(d.sum()-d[largest])/(self.n.sum()-self.n[largest])
        return {"cold_per_fn_difference":float(point),"cold_ci_low":float(ci[0]),"cold_ci_high":float(ci[1]),
                "csr_difference_pp":float(100*d.sum()/self.inv.sum()),
                "csr_ci_low_pp":float(np.quantile(csr,.025)),"csr_ci_high_pp":float(np.quantile(csr,.975)),
                "wm_ratio":float(wm_a.sum()/wm_b.sum()),"wm_ratio_ci_low":float(wm_ratio[0]),"wm_ratio_ci_high":float(wm_ratio[1]),
                "allocated_ratio":float(allocated_a.sum()/allocated_b.sum()),
                "allocated_ratio_ci_low":float(allocated_ratio[0]),"allocated_ratio_ci_high":float(allocated_ratio[1]),
                "approximately_matched_wm":bool(wm_ratio[0]>=.99 and wm_ratio[1]<=1.01),
                "bootstrap_t_p":float(p),"fixed_cohort_mc_se_cold_per_fn":float(seed_diff.std(ddof=1)/np.sqrt(len(seed_diff))),
                "leave_largest_invocation_group_out_cold_per_fn":float(leave)}


def summaries():
    rows=[]
    for cohort in ("azure_primary","azure_evaluation","huawei"):
        z=load_cohort(cohort)
        for directory in sorted((OUT/"evaluation"/cohort).iterdir()):
            rho=float(re.search(r"rho([\d.]+)$",directory.name)[1])
            for path in sorted(directory.glob("*.npz")):
                data=np.load(path)
                metrics=data["metrics"]
                for interval,bins in INTERVALS.items():
                    total=metrics[:,:,bins,:].sum(axis=2).mean(axis=1).sum(axis=0)
                    inv=total[0]
                    if inv==0:
                        continue
                    rows.append({"cohort":cohort,"tag":directory.name,"rho":rho,"arm":path.stem,"interval":interval,
                                 "n_functions":len(z["counts"]),"n_groups":len(set(z["apps"])),
                                 "invocations":float(inv),"cold":float(total[1]),"csr_pct":100*total[1]/inv,
                                 "wm_per_1k":1000*total[2]/inv,"init_per_1k":1000*total[3]/inv,
                                 "execution_per_1k":1000*total[4]/inv,"allocated_per_1k":1000*total[2:5].sum()/inv,
                                 "modeled_cost_per_1k":1000*(total[2]+15*rho*total[1])/inv,
                                 "drain_idle_per_1k_4h_requests":1000*metrics[:,:,4,2].mean(axis=1).sum()/z["counts"].sum()})
    csv_write(OUT/"endpoints.csv",rows)
    return rows


def load_metric(cohort,tag,arm):
    return np.load(OUT/"evaluation"/cohort/tag/(arm+".npz"))["metrics"]


def window_contrast(bootstrap,a,b):
    result=bootstrap.contrast(a[:,:,:4].sum(2),b[:,:,:4].sum(2))
    av=a.sum(2).mean(1); bv=b.sum(2).mean(1)
    for name,columns in (("wm",[2]),("allocated",[2,3,4])):
        ga=bootstrap.group_sum(av[:,columns].sum(1))
        gb=bootstrap.group_sum(bv[:,columns].sum(1))
        lo,hi=np.quantile((bootstrap.weights@ga)/(bootstrap.weights@gb),[.025,.975])
        prefix="drain_inclusive_"+name+"_ratio"
        result.update({prefix:float(ga.sum()/gb.sum()),prefix+"_ci_low":float(lo),prefix+"_ci_high":float(hi)})
    return result


def comparisons():
    selections=read_json(OUT/"selection.json")
    reps=read_json(OUT/"representation_selection.json")
    rows=[]
    for cohort in ("azure_primary","azure_evaluation","huawei"):
        z=load_cohort(cohort)
        bootstrap=PairedBootstrap(z["apps"],z["counts"].sum(1))
        for rho in (10.,1.,100.):
            for condition in ("main","strict"):
                tag=f"{condition}_rho{rho:g}"
                ewma=load_metric(cohort,tag,"ewma")
                for arm in ("component","gate"):
                    a=load_metric(cohort,tag,arm)
                    pairs=[("ewma",ewma,0.),("ageka60",load_metric(cohort,tag,"ageka60"),0.)]
                    for other in (["no_prior","prototype_only"] if arm=="component" else []):
                        pairs.append((other,load_metric(cohort,tag,other),0.))
                    for other in (arm+"_q_ewma_ttl",arm+"_ttl_ewma_q"):
                        pairs.append((other,load_metric(cohort,tag,other),0.))
                        result=window_contrast(bootstrap,load_metric(cohort,tag,other),ewma)
                        result.update(cohort=cohort,condition=condition,rho=rho,arm=other,comparator="ewma",
                                      budget_multiplier=0.,primary=False)
                        rows.append(result)
                    for choice in selections[str(rho)][arm]["choices"]:
                        if choice["config"]:
                            name=choice["config"]["id"]
                            pairs.append((name,load_metric(cohort,tag,name),choice["multiplier"]))
                    for name,b,multiplier in pairs:
                        result=window_contrast(bootstrap,a,b)
                        result.update(cohort=cohort,condition=condition,rho=rho,arm=arm,comparator=name,budget_multiplier=multiplier,
                                      primary=(rho==10. and condition=="main" and multiplier==1.))
                        rows.append(result)
            for seed in (1,2):
                for arm in ("component","gate"):
                    a=load_metric(cohort,f"model{seed}_rho{rho:g}",arm)
                    choice=next(c for c in selections[str(rho)][arm]["choices"] if c["multiplier"]==1.)
                    names=["ewma"]+([choice["config"]["id"]] if choice["config"] else [])
                    for name in names:
                        b=load_metric(cohort,f"main_rho{rho:g}",name)
                        result=window_contrast(bootstrap,a,b)
                        result.update(cohort=cohort,condition=f"model{seed}",rho=rho,arm=arm,comparator=name,
                                      budget_multiplier=0. if name=="ewma" else 1.,primary=False)
                        rows.append(result)
            # Prespecified average over five random-initialization policies.
            for multiplier in (1.,):
                tc=next(c for c in reps[str(rho)]["trained0"] if c["multiplier"]==multiplier)
                rc=[next(c for c in reps[str(rho)][f"random{s}"] if c["multiplier"]==multiplier) for s in range(5)]
                if tc["lambda_index"] is not None and all(c["lambda_index"] is not None for c in rc):
                    a=load_metric(cohort,f"repr_trained0_rho{rho:g}",f"lambda{tc['lambda_index']}")
                    b=np.mean([load_metric(cohort,f"repr_random{s}_rho{rho:g}",f"lambda{rc[s]['lambda_index']}") for s in range(5)],axis=0)
                    result=window_contrast(bootstrap,a,b)
                    result.update(cohort=cohort,condition="representation",rho=rho,arm="learned_no_prior_selected_lambda",
                                  comparator="mean_of_five_random_policies_selected_lambdas",budget_multiplier=1.,primary=(rho==10.))
                    rows.append(result)
    primary=[i for i,r in enumerate(rows) if r["primary"]]
    # Missing primary representation contrasts remain p=1 in the nine-test family.
    p=[rows[i]["bootstrap_t_p"] for i in primary]+[1.]*(9-len(primary))
    corrected=holm(p)
    for j,i in enumerate(primary):
        rows[i]["holm_primary_nine_p"]=float(corrected[j])
        rows[i]["primary_cold_rejects_zero_after_holm"]=bool(corrected[j]<.05)
    for r in rows:
        if r["cold_ci_high"]<0 and r["wm_ratio_ci_high"]<=1 and r["allocated_ratio_ci_high"]<=1:
            r["joint_direction"]="arm_better_cold_no_more_memory"
        elif r["cold_ci_low"]>0 and r["wm_ratio_ci_low"]>=1 and r["allocated_ratio_ci_low"]>=1:
            r["joint_direction"]="comparator_better_cold_no_more_memory"
        else:
            r["joint_direction"]="tradeoff_or_inconclusive"
        r["arm_joint_direction_survives_drain"]=bool(r["joint_direction"]=="arm_better_cold_no_more_memory"
               and r["drain_inclusive_wm_ratio_ci_high"]<=1 and r["drain_inclusive_allocated_ratio_ci_high"]<=1)
        r["comparator_joint_direction_survives_drain"]=bool(r["joint_direction"]=="comparator_better_cold_no_more_memory"
               and r["drain_inclusive_wm_ratio_ci_low"]>=1 and r["drain_inclusive_allocated_ratio_ci_low"]>=1)
    write_json(OUT/"comparisons.json",rows)
    csv_write(OUT/"comparisons.csv",rows)
    return rows


def time_and_interaction():
    rows=[]
    for cohort in ("azure_primary","azure_evaluation","huawei"):
        z=load_cohort(cohort)
        boot=PairedBootstrap(z["apps"],z["counts"].sum(1))
        for rho in (1.,10.,100.):
            tag=f"main_rho{rho:g}"
            e=load_metric(cohort,tag,"ewma")
            for arm in ("component","gate"):
                w=load_metric(cohort,tag,arm)
                q=load_metric(cohort,tag,arm+"_q_ewma_ttl")
                t=load_metric(cohort,tag,arm+"_ttl_ewma_q")
                for label,bins in INTERVALS.items():
                    for contrast,difference in (("arm_minus_ewma",w-e),("q_ttl_interaction",w-q-t+e)):
                        d=difference[:,:,bins,1].sum(2).mean(1)
                        samples=(boot.weights@boot.group_sum(d))/boot.bn
                        lo,hi=np.quantile(samples,[.025,.975])
                        wm=difference[:,:,bins,2].sum(2).mean(1)
                        rows.append({"cohort":cohort,"rho":rho,"arm":arm,"interval":label,"contrast":contrast,
                                     "cold_per_fn":float(d.mean()),"ci_low":float(lo),"ci_high":float(hi),
                                     "idle_gb_s_per_fn":float(wm.mean())})
    csv_write(OUT/"time_and_interaction.csv",rows)
    return rows


def figures(rows):
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    folder=OUT/"figures"; folder.mkdir(exist_ok=True)
    colors={"ewma":"#555555","component":"#2274A5","gate":"#C94532","ageka60":"#338568","b2f":"#9B6C22"}
    cohorts=["azure_primary","azure_evaluation","huawei"]
    titles={"azure_primary":"Azure 2019 (100 functions)","azure_evaluation":"Azure 2019 (970 functions)","huawei":"Huawei (76 functions)"}
    selected=read_json(OUT/"selection.json")
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout="constrained")
    for ax,cohort in zip(axes,cohorts):
        subset=[r for r in rows if r["cohort"]==cohort and r["tag"]=="main_rho10" and r["interval"]=="first240"]
        lookup={r["arm"]:r for r in subset}
        for arm,color in colors.items():
            r=lookup[arm]; ax.scatter(r["wm_per_1k"],r["csr_pct"],color=color,s=45,label=arm,zorder=3)
        ids={c["config"]["id"] for a in ("component","gate") for c in selected["10.0"][a]["choices"] if c["config"]}
        points=sorted((lookup[k]["wm_per_1k"],lookup[k]["csr_pct"]) for k in ids)
        ax.scatter(*np.array(points).T,facecolors="none",edgecolors="#333333",marker="D",label="validation-selected simple policies")
        ax.set(title=titles[cohort],xlabel="Idle WM (GB s / 1k requests)",ylabel="Cold-start ratio (%)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncols=3,fontsize=8,frameon=False)
    fig.savefig(folder/"01_resource_operating_points.pdf"); fig.savefig(folder/"01_resource_operating_points.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.3),layout="constrained")
    arms=["ewma","component_q_ewma_ttl","component_ttl_ewma_q","component"]
    labels=["EWMA q / TTL","WINTER q\nEWMA TTL","EWMA q\nWINTER TTL","WINTER q / TTL"]
    for ax,cohort in zip(axes,cohorts):
        lookup={r["arm"]:r for r in rows if r["cohort"]==cohort and r["tag"]=="main_rho10" and r["interval"]=="first240"}
        vals=[lookup[a]["csr_pct"] for a in arms]
        ax.bar(range(4),vals,color=["#666666","#268A75","#C99438","#2274A5"])
        ax.set_xticks(range(4),labels,fontsize=7)
        ax.set(title=titles[cohort],ylabel="Cold-start ratio (%)")
        for i,a in enumerate(arms):
            ax.text(i,vals[i],f"WM {lookup[a]['wm_per_1k']:.0f}",ha="center",va="bottom",fontsize=7)
        ax.set_ylim(0,max(vals)*1.25)
    fig.supxlabel("WM annotations: idle GB s per 1,000 requests",fontsize=8)
    fig.savefig(folder/"02_action_channel_swap.pdf"); fig.savefig(folder/"02_action_channel_swap.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,5.8),layout="constrained")
    for j,cohort in enumerate(cohorts):
        z=load_cohort(cohort)
        bootstrap=PairedBootstrap(z["apps"],z["counts"].sum(1))
        e=load_metric(cohort,"main_rho10","ewma").mean(axis=1)
        for arm in ("component","gate","ageka60"):
            a=load_metric(cohort,"main_rho10",arm).mean(axis=1)
            cold_fn=(e[:,:4,1]-a[:,:4,1]).cumsum(axis=1)
            wm_fn=(a[:,:4,2]-e[:,:4,2]).cumsum(axis=1)
            for row,values in enumerate((cold_fn,wm_fn)):
                point=values.mean(axis=0)
                samples=np.stack([(bootstrap.weights@bootstrap.group_sum(values[:,i]))/bootstrap.bn for i in range(4)],axis=1)
                low,high=np.quantile(samples,[.025,.975],axis=0)
                axes[row,j].plot([1,15,60,240],point,"o-",label=arm,color=colors[arm])
                axes[row,j].fill_between([1,15,60,240],low,high,color=colors[arm],alpha=.10)
        axes[0,j].set(title=titles[cohort],ylabel="Cumulative avoided cold / function")
        axes[1,j].set(xlabel="Minutes since first-arrival minute",ylabel="Cumulative extra idle GB s / function")
        for ax in axes[:,j]:
            ax.axhline(0,color="#999999",lw=.7); ax.set_xscale("log"); ax.set_xticks([1,15,60,240],[1,15,60,240]); ax.grid(alpha=.15)
    axes[0,-1].legend(fontsize=8)
    fig.savefig(folder/"03_time_and_resources.pdf"); fig.savefig(folder/"03_time_and_resources.png",dpi=180); plt.close(fig)


def report(endpoints,comparisons):
    lines=["# Onboarding Attribution Results","", "## Scope",
           "Frozen-source transfer versus conservative warming on previously examined trace cohorts. No meta-learning-versus-supervised superiority claim.",
           "Calibration: 625 Azure functions / 426 applications. Additional evaluation: 970 functions / 663 disjoint applications. Original Azure: 100 / 97. Huawei: 76 functions, application IDs unavailable.",
           "Main endpoints use common events, prospective TTL, and exact init/execution/idle interval accounting. Existing paper numbers are not silently replaced.","",
           "## Rho 10, First Four Hours",
           "| Cohort | Policy | CSR (%) | Idle GB s/1k | Allocated GB s/1k | Drain idle GB s/1k |",
           "|---|---|---:|---:|---:|---:|"]
    for r in endpoints:
        if r["tag"]=="main_rho10" and r["interval"]=="first240" and r["arm"] in ("ewma","component","gate","ageka60","b2f","fleetprior"):
            lines.append(f"| {r['cohort']} | {r['arm']} | {r['csr_pct']:.4f} | {r['wm_per_1k']:.1f} | {r['allocated_per_1k']:.1f} | {r['drain_idle_per_1k_4h_requests']:.1f} |")
    lines.extend(["","## Primary Contrasts","Differences and ratios are arm minus/divided by comparator. Budget 1.0 means selection at the validation budget, NOT guaranteed equality on evaluation data.",
                  "| Cohort | Arm / comparator | Cold/function difference [95% CI] | WM ratio [95% CI] | Holm p | Direction |",
                  "|---|---|---|---|---:|---|"])
    for r in comparisons:
        if r["primary"]:
            lines.append(f"| {r['cohort']} | {r['arm']} / {r['comparator']} | {r['cold_per_fn_difference']:.3f} [{r['cold_ci_low']:.3f}, {r['cold_ci_high']:.3f}] | {r['wm_ratio']:.3f} [{r['wm_ratio_ci_low']:.3f}, {r['wm_ratio_ci_high']:.3f}] | {r.get('holm_primary_nine_p',1):.4g} | {r['joint_direction']} |")
    lines.extend(["","## Accounting Bridge",
                  "These three-seed bridge rows use the SAME common request events for old versus corrected TTL. They are not the twenty-seed evaluation rows above. The correction expires old deadlines before changing TTL, and accounts for already elapsed idle time.",
                  "| Cohort | Policy | Old TTL CSR (%) | Corrected CSR (%) | Corrected / old idle WM |",
                  "|---|---|---:|---:|---:|"])
    bridge=read_json(OUT/"bridge.json")
    for cohort in ("azure_primary","huawei"):
        for arm in ("ewma","component","gate","ageka60"):
            r=bridge["cohorts"][cohort]["10.0"][arm]
            lines.append(f"| {cohort} | {arm} | {r['common_events_legacy_ttl_csr_pct']:.4f} | {r['corrected_csr_pct']:.4f} | {r['corrected_wm']/r['common_events_legacy_ttl_wm']:.4f} |")
    lines.extend(["","## Interpretation Rules",
                  "- A validation-selected budget is not an executed hard quota. No equality claim is made unless the evaluation WM ratio CI fits [0.99,1.01].",
                  "- A baseline without source-trained model weights that reproduces or improves the point estimate does not prove that transfer contains no information; uncertainty and resource differences remain explicit. Fleet-prior and age-profile baselines themselves use simple cross-function information.",
                  "- Action-channel swaps distinguish pool targets from TTL. They do not identify a unique percentage of the effect attributable to information.",
                  "- Representation controls share features, support alignment, cadence, and a symmetric lambda grid; only native controller operating points are searched for these controls. Missing resource overlap remains inconclusive.",
                  "- Random-policy averaging is over five independent initializations, not an ensemble of their rate predictions.",
                  "- Random-body controls preserve all input channels, including Azure application co-rate: they remove source-trained weights, not all cross-function information.",
                  "- Intervals are paired application-cluster percentile bootstrap CIs (Huawei function clusters). P values use centered studentized cluster bootstrap, Holm-adjusted over nine primary contrasts; absent contrasts count as p=1.",
                  "- Joint directions are descriptive marginal-CI checks, not multiplicity-adjusted superiority decisions. The primary Holm test and its rejection flag must be read separately; secondary intervals are not confirmatory tests.",
                  "- DES-seed Monte Carlo SE is separate from population-bootstrap uncertainty. Largest-invocation-group deletion diagnostics are in comparisons.csv.",
                  "- The joint-direction column describes the first four hours only. Separate drain-inclusive ratios and survival flags in comparisons.csv test whether its direction persists after lifecycle closure. Predictor inference/refit cost is excluded.",
                  "","## Files","endpoints.csv: every arm, time interval, model seed and start condition.",
                  "comparisons.csv/json: effect sizes, paired uncertainty, actual resource ratios, and frozen comparator IDs.",
                  "bridge.json: original archives, original-DES replay, common-event old TTL, corrected TTL/accounting.",
                  "figures/: resource operating points, action-channel swaps, and benefit/resource evolution."])
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    rows=summaries()
    contrasts=comparisons()
    time_and_interaction()
    figures(rows)
    report(rows,contrasts)
