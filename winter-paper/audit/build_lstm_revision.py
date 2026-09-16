"""Generate manuscript tables and figures from completed common-request runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np
import build_final_evidence
import build_winter_g_evidence
import build_calibration_figure

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT.parent / "serverless-fewshot/results/fixed_ewma_v1"
DEST = ROOT / "audit/lstm_comparison"
GEN = ROOT / "generated"
EXPECTED = ["initial_azure_primary","initial_huawei","initial_azure_evaluation","continuous_48h"]
EXPECTED += [f"steady_{p}_{s}" for p in ("azure2021","azure2019","huawei")
             for s in (["h_mixed","h_sparse","h_saturated"] if p == "huawei" else ["S1","S2","S3"])]
EXPECTED += [f"drift_{p}_{s}" for p in ("azure2021","azure2019","huawei") for s in ("natural","synthetic")]
NAMES = {"WINTER":r"\sys{}", "WINTER_G":r"\sysG{}", "LSTM_Fifer":"LSTM--Fifer", "LSTM_shared":"LSTM--shared",
         "EWMA_0.3":"EWMA","Hybrid":"Hybrid histogram", "Fourier":"Spectral",
         "Chronos":"Chronos-Bolt", "Keep_alive":"Keep-alive", "No_handoff":"No age hand-off", "WINTER_no_prior":"WINTER, no prior"}
PLOT_NAMES = {k:v.replace(r"\sys{}","WINTER").replace(r"\sysG{}","WINTER-G").replace("--","–") for k,v in NAMES.items()}
COLORS = {"WINTER":"#222222", "WINTER_G":"#0077BB", "LSTM_Fifer":"#AA3377", "LSTM_shared":"#EE7733",
          "EWMA_0.3":"#009988", "Hybrid":"#BBBBBB", "Fourier":"#4477AA", "Chronos":"#998800"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9,"axes.labelsize":8,
                     "xtick.labelsize":7,"ytick.labelsize":7,"pdf.fonttype":42})
OUTPUTS = {}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
def get(data,name,method,rho=10):return data[name]["by_action"][f"{method}__rho{rho:g}"]
def num(x,digits=3):return "--" if x is None else f"{x:.{digits}f}"
def lag(row):return "--" if row["adaptation_lag"] is None else str(row["adaptation_lag"])
def savefig(fig,name):
    path=ROOT/(name+".pdf")
    fig.savefig(path,bbox_inches="tight",pad_inches=.06)
    preview=DEST/"visual";preview.mkdir(exist_ok=True)
    fig.savefig(preview/(name+".png"),dpi=180,bbox_inches="tight")
    plt.close(fig);OUTPUTS[path.name]=sha(path)


def initial(data):
    cases=[("initial_azure_primary","Azure 2019 (100 functions)"),("initial_huawei","Huawei (76 functions)")]
    if not all(k in data for k,_ in cases):return
    methods=["EWMA_0.3","Hybrid","Fourier","LSTM_Fifer","LSTM_shared","Chronos","WINTER","WINTER_G"]
    rows=[]
    for case,title in cases:
        rows.append(r"\multicolumn{7}{@{}l}{\textit{"+title+r"}} \\")
        for m in methods:
            if f"{m}__rho10" not in data[case]["by_action"]:continue
            v=get(data,case,m);w=v["windows"]
            rows.append(" & ".join([NAMES[m],num(w["post_15"]["csr_pct"]),num(w["post_60"]["csr_pct"]),
                num(w["full"]["csr_pct"]),lag(v),f'{w["full"]["idle_per_1k"]:,.0f}',f'{w["full"]["cost_per_1k"]:,.0f}'])+r" \\")
        rows.append(r"\midrule")
    (GEN/"lstm_initial_rows.tex").write_text("\n".join(rows[:-1])+"\n")
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.8))
    for ax,(case,title) in zip(axes,cases):
        for m in methods:
            if f"{m}__rho10" not in data[case]["by_action"]:continue
            w=get(data,case,m)["windows"]["full"]
            ax.scatter(w["idle_per_1k"],w["csr_pct"],s=45 if m=="WINTER" else 32,
                       marker="*" if m=="WINTER" else ("s" if m=="WINTER_G" else "o"),c=COLORS[m],label=PLOT_NAMES[m],zorder=3)
        control=build_final_evidence.conservative(case.removeprefix("initial_"))
        ax.scatter(float(control["wm_per_1k"]),float(control["csr_pct"]),
                   s=36,marker="D",facecolors="none",edgecolors="#882255",
                   label="Conservative EWMA",zorder=4)
        ax.set(title=title,xlabel="Idle GB s / 1,000 invocations",ylabel="Four-hour CSR (%)")
        ax.grid(alpha=.2,lw=.5);ax.ticklabel_format(axis="x",style="sci",scilimits=(4,4))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,-.19),fontsize=7)
    fig.tight_layout(w_pad=2)
    savefig(fig,"fig15_initial_operating_points")
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.65))
    for ax,(case,title) in zip(axes,cases):
        with np.load(RUN/"cases"/case/"aggregate.npz") as z:
            names=list(z["action_names"]);curves=z["timeline"];total=z["total_curve"].sum()
        for m in ["EWMA_0.3","LSTM_Fifer","WINTER","WINTER_G","Chronos"]:
            key=f"{m}__rho10"
            if key in names:
                y=np.r_[0.,1000*np.cumsum(curves[names.index(key),:-1,0])/total]
                ax.plot(np.arange(241),y,color=COLORS[m],label=PLOT_NAMES[m],lw=1.3,ls="--" if m=="WINTER_G" else "-")
        ax.set(title=title,xlabel="Minutes since first observed arrival",ylabel="Cumulative cold starts / 1,000")
        ax.set_xlim(0,240);ax.grid(alpha=.2,lw=.5)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(.5,-.07),fontsize=7)
    fig.tight_layout(w_pad=2);savefig(fig,"fig16_cold_count_windows")
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.25))
    for ax,(case,title) in zip(axes,cases):
        deltas=[get(data,case,"WINTER",r)["windows"]["full"]["csr_pct"]-get(data,case,"WINTER_no_prior",r)["windows"]["full"]["csr_pct"] for r in (1,10,100)]
        ax.bar(np.arange(3),deltas,color="#4477AA",width=.55)
        ax.axhline(0,color="black",lw=.7);ax.set_xticks(np.arange(3),["1","10","100"])
        ax.set(title=title,xlabel=r"Cost ratio $\rho$",ylabel="Prior minus no-prior CSR (pp)");ax.grid(axis="y",alpha=.2)
    fig.tight_layout(w_pad=2);savefig(fig,"fig17_prior_increment")


def continuous(data):
    case="continuous_48h"
    if case not in data:return
    rows=[]
    for m in ["EWMA_0.3","Hybrid","Fourier","LSTM_Fifer","LSTM_shared","WINTER","No_handoff","WINTER_G"]:
        v=get(data,case,m)["windows"]["full"]
        rows.append(" & ".join([NAMES[m],num(v["csr_pct"],4),f'{v["idle_per_1k"]:,.0f}',f'{v["cost_per_1k"]:,.0f}'])+r" \\")
    (GEN/"lstm_holdout_rows.tex").write_text("\n".join(rows)+"\n")
    with np.load(RUN/"cases"/case/"aggregate.npz") as z:
        names=list(z["action_names"]);timeline=z["timeline"];total=z["total_curve"].sum()
    base=names.index("EWMA_0.3__rho10")
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.55))
    for m in ["LSTM_Fifer","LSTM_shared","WINTER","WINTER_G","No_handoff"]:
        k=names.index(m+"__rho10");color=COLORS.get(m,"#777777")
        for j,(axis,col,scale) in enumerate([(axes[0],0,1000/total),(axes[1],1,1000/total)]):
            axis.plot(np.arange(2881)/60,np.r_[0.,np.cumsum(timeline[k,:-1,col]-timeline[base,:-1,col])*scale],
                      label=PLOT_NAMES[m],color=color,lw=1.15,ls="--" if m=="WINTER_G" else "-")
    for ax in axes:ax.axhline(0,color="black",lw=.6);ax.axvline(12,color="#777777",ls=":",lw=.8);ax.set(xlabel="Hours since first observed arrival",xlim=(0,48));ax.grid(alpha=.2)
    axes[0].set_ylabel("Cumulative extra cold starts / 1,000")
    axes[1].set_ylabel("Cumulative extra idle GB s / 1,000")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(.5,-.08),fontsize=7)
    fig.tight_layout(w_pad=2);savefig(fig,"fig18_continuous_comparison")


def steady(data):
    methods=["EWMA_0.3","LSTM_Fifer","WINTER","WINTER_G"]
    for group,providers in [("primary",["azure2021"]),("provider",["azure2019","huawei"])]:
        rows=[]
        for provider in providers:
            for split in (["h_mixed","h_sparse","h_saturated"] if provider=="huawei" else ["S1","S2","S3"]):
                case=f"steady_{provider}_{split}"
                if case not in data:return
                label=("Huawei " + split.replace("h_","")) if provider=="huawei" else ("Azure 2019 " + split if provider=="azure2019" else split)
                for index,m in enumerate(methods):
                    v=get(data,case,m)["windows"]["full"]
                    rows.append(" & ".join([label if index==0 else "",NAMES[m],num(v["csr_pct"]),f'{v["idle_per_1k"]:,.0f}'])+r" \\")
                rows.append(r"\midrule")
        (GEN/f"lstm_steady_{group}_rows.tex").write_text("\n".join(rows[:-1])+"\n")


def active_plot(data):
    cases=[c for c in EXPECTED if c.startswith("steady_")]
    if not all(c in data for c in cases):return
    methods=["EWMA_0.3","Hybrid","Fourier","LSTM_Fifer","LSTM_shared","WINTER","WINTER_G","Chronos"]
    fig,axes=plt.subplots(3,3,figsize=(7.16,5.7))
    handles={}
    for ax,case in zip(axes.flat,cases):
        baseline=get(data,case,"EWMA_0.3")["windows"]["full"]["idle_per_1k"]
        for m in methods:
            if m+"__rho10" not in data[case]["by_action"]:continue
            v=get(data,case,m)["windows"]["full"]
            h=ax.scatter(v["idle_per_1k"]/baseline,v["csr_pct"],s=32 if m=="WINTER" else 21,c=COLORS[m],
                marker="*" if m=="WINTER" else ("s" if m=="WINTER_G" else "o"),label=PLOT_NAMES[m],zorder=3)
            handles[m]=h
        title=case.replace("steady_azure2021_","Azure 2021 · ").replace("steady_azure2019_","Azure 2019 · ").replace("steady_huawei_h_","Huawei · ")
        ax.set_title(title,fontsize=8);ax.set_xscale("log");ax.axvline(1,color="#999999",lw=.6,ls=":");ax.grid(alpha=.18,lw=.5)
        ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        lo,hi=ax.get_xlim()
        if hi/lo < 2:
            ticks=MaxNLocator(nbins=3).tick_values(lo,hi)
            ticks=[x for x in ticks if lo <= x <= hi]
            if lo <= 1 <= hi:ticks.append(1.)
            ax.xaxis.set_major_locator(FixedLocator(sorted(set(ticks))))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("CSR (%)",fontsize=7)
    for ax in axes[-1]:ax.set_xlabel("Idle memory / EWMA",fontsize=7)
    fig.legend([handles[m] for m in methods if m in handles],[PLOT_NAMES[m] for m in methods if m in handles],
               loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(.5,0),fontsize=6.5)
    fig.tight_layout(rect=(0,.1,1,1),w_pad=1.4,h_pad=1.7);savefig(fig,"fig20_active_lstm_comparison")


def supplementary(data):
    rows=[]
    for case in ["initial_azure_primary","initial_huawei","initial_azure_evaluation"]:
        if case not in data:continue
        label={"initial_azure_primary":"Azure, 100","initial_huawei":"Huawei, 76","initial_azure_evaluation":"Azure, 970"}[case]
        for rho in (1,10,100):
            winter,lstm=get(data,case,"WINTER",rho),get(data,case,"LSTM_Fifer",rho)
            w,l=winter["windows"]["full"],lstm["windows"]["full"]
            interval=data[case]["paired_bootstrap"]["contrasts"][f"WINTER_vs_LSTM_Fifer__rho{rho}"]["full"]["winter_minus_comparator_csr_pp_ci95"]
            rows.append(" & ".join([label,str(rho),num(w["csr_pct"]),num(l["csr_pct"]),num(w["idle_per_1k"]/l["idle_per_1k"],2),
                f'[{interval[0]:+.3f}, {interval[1]:+.3f}]',lag(winter),lag(lstm)])+r" \\")
    (GEN/"lstm_initial_grid_rows.tex").write_text("\n".join(rows)+"\n")
    rows=[]
    for case in EXPECTED:
        if not case.startswith("steady_") or case not in data:continue
        label=case.replace("steady_azure2021_","A21 ").replace("steady_azure2019_","A19 ").replace("steady_huawei_h_","Huawei ")
        for rho in (.1,1,10,100):
            v={m:get(data,case,m,rho)["windows"]["full"] for m in ["EWMA_0.3","LSTM_Fifer","WINTER","WINTER_G"]}
            ratios=[v[m]["idle_per_1k"]/v["EWMA_0.3"]["idle_per_1k"] for m in ["LSTM_Fifer","WINTER","WINTER_G"]]
            rows.append(" & ".join([label,f'{rho:g}']+[num(v[m]["csr_pct"]) for m in v]+[num(r,2) for r in ratios])+r" \\")
    (GEN/"lstm_steady_grid_rows.tex").write_text("\n".join(rows)+"\n")
    rows=[];records=[]
    for case in EXPECTED:
        if not case.startswith("initial_") or case not in data:continue
        with np.load(RUN/"cases"/case/"aggregate.npz") as z:
            names=list(z["action_names"]);curves=z["timeline"];counts=z["total_curve"]
        offset=0
        references={}
        for m in ["WINTER","LSTM_Fifer"]:
            cold=curves[names.index(m+"__rho10"),offset:offset+240,0];total=counts[offset:offset+240]
            den=np.convolve(total,np.ones(15),mode="same")
            curve=np.divide(np.convolve(cold,np.ones(15),mode="same"),den,out=np.full_like(den,np.nan),where=den>0)
            references[m]=float(np.nanmean(curve[180:])*100)
            records.append(dict(case=case,method=m,rho=10,reference_csr_pct=references[m],threshold_csr_pct=1.1*references[m],adaptation_lag=get(data,case,m)["adaptation_lag"]))
        label=case.replace("initial_azure_primary","Initial A19, 100").replace("initial_azure_evaluation","Initial A19, 970").replace("initial_huawei","Initial Huawei, 76").replace("drift_azure2021_","A21 ").replace("drift_azure2019_","A19 ").replace("drift_huawei_","Huawei ")
        rows.append(" & ".join([label,num(references["WINTER"]),lag(get(data,case,"WINTER")),num(references["LSTM_Fifer"]),lag(get(data,case,"LSTM_Fifer"))])+r" \\")
    (GEN/"lstm_reference_rows.tex").write_text("\n".join(rows)+"\n")
    dump(DEST/"timing_references.json",records)


def validation():
    path=RUN/"benchmark.json"
    if path.exists():
        benchmark=json.loads(path.read_text());shutil.copy2(path,DEST/path.name)
        rows=[]
        for batch in (1,100,1000):
            for model,label in [("WINTER_body_scalar_read","WINTER body/read"),("LSTM_peak","LSTM peak")]:
                r=[next(x for x in benchmark["rows"] if x["batch"]==batch and x["model"]==model and x["device"]==d) for d in ("cpu","cuda")]
                rows.append(" & ".join([str(batch),label]+[num(x["median_us_per_function"],2) for x in r])+r" \\")
        (GEN/"lstm_overhead_rows.tex").write_text("\n".join(rows)+"\n")
    path=RUN/"strict_round/live/results.json"
    if path.exists():
        live=json.loads(path.read_text());shutil.copy2(path,DEST/"live_results.json")
        rows=[]
        for m,label in [("reactive","Reactive"),("keepalive10","Keep-alive"),("ewma","EWMA"),("protowarm","Learned schedule"),("lstm_fifer","LSTM--Fifer schedule")]:
            v=live["arms"][m]
            rows.append(" & ".join([label,num(v["csr_event"]*100),num(v["csr_latency"]*100),f'{v["pod_seconds"]:,.0f}',str(v["failed_requests"])])+r" \\")
        (GEN/"lstm_live_rows.tex").write_text("\n".join(rows)+"\n")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--partial",action="store_true");args=ap.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    data={}
    for name in EXPECTED:
        p=RUN/"cases"/name/"summary.json"
        if p.exists():
            data[name]=json.loads(p.read_text());shutil.copy2(p,DEST/(name+".json"))
    missing=[n for n in EXPECTED if n not in data]
    if missing and not args.partial:raise SystemExit("Awaiting completed cases: "+", ".join(missing))
    initial(data);continuous(data);steady(data);active_plot(data);supplementary(data);validation()
    build_calibration_figure.generate()
    build_final_evidence.generate()
    gate_evidence = build_winter_g_evidence.generate()
    columns=["case","method","rho","window","csr_pct","idle_per_1k","cost_per_1k","invocations","cold","idle","init","execution","adaptation_lag"]
    with (DEST/"all_results.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=columns,lineterminator="\n");writer.writeheader()
        for name,d in data.items():
            for row in d["by_action"].values():
                for window,v in row["windows"].items():
                    record=dict(case=name,method=row["method"],rho=row["rho"],window=window,adaptation_lag=row["adaptation_lag"])
                    record.update({k:v[k] for k in columns if k in v});writer.writerow(record)
    # Matched shift comparisons have a separate companion with paired intervals.
    with (DEST/"all_results.csv").open(newline="") as file:
        current_rows=[row for row in csv.DictReader(file) if not row["case"].startswith("drift_")]
    with (DEST/"current_results.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=columns,lineterminator="\n")
        writer.writeheader();writer.writerows(current_rows)
    manifests={}
    for name in data:
        d=RUN/"cases"/name
        manifests[name]={p:sha(d/p) for p in ["case.json","execution.json","aggregate.npz","summary.json"]}
    verification=RUN/"verification/report.json"
    status="partial"
    if verification.exists():
        shutil.copy2(verification,DEST/"verification.json")
        status=json.loads(verification.read_text())["status"]
    dump(DEST/"manifest.json",dict(expected_cases=EXPECTED,completed_cases=list(data),missing_cases=missing,
        ewma_alpha=.3,affected_replay_cases=19,
        run_root=str(RUN),source_manifests=manifests,figures=OUTPUTS,build_script_sha256=sha(__file__),
        verification_sha256=sha(verification) if verification.exists() else None,
        measurement="common request streams, prospective TTL, disjoint wall-clock phase accounting",
        measurement_status=status,
        additional_gate_sources={"manifest":str(build_winter_g_evidence.MANIFEST.relative_to(ROOT)),
            "sha256":sha(build_winter_g_evidence.MANIFEST),"additional_replay_cases":6},
        matched_winter_sources={"manifest":str(build_winter_g_evidence.MANIFEST.relative_to(ROOT)),
            "sha256":sha(build_winter_g_evidence.MANIFEST),"matched_winter_replay_cases":6,
            "validation":gate_evidence["matched_winter_validation"]},
        additional_control_sources={"manifest":"audit/final_revision_2026-09-16/evidence_manifest.json",
            "sha256":sha(ROOT/"audit/final_revision_2026-09-16/evidence_manifest.json")}))
    if not missing:
        outputs={"fig14_information_protocol.pdf":sha(ROOT/"fig14_information_protocol.pdf"),**OUTPUTS,**gate_evidence["figures"]}
        dump(ROOT/"audit/revision_figure_sources.json",dict(
            measurement_status=status,source="audit/lstm_comparison/manifest.json",
            source_sha256=sha(DEST/"manifest.json"),outputs=outputs,
            additional_replay_cases=6,
            additional_gate_manifest=str(build_winter_g_evidence.MANIFEST.relative_to(ROOT)),
            additional_gate_manifest_sha256=sha(build_winter_g_evidence.MANIFEST),
            matched_winter_replay_cases=6,
            matched_winter_validation=gate_evidence["matched_winter_validation"],
            new_replay_cases=len(data),verification="audit/lstm_comparison/verification.json",
            verification_sha256=sha(verification) if verification.exists() else None,
            static_diagram="fig14_information_protocol.pdf",
            calibration_figure="fig11_testbed_latency_cdf.pdf retains the measured initialization-latency calibration; historical result figures are excluded from both documents."))
    print("generated from",len(data),"completed cases; remaining",len(missing))


if __name__=="__main__":main()
