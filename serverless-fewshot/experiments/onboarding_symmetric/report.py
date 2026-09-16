"""Paired inference and figures from frozen selections, with resource caveats."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from experiment import COHORTS, FAMILIES, MULTIPLIERS, OUT, RHOS, load_cohort, read_json, write_json
from analyze import PairedBootstrap, csv_write, holm, window_contrast


def load_metrics(cohort, rho, condition):
    directory = OUT / "evaluation" / cohort / f"{condition}_rho{rho:g}"
    meta = read_json(directory / "complete.json")
    result = {name: [] for name in meta["names"]}
    for f in range(meta["n"]):
        with np.load(directory / f"f{f:04d}.npz") as z:
            metrics = z["metrics"]
            for i, name in enumerate(z["names"]):
                result[str(name)].append(metrics[i])
    return {name: np.stack(values) for name, values in result.items()}


def endpoint(cohort, rho, condition, name, m, bins, interval):
    a = m[:, :, bins].sum(2).mean(1).sum(0)
    inv = a[0]
    full_inv = m[:, :, :4, 0].sum(2).mean(1).sum()
    drain = m[:, :, 4, 2].mean(1).sum()
    return dict(cohort=cohort, rho=rho, condition=condition, arm=name, interval=interval,
                functions=len(m), invocations=float(inv), cold=float(a[1]), csr_pct=float(100*a[1]/inv),
                wm_per1k=float(1000*a[2]/inv), allocated_per1k=float(1000*a[2:5].sum()/inv),
                modeled_cost_per1k=float(1000*(a[2]+15*rho*a[1])/inv),
                drain_idle_per1k_4h=float(1000*drain/full_inv),
                cost_plus_drain_per1k=float(1000*(a[2]+15*rho*a[1]+drain)/inv) if interval == "first240" else None)


def cost_contrast(bootstrap, a, b, rho):
    result = {}
    for drain in (False, True):
        cold = (a[:, :, :4, 1]-b[:, :, :4, 1]).sum(2).mean(1)
        idle = (a[:, :, :5 if drain else 4, 2]-b[:, :, :5 if drain else 4, 2]).sum(2).mean(1)
        d = bootstrap.group_sum(idle+15*rho*cold)
        values = 1000*(bootstrap.weights@d)/bootstrap.bi
        lo, hi = np.quantile(values, [.025, .975])
        label = "drain_cost_delta_per1k" if drain else "cost_delta_per1k"
        result.update({label: float(1000*d.sum()/bootstrap.inv.sum()), label+"_ci_low": float(lo), label+"_ci_high": float(hi)})
    return result


def summarize():
    endpoints, contrasts, times = [], [], []
    for cohort in COHORTS[1:]:
        z = load_cohort(cohort)
        bs = PairedBootstrap(z["apps"], z["counts"].sum(1))
        for rho in RHOS:
            for condition in ("main", "strict"):
                mm = load_metrics(cohort, rho, condition)
                # Means are over policy outcomes, not an ensemble of forecasts.
                for suffix in ["native"] + [f"b{i}" for i in range(6)]:
                    names = [f"random{s}__{suffix}" for s in range(5)]
                    if all(name in mm for name in names):
                        mm[f"random_mean__{suffix}"] = np.mean([mm[name] for name in names], axis=0)
                names = [f"random{s}__trained_lambda_wrapper" for s in range(5)]
                if all(name in mm for name in names):
                    mm["random_mean__trained_lambda_wrapper"] = np.mean([mm[name] for name in names], axis=0)
                for name, m in mm.items():
                    for interval, bins in (("first15", [0, 1]), ("first60", [0, 1, 2]), ("first240", [0, 1, 2, 3])):
                        endpoints.append(endpoint(cohort, rho, condition, name, m, bins, interval))
                pairs = []
                for bi in range(6):
                    for family in ("component", "gate"):
                        pairs.append((f"{family}__b{bi}", f"strong_simple__b{bi}", bi, "system"))
                for a, b in (("component__native", "ewma"), ("gate__native", "ewma"),
                             ("component__b2", "no_prior__b2"), ("component__b2", "pooled_prior__b2"),
                             ("trained_lambda__b2", "random_mean__b2"),
                             ("component__b2", "no_prior__component_wrapper"),
                             ("component__b2", "pooled_prior__component_wrapper"),
                             ("trained_lambda__b2", "random_mean__trained_lambda_wrapper"),
                             ("component__native", "no_prior__native"), ("component__native", "pooled_prior__native")):
                    pairs.append((a, b, 2 if a.endswith("b2") else -1, "mechanism"))
                for a, b, bi, kind in pairs:
                    if a not in mm or b not in mm:
                        continue
                    row = window_contrast(bs, mm[a], mm[b])
                    row.update(cost_contrast(bs, mm[a], mm[b], rho))
                    primary = rho == 10 and a == "component__b2" and b == "strong_simple__b2"
                    secondary = rho == 10 and condition == "main" and (a, b) in (
                        ("component__b2", "no_prior__b2"), ("component__b2", "pooled_prior__b2"),
                        ("trained_lambda__b2", "random_mean__b2"))
                    row.update(cohort=cohort, rho=rho, condition=condition, arm=a, comparator=b,
                               budget_index=bi, primary=primary, mechanistic_family=secondary)
                    contrasts.append(row)
                if rho == 10:
                    for a, b in (("component__b2", "strong_simple__b2"), ("component__b2", "ewma")):
                        if a not in mm or b not in mm:
                            continue
                        for hi, label in ((1, "minute1"), (2, "minute15"), (3, "minute60"), (4, "minute240")):
                            diff = (mm[a][:, :, :hi, 1]-mm[b][:, :, :hi, 1]).sum(2).mean(1)
                            draws = (bs.weights@bs.group_sum(diff))/bs.bn
                            ci = np.quantile(draws, [.025, .975])
                            times.append(dict(cohort=cohort, condition=condition, arm=a, comparator=b, until=label,
                                              cold_per_fn_difference=float(diff.mean()), ci_low=float(ci[0]), ci_high=float(ci[1])))
                print("analysis", cohort, rho, condition, flush=True)
    for field, size, output in (("primary", 6, "holm6_p"), ("mechanistic_family", 9, "holm9_p")):
        rows = [x for x in contrasts if x[field]]
        if len(rows) > size:
            raise AssertionError("Hypothesis family expanded")
        adjusted = holm([x["bootstrap_t_p"] for x in rows] + [1.]*(size-len(rows)))
        for row, p in zip(rows, adjusted):
            row[output] = float(p)
    csv_write(OUT / "endpoints.csv", endpoints)
    csv_write(OUT / "contrasts.csv", contrasts)
    csv_write(OUT / "time_effects.csv", times)
    write_json(OUT / "contrasts.json", contrasts)
    return endpoints, contrasts


def figures(rows, contrasts):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    titles = dict(azure_primary="Azure 2019: 100 functions", azure_evaluation="Azure 2019: 970 functions", huawei="Huawei: 76 functions")
    colors = dict(component="#166f96", strong_simple="#c5433d", gate="#377b51", pooled_prior="#8563a0")
    labels = dict(component="Calibrated component", strong_simple="Strong simple policy", gate="Calibrated WINTER-G", pooled_prior="Calibrated pooled prior")
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    fig.suptitle("rho = 10 | settings selected on calibration only", fontsize=11)
    for ci, cohort in enumerate(COHORTS[1:]):
        for ri, condition in enumerate(("main", "strict")):
            ax = axes[ri, ci]
            rr = [r for r in rows if r["cohort"] == cohort and r["rho"] == 10 and r["condition"] == condition and r["interval"] == "first240"]
            for family in colors:
                pts = [r for r in rr if r["arm"].startswith(family+"__b")]
                pts.sort(key=lambda r: r["arm"])
                ax.plot([p["wm_per1k"] for p in pts], [p["csr_pct"] for p in pts], "o-", ms=4,
                        color=colors[family], label=labels[family], alpha=.85)
                primary = next((p for p in pts if p["arm"] == family+"__b2"), None)
                if primary:
                    ax.scatter([primary["wm_per1k"]], [primary["csr_pct"]], s=95, marker="s", facecolors="none", edgecolors=colors[family])
            ax.set_title(titles[cohort]+(" / original start" if ri == 0 else " / strict start"), fontsize=10)
            ax.set_xlabel("Idle memory (GB s / 1,000 requests)")
            ax.set_ylabel("Cold-start ratio (%)")
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.grid(alpha=.18)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", bbox_to_anchor=(.5, .028), ncol=4, frameon=False)
    fig.text(.5, .009, "Squares: calibration budget 1. Lines join operating points, not Pareto frontiers; evaluation memory is not matched.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .96))
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"01_operating_points.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    fig.suptitle("Calibrated component vs strong simple | rho = 10, calibration budget 1 | marginal 95% CIs", fontsize=10)
    primary = [r for r in contrasts if r["primary"]]
    for i, row in enumerate(primary):
        axes[0].errorbar(row["cold_per_fn_difference"], i,
                         xerr=[[row["cold_per_fn_difference"]-row["cold_ci_low"]], [row["cold_ci_high"]-row["cold_per_fn_difference"]]], fmt="o", color="#166f96")
        axes[1].errorbar(row["wm_ratio"], i,
                         xerr=[[row["wm_ratio"]-row["wm_ratio_ci_low"]], [row["wm_ratio_ci_high"]-row["wm_ratio"]]], fmt="o", color="#c5433d")
    labels_ = [titles[r["cohort"]].replace(": ", " / ")+" / "+r["condition"] for r in primary]
    axes[0].set_yticks(range(len(primary)), labels_)
    axes[1].set_yticks(range(len(primary)), [""]*len(primary))
    axes[0].axvline(0, ls="--", color="#777777", lw=1)
    axes[1].axvline(1, ls="--", color="#777777", lw=1)
    axes[0].set_xlabel("WINTER - strong simple: cold starts / function")
    axes[1].set_xlabel("WINTER / strong simple: idle memory ratio")
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"02_primary_effects.{ext}", dpi=180)
    plt.close(fig)


def markdown(rows, contrasts):
    lines = ["# Controller-symmetric follow-up: frozen-policy results", "",
             "This is a post-v1 follow-up, not a new blind holdout. New controller choices used calibration only; checkpoints and representation lambdas were frozen from v1.", "",
             "## Primary comparisons", "",
             "rho=10; common calibration budget equals native component WM. Difference is WINTER minus strong simple.", "",
             "| Cohort | Start | WINTER CSR % | Simple CSR % | Cold/fn difference [95% CI] | WM ratio [95% CI] | Holm6 p |",
             "|---|---|---:|---:|---|---|---:|"]
    index = {(r["cohort"], r["condition"], r["rho"], r["arm"], r["interval"]): r for r in rows}
    for r in (r for r in contrasts if r["primary"]):
        a = index[(r["cohort"], r["condition"], 10., r["arm"], "first240")]
        b = index[(r["cohort"], r["condition"], 10., r["comparator"], "first240")]
        lines.append(f"| {r['cohort']} | {r['condition']} | {a['csr_pct']:.5f} | {b['csr_pct']:.5f} | {r['cold_per_fn_difference']:+.4f} [{r['cold_ci_low']:.4f}, {r['cold_ci_high']:.4f}] | {r['wm_ratio']:.4f} [{r['wm_ratio_ci_low']:.4f}, {r['wm_ratio_ci_high']:.4f}] | {r['holm6_p']:.5g} |")
    lines += ["", "## Cost and terminal sensitivity", "", "Model cost = idle GB s + 15*rho*cold; excludes predictor costs. Positive delta favors simple.", "",
              "| Cohort | Start | Cost delta /1k [CI] | Cost + drain delta /1k [CI] |", "|---|---|---|---|"]
    for r in (r for r in contrasts if r["primary"]):
        lines.append(f"| {r['cohort']} | {r['condition']} | {r['cost_delta_per1k']:+.1f} [{r['cost_delta_per1k_ci_low']:.1f}, {r['cost_delta_per1k_ci_high']:.1f}] | {r['drain_cost_delta_per1k']:+.1f} [{r['drain_cost_delta_per1k_ci_low']:.1f}, {r['drain_cost_delta_per1k_ci_high']:.1f}] |")
    lines += ["", "## Mechanistic contrasts", "", "Main start, rho10, budget1. Independently calibrated controllers: not a fixed-action causal decomposition.", "",
              "| Cohort | Arm | Comparator | Cold/fn delta [CI] | WM ratio | Holm9 p |", "|---|---|---|---|---:|---:|"]
    for r in (r for r in contrasts if r["mechanistic_family"]):
        lines.append(f"| {r['cohort']} | {r['arm']} | {r['comparator']} | {r['cold_per_fn_difference']:+.4f} [{r['cold_ci_low']:.4f}, {r['cold_ci_high']:.4f}] | {r['wm_ratio']:.4f} | {r['holm9_p']:.5g} |")
    lines += ["", "## Reading the evidence", "",
              "- Shared validation budgets are not actual matched evaluation memory. See ratio CIs.",
              "- Marginal percentile CIs and centered studentized bootstrap tests are not inverse procedures.",
              "- No strict superiority or equivalence follows from a nonsignificant contrast.",
              "- Pooled prior uses the same 76 source functions / 1520 support rows as C16; no target labels.",
              "- Random mean is the average of five policy outcomes, not a forecasting ensemble.",
              "- The primary family has six tests, mechanisms a separate nine. Other contrasts are descriptive.",
              "- Operating-point lines join fixed calibration-budget settings; they are not guaranteed Pareto frontiers.",
              "- Squares mark calibration budget1, not equal evaluation WM. Primary-effect bars show marginal 95% CIs.",
              "- Drain is the artificial no-new-request terminal scenario, not measured future production utility.",
              "- The 4h replay does not evaluate the 720-minute handoff. Mature/drift economics remain unaudited here.",
              "- See protocol, selection/proofs, endpoints.csv, contrasts.csv, and verification.json for provenance.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    endpoints, contrasts = summarize()
    figures(endpoints, contrasts)
    markdown(endpoints, contrasts)
