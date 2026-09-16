"""Regenerate current LSTM-comparison figures and tables from measured results.

The explicit --archived-full option retains the earlier figure workflow.
"""
import json
import os
import sys
from pathlib import Path

# Current figures require completed common-request comparison results.
if __name__ == "__main__" and "--archived-full" not in sys.argv:
    from audit.build_lstm_revision import main
    main()
    raise SystemExit(0)

EXPERIMENT = Path(os.environ.get("WINTER_EXPERIMENT_ROOT", Path(__file__).resolve().parent.parent / "serverless-fewshot"))
if __name__ == "__main__" and not EXPERIMENT.is_dir():
    raise SystemExit("Full archived figure regeneration requires WINTER_EXPERIMENT_ROOT; no figures were overwritten.")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from audit import figure_legacy as legacy
plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]

ROOT = Path(__file__).resolve().parent
RUNS = EXPERIMENT / "results/runs"
BLUE, BLACK, RED, TEAL, ORANGE = "#0077BB", "#222222", "#CC3311", "#008877", "#B26B00"
STYLES = {
    "WINTER-G": dict(color=BLUE, marker="s", linestyle="-"),
    "WINTER component": dict(color=BLACK, marker="o", linestyle="--"),
    "EWMA (0.1)": dict(color=RED, marker="^", linestyle="-."),
    "EWMA (0.3)": dict(color=TEAL, marker="d", linestyle=":"),
    "B2f histogram": dict(color="#8135A7", marker="v", linestyle=":"),
    "Keep-alive": dict(color=TEAL, marker="D", linestyle="--"),
    "Spectral": dict(color=ORANGE, marker="<", linestyle="-."),
    "Oracle": dict(color="#888888", marker="x", linestyle=":"),
}


def read(name):
    return json.loads((RUNS / name).read_text())


def save(fig, name):
    (ROOT / "thumbnails").mkdir(exist_ok=True)
    fig.savefig(ROOT / (name + ".pdf"), bbox_inches="tight", pad_inches=.04)
    fig.savefig(ROOT / "thumbnails" / (name + ".png"), dpi=180,
                bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    print("wrote", name, flush=True)


def box(ax, xy, size, title, detail, color=BLACK):
    x, y = xy
    w, h = size
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color, linewidth=.9))
    ax.text(x + w / 2, y + h * .69, title, ha="center", va="center", fontsize=8, color=color, weight="bold")
    ax.text(x + w / 2, y + h * .28, detail, ha="center", va="center", fontsize=7)


def arrow(ax, start, end, **kw):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", lw=.8, color=BLACK, **kw))


def lifecycle():
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")
    ax.text(0, 9.65, "History state", fontsize=8, weight="bold")
    ax.text(6.6, 9.65, "Selected output", fontsize=8, weight="bold")
    rows = [(8.25, "No local arrivals", r"$N_t=0$", "Prototype", BLACK),
            (6.5, "Few arrivals", r"$0<N_t<100$", "EWMA", RED),
            (4.75, "Young, count-qualified", r"$N_t\geq100,\ A_t<720$", "Ridge", BLUE),
            (3., "Mature, count-qualified", r"$N_t\geq100,\ A_t\geq720$", "EWMA", RED)]
    for y, title, condition, output, color in rows:
        ax.text(0, y + .22, title, fontsize=7.5, va="center")
        ax.text(0, y - .42, condition, fontsize=7, va="center")
        arrow(ax, (5.7, y), (6.5, y))
        ax.text(6.7, y, output, fontsize=8, color=color, va="center")
        ax.plot([0, 9.8], [y - .85, y - .85], color="#dddddd", lw=.5)
    ax.text(0, 1.65, "Independent of history state", fontsize=7.5, weight="bold")
    ax.text(0, .75, "Sparse phase structure may favor spectral prediction.\nSaturation can leave little cold-start headroom.", fontsize=7, va="center", linespacing=1.6)
    save(fig, "fig01_regime_map_1col")


def architecture():
    fig, ax = plt.subplots(figsize=(7.16, 3.15))
    ax.set(xlim=(0, 16), ylim=(0, 8))
    ax.axis("off")
    box(ax, (.1, 5.9), (3.15, 1.45), "Offline training", "Source functions:\nbody and prototypes")
    box(ax, (.1, 3.45), (3.15, 1.45), "Causal features", "Past window only:\n" + r"minutes $[t-60,t)$")
    box(ax, (4.1, 3.45), (2.35, 1.45), "Shared TCN", r"$\phi_t\in\mathbb{R}^{64}$")
    box(ax, (7.25, 5.9), (3.05, 1.45), "Prototype / ridge", "Campaign-specific\nsupport and fit")
    box(ax, (7.25, 3.45), (3.05, 1.45), "EWMA state", "Past log counts", RED)
    box(ax, (11.1, 4.65), (1.9, 1.45), "Router", r"$N_t, A_t$", BLUE)
    box(ax, (13.8, 4.65), (2.05, 1.45), "Controller", "Prewarm\nand lifetime")
    box(ax, (4.1, .8), (6.2, 1.45), "Completed minute", "Update support, EWMA, count, and age")
    arrow(ax, (3.25, 4.17), (4.1, 4.17))
    arrow(ax, (3.25, 6.62), (7.25, 6.62))
    arrow(ax, (5.27, 5.9), (5.27, 4.9))
    ax.plot([5.27, 5.27], [5.9, 6.62], color=BLACK, lw=.8)
    arrow(ax, (6.45, 4.45), (7.25, 6.15))
    arrow(ax, (10.3, 6.5), (11.1, 5.75))
    arrow(ax, (10.3, 4.17), (11.1, 5.0))
    arrow(ax, (13., 5.38), (13.8, 5.38))
    arrow(ax, (14.8, 4.65), (10.3, 1.5), connectionstyle="angle,angleA=-90,angleB=0,rad=0")
    arrow(ax, (7.2, 2.25), (7.6, 3.45))
    arrow(ax, (10.3, 2.0), (11.65, 4.65), connectionstyle="angle,angleA=0,angleB=-90,rad=0")
    arrow(ax, (4.1, 1.5), (1.7, 3.45), connectionstyle="angle,angleA=180,angleB=-90,rad=0")
    ax.text(8., .15, "Routing to EWMA does not imply stopping TCN execution, refitting, or retained state.", fontsize=7, ha="center")
    save(fig, "fig02_architecture_v2")


def onboarding_effects():
    data = json.loads((ROOT / "audit/onboarding_metrics.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.8), sharex=True)
    x = np.arange(3)
    for j, (cohort, title) in enumerate((("azure2019", "Azure 2019 (100 functions)"), ("huawei", "Huawei (76 functions)"))):
        axes[0, j].set_title(title)
        for arm, label, offset in [("gated_v3", "WINTER-G", -.08), ("A5_proto", "WINTER component", .08)]:
            rows = [data[cohort]["by_rho"][str(r)][arm] for r in (1., 10., 100.)]
            y = np.array([r["delta_cold_per_fn"] for r in rows])
            bounds = np.array([r["delta_cold_ci95"] for r in rows]).T
            axes[0, j].errorbar(x + offset, y, yerr=np.stack([y - bounds[0], bounds[1] - y]),
                               capsize=3, markersize=4, linewidth=1, label=label, **STYLES[label])
            axes[1, j].plot(x + offset, [r["wm_ratio_ewma"] for r in rows], markersize=4, label=label, **STYLES[label])
        axes[0, j].axhline(0, color=RED, lw=.8, linestyle=":")
        axes[1, j].axhline(1, color=RED, lw=.8, linestyle=":")
        axes[1, j].set_xticks(x, ["1", "10", "100"])
        axes[1, j].set_xlabel(r"Cost ratio $\rho$")
        for ax in axes[:, j]:
            ax.grid(axis="y", alpha=.25, lw=.4)
            ax.set_xlim(-.35, 2.35)
    axes[0, 0].set_ylabel("Paired cold starts / function\n(difference from EWMA)")
    axes[1, 0].set_ylabel("Warm-memory ratio to EWMA")
    axes[1, 0].set_ylim(.9, 2.9)
    axes[1, 1].set_ylim(.9, 2.9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=1.3)
    save(fig, "fig06_onboarding_cohort_2019")


def timing():
    alpha = read("revision_r22_ewma_alpha_frontier.json")
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.6))
    for j, (cohort, filename, title) in enumerate([
        ("azure2019", "revision_a4_crosstrace.json", "Azure 2019"),
        ("huawei", "revision_h1_huawei_cohort.json", "Huawei")]):
        rows = read(filename)["by_rho"]["10.0"]
        curves = [("WINTER-G", rows["gated_v3"]), ("WINTER component", rows["A5_proto"]),
                  ("EWMA (0.1)", rows["B4a_ewma"]),
                  ("EWMA (0.3)", alpha["cohorts"][cohort]["by_rho"]["10.0"]["ewma_alpha"]["0.3"])]
        for i, limit in enumerate((60, 240)):
            ax = axes[i, j]
            for label, row in curves:
                ax.plot(100 * legacy.arr(row["rolling_csr"]), label=label, markersize=2.5,
                        markevery=12 if i == 0 else 40, lw=1.1, **STYLES[label])
            ax.set_xlim(0, limit)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.25, lw=.4)
            ax.set_title(title + (": first hour" if i == 0 else ": four hours"))
            ax.set_xlabel("Minutes since first observed arrival")
    for ax in axes[:, 0]:
        ax.set_ylabel("Rolling cohort CSR (%)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.035))
    fig.tight_layout(rect=(0, 0, 1, .96), h_pad=1.1)
    save(fig, "fig13_onboarding_timing")


def operating_curves():
    sources = [("revision_r16_faithful_primary_experiment.json", {"A5_faithful": "WINTER component"}),
               ("revision_r17_faithful_gates_2021.json", {"A5_gated_aq_faithful": "WINTER-G"}),
               ("sim_results_des_revision.json", {"B4a_ewma": "EWMA (0.1)", "B1_fixed_keepalive": "Keep-alive", "Oracle": "Oracle"}),
               ("revision_v1_hybridfull_2021.json", {"B2f_hybrid_full": "B2f histogram"}),
               ("revision_f1_fourier_steady.json", {"B3_fourier": "Spectral"})]
    points = {}
    for filename, mapping in sources:
        data = read(filename)
        for row in data if isinstance(data, list) else data["results"]:
            if row["method"] not in mapping or row["cost_ratio"] not in (.1, 1., 10., 100.):
                continue
            key = (row["split"], mapping[row["method"]], float(row["cost_ratio"]))
            points.setdefault(key, []).append(row)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
    for ax, split, n in zip(axes, ("S1", "S2", "S3"), (35, 86, 172)):
        for label in ("WINTER-G", "WINTER component", "EWMA (0.1)", "B2f histogram", "Keep-alive", "Spectral", "Oracle"):
            series = []
            for rho in (.1, 1., 10., 100.):
                rows = points[(split, label, rho)]
                assert len(rows) == 10 and len({r["seed"] for r in rows}) == 10, (split, label, rho)
                series.append((np.mean([r["wm_per_1k_inv"] for r in rows]), 100 * np.mean([r["csr"] for r in rows])))
            a = np.asarray(series)
            ax.plot(a[:, 0], a[:, 1], label=label, markersize=3, lw=1, **STYLES[label])
            ax.scatter(*a[2], s=35, marker="o" if label == "Oracle" else STYLES[label]["marker"], facecolors="white",
                       edgecolors=STYLES[label]["color"], linewidths=1.1, zorder=8)
        ax.set_title(f"{split} ({n} functions)")
        ax.set_xlabel("WM / 1k invocations (GB s)")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3), useMathText=True)
        ax.grid(alpha=.2, lw=.4)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("CSR (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.14), columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=1.)
    save(fig, "fig03_steady_pareto")


def rolling_supplement():
    data = read("revision_a4_crosstrace.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.1), sharey=True)
    arms = [("gated_v3", "WINTER-G"), ("A5_proto", "WINTER component"),
            ("B4a_ewma", "EWMA (0.1)"), ("B1_fixed_keepalive", "Keep-alive")]
    for ax, rho in zip(axes, ("1.0", "10.0", "100.0")):
        for arm, label in arms:
            ax.plot(100 * legacy.arr(data["by_rho"][rho][arm]["rolling_csr"]), label=label,
                    markersize=3, markevery=40, lw=1.1, **STYLES[label])
        ax.set(xlim=(0, 240), xlabel="Minutes since first observed arrival", title=rf"$\rho={float(rho):g}$")
        ax.grid(axis="y", alpha=.2, lw=.4)
    axes[0].set_ylabel("Rolling cohort CSR (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.1))
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, "fig07_rolling_adaptation_lag")


def synthetic_heads():
    from matplotlib.colors import LogNorm
    ob = legacy.load("onboarding_drift_results.json")["onboarding"]
    hm = legacy.load("ablation_results.json")["heatmap_k_head"]
    fig = plt.figure(figsize=(7.16, 4.25))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.], hspace=.6, wspace=.33)
    a, b, c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    arms = [("Oracle", "Oracle"), ("B1_fixed_keepalive", "Keep-alive"),
            ("B4a_ewma", "EWMA"), ("A5_noproto", "Without prior"), ("A5_proto", "WINTER component")]
    for arm, label in arms:
        style_label = "Learned component, no prototype" if arm == "A5_noproto" else label
        a.plot(100 * legacy.arr(ob["by_rho"]["1.0"][arm]["rolling_csr"]), label=label,
               **legacy.line_kw(style_label, markevery=40, ms=3))
    for arm, label in arms[-2:]:
        style_label = "Learned component, no prototype" if arm == "A5_noproto" else label
        b.plot(legacy.arr(ob["crps_vs_time"][arm]), label=label,
               **legacy.line_kw(style_label, markevery=12, ms=3))
    a.set(xlabel="Minutes since injection", ylabel="Rolling CSR (%)", title="(a) Synthetic injection: policy effect", xlim=(0, 240))
    b.set(xlabel="Minutes since injection", ylabel="CRPS", title="(b) First-hour prediction error", xlim=(0, 60))
    for ax in (a, b):
        ax.grid(axis="y", alpha=.2, lw=.4)
    values = np.asarray(hm["crps_matrix"], float)
    im = c.imshow(values.T, aspect="auto", cmap="Greys_r", origin="lower",
                  norm=LogNorm(vmin=max(values.min(), .001), vmax=values.max()))
    c.set_xticks(range(len(hm["k_values"])), hm["k_values"])
    c.set_yticks(range(3), ["Ridge\n(zero at K=0)", "ANIL-GD", "Bayes"])
    c.tick_params(axis="y", length=0, pad=8)
    c.set_xlabel("Adaptation shots K")
    c.set_title("(c) Head comparison: CRPS (different forecast representations)")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            c.text(i, j, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8,
                   color="white" if im.norm(values[i, j]) < .48 else BLACK)
    c.spines[:].set_visible(False)
    fig.legend(*a.get_legend_handles_labels(), ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.01))
    fig.subplots_adjust(top=.88, left=.13, right=.98, bottom=.09)
    save(fig, "fig09_onboarding_heads")


if __name__ == "__main__":
    legacy.RUNS = str(EXPERIMENT / "results_azure2021/runs")
    legacy.LIVE = str(EXPERIMENT / "results/runs")
    legacy.TABLES = str(EXPERIMENT / "results/tables")
    legacy.save = save
    lifecycle()
    architecture()
    onboarding_effects()
    timing()
    operating_curves()
    rolling_supplement()
    synthetic_heads()
    for generator in (legacy.fig_e2_forest, legacy.fig_testbed, legacy.fig_stratified,
                      legacy.fig_lag_fractile, legacy.fig_pareto_2019):
        generator()
