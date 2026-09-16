"""Draw revision figures from retained summaries, without rerunning policies."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "audit/onboarding_metrics.json"
DATA = json.loads(SOURCE.read_text())
COLORS = {"B4a_ewma": "#CC3311", "EWMA_selected": "#008877",
          "A5_proto": "#222222", "gated_v3": "#0077BB"}
NAMES = {"B4a_ewma": "EWMA (0.1)", "EWMA_selected": "Selected EWMA (0.3)",
         "A5_proto": "Learned (C=16)", "gated_v3": "WINTER-G"}
RHOS = (1., 10., 100.)
COHORTS = (("azure2019", "Azure 2019 (100 functions)"),
           ("huawei", "Huawei (76 functions)"))
OUTPUTS = {}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(ROOT / (name + ".pdf"), bbox_inches="tight", pad_inches=.07)
    preview = ROOT / "audit/visual/revision-figures"
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    OUTPUTS[name + ".pdf"] = hashlib.sha256((ROOT / (name + ".pdf")).read_bytes()).hexdigest()


def information():
    fig, ax = plt.subplots(figsize=(7.16, 3.6))
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")

    def box(x, y, w, h, title, detail):
        ax.add_patch(Rectangle((x, y), w, h, ec="#555555", fc="white", lw=.7))
        ax.text(x+w/2, y+h-5, title, ha="center", va="center", weight="bold", fontsize=8)
        ax.text(x+w/2, y+h/2-3, detail, ha="center", va="center", fontsize=7)

    def arrow(start, end):
        ax.annotate("", end, start, arrowprops={"arrowstyle": "->", "lw": .8})

    ax.text(0, 97, "(a) Information and output selection", weight="bold")
    box(0, 66, 22, 24, "Source training", "Shared body\nand prototype centers")
    box(0, 34, 22, 24, "Known before t", "Past features\nAge and count")
    box(30, 66, 27, 24, "Prototype / adapter", "Onboarding or\nsteady-window procedure")
    box(30, 34, 27, 24, "EWMA", "Past log-count state")
    box(65, 49, 15, 25, "Router", "Count and\nobserved age")
    box(86, 49, 14, 25, "Controller", "Target and\nidle lifetime")
    arrow((22, 78), (30, 78))
    arrow((22, 50), (30, 69))
    arrow((22, 43), (30, 43))
    arrow((57, 77), (65, 67))
    arrow((57, 46), (65, 55))
    arrow((80, 61), (86, 61))
    ax.text(30, 27, "Both branches remain maintained after output selection.", fontsize=7)
    ax.text(0, 18, "(b) Update targets", weight="bold")
    ax.text(0, 10, r"Onboarding: past $\phi_t$ is paired with $c_t$ after minute $t$ ends.", fontsize=7.5)
    ax.text(0, 2, r"Steady window: $\phi_t$ is paired with already observed $c_{t-1}$.", fontsize=7.5)
    save(fig, "fig14_information_protocol")


def operating_points():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    for ax, (cohort, title) in zip(axes, COHORTS):
        for arm, color in COLORS.items():
            for rho, marker in zip(RHOS, ("o", "s", "^")):
                row = DATA[cohort]["by_rho"][str(rho)][arm]
                ax.scatter(row["wm_per_1k"], row["csr_pct"], color=color, marker=marker, s=26)
        ax.set(title=title, xlabel="Archived idle GB s / 1,000 invocations", ylabel="CSR (%)")
        ax.grid(alpha=.2, lw=.5)
    handles = [Line2D([], [], color=c, marker="o", ls="", label=NAMES[a]) for a, c in COLORS.items()]
    handles += [Line2D([], [], color="#666666", marker=m, ls="", label=f"rho={r:g}")
                for r, m in zip(RHOS, ("o", "s", "^"))]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .82))
    save(fig, "fig15_initial_operating_points")


def windows():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65))
    x = np.arange(3)
    for ax, (cohort, title) in zip(axes, COHORTS):
        for arm, offset in (("A5_proto", -.1), ("gated_v3", .1)):
            rows = [DATA[cohort]["by_rho"][str(rho)][arm] for rho in RHOS]
            first = [r["avoided_first_hour_per_fn"] for r in rows]
            total = [r["avoided_four_hours_per_fn"] for r in rows]
            ax.vlines(x+offset, first, total, color=COLORS[arm], lw=.8)
            ax.scatter(x+offset, first, facecolors="white", edgecolors=COLORS[arm], s=26)
            ax.scatter(x+offset, total, color=COLORS[arm], marker="s", s=22)
        ax.axhline(0, color="#777777", lw=.6)
        ax.set(title=title, xticks=x, xticklabels=["1", "10", "100"], xlabel="Cost ratio rho",
               ylabel="Archived avoided cold starts / function")
        ax.grid(axis="y", alpha=.2)
    handles = [Line2D([], [], color=COLORS[a], lw=1, label=NAMES[a]) for a in ("A5_proto", "gated_v3")]
    handles += [Line2D([], [], color="#555555", marker="o", mfc="white", ls="", label="First hour"),
                Line2D([], [], color="#555555", marker="s", ls="", label="Four hours")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .88))
    save(fig, "fig16_cold_count_windows")


def prior():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45))
    for ax, (cohort, title) in zip(axes, COHORTS):
        changes = [DATA[cohort]["by_rho"][str(r)]["A5_proto"]["csr_pct"]
                   - DATA[cohort]["by_rho"][str(r)]["A5_noproto"]["csr_pct"] for r in RHOS]
        ax.scatter(np.arange(3), changes, color="#0077BB", s=27)
        ax.axhline(0, color="#777777", lw=.8)
        ax.set(title=title, xlabel="Cost ratio rho", ylabel="Prior minus no-prior CSR (pp)",
               xticks=np.arange(3), xticklabels=["1", "10", "100"], xlim=(-.4, 2.4))
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save(fig, "fig17_prior_increment")


def main():
    information()
    operating_points()
    prior()
    windows()
    report = {"measurement_status": "archived_unvalidated", "source": str(SOURCE.relative_to(ROOT)),
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "outputs": OUTPUTS,
              "new_policy_replays": 0, "new_bootstrap_intervals": 0}
    (ROOT / "audit/revision_figure_sources.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
