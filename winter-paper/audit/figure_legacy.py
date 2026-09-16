#!/usr/bin/env python3.13
"""Generate all measured figures for the WINTER FGCS paper.

Reads canonical Azure-2021 result JSONs from ../serverless-fewshot/results_azure2021/runs
and writes PDF (for LaTeX) + PNG (for visual inspection) into ./figures/.

Style: grayscale-only (print-safe). Identity is carried by a fixed
(gray level, linestyle) pair per method for lines and a fixed
(fill, hatch) pair for bars, consistent across all figures; one axis
per panel.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RUNS = "/data/260715/serverless-fewshot/results_azure2021/runs"
LIVE = "/data/260715/serverless-fewshot/results/runs"
TABLES = "/data/260715/serverless-fewshot/results/tables"
DERIVED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lag_tables.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.size": 7.5,
    # Arial for all figures; Liberation Sans is the metric-compatible fallback
    # used when Arial is not installed (renders as real Arial on Overleaf / any
    # host that has it).
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 300,
})

# Per-policy identity, following IEEE/Elsevier figure guidance: encode each
# series REDUNDANTLY on color + linestyle + marker, so identity survives both
# color-vision deficiency and a grayscale printout. Colors are a validated
# colorblind-safe set (scripts/validate_palette.js: CVD dE>=8, contrast>=3:1).
# Learned arms are black; Oracle (a bound) is muted light gray.
_BLACK, _BLUE, _RED, _TEAL = "#000000", "#0077BB", "#CC3311", "#009988"
_PURPLE, _GOLD, _MAGENTA = "#8135A7", "#B8860B", "#EE3377"
_GRAY, _GRAY2 = "#9e9e9e", "#777777"
_ORANGE = "#EE7733"   # spectral (B3); unused elsewhere in the palette
_SOLID = "-"
_DASH = (0, (4, 1.6))
_DASHDOT = (0, (4, 1.4, 1, 1.4))
_DOT = (0, (1, 1.3))
_SHORTDASH = (0, (2, 1.4))

# identity: label -> dict(color, ls, marker, lw, open=marker is open-faced)
IDENT = {
    "WINTER head":       dict(color=_BLACK,   ls=_SOLID,     marker="o", lw=1.9),
    "Section-4 always-on learned head": dict(color=_BLACK, ls=_SOLID, marker="o", lw=1.9),
    "WINTER component":  dict(color=_BLACK,   ls=_SOLID,     marker="o", lw=1.9),
    "Learned serving arm": dict(color=_BLACK, ls=_SOLID,     marker="o", lw=1.9),
    "Archived blended arm": dict(color=_BLACK, ls=_DOT,      marker="",  lw=1.2),
    "Archived serving variant": dict(color=_BLACK, ls=_DOT,  marker="",  lw=1.2),
    "Learned arm":       dict(color=_BLACK,   ls=_SOLID,     marker="o", lw=1.5),
    # Thinner than the other hero entries (1.5 vs 1.9): in the drift figure the
    # refit curve runs on top of the baselines, and the heavy stroke hid them.
    "Learned arm, refit": dict(color=_BLACK,  ls=_SOLID,     marker="o", lw=1.5),
    "WINTER-G":          dict(color=_BLUE,    ls=_SOLID,     marker="s", lw=1.3),
    "WINTER-G (720 min)": dict(color=_BLUE,   ls=_SOLID,     marker="s", lw=1.3),
    "WINTER-G v2":       dict(color=_BLUE,    ls=_SOLID,     marker="s", lw=1.3),
    "Prototype zero-shot":  dict(color=_BLACK,   ls=_SHORTDASH, marker="o", lw=1.2, open=True),
    "Learned component, no prototype": dict(color=_BLACK, ls=_SHORTDASH, marker="o", lw=1.2, open=True),
    "Learned arm, frozen": dict(color=_BLACK, ls=_DOT,       marker="",  lw=1.3),
    "EWMA":                 dict(color=_RED,     ls=_DASH,      marker="^", lw=1.3),
    "Keep-alive":           dict(color=_TEAL,    ls=_SOLID,     marker="D", lw=1.3),
    "Histogram":            dict(color=_GOLD,    ls=_DASHDOT,   marker="v", lw=1.3),
    "Global":               dict(color=_PURPLE,  ls=_DASHDOT,   marker="P", lw=1.3),
    "TSFM":                 dict(color=_MAGENTA, ls=_DASHDOT,   marker="h", lw=1.3),
    # One name per gate variant across every figure and both tables: the body
    # and Tables 1-2 call these "WINTER-G v3/v4", so the legends do too. v3
    # keeps the WINTER-G identity (blue, solid, square) it carries in the
    # Pareto figures; v4 stays purple/dotted, the only pair that must separate.
    "WINTER-G v4":          dict(color=_PURPLE,  ls=_DOT,       marker="*", lw=1.2),
    "WINTER-G v3":          dict(color=_BLUE,    ls=_SOLID,     marker="s", lw=1.3),
    "LSTM":                 dict(color=_PURPLE,  ls=_DASH,      marker="x", lw=1.3),
    "Oracle":               dict(color=_GRAY,    ls=_DOT,       marker="",  lw=1.1),
    # B3, the registered spectral arm (2026-08-03): its whole story is that it
    # sits on a different part of the CSR-memory plane, so it must be visible
    # in the frontier and cohort figures rather than described only in prose.
    "Spectral (B3)":        dict(color=_ORANGE,  ls=_DASHDOT,   marker="<", lw=1.3),
}


def _ident(label):
    """Resolve a display label (incl. parenthesised variants) to its identity."""
    if label in IDENT:
        return IDENT[label]
    base = label.split(" (")[0].split(" [")[0]
    return IDENT.get(base, dict(color="#333333", ls=_SOLID, marker="", lw=1.3))


# Convenience views used across the figure functions.
C = {k: v["color"] for k, v in IDENT.items()}
LS = {k: v["ls"] for k, v in IDENT.items()}
MARK = {k: v["marker"] for k, v in IDENT.items()}

# Bars: fill = policy color, plus a hatch as the grayscale/CVD-safe redundancy.
BAR = {
    "WINTER head":          (_BLACK,   None),
    "WINTER component":     (_BLACK,   None),
    "Learned serving arm":  (_BLACK,   None),
    "Archived blended arm": ("#ffffff", "\\\\\\"),
    "Archived serving variant": ("#ffffff", "\\\\\\"),
    "Learned arm":          (_BLACK,   None),
    "Learned arm, refit":   (_BLACK,   None),
    "Learned arm, frozen":  ("#ffffff", "///"),
    "Learned component, no prototype": ("#ffffff", "///"),
    "EWMA":                 (_RED,     None),
    "Keep-alive":           (_TEAL,    None),
    "Histogram":            (_GOLD,    "xxx"),
    "Histogram (B2)":       (_GOLD,    "xxx"),
    "Oracle":               (_GRAY,    None),
    "Spectral (B3)":        (_ORANGE,  "++"),
}
# Display labels carry the registry index wherever two policies share a word:
# B2 (percentile-only) is what the frontier figures plot, NOT B2f (the full
# production policy reported in Table 1 / main 10.1); and the gate curve is
# v3 specifically, not the legacy v2 documented only in the supplement.
METHOD_MAP = {
    "A5_full_system": "Archived serving variant",
    "A5_faithful": "Section-4 always-on learned head",
    "A5_gated": "WINTER-G v3",
    "A5_gated_v3": "WINTER-G v3",
    "A5_gated_v3_faithful": "WINTER-G v3",
    "A5_gated_v4_faithful": "WINTER-G v4",
    "A5_gated_aq_faithful": "WINTER-G (720 min)",
    "G_WE_A720": "WINTER-G (720 min)",
    "G_WE_A480": "WINTER-G",
    "A5_gated_v2": "WINTER-G v2 (legacy)",
    "A5_proto": "WINTER component",
    "A5_noproto": "Learned component, no prototype",
    "A5_refit": "Hybrid online refit",
    "A5_frozen": "Learned arm, frozen",
    "B4a_ewma": "EWMA",
    "B1_fixed_keepalive": "Keep-alive",
    "B2_histogram": "Histogram (B2)",
    "B5_global": "Global",
    "Oracle": "Oracle",
    "B3_fourier": "Spectral (B3)",
}


def load_live(name):
    with open(os.path.join(LIVE, name)) as f:
        return json.load(f)


def load_derived():
    with open(DERIVED) as f:
        return json.load(f)


def load(name):
    with open(os.path.join(RUNS, name)) as f:
        return json.load(f)


def load_chronos_native():
    return load_live("revision_e5d_chronos_native.json")


def chronos_native_cohort(rho):
    return load_chronos_native()["cohort"]["by_rho"][rho]["native_direct"]


def arr(x):
    """list -> float array, None -> NaN (gaps in rolling windows)."""
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def style_of(label):
    """(color, linestyle) for a display label (variants inherit base identity)."""
    d = _ident(label)
    return d["color"], d["ls"]


def line_kw(label, markevery=None, ms=5.0):
    """Full plot kwargs for a policy line: redundant color+ls+marker+lw+zorder.
    Learned-family curves draw on top; open-faced markers for learned variants
    so they read apart from the filled component marker."""
    d = _ident(label)
    kw = dict(color=d["color"], linestyle=d["ls"], linewidth=d["lw"],
              marker=d["marker"] or None, markersize=ms, markeredgewidth=0.6,
              zorder=5 if d["color"] == _BLACK else 3)
    if d.get("open"):
        kw.update(markerfacecolor="white", markeredgecolor=d["color"],
                  markeredgewidth=0.9)
    if markevery is not None:
        kw["markevery"] = markevery
    return kw


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# ================================================================ F1 Pareto
def fig_pareto():
    data = load("sim_results_des.json")
    splits = ["S1", "S2", "S3"]
    titles = {"S1": "S1 (35 functions)", "S2": "S2 (86 functions)", "S3": "S3 (172 functions)"}
    # seed-mean per (split, method, rho). The 2021 learned path now uses the
    # faithful Section-3 head and the final 720-minute age-qualified gate; EWMA,
    # keep-alive, and Oracle remain from the archived 10-seed campaign.
    # Histogram and Global stay at 5 seeds. Core rho grid is restricted to the
    # original {0.1..100} so curve extent is unchanged.
    RHO0 = {0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0}
    CORE = {"B4a_ewma", "Oracle", "B1_fixed_keepalive"}
    agg = {}
    for e in data:
        if e["method"] in ("B2_histogram", "B5_global"):
            key = (e["split"], e["method"], e["cost_ratio"])
            agg.setdefault(key, []).append((e["csr"], e["wm_per_1k_inv"]))
    for e in load_live("sim_results_des_revision.json")["results"]:
        if e["method"] in CORE and e["cost_ratio"] in RHO0:
            key = (e["split"], e["method"], e["cost_ratio"])
            agg.setdefault(key, []).append((e["csr"], e["wm_per_1k_inv"]))
    for e in load_live("revision_r16_faithful_primary_experiment.json")["results"]:
        if e["method"] == "A5_faithful" and e["cost_ratio"] in RHO0:
            key = (e["split"], e["method"], e["cost_ratio"])
            agg.setdefault(key, []).append((e["csr"], e["wm_per_1k_inv"]))
    for e in load_live("revision_r17_faithful_gates_2021.json")["results"]:
        if e["method"] == "A5_gated_aq_faithful" and e["cost_ratio"] in RHO0:
            key = (e["split"], e["method"], e["cost_ratio"])
            agg.setdefault(key, []).append((e["csr"], e["wm_per_1k_inv"]))
    for e in load_live("revision_f1_fourier_steady.json")["results"]:
        key = (e["split"], e["method"], e["cost_ratio"])
        agg.setdefault(key, []).append((e["csr"], e["wm_per_1k_inv"]))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.95), sharey=False)
    order = ["Section-4 always-on learned head", "WINTER-G (720 min)", "EWMA",
             "Spectral (B3)", "Keep-alive", "Histogram (B2)",
             "Global", "Oracle"]
    for ax, sp in zip(axes, splits):
        methods = sorted({m for (s, m, r) in agg if s == sp})
        series = {}
        for m in methods:
            pts = sorted(
                (np.mean([w for _, w in agg[(sp, m, r)]]),
                 100 * np.mean([c for c, _ in agg[(sp, m, r)]]))
                for r in sorted({r for (s, mm, r) in agg if s == sp and mm == m})
            )
            series[METHOD_MAP.get(m, m)] = pts
        for label in order:
            if label not in series:
                continue
            pts = series[label]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, label=label, **line_kw(label, ms=4.5))
        ax.set_xscale("log")
        _log_ticks_125(ax)
        ax.set_title(titles.get(sp, sp), fontsize=8)
        ax.set_xlabel("Warm memory (GB·s per 1000 inv.)")
        ax.grid(axis="y", lw=0.3, alpha=0.35)
    axes[0].set_ylabel("Cold-start rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8,
               bbox_to_anchor=(0.5, 1.14), columnspacing=0.75,
               handlelength=1.25, fontsize=6.7)
    save(fig, "fig03_steady_pareto")


# ============================ archived merged steady-state frontier, both traces
def _knot(v, _pos=None):
    """k/M-notation tick label. Elsevier artwork sizing requires printed
    sub/superscripts >= 6pt; plain k-notation sidesteps mathtext exponents
    entirely (10^3 -> 1k) and is shorter, so labels collide less."""
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def _log_knot(ax):
    """k/M tick labels on a log x-axis (no superscript exponents)."""
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_knot))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _log_ticks_125(ax):
    """Label a log x-axis only at the 1/2/5 subdivisions, in k-notation.

    1/2/3/5 was tried first and still collides: at this panel width 2x10^3
    and 3x10^3 touch.
    """
    lo, hi = ax.get_xlim()
    ticks = [m * 10 ** k for k in range(-2, 8) for m in (1, 2, 5)
             if lo <= m * 10 ** k <= hi]

    fmt = _knot

    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt))


def _pareto_kw(label, ms=4.0):
    """Fig. 3-local override of line_kw() for the coincident hero pair.

    The archived blended arm and WINTER-G v3 run 0.1-1.3pt apart in all six panels (measured on
    the rendered ~100pt-tall axes) while the shared hero stroke is 1.9pt wide
    with 4pt markers, so the gate curve was being drawn inside the learned arm's line.
    Draw the gate as a wide translucent band underneath and the hero as a thin
    line on top: where they coincide the reader sees a black core inside a blue
    band, where they diverge the band emerges. The hero marker shrinks to 3.0
    so it no longer swallows the 3.0pt band at every rho point. Local to this
    figure -- the shared IDENT identity the other figures use is unchanged.
    """
    kw = line_kw(label, ms=ms)
    if label == "Learned serving arm":
        kw.update(linewidth=1.3, markersize=3.0, zorder=6)
    elif label == "WINTER-G v3":
        kw.update(linewidth=3.0, alpha=0.45, markersize=2.6,
                  markeredgewidth=0.0, zorder=2)
    return kw


def fig_pareto_merged():
    """2021 (top row) and 2019 (bottom row) frontiers in one float.

    The two were separate figures eight pages apart while plotting the same
    axes, the same policies and the same three splits; stacking them puts the
    replication claim of main 10.1 in front of the reader instead of asking
    them to hold Fig. 3 in memory. Row 1 is the 10-seed 2021 campaign, row 2
    the population-weighted 2019 re-run.
    """
    splits = ["S1", "S2", "S3"]
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.55))

    # ---- row 0: Azure 2021 (body of the former fig_pareto) ----
    data = load("sim_results_des.json")
    titles = {"S1": "S1 (35 functions)", "S2": "S2 (86 functions)",
              "S3": "S3 (172 functions)"}
    RHO0 = {0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0}
    CORE = {"B4a_ewma", "Oracle", "B1_fixed_keepalive"}
    agg = {}
    for e in data:
        if e["method"] in ("B2_histogram", "B5_global"):
            agg.setdefault((e["split"], e["method"], e["cost_ratio"]), []).append(
                (e["csr"], e["wm_per_1k_inv"]))
    for e in load_live("sim_results_des_revision.json")["results"]:
        if e["method"] in CORE and e["cost_ratio"] in RHO0:
            agg.setdefault((e["split"], e["method"], e["cost_ratio"]), []).append(
                (e["csr"], e["wm_per_1k_inv"]))
    for e in load_live("revision_r16_faithful_primary_experiment.json")["results"]:
        if e["method"] == "A5_faithful" and e["cost_ratio"] in RHO0:
            agg.setdefault((e["split"], e["method"], e["cost_ratio"]), []).append(
                (e["csr"], e["wm_per_1k_inv"]))
    for e in load_live("revision_r17_faithful_gates_2021.json")["results"]:
        if e["method"] == "A5_gated_aq_faithful" and e["cost_ratio"] in RHO0:
            agg.setdefault((e["split"], e["method"], e["cost_ratio"]), []).append(
                (e["csr"], e["wm_per_1k_inv"]))
    # B3 (spectral) ran the same campaign on a 4-point rho grid (F1). It is
    # the arm whose position on this plane is the finding, so it is plotted
    # rather than only tabulated.
    for e in load_live("revision_f1_fourier_steady.json")["results"]:
        agg.setdefault((e["split"], e["method"], e["cost_ratio"]),
                       []).append((e["csr"], e["wm_per_1k_inv"]))
    order = ["Section-4 always-on learned head", "WINTER-G (720 min)", "EWMA", "Spectral (B3)", "Keep-alive",
             "Histogram (B2)", "Global", "Oracle"]
    for ax, sp in zip(axes[0], splits):
        series = {}
        for m in sorted({m for (s, m, r) in agg if s == sp}):
            pts = sorted(
                (np.mean([w for _, w in agg[(sp, m, r)]]),
                 100 * np.mean([c for c, _ in agg[(sp, m, r)]]))
                for r in sorted({r for (s, mm, r) in agg if s == sp and mm == m})
            )
            series[METHOD_MAP.get(m, m)] = pts
        for label in order:
            if label not in series:
                continue
            xs, ys = zip(*series[label])
            ax.plot(xs, ys, label=label, **_pareto_kw(label, ms=4.0))
        ax.set_xscale("log")
        _log_ticks_125(ax)
        ax.set_title(titles.get(sp, sp), fontsize=8)
        # 8% y-headroom (default 5%): the S3 Histogram point and the S2
        # Global point sat ~2pt from the (spineless) frame top, which reads
        # as clipping at print size.
        ax.margins(y=0.08)
        ax.grid(axis="y", lw=0.3, alpha=0.35)
        ax.tick_params(labelsize=7.8)
    axes[0][0].set_ylabel("Cold-start rate (%)", fontsize=8)

    # ---- row 1: Azure 2019 (body of the former fig_pareto_2019) ----
    # des_2019_stats.json carries the v2 gate (the legacy variant, supplement
    # only); the deployed v3 was re-run later and population-weighted by the
    # same estimator into revision_2019_new_stats.json, which is what main 10.1
    # reports. Merge the two so the row plots the gate the text discusses --
    # identical key format (split|rho|method) and identical weighting.
    with open("/data/260715/serverless-fewshot/results/runs/des_2019_stats.json") as f:
        stats = json.load(f)
    with open("/data/260715/serverless-fewshot/results/runs/"
              "revision_2019_new_stats.json") as f:
        stats = {**stats, **json.load(f)}
    keys = [k.split("|") for k in stats]
    # B3's 2019 grid is population-weighted by the same estimator in F5; fold
    # it into the same key format so the row plots it like any other curve.
    try:
        f5 = load_live("revision_f5_fourier_stats.json")["surfaces"]
        for sp_, v in f5.get("azure2019_steady", {}).items():
            for c in v["curves"].get("B3_fourier", []):
                stats[f"{sp_}|{c['rho']}|B3_fourier"] = {
                    "wm_per_1k_weighted": c["wm_per_1k"],
                    "pop_csr": c["csr_pp"] / 100.0}
        keys = [k.split("|") for k in stats]
    except Exception as exc:                       # pragma: no cover
        print(f"  (2019 B3 curve unavailable: {exc})")
    curve_methods = ["A5_full_system", "A5_gated_v3", "B4a_ewma",
                     "B5_global", "B3_fourier", "Oracle"]
    point_methods = ["B1_fixed_keepalive", "B2_histogram"]
    for ax, sp in zip(axes[1], splits):
        rhos = sorted({float(r) for s, r, m in keys if s == sp})
        for m in curve_methods:
            xs, ys = [], []
            for r in rhos:
                k = f"{sp}|{r}|{m}"
                if k in stats:
                    xs.append(stats[k]["wm_per_1k_weighted"])
                    ys.append(stats[k]["pop_csr"] * 100)
            lbl = METHOD_MAP.get(m, m)
            ax.plot(xs, ys, label=lbl if sp == "S1" else None,
                    **_pareto_kw(lbl, ms=4.0))
        for m in point_methods:
            k = f"{sp}|10.0|{m}"
            if k in stats:
                lbl = METHOD_MAP.get(m, m)
                d = _ident(lbl)
                ax.plot(stats[k]["wm_per_1k_weighted"], stats[k]["pop_csr"] * 100,
                        marker=d["marker"] or "D", ms=5.5, color=d["color"],
                        ls="", markeredgewidth=0.6,
                        label=lbl if sp == "S1" else None)
        ax.set_xscale("log")
        # The 2019 row spans barely one decade, so matplotlib's default log
        # ticker labels every minor subdivision and 3x10^3 collides with
        # 4x10^3 in all three panels. Place ticks explicitly at the 1/2/5
        # subdivisions; a LogLocator with subs= is not enough, because it
        # falls back to a linear tick set below one decade.
        _log_ticks_125(ax)
        if sp == "S1":
            # Default locator labels only 0.6/0.8 here while the data reaches
            # 0.43; 0.1-steps add ticks bracketing the plotted range.
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.set_xlabel("Warm memory (GB·s per 1000 inv.)", fontsize=8)
        ax.grid(axis="y", lw=0.3, alpha=0.35)
        ax.tick_params(labelsize=7.8)
    axes[1][0].set_ylabel("Population CSR (%)", fontsize=8)

    # row identifiers, so the two vintages are never confused
    axes[0][0].text(-0.30, 0.5, "Azure 2021", transform=axes[0][0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=8.5)
    axes[1][0].text(-0.30, 0.5, "Azure 2019", transform=axes[1][0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=8.5)

    fig.tight_layout()
    handles, labels = [], []
    seen = set()
    for ax in (axes[0][0], axes[1][0]):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab and lab not in seen:
                handles.append(h)
                labels.append(lab)
                seen.add(lab)
    fig.legend(handles, labels, loc="upper center", ncol=8,
               bbox_to_anchor=(0.5, 1.09), columnspacing=0.75,
               handlelength=1.25, fontsize=6.7)
    save(fig, "fig03_steady_pareto_legacy_merged_unused")


# ================================= F2 onboarding + heads (merged, 3 panels)
def fig_onboarding_heads():
    ob = load("onboarding_drift_results.json")["onboarding"]
    ab = load("ablation_results.json")
    rho = "1.0"
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.9),
                             gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})

    # (a) rolling CSR, synthetic zero-history injection
    ax = axes[0]
    methods = ["Oracle", "B1_fixed_keepalive", "B4a_ewma", "A5_noproto", "A5_proto"]
    for m in methods:
        v = ob["by_rho"][rho].get(m)
        if v is None:
            continue
        label = METHOD_MAP.get(m, m)
        r = arr(v["rolling_csr"]) * 100
        ax.plot(np.arange(len(r)), r, label=label,
                **line_kw(label, markevery=28, ms=4.0))
    ax.set_xlabel("Minutes since injection", fontsize=7.7)
    ax.set_ylabel("Rolling CSR (%, 15-min window)", fontsize=7.7)
    ax.tick_params(labelsize=7.7)
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    # Place the legend ABOVE panel (a): the ~28% mid-window spike leaves no
    # in-panel space wide enough for the 5-entry legend without a collision.
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2,
              handlelength=1.3, columnspacing=1.0, fontsize=7.7,
              frameon=False)

    # (b) CRPS by head type and K
    ax = axes[1]
    hm = ab["heatmap_k_head"]
    ks, heads = hm["k_values"], hm["head_types"]
    M = np.asarray(hm["crps_matrix"], dtype=float)
    from matplotlib.colors import LogNorm
    im = ax.imshow(M.T, aspect="auto", cmap="Greys_r", origin="lower",
                   norm=LogNorm(vmin=max(M.min(), 1e-3), vmax=M.max()))
    ax.set_xticks(range(len(ks)), [str(k) for k in ks], fontsize=7.7)
    ax.set_yticks(range(len(heads)), ["Ridge (zero at K=0)", "ANIL-GD", "Bayes"],
                  fontsize=7.7)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlabel("Adaptation shots $K$", fontsize=7.7)
    for i in range(len(ks)):
        for j in range(len(heads)):
            # leading zero stripped (".030") so 7.7pt values fit the cells
            s = f"{M[i, j]:.3f}" if M[i, j] < 1 else f"{M[i, j]:.2f}"
            ax.text(i, j, s[1:] if s.startswith("0.") else s,
                    ha="center", va="center", fontsize=7.7,
                    color="white" if im.norm(M[i, j]) < 0.48 else "#1a1a1a")
    ax.spines[:].set_visible(False)
    ax.set_title("CRPS by head and $K$", fontsize=8)
    ax.tick_params(axis="y", labelsize=6.2)

    # (c) first-hour CRPS, prototype vs no-prototype
    ax = axes[2]
    crps = ob["crps_vs_time"]
    for m, label in [("A5_noproto", "Learned component, no prototype"),
                     ("A5_proto", "WINTER component")]:
        r = arr(crps[m])
        ax.plot(np.arange(len(r)), r, label=label,
                **line_kw(label, markevery=9, ms=4.0))
    ax.set_xlim(0, 60)
    ax.set_xlabel("Minutes since injection", fontsize=7.7)
    ax.set_ylabel("CRPS (new functions)", fontsize=7.7)
    ax.tick_params(labelsize=7.7)
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    ax.legend(handlelength=1.6, fontsize=7.7)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig09_onboarding_heads")


# ================================================================= F3 drift
def fig_drift():
    best = pd.read_csv(os.path.join(TABLES, "T_r23_drift_v2_cross_provider_winter_best.csv"))
    overall = pd.read_csv(os.path.join(TABLES, "T_r23_drift_v2_cross_provider_overall.csv"))

    arm_label = {
        "winter_scheduled": "scheduled",
        "winter_trigger": "trigger",
        "winter_sched_trigger": "sched.+trigger",
    }
    cond_labels = {
        "natural_burst_down": "Nat. burst down",
        "natural_burst_up": "Nat. burst up",
        "natural_collapse": "Nat. collapse",
        "natural_duty_down": "Nat. duty down",
        "natural_duty_up": "Nat. duty up",
        "natural_growth": "Nat. growth",
        "burst_compress_4": "Syn. burst compress",
        "phase_circular_120": "Syn. phase shift",
        "ramp_down_0.2_120": "Syn. ramp down",
        "ramp_up_5_120": "Syn. ramp up",
        "scale_down_0.2": "Syn. scale down $\\times0.2$",
        "scale_down_0.5": "Syn. scale down $\\times0.5$",
        "scale_up_2": "Syn. scale up $\\times2$",
        "scale_up_5": "Syn. scale up $\\times5$",
    }
    cond_order = [
        "natural_burst_down", "natural_burst_up", "natural_collapse",
        "natural_duty_down", "natural_duty_up", "natural_growth",
        "burst_compress_4", "phase_circular_120", "ramp_down_0.2_120",
        "ramp_up_5_120", "scale_down_0.2", "scale_down_0.5",
        "scale_up_2", "scale_up_5",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # (a) Best practical re-adaptation against frozen, by drift condition.
    ax = axes[0]
    pooled = best[(best["provider"] == "pooled") &
                  (best["source"].isin(["natural", "synthetic"]))]
    ybase = np.arange(len(cond_order))
    for rho, marker, offset, color in [(1.0, "o", -0.12, _BLACK),
                                       (10.0, "s", 0.12, _BLUE)]:
        xs = []
        ys = []
        for i, kind in enumerate(cond_order):
            row = pooled[(pooled["kind"] == kind) & (pooled["rho"] == rho)]
            xs.append(float(row["best_delta240_pp"].iloc[0]))
            ys.append(i + offset)
        ax.scatter(xs, ys, s=20, marker=marker, color=color,
                   edgecolor="black", linewidth=0.35,
                   label=f"$\\rho={int(rho)}$", zorder=3)
    ax.axvline(0, color="#777777", lw=0.7, ls=":")
    ax.set_yticks(ybase, [cond_labels[k] for k in cond_order])
    ax.invert_yaxis()
    ax.set_xlim(-0.95, 0.25)
    ax.set_xlabel("$\\Delta$CSR vs. frozen, pp")
    ax.set_title("Best practical WINTER arm by condition", fontsize=8)
    ax.grid(axis="x", lw=0.3, alpha=0.35)
    ax.legend(handletextpad=0.3, borderaxespad=0.2, loc="lower left")

    # (b) Pooled source-cost aggregates.
    ax = axes[1]
    groups = [("Natural\n$\\rho=1$", "natural", 1.0),
              ("Natural\n$\\rho=10$", "natural", 10.0),
              ("Synthetic\n$\\rho=1$", "synthetic", 1.0),
              ("Synthetic\n$\\rho=10$", "synthetic", 10.0)]
    frozen_vals = []
    adaptive_vals = []
    rel_vals = []
    for _, source, rho in groups:
        sub = overall[(overall["provider"] == "pooled") &
                      (overall["source"] == source) &
                      (overall["rho"] == rho)]
        frozen = float(sub[sub["arm"] == "winter_frozen"]["csr_240_pct"].iloc[0])
        cand = sub[sub["arm"].isin(arm_label)]
        chosen = cand.loc[cand["csr_240_pct"].idxmin()]
        adaptive = float(chosen["csr_240_pct"])
        frozen_vals.append(frozen)
        adaptive_vals.append(adaptive)
        rel_vals.append(-100.0 * float(chosen["delta_vs_frozen_csr_240_pp"]) / frozen)
    x = np.arange(len(groups))
    w = 0.34
    ax.bar(x - w / 2, frozen_vals, w, color="white", hatch="///",
           edgecolor="black", linewidth=0.45, label="frozen")
    ax.bar(x + w / 2, adaptive_vals, w, color=_BLACK,
           edgecolor="black", linewidth=0.45, label="best adaptive")
    for i, (val, rel) in enumerate(zip(adaptive_vals, rel_vals)):
        ax.text(i + w / 2, val + 0.012, f"{rel:.0f}%", ha="center",
                va="bottom", fontsize=6.6)
    ax.set_xticks(x, [g[0] for g in groups])
    ax.set_ylim(0, max(frozen_vals + adaptive_vals) * 1.22)
    ax.set_ylabel("CSR, 240 min post-onset (%)")
    ax.set_title("Pooled aggregate reduction", fontsize=8)
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    ax.legend(handlelength=1.2, loc="upper left")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig10_drift_response")


# ============================== lag figures (merged from the lag analysis)
RHOS_L = ["1.0", "10.0", "100.0"]
ARM_L = {
    "A5_proto": "WINTER component",
    "A5_proto_zero": "Prototype zero-shot",
    "B4a_ewma": "EWMA",
    "B1_fixed_keepalive": "Keep-alive",
    "gated_v4": "WINTER-G v4",
    "Oracle": "Oracle",
    "B3_fourier": "Spectral (B3)",
}


def _lag_style(label):
    if label == "Prototype zero-shot":
        return C["Prototype zero-shot"], LS["Prototype zero-shot"]
    return style_of(label)


def fig_lag_fractile():
    """3 panels: best-lag histogram, alignment-r dumbbells, pinball vs tau."""
    d = load_derived()
    la = d["lag_audit"]
    names = [("A5_learned", "Archived blended arm"), ("B4a_ewma", "EWMA"),
             ("B7_lstm_median", "LSTM")]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.16, 1.8),
                                  gridspec_kw={"width_ratios": [1.2, 0.9, 1.2]})
    width = 0.26
    x = np.arange(6)
    hat = {"Archived blended arm": None, "EWMA": "///", "LSTM": "xxx"}
    for i, (k, lab) in enumerate(names):
        h = np.asarray(la[k]["lag_hist_0_5"], float)
        a.bar(x + (i - 1) * width, h / h.sum() * 100, width,
              facecolor=C[lab], hatch=hat[lab], edgecolor="black",
              linewidth=0.4, label=lab)
    a.set_xticks(x, ["0", "1", "2", "3", "4", "$\\geq$5"])
    a.set_xlabel("Best cross-correlation lag (min)")
    a.set_ylabel("% of functions")
    a.legend(fontsize=6.5)
    a.grid(axis="y", lw=0.3, alpha=0.35)
    for i, (k, lab) in enumerate(names):
        r0, rb = la[k]["mean_r_lag0"], la[k]["mean_r_best"]
        b.plot([i, i], [r0, rb], color=C[lab], lw=1.4, zorder=1)
        b.scatter([i], [r0], marker="o", facecolor="white", edgecolor=C[lab],
                  linewidth=0.9, s=22, zorder=2,
                  label="$r$ at lag 0" if i == 0 else None)
        b.scatter([i], [rb], marker="o", color=C[lab], s=22, zorder=2,
                  label="$r$ at best lag" if i == 0 else None)
        b.annotate(f"lag {la[k]['median_best_lag']:.0f}", (i, rb),
                   textcoords="offset points", xytext=(5, -2), fontsize=6.5)
    b.set_xticks(range(3), ["Archived\nblended", "EWMA", "LSTM"], fontsize=6.5)
    b.set_xlim(-0.5, 2.9)
    b.set_ylabel("Mean alignment $r$")
    b.set_ylim(0, 1)
    b.legend(loc="lower left", fontsize=6.5)
    taus = sorted(float(t) for t in d["fractile_pinball"])
    for k, lab in [("A5_learned", "Archived blended arm"), ("B4a_ewma", "EWMA"),
                   ("B7_lstm", "LSTM")]:
        y = [d["fractile_pinball"][f"{t:.6f}"][k] for t in taus]
        c.plot(taus, y, label=lab, **line_kw(lab, markevery=4, ms=4.0))
    c.axvspan(0.0, 0.10, color="#000000", alpha=0.07, lw=0)
    c.text(0.055, 0.135, "LSTM\nwins", fontsize=6.5, ha="center")
    tau_star = 10.0 / 11.0
    c.axvline(tau_star, color="black", lw=0.6, ls=(0, (1, 1)))
    # fontsize 7.9: mathtext scripts render at ~0.7x, and Elsevier requires
    # printed sub/superscripts >= 6pt (this figure prints at 1.085x).
    c.text(tau_star - 0.02, 0.02, "$\\tau^{*}(\\rho{=}10)$", fontsize=7.9,
           ha="right")
    c.set_xlabel("Quantile level $\\tau$")
    c.set_ylabel("Pinball loss")
    c.legend(loc="upper left", bbox_to_anchor=(0.14, 1.02), fontsize=6.5)
    c.grid(axis="y", lw=0.3, alpha=0.35)
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig04_persistence_lag_fractile")


def fig_rolling_al():
    """Rolling cohort CSR per rho with AL(0.1) marks; includes corrected TSFM."""
    a4 = load_live("revision_a4_crosstrace.json")
    e5c = load_chronos_native()["cohort"]
    # B3 was simulated on the same cohort under the same protocol (F4), so its
    # curve belongs here: it is the one arm whose transient never separates
    # from keep-alive's.
    b3 = load_live("revision_f4_fourier_cohort_azure2019.json")
    d = load_derived()
    arms = ["A5_proto", "A5_proto_zero", "gated_v4", "B4a_ewma",
            "B1_fixed_keepalive", "Oracle"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.75), sharey=True)
    for ax, rho in zip(axes, RHOS_L):
        for arm in arms:
            lab = ARM_L[arm]
            ax.plot(arr(a4["by_rho"][rho][arm]["rolling_csr"]) * 100,
                    label=lab, **line_kw(lab, markevery=30, ms=3.6))
        ax.plot(arr(e5c["by_rho"][rho]["native_direct"]["rolling_csr"]) * 100,
                label="TSFM", **line_kw("TSFM", markevery=30, ms=3.6))
        lb3 = ARM_L["B3_fourier"]
        ax.plot(arr(b3["by_rho"][rho]["rolling_csr"]) * 100,
                label=lb3, **line_kw(lb3, markevery=30, ms=3.6))
        for arm, dy in [("A5_proto", 0.86), ("B4a_ewma", 0.70)]:
            al = d["onboarding_AL_minutes"][rho][arm]["eps0.1"]
            if al is not None and al > 0:
                lab = ARM_L[arm]
                col = C[lab]
                ax.axvline(al, color=col, ls=LS[lab], lw=0.7, alpha=0.7)
                ax.text(al + 3, ax.get_ylim()[1] * dy, f"AL={al}",
                        fontsize=6.7, color=col)
        ax.set_title(f"$\\rho$={float(rho):g}", fontsize=8)
        ax.set_xlabel("Minutes since first observed invocation")
        ax.set_xlim(0, 240)
        ax.grid(axis="y", lw=0.3, alpha=0.35)
    axes[0].set_ylabel("Rolling CSR (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=8, loc="upper center",
               bbox_to_anchor=(0.5, 1.16), handlelength=1.6,
               columnspacing=0.9, fontsize=6.7)
    save(fig, "fig07_rolling_adaptation_lag")


def fig_gate_truncation():
    """WINTER-G v4 hugs EWMA through the lag window (rho=10)."""
    a4 = load_live("revision_a4_crosstrace.json")
    d = load_derived()
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    rho = "10.0"
    offsets = {"A5_proto": (4, 8), "gated_v4": (4, -12), "B4a_ewma": (4, 10)}
    for arm in ["A5_proto", "gated_v4", "B4a_ewma"]:
        lab = ARM_L[arm]
        col, ls = _lag_style(lab)
        ax.plot(arr(a4["by_rho"][rho][arm]["rolling_csr"]) * 100,
                color=col, ls=ls, label=lab, lw=1.1)
        al = d["onboarding_AL_minutes"][rho][arm]["eps0.1"]
        ax.annotate(f"{lab}: AL={al}",
                    (al, arr(a4["by_rho"][rho][arm]["rolling_csr"])[al] * 100),
                    textcoords="offset points", xytext=offsets[arm],
                    fontsize=6, color=col)
    ax.set_xlabel("Minutes since first observed invocation")
    ax.set_ylabel("Rolling CSR (%)")
    ax.set_xlim(0, 120)
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    ax.legend(title="$\\rho$=10", title_fontsize=6.8, fontsize=6.4)
    save(fig, "fig_gate_truncation")


def fig_frontier():
    """Cohort CSR vs measured per-function serving forecast cost (rho=10)."""
    d = load_derived()
    e5 = load_chronos_native()
    e5_infer = load_live("revision_e5_chronos.json")
    r9 = load_live("revision_r9_forecast_cost.json")
    rho = "10.0"
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    winter_us = r9["comparison"]["winter_us_per_forecast_gpu_bf16_b10k"]
    chron_us = e5_infer["inference"]["mean_batch100_sec"] / 100 * 1e6
    pts = [
        ("WINTER component", winter_us, d["overall_csr"][rho]["A5_proto"] * 100,
         "8 KB state", "o", (7, 2)),
        ("TSFM", chron_us, e5["cohort"]["by_rho"][rho]["native_direct"]["overall_csr"] * 100,
         f"{e5_infer['inference']['params']/1e6:.1f}M params", "D", (-8, 8)),
    ]
    for lab, x, y, note, mk, off in pts:
        ax.scatter([x], [y], s=36, marker=mk, facecolor="white",
                   edgecolor="black", lw=0.9, zorder=3)
        ax.annotate(f"{lab}\n({note})", (x, y), textcoords="offset points",
                    xytext=off, fontsize=6.5,
                    ha="right" if off[0] < 0 else "left")
    for base, lab in [("B4a_ewma", "EWMA"), ("B1_fixed_keepalive", "Keep-alive")]:
        y = d["overall_csr"][rho][base] * 100
        col, ls = style_of(lab)
        ax.axhline(y, color=col, ls=ls, lw=0.9)
        ax.text(0.3, y + 0.012, lab, fontsize=6.5, va="bottom", color=col)
    ax.set_xscale("log")
    ax.set_xlim(0.8, chron_us * 4)
    ax.set_ylim(0.15, 0.95)
    ax.set_xlabel("Per-function serving forecast/read cost ($\\mu$s, measured)")
    ax.set_ylabel("Cohort CSR (%), $\\rho$=10")
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    save(fig, "fig_frontier")


def fig_e2_forest():
    """Supplement: cohort-definition sweep forest plot (27 cells)."""
    e2 = load_live("revision_e2_cohort_sweep.json")
    cells = list(e2["cells"].keys())
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0), sharey=True)
    for ax, rho in zip(axes, RHOS_L):
        for i, cell in enumerate(cells):
            rec = e2["cells"][cell]["by_rho"][rho]["paired_vs_A5_proto"]["B4a_ewma"]
            ax.scatter([rec["mean_diff"]], [i], marker="o", s=18,
                       facecolor="black" if rec["wilcoxon_p"] < 0.05 else "white",
                       edgecolor="black", lw=0.8, zorder=3)
        ax.axvline(0, color="#8c8c8c", lw=0.6)
        ax.set_title(f"$\\rho$={float(rho):g}", fontsize=8)
        ax.set_xlabel("Mean paired cold-start\nadvantage vs EWMA")
        ax.grid(axis="x", lw=0.3, alpha=0.35)
    axes[0].set_yticks(range(len(cells)),
                       [c.replace("gap", "gap ").replace("_thr", ", thr ")
                        for c in cells])
    fig.text(0.005, 0.96, "filled = $p<0.05$", fontsize=6.5)
    save(fig, "fig08_cohort_robustness_forest")


# ============================================================== F5 testbed
def fig_testbed():
    cdf = load("testbed_cdf.json")
    runtimes = [("python-ml", "Python (ML deps)"), ("node-api-real", "Node.js API"),
                ("java-svc", "Java service")]
    styles = [("#000000", "-"), ("#666666", (0, (4, 1.6))),
              ("#9e9e9e", (0, (1, 1.3)))]
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.4))
    for (rt, label), (color, ls) in zip(runtimes, styles):
        for ax, kind in zip(axes, ("cold", "warm")):
            v = np.sort(np.asarray(cdf[kind][rt], dtype=float))
            y = np.arange(1, len(v) + 1) / len(v)
            xs = v if kind == "cold" else v * 1e3
            ax.step(xs, y, where="post", color=color, linestyle=ls, label=f"{label} (n={len(v)})")
    axes[0].set_xlabel("Cold-start latency (s)")
    axes[1].set_xlabel("Warm latency (ms)")
    axes[0].set_ylabel("CDF")
    axes[0].set_title("Cold invocations", fontsize=8)
    axes[1].set_title("Warm invocations", fontsize=8)
    axes[1].set_ylabel("CDF")
    for ax in axes:
        ax.grid(axis="y", lw=0.3, alpha=0.35)
        ax.set_ylim(0, 1.02)
    # Upper-left is the empty low-CDF corner; lower-right sat under the
    # rising CDF curves (the Python step passed through the legend).
    axes[0].legend(handlelength=1.6, loc="upper left", fontsize=6.2)
    axes[1].legend(handlelength=1.6, loc="upper left", fontsize=6.2)
    fig.subplots_adjust(hspace=0.52)
    save(fig, "fig11_testbed_latency_cdf")


# =========================================================== F6 stratified
def fig_stratified():
    data = load("sim_results_des.json")
    sp, rho = "S3", 1.0
    show = ["A5_full_system", "B4a_ewma", "B1_fixed_keepalive", "B2_histogram"]
    # per-function arrays, seed-mean of per-func CSR
    per = {}
    for e in data:
        if e["split"] != sp or e["cost_ratio"] != rho or e["method"] not in show:
            continue
        per.setdefault(e["method"], []).append(
            (np.asarray(e["func_total"], dtype=float),
             np.asarray(e["func_cold"], dtype=float)))
    edges = [0, 100, 1_000, 10_000, np.inf]
    blabels = ["<100", "100–1k", "1k–10k", ">10k"]
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    width = 0.2
    for mi, m in enumerate(show):
        runs = per[m]
        tot = np.mean([t for t, _ in runs], axis=0)
        cold = np.mean([c for _, c in runs], axis=0)
        ys = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (tot >= lo) & (tot < hi)
            ys.append(100 * cold[sel].sum() / max(tot[sel].sum(), 1))
        label = METHOD_MAP[m]
        fill, hatch = BAR.get(label, ("#888888", None))
        ax.bar(np.arange(len(ys)) + (mi - 1.5) * width, ys, width * 0.92,
               color=fill, label=label, hatch=hatch,
               edgecolor="black", linewidth=0.35)
    ax.set_xticks(range(len(blabels)), blabels)
    ax.set_xlabel("Function invocation volume (evaluation window)")
    ax.set_ylabel("Cold-start rate (%)")
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    ax.legend(ncol=2, handlelength=1.2)
    save(fig, "fig05_stratified_by_volume")




# ============================================================ F7 2019 Pareto
def fig_pareto_2019():
    with open("/data/260715/serverless-fewshot/results/runs/des_2019_stats.json") as f:
        stats = json.load(f)
    with open("/data/260715/serverless-fewshot/results/runs/"
              "revision_2019_new_stats.json") as f:
        stats = {**stats, **json.load(f)}
    keys = [k.split("|") for k in stats]
    try:
        f5 = load_live("revision_f5_fourier_stats.json")["surfaces"]
        for sp_, v in f5.get("azure2019_steady", {}).items():
            for c in v["curves"].get("B3_fourier", []):
                stats[f"{sp_}|{c['rho']}|B3_fourier"] = {
                    "wm_per_1k_weighted": c["wm_per_1k"],
                    "pop_csr": c["csr_pp"] / 100.0}
        keys = [k.split("|") for k in stats]
    except Exception as exc:                       # pragma: no cover
        print(f"  (2019 B3 curve unavailable: {exc})")
    splits = ["S1", "S2", "S3"]
    curve_methods = ["A5_full_system", "A5_gated_v3", "B4a_ewma",
                     "B5_global", "B3_fourier", "Oracle"]
    point_methods = ["B1_fixed_keepalive", "B2_histogram"]
    # Sized to the two-column span (\textwidth ~7.16in) so it renders ~1:1 as a
    # figure* float rather than being downscaled from 9.6in in a single column.
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.1), sharey=False)
    for ax, sp in zip(axes, splits):
        rhos = sorted({float(r) for s, r, m in keys if s == sp})
        for m in curve_methods:
            xs, ys = [], []
            for r in rhos:
                k = f"{sp}|{r}|{m}"
                if k in stats:
                    xs.append(stats[k]["wm_per_1k_weighted"])
                    ys.append(stats[k]["pop_csr"] * 100)
            lbl = METHOD_MAP.get(m, m)
            ax.plot(xs, ys, label=lbl if sp == "S1" else None,
                    **_pareto_kw(lbl, ms=4.0))
        for m in point_methods:
            k = f"{sp}|10.0|{m}"
            if k in stats:
                lbl = METHOD_MAP.get(m, m)
                d = _ident(lbl)
                ax.plot(stats[k]["wm_per_1k_weighted"],
                        stats[k]["pop_csr"] * 100,
                        marker=d["marker"] or "D", ms=6, color=d["color"], ls="",
                        markeredgewidth=0.6,
                        label=lbl + " (pctl-only)" if "Hist" in lbl and sp == "S1"
                        else (lbl if sp == "S1" else None))
        ax.set_xscale("log")
        _log_ticks_125(ax)
        ax.set_title(sp, fontsize=9)
        ax.set_xlabel("Warm memory (GB·s per 1000 inv.)", fontsize=8)
        ax.grid(axis="y", lw=0.3, alpha=0.35)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("population CSR (%)", fontsize=8)
    fig.tight_layout()
    # Figure-level legend above the panels (matches fig03); the in-panel
    # 'best' placement collided with the S1 y-axis ticks and title.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7,
               bbox_to_anchor=(0.5, 1.16), columnspacing=1.1,
               handlelength=1.6, fontsize=6.5)
    save(fig, "fig03_steady_pareto_2019_archived")


# ======================================================== F8 Natural cohort
def fig_cohort():
    LIVE = "/data/260715/serverless-fewshot/results/runs"
    with open(os.path.join(LIVE, "revision_a4_crosstrace.json")) as f:
        d = json.load(f)
    try:
        with open(os.path.join(LIVE, "revision_e5d_chronos_native.json")) as f:
            ch = json.load(f)
    except Exception:
        ch = None
    rhos = ["1.0", "10.0", "100.0"]
    # (arm key, legend label, fill color, hatch) — policy palette + hatch as the
    # grayscale/CVD-safe redundancy that separates same-colour learned variants.
    # The zero-shot arm is a learned variant, so it keeps the black edge/hatch
    # family identity but takes a white fill: a black hatch on a black fill is
    # invisible, which left these two bars (and their legend swatches)
    # indistinguishable. Same device as BAR['Learned component, no prototype'].
    arms = [("A5_proto", "WINTER component", _BLACK, ""),
            ("A5_proto_zero", "Prototype zero-shot", "#ffffff", "//"),
            ("gated_v3", "WINTER-G", _BLUE, "xx"),
            ("gated_v4", "WINTER-G v4", _PURPLE, "\\\\"),
            ("B4a_ewma", "EWMA", _RED, ""),
            ("B3_fourier", "Spectral (B3)", _ORANGE, "++"),
            ("B1_fixed_keepalive", "Keep-alive", _TEAL, ".."),
            ("Oracle", "Oracle", _GRAY, "")]
    # B3 lives in its own archive (F4, same cohort/seeds/protocol); splice it
    # into the arm lookup so the bar loop stays one loop.
    b3d = json.load(open(os.path.join(LIVE,
                                      "revision_f4_fourier_cohort_azure2019.json")))
    for r in rhos:
        d["by_rho"][r]["B3_fourier"] = b3d["by_rho"][r]
    fig, (ax1, axf) = plt.subplots(2, 1, figsize=(3.5, 3.9))
    n = len(arms) + (1 if ch else 0)
    w = 0.8 / n
    for i, (m, lbl, col, hatch) in enumerate(arms):
        vals = [d["by_rho"][r][m]["overall_csr"] * 100 for r in rhos]
        ax1.bar(np.arange(3) + i * w, vals, w, color=col, edgecolor="black",
                lw=0.4, hatch=hatch, label=lbl)
    if ch:
        vals = [ch["cohort"]["by_rho"][r]["native_direct"]["overall_csr"] * 100 for r in rhos]
        ax1.bar(np.arange(3) + len(arms) * w, vals, w, color=_MAGENTA,
                edgecolor="black", lw=0.5, hatch="oo", label="TSFM (Chronos)")
    ax1.set_xticks(np.arange(3) + 0.4)
    ax1.set_xticklabels([r"$\rho$=1", r"$\rho$=10", r"$\rho$=100"], fontsize=8)
    # The rho=1 Spectral bar is 1.0006%, coincident with the 1.00 tick under
    # autoscale; ~11% headroom keeps the top gridline clear of the tallest bar.
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("CSR, first 4 h (%)", fontsize=8)
    ax1.tick_params(labelsize=7.6)
    ax1.legend(fontsize=7.0, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 1.02), columnspacing=0.8,
               handlelength=1.2, handletextpad=0.4)
    # (b) serving-cost frontier (rho=10), merged from fig_frontier
    dl = load_derived()
    e5f = load_chronos_native()
    e5_infer = load_live("revision_e5_chronos.json")
    r9 = load_live("revision_r9_forecast_cost.json")
    winter_us = r9["comparison"]["winter_us_per_forecast_gpu_bf16_b10k"]
    chron_us = e5_infer["inference"]["mean_batch100_sec"] / 100 * 1e6
    f7 = load_live("revision_f7_fourier_cost.json")
    b3_us = f7["scalability"]["amortized_us_per_func_per_tick"]
    for lab, x, y, note, off in [
            ("WINTER component", winter_us,
             dl["overall_csr"]["10.0"]["A5_proto"] * 100, "8 KB state", (7, 2)),
            ("Spectral (B3)", b3_us,
             b3d["by_rho"]["10.0"]["overall_csr"] * 100, "5.6 KB state", (8, 3)),
            ("TSFM", chron_us,
             e5f["cohort"]["by_rho"]["10.0"]["native_direct"]["overall_csr"] * 100,
             f"{e5_infer['inference']['params']/1e6:.1f}M params", (-8, 8))]:
        di = _ident(lab)
        axf.scatter([x], [y], s=44, marker=di["marker"], color=di["color"],
                    edgecolor="black", lw=0.6, zorder=3)
        axf.annotate(f"{lab}\n({note})", (x, y), textcoords="offset points",
                     xytext=off, fontsize=7.6,
                     ha="right" if off[0] < 0 else "left")
    for base, lab in [("B4a_ewma", "EWMA"),
                      ("B1_fixed_keepalive", "Keep-alive")]:
        y = dl["overall_csr"]["10.0"][base] * 100
        col, ls = style_of(lab)
        axf.axhline(y, color=col, ls=ls, lw=0.9)
        # Labels sit on the right half: the left is where the two cheap arms
        # (learned component, B3) plot, and B3's annotation collided with them there.
        axf.text(chron_us * 0.30, y + 0.012, lab, fontsize=7.6, va="bottom",
                 ha="center", color=col)
    axf.set_xscale("log")
    _log_knot(axf)
    axf.set_xlim(0.8, chron_us * 4)
    axf.set_ylim(0.15, 1.05)
    axf.tick_params(labelsize=7.6)
    axf.set_xlabel("Per-function serving forecast/read cost ($\\mu$s, measured)",
                   fontsize=7.6)
    axf.set_ylabel("Cohort CSR (%), $\\rho$=10", fontsize=7.6)
    axf.grid(axis="y", lw=0.3, alpha=0.35)
    fig.subplots_adjust(hspace=0.5)
    fig.tight_layout()
    save(fig, "fig06_onboarding_cohort_2019")


# ========================================================== F9 Regime map
def fig_regimemap():
    cells = [
        ("Early-life\nonboarding ($\\S$7)", "LEARNING LOWERS COLD STARTS\n$-$32...$-$46% Azure, $-$16...$-$24% Huawei\n(magnitude provider-dependent)", 0.15),
        ("Active-pool\nsteady window ($\\S$6)", "ROUTE TO LOW-COST ARMS\nfaithful equivalent only on Azure 2021 S1;\nEWMA better on S2/S3; Huawei archived arm loses", 0.55),
        ("Sparse / saturated ($\\S$11)", "Sparse: phase-sensitive trade-offs\nSaturated: limited predictor headroom\nunder the evaluated controller", 0.9),
        ("Event-level drift ($\\S$8)", "REFIT LOWERS CSR\nall pooled source-cost cells;\n23 of 28 condition cells", 0.35),
        ("Drift guardrails ($\\S$8)", "DIRECTION MATTERS\nduty-down, phase shift,\nand one collapse cell can harm", 0.75),
    ]
    # Designed at the two-column span width (~7.16in). Boxes widened to 0.32
    # (full span, small gaps) with trimmed fonts so the longest single-line
    # titles sit inside the borders. Arial font, with Liberation Sans as the
    # metric-compatible fallback when Arial is not installed. Extra vertical
    # room is left above/below the central "history/traffic density" band.
    AR = ["Arial", "Liberation Sans", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.16, 2.85))
    ax.axis("off")
    W = 0.32
    xs = [0.0, 0.34, 0.68]
    for i, (title, body, gray) in enumerate(cells[:3]):
        r = plt.Rectangle((xs[i], 0.55), W, 0.44, fc=str(0.98 - 0.13 * i),
                          ec="black", lw=0.8)
        ax.add_patch(r)
        cx = xs[i] + W / 2
        ax.text(cx, 0.95, title, ha="center", va="top",
                fontsize=6.9, fontweight="bold", family=AR)
        ax.text(cx, 0.78, body, ha="center", va="top", fontsize=6.7, family=AR)
    for j, (title, body, gray) in enumerate(cells[3:]):
        x = 0.17 + 0.34 * j
        r = plt.Rectangle((x, 0.01), W, 0.25, fc="0.93", ec="black",
                          lw=0.8, ls=(0, (3, 2)))
        ax.add_patch(r)
        cx = x + W / 2
        ax.text(cx, 0.225, title, ha="center", va="top", fontsize=6.8,
                fontweight="bold", family=AR)
        ax.text(cx, 0.125, body, ha="center", va="top", fontsize=6.7, family=AR)
    ax.annotate("", xy=(1.0, 0.44), xytext=(0.0, 0.44),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.5, 0.395, "history / traffic density $\\rightarrow$",
            ha="center", va="top", fontsize=6.8, family=AR)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig01_regime_map")


# =================================================== F9b Regime map, 1 column
def fig_regimemap_1col():
    """Single-column redraw of the regime map (fig_regimemap stays as-is).

    Direction only: every magnitude the two-column version printed inside the
    boxes is stated at least twice elsewhere (Sections 6-8 and 11,
    Table 5),
    and at this figure's location -- Section 2.2, before the methodology --
    the split names, TOST and the Huawei trace are not yet defined, so the
    numbers were forward references. The claim-limiting qualifiers they
    carried move to the caption, not out of the paper.

    Sizing: drawn on a full-figure axes (no subplot margins) at exactly
    \\columnwidth minus the 0.02in tight-bbox pad on each side, so
    \\includegraphics[width=\\linewidth] prints it at scale 1.0 and the 7pt
    lettering stays 7pt. Do NOT re-add a width fraction at the include site.
    """
    top = [
        ("Early-life\nonboarding ($\\S$7)", "LEARNING LOWERS\nCOLD STARTS", 0.98),
        ("Active-pool\nsteady window ($\\S$6)", "ROUTE TO\nLOW-COST ARMS", 0.85),
        ("Sparse / saturated\n($\\S$11)", "LIMITED GAINS OR\nHIGH MEMORY COST", 0.72),
    ]
    bot = [
        ("Event-level\ndrift ($\\S$8)", "REFIT LOWERS\nCOLD STARTS"),
        ("Drift guardrails\n($\\S$8)", "DIRECTION-AWARE\nGATES NEEDED"),
    ]
    AR = ["Arial", "Liberation Sans", "DejaVu Sans"]
    FS = 7.0
    fig = plt.figure(figsize=(3.256, 1.38))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    W = 0.32
    for i, (title, body, gray) in enumerate(top):
        x = 0.34 * i
        ax.add_patch(plt.Rectangle((x, 0.555), W, 0.44, fc=str(gray),
                                   ec="black", lw=0.8))
        cx = x + W / 2
        ax.text(cx, 0.967, title, ha="center", va="top",
                fontsize=FS, fontweight="bold", family=AR)
        ax.text(cx, 0.775, body, ha="center", va="top", fontsize=FS, family=AR)
    ax.annotate("", xy=(0.995, 0.505), xytext=(0.005, 0.505),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.5, 0.487, "history / traffic density $\\rightarrow$",
            ha="center", va="top", fontsize=FS, family=AR)
    for j, (title, body) in enumerate(bot):
        x = 0.055 + 0.47 * j
        ax.add_patch(plt.Rectangle((x, 0.018), 0.42, 0.405, fc="0.93",
                                   ec="black", lw=0.8, ls=(0, (3, 2))))
        cx = x + 0.42 / 2
        ax.text(cx, 0.392, title, ha="center", va="top", fontsize=FS,
                fontweight="bold", family=AR)
        body_y = 0.180 if j == 0 else 0.210
        ax.text(cx, body_y, body, ha="center", va="top", fontsize=FS, family=AR)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig01_regime_map_1col")

if __name__ == "__main__":
    raise SystemExit("Historical helpers only. Run ../make_figures.py; drift assets are excluded.")
