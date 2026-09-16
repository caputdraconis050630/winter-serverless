# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP8: Paper-ready figures and tables — every number from measured JSONs.

Sources (all produced by experiments in this repo):
  sim_results_des.json          steady-state DES sweep (per-function arrays)
  steady_state_stats.json       paired statistics + decision gate
  onboarding_drift_results.json onboarding/drift rolling-CSR curves, T5 input
  ablation_results.json         K x head heatmap (measured), scalability
  extended_results.json         B6/B7/B8 baselines
  multiseed_training.json       training-seed robustness
  testbed_cdf.json              Knative real-runtime cold/warm CDFs
  gate_routing_stats.json       hybrid gate routing fractions
No literals: if a source is missing, the artifact is skipped with a warning.
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 8.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

STYLE = {
    "B1_fixed_keepalive": ("#888888", "s", "Fixed keep-alive (10 min)"),
    "B2_histogram": ("#e377c2", "D", "Hybrid histogram"),
    "B4a_ewma": ("#ff7f0e", "^", "EWMA + decision layer"),
    "B5_global": ("#d62728", "x", "Global head (no adapt.)"),
    "A5_full_system": ("#1f77b4", "o", "Ours (meta + per-func ridge)"),
    "A5_gated": ("#17becf", "P", "Ours + hybrid gate"),
    "A5_proto": ("#1f77b4", "o", "Ours (prototype + biased ridge)"),
    "A5_noproto": ("#9467bd", "v", "Ours (no prototype)"),
    "A5_refit": ("#1f77b4", "o", "Ours (trigger refit)"),
    "A5_frozen": ("#9467bd", "v", "Ours (frozen head)"),
    "Oracle": ("#000000", "*", "Oracle"),
}


def load(name):
    p = RUNS_DIR / name
    if not p.exists():
        print(f"  [skip] {name} not found")
        return None
    with open(p) as f:
        return json.load(f)


def sty(m):
    return STYLE.get(m, ("#333333", ".", m))


# ---------------------------------------------------------------- F1
def fig_f1(sim):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    for ax, split in zip(axes, ["S1", "S2", "S3"]):
        methods = sorted(set(r["method"] for r in sim if r["split"] == split))
        for m in methods:
            pts = {}
            for r in sim:
                if r["method"] == m and r["split"] == split:
                    pts.setdefault(r["cost_ratio"], []).append(r)
            rows = sorted(
                ((np.mean([x["wm_per_1k_inv"] for x in v]),
                  np.mean([x["csr"] for x in v]),
                  np.std([x["csr"] for x in v])) for v in pts.values()),
                key=lambda z: z[0])
            wm = [r[0] for r in rows]; cs = [r[1] for r in rows]
            er = [r[2] for r in rows]
            c, mk, lb = sty(m)
            if m in ("B1_fixed_keepalive", "B2_histogram"):
                ax.errorbar(wm[:1], cs[:1], yerr=er[:1], fmt=mk, color=c,
                            label=lb, markersize=9, capsize=2)
            else:
                ax.errorbar(wm, cs, yerr=er, fmt=f"-{mk}", color=c, label=lb,
                            markersize=5, linewidth=1.4, capsize=2, alpha=0.9)
        ax.set_xlabel("Wasted memory (GB·s / 1k inv.)")
        ax.set_title(f"{split}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Cold-start ratio")
    axes[0].legend(loc="upper right")
    fig.suptitle("CSR vs. wasted-memory Pareto frontier (5 seeds, event-level DES)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "F1_pareto_csr_vs_wm.pdf")
    fig.savefig(FIGURES_DIR / "F1_pareto_csr_vs_wm.png")
    plt.close(fig)
    print("  F1 saved")


# ---------------------------------------------------------------- F2
def fig_f2(od):
    onb = od["onboarding"]["by_rho"]
    drift = od["drift"]["drift_curves"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # (a) onboarding rolling CSR at rho=10
    ax = axes[0]
    rho_key = "10.0" if "10.0" in onb else list(onb)[-1]
    for m in ["A5_proto", "A5_noproto", "B4a_ewma", "B1_fixed_keepalive", "Oracle"]:
        if m not in onb[rho_key]:
            continue
        c, mk, lb = sty(m)
        y = np.array([np.nan if v is None else v
                      for v in onb[rho_key][m]["rolling_csr"]], dtype=float)
        ax.plot(np.arange(len(y)), y, color=c, label=lb, linewidth=1.6,
                alpha=0.9)
    ax.set_xlabel("Minutes since onboarding")
    ax.set_ylabel("Rolling CSR (15-min window)")
    ax.set_title(f"(a) Onboarding (n={od['onboarding']['n_functions']} functions, "
                 f"rho={rho_key})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) drift: scale(0.5) shows the clearest post-drift contrast
    # (splice/scale-up make the post-drift regime trivially warm -> all ~0)
    ax = axes[1]
    dkey = "scale(0.5)" if "scale(0.5)" in drift else list(drift)[0]
    rkey = "10.0" if "10.0" in drift[dkey] else list(drift[dkey])[0]
    W = None
    for m in ["A5_refit", "A5_frozen", "B4a_ewma", "B1_fixed_keepalive", "Oracle"]:
        if m not in drift[dkey][rkey]:
            continue
        c, mk, lb = sty(m)
        y = np.array([np.nan if v is None else v
                      for v in drift[dkey][rkey][m]["rolling_csr"]], dtype=float)
        W = len(y)
        x = np.arange(W) - W // 2
        ax.plot(x, y, color=c, label=lb, linewidth=1.6, alpha=0.9)
    if W:
        ax.axvline(0, color="black", linestyle="--", alpha=0.6)
        ax.set_xlim(-60, 120)
        ax.relim(); ax.autoscale_view()
        ymax = ax.get_ylim()[1]
        ax.set_ylim(-0.02 * ymax, ymax)
        ax.annotate("drift", (3, ymax * 0.9), fontsize=9)
    ax.set_xlabel("Minutes relative to drift event")
    ax.set_title(f"(b) Drift: {dkey}, rho={rkey}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "F2_adaptation_curves.pdf")
    fig.savefig(FIGURES_DIR / "F2_adaptation_curves.png")
    plt.close(fig)
    print("  F2 saved")


# ---------------------------------------------------------------- F3
def fig_f3(abl, ext):
    scale = abl.get("scalability", []) if abl else []
    if not scale:
        print("  [skip] F3 (no scalability data)")
        return
    N_values = [r["N"] for r in scale]
    ours_kb = [r["per_func_kb"] for r in scale]
    ours_lat = [r["adapt_ms_per_func"] for r in scale]
    b6 = (ext or {}).get("B6_full_finetune", {})
    b7 = (ext or {}).get("B7_per_func_lstm", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.plot(N_values, ours_kb, "-o", color="#1f77b4", label="Ours (head only)", linewidth=2)
    if b6:
        ax.plot(N_values, [b6["per_func_bytes"] / 1024] * len(N_values), "-x",
                color="#d62728", label="Full fine-tune (B6)", linewidth=1.5)
    if b7:
        ax.plot(N_values, [b7["per_func_bytes"] / 1024] * len(N_values), "-s",
                color="#888888", label="Per-function LSTM (B7)", linewidth=1.5)
    ax.axhline(32, color="black", linestyle=":", alpha=0.5, label="32 KB target")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of functions"); ax.set_ylabel("Per-function state (KB)")
    ax.set_title("(a) Per-function state"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(N_values, ours_lat, "-o", color="#1f77b4",
            label="Ours (batched ridge)", linewidth=2)
    if b7.get("train_time_per_func"):
        ax.plot(N_values, [b7["train_time_per_func"] * 1000] * len(N_values),
                "-s", color="#888888", label="Per-function LSTM (B7, measured)",
                linewidth=1.5)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="1 ms target")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of functions")
    ax.set_ylabel("Adaptation latency (ms/function)")
    ax.set_title("(b) Adaptation latency"); ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "F3_scalability.pdf")
    fig.savefig(FIGURES_DIR / "F3_scalability.png")
    plt.close(fig)
    print("  F3 saved")


# ---------------------------------------------------------------- F4
def fig_f4(sim):
    """Stratified CSR/WM by frequency bucket — real per-function data."""
    counts = np.load(PROCESSED_DIR / "counts.npy")
    splits_data = np.load(PROCESSED_DIR / "splits.npz")
    test_idx = splits_data["s1_test"]
    T = counts.shape[1]
    rate = counts[test_idx].sum(axis=1) / T  # per-minute
    bucket_edges = [(0, 1 / 1440, "<1/day"), (1 / 1440, 1 / 60, "1/day-1/h"),
                    (1 / 60, 1.0, "1/h-1/min"), (1.0, np.inf, ">1/min")]
    bucket_of = np.array([next(i for i, (lo, hi, _) in enumerate(bucket_edges)
                               if lo <= r < hi) for r in rate])

    methods = ["B1_fixed_keepalive", "B4a_ewma", "A5_full_system"]
    rho = 10.0
    bucket_csr = {m: [] for m in methods}
    bucket_wm = {m: [] for m in methods}
    labels = [b[2] for b in bucket_edges]

    for m in methods:
        runs = [r for r in sim if r["method"] == m and r["split"] == "S1"
                and abs(r["cost_ratio"] - rho) < 1e-9]
        fc = np.array([r["func_cold"] for r in runs]).mean(axis=0)
        ft = np.array(runs[0]["func_total"], dtype=float)
        fw = np.array([r["func_wm"] for r in runs]).mean(axis=0)
        for b in range(4):
            mask = bucket_of == b
            tot = ft[mask].sum()
            bucket_csr[m].append(fc[mask].sum() / tot if tot > 0 else np.nan)
            bucket_wm[m].append(fw[mask].sum() / max(tot / 1000, 1e-9)
                                if tot > 0 else np.nan)

    x = np.arange(4); w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, data, ylab, ttl in [
            (axes[0], bucket_csr, "Cold-start ratio", "(a) CSR by frequency bucket"),
            (axes[1], bucket_wm, "Wasted memory (GB·s/1k inv.)",
             "(b) WM by frequency bucket")]:
        for i, m in enumerate(methods):
            c, mk, lb = sty(m)
            ax.bar(x + (i - 1) * w, data[m], w, label=lb, color=c, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12)
        ax.set_ylabel(ylab); ax.set_title(ttl)
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "F4_stratified_gains.pdf")
    fig.savefig(FIGURES_DIR / "F4_stratified_gains.png")
    plt.close(fig)
    print("  F4 saved (measured per-function data)")


# ---------------------------------------------------------------- F5
def fig_f5(abl):
    hm = (abl or {}).get("heatmap_k_head", {})
    if not hm or "measured" not in str(hm.get("source", "")):
        print("  [warn] F5 heatmap not yet measured (run phase8_heads.py); skipping")
        return
    k_values = hm["k_values"]; head_types = hm["head_types"]
    mat = np.array(hm["crps_matrix"], dtype=float)
    lbl = {"ridge": "Ridge (H2)", "anil_gd": "ANIL-GD (H1)",
           "bayesian": "Bayesian (H3)"}
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    im = ax.imshow(np.where(np.isnan(mat), np.nanmax(mat), mat),
                   cmap="YlOrRd_r", aspect="auto")
    ax.set_xticks(range(len(head_types)))
    ax.set_xticklabels([lbl.get(h, h) for h in head_types])
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([str(k) for k in k_values])
    ax.set_xlabel("Head type"); ax.set_ylabel("K (support size)")
    ax.set_title("CRPS by head type and K (measured)")
    for i in range(len(k_values)):
        for j in range(len(head_types)):
            v = mat[i, j]
            txt = "N/A" if np.isnan(v) else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="black")
    fig.colorbar(im, label="CRPS")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "F5_ablation_heatmap.pdf")
    fig.savefig(FIGURES_DIR / "F5_ablation_heatmap.png")
    plt.close(fig)
    print("  F5 saved (measured)")


# ---------------------------------------------------------------- F6
def fig_f6(cdf):
    if not cdf:
        print("  [skip] F6 (testbed CDF pending)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.8))
    colors = {"python-ml": "#1f77b4", "node-api-real": "#ff7f0e",
              "java-svc": "#d62728"}
    for fn, lats in cdf["cold"].items():
        if not lats:
            continue
        s = np.sort(lats); c = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, c, color=colors.get(fn, "#333"), linewidth=2,
                label=f"{fn} cold (n={len(s)})")
    for fn, lats in cdf["warm"].items():
        if not lats:
            continue
        s = np.sort(lats); c = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, c, "--", color=colors.get(fn, "#333"), linewidth=1.2,
                alpha=0.7, label=f"{fn} warm (n={len(s)})")
    ax.set_xscale("log")
    ax.set_xlabel("End-to-end latency (s)"); ax.set_ylabel("CDF")
    ax.set_title("Knative testbed: cold vs. warm latency (real runtimes)")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "F6_testbed_cdf.pdf")
    fig.savefig(FIGURES_DIR / "F6_testbed_cdf.png")
    plt.close(fig)
    print("  F6 saved (measured Knative data)")


# ---------------------------------------------------------------- T3
def table_t3(stats):
    rows = []
    for split, rep in stats.items():
        if split.startswith("_"):
            continue
        for m, agg in rep["aggregate"].items():
            pr = rep["paired"].get(m, {})
            star = ""
            if pr:
                p = pr.get("p_holm", 1.0)
                star = "***" if p < 0.001 else "**" if p < 0.01 else \
                       "*" if p < 0.05 else ""
            rows.append({
                "split": split, "method": m,
                "csr_mean": agg["csr"], "csr_std": agg["csr_std_seeds"],
                "csr_func_ci_lo": agg["csr_func_ci"][0],
                "csr_func_ci_hi": agg["csr_func_ci"][1],
                "wm_mean": agg["wm"], "wm_std": agg["wm_std_seeds"],
                "sig_vs_A5": star,
                "cliffs_delta_vs_A5": pr.get("cliffs_delta"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "T3_main_comparison.csv", index=False)
    print("  T3 saved (CI + significance)")


# ---------------------------------------------------------------- T4
def table_t4(abl, ext, ms):
    scale = (abl or {}).get("scalability", [])
    ours_ms = scale[1]["adapt_ms_per_func"] if len(scale) > 1 else None
    ours_kb = scale[0]["per_func_kb"] if scale else None
    rows = [{
        "Method": "Ours (shared body + ridge head)",
        "Per_Func_KB": ours_kb, "Adapt_ms_per_func": ours_ms,
        "CRPS_S2": (ms or {}).get("crps_mean"),
        "Notes": f"train-seed std {ms['crps_std']:.4f} (n={ms['n_seeds']})" if ms else "",
    }]
    for key, label in [("B6_full_finetune", "B6 full fine-tune"),
                       ("B7_per_func_lstm", "B7 per-function LSTM"),
                       ("B8_full_maml", "B8 full MAML")]:
        b = (ext or {}).get(key, {})
        if b:
            rows.append({
                "Method": label,
                "Per_Func_KB": b["per_func_bytes"] / 1024,
                "Adapt_ms_per_func": (b.get("train_time_per_func", 0) * 1000
                                      or None),
                "CRPS_S2": b.get("crps"),
                "Notes": f"n_eval={b.get('n_eval')}",
            })
    pd.DataFrame(rows).to_csv(TABLES_DIR / "T4_overhead_accounting.csv",
                              index=False)
    print("  T4 saved")


def main():
    sim = load("sim_results_des.json")
    stats = load("steady_state_stats.json")
    od = load("onboarding_drift_results.json")
    abl = load("ablation_results.json")
    ext = load("extended_results.json")
    ms = load("multiseed_training.json")
    cdf = load("testbed_cdf.json")

    print("Figures:")
    if sim: fig_f1(sim)
    if od: fig_f2(od)
    fig_f3(abl, ext)
    if sim: fig_f4(sim)
    fig_f5(abl)
    fig_f6(cdf)

    print("Tables:")
    if stats: table_t3(stats)
    table_t4(abl, ext, ms)
    print("Done.")


if __name__ == "__main__":
    main()
