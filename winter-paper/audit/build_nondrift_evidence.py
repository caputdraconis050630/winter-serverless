"""Reaggregate existing non-drift records with explicit units and provenance."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
EXPERIMENT = Path(os.environ.get("WINTER_EXPERIMENT_ROOT", PAPER.parent / "serverless-fewshot")).resolve()
RUNS = EXPERIMENT / "results/runs"
TABLES = EXPERIMENT / "results/tables"
SOURCES = {}
NBOOT = 10000
SEED = 260907
LABELS = {
    "gated_v3": r"\sysG{}", "A5_proto": r"\sys{} component",
    "B4a_ewma": r"EWMA ($\alpha=0.1$)",
    "EWMA_selected": r"EWMA ($\alpha=0.3$)",
    "B1_fixed_keepalive": "Keep-alive (10 min)",
    "B2f_hybrid_full": "Hybrid histogram (B2f)",
}
ORDER = list(LABELS)


def read_json(path):
    raw = path.read_bytes()
    label = str(path.relative_to(PAPER.parent)) if path.is_relative_to(PAPER.parent) else str(path)
    SOURCES[label] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def ci(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def strict_al(curve):
    """Complete inclusive 30-minute interval; no partial-horizon success."""
    y = np.asarray(curve, dtype=float)
    reference = float(np.nanmean(y[int(len(y) * .75):]))
    for t in range(len(y) - 30):
        window = y[t:t + 31]
        if np.all(np.isfinite(window)) and np.all(window <= 1.1 * reference + 1e-10):
            return t, reference * 100
    return None, reference * 100


def onboarding():
    replay = read_json(HERE / "onboarding_b2f.json")
    alpha = read_json(RUNS / "revision_r22_ewma_alpha_frontier.json")
    output = {}
    flat = []
    for name, filename in [("azure2019", "revision_a4_crosstrace.json"),
                           ("huawei", "revision_h1_huawei_cohort.json")]:
        data = read_json(RUNS / filename)
        inv = np.asarray(replay["cohorts"][name]["invocations_per_function"])
        n = len(inv)
        assert n == data["n_functions"]
        assert inv.sum() == alpha["cohorts"][name]["invocations"]
        samples = np.random.default_rng(SEED).integers(0, n, (NBOOT, n))
        output[name] = {"n_functions": n, "invocations": int(inv.sum()),
                        "prototype_count": len(data["assign_hist_k0"]), "by_rho": {}}
        for rho in (1., 10., 100.):
            key = str(rho)
            records = dict(data["by_rho"][key])
            selected = alpha["selected_alpha"]["by_csr"][key]
            records["EWMA_selected"] = alpha["cohorts"][name]["by_rho"][key]["ewma_alpha"][str(selected)]
            records["B2f_hybrid_full"] = replay["cohorts"][name]["B2f_hybrid_full"]
            base = records["B4a_ewma"]
            basecold = np.asarray(base["func_cold_all"])
            rows = {}
            for arm, rec in records.items():
                if arm.startswith("_"):
                    continue
                cold = np.asarray(rec["func_cold_all"])
                assert len(cold) == n
                np.testing.assert_allclose(cold.sum() / inv.sum(), rec["overall_csr"], atol=1e-12, rtol=0)
                diff = cold - basecold
                csr = 100 * cold.sum() / inv.sum()
                wm = rec["wm_total"] / inv.sum() * 1000
                al, ref = strict_al(rec["rolling_csr"])
                nz = diff[diff != 0]
                row = {
                    "cohort": name, "rho": rho, "arm": arm,
                    "csr_pct": csr,
                    "csr_ci95_pct": ci(100 * cold[samples].sum(axis=1) / inv[samples].sum(axis=1)),
                    "delta_csr_pp": csr - 100 * base["overall_csr"],
                    "delta_cold_per_fn": float(diff.mean()),
                    "delta_cold_ci95": ci(diff[samples].mean(axis=1)),
                    "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) else 1.,
                    "wm_per_1k": wm, "wm_ratio_ewma": rec["wm_total"] / base["wm_total"],
                    "cost_per_1k": wm + 150 * rho * (csr / 1.),
                    "al_min": al, "reference_csr_pct": ref,
                }
                # 1000 * 15 * rho * (CSR percent / 100) = 150 * rho * CSR.
                first_diff = np.asarray(base["func_cold60"]) - np.asarray(rec["func_cold60"])
                total_diff = -diff
                row["avoided_first_hour_per_fn"] = float(first_diff.mean())
                row["avoided_four_hours_per_fn"] = float(total_diff.mean())
                if total_diff.sum() > 0:
                    row["first_hour_share"] = float(first_diff.sum() / total_diff.sum())
                rows[arm] = row
                flat.append(row)
            # Holm adjustment over the three rho values is added below for
            # each cohort/arm family; bootstrap intervals estimate the mean.
            output[name]["by_rho"][key] = rows
        for arm in output[name]["by_rho"]["1.0"]:
            rows = [output[name]["by_rho"][str(r)][arm] for r in (1., 10., 100.)]
            order = np.argsort([r["wilcoxon_p"] for r in rows])
            running = 0.
            for rank, idx in enumerate(order):
                running = max(running, min(1., (3 - rank) * rows[idx]["wilcoxon_p"]))
                rows[idx]["wilcoxon_holm_three_rhos"] = running
    (HERE / "onboarding_metrics.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    with (HERE / "onboarding_metrics.csv").open("w", newline="") as stream:
        fields = list(dict.fromkeys(k for row in flat for k in row))
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(flat)
    return output


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


def table_rows(data, rhos):
    lines = []
    for name, title in [("azure2019", "Azure 2019, 100 functions"), ("huawei", "Huawei, 76 functions")]:
        for rho in rhos:
            lines.append(r"\multicolumn{7}{@{}l}{\emph{" + title + r", $\rho=" + f"{rho:g}" + r"$}} \\")
            for arm in ORDER:
                r = data[name]["by_rho"][str(float(rho))][arm]
                lo, hi = r["delta_cold_ci95"]
                effect = "---" if arm == "B4a_ewma" else f"{r['delta_cold_per_fn']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                lines.append(f"{LABELS[arm]} & {r['csr_pct']:.3f} & {r['delta_csr_pp']:+.3f} & {effect} & {r['wm_per_1k']:,.0f} & {r['wm_ratio_ewma']:.2f} & {r['cost_per_1k']:,.0f}".replace(",", r"{,}") + r" \\")
            lines.append(r"\addlinespace")
    return "\n".join(lines) + "\n"


def crossprovider_tables():
    path = TABLES / "T_r21_reviewer_full_faithful_fullrho.csv"
    SOURCES[str(path.relative_to(PAPER.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = list(csv.DictReader(path.open()))
    full, main = [], []
    for r in rows:
        if r["method"] not in ("A5_faithful", "G_WE_A720"):
            continue
        name = {"h_mixed": "Huawei mixed", "h_sparse": "Huawei sparse", "h_saturated": "Huawei saturated"}.get(r["split"], "Azure 2019 " + r["split"])
        arm = r"\sys{}" if r["method"] == "A5_faithful" else r"\sysG{}"
        basewm = float(r["wm_per_1k_inv"]) - float(r["delta_wm_per_1k_vs_ewma"])
        row = (f"{name} & {float(r['rho']):g} & {arm} & {float(r['csr_pct']):.4f} & "
               f"{float(r['delta_csr_pp_vs_ewma']):+.4f} [{float(r['boot_ci95_lo_pp']):+.4f}, {float(r['boot_ci95_hi_pp']):+.4f}] & "
               f"{float(r['wm_per_1k_inv'])/basewm:.3f} & {float(r['delta_cost_per_1k_vs_ewma']):+,.0f}").replace(",", r"{,}") + r" \\"
        full.append(row)
        if float(r["rho"]) == 10:
            main.append(row)
    (PAPER / "generated/crossprovider_full_rows.tex").write_text("\n".join(full) + "\n")
    (PAPER / "generated/crossprovider_main_rows.tex").write_text("\n".join(main) + "\n")


def timing_and_memory_tables(data):
    alpha = read_json(RUNS / "revision_r22_ewma_alpha_frontier.json")
    alrows, memoryrows, diagnostics = [], [], []
    for name, title in [("azure2019", "Azure 2019"), ("huawei", "Huawei")]:
        for rho in (1., 10., 100.):
            key = str(rho)
            for arm in ("gated_v3", "A5_proto", "B4a_ewma"):
                row = data[name]["by_rho"][key][arm]
                al = "---" if row["al_min"] is None else str(row["al_min"])
                alrows.append(f"{title} & {rho:g} & {LABELS[arm]} & {al} & {row['reference_csr_pct']:.3f}" + r" \\")
            points = sorted((r["wm_per_1k_inv"], r["csr_pct"]) for r in alpha["cohorts"][name]["by_rho"][key]["ewma_alpha"].values())
            frontier = []
            for wm, csr in points:
                if not frontier or csr < frontier[-1][1]:
                    frontier.append((wm, csr))
            for arm in ("gated_v3", "A5_proto"):
                row = data[name]["by_rho"][key][arm]
                target = row["wm_per_1k"]
                delta = None
                if frontier[0][0] <= target <= frontier[-1][0]:
                    baseline = np.interp(target, [p[0] for p in frontier], [p[1] for p in frontier])
                    delta = row["csr_pct"] - baseline
                diagnostics.append({"cohort": name, "rho": rho, "arm": arm,
                                    "wm_target": target, "frontier": frontier,
                                    "interpolated_delta_csr_pp": delta})
                memoryrows.append(f"{title} & {rho:g} & {LABELS[arm]} & {target:,.0f} & "
                                  + ("---" if delta is None else f"{delta:+.3f}") + r" \\")
    def write_table(path, caption, label, spec, heading, rows):
        text = (r"\begin{table}[htbp]" + "\n" + r"\centering\small" + "\n"
                + r"\caption{" + caption + r"}\label{" + label + "}\n"
                + r"\begin{tabular}{@{}" + spec + r"@{}}" + "\n"
                + r"\toprule" + "\n" + heading + r" \\" + "\n" + r"\midrule" + "\n"
                + "\n".join(rows).replace(",", r"{,}") + "\n"
                + r"\bottomrule\end{tabular}\end{table}" + "\n")
        (PAPER / "generated" / path).write_text(text)
    write_table("al_complete_table.tex", "Complete-window AL and policy-specific reference CSR. Reference CSR is percent; AL is minutes. A dash means no complete qualifying interval occurs within four hours.",
                "stab:alcomplete", "lrlrr", r"Cohort & $\rho$ & Arm & AL & Reference CSR", alrows)
    write_table("matched_memory_table.tex", r"Approximate same-memory comparison against the nondominated EWMA-alpha operating points at the same cost ratio. Negative differences favor the transfer arm. A dash means outside the measured frontier span, so no interpolation is reported. WM is GB$\cdot$s/1k.",
                "stab:matchedmemory", "lrlrr", r"Cohort & $\rho$ & Arm & Target WM & $\Delta$CSR (pp)", memoryrows)
    (HERE / "matched_memory.json").write_text(json.dumps(diagnostics, indent=2))


def main():
    if not RUNS.is_dir():
        raise SystemExit("Reaggregation requires WINTER_EXPERIMENT_ROOT with results/runs; archived outputs were not modified.")
    (PAPER / "generated").mkdir(exist_ok=True)
    data = onboarding()
    (PAPER / "generated/onboarding_main_rows.tex").write_text(table_rows(data, [10]))
    (PAPER / "generated/onboarding_full_rows.tex").write_text(table_rows(data, [1, 10, 100]))
    crossprovider_tables()
    timing_and_memory_tables(data)
    (HERE / "nondrift_sources.json").write_text(json.dumps({"bootstrap_resamples": NBOOT, "bootstrap_seed": SEED, "input_sha256": SOURCES}, indent=2))
    for name, d in data.items():
        for rho, rows in d["by_rho"].items():
            for arm in ("gated_v3", "A5_proto", "B2f_hybrid_full"):
                r = rows[arm]
                print(name, rho, arm, "CSR", fmt(r["csr_pct"]), "WM/EWMA", fmt(r["wm_ratio_ewma"]), "cost", fmt(r["cost_per_1k"], 0), "cold/fn CI", r["delta_cold_ci95"], flush=True)


if __name__ == "__main__":
    main()
