# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP-H5 diagnostic, part 3: is the cell-2 split a *provider* boundary or a
*predictability* boundary?

Parts 1-2 established what the Huawei steady-state failure is not: not a scale
mis-calibration (mean log-residual ~0; correcting it hurts) and not a loss of
discrimination (the learned arm's lift over blind prewarming at equal duty is
comparable to or better than EWMA's on every pool, Huawei included).  What does
differ is *duty*: the learned blend crosses the prewarm threshold far less often
than EWMA -- 15-19 % less on the Azure pools, 49 % less on Huawei -- and Huawei's
mixed pool is intrinsically much harder (EWMA lift 2.5x there against 5.7-14.8x
on Azure).

That suggests a single explanation for both providers: the learned path is
uniformly more conservative, and the cost of that conservatism scales with how
unpredictable the pool is.  If so, per-function (learned CSR - EWMA CSR) should
track a per-function predictability statistic along the *same* curve on both
providers, and "Azure vs Huawei" is not the operative variable.

Per-function statistics, all at the rho=10 operating point:
  predictability  EWMA's lift = P(prewarmed | arrival) / P(prewarmed)
  gap             CSR(A5) - CSR(EWMA) in percentage points

Reads the recal-arm DES results and job dirs; writes
results/runs/revision_h7_predictability.json and
results/tables/T_huawei_predictability.csv.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sstats

PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RHO = 10.0
ARMS = ["huawei", "huawei_s1ckpt", "azure2021", "azure2021_s1", "azure2021_s3"]
PROVIDER = {"huawei": "huawei", "huawei_s1ckpt": "huawei",
            "azure2021": "azure", "azure2021_s1": "azure",
            "azure2021_s3": "azure"}


def per_function_csr(rows, method, rho):
    """Seed-averaged per-function CSR for one method at one rho."""
    sel = [r for r in rows if r["method"] == method and r["cost_ratio"] == rho]
    if not sel:
        return None
    cold = np.mean([r["func_cold"] for r in sel], axis=0)
    total = np.mean([r["func_total"] for r in sel], axis=0)
    return cold / np.maximum(total, 1e-9), total


def trace_descriptors(counts, min_per_day=1440):
    """Policy-independent per-function predictability descriptors.

    Registered definitions, copied from phase1c_huawei.descriptors so the axis
    is a property of the arrival process alone -- neither EWMA's decisions nor
    its simulated CSR enter, which is what makes the provider comparison below
    a fair one.
    """
    c = np.asarray(counts, dtype=np.float64)
    n_days = max(c.shape[1] // min_per_day, 1)
    mean = c.mean(axis=1)
    var = c.var(axis=1)
    fano = np.where(mean > 0, var / np.maximum(mean, 1e-12), 0.0)
    F = np.abs(np.fft.rfft(c, axis=1)) ** 2
    tot_e = F[:, 1:].sum(axis=1) + 1e-12
    e_daily = F[:, min(n_days, F.shape[1] - 1)] / tot_e
    nz_frac = (c > 0).mean(axis=1)
    iat_cv = np.zeros(c.shape[0])
    for i in range(c.shape[0]):
        nzi = np.flatnonzero(c[i])
        if len(nzi) > 2:
            iats = np.diff(nzi)
            m = iats.mean()
            iat_cv[i] = iats.std() / m if m > 0 else 0.0
    return {"fano": fano, "e_daily": e_daily, "nz_frac": nz_frac,
            "iat_cv": iat_cv}


def per_function_lift(prewarm, counts):
    """EWMA's per-function discrimination: recall / duty."""
    arrival = counts > 0
    act = prewarm > 0
    n_arr = arrival.sum(axis=1).astype(np.float64)
    n_act = act.sum(axis=1).astype(np.float64)
    hit = (act & arrival).sum(axis=1).astype(np.float64)
    T = counts.shape[1]
    recall = np.where(n_arr > 0, hit / np.maximum(n_arr, 1), np.nan)
    duty = n_act / T
    return np.where(duty > 0, recall / np.maximum(duty, 1e-12), np.nan)


def main():
    recs = []
    for arm in ARMS:
        rp = RUNS_DIR / f"revision_h5_recal_{arm}.json"
        jroot = RUNS_DIR / f"split_jobs_recal_{arm}"
        if not rp.exists() or not jroot.exists():
            print(f"  missing artefacts for {arm} -- skipped")
            continue
        rows = json.load(open(rp))
        jdir = [d for d in jroot.iterdir() if (d / "shared.npz").exists()][0]
        counts = np.load(jdir / "shared.npz")["counts"]
        pw_ewma = np.load(jdir / f"B4a_ewma__rho{RHO}.npz")["prewarm"]

        a5, total = per_function_csr(rows, "A5_full_system", RHO)
        ew, _ = per_function_csr(rows, "B4a_ewma", RHO)
        lift = per_function_lift(pw_ewma, counts)
        desc = trace_descriptors(counts)
        gap = (a5 - ew) * 100.0

        for i in range(len(gap)):
            if np.isfinite(lift[i]) and total[i] > 0:
                recs.append({"arm": arm, "provider": PROVIDER[arm],
                             "fn": i, "lift": float(lift[i]),
                             # primary predictability axis: how well the best
                             # simple predictor does on this function (low =
                             # predictable).  Better behaved than lift, which
                             # degenerates to 0 whenever EWMA catches nothing.
                             "ewma_csr_pct": float(ew[i] * 100.0),
                             "e_daily": float(desc["e_daily"][i]),
                             "fano": float(desc["fano"][i]),
                             "iat_cv": float(desc["iat_cv"][i]),
                             "nz_frac": float(desc["nz_frac"][i]),
                             "gap_pp": float(gap[i]),
                             "invocations": float(total[i])})
        print(f"  {arm:<15s} {len(gap)} functions, median EWMA CSR "
              f"{np.nanmedian(ew)*100:.2f}%, median EWMA lift "
              f"{np.nanmedian(lift):.2f}x, median gap {np.nanmedian(gap):+.3f} pp")

    if not recs:
        raise SystemExit("no arms available")

    lift = np.array([r["lift"] for r in recs])
    gap = np.array([r["gap_pp"] for r in recs])
    prov = np.array([r["provider"] for r in recs])
    axes = {k: np.array([r[k] for r in recs])
            for k in ("e_daily", "fano", "iat_cv", "nz_frac", "ewma_csr_pct")}
    # Primary axis must be policy-independent.  `ewma_csr_pct` is reported for
    # completeness but is NOT usable as the axis: gap = CSR(A5) - CSR(EWMA)
    # shares its -CSR(EWMA) term, which induces correlation by construction.
    xpred = axes["e_daily"]

    out = {"rho": RHO, "n_functions": len(recs), "arms": ARMS,
           "predictability_axis": "daily spectral energy fraction "
                                  "(higher = more periodic = more predictable)",
           "axis_caveat": "ewma_csr_pct is confounded with the gap by "
                          "construction and is reported, not used",
           "by_provider": {}, "quantile_bins": []}

    for name, x in axes.items():
        r, pv = sstats.spearmanr(x, gap)
        out[f"spearman_all_{name}"] = {"rho": float(r), "p": float(pv)}
    for p in ["azure", "huawei"]:
        m = prov == p
        if m.sum() > 5:
            r, pv = sstats.spearmanr(xpred[m], gap[m])
            out["by_provider"][p] = {
                "n": int(m.sum()), "spearman_rho": float(r), "spearman_p": float(pv),
                "median_e_daily": float(np.median(xpred[m])),
                "median_fano": float(np.median(axes["fano"][m])),
                "median_lift": float(np.median(lift[m])),
                "median_gap_pp": float(np.median(gap[m]))}

    # Same-curve test: bin by predictability, compare providers inside each bin.
    # Quantile edges are de-duplicated -- EWMA CSR has heavy ties at 0.
    edges = np.unique(np.nanquantile(xpred, np.linspace(0, 1, 6)))
    nb = len(edges) - 1
    lines = ["bin,e_daily_lo,e_daily_hi,n_azure,n_huawei,median_gap_azure_pp,"
             "median_gap_huawei_pp,mannwhitney_p"]
    for b in range(nb):
        lo, hi = edges[b], edges[b + 1]
        sel = (xpred >= lo) & (xpred <= hi if b == nb - 1 else xpred < hi)
        ga = gap[sel & (prov == "azure")]
        gh = gap[sel & (prov == "huawei")]
        mw = float("nan")
        if len(ga) > 3 and len(gh) > 3:
            mw = float(sstats.mannwhitneyu(ga, gh).pvalue)
        entry = {"bin": b, "e_daily_range": [float(lo), float(hi)],
                 "n_azure": int(len(ga)), "n_huawei": int(len(gh)),
                 "median_gap_azure_pp": float(np.median(ga)) if len(ga) else None,
                 "median_gap_huawei_pp": float(np.median(gh)) if len(gh) else None,
                 "mannwhitney_p": mw}
        out["quantile_bins"].append(entry)
        lines.append(f"{b},{lo:.4f},{hi:.4f},{len(ga)},{len(gh)},"
                     f"{np.median(ga) if len(ga) else float('nan'):.4f},"
                     f"{np.median(gh) if len(gh) else float('nan'):.4f},{mw:.3e}")

    json.dump(out, open(RUNS_DIR / "revision_h7_predictability.json", "w"),
              indent=2, default=float)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "T_huawei_predictability.csv").write_text("\n".join(lines) + "\n")

    print(f"\nSpearman(axis, learned-EWMA gap) over {len(recs)} functions:")
    for k in ("e_daily", "fano", "iat_cv", "nz_frac", "ewma_csr_pct"):
        sa = out[f"spearman_all_{k}"]
        tag = "  [confounded, not used]" if k == "ewma_csr_pct" else ""
        print(f"  {k:<14s} rho={sa['rho']:+.3f} (p={sa['p']:.2e}){tag}")
    for p, v in out["by_provider"].items():
        print(f"  {p:<7s} n={v['n']:<5d} spearman {v['spearman_rho']:+.3f} "
              f"(p={v['spearman_p']:.2e})  median e_daily "
              f"{v['median_e_daily']:.4f}  median fano {v['median_fano']:.1f}  "
              f"median gap {v['median_gap_pp']:+.3f} pp")
    print("\nSame-curve test -- median learned-EWMA gap by predictability bin:")
    for e in out["quantile_bins"]:
        fa = ("%8.3f" % e["median_gap_azure_pp"]) if e["median_gap_azure_pp"] is not None else "     -  "
        fh = ("%8.3f" % e["median_gap_huawei_pp"]) if e["median_gap_huawei_pp"] is not None else "     -  "
        print(f"  e_daily {e['e_daily_range'][0]:7.5f}-{e['e_daily_range'][1]:7.5f}  "
              f"azure n={e['n_azure']:<4d}{fa} pp  |  "
              f"huawei n={e['n_huawei']:<4d}{fh} pp  "
              f"(MW p={e['mannwhitney_p']:.2e})")
    print("\nSaved revision_h7_predictability.json + T_huawei_predictability.csv")


if __name__ == "__main__":
    main()
