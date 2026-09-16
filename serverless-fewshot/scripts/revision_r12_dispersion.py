# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""R12: Poisson vs negative-binomial decision-layer sensitivity
(pre-registered in revision_r12_PREREG.md).

  --s2   (numba env AFTER --export-s2 in torch env is NOT needed: EWMA rates
         are CPU) exports 2019 S2 EWMA NB decisions then prints runner cmd
  --s3   (torch env) 2021 S3 A5+EWMA in-process
  --report joins everything

Order: --s3 (GPU+CPU), --s2 export + des_runner_fast, --report.
"""

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats as sps

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.phase6_des import (  # noqa: E402
    rates_ewma, rates_a5, decisions_from_rates, run_config, COLD_INIT,
)
from src.decision.newsvendor import newsvendor_quantile  # noqa: E402

RUNS = PROJECT_ROOT / "results" / "runs"
RHO = 10.0
OUT = RUNS / "revision_r12_dispersion.json"
PHIS = {"phi2": 2.0, "phi5": 5.0}


def decisions_nb(rates, tau, phi):
    """NB2 gate/sizing; keep-alive from the registered block unchanged."""
    lam = np.maximum(rates, 1e-12)
    ph = np.broadcast_to(phi, lam.shape) if np.ndim(phi) else np.full_like(lam, phi)
    ph = np.maximum(ph, 1.01)
    r = lam / (ph - 1.0)
    p_zero = np.power(r / (r + lam), r)
    pw = (1.0 - p_zero > (1.0 - tau)).astype(np.int32)
    hi = rates > 2.0
    if hi.any():
        pnb = r[hi] / (r[hi] + lam[hi])
        pw[hi] = np.maximum(pw[hi], sps.nbinom.ppf(tau, r[hi], pnb).astype(np.int32))
    _, ka = decisions_from_rates(rates, tau)
    return pw, ka


def s3():
    import pandas as pd
    import torch
    from src.meta.trainer import ANILMetaTrainer
    from src.models.heads import N_QUANTILES
    torch.set_num_threads(1)
    P21 = PROJECT_ROOT / "data" / "processed"
    features = np.load(P21 / "features.npy")
    counts_all = np.load(P21 / "counts.npy")
    splits = np.load(P21 / "splits.npz")
    dur_df = pd.read_csv(P21 / "duration_stats.csv")
    idx = splits["s3_test"]
    t0 = int(splits["s3_test_t_start"][0])
    counts = counts_all[idx][:, t0:].astype(np.float32)
    feats = features[idx][:, t0:, :]
    pre = counts_all[idx][:, :t0].astype(np.float64)
    mean = pre.mean(axis=1)
    var = pre.var(axis=1)
    phi_hat = np.clip(np.where(mean > 0, var / np.maximum(mean, 1e-9), 1.01),
                      1.01, 20.0)[:, None]
    dm = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_mean"] if fi < len(dur_df)
                                 else 1.0 for fi in idx]), nan=1.0)
    ds = np.nan_to_num(np.array([dur_df.iloc[fi]["dur_std"] if fi < len(dur_df)
                                 else 0.5 for fi in idx]), nan=0.5)
    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=features.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1, device="cuda")
    trainer.load(RUNS / "best_anil_ridge_s1_s0.pt")
    a5 = rates_a5(counts, feats, trainer, "cuda")
    ew = rates_ewma(counts)
    tau = newsvendor_quantile(RHO)
    jobs = []
    for arm, rates in [("A5_full_system", a5), ("B4a_ewma", ew)]:
        for tag, phi in list(PHIS.items()) + [("phihat", phi_hat)]:
            pw, ka = decisions_nb(rates, tau, phi)
            jobs.append((f"{arm}__nb_{tag}", RHO, list(range(10)), counts,
                         pw, ka, dm, ds, COLD_INIT, "S3"))
    out = json.load(open(OUT)) if OUT.exists() else {
        "prereg": "revision_r12_PREREG.md", "results": []}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for res in ex.map(run_config, jobs):
            for r in res:
                r.pop("func_csr", None); r.pop("func_cold", None)
                r.pop("func_total", None); r.pop("func_wm", None)
            out["results"].extend(res)
            r0 = res[0]
            print(f"  S3 {r0['method']:26s} csr={np.mean([x['csr'] for x in res])*100:.3f}% "
                  f"wm={np.mean([x['wm_per_1k_inv'] for x in res]):.0f}", flush=True)
            json.dump(out, open(OUT, "w"), default=float)
    print("s3 done")


def s2():
    shared = np.load(RUNS / "des_jobs_2019" / "S2" / "shared.npz")
    counts = shared["counts"].astype(np.float32)
    ew = rates_ewma(counts)
    tau = newsvendor_quantile(RHO)
    jdir = RUNS / "des_jobs_r12" / "S2_2019"
    jdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUNS / "des_jobs_2019" / "S2" / "shared.npz", jdir / "shared.npz")
    meta = []
    for tag, phi in PHIS.items():
        pw, ka = decisions_nb(ew, tau, phi)
        f = f"B4a_ewma__nb_{tag}__rho{RHO}.npz"
        np.savez_compressed(jdir / f, prewarm=pw, keepalive=ka)
        meta.append({"method": f"B4a_ewma__nb_{tag}", "rho": RHO, "file": f})
    json.dump({"split": "S2_2019", "seeds": [0, 1, 2], "cold_init": COLD_INIT,
               "jobs": meta}, open(jdir / "jobs.json", "w"))
    print("exported; run:\n  PYTHONPATH=/data/260715/site-packages-des:. "
          "python3.13 scripts/des_runner_fast.py --jobs results/runs/des_jobs_r12 "
          "--out results/runs/revision_r12_des_s2.json")


def report():
    out = json.load(open(OUT))
    camp = json.load(open(RUNS / "sim_results_des_revision.json"))["results"]
    ref = {}
    for m in ["A5_full_system", "B4a_ewma"]:
        rows = [r for r in camp if r["method"] == m and r["split"] == "S3"
                and r["cost_ratio"] == RHO]
        ref[m] = {"csr": float(np.mean([r["csr"] for r in rows])),
                  "wm": float(np.mean([r["wm_per_1k_inv"] for r in rows]))}
    tab = {}
    for r in out["results"]:
        tab.setdefault(r["method"], []).append(r)
    summary = {"S3_poisson_ref": ref, "S3_nb": {}, "S2_2019_nb": {}}
    for m, rows in tab.items():
        summary["S3_nb"][m] = {"csr": float(np.mean([r["csr"] for r in rows])),
                               "wm": float(np.mean([r["wm_per_1k_inv"] for r in rows]))}
    s2p = RUNS / "revision_r12_des_s2.json"
    if s2p.exists():
        s2res = json.load(open(s2p))
        rows2 = s2res["results"] if isinstance(s2res, dict) else s2res
        camp19 = json.load(open(RUNS / "sim_results_des_2019.json"))
        c19 = camp19["results"] if isinstance(camp19, dict) and "results" in camp19 else camp19
        ewref = [r for r in c19 if r["method"] == "B4a_ewma" and r["split"] == "S2"
                 and r["cost_ratio"] == RHO]
        summary["S2_2019_nb"]["poisson_ref_csr"] = float(np.mean([r["csr"] for r in ewref]))
        for m in {r["method"] for r in rows2}:
            rs = [r for r in rows2 if r["method"] == m]
            summary["S2_2019_nb"][m] = float(np.mean([r["csr"] for r in rs]))
    out["summary"] = summary
    json.dump(out, open(OUT, "w"), default=float)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", action="store_true")
    ap.add_argument("--s2", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.s3:
        s3()
    elif a.s2:
        s2()
    elif a.report:
        report()
    else:
        ap.error("pass --s3 / --s2 / --report")
