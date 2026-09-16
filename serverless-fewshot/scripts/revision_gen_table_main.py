# -*- coding: utf-8 -*-
"""Generate the paper's steady-state table (tab:main) rows from archived
JSONs — every number traceable. 10-seed campaign for headline methods
(A5, gated v3, gated v4, EWMA, B1, Oracle: sim_results_des_revision.json +
revision_e1_v4_2021.json), 10-seed B2f (revision_v1_hybridfull_2021.json),
legacy 3-seed rows for percentile-only B2 and Global B5
(results_azure2021/runs/sim_results_des.json). rho=10. Output: LaTeX rows
to stdout + results/tables/T_main_2021_revised.csv.
"""
import json, sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).parent.parent
RUNS = ROOT / "results" / "runs"

src10 = json.load(open(RUNS / "sim_results_des_revision.json"))["results"]
src10 += json.load(open(RUNS / "revision_e1_v4_2021.json"))["results"]
srcb2f = json.load(open(RUNS / "revision_v1_hybridfull_2021.json"))
leg_path = ROOT / "results_azure2021" / "runs" / "sim_results_des.json"
leg = json.load(open(leg_path))
if isinstance(leg, dict):
    leg = leg.get("results", leg)

def rows_for(src, method, split, rho=10.0):
    return [r for r in src if r["method"] == method and r["split"] == split
            and abs(r["cost_ratio"] - rho) < 1e-9]

ORDER = [
    ("A5_full_system", "\\sys{} (learned path)", src10),
    ("A5_gated_v3", "\\sysG{} v3", src10),
    ("A5_gated_v4", "\\sysG{} v4", src10),
    ("B4a_ewma", "EWMA", src10),
    ("B1_fixed_keepalive", "Keep-alive", src10),
    ("B2f_hybrid_full", "Hybrid histogram (full)", srcb2f),
    ("B2_histogram", "Histogram (pctl-only)$^{\\dagger}$", leg),
    ("B5_global", "Global$^{\\dagger}$", leg),
    ("Oracle", "Oracle", src10),
]

lines = ["split,policy,csr_mean,csr_std,wm,nseeds"]
for split in ["S1", "S2", "S3"]:
    print(f"% ---- {split} ----")
    for method, label, src in ORDER:
        rs = rows_for(src, method, split)
        if not rs:
            print(f"%   {label}: NO DATA")
            continue
        csr = np.array([r["csr"] for r in rs])
        wm = np.mean([r["wm_per_1k_inv"] for r in rs])
        print(f" & {label:34s} & ${csr.mean()*100:.2f}\\pm{csr.std()*100:.2f}$ "
              f"& {wm:,.0f} \\\\  % n={len(rs)} seeds")
        lines.append(f"{split},{method},{csr.mean():.6f},{csr.std():.6f},"
                     f"{wm:.1f},{len(rs)}")
out = ROOT / "results" / "tables" / "T_main_2021_revised.csv"
out.write_text("\n".join(lines) + "\n")
b = out.read_bytes(); assert b and b.count(0) == 0
print(f"% saved {out}")
