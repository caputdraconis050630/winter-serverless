# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""M1 rerun with the 2021-trained S2 checkpoints (fixes F01).

Why this exists
---------------
revision_m1_kproto.py resolves checkpoints through
`phase63_onboarding_drift.RUNS_DIR`, which is hardcoded to results/runs.
On 7/16 the 2019 recovery chain (run_recovery.sh L40/L52/L56, run with
SF_DATA_DIR=processed_2019) overwrote results/runs/best_anil_ridge_s2_s{0,1,2}.pt
with 2019-trained bodies.  Every M1 run after that date therefore evaluated
the S2 cluster-hold-out campaigns with a 2019 body on 2021 cohort data:

    results/runs           s2_s0 ff148276… s2_s1 53f8cefe… s2_s2 9a6be99c…
    results_azure2021/runs s2_s0 a2aad046… s2_s1 98bbef63… s2_s2 b9b26913…

That is a cross-trace configuration, not the "S2 hold-out, three model seeds"
(2021 body) setup the manuscript describes.  The sibling campaigns
(revision_m1_kproto_2019.py:55, revision_m1_kproto_huawei.py:58) avoid the
problem by routing through eval_adapt_biased, which honours SF_RUNS_DIR;
this script is the Azure-2021 equivalent of that fix.  S1 is unaffected --
best_anil_ridge_s1_s0.pt is md5-identical in both directories (989b0ef6…) --
so the S1 and S1_val campaigns double as a regression check: they must
reproduce revision_m1_kproto_v2.json exactly.

What it runs
------------
The v2 campaign set (S1, S2_m0, S2_m1, S2_m2, S1_val) with checkpoints loaded
from results_azure2021/runs.  Everything else -- cohorts, prototypes, DES
seeds, rho grid, contrast list, bootstrap seed and the order in which the
shared RNG is consumed -- is inherited verbatim from revision_m1_kproto so the
only admissible source of difference is the S2 model body.

Writes results/runs/revision_m1_kproto_v3.json.  Touches no existing archive.
"""

import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.revision_m1_kproto as m1                        # noqa: E402
from scripts.revision_m1_kproto import (                       # noqa: E402
    CAMPAIGNS, VAL_CAMPAIGN, RHOS, KS, BOOT, BOOT_SEED,
    run_campaign, paired_stats, holm,
)
from scripts.phase63_onboarding_drift import (                 # noqa: E402
    PROCESSED_DIR, DES_SEEDS, ONBOARD_WINDOW,
)

# The archived (contaminated) checkpoint directory stays the output location;
# only the *load* path moves to the 2021 training archive.
ARCHIVE_RUNS = PROJECT_ROOT / "results" / "runs"
CKPT_DIR = PROJECT_ROOT / "results_azure2021" / "runs"

CONTRASTS = [("k1", "noproto", "prior_at_all"),
             ("k16", "k1", "clustering_k16_vs_k1"),
             ("k4", "k1", "clustering_k4_vs_k1"),
             ("k64", "k1", "clustering_k64_vs_k1"),
             ("k16", "k4", "granularity_16_vs_4"),
             ("k64", "k16", "granularity_64_vs_16"),
             ("k16", "noproto", "total_prototype_effect")]


def main():
    t0 = time.time()

    # Redirect checkpoint loading inside run_campaign (it resolves the
    # module-global RUNS_DIR at call time) to the 2021 training archive.
    for tag, ckpt, _, _ in list(CAMPAIGNS) + [VAL_CAMPAIGN[:4]]:
        assert (CKPT_DIR / ckpt).exists(), f"missing clean checkpoint {ckpt}"
    m1.RUNS_DIR = CKPT_DIR
    print(f"checkpoints <- {CKPT_DIR}")
    print(f"data        <- {PROCESSED_DIR}")

    features = np.load(PROCESSED_DIR / "features.npy")
    counts = np.load(PROCESSED_DIR / "counts.npy")
    splits = np.load(PROCESSED_DIR / "splits.npz")
    dur_df = pd.read_csv(PROCESSED_DIR / "duration_stats.csv")

    import hashlib
    ckpt_md5 = {}
    for _, ckpt, _, _ in CAMPAIGNS:
        ckpt_md5[ckpt] = hashlib.md5(
            open(CKPT_DIR / ckpt, "rb").read()).hexdigest()

    out = {
        "config": "M1 decision-level prototype-granularity sweep "
                  "(v3: 2021-trained S2 checkpoints)",
        "supersedes": "revision_m1_kproto_v2.json",
        "fix": "S2_m{0,1,2} in v1/v2 loaded results/runs/best_anil_ridge_s2_s*.pt, "
               "which the 7/16 2019 recovery chain had overwritten with "
               "2019-trained bodies; this run loads them from "
               "results_azure2021/runs",
        "checkpoint_dir": str(CKPT_DIR),
        "checkpoint_md5": ckpt_md5,
        "constants": {"ks": KS, "rhos": RHOS, "des_seeds": DES_SEEDS,
                      "onboard_window_min": ONBOARD_WINDOW,
                      "bootstrap_resamples": BOOT, "bootstrap_seed": BOOT_SEED,
                      "reassignment": "per-tick nearest centroid (as archived)",
                      "note_val": (
                          "S1_val cohort = s1_test + s1_val; s1_val informed "
                          "early stopping, so this campaign is a power "
                          "sensitivity, not a primary result")},
        "campaigns": {},
    }

    campaigns = list(CAMPAIGNS) + [VAL_CAMPAIGN]
    for tag, ckpt, tr, co in campaigns:
        out["campaigns"][tag] = run_campaign(tag, ckpt, tr, co, features,
                                             counts, splits, dur_df)

    # ---- paired contrasts, per campaign (RNG order identical to v2) ----
    rng = np.random.default_rng(BOOT_SEED)
    out["stats"] = {}
    for tag, camp in out["campaigns"].items():
        st = {}
        for rho in RHOS:
            rd = camp["by_rho"][str(rho)]
            row = {}
            for a, b, name in CONTRASTS:
                row[name] = paired_stats(rd[a]["func_cold60"],
                                         rd[b]["func_cold60"], rng)
            st[str(rho)] = row
        for _, _, name in CONTRASTS:
            ps = [st[str(r)][name]["wilcoxon_p"] for r in RHOS]
            for r, v in zip(RHOS, holm(ps)):
                st[str(r)][name]["holm_p_across_rho"] = v
        out["stats"][tag] = st

    # ---- pooled over unique functions (S2_m0 preferred, S1-only added) ----
    s1, s2 = out["campaigns"]["S1"], out["campaigns"]["S2_m0"]
    s2_ids = set(s2["func_ids"])
    s1_only = [i for i, fi in enumerate(s1["func_ids"]) if fi not in s2_ids]
    pooled = {"n_s2": len(s2["func_ids"]), "n_s1_only": len(s1_only),
              "n_total": len(s2["func_ids"]) + len(s1_only),
              "note": "S1 functions already in the S2 cohort are taken from "
                      "the S2 campaign only, so no function is counted twice",
              "by_rho": {}}
    for rho in RHOS:
        r1, r2 = s1["by_rho"][str(rho)], s2["by_rho"][str(rho)]
        row = {}
        for a, b, name in CONTRASTS:
            va = np.concatenate([np.array(r2[a]["func_cold60"]),
                                 np.array(r1[a]["func_cold60"])[s1_only]])
            vb = np.concatenate([np.array(r2[b]["func_cold60"]),
                                 np.array(r1[b]["func_cold60"])[s1_only]])
            row[name] = paired_stats(va, vb, rng)
        pooled["by_rho"][str(rho)] = row
    for _, _, name in CONTRASTS:
        ps = [pooled["by_rho"][str(r)][name]["wilcoxon_p"] for r in RHOS]
        for r, v in zip(RHOS, holm(ps)):
            pooled["by_rho"][str(r)][name]["holm_p_across_rho"] = v
    out["pooled_unique"] = pooled

    # ---- replication check vs archived a1 (S1 / k16 == 'nearest') ----
    a1p = ARCHIVE_RUNS / "revision_a1_onboarding.json"
    assert a1p.exists(), f"missing replication anchor {a1p}"
    a1 = json.load(open(a1p))
    rep = {}
    for rho in RHOS:
        new = out["campaigns"]["S1"]["by_rho"][str(rho)]
        old = a1["by_rho"][str(rho)]
        rep[str(rho)] = {
            "k16_vs_archived_nearest_csr": [
                new["k16"]["overall_csr"], old["nearest"]["overall_csr"]],
            "k1_vs_archived_global_csr": [
                new["k1"]["overall_csr"], old["global"]["overall_csr"]],
            "noproto_vs_archived_noproto_csr": [
                new["noproto"]["overall_csr"], old["noproto"]["overall_csr"]],
        }
    out["replication_check"] = rep

    out["wall_sec"] = time.time() - t0
    path = ARCHIVE_RUNS / os.environ.get("SF_M1_OUT", "revision_m1_kproto_v3.json")
    assert not path.exists(), f"refusing to overwrite {path}"
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    blob = open(path, "rb").read()
    assert len(blob) > 0 and blob.count(0) == 0, "corrupt write (sdb guard)"
    json.load(open(path))
    print(f"\nSaved + verified {path}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
