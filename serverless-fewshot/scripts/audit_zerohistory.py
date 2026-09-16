#!/usr/bin/env python3
"""Audit: what the gate's zero-history arm actually counts (main 3.4, 5).

The v3/v4/age-qualified gates route a function-minute to the prototype arm
when the function's cumulative invocation count *inside the evaluation
window* is zero (revision_a2_des.py:60-62 builds `cum_prev` from a cumsum
over the sliced window, and `zero_mask = cum_prev == 0`).

For S1 and S2 the window is the whole 2021 trace, so "zero history" and "no
invocation on record" coincide. For S3 the window starts at
`s3_test_t_start` (day 10) and the counter restarts there, so a function
that was active during the training days re-enters the trace as
zero-history. This script quantifies that, so the supplementary can state
the size of the effect from an archive rather than from an assertion.

Nothing here re-runs the DES: every policy faced the identical reset, so no
CSR/WM comparison depends on the labelling. Output:
results/runs/audit_zerohistory.json
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROJECT_ROOT / "results" / "runs" / "audit_zerohistory.json"


def zero_prefix_mask(counts):
    """Ticks strictly before the function's first invocation in the window."""
    cum = np.cumsum(counts, axis=1)
    cum_prev = np.concatenate([np.zeros((counts.shape[0], 1)), cum[:, :-1]],
                              axis=1)
    return cum_prev == 0


def main():
    counts = np.load(PROC / "counts.npy").astype(np.float64)
    splits = np.load(PROC / "splits.npz")
    t0 = int(splits["s3_test_t_start"][0])

    out = {
        "definition": ("zero-history = cumulative invocations inside the "
                       "evaluation window == 0 (revision_a2_des.py:60-62)"),
        "source": "data/processed/{counts.npy,splits.npz}",
        "s3_test_t_start": t0,
        "by_split": {},
    }

    for split, key in [("S1", "s1_test"), ("S2", "s2_test"), ("S3", "s3_test")]:
        idx = splits[key]
        full = counts[idx]
        window = full[:, t0:] if split == "S3" else full
        zm = zero_prefix_mask(window)
        prefix = zm.sum(axis=1)
        rec = {
            "n_functions": int(len(idx)),
            "window_ticks": int(window.shape[1]),
            "window_starts_at_trace_start": split != "S3",
            "zero_history_share": float(zm.mean()),
            "prefix_minutes_median": float(np.median(prefix)),
            "prefix_minutes_mean": float(prefix.mean()),
            # Composition: how much of the zero-history mass belongs to
            # functions whose in-window silence meets the natural cohort's
            # three-day idle-prefix criterion (main 4.1). On S1/S2 this is
            # the defense against reading the share as a window artifact.
            "share_from_ge3day_idle_prefix": float(
                prefix[prefix >= 3 * 1440].sum() / prefix.sum()),
            "n_functions_ge3day_prefix": int((prefix >= 3 * 1440).sum()),
        }
        if split == "S3":
            pre = full[:, :t0].sum(axis=1)
            mass = zm.sum(axis=1)
            rec["functions_with_pre_window_invocations"] = int((pre > 0).sum())
            rec["zero_history_minutes_from_those_functions"] = float(
                mass[pre > 0].sum() / mass.sum())
            rec["zero_history_share_no_earlier_record"] = float(
                mass[pre == 0].sum() / zm.size)
        out["by_split"][split] = rec

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
