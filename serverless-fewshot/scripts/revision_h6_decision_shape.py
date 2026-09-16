# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP-H5 diagnostic, part 2: if not scale, then what?

The recalibration test (revision_h5_scale_recal.py) falsified the
under-forecasting hypothesis: the transferred predictor's mean log-residual is
~0 on every pool, and correcting it makes cold starts worse.  So the learned
path's steady-state deficit is not in the *level* of the rate estimate.

The remaining candidate is *shape*: does the estimate still separate the ticks
that receive arrivals from the ticks that do not?  That is answerable directly
from the exported decision matrices -- no model, no GPU, no simulation:

  recall     P(prewarmed | arrival)      -- the quantity CSR is one minus
  precision  P(arrival | prewarmed)      -- what the prewarms were worth
  duty       P(prewarmed)                -- how often it acts at all

A predictor that is unbiased in the mean but over-smoothed shows the signature
of low recall at comparable-or-higher duty: it spends its prewarms on the wrong
ticks.  Comparing the learned arm against EWMA on the same pool separates that
from "the pool is simply unpredictable" (both arms low).

Reads results/runs/split_jobs_recal_*/ ; writes
results/runs/revision_h6_decision_shape.json and
results/tables/T_huawei_decision_shape.csv.
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RHO = 10.0                      # the paper's headline operating point
METHODS = ["B4a_ewma", "A5_full_system", "A5_recalibrated"]


def stats(prewarm, keepalive, counts):
    """Decision quality against the realized arrivals.

    `keepalive` is a dwell time in minutes (clipped to 5..30), not a binary
    readiness flag, so only `prewarm` -- the number of containers the policy
    asks to have ready at tick t -- carries the tick-level decision.  It is
    reported separately as a mean dwell.
    """
    arrival = counts > 0
    act = prewarm > 0
    n_arr = float(arrival.sum())
    n_act = float(act.sum())
    hit = float((act & arrival).sum())
    recall = hit / max(n_arr, 1.0)
    duty = float(act.mean())
    base = float(arrival.mean())
    return {
        "recall": recall,
        "precision": hit / max(n_act, 1.0),
        "duty": duty,
        # how much better than prewarming blindly at the same duty
        "lift": (recall / duty) if duty > 0 else float("nan"),
        "arrival_rate": base,
        "mean_keepalive_min": float(np.asarray(keepalive, dtype=np.float64).mean()),
    }


def main():
    out, lines = {}, ["arm,method,recall,precision,duty,lift,arrival_rate,"
                      "mean_keepalive_min"]
    for jroot in sorted(RUNS_DIR.glob("split_jobs_recal_*")):
        arm = jroot.name.replace("split_jobs_recal_", "")
        sub = [d for d in jroot.iterdir() if (d / "shared.npz").exists()]
        if not sub:
            continue
        jdir = sub[0]
        counts = np.load(jdir / "shared.npz")["counts"]
        out[arm] = {"n_functions": int(counts.shape[0]),
                    "n_ticks": int(counts.shape[1]), "rho": RHO, "methods": {}}
        for m in METHODS:
            f = jdir / f"{m}__rho{RHO}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            s = stats(z["prewarm"], z["keepalive"], counts)
            out[arm]["methods"][m] = s
            lines.append(f"{arm},{m},{s['recall']:.5f},{s['precision']:.5f},"
                         f"{s['duty']:.5f},{s['lift']:.4f},"
                         f"{s['arrival_rate']:.5f},{s['mean_keepalive_min']:.3f}")

    json.dump(out, open(RUNS_DIR / "revision_h6_decision_shape.json", "w"),
              indent=2, default=float)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "T_huawei_decision_shape.csv").write_text("\n".join(lines) + "\n")

    for arm, a in out.items():
        print(f"\n--- {arm} ({a['n_functions']} fns, arrival ticks "
              f"{a['methods'].get('B4a_ewma', {}).get('arrival_rate', float('nan')):.4f}) ---")
        for m, s in a["methods"].items():
            print(f"  {m:<18s} recall {s['recall']:.4f}  precision {s['precision']:.4f}"
                  f"  duty {s['duty']:.4f}  lift {s['lift']:.2f}x"
                  f"  keepalive {s['mean_keepalive_min']:.1f} min")
        e = a["methods"].get("B4a_ewma")
        l = a["methods"].get("A5_full_system")
        if e and l:
            print(f"  -> learned vs EWMA: recall {l['recall']-e['recall']:+.4f}, "
                  f"duty {l['duty']-e['duty']:+.4f}, "
                  f"lift {l['lift']:.2f}x vs {e['lift']:.2f}x")
    print("\nSaved revision_h6_decision_shape.json + T_huawei_decision_shape.csv")


if __name__ == "__main__":
    main()
