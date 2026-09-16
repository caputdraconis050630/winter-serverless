# Readiness of the Historical 48-Hour Inputs

This is a dependency audit, not a new performance experiment. Reproduce it with
the environment in README and this command:

```bash
python3.13 experiments/onboarding_symmetric/audit_48h_inputs.py
```

## Available Inputs

Directory: `results/runs/des_jobs_r20_2019_holdout/azure2019_holdout/`.

- `shared.npz`: 1755 x 2880 counts, duration means and standard deviations.
- `B4a_ewma__rho10.npz`: the actual EWMA comparator. `G_WE_A0` is not a
  substitute because the gate retains a zero-history prior branch.
- `G_WE_A720__rho10.npz`: historical final age gate.
- `G_WE_Ainf__rho10.npz`: no-age-handoff comparator.
- `A5_proto__rho10.npz`: always-on historical component.
- Equivalent files exist for rho 1 and rho 100; this audit checked rho 10 only.

`dependency_48h_inputs.json` records hashes and checks. Reconstructed function
order and every count match the original trace; duration summaries also match.
The exclusion union contains 772 functions and the resulting cohort 1755.
The gate action schedules equal no-age-handoff before minute 720 and EWMA from
minute 720 onward. This is an action identity, not an outcome identity: inherited
sandbox state can differ after handoff.

## Compatibility Limit

The saved targets are uncapped desired quantities; the historical simulator
caps the retained pool at 200. Compare executed targets after this cap, not raw
arrays. Even after doing so, the first 240 minutes are not bitwise identical to
the v1/v2 cached onboarding schedules:

| Shared cohort | Arm | Different capped q function-minutes | Total function-minutes | Mean absolute TTL difference, minutes |
|---|---|---:|---:|---:|
| Calibration 625 | Component | 284 | 150000 | 0.022590 |
| Calibration 625 | Gate | 237 | 150000 | 0.006635 |
| Evaluation 970 | Component | 518 | 232800 | 0.022308 |
| Evaluation 970 | Gate | 455 | 232800 | 0.007103 |

The maximum absolute TTL difference is 23.863636 minutes, so a small mean is not
a guarantee that every decision is close. The historical 48h code accumulates
ridge sufficient statistics and solves in float64; the original onboarding
canonical helper solves in float32. Embedding batching also differs. These are
documented implementation differences, not a proof that one isolated factor
explains every action discrepancy. No new cold/WM effect is inferred from these
array comparisons.

## Two Distinct Next Tasks

1. **Correct the historical table's accounting:** preserve these frozen 48h
   action schedules and replay them through a separately versioned prospective
   TTL simulator. This does not require retraining. It still evaluates this
   historical implementation, not the new controller-calibrated wrapper.
2. **Claim one end-to-end policy:** first specify one numerical/update path and
   lifecycle schedule, then validate that path over both short and long
   horizons. If a new wrapper or precision path is chosen, name it and perform
   the corresponding calibration/evaluation; do not silently replace earlier
   results.

The current v1/v2 DES has four-hour accounting boundaries and a fixed drain bin.
Passing 2880 ticks to it unchanged would mislabel post-four-hour memory and
request windows. A generalized implementation needs horizon/bin-boundary
tests, busy work across minutes 720/2880, expiration on a boundary, terminal drain,
request/execution conservation, and reproduction of existing 240-minute
fixtures. Common event generation must use stable function identities, not a
new incidental row order. Benchmark before promising a runtime.

This audit supplies reusable inputs and identifies a compatibility issue; it
does not mark the old 48h CSR/WM/cost figures as verified or submission-ready.
