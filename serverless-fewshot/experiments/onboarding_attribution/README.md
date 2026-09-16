# Onboarding Attribution Study

This independent study implements the agreed resource-versus-information
attribution plan. It does not modify the historical simulator, manuscript,
archived results, or running drift campaign.

## Execution

Run from `serverless-fewshot` using Python 3.13:

```bash
export PYTHONPATH=/data/260715/site-packages-des:/data/260715/site-packages
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3.13 experiments/onboarding_attribution/run_all.py
```

Results live in `results/onboarding_attribution_v1/`. Stages resume from
completed files; temporary files are never accepted as completed results.
The protocol and original-source/data hashes are in `protocol.json` and
`manifest.json`. The main model is Azure-2021 S2 seed 0, with C=16 source
prototypes. Seeds 1 and 2 are separate robustness arms, not substituted models.

## Experimental Contract

- Azure primary: existing 100 functions, 97 applications.
- Calibration: 625 functions, 426 applications.
- Additional evaluation: 970 functions, 663 applications, disjoint from both
  calibration and the original primary applications.
- Huawei: existing 76 functions; no Huawei parameter selection.
- These are previously inspected traces, not pristine new hold-outs.
- Calibration event seeds 0..4; final evaluation seeds 1000..1019.
- Time bins: [0,1), [1,15), [15,60), [60,240) minutes, then drain.
- Both providers use the reproduced first-arrival-minute cohort clock in this
  study, not a new deployment-presence experiment. The main start permits
  provisioning at that minute boundary; the strict start disables it until
  the following minute.
- Inputs for minute t end at t-1. Ridge fits only earlier targets, each tick.
- All arms share request times, execution durations and semantically indexed
  potential cold-init draws. Common seeds alone would not provide this.
- TTL updates cannot resurrect expired containers or retroactively erase
  allocated memory. Idle, initializing and executing memory are disjoint.
- Cap 200 limits retained containers; overflow requests execute in charged,
  ephemeral containers. This is not a global admission/capacity guarantee.
- On horizon exit, disable new prewarm; charge all pending work and retained
  idle lifetimes to the separate drain interval using the last TTL.

`study.py screen` enumerates the full finite simple-policy family. To avoid
repeating billions of irrelevant events, the default workflow fully screens
low-volume functions, then `prune_screen.py` finishes with exact partial-sum
bounds. A candidate is dropped only if nonnegative remaining cold/memory
cannot improve any registered budget. Every pruning certificate is checked
against the final best upper bounds. This is not subsampling or approximate
simulation. Fully evaluated candidates and lower-bound certificates remain
available. The coarse lower bound on first-arrival cold starts assumes every
potential initial warm container were available; it is never used as a
deployable comparator.

Identical action arrays reuse identical simulations. A second exact reuse
certificate applies only when a five-minute-TTL replay has no expiry and the
alternative policy never requests more than its existing pool. Request paths
then match exactly, and only final drain idle time needs adjustment. An
independent event-queue oracle and an indexed-heap implementation cross-check
the production array kernel; the heap implementation is not the default.

## Interpretation Limits

The budget grid chooses fixed policies on calibration data. It is not an
online hard-budget limiter. A WM-ratio CI within [0.99,1.01] is required for
the approximate matched-memory label on evaluation data. Other points remain
resource trade-offs or directional comparisons. No extrapolated frontier is
used for a headline claim.

The random-body and learned-no-prior controls share the native controller
and the same nine lambda values. Their operating-point coverage is narrower
than the conservative-EWMA search; unmatched cells cannot establish
representation value. The age-profile predictor and fleet-prior EWMA are
simple cross-function-information baselines, not information-free policies.
Meta-learning versus supervised-training superiority is outside this study.
The random bodies retain the same input channels, including Azure application
co-rate. They remove source-trained weights, not every form of cross-function
information. The eligibility rule also conditions on subsequent first-day
activity; results are conditional on these cohorts, not all registrations.

The retrospective bridge can differ slightly from the old GPU run because
the embedding batch layout changes floating-point rounding. This residual is
reported separately from changes to random events and TTL/accounting.
Cost is idle GB s plus the original 15*rho penalty per cold request. Predictor
compute, shared parameters and per-function buffers are not hidden in this
provisioning cost.

## Deliverables

- `bridge.json`: four-stage provenance/accounting comparison.
- `selection.json`: frozen simple-policy choices and validation budgets.
- `representation_selection.json`: all five random bodies and learned control.
- `evaluation/`: function x seed x interval metrics plus actual q/TTL arrays.
- `events/`, `ledgers/`: reproducible event tapes and phase-interval checks.
- `endpoints.csv`, `comparisons.csv`, `time_and_interaction.csv`.
- `figures/`: PDF and PNG plots; four time checkpoints are shown, not a
  continuously measured per-minute curve. Resource points and action-swap bars
  are point estimates; paired uncertainty is in the comparison tables. Time
  bands are marginal application/function-cluster 95% intervals.
- `RESULTS.md` and `RESULTS_KO.md`: scientific interpretation and paper impact.
