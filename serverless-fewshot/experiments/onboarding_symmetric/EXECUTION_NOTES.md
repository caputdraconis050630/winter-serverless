# Execution notes and pre-evaluation diagnostic addendum

The main protocol was frozen before the new controller screening. Historical
v1 outcomes had already been examined, so this remains an explicitly post-v1
follow-up. No model, grid, calibration objective, cohort, primary test, or
selection rule has been changed based on evaluation output.

## Execution-only refinements

- Independent rho screens/selectors run concurrently with separate caches.
  File locks prevent duplicate selection writers. The main runner waits for
  all three fixed selections before any new test simulation.
- Progress logging and an explicit evaluation-lock check were added during
  calibration. They do not change the simulator or candidate scores.
- Large-function seed evaluations use two threads with ordered result collection;
  a dedicated test checks exact equality to sequential seed reduction. Selectors
  were restarted from completed candidate/action caches to adopt this change.
- NumPy indices in pruning proofs are converted to Python integers for JSON
  serialization. This storage fix does not alter candidate selection.
- A generalized no-expiration certificate was tested separately but had no hits
  for the bottleneck-function probe. It is not used in the production runner;
  the original tested certificate and DES remain unchanged.
- Pooled source reconstruction reproduced C16 prior coefficients exactly. The
  pooled head uses the same union of 76 functions and 1,520 support rows.
- NumPy source memmaps are read-only. The PyTorch conversion warning during
  pooling did not correspond to a write; source hashes are verified afterward.

## Diagnostic addendum, before any new test simulation

Independent controller calibration compares policy families, not a pure prior
substitution at a fixed wrapper. Therefore seven additional evaluation arms are
specified before evaluation:

1. Apply the component's budget1 selected wrapper to native no-prior and pooled
   priors, without reselecting their wrappers.
2. Apply the trained-lambda budget1 selected wrapper to each of five random-body
   policies, using their already frozen v1 lambdas.

These are settings already inside the common grid, not additional search
candidates. Their contrasts are descriptive, outside the unchanged Holm6 and
Holm9 test families. They are evaluated under both starts and every rho. They
control the wrapper hyperparameters, not realized memory or computation; they
do not identify a percentage of gain caused by information. No evaluation
outcomes were available when this addendum was implemented.

Prequential log-count MAE at 1/5/15/60/240 minutes and fixed-wrapper q/TTL
agreement are also descriptive diagnostics from cached forecasts. They do not
change policy choices or the primary/secondary test families. Log-MAE is not a
predictive-distribution score or a substitute for decision/resource outcomes.
The standalone diagnostic is run after the evaluation lock is created.

Numerical verification also checks that allowing 1e-7 GB s of feasibility
rounding does not change any selected setting. Aggregating native rho1 WM in a
different order differed from its reference budget by 1.9e-9 GB s; the selected
controller is unchanged. The strict original budget rule itself is retained.
# Reporting and independent-cache audit

Five analytic reporting fixtures passed: percentage/percentage-point units,
seed averaging, partial windows, terminal idle cost, and Holm ordering. These
are separate from the frozen 27 simulator/controller tests.

The early calibration-only verification found selected complete totals whose
per-function memo entries had not been flushed before the earlier restart.
There were 107 missing action/function entries at rho10 and 140 at rho100,
with none at rho1. The verifier recomputes these on the same five calibration
event tapes and stores them separately under `verification_cache/`; it does
not alter selections, candidate totals, original caches, or action code.
The independently reconstructed totals must match the saved complete totals.
This is an audit-cache recovery, not a new policy selection.
