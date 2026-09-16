# Symmetric controller calibration (v2)

This is a follow-up to the completed attribution v1, designed after its results
were inspected. It does not replace or mutate the original simulator, results,
paper, or running drift campaign. It is not a blind or externally preregistered
study. No new body training or future-aware routing is introduced.

## Fixed Plan

1. Hash v1 inputs; freeze `protocol.json` before new controller evaluation.
2. Build a pooled prior on exactly the same source-support union as C16.
3. Retain the existing component, gate, native no-prior, and six representations
   with their v1 budget-1 calibration-selected lambdas.
4. Apply all 180 common q/TTL transformations to each of ten forecast families.
5. Select each family's minimum calibration cold count under six common absolute
   WM limits. Retain the previous exhaustive 3,701-candidate simple-policy winner.
6. Freeze all three rho selections before any new test-set simulation.
7. Replay three cohorts with 20 common-event seeds, original and strict starts.
8. Report paired cold effects, actual WM, modeled cost, and drain sensitivity;
   verify conservation, baseline reproduction, pruning, and original input hashes.

Random policies are averaged as outcomes, not predictions. Component/no-prior/
pooled-prior use native lambda and identical body. `trained_lambda` versus five
random bodies controls the representation's calibration stage; its lambda is
frozen from v1, not jointly reselected with the controller. This is controller
symmetry, not exhaustive global hyperparameter optimization.

Pooled/C16 priors share the source-row union; individual head sample sizes and
effective shrinkage still differ. This tests prior construction, not routing
alone. See `INTERPRETATION.md` for the limits of each contrast.

The six primary tests compare component versus strong simple at rho 10, budget
1, across three cohorts and two starts (Holm6). Nine secondary mechanistic tests
use main start, rho 10, budget 1: component/no-prior, component/pooled-prior, and
trained-lambda/mean-five-random, across the three cohorts (Holm9).

Calibration uses every function. Cheap-function screening and exact nonnegative
partial-sum bounds can prove a candidate cannot improve any budget. This is not
cohort subsampling. All proofs and fully computed candidate totals are retained.

## Run

From `/data/260715/serverless-fewshot`:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/data/260715/site-packages-des:/data/260715/site-packages \
  python3.13 experiments/onboarding_symmetric/run_all.py
```

Outputs: `results/onboarding_symmetric_v2/`. Every stage is resumable and logs
its completion status. A failure stops the pipeline. The evaluation lock hashes
all fixed selected policies. Existing v1 event tapes and model caches are read.

## Scope and Stop Rules

The priority is adjudicating existing forecasts under symmetric control, not
searching new architectures until a favorable test result appears. No parameter
changes based on new evaluation output are permitted. Workload selection and
historical inspection remain limitations. Shared calibration WM budgets are not
hard runtime quotas and do not imply equal evaluation WM.

The 48-hour handoff, mature traffic, and drift campaigns are not revalidated by
these four-hour replays. Paper integration depends on their accounting audit if
their claims are retained. A weak or adverse result is reported, not replaced by
a newly favorable regime or threshold. A report does not certify submission
readiness when dependent paper results remain unverified.
