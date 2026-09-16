# Global EWMA alpha revision

This campaign fixes the weight of the new log-count observation to **0.3**.
It preserves source models, cohorts, seeds, requests, cost ratios, controller
rules and the existing gate. Original result directories remain immutable.

## Scope

- Three initial cohorts, 20 request seeds, rho 1/10/100.
- One 48-hour cohort, three seeds, rho 1/10/100.
- Six natural/injected shift cases, three seeds, rho 1/10, replayed from the
  start of the 24-hour burn-in through the four-hour post-shift interval.
- Nine active pools, three seeds, rho 0.1/1/10/100.
- Source-validation reselection of the conservative controller family, with
  alpha fixed to 0.3: 497 candidates per rho, five validation seeds. Evaluation
  retains 20 seeds, the three cohorts, budget grid, strict-start controls and
  representation controls from the earlier attribution study.
- A fresh five-arm, one-hour Knative actuation replay. Both independent EWMA
  and the count-gated learned schedule use alpha 0.3. All response attempts,
  HTTP status and readiness intervals are retained.

The active-pool WINTER startup previously used alpha **0.15**, in addition to
the gate's separate EWMA 0.1. Both become 0.3. Its startup emission order is
retained, including the documented repeated incorporation of tick-zero count.
The learned rates after the first refit remain unchanged. This revision does
not retune the count/age gate or retrain the body or LSTM.

## Execution

From the project root, use Python 3.13 with:

```bash
export PYTHONPATH=/data/260715/site-packages-des:/data/260715/site-packages
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3.13 experiments/fixed_ewma/campaign.py prepare
python3.13 experiments/fixed_ewma/controls.py prepare
python3.13 experiments/fixed_ewma/test_replay.py
python3.13 experiments/fixed_ewma/live_replay.py
python3.13 experiments/fixed_ewma/run_all.py
python3.13 experiments/fixed_ewma/analyze.py
python3.13 experiments/fixed_ewma/verify_controls.py
python3.13 experiments/fixed_ewma/verify.py
```

The live command may run in a separate process. The orchestrator waits for its
completed result before starting bulk DES, then runs DES and controller
calibration concurrently. Each stage records its return code and elapsed time.
Long simulation jobs checkpoint every five minutes. Resume with the same code
and frozen settings using `resume.py`; it validates all originally frozen code
hashes while allowing later analysis/export tools to exist in this directory.
Do not launch a second orchestrator while one is active.
`runner.py CASE --workers N` runs one prepared case;
`merge.py --case CASE` composes its new and invariant outcomes.

`postprocess.py` waits for all measurements, runs paired analysis and both
verification steps, copies the verified paper-local evidence, and regenerates
the manuscript's figures and tables. Prose review and the coupled PDF build
follow separately. The initial execution plan freezes measurement code;
analysis, verification, and export records additionally identify their own
code hashes.

After the control branch completed, `finish_parallel.py` reassigned its two
workers to the remaining Huawei cases while Azure 2019 S3 retained four.
It paused only the parent orchestrator, resumed it in a `finally` block, and
used the unchanged runner and stage completion records. This kept the original
six-worker budget; `parallel_completion.json` records the execution adjustment.
Once Azure S3 completed, `resume_last_pool.py` continued the final Huawei pool
with six workers from saved checkpoints. Completed jobs and interrupted
temporary writes were preserved; `worker_reallocation.json` records the
checkpoint hashes and the orchestrator's eventual resumption.

## Reuse and verification

The existing independent EWMA 0.3 outcomes are retained and their action arrays
are independently recomputed during verification. The new gate replaces only
the appropriate EWMA-selected actions, after matching the original actions to
EWMA 0.1 on every affected minute. Active-pool startup actions are similarly
checked against the original alpha 0.15 implementation before substitution.

The heavy-case runner may reuse a saved per-function/seed outcome only when
the **entire** q/TTL schedule is byte-identical and the cohort, seed and request
key match. A suffix difference prevents reuse. A test compares this execution
path with a complete fresh simulation. No approximate suffix stitching or
outcome-based early stopping is used.

`replay/` contains the affected arms and their complete ledgers. `cases/`
combines them with verified invariant results, preserving source hashes and
action-level origins. The old scheduled drift configuration is excluded.
`analysis/` contains the paired 2,000-resample cluster contrasts; controller
comparisons retain their original 10,000-resample protocol.

No outcome of this revision is used to change alpha. The earlier alpha sweep
was post hoc, and its validation population is also used for the 48-hour audit;
this revision does not describe that population as an independent validation.
Release packaging and portable commands are described in the repository-root README.
