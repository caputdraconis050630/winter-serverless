# Fifer-inspired LSTM comparison

This package adds a source-trained recurrent prewarmer to the WINTER paper's
initial, continuous, active-pool, drift and live comparisons. It preserves the
retained cohorts and WINTER policy variants, while replaying every compared
policy with common exogenous requests and prospective lifetime accounting.

## Model and literature fidelity

Primary paper: Gunasekaran et al., *Fifer: Tackling Resource Underutilization
in the Serverless Era*, Middleware 2020, DOI
[10.1145/3423211.3425683](https://doi.org/10.1145/3423211.3425683).
[Author manuscript](https://arxiv.org/abs/2008.12819).

Implemented components: forecast future peak demand from a recent history,
create predicted excess capacity, react to unpredicted requests, and remove
idle capacity after ten minutes. The original twenty five-second bins and
ten-second monitoring become twenty minute bins and minute decisions because
the public traces have minute resolution; the ten-minute peak horizon remains.
Independent functions use batch size one. This does not reproduce the original
DAG stage partitioning, batch-size search, bin packing or Brigade deployment.

`model.py` registers our one-layer 64-unit LSTM, scalar readout, log transforms,
Adam settings, source split and stopping rules. Fifer's published prototype
(Section 5.1) reports two layers, 32 neurons, 100 epochs and training batch
size one in Keras/TensorFlow. Our architecture and training settings are
adaptations, not unspecified details of that prototype. The published paper is
available from the [first author's site](https://jashwantraj92.github.io/assets/files/middleware%2720.pdf).
Source-only validation chooses the epoch;
target outcomes do not select the architecture, epoch, policy constants or seed.
The primary model seed is zero. Additional source checkpoints are retained as
training records, not counted as independent target-policy repetitions.

- `LSTM_Fifer`: ceil(predicted ten-minute peak), cap 200, idle timeout 10 minutes.
- `LSTM_shared`: identical peak forecast through the common quantile/lifetime
  controller. This is a controller ablation, not a second trained model.
- S2 source: initial and 48-hour comparisons; S1 source: all active pools and live.
- Drift: S1 for Azure 2021; S2 for Azure 2019 and Huawei, matching the retained
  source-checkpoint protocol. The scheduled ridge adapter reads completed bins.

## Environment

Executed with Python 3.13, PyTorch 2.10.0+cu128, NumPy 2.4.6 and Numba 0.66.0.
The workspace also supplies pandas, SciPy, Matplotlib and scikit-learn.
Source training and forecast export use an RTX 4090. The DES uses CPU workers.

From `/data/260715/serverless-fewshot`, prefix commands with:

```bash
env PYTHONPATH=/data/260715/site-packages-des:/data/260715/site-packages OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3.13
```

The commands below abbreviate that executable as `python3.13`. Inputs are the
processed Azure/Huawei releases and retained policy caches in this experiment
workspace. Their exact paths, function IDs, action arrays and hashes are recorded
in each case. `common.py` is the single location for workspace and output paths.

## Reproduction order

1. Train source models (skip when the verified checkpoints are already present):

   ```bash
   python3.13 experiments/lstm_comparison/model.py --split s1 --seed 0
   python3.13 experiments/lstm_comparison/model.py --split s2 --seed 0
   ```

2. Export immutable cohorts, forecasts and actions:

   ```bash
   python3.13 experiments/lstm_comparison/cases.py initial --cohort azure_primary
   python3.13 experiments/lstm_comparison/cases.py initial --cohort huawei
   python3.13 experiments/lstm_comparison/cases.py initial --cohort azure_evaluation
   python3.13 experiments/lstm_comparison/cases.py continuous
   python3.13 experiments/lstm_comparison/steady.py all
   python3.13 experiments/lstm_comparison/drift.py all
   ```

3. Replay each of the nineteen case directories listed in `verify_results.EXPECTED`:

   ```bash
   python3.13 experiments/lstm_comparison/run.py initial_azure_primary --workers 4
   python3.13 experiments/lstm_comparison/run_seeds.py steady_azure2019_S1 --workers 5
   ```

   `run.py` is the reference minute ledger. `run_seeds.py` schedules by
   function/seed, checkpoints long jobs, and uses the equivalent optimized
   completion-order engine. Either produces the same aggregate schema. Keep an
   existing case's execution contract when resuming it; a changed runner is not
   silently accepted. The initial comparisons have 20 DES seeds; the other DES
   comparisons have three. These are request-randomness seeds, not model seeds.
   Native policies with identical schedules across cost ratios execute once
   per function/seed and are counted once in physical execution totals.

4. Measure model operations with `benchmark.py` on an otherwise idle measurement
   CPU/GPU. Five warm-ups and thirty repetitions compare equal batch sizes.
   The recorded benchmark excludes feature construction, transfer, controller,
   ridge fitting and full-service overhead.

5. On the configured Knative testbed, `live.py` is the development actuation
   replay. `live_strict.py` waits for its completion and service cleanup, then
   executes the primary 60-minute five-arm replay with per-request HTTP status.
   The primary files are `strict_round/live/`. Success requires HTTP 2xx; every
   attempt is retained in CSR. This is precomputed one-pod schedule actuation,
   not deployment of an online predictor/adapter. All five arms share one node.
   Only experiment-created services in `winter-lstm-eval-v2` are removed.

6. Verify measurements and generate the paper:

   ```bash
   python3.13 experiments/lstm_comparison/check_tests.py
   python3.13 experiments/lstm_comparison/verify_results.py
   ```

   From `winter-paper`:

   ```bash
   python3.13 audit/build_lstm_revision.py
   python3.13 audit/update_highlights_docx.py
   python3.13 audit/check_manuscript.py
   python3.13 audit/build_paper.py --build-dir build-lstm
   python3.13 audit/inspect_pdfs.py
   ```

   `--partial` on the result verifier or figure builder permits incremental
   inspection and explicitly records missing cases; it does not certify the
   complete suite. Main PDFs should be delivered only after final validation.

## Measurement contract and artifacts

Outputs are under `results/lstm_comparison_v1/`.

- `models/<split>_seed<seed>/`: frozen source protocol, checkpoint, learning curve.
- `cases/<name>/case.json`: exact cohort, horizon, model source, seeds, cost grid,
  source-cache links and hashes of all policy actions.
- `data.npz`: counts, duration parameters, function/application IDs and weights.
- `execution.json`: simulator/runner/event-generator hashes and accounting rules.
- `functions/`: per-function seed totals and minute cold/idle curves.
- `seed_functions/`, `partial_states/`: completed seed jobs and resumable states
  for high-volume cases; original engine contracts are retained during migration.
- `aggregate.npz`, `summary.json`: weighted/paired aggregates, all windows, AL,
  and 2,000-replicate cluster intervals.
- `verification/`: independent reconstruction checks, source hashes and test logs.
- `strict_round/live/`: primary schedules, service manifest, individual requests,
  response status, ready-pod timeline and aggregate outcomes.

Arrivals, service durations, reactive initialization and proactive initialization
use separate stable semantic streams. Requests are identical across policies and
chunk sizes. Initialization, execution and ready-idle time occupy distinct
wall-clock phases. Main endpoints exclude the separately recorded terminal
drain. Execution demand is conserved across policies over the horizon **plus**
drain; cold initialization can move execution across the horizon boundary.

CSR is 100 times cold requests divided by invocations. Memory is idle GB seconds
per 1,000 invocations. Modeled cost is `(idle + 15*rho*cold)/invocations*1000`.
AL uses centered 15-minute CSR, a policy's own final-quarter mean, and the first
complete 31-minute interval at or below 1.1 times that mean. It is retrospective;
zero AL is possible for a consistently poor policy.

Confidence intervals resample seed-averaged application clusters; Huawei uses
function clusters because application IDs are absent. Azure 2019 active-pool
weights are the retained inverse inclusion probabilities. All synthetic/natural
windows of the same function/application remain in one cluster. The legacy
aggregate schema calls this field `application`; use the recorded IDs and
provider-specific unit described here when interpreting it.

`superseded_precheck` cohorts, integer-overflow action backups, and live capacity
preflights are preserved for provenance and excluded from the final suite. An
extreme-rate guard clips rates before an archived int32 Poisson conversion; at
that bound the target already equals cap 200 for every evaluated cost ratio.
The new run does not revise older archived measurements in place.
