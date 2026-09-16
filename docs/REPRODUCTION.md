# Reproduction

## Supported entry points

Run these commands from the repository root with Python 3.13 and the packages
in `env/requirements.txt`. Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1` for the recorded numerical-check configuration; the entry
point sets these defaults when they are absent.

| Command | Inputs and behavior |
|---|---|
| `python tools/reproduce.py checksums` | Verify the code snapshot; no results needed |
| `python tools/reproduce.py test` | Simulator, model, causal-input, optimized-engine, and event-stream tests; no archived outcomes needed |
| `python tools/reproduce.py checksums --data` | Verify every restored code/data file against the release manifests |
| `python tools/reproduce.py verify` | Verify all 19 final cohort/action sets and every included generated table |
| `python tools/reproduce.py verify --statistics` | Also reaggregate outcomes, recompute 3,666 paired contrasts and their cluster intervals, and check control/live/timing records |
| `python tools/reproduce.py paper` | Compile the restored flat sources into a new `.repro-output/paper-*` directory |
| `python tools/reproduce.py figures` | Regenerate current figures/tables in a new copy of the paper directory |
| `python tools/reproduce.py replay --case initial_azure_primary --workers 4 --output /path/to/new-replay` | Execute the complete fixed policy arrays on regenerated common request streams, writing new results |

The last six commands require the matching DOI data archive. The document
build additionally requires `pdflatex`, `bibtex`, and the standard packages of
TeX Live. The included journal class and bibliography styles retain their own
licenses. Builds may paginate differently with a different TeX installation;
the frozen submission PDFs and packaging validation record identify the
reviewed output.

## Data layout

The data archive restores these paths below `winter-serverless/`:

```text
serverless-fewshot/data/processed/
serverless-fewshot/results/fixed_ewma_v1/
serverless-fewshot/results/lstm_comparison_v1/
serverless-fewshot/results/winter_drift_v1/
serverless-fewshot/results/winter_g_drift_v1/
serverless-fewshot/results/onboarding_attribution_v1/
serverless-fewshot/results/runs/
serverless-fewshot/results_azure2021/
winter-paper/
submission/
provenance/
```

`fixed_ewma_v1/cases/` is the final 19-condition dataset. Its `case.json`
identifies inputs and policies; `aggregate.npz` retains weighted function-level
and seed-level outcomes; `summary.json` contains derived metrics. The composed
final records identify their original action-level measurement sources.
`fixed_ewma_v1/replay/` contains changed-policy function/seed ledgers, and the
source campaign directories retain invariant-policy and matched-WINTER ledgers.
`fixed_ewma_v1/controls/` contains validation selection, evaluation, and
representation controls. `strict_round/live/` contains the final live records.

Original JSON provenance strings and frozen execution contracts retain the
measurement-time `/data/260715/` prefix. Their suffix after that prefix maps
to the corresponding release-relative path. They are not requirements to
install the artifact at that location. The supported commands above resolve
their inputs relative to the repository and leave recorded results intact.

## Full execution

Fresh replay reproduces simulation from frozen policy actions; it does not
retrain WINTER or LSTM. Source training and trace preprocessing code are also
included, with checkpoints, source splits and processed Azure-2021 inputs in
the data archive. Reconstructing all cohorts from provider originals requires
the public traces listed in `DATA.md`. Chronos model weights are obtained
separately from their original release.

Physical Knative timing depends on the cluster and runtime. The live scripts
create and remove experiment services and require a configured test cluster;
the verifier does not invoke them. The archive preserves the actual request,
readiness, response and predictor-timing records for the reported measurements.

Retained source campaign scripts include local machine paths. Consult
`IMPLEMENTATION.md` before adapting them to a new machine. Use new output
directories for new experiments. The supported release commands use relative
paths and the selected data inventory.

## Corrected release metadata

Six final drift `case.json` descriptions inherited the source campaign's older
scheduled/frozen adapter labels. The release descriptions now identify the
onboarding update used by the actual WINTER and WINTER-G action arrays. The
enclosing `execution.json` case hashes are updated accordingly. All measured
arrays, action schedules, summaries and measurement-time verification reports
retain their original bytes.

`provenance/metadata-corrections.json` maps original and release hashes. The
twelve original case/contract files are retained under
`provenance/metadata-before-correction/`. `verify` checks the exact permitted
description changes, the original report hashes and the new case hashes before
checking the outcomes. Historical source/replay contracts remain unchanged;
their policy arrays and action-level source links identify the measured methods.
`tools/release_metadata.py` records and verifies this export correction; it is
idempotent on the corrected release. Original measurement code is preserved.
