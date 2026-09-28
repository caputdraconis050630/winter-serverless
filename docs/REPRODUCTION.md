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
| `python tools/verify_core_evidence.py` | Recheck the three 497-candidate controller selections and the published representation-control selection and intervals from retained measurements |
| `python tools/reproduce.py paper` | Build the restored IEEE Access main text and supplement in a new `.repro-output/paper-*` directory |
| `python tools/reproduce.py figures` | Regenerate current figures/tables in a new copy of the paper directory |
| `python tools/reproduce.py replay --case initial_azure_primary --workers 4 --output /path/to/new-replay` | Execute the complete fixed policy arrays on regenerated common request streams, writing new results |

`checksums --data`, `verify`, `verify --statistics`, `figures`, and `replay`
require the companion data archive from
[Zenodo](https://doi.org/10.5281/zenodo.21754571). First download all 67 data
files, `data-bundle.json` and `restore-data.py`, and run `python3 restore-data.py`
in the download directory. Two volumes are supplied whole; six are transported
as 65 binary parts of at most 8 MB. The helper verifies the input parts and
reconstructed volume hashes using only the Python standard library.
Then extract every
`winter-data-core-ieee-access-20260928-*.tar.gz` beside the matching code ZIP
or GitHub snapshot. Each volume is independently extractable; all are needed
for the complete workspace, and they must not be concatenated. The bundle
preserves selected `2026.09.17` experimental files and supplies the current
manuscript under `submission/`. `CORE_DATA_SCOPE.md` and `data-inventory.json`
describe the exact retained scope. The earlier 18.1 GB archive and its binary
`.partNNN` chunks are superseded and are not needed.
The manuscript build requires that restored source package, Python 3,
`pdflatex`, `bibtex`, and the standard packages of TeX Live. Its journal class,
bibliography style, fonts, and logos retain their own licenses.

## Manuscript build

The current submission source is the complete `ieee-access-20260922` package,
restored under `submission/` by the current data archive.
Keep `authors/`, the figures, generated tables, class/style files, font files
(`.pfb`, `.tfm`, `.map`, `.fd`), and `build.py` together. Run:

```bash
python tools/reproduce.py paper
```

The entry point copies the package to a fresh `.repro-output/paper-*` directory
and runs its `build.py` there. The build refreshes main/supplement cross-references
and checks the LaTeX logs before returning the two PDFs. It writes `build.json`
with the source hashes, journal format, page counts, and PDF hashes. The input
package is left intact. The supplied PDFs have 13 and 18 pages respectively;
pagination can vary with the TeX installation. The local validation on
2026-09-26 produced 13 and 17 pages, with no unresolved references, citations,
or missing-character errors.

The standalone `winter-manuscript-ieee-access-20260926.zip` contains the same
source package under `ieee-access-20260922/`. It can be built using
`python tools/reproduce.py paper --source /path/to/ieee-access-20260922`.
The manuscript PDFs and corresponding LaTeX `pdfauthor` fields name the three
human authors. This metadata correction preserves all 31 supplied pages;
the manuscript's AI-assistance acknowledgment remains part of its text.

The old `2026.09.17` data archive remains usable with these tools for historical
reproduction. Its earlier document format is identified as `archived-cas` in
`build.json`; it is superseded by the current IEEE Access manuscript package.

## Data layout

The data archive restores these paths below `winter-serverless/`:

```text
serverless-fewshot/data/processed/       # partition/identity metadata only
serverless-fewshot/results/fixed_ewma_v1/
serverless-fewshot/results/lstm_comparison_v1/
serverless-fewshot/results/winter_drift_v1/
serverless-fewshot/results/winter_g_drift_v1/  # prototype parameters
serverless-fewshot/results/onboarding_attribution_v1/  # selected models/provenance
serverless-fewshot/results/runs/
serverless-fewshot/results_azure2021/
winter-paper/
submission/
provenance/
validation/
```

`fixed_ewma_v1/cases/` is the final 19-condition dataset. Its `case.json`
identifies inputs and policies; `aggregate.npz` retains weighted function-level
and seed-level outcomes; `summary.json` contains derived metrics. The composed
final records identify their original action-level measurement sources.
`fixed_ewma_v1/replay/` and the source campaign directories retain the original
aggregates and contracts cited by the final action-level composition. The
large intermediate `functions/` and `seed_functions/` files are omitted;
fresh replay can regenerate them. Final aggregates retain function-level
seed-averaged outcomes, seed-level totals, timelines, identities, and weights.
`fixed_ewma_v1/controls/` retains selection evidence and raw per-function,
per-seed outcomes for the reported main, strict-start, and representation
comparisons. `strict_round/live/` contains the actual live records.

Original JSON provenance strings and frozen execution contracts retain the
measurement-time `/data/260715/` prefix. Their suffix after that prefix maps
to the corresponding release-relative path. They are not requirements to
install the artifact at that location. The supported commands above resolve
their inputs relative to the repository and leave recorded results intact.

## Full execution

Fresh replay reproduces simulation from frozen policy actions; it does not
retrain WINTER or LSTM. Source training and trace preprocessing code are also
included, with checkpoints and source partition metadata in the core data
bundle. Provider datasets and full processed Azure-2021/2019/Huawei arrays
are excluded. The selected final replay inputs are only 16.4 MB; these frozen
experiment slices are necessary to run the existing replay command on exactly
the same cohorts. Reconstructing all cohorts and training inputs from provider
originals requires the public traces listed in `DATA.md` and adapting the
source scripts. That complete pipeline has not been rerun or validated as a
single portable command. Chronos model weights are obtained separately from
their original release; measured fixed policy arrays are retained.

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
