# WINTER

Code for **WINTER: Adaptive Serverless Prewarming with Limited Invocation History**.

Guntak Kim, Jung JaeHong, and Gu-In Kwon, Inha University.

WINTER combines a source-trained temporal convolutional representation,
prototype initialization, and local ridge adaptation. WINTER-G selects
prototype, adapted, or exponentially weighted moving average predictions using
observed age and invocation count. Operational comparisons use the same
provisioning controller with and without the gate.

This repository contains **code and documentation**. Measured results,
checkpoints, trace-derived arrays, figures, and the submission documents are
distributed in the companion data archive. The manuscript's archive identifier
is [10.5281/zenodo.21754571](https://doi.org/10.5281/zenodo.21754571).
The DOI files are prepared separately for publication after final confirmation.

## Version and experiment scope

Release version: **2026.09.16.1**, corresponding to reviewed manuscript build
`run-c555y8ho`.

The new-observation EWMA weight is fixed at **α = 0.3** in every evaluated
component. The current campaign covers three initial cohorts, one continuous
48-hour replay, six workload-shift conditions, nine active-function pools,
additional controller and representation controls, and a five-arm live replay.
The initial, continuous, and shift comparisons use the onboarding update
procedure. Active-pool diagnostics use the separate steady-window procedure.

## Install and check the code

The recorded environment uses Python 3.13. Create an environment and install
the verification dependencies:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r env/requirements.txt
python tools/reproduce.py checksums
python tools/reproduce.py test
```

The pinned Torch version is sufficient for CPU checks. The original GPU
measurements used the CUDA 12.8 wheel, recorded as `torch 2.10.0+cu128`.
Additional preprocessing and training dependencies are declared in
`serverless-fewshot/pyproject.toml`.

## Restore the companion data

The code ZIP and data TAR.GZ both contain a top-level `winter-serverless/`
directory. Extract them into the same parent directory, or extract the data
archive next to an existing clone named `winter-serverless`.

```bash
# Run from the directory containing the repository.
tar -xzf winter-data-20260916.tar.gz
cd winter-serverless
python tools/reproduce.py checksums --data
python tools/reproduce.py verify --statistics
python tools/reproduce.py paper
```

Verification recomputes reported statistics from saved measurements; it does
not rerun all simulations. The `paper` command builds the frozen flat sources
in a new output directory. Further commands regenerate figures or perform a
fresh replay into a new directory. See [reproduction instructions](docs/REPRODUCTION.md).

## Layout

| Path | Purpose |
|---|---|
| `serverless-fewshot/src/` | Models, adaptation, features, baselines, and simulation support |
| `serverless-fewshot/experiments/fixed_ewma/` | Current fixed-alpha campaign and verification |
| `serverless-fewshot/experiments/lstm_comparison/` | Common-request simulator, LSTM, tests, and source campaign |
| `serverless-fewshot/experiments/winter_drift/` | Matched ungated WINTER shift measurements |
| `serverless-fewshot/experiments/winter_g_drift/` | Onboarding forecasts and gate shift protocol |
| `serverless-fewshot/scripts/`, `configs/`, `testbed/` | Training, preprocessing, and testbed implementation |
| `winter-paper/` | Table/figure generation and numerical audit code; paper/data files arrive with the DOI archive |
| `tools/reproduce.py` | Portable release entry points |
| `docs/` | Protocol map, data provenance, and reproduction instructions |

The authoritative current result directory, after restoring the data, is
`serverless-fewshot/results/fixed_ewma_v1/`. Other result directories preserve
source measurements and attribution controls; their results must not be pooled
with the final campaign. Shared source modules and recorded simulator versions
are retained where the current measurements depend on them. Unused experiments
and paper assets are excluded. See [release scope](docs/SCOPE.md).

## Citation and licenses

Use [CITATION.cff](CITATION.cff) for the manuscript and authors. The manuscript
is prepared for submission to *Future Generation Computer Systems*.

Author-written code uses the existing [MIT license](LICENSE). Measurements and
generated assets in the DOI package use [LICENSE-DATA](LICENSE-DATA).
Public trace derivatives retain their upstream attribution and license terms;
see [data provenance](docs/DATA.md) and [third-party notices](THIRD_PARTY.md).
