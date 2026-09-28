# WINTER

Code for **WINTER: Adaptive Serverless Prewarming with Limited Invocation History**.

Guntak Kim, Jaehong Jung, and Gu-In Kwon, Inha University.

Manuscript prepared for **IEEE Access**, source revision
`ieee-access-20260922`.

WINTER combines a source-trained temporal convolutional representation,
prototype initialization, and local ridge adaptation. WINTER-G selects
prototype, adapted, or exponentially weighted moving average predictions using
observed age and invocation count. Operational comparisons use the same
provisioning controller with and without the gate.

This repository contains **code and documentation**. The companion archive for
measured results, checkpoints, trace-derived arrays, figures, and archived
submission documents is on **Zenodo**:
[10.5281/zenodo.21754571](https://doi.org/10.5281/zenodo.21754571).
This DOI identifies the research artifact; the IEEE Access manuscript is a
submission manuscript.

## Version and experiment scope

Repository version: **2026.09.28**, aligned with the IEEE Access manuscript
package `ieee-access-20260922` (main text: 13 pages; supplement: 18 pages).
The experimental snapshot remains **2026.09.17**. The `2026.09.28` core data
bundle retains the inputs and evidence needed for this manuscript, omitting
provider datasets, full preprocessing arrays, cached representations, and
redundant detailed simulation ledgers. The measured arrays and manuscript
contents retain their original bytes.
PDF author metadata identifies Guntak Kim, Jaehong Jung, and Gu-In Kwon;
the supplied page content is preserved.

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

Obtain `winter-code-ieee-access-20260928.zip` and **all**
`winter-data-core-ieee-access-20260928-*.tar.gz` files from the
[Zenodo record](https://doi.org/10.5281/zenodo.21754571). Each data volume is an
independent TAR.GZ with disjoint files under `winter-serverless/`; do not
concatenate them. Extract the code ZIP and every data volume into the same
parent directory. `data-bundle.json` identifies the complete volume set, and
the record's `SHA256SUMS.txt` verifies downloaded files.

```bash
# Run from the directory containing the repository.
for archive in winter-data-core-ieee-access-20260928-*.tar.gz; do
  tar -xzf "$archive"
done
cd winter-serverless
python tools/reproduce.py checksums --data
python tools/reproduce.py verify --statistics
python tools/verify_core_evidence.py
```

Verification recomputes reported statistics from saved measurements; it does
not rerun all simulations. Further commands regenerate figures or perform a
fresh replay into a new directory. See [reproduction instructions](docs/REPRODUCTION.md).

## Build the IEEE Access manuscript

The current data archive restores the complete `ieee-access-20260922` source
package under `submission/`, including `authors/`, the class and font files,
figures, and `build.py`:

```bash
python tools/reproduce.py paper
```

The command builds a copy under `.repro-output/paper-*`, refreshes cross-document
references, and produces `main.pdf`, `supplementary.pdf`, and `build.json`.
The standalone `winter-manuscript-ieee-access-20260926.zip` contains the same
manuscript package. To build it separately, use
`python tools/reproduce.py paper --source /path/to/ieee-access-20260922`.
See [manuscript build instructions](docs/REPRODUCTION.md#manuscript-build).

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
| `submission/` | Current IEEE Access sources and supplied PDFs, restored from the data archive |
| `validation/` | Checks of the packaged measurements, generated assets, and manuscript; restored from the data archive |
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
is prepared for submission to *IEEE Access*. Cite the
[Zenodo DOI](https://doi.org/10.5281/zenodo.21754571) when referring to the
archived research artifact.

Author-written code uses the existing [MIT license](LICENSE). Measurements and
generated assets in the DOI package use [LICENSE-DATA](LICENSE-DATA).
Public trace derivatives retain their upstream attribution and license terms;
see [data provenance](docs/DATA.md) and [third-party notices](THIRD_PARTY.md).
