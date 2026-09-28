# Data and provenance

The companion results archive is on
[Zenodo, DOI 10.5281/zenodo.21754571](https://doi.org/10.5281/zenodo.21754571),
separately from GitHub code. The `2026.09.28` core code/data archives use the unchanged
`2026.09.17` experimental snapshot for the IEEE Access manuscript.
The final experimental record is `results/fixed_ewma_v1/`, with EWMA's
new-observation weight fixed to 0.3 throughout. The other included campaigns
provide the source-model, fixed-action, invariant-outcome, calibration, and
control-study dependencies of that record.

The core bundle includes fixed cohort arrays, action schedules, source
checkpoints, function-level outcomes averaged over seeds, seed-level totals,
paired statistics, selected control-study function/seed outcomes, live request
records, table/figure data, and the current IEEE Access manuscript.
It omits the much larger intermediate per-minute function/seed ledgers,
cached feature/rate matrices, and evaluations outside the reported controls.
Its source package is `ieee-access-20260922`, restored under `submission/`; see
[manuscript build instructions](REPRODUCTION.md#manuscript-build).
The archive materializes workspace symlinks as ordinary files. Temporary checkpoints, caches, Python
environments, raw provider archives, and earlier paper build directories are
excluded.

## Trace sources

- Microsoft Azure Functions 2019: Shahrad et al., *Serverless in the Wild:
  Characterizing and Optimizing the Serverless Workload at a Large Cloud
  Provider*, USENIX ATC 2020.
- Microsoft Azure Functions Invocation Trace 2021: Zhang et al., *Faster and
  Cheaper Serverless Computing on Harvested Resources*, SOSP 2021.
  Release: <https://github.com/Azure/AzurePublicDataset>.
- Huawei SIR Lab 2023: Joosen et al., *How Does It Function? Characterizing
  Long-term Trends in Production Serverless Workloads*, ACM SoCC 2023.
  Release: <https://github.com/sir-lab/data-release>.

Upstream license/attribution records are supplied in `licenses/`. Derived
arrays preserve the upstream CC BY 4.0 attribution requirements. Preprocessing
selects functions and windows, transforms counts, constructs causal features,
and assigns duration parameters as described in the manuscript and source code.

Provider-distributed datasets and complete processed training/evaluation
universes for all three trace sources are excluded. Only source split/function
identities, the selected final replay slices (16.4 MB), and control-study
cohorts are retained. They fix the exact function selection, request counts,
duration parameters and grouping used in this study; they are not a mirror of
the external datasets. The frozen arrays support the documented verification
and fixed-policy replay commands without a provider download.

## Manifests

`MANIFEST-code.sha256` covers the code release. `MANIFEST-data.sha256` covers
every companion payload file. `data-inventory.json` identifies the data file
sizes, hashes and inclusion reasons without machine-specific source paths.
Historical contracts and verification reports may refer to omitted
intermediate files or full training inputs. They record the original run;
`MANIFEST-data.sha256` alone defines the current payload. Measurement-time
provenance is retained. The six corrected final drift
descriptions have an explicit original-to-release hash ledger; see
`REPRODUCTION.md`. `docs/SCOPE.md` explains the selected artifact boundaries.
