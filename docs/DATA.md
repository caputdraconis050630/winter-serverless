# Data and provenance

Results are distributed in the DOI data archive, separately from GitHub code.
The final experimental record is `results/fixed_ewma_v1/`, with EWMA's
new-observation weight fixed to 0.3 throughout. The other included campaigns
provide the source-model, fixed-action, invariant-outcome, calibration, and
control-study dependencies of that record.

The archive includes fixed cohort arrays, action schedules, source checkpoints,
per-function/per-seed ledgers, aggregate outcomes, paired statistics, live
request records, table/figure data, and the final paper. It materializes
workspace symlinks as ordinary files. Temporary checkpoints, caches, Python
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

Processed Azure-2021 source arrays are included. Complete processed Azure-2019
and Huawei universes and provider-distributed raw archives are not included;
the selected replay arrays are sufficient for the supported verification and
fixed-policy replay commands.

## Manifests

`MANIFEST-code.sha256` covers the code release. `MANIFEST-data.sha256` covers
every companion payload file. `data-inventory.json` identifies the data file
sizes and hashes without machine-specific source paths. Scientific provenance
inside the original experiment manifests is preserved unchanged.
