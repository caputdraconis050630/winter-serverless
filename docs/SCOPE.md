# Artifact scope

Repository version `2026.09.26` supports the IEEE Access manuscript package
`ieee-access-20260922`, supplied as a 13-page main text and an 18-page
supplement. The main text uses the IEEE Access class and IEEE bibliography
style. The repository metadata follows that manuscript's title, author names,
and journal. The manuscript package
includes its AI-assistance acknowledgment.

The experiment code and measured outcomes retain the `2026.09.17` snapshot.
All ten figure PDFs, twenty generated table fragments, and the bibliography in
the IEEE Access package match that snapshot. The companion research artifact
is identified by [Zenodo DOI 10.5281/zenodo.21754571](https://doi.org/10.5281/zenodo.21754571).
The `2026.09.26` data archive restores the current IEEE Access source package
and its supplied PDFs under `submission/`. The PDF author metadata is corrected
to identify the three human authors, preserving all supplied page contents.
The canonical `winter-paper/` sources are derived from the same package for
numerical auditing and figure regeneration. The code repository contains the
build entry point and instructions; manuscript assets and experimental data
are distributed in the companion archive.

| Included component | Role |
|---|---|
| `fixed_ewma_v1/cases/` | The 19 final condition/action sets, with EWMA alpha 0.3 |
| `fixed_ewma_v1/replay/`, `controls/`, `strict_round/` | Revised-policy ledgers, reported control studies and final live replay |
| Selected LSTM, WINTER-drift and WINTER-G-drift source records | Original measurements underlying final action-level composition |
| Onboarding-attribution cohorts, models and source ledgers | Source of shared cohorts, prototypes, representation controls and reused outcomes |
| Selected `results/runs/` records | Shared fixed inputs, source policy schedules, Chronos forecasts, prior alpha selection and latency calibration |
| Source preprocessing/training code, selected checkpoints and processed Azure-2021 data | Reconstructing the learned predictors and selected inputs |
| Current paper sources and evidence inputs | Ten referenced figures and twenty referenced generated table fragments |
| `submission/` in the current data archive | Complete IEEE Access manuscript, supplement, author images, class/font assets and `build.py` |
| Standalone manuscript ZIP | The same source package under `ieee-access-20260922/`, also usable with `paper --source` |
| `provenance/` | Original measurement metadata, the six-case correction ledger, and manuscript source/metadata correspondence |
| `validation/` | Fresh artifact checks and their precise scope |

Separate R30–R33 campaigns, the onboarding-symmetric study, unreferenced paper
figures/tables, unused model variants and unrelated historical result trees
are excluded. File selection follows current entry points, their imports and
concrete measurement-source references rather than copying whole result trees.
Every data file has an inclusion reason in `data-inventory.json`.

Some source files contain multiple policies in one hashed array or record.
Those files are retained intact to verify the original measurements. Their
older policy names do not add comparison arms to the paper. The authoritative
arms and original measurement sources are listed in each final `case.json`.
Final WINTER/WINTER-G drift descriptions use the measured onboarding update;
older source contracts are retained as historical records and linked through
the metadata correction ledger.

Raw provider downloads, complete Azure-2019/Huawei preprocessing universes and
Chronos model weights must be obtained from their upstream sources for a full
rebuild from raw data. Frozen inputs support the documented verification and
policy replay without those downloads.
