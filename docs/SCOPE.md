# Artifact scope

Version `2026.09.16.1` contains the reviewed manuscript (`run-c555y8ho`), its
reported measurements and the code/data needed to inspect and reproduce them.
The paper text, numerical outcomes and submission PDFs are unchanged from
version `2026.09.16`.

| Included component | Role |
|---|---|
| `fixed_ewma_v1/cases/` | The 19 final condition/action sets, with EWMA alpha 0.3 |
| `fixed_ewma_v1/replay/`, `controls/`, `strict_round/` | Revised-policy ledgers, reported control studies and final live replay |
| Selected LSTM, WINTER-drift and WINTER-G-drift source records | Original measurements underlying final action-level composition |
| Onboarding-attribution cohorts, models and source ledgers | Source of shared cohorts, prototypes, representation controls and reused outcomes |
| Selected `results/runs/` records | Shared fixed inputs, source policy schedules, Chronos forecasts, prior alpha selection and latency calibration |
| Source preprocessing/training code, selected checkpoints and processed Azure-2021 data | Reconstructing the learned predictors and selected inputs |
| Current paper sources and evidence inputs | Ten referenced figures and twenty referenced generated table fragments |
| `submission/` | Flat Elsevier source files, reviewed PDFs and submission support documents |
| `provenance/` | Twelve original metadata files and the exact six-case description correction ledger |

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
