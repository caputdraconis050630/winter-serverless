# Artifact scope

Repository version `2026.09.28` supports the IEEE Access manuscript package
`ieee-access-20260922`, supplied as a 13-page main text and an 18-page
supplement. The main text uses the IEEE Access class and IEEE bibliography
style. The repository metadata follows that manuscript's title, author names,
and journal. The manuscript package
includes its AI-assistance acknowledgment.

The experiment code and measured outcomes retain the `2026.09.17` snapshot.
All ten figure PDFs, twenty generated table fragments, and the bibliography in
the IEEE Access package match that snapshot. The companion research artifact
is identified by [Zenodo DOI 10.5281/zenodo.21754571](https://doi.org/10.5281/zenodo.21754571).
The `2026.09.28` core data bundle restores the current IEEE Access source package
and its supplied PDFs under `submission/`. The PDF author metadata is corrected
to identify the three human authors, preserving all supplied page contents.
The canonical `winter-paper/` sources are derived from the same package for
numerical auditing and figure regeneration. The code repository contains the
build entry point and instructions; manuscript assets and experimental data
are distributed in the companion archive.

| Included component | Role |
|---|---|
| `fixed_ewma_v1/cases/` | The 19 final condition/action sets, with EWMA alpha 0.3 |
| `fixed_ewma_v1/replay/`, `controls/`, `strict_round/` | Source aggregates/contracts, reported control-study evidence and final live replay |
| Selected LSTM, WINTER-drift and WINTER-G-drift source records | Original measurements underlying final action-level composition |
| Selected onboarding-attribution model/protocol records | Learned/random representations, prototypes and the original representation-selection budget |
| Selected `results/runs/` records | Prior alpha-selection evidence and measured latency-calibration samples |
| Source preprocessing/training code, checkpoints and source partition metadata | Model/partition provenance; full training arrays must be reconstructed from upstream traces |
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

Raw provider datasets, full processed arrays for all three trace sources,
cached representations and rate matrices, and intermediate per-minute
function/seed ledgers are excluded from the core bundle. Chronos model weights
must be obtained from their upstream source for a full rebuild. Frozen inputs
support the documented verification and policy replay without those downloads.
The reduced archive is distributed in independent TAR.GZ volumes below 100 MB
each. It supersedes the earlier 18.1 GB data archive and its binary parts.
See `CORE_DATA_SCOPE.md` in the restored data for the precise evidence boundary;
historical provenance references are not a promise to include every old file.
