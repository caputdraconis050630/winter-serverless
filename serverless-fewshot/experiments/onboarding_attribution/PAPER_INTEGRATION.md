# Paper Integration Requirements

The attribution study is a separate, versioned experiment. It does not silently
replace the historical simulator or any manuscript artifact. Read the final
`results/onboarding_attribution_v1/RESULTS_KO.md` before revising claims.

## Evidence Mapping

| Question | Evidence | What it does not establish |
|---|---|---|
| Does increasing TTL explain the original improvement? | Paired 2 x 2 swaps of retained-pool target and TTL, including interactions | A unique percentage attributable to information |
| Can conservative policies reproduce the improvement? | Calibration-selected EWMA boosts and fixed targets, evaluated without retuning | Universal optimality of the finite comparator family |
| Is the learned representation necessary? | Same-feature, same-adapter learned and five random frozen bodies with the same lambda grid | Superiority of meta-learning over supervised training |
| Does a prior explain the entire gain? | Component, no-prior, prototype-only, fleet-prior EWMA, age-profile outcomes | That all forms of transferred information are interchangeable |
| Does advance function registration drive the result? | Common minute-zero no-prewarm condition | Event-level detection and actuation at the exact first request |
| Are costs merely shifted outside the measurement window? | Separate drain intervals and drain-inclusive resource contrasts | Production infrastructure or predictor-compute cost |

## Accounting Correction

The old TTL path recomputes a container deadline from its last completion and
the newly selected TTL before checking expiration. A TTL decrease can erase
already incurred idle time; an increase can preserve a container past a deadline
that should already have expired. The new path first expires old deadlines,
then applies changed TTL prospectively. This changes simulation semantics, not
just the presentation of a metric. The exact bridge separates:

1. Archived results.
2. Original simulator with reconstructed model predictions and original seeds.
3. Common request events with old TTL semantics.
4. The same common events with corrected TTL and interval accounting.

Small original-replay residuals from floating-point inference are reported
separately. Cold-start and memory changes must not all be attributed to TTL
when the event tape or inference execution also changed.

## Affected Manuscript Artifacts

Review these after accepting the new protocol and interpreting its results:

- `winter-paper/sections/onboarding.tex`: headline component/gate effects,
  resource premiums, prior attribution, first-hour evidence and comparator scope.
- `winter-paper/generated/onboarding_main_rows.tex` and
  `winter-paper/generated/onboarding_full_rows.tex`: historical three-seed
  results versus new twenty-seed common-event estimates must be distinguished.
- `winter-paper/fig06_onboarding_cohort_2019.pdf` and
  `winter-paper/fig13_onboarding_timing.pdf`: regenerate only from the chosen
  versioned evidence, with cluster-level uncertainty and correct time resolution.
- `winter-paper/generated/matched_memory_table.tex`: calibration budgets are
  not observed equal-memory points. No extrapolated or unmatched comparison
  should become a matched-memory claim.
- Abstract, introduction, discussion, conclusion and highlights: limit transfer
  information claims to the identified comparisons; report resource trade-offs.
- Simulator specification and reproducibility supplement: document prospective
  deadlines, retained-container cap, overflow, common events and drain accounting.

Other historical campaigns using changing TTLs need a separate code-path audit
and, where affected, corrected replay. This study has not revalidated mature
traffic, 48-hour handoff, Chronos, or drift results. It neither changes nor stops
the ongoing drift campaign. A single accounting correction must not be patched
into old tables without rerunning the corresponding policies.

## Scope of a Defensible Conclusion

Distinguish three statements: shorter or longer retention time, a larger pool of
retained sandboxes, and more informative allocation of that pool. The first two
are actuation mechanisms; source information can influence either. A TTL swap
alone cannot establish information value, and failure to establish information
value does not prove that the representation carries no information.
Random-body controls still receive the same application co-rate channel. Their
comparison cannot establish that every type of cross-function information is
unnecessary. A fleet-prior baseline winning without a trained TCN would support
a simpler transfer mechanism, not an information-free mechanism.

The four-hour gate uses its specified onboarding loop and never reaches the
720-minute handoff. No lifecycle-policy or boundary-optimality claim follows.
Previously inspected trace cohorts remain previously inspected, even though
application partitions and new comparator settings were frozen before evaluation.
