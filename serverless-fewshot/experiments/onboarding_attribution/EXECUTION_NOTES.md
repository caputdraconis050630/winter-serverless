# Execution and Analysis Notes

The versioned protocol records the cohorts, policy grid, model seeds, DES seeds,
budget multipliers and nine primary contrasts before the new evaluation runs.
This is a local frozen analysis protocol, not an external preregistration, and
the underlying trace cohorts had already been examined in earlier work.

The following implementation refinements were made during execution:

- NPZ cohort arrays are eagerly loaded once, avoiding repeated decompression.
- Exact action caching and a tested no-expiry certificate avoid identical DES
  executions. The heap implementation remains a cross-check, not the main kernel.
- The finite candidate grid is screened with exact nonnegative partial-sum
  bounds. Deferred functions are evaluated in descending native calibration
  cold-start contribution; changing their execution order does not change the
  optimization objective or omit workloads.
- Independent rho shards use separate writable caches. The final merge accepts
  only completed shard score files and their pruning certificates.
- Integer cold-count sums over the five calibration seeds determine cold-count
  ties. Conservative numerical tolerance retains borderline pruning candidates.
- Cached policy results are accepted only when q, TTL and DES seed arrays match
  the requested execution. Representation selections are persisted before their
  evaluation runs.
- B2f intentionally permits TTL zero. A dedicated oracle test checks that this
  removes idle retention without terminating busy work.
- A simulation-contract guard was added at the final audit, using the unchanged
  tested kernel hash. Future kernel changes cannot reuse this version's metrics
  merely by rerunning unit tests; they require a new output version.

Core policy evaluation and representation evaluation were allowed to run while
the conservative-policy grid was still being screened. Intermediate core
results were inspected for correctness. No evaluation results were used to
change the candidate family, lambda grid, resource budgets, selected policies,
cohort partition or primary hypothesis family.

Supplementary diagnostics were expanded during implementation: conditional
action-swap contrasts against EWMA, separate model-seed contrasts,
drain-inclusive resource ratios, and confidence bands at the four cumulative
time checkpoints. These are explicitly secondary, not new primary hypotheses.
Their marginal 95% intervals do not imply multiplicity-adjusted confirmation.

Source files, model artifacts, cohort artifacts, selected configurations and
final effect tables are hashed in the final verification products. Historical
results and the running drift campaign are not overwritten or stopped.
