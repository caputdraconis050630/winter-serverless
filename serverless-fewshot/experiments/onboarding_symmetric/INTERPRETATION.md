# Interpretation Rules

These rules explain the frozen protocol; they do not introduce a selection rule,
new hypothesis family, or retrospective subgroup.

## More Information Is an Opportunity, Not a Model Guarantee

With the same actions, loss, and a free option to ignore extra information, an
optimal decision rule cannot have higher minimum expected loss merely because
it has access to more information. This statement is about the best attainable
rule. It does not establish strict improvement for a finite trained predictor,
a misspecified prior, a different resource allocation, or a particular
heuristic controller. Blackwell's comparison of experiments formalizes the
decision-theoretic distinction between information structures and their value:
[Blackwell theorem proof](https://doi.org/10.1016/j.econlet.2024.112146).

Transfer from a less related source can hurt target performance. This is a
documented possibility, not evidence that it happened in WINTER:
[Wang et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Characterizing_and_Avoiding_Negative_Transfer_CVPR_2019_paper.html).

In this application, an initial function may have no causal signal that
distinguishes its future demand from the fleet. A fleet prior is itself a form
of cross-function information. Beating zero-initialized EWMA is not sufficient
to establish the necessity of learned representations or clustered prototypes.
Conversely, failure to beat a strong simple policy is not proof that the
forecasts contain no information.

## What Each Comparison Can Establish

| Comparison | Supports | Does not establish |
|---|---|---|
| Component vs strong simple, separately calibrated | Relative performance of the evaluated policy pipelines | Pure information effect or equal realized memory |
| Clustered vs pooled prior, same source rows | Incremental value of the evaluated prior construction | Universal superiority of clustering |
| Prior vs no prior, same body | Incremental initialization effect conditional on the trained body | Necessity of meta-training |
| Trained vs random bodies with adaptation | Value beyond these random-feature controls | Superiority over supervised pooled representation learning |
| Fixed-wrapper substitutions | Effect of changing the forecast source with controller settings held fixed | Equal actions or equal memory; changed forecasts can change both |
| Forecast log-count MAE | Predictive accuracy for the stated horizon and weighting | Optimal provisioning or calibrated tail probabilities |
| Original vs strict first minute | Sensitivity to the initial prewarming opportunity | Full reconstruction of deployment-time visibility |
| Four-hour cost plus idle drain | Sensitivity to an artificial no-future-arrivals boundary | Real future production cost |

If q and TTL schedules are both fixed, the simulator cannot detect which
predictor produced them. Information can affect the measured policy outcome
only through actions. The relevant resource question is therefore whether the
predictor leads to better action timing/allocation at comparable cost, not
whether it helps without changing any action.

The pooled/C16 comparison equalizes the union of source support rows, not the
number of rows per fitted head. Pooling changes aggregation and the effective
regularization relative to each head's support size at fixed lambda. It tests
the complete prior-construction choice, not routing in isolation.

Inference conditions on the frozen learned checkpoint and the five fixed
random initializations. Event-seed repetition is not repetition of model
training. Huawei lacks application identifiers here, so its function-level
resampling cannot account for unknown within-application dependence.

## Submission Decision

1. A favorable primary contrast must be reported together with actual WM and
   modeled-cost uncertainty. A common calibration constraint is not a matched
   evaluation-memory design.
2. A favorable system contrast and weak mechanism contrasts support a pipeline
   result, not a claim that clustered meta priors caused the full improvement.
3. A weak system contrast does not justify selecting a new favorable threshold,
   cohort, or budget from this evaluation. A redesigned model requires another
   explicitly versioned study and an honest account of historical data access.
4. These four-hour results cannot validate the 720-minute handoff or repair
   mature/drift measurements produced by an incompatible TTL accounting path.
5. Drift requires causal alarm generation, comparable reset/adaptation rules,
   false-alarm accounting, and the same resource accounting. An event-aligned
   reset study alone supports conditional recovery, not deployable detection.

The experiment answers a bounded attribution and fairness question. It does
not certify the current paper as ready for submission.
