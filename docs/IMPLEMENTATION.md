# Implementation map for the submitted version

| Manuscript component | Code |
|---|---|
| TCN and closed-form ridge adaptation | `src/models/bodies.py`, `heads.py`, `prototypes.py`, `src/meta/trainer.py`; training entry point `scripts/phase4_train.py` |
| Fifer-inspired LSTM and common replay | `experiments/lstm_comparison/model.py`, `cases.py`, `events.py`, `simulator.py`, `simulator_events.py`, `run_seeds.py` |
| Fixed EWMA weight and final composed outcomes | `experiments/fixed_ewma/campaign.py`, `runner.py`, `merge.py` |
| Conservative controller selection | `experiments/fixed_ewma/controls.py`, `verify_controls.py` and `experiments/onboarding_attribution/` |
| Matched WINTER shift replay | `experiments/winter_drift/` |
| Onboarding forecasts and gate under shifts | `experiments/winter_g_drift/` |
| Final statistics and curves | `experiments/fixed_ewma/analyze.py` and `winter-paper/audit/build_lstm_revision.py` |
| Scalar/control/timing evidence | `winter-paper/audit/build_final_evidence.py` |
| Gate contrasts and shift figures | `winter-paper/audit/build_winter_g_evidence.py` |
| Measured latency CDF | `winter-paper/audit/build_calibration_figure.py` |
| Independent numerical and table review | `winter-paper/audit/submission_sentence_review_2026-09-16/` |
| Live schedules | `experiments/fixed_ewma/live_replay.py`, `verify_live.py`, `experiments/lstm_comparison/live_strict.py` |

Paths beginning with `experiments/`, `src/`, or `scripts/` are relative to
`serverless-fewshot/`. Original measurement code is preserved byte-for-byte;
release entry points are separate under `tools/`.

WINTER is the prediction model. WINTER-G is its output-selection gate. The
no-age-hand-off control preserves the gate's prototype/low-count routes and
removes only its mature-age switch; it is not identical to ungated WINTER.
Both branches remain maintained in the evaluated gated implementation.

Source checkpoints, shared inputs, recorded simulator versions and invariant
measurements needed by the final campaign are retained. The final
`fixed_ewma_v1/cases/*/case.json` files define the comparison arms. Some immutable
source arrays also contain older scheduled drift and non-0.3 EWMA results;
these are not extracted into final comparisons. Unused follow-up campaigns,
including R30–R33 and onboarding-symmetric experiments, are excluded.
The scope and metadata correction are documented in `SCOPE.md` and
`REPRODUCTION.md`.
