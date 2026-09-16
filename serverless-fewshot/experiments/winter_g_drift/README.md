# Direct WINTER-G drift extension

This independent campaign adds the fixed age/count router and its no-age-handoff
control to all six retained drift cohorts. Old cohorts, forecasts, simulator
results, source checkpoints, and manuscript files remain intact.

## Protocol

- 3,420 event windows across Azure 2019, Azure 2021 and Huawei, natural and
  injected shifts; multiple windows/kinds from one function stay clustered.
- 24-hour burn-in followed by four hours after the shift; full sandbox state
  continues through the shift. Three DES seeds (0, 1, 2), rho = 1 and 10.
- Added branches use the paper's onboarding update procedure: causal past
  60-minute features, next-minute supervised pairs, all earlier observed rows,
  a prototype-biased ridge solve each minute, meta-learned regularization.
- Source checkpoints match each retained drift case and its LSTM: Azure 2021
  S1 for Azure 2021, S2 for Azure 2019/Huawei. Sixteen source-only prototypes
  use the onboarding builder; S2 reuses the verified primary prototype cache.
  This is an extension of the onboarding procedure to retained shift windows,
  distinct from the old scheduled, trailing-120-row, fixed-lambda adapter.
- Support starts at the replay observation boundary, including observed zero
  rows. Azure gate age begins at the first in-window arrival; Huawei age uses
  recorded presence at replay start. This does not assert a function's true
  deployment age before the available observation window.
- Routing: N=0 prototype; 0<N<100 EWMA(0.1); N>=100 and age<720 learned;
  otherwise EWMA(0.1). No-handoff changes only the last age condition.
- No drift detector, oracle, state reset, threshold search or target selection.
- The first 1/15/30/60/240 post-shift minutes and four disjoint intervals are
  retained. The primary registered contrast is rho=10, the full four hours.
- COST = idle GB seconds / 1k calls + 150*rho*CSR_percent, with each window's
  own calls as denominator. Predictor/control-plane compute and terminal drain
  are excluded. These are modeled provisioning costs, not cloud invoices.
- 2,000 paired cluster bootstrap samples; Azure applications, Huawei functions;
  seeds averaged first. Intervals are exploratory marginal 95% intervals,
  conditional on source checkpoints and cohorts, without multiplicity adjustment.

## Execution

From `/data/260715/serverless-fewshot`, use this environment prefix:

```bash
env PYTHONPATH=/data/260715/site-packages-des:/data/260715/site-packages \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python3.13 experiments/winter_g_drift/test_protocol.py
```

With the same environment, execute in order:

```bash
python3.13 experiments/winter_g_drift/campaign.py --workers 5
python3.13 experiments/winter_g_drift/verify_baselines.py
python3.13 experiments/winter_g_drift/analyze.py
```

Forecasting and ridge adaptation use CUDA when available; the unchanged DES
uses CPU threads. Completed forecast chunks and function/seed jobs are reusable.
Do not change a frozen campaign's code/protocol to resume it; use a new version.

## Verification and artifacts

Outputs: `results/winter_g_drift_v1/`.

- `protocol.json`: frozen grid, arms, source case/aggregate hashes and script.
- `models/`: checkpoint/prototype identity, lambda and source-only prototypes.
- `cases/`: exact data and action arrays, routing states, forecast checks,
  execution contracts, per-seed/function ledgers, aggregate and summaries.
- `baseline_replay_checks.json`: each case's median-volume row replayed under
  native LSTM, shared LSTM and EWMA at all three seeds and all saved windows;
  counts match exactly, phase ledgers match numerical tolerances.
- `analysis.json`, `comparisons.csv`: 648 paired condition/window comparisons,
  source/input/accounting/action identity checks and confidence intervals.
- `figures/post_shift_{csr,cost}.{pdf,png}`: cumulative post-shift outcomes.

The protocol tests check causal feature/target boundaries, count/age boundaries,
no reset at shift, and incremental versus direct ridge solutions. The existing
optimized-engine tests separately compare the retained simulator implementations.
The analyzer verifies that new G actions equal source EWMA actions in W/M states
and no-handoff actions outside M, and that all baselines retain their old hashes.
