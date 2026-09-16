# WINTER and WINTER-G with one online adaptation procedure

This additive campaign supplies standalone WINTER on the six retained shift
cohorts. It uses the exact `learned` array already used by WINTER-G, at every
minute, including the below-count-threshold period. It is not the no-age-handoff
ablation, which retains the low-count EWMA branch.

The source body, 16 prototypes, per-minute all-past-row ridge update, source
regularizer, fixed controller, inputs, observation boundary, request streams,
and seeds match the completed WINTER-G campaign. No new fitting or tuning occurs.
Each policy starts from its own empty containers at the same replay boundary and
retains its resulting state through 24 hours of burn-in and four post-shift hours.
There is no drift signal or reset. The complete 3,420 event windows, three seeds,
and rho 1/10 grid are retained. Old results and frozen scripts remain untouched.

The primary window is four post-shift hours at rho 10. All cumulative
1/15/30/60/240-minute and disjoint 1–15/15–30/30–60/60–240-minute results are saved.
Paired cluster bootstrap uses the same 2,000 draws (seed 260915), seed-averaged
per-window totals, Azure application/Huawei function clusters, and marginal 95%
intervals as the earlier analysis. Existing G comparisons must reproduce those
earlier estimates and CIs. Cost and accounting conventions are unchanged.

From the experiment root, with Python 3.13 and the workspace NumPy/Numba stack:

```bash
export PYTHONPATH=/data/260715/site-packages-des:/data/260715/site-packages
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3.13 experiments/winter_drift/campaign.py --workers 6
python3.13 experiments/winter_drift/verify.py
python3.13 experiments/winter_drift/analyze.py
```

Results are in `results/winter_drift_v1/`. Prediction arrays originate in the
verified CUDA-enabled earlier campaign; only CPU discrete-event replay is needed
here. Execution is resumable using the unchanged function/seed runner.
`protocol.json` freezes source hashes before execution. `verification.json`
records action identity, all function/seed job contracts and independent replay.
`analysis.json`, the two numerical CSVs, and `figure_curves.csv` provide the
paper-local evidence. Never edit a frozen campaign script to resume execution.
