#!/usr/bin/env bash
# Chain: wait for E4 (testbed_e2e_diag) -> smoke-test E5 -> full E5 -> E6.
#
# The full cohort run is ~4.5 h of unattended wall time on a script that has
# never executed end to end, so a 3-function / 2-minute smoke pass runs first
# and the full run only starts if it produced a valid result file.
set -uo pipefail
cd /data/260715/serverless-fewshot
export PYTHONPATH=/data/260715/site-packages:/data/260715/site-packages-des
LOG=/tmp/claude-1000/-data-260715/283d29e5-4028-4fe3-bfc3-b75155cc6721/scratchpad
RUNS=results/runs

# Wait for E4 only if it has not already produced its result. NOTE: do not
# poll with `pgrep -f testbed_e2e_diag` -- any watcher shell carrying that
# string in its own command line matches too, so two waiters deadlock on
# each other. Match the python process explicitly instead.
if [ ! -f "$RUNS/testbed_e2e_diag.json" ]; then
  echo "[chain] waiting for E4 to finish..."
  while pgrep -f '^python3\.13 scripts/testbed_e2e_diag' >/dev/null; do
    sleep 15
  done
  if [ ! -f "$RUNS/testbed_e2e_diag.json" ]; then
    echo "[chain] ABORT: E4 produced no output"; exit 1
  fi
fi
echo "[chain] E4 result present."

echo "[chain] smoke test (3 funcs x 2 min x 4 arms)"
python3.13 scripts/testbed_cohort.py --minutes 2 --funcs 3 \
    > "$LOG/e5_smoke.log" 2>&1
if [ ! -f "$RUNS/testbed_cohort.json" ]; then
  echo "[chain] ABORT: smoke test failed, see e5_smoke.log"; exit 1
fi
echo "[chain] smoke OK"
# Never leave the smoke result behind masquerading as the real run.
rm -f "$RUNS/testbed_cohort.json"

echo "[chain] full run (10 funcs x 60 min x 4 arms)"
python3.13 scripts/testbed_cohort.py > "$LOG/e5.log" 2>&1
if [ ! -f "$RUNS/testbed_cohort.json" ]; then
  echo "[chain] ABORT: full run failed, see e5.log"; exit 1
fi
echo "[chain] E5 done."

echo "[chain] DES mirror"
python3.13 scripts/testbed_cohort_desmirror.py > "$LOG/e6.log" 2>&1
if [ ! -f "$RUNS/testbed_cohort_desmirror.json" ]; then
  echo "[chain] ABORT: mirror failed, see e6.log"; exit 1
fi
echo "[chain] E6 done. all complete."
