#!/bin/bash
# Resume of run_recovery.sh after step-8 numba-cache failure (2026-07-16):
# corrupted src/sim/__pycache__/des_fast.*.nbi/.nbc deleted; steps 1-7 already
# verified PASSED. Runs steps 8-9, then chains run_followup.sh.
set -uo pipefail
cd /data/260715/serverless-fewshot
export PYTHONPATH=/data/260715/site-packages
DESPP=/data/260715/site-packages-des:.
LOG=results/logs
PY=python3.13

step() { echo ""; echo "===== [$(date '+%F %T')] $1 ====="; }
fail() { echo "!!!!! RESUME FAILED at: $1 (exit $2)"; exit 1; }

step "8/9 DES run (numba env)"
PYTHONPATH=$DESPP $PY -u scripts/des_runner_fast.py --jobs results/runs/des_jobs_2019 --out results/runs/sim_results_des_2019.json > $LOG/des_run_2019_rerun.log 2>&1 || fail des-run $?
$PY -c "import json; r=json.load(open('results/runs/sim_results_des_2019.json')); print('des run verify OK, entries:', len(r))" || fail des-run-verify $?
sync

step "9/9 DES stats (T3 table)"
$PY -u scripts/des_2019_stats.py > $LOG/des_stats_2019_rerun.log 2>&1 || fail des-stats $?
$PY - <<'PYEOF' || fail des-stats-verify $?
import json
s=json.load(open("results/runs/des_2019_stats.json"))
b=open("results/tables/T3_des_2019.csv","rb").read()
assert len(b)>0 and b.count(0)==0
print("des stats verify OK")
PYEOF
sync

echo ""
echo "===== [$(date '+%F %T')] RECOVERY CHAIN COMPLETE (resumed 8-9) ====="

step "chaining run_followup.sh"
bash scripts/run_followup.sh > $LOG/run_followup.log 2>&1 || fail followup $?
echo "===== [$(date '+%F %T')] ALL DONE ====="
