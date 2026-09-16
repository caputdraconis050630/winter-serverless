#!/bin/bash
# Follow-up after run_recovery.sh completes:
#   A. des_subsample_validation (regenerate estimator-validation artifact, 2021 inputs only)
#   B. val2021 job export (2021 S1, full-172 take-all) — with 2019 manifest backup/restore
#   C. validate_des_fast: numba vs python statistical validation (28 configs)
#   D. read-back verification + small-artifact backup to root fs
set -uo pipefail
cd /data/260715/serverless-fewshot
export PYTHONPATH=/data/260715/site-packages
DESPP=/data/260715/site-packages-des:.
LOG=results/logs
PY=python3.13

step() { echo ""; echo "===== [$(date '+%F %T')] $1 ====="; }
fail() { echo "!!!!! FOLLOWUP FAILED at: $1 (exit $2)"; exit 1; }

step "A. des_subsample_validation"
$PY -u scripts/des_subsample_validation.py > $LOG/des_subsample_validation_rerun.log 2>&1 || fail subsample $?
$PY -c "import json; json.load(open('results/runs/des_subsample_validation.json')); print('subsample verify OK')" || fail subsample-verify $?
sync

step "B. val2021 export (manifest-safe)"
cp results/runs/des_sample_manifest.json results/runs/des_sample_manifest_2019.json || fail manifest-backup $?
cp results/runs/gate_v2_routing_stats.json results/runs/gate_v2_routing_stats_2019.json || fail gatestats-backup $?
SF_DATA_DIR=processed SF_CKPT_DIR=results_azure2021/runs $PY -u scripts/phase6_des_sampled.py \
    --export-jobs results/runs/des_jobs_val2021 --splits S1 > $LOG/des_export_val2021_rerun.log 2>&1 || fail val2021-export $?
mv results/runs/des_sample_manifest.json results/runs/des_sample_manifest_val2021.json
mv results/runs/gate_v2_routing_stats.json results/runs/gate_v2_routing_stats_val2021.json
cp results/runs/des_sample_manifest_2019.json results/runs/des_sample_manifest.json
cp results/runs/gate_v2_routing_stats_2019.json results/runs/gate_v2_routing_stats.json
$PY - <<'PYEOF' || fail val2021-verify $?
import numpy as np, glob, json
jobs=sorted(glob.glob("results/runs/des_jobs_val2021/**/*.npz", recursive=True))
assert jobs, "no val2021 jobs"
for f in jobs:
    d=np.load(f, allow_pickle=True); [d[k] for k in d.files]
print("val2021 export verify OK:", len(jobs), "npz files")
PYEOF
sync

step "C. validate_des_fast (numba vs python)"
PYTHONPATH=$DESPP $PY -u scripts/validate_des_fast.py --jobs results/runs/des_jobs_val2021 > $LOG/validate_des_fast_rerun.log 2>&1 || fail validate $?
$PY - <<'PYEOF' || fail validate-verify $?
import json
r=json.load(open("results/runs/des_fast_validation.json"))
print("des_fast_validation regenerated; summary keys:", list(r)[:8])
PYEOF
sync

step "D. backup small artifacts"
B=/home/cnlab/.claude/backups/sf-260716
cp results/runs/go_nogo_gate.json results/runs/multiseed_training.json \
   results/runs/diag_gate2019_ksweep.json results/runs/router_analysis_2019.json \
   results/runs/router_analysis_azure2021.json results/runs/des_subsample_validation.json \
   results/runs/des_fast_validation.json results/runs/des_sample_manifest_2019.json \
   results/runs/gate_v2_routing_stats_2019.json results/runs/des_2019_stats.json \
   results/tables/T3_des_2019.csv $B/ 2>/dev/null
ls -la $B/

echo ""
echo "===== [$(date '+%F %T')] FOLLOWUP COMPLETE ====="
