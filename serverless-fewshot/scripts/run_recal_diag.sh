#!/usr/bin/env bash
# WP-H5 diagnostic chain: scale recalibration on Huawei h_mixed and on the
# Azure-2021 steady pool (the in-distribution reference), then the report.
# Emits STEP/OK/FAIL lines for a monitor.

set -u
cd /data/260715/serverless-fewshot || exit 1

RUNS=results/runs
LOGS=results/logs
SENTINEL="$RUNS/.recal_diag_done"
PY_TORCH="PYTHONPATH=/data/260715/site-packages:."
PY_DES="PYTHONPATH=/data/260715/site-packages-des:."
rm -f "$SENTINEL"
mkdir -p "$LOGS"

say() { echo "[$(date -u -d '+9 hours' '+%H:%M KST')] $*"; }
fail() { say "FAIL $1"; echo "FAIL $1" > "$SENTINEL"; exit 1; }

for trace in huawei azure2021; do
  say "STEP recal export: $trace"
  env $PY_TORCH python3.13 -u scripts/revision_h5_scale_recal.py \
      --trace "$trace" --export > "$LOGS/revision_h5_recal_export_$trace.log" 2>&1 \
    || fail "recal export $trace (see $LOGS/revision_h5_recal_export_$trace.log)"

  say "STEP recal DES: $trace"
  env $PY_DES nice -n 5 python3.13 -u scripts/des_runner_fast.py \
      --jobs "$RUNS/split_jobs_recal_$trace" \
      --out "$RUNS/revision_h5_recal_$trace.json" \
      > "$LOGS/revision_h5_recal_des_$trace.log" 2>&1 \
    || fail "recal DES $trace (see $LOGS/revision_h5_recal_des_$trace.log)"
  say "OK $trace"
done

say "STEP recal report"
env $PY_TORCH python3.13 -u scripts/revision_h5_scale_recal.py --report \
    > "$LOGS/revision_h5_recal_report.log" 2>&1 \
  || fail "recal report (see $LOGS/revision_h5_recal_report.log)"

say "DONE scale-recalibration diagnostic complete"
echo "OK" > "$SENTINEL"
