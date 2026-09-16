#!/usr/bin/env bash
# Second pass of the WP-H5 diagnostic: the arms that separate the provider
# effect from the checkpoint choice and from the split the paper actually
# claims parity on.
#   huawei_s1ckpt  Huawei h_mixed under the checkpoint the Azure steady-state
#                  campaign uses (s1_s0) instead of the transfer checkpoint
#   azure2021_s1   Azure S1 test  -- paper claims TOST parity here
#   azure2021_s3   Azure S3 test  -- paper claims TOST parity here
# (huawei/s2_s0 and azure2021/s2_test already ran in run_recal_diag.sh.)

set -u
cd /data/260715/serverless-fewshot || exit 1

RUNS=results/runs
LOGS=results/logs
SENTINEL="$RUNS/.recal_diag2_done"
PY_TORCH="PYTHONPATH=/data/260715/site-packages:."
PY_DES="PYTHONPATH=/data/260715/site-packages-des:."
rm -f "$SENTINEL"

say() { echo "[$(date -u -d '+9 hours' '+%H:%M KST')] $*"; }
fail() { say "FAIL $1"; echo "FAIL $1" > "$SENTINEL"; exit 1; }

for trace in huawei_s1ckpt azure2021_s1 azure2021_s3; do
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

say "STEP recal report (all arms)"
env $PY_TORCH python3.13 -u scripts/revision_h5_scale_recal.py --report \
    > "$LOGS/revision_h5_recal_report.log" 2>&1 \
  || fail "recal report (see $LOGS/revision_h5_recal_report.log)"

say "DONE diagnostic pass 2 complete"
echo "OK" > "$SENTINEL"
