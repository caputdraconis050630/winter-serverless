#!/usr/bin/env bash
# WP-H3 tail + WP-H4 controls, chained unattended.
#
# Waits for the h_saturated DES runner to land its results, then runs every
# remaining step up to (but not including) paper integration:
#   1. cell 2/3 statistics over all three Huawei pools
#   2. duration-sensitivity export + 3 DES variants + report
#   3. joins the channel-10 control (launched separately) and reports its state
# Emits one STEP/OK/FAIL line per stage on stdout so a monitor can follow it.

set -u
cd /data/260715/serverless-fewshot || exit 1

RUNS=results/runs
LOGS=results/logs
SENTINEL="$RUNS/.huawei_followup_done"
PY_TORCH="PYTHONPATH=/data/260715/site-packages:."
PY_DES="PYTHONPATH=/data/260715/site-packages-des:."
rm -f "$SENTINEL"
mkdir -p "$LOGS"

say() { echo "[$(date -u -d '+9 hours' '+%H:%M KST')] $*"; }
fail() { say "FAIL $1"; echo "FAIL $1" > "$SENTINEL"; exit 1; }

# ---- 1. wait for the saturated pool -------------------------------------
say "STEP wait h_saturated DES"
while [ ! -f "$RUNS/revision_h2_huawei_des_h_saturated.json" ]; do
  if ! pgrep -f "split_jobs_huawei_h_saturated" > /dev/null; then
    fail "h_saturated runner died without writing results"
  fi
  sleep 60
done
say "OK h_saturated DES results present"

# ---- 2. cell 2/3 statistics ---------------------------------------------
say "STEP cell 2/3 statistics"
env $PY_TORCH python3.13 -u scripts/revision_h2_huawei_stats.py \
    > "$LOGS/revision_h2_huawei_stats.log" 2>&1 \
  || fail "revision_h2_huawei_stats.py (see $LOGS/revision_h2_huawei_stats.log)"
say "OK cell 2/3 statistics -> revision_h2_huawei_stats.json"

# ---- 3. duration sensitivity --------------------------------------------
say "STEP duration sensitivity export"
env $PY_TORCH python3.13 -u scripts/revision_h4_dur_sensitivity.py --export \
    > "$LOGS/revision_h4_dur_export.log" 2>&1 \
  || fail "duration-sensitivity export (see $LOGS/revision_h4_dur_export.log)"

for v in huawei_private azure2021 const1s; do
  say "STEP duration sensitivity DES: $v"
  env $PY_DES nice -n 5 python3.13 -u scripts/des_runner_fast.py \
      --jobs "$RUNS/split_jobs_hdur_$v" \
      --out "$RUNS/revision_h4_dur_$v.json" \
      > "$LOGS/revision_h4_dur_$v.log" 2>&1 \
    || fail "duration-sensitivity DES $v (see $LOGS/revision_h4_dur_$v.log)"
done

say "STEP duration sensitivity report"
env $PY_TORCH python3.13 -u scripts/revision_h4_dur_sensitivity.py --report \
    > "$LOGS/revision_h4_dur_report.log" 2>&1 \
  || fail "duration-sensitivity report (see $LOGS/revision_h4_dur_report.log)"
say "OK duration sensitivity -> revision_h4_dur_sensitivity.json"

# ---- 4. channel-10 control ----------------------------------------------
say "STEP wait channel-10 control"
while [ ! -f "$RUNS/revision_h3_ch10_control.json" ]; do
  if ! pgrep -f "revision_h3_ch10_control.py" > /dev/null; then
    fail "channel-10 control died without writing results"
  fi
  sleep 60
done
say "OK channel-10 control -> revision_h3_ch10_control.json"

say "DONE all WP-H3/H4 work complete; paper integration (WP-H5) is next"
echo "OK" > "$SENTINEL"
