#!/bin/bash
# Revision supervisor (2026-07-18): monitors every running/queued experiment,
# detects failures in near-real-time (2-min cycle), and auto-restarts with
# capped retries. Units:
#   [disk]    /data must be rw — if ro, pause all restarts (manual reboot+fstrim)
#   [chain]   run_resume_89.sh (PID 18636): on RESUME FAILED -> clear numba
#             cache, restart chain script (retry cap 1 — it is a ~1.5-day job)
#   [watcher] run_revision_des_2019.sh: if dead before launching runners while
#             chain still runs -> relaunch
#   [runner]  8 per-split DES runners: if died without a valid output JSON ->
#             clear numba cache, relaunch that unit (retry cap 2 each)
#   [merge]   when all 8 part outputs verify -> (re)run revision_merge_2019.py
# Status/log: results/logs/supervisor.log ; retry counters in
# results/logs/supervisor_state/. Lines starting with CRITICAL/RESTART are the
# alert surface consumed by the session Monitor.
set -u
cd /data/260715/serverless-fewshot
LOG=results/logs/supervisor.log
STATE=results/logs/supervisor_state
mkdir -p "$STATE"
DESPP=/data/260715/site-packages-des:.
PY=python3.13
CHAIN_PID_FILE=$STATE/chain_pid
echo 18636 > "$CHAIN_PID_FILE"  # initial chain PID

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
rc(){ cat "$STATE/$1" 2>/dev/null || echo 0; }
inc(){ echo $(( $(rc "$1") + 1 )) > "$STATE/$1"; }
clear_numba(){ rm -f src/sim/__pycache__/des_fast.*.nbi src/sim/__pycache__/des_fast.*.nbc 2>/dev/null; log "numba cache cleared"; }

verify_json(){ # verify_json <path> -> 0 if valid non-empty JSON without NULs
  $PY - "$1" <<'PYEOF' >/dev/null 2>&1
import json, sys
b = open(sys.argv[1], "rb").read()
assert b and b.count(0) == 0
r = json.loads(b)
rows = r["results"] if isinstance(r, dict) and "results" in r else r
assert rows
PYEOF
}

PAIRS="v1:S1 v1:S2 v1:S3 e1:S2 e1a:S1 e1a:S3 e1b:S1 e1b:S3"
log "supervisor started (pid $$)"

while true; do
  # ---- disk ----
  if ! awk '$2=="/data" && $4~/(^|,)rw(,|$)/{f=1} END{exit !f}' /proc/mounts; then
    log "CRITICAL disk: /data not mounted rw — restarts paused, manual reboot+fstrim required"
    sleep 300; continue
  fi

  # ---- chain ----
  cpid=$(cat "$CHAIN_PID_FILE")
  if ! kill -0 "$cpid" 2>/dev/null; then
    if grep -q "ALL DONE" results/logs/run_resume_89.log 2>/dev/null || \
       [ -f "$STATE/chain_done" ]; then
      [ -f "$STATE/chain_done" ] || { log "chain complete (ALL DONE)"; touch "$STATE/chain_done"; }
    elif [ -f "$STATE/chain_retrying" ] && pgrep -f "run_resume_89.sh" >/dev/null; then
      : # a restarted chain is running under a new pid; track it
      pgrep -f "run_resume_89.sh" | head -1 > "$CHAIN_PID_FILE"
    else
      n=$(rc chain)
      if [ "$n" -lt 1 ]; then
        inc chain; touch "$STATE/chain_retrying"
        log "CRITICAL chain: exited without ALL DONE (log tail: $(tail -1 results/logs/run_resume_89.log 2>/dev/null)) — RESTART attempt $((n+1)) after numba cache clear"
        clear_numba
        nohup setsid bash scripts/run_resume_89.sh >> results/logs/run_resume_89.log 2>&1 < /dev/null &
        echo $! > "$CHAIN_PID_FILE"
      elif [ ! -f "$STATE/chain_gave_up" ]; then
        log "CRITICAL chain: retry cap reached — manual intervention required"
        touch "$STATE/chain_gave_up"
      fi
    fi
  fi

  # ---- runners (only meaningful once the watcher launched them) ----
  all_parts_ok=1
  for pair in $PAIRS; do
    set="${pair%%:*}"; sp="${pair##*:}"
    out=results/runs/revision_${set}_2019_${sp}.json
    runlog=results/logs/revision_${set}_des_2019_${sp}.log
    if verify_json "$out"; then continue; fi
    all_parts_ok=0
    [ -f "$runlog" ] || continue            # not launched yet
    if pgrep -f "des_runner_fast.py --jobs .*split_jobs_${set}_${sp} " >/dev/null || \
       pgrep -f "split_jobs_${set}_${sp}" >/dev/null; then
      continue                              # still running
    fi
    n=$(rc "runner_${set}_${sp}")
    if [ "$n" -lt 2 ]; then
      inc "runner_${set}_${sp}"
      log "RESTART runner ${set}/${sp} (attempt $((n+1)); log tail: $(tail -1 "$runlog" 2>/dev/null | head -c 160))"
      clear_numba
      wrap=results/runs/split_jobs_${set}_${sp}
      mkdir -p "$wrap"
      ln -sfn "$PWD/results/runs/des_jobs_2019_${set}/${sp}" "$wrap/${sp}"
      PYTHONPATH=$DESPP nice -n 5 nohup setsid $PY -u scripts/des_runner_fast.py \
          --jobs "$wrap" --out "$out" >> "$runlog" 2>&1 < /dev/null &
    elif [ ! -f "$STATE/runner_${set}_${sp}_gave_up" ]; then
      log "CRITICAL runner ${set}/${sp}: retry cap reached — manual intervention required"
      touch "$STATE/runner_${set}_${sp}_gave_up"
    fi
  done

  # ---- watcher liveness (needed only until runners are launched) ----
  if [ ! -f "$STATE/chain_done" ] || [ "$all_parts_ok" -eq 0 ]; then
    first_runlog=results/logs/revision_v1_des_2019_S1.log
    if ! pgrep -f "run_revision_des_2019.sh" >/dev/null && [ ! -f "$first_runlog" ]; then
      n=$(rc watcher)
      if [ "$n" -lt 3 ]; then
        inc watcher
        log "RESTART watcher (attempt $((n+1)))"
        nohup setsid bash scripts/run_revision_des_2019.sh >> results/logs/revision_watcher.log 2>&1 < /dev/null &
      fi
    fi
  fi

  # ---- merge when everything is in ----
  if [ "$all_parts_ok" -eq 1 ] && [ ! -f "$STATE/merged_ok" ]; then
    if $PY scripts/revision_merge_2019.py >> "$LOG" 2>&1; then
      log "ALL EXPERIMENTS COMPLETE — merged outputs verified"
      touch "$STATE/merged_ok"
    else
      log "merge attempted but parts incomplete (will retry)"
    fi
  fi

  if [ -f "$STATE/merged_ok" ] && [ -f "$STATE/chain_done" ]; then
    log "supervisor: all units complete, exiting"
    exit 0
  fi
  sleep 120
done
