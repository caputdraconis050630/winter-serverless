#!/bin/bash
# Waits for run_recovery.sh terminal state; on success, launches run_followup.sh.
LOG=/data/260715/serverless-fewshot/results/logs/run_recovery.log
while true; do
  if grep -q "RECOVERY CHAIN COMPLETE" "$LOG" 2>/dev/null; then
    bash /data/260715/serverless-fewshot/scripts/run_followup.sh \
      > /data/260715/serverless-fewshot/results/logs/run_followup.log 2>&1
    exit 0
  fi
  if grep -q "RECOVERY FAILED" "$LOG" 2>/dev/null; then
    echo "[watcher] recovery chain failed — followup NOT started" \
      >> /data/260715/serverless-fewshot/results/logs/run_followup.log
    exit 1
  fi
  sleep 30
done
