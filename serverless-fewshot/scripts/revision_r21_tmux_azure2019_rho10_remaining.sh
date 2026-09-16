#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHONPATH_DES="/data/260715/site-packages-des:."
LOG_DIR="results/runs/logs_r21_azure2019_rho10"
mkdir -p "$LOG_DIR"

launch_one() {
  local name="$1"
  local jobs="$2"
  local out="$3"
  local log="$LOG_DIR/${name}.log"
  local session="r21_${name}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "${name}: tmux session already running (${session})"
    return
  fi
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && env PYTHONPATH='$PYTHONPATH_DES' python3.13 scripts/revision_r21_des_runner_resume.py --jobs '$jobs' --out '$out' 2>&1 | tee -a '$log'"
  echo "${name}: tmux=${session} log=$log"
}

launch_one S1 \
  results/runs/des_jobs_r21_azure2019_rho10_only_S1 \
  results/runs/revision_r21_azure2019_rho10_S1_faithful_fast_runs.json

launch_one S2 \
  results/runs/des_jobs_r21_azure2019_rho10_split_S2 \
  results/runs/revision_r21_azure2019_rho10_S2_faithful_fast_runs.json

launch_one S3_B4a_ewma \
  results/runs/des_jobs_r21_azure2019_rho10_S3_by_method/B4a_ewma \
  results/runs/revision_r21_azure2019_rho10_S3_B4a_ewma_fast_runs.json

launch_one S3_A5_faithful \
  results/runs/des_jobs_r21_azure2019_rho10_S3_by_method/A5_faithful \
  results/runs/revision_r21_azure2019_rho10_S3_A5_faithful_fast_runs.json

launch_one S3_G_WE_Ainf \
  results/runs/des_jobs_r21_azure2019_rho10_S3_by_method/G_WE_Ainf \
  results/runs/revision_r21_azure2019_rho10_S3_G_WE_Ainf_fast_runs.json

launch_one S3_G_WL_A0 \
  results/runs/des_jobs_r21_azure2019_rho10_S3_by_method/G_WL_A0 \
  results/runs/revision_r21_azure2019_rho10_S3_G_WL_A0_fast_runs.json

launch_one S3_G_WE_A720 \
  results/runs/des_jobs_r21_azure2019_rho10_S3_by_method/G_WE_A720 \
  results/runs/revision_r21_azure2019_rho10_S3_G_WE_A720_fast_runs.json
