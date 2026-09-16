#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC="results/runs/des_jobs_r21_huawei_hsaturated_remaining/h_saturated"
SPLIT_ROOT="results/runs/des_jobs_r21_huawei_hsaturated_remaining_by_job"
OUT_ROOT="results/runs/revision_r21_huawei_hsaturated_remaining_by_job"
LOG_DIR="results/runs/logs_r21_huawei_hsaturated_remaining"
PYTHONPATH_DES="/data/260715/site-packages-des:."

mkdir -p "$SPLIT_ROOT" "$OUT_ROOT" "$LOG_DIR"

python3 - <<'PY'
import json
import shutil
from pathlib import Path

src = Path("results/runs/des_jobs_r21_huawei_hsaturated_remaining/h_saturated")
dst_root = Path("results/runs/des_jobs_r21_huawei_hsaturated_remaining_by_job")
meta = json.load(open(src / "jobs.json"))

for job in meta["jobs"]:
    name = f"{job['method']}__rho{job['rho']:g}".replace(".", "p")
    dst = dst_root / name / "h_saturated"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "shared.npz", dst / "shared.npz")
    shutil.copy2(src / job["file"], dst / job["file"])
    one = dict(meta)
    one["jobs"] = [job]
    with open(dst / "jobs.json", "w") as f:
        json.dump(one, f, indent=2)
PY

launch_one() {
  local name="$1"
  local jobs="$SPLIT_ROOT/$name"
  local out="$OUT_ROOT/${name}.json"
  local log="$LOG_DIR/${name}.log"
  local session="r21_hsat_${name}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "${name}: tmux session already running (${session})"
    return
  fi
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && env PYTHONPATH='$PYTHONPATH_DES' python3.13 scripts/revision_r21_des_runner_resume.py --jobs '$jobs' --out '$out' 2>&1 | tee -a '$log'"
  echo "${name}: tmux=${session} log=$log"
}

for d in "$SPLIT_ROOT"/*; do
  [ -d "$d" ] || continue
  launch_one "$(basename "$d")"
done
