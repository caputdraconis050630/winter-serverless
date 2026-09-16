#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/260715/serverless-fewshot"
cd "$ROOT"

PY="python3.13"
ENV_PYTHONPATH="/data/260715/site-packages-des:."
RUNS="$ROOT/results/runs"
TABLES="$ROOT/results/tables"
LOG_DIR="$RUNS/r33_adaptive_ewma_logs"
mkdir -p "$RUNS" "$TABLES" "$LOG_DIR"

SEEDS_AZ21="0,1,2,3,4,5,6,7,8,9"
SEEDS_XTRACE="0,1,2"
RHOS_SCREEN="10"
RHOS_FULL="0.1,1,10,100"
WORKERS="${R33_WORKERS:-2}"

candidate_args_from_file() {
  local path="$1"
  local out=()
  while IFS= read -r cand; do
    [[ -z "$cand" ]] && continue
    out+=(--candidate "$cand")
  done < "$path"
  printf '%s\n' "${out[@]}"
}

run_export() {
  local stage="$1"
  local trace="$2"
  local rhos="$3"
  local seeds="$4"
  local jobs_root="$5"
  local manifest="$6"
  local cand_file=""
  if [[ $# -ge 7 ]]; then
    cand_file="$7"
    shift 7
  else
    shift 6
  fi
  local extra_args=("$@")
  local cand_args=()
  local default_args=()

  if [[ -n "$cand_file" ]]; then
    default_args=(--no-defaults)
    mapfile -t cand_args < <(candidate_args_from_file "$cand_file")
  fi

  if [[ -s "$manifest" && -d "$jobs_root" ]]; then
    echo "[$stage][$trace] export exists; reusing $jobs_root"
    return
  fi

  echo "[$stage][$trace] export start"
  PYTHONPATH="$ENV_PYTHONPATH" "$PY" -u scripts/revision_r33_adaptive_ewma.py \
    --export \
    --trace "$trace" \
    "${default_args[@]}" \
    "${cand_args[@]}" \
    --rhos "$rhos" \
    --seeds "$seeds" \
    --jobs-root "$jobs_root" \
    --manifest-out "$manifest" \
    "${extra_args[@]}"
  echo "[$stage][$trace] export done"
}

run_des() {
  local stage="$1"
  local trace="$2"
  local jobs_root="$3"
  local out_json="$4"
  echo "[$stage][$trace] DES start -> $out_json"
  NUMBA_NUM_THREADS=1 PYTHONPATH="$ENV_PYTHONPATH" "$PY" -u scripts/des_runner_fast_parallel.py \
    --jobs "$jobs_root" \
    --out "$out_json" \
    --workers "$WORKERS" \
    --resume
  echo "[$stage][$trace] DES done"
}

summarize_trace() {
  local stage="$1"
  local trace="$2"
  local manifest="$3"
  local candidate_json="$4"
  local prefix="$5"
  echo "[$stage][$trace] summarize start"
  PYTHONPATH="$ENV_PYTHONPATH" "$PY" -u scripts/revision_r33_adaptive_ewma.py \
    --summarize \
    --inputs "$candidate_json" \
    --manifest "$manifest" \
    --prefix "$prefix"
  echo "[$stage][$trace] summarize done"
}

select_finalists() {
  local out_file="$1"
  "$PY" - "$out_file" "$RUNS/revision_r33_adaptive_ewma_screen_azure2021_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

out_file = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
cand_map = {
    str(c["name"]): c
    for c in json.load(open(manifest_path))["candidates"]
}
paths = {
    "azure2021": Path("results/tables/T_r33_adaptive_ewma_screen_azure2021_rho10.csv"),
    "azure2019": Path("results/tables/T_r33_adaptive_ewma_screen_azure2019_rho10.csv"),
    "huawei": Path("results/tables/T_r33_adaptive_ewma_screen_huawei_rho10.csv"),
}
frames = []
for dataset, path in paths.items():
    df = pd.read_csv(path)
    df = df[df["method"].astype(str) != "B4a_ewma"].copy()
    df["dataset"] = dataset
    frames.append(df)
all_rows = pd.concat(frames, ignore_index=True)
grouped = all_rows.groupby("method", dropna=False).agg(
    cells=("rho", "size"),
    ewma_nonworse=("delta_csr_pp_vs_ewma", lambda s: int((s <= 0.0).sum())),
    mean_dcsr=("delta_csr_pp_vs_ewma", "mean"),
    max_dcsr=("delta_csr_pp_vs_ewma", "max"),
    min_dcsr=("delta_csr_pp_vs_ewma", "min"),
    mean_dcost=("delta_cost_pct_vs_ewma", "mean"),
    max_dcost=("delta_cost_pct_vs_ewma", "max"),
    mean_fast_share=("route_fast_ewma", "mean"),
    mean_entries=("mean_entries_per_function", "mean"),
).reset_index()
grouped["strict_pass"] = (
    (grouped["max_dcost"] <= 3.0)
    & (grouped["max_dcsr"] <= 0.02)
)
ranked = grouped.sort_values(
    by=[
        "strict_pass",
        "ewma_nonworse",
        "max_dcsr",
        "mean_dcsr",
        "max_dcost",
        "mean_fast_share",
    ],
    ascending=[False, False, True, True, True, True],
).reset_index(drop=True)
ranked.to_csv("results/tables/T_r33_adaptive_ewma_screening_rank.csv", index=False)
finalists = ranked[ranked["strict_pass"]]["method"].tolist()[:4]
if len(finalists) < 3:
    for name in ranked["method"].tolist():
        if name not in finalists:
            finalists.append(name)
        if len(finalists) >= 4:
            break

def raw_candidate(rec):
    keys = [
        "name", "mode", "trigger_mode", "alpha_base", "alpha_fast",
        "ttl", "cooldown", "trigger_alpha", "trigger_delta", "trigger_m",
        "trigger_calib_size", "trigger_window", "trigger_min_calib",
        "min_under_log",
    ]
    return ",".join(f"{k}={rec[k]}" for k in keys if k in rec)

with open(out_file, "w") as f:
    for name in finalists:
        f.write(raw_candidate(cand_map[name]) + "\n")
selection = {
    "criteria": {
        "cost_cap_pct_vs_ewma": 3.0,
        "max_csr_regression_pp_vs_ewma": 0.02,
        "rank_order": [
            "strict_pass desc",
            "ewma_nonworse desc",
            "max_dcsr asc",
            "mean_dcsr asc",
            "max_dcost asc",
            "mean_fast_share asc",
        ],
    },
    "strict_pass_count": int(ranked["strict_pass"].sum()),
    "finalists": finalists,
}
Path("results/runs/revision_r33_adaptive_ewma_screening_selection.json").write_text(
    json.dumps(selection, indent=1)
)
print("selected finalists:", ",".join(finalists))
print("strict_pass_count:", int(ranked["strict_pass"].sum()))
PY
}

compact_full_results() {
  "$PY" <<'PY'
import json
from pathlib import Path

import pandas as pd

finalists_path = Path("results/runs/revision_r33_adaptive_ewma_finalists.txt")
finalists = []
for line in finalists_path.read_text().splitlines():
    if not line.strip():
        continue
    parts = dict(part.split("=", 1) for part in line.split(",") if "=" in part)
    finalists.append(parts["name"])
paths = {
    "azure2021": Path("results/tables/T_r33_adaptive_ewma_full_azure2021_fullrho.csv"),
    "azure2019": Path("results/tables/T_r33_adaptive_ewma_full_azure2019_fullrho.csv"),
    "huawei": Path("results/tables/T_r33_adaptive_ewma_full_huawei_fullrho.csv"),
}
frames = []
for dataset, path in paths.items():
    df = pd.read_csv(path)
    df = df[df["method"].isin(["B4a_ewma"] + finalists)].copy()
    df["dataset"] = dataset
    frames.append(df)
all_rows = pd.concat(frames, ignore_index=True)
compact_cols = [
    "dataset", "split", "method", "rho", "csr_pct",
    "delta_csr_pp_vs_ewma", "delta_cost_pct_vs_ewma",
    "route_fast_ewma", "mean_fires_per_function",
    "mean_entries_per_function",
]
all_rows[compact_cols].to_csv(
    "results/tables/T_r33_adaptive_ewma_cross_trace_compact_fullrho.csv",
    index=False,
)
rank = all_rows[all_rows["method"].isin(finalists)].groupby("method", dropna=False).agg(
    cells=("rho", "size"),
    ewma_nonworse=("delta_csr_pp_vs_ewma", lambda s: int((s <= 0.0).sum())),
    mean_dcsr=("delta_csr_pp_vs_ewma", "mean"),
    max_dcsr=("delta_csr_pp_vs_ewma", "max"),
    mean_dcost=("delta_cost_pct_vs_ewma", "mean"),
    max_dcost=("delta_cost_pct_vs_ewma", "max"),
    mean_fast_share=("route_fast_ewma", "mean"),
).reset_index().sort_values(
    ["ewma_nonworse", "max_dcsr", "mean_dcsr", "max_dcost"],
    ascending=[False, True, True, True],
)
rank.to_csv("results/tables/T_r33_adaptive_ewma_finalist_rank_fullrho.csv", index=False)
Path("results/runs/revision_r33_adaptive_ewma_final_summary.json").write_text(
    json.dumps({"finalists": finalists, "rank_rows": json.loads(rank.to_json(orient="records"))}, indent=1)
)
print(rank.to_string(index=False))
PY
}

echo "R33 adaptive EWMA run started at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
PYTHONPATH="$ENV_PYTHONPATH" "$PY" -m py_compile scripts/revision_r33_adaptive_ewma.py scripts/des_runner_fast_parallel.py

SCREEN_AZ21_JOBS="$RUNS/des_jobs_r33_adaptive_ewma_screen_azure2021"
SCREEN_AZ19_JOBS="$RUNS/des_jobs_r33_adaptive_ewma_screen_azure2019"
SCREEN_HW_JOBS="$RUNS/des_jobs_r33_adaptive_ewma_screen_huawei"
SCREEN_AZ21_MAN="$RUNS/revision_r33_adaptive_ewma_screen_azure2021_manifest.json"
SCREEN_AZ19_MAN="$RUNS/revision_r33_adaptive_ewma_screen_azure2019_manifest.json"
SCREEN_HW_MAN="$RUNS/revision_r33_adaptive_ewma_screen_huawei_manifest.json"
SCREEN_AZ21_DES="$RUNS/revision_r33_adaptive_ewma_screen_azure2021_fast_runs.json"
SCREEN_AZ19_DES="$RUNS/revision_r33_adaptive_ewma_screen_azure2019_fast_runs.json"
SCREEN_HW_DES="$RUNS/revision_r33_adaptive_ewma_screen_huawei_fast_runs.json"

run_export screen azure2021 "$RHOS_SCREEN" "$SEEDS_AZ21" "$SCREEN_AZ21_JOBS" "$SCREEN_AZ21_MAN"
run_des screen azure2021 "$SCREEN_AZ21_JOBS" "$SCREEN_AZ21_DES"
summarize_trace screen azure2021 "$SCREEN_AZ21_MAN" "$SCREEN_AZ21_DES" "T_r33_adaptive_ewma_screen_azure2021"

run_export screen azure2019 "$RHOS_SCREEN" "$SEEDS_XTRACE" "$SCREEN_AZ19_JOBS" "$SCREEN_AZ19_MAN"
run_des screen azure2019 "$SCREEN_AZ19_JOBS" "$SCREEN_AZ19_DES"
summarize_trace screen azure2019 "$SCREEN_AZ19_MAN" "$SCREEN_AZ19_DES" "T_r33_adaptive_ewma_screen_azure2019"

run_export screen huawei "$RHOS_SCREEN" "$SEEDS_XTRACE" "$SCREEN_HW_JOBS" "$SCREEN_HW_MAN"
run_des screen huawei "$SCREEN_HW_JOBS" "$SCREEN_HW_DES"
summarize_trace screen huawei "$SCREEN_HW_MAN" "$SCREEN_HW_DES" "T_r33_adaptive_ewma_screen_huawei"

FINALISTS="$RUNS/revision_r33_adaptive_ewma_finalists.txt"
select_finalists "$FINALISTS"

FULL_AZ21_JOBS="$RUNS/des_jobs_r33_adaptive_ewma_full_azure2021"
FULL_AZ19_JOBS="$RUNS/des_jobs_r33_adaptive_ewma_full_azure2019"
FULL_HW_JOBS="$RUNS/des_jobs_r33_adaptive_ewma_full_huawei"
FULL_AZ21_MAN="$RUNS/revision_r33_adaptive_ewma_full_azure2021_manifest.json"
FULL_AZ19_MAN="$RUNS/revision_r33_adaptive_ewma_full_azure2019_manifest.json"
FULL_HW_MAN="$RUNS/revision_r33_adaptive_ewma_full_huawei_manifest.json"
FULL_AZ21_DES="$RUNS/revision_r33_adaptive_ewma_full_azure2021_fast_runs.json"
FULL_AZ19_DES="$RUNS/revision_r33_adaptive_ewma_full_azure2019_fast_runs.json"
FULL_HW_DES="$RUNS/revision_r33_adaptive_ewma_full_huawei_fast_runs.json"

run_export full azure2021 "$RHOS_FULL" "$SEEDS_AZ21" "$FULL_AZ21_JOBS" "$FULL_AZ21_MAN" "$FINALISTS"
run_des full azure2021 "$FULL_AZ21_JOBS" "$FULL_AZ21_DES"
summarize_trace full azure2021 "$FULL_AZ21_MAN" "$FULL_AZ21_DES" "T_r33_adaptive_ewma_full_azure2021"

run_export full azure2019 "$RHOS_FULL" "$SEEDS_XTRACE" "$FULL_AZ19_JOBS" "$FULL_AZ19_MAN" "$FINALISTS"
run_des full azure2019 "$FULL_AZ19_JOBS" "$FULL_AZ19_DES"
summarize_trace full azure2019 "$FULL_AZ19_MAN" "$FULL_AZ19_DES" "T_r33_adaptive_ewma_full_azure2019"

run_export full huawei "$RHOS_FULL" "$SEEDS_XTRACE" "$FULL_HW_JOBS" "$FULL_HW_MAN" "$FINALISTS"
run_des full huawei "$FULL_HW_JOBS" "$FULL_HW_DES"
summarize_trace full huawei "$FULL_HW_MAN" "$FULL_HW_DES" "T_r33_adaptive_ewma_full_huawei"

compact_full_results
echo "R33 adaptive EWMA run finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
