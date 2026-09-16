#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/260715/serverless-fewshot"
cd "$ROOT"

PY="python3.13"
ENV_PYTHONPATH="/data/260715/site-packages-des:."
RUNS="$ROOT/results/runs"
TABLES="$ROOT/results/tables"
LOG_DIR="$RUNS/r32_lifecycle_guarded_logs"
mkdir -p "$RUNS" "$TABLES" "$LOG_DIR"

AZ21_BASE="$RUNS/revision_r31_fullrho_tune6_top6_fast_runs.json"
AZ19_BASE="$RUNS/revision_r21_azure2019_full_faithful_fast_runs.json"
HW_BASE_MIXED="$RUNS/revision_r21_huawei_mixed_sparse_faithful_fast_runs.json"
HW_BASE_SAT="$RUNS/revision_r21_huawei_hsaturated_full_faithful_fast_runs.json"

SEEDS_AZ21="0,1,2,3,4,5,6,7,8,9"
SEEDS_XTRACE="0,1,2"
RHOS_SCREEN="10"
RHOS_FULL="0.1,1,10,100"
WORKERS="${R32_WORKERS:-6}"

COMMON="reset_mode=context,trigger_mode=signed_under,stable_exit=30,trigger_alpha=0.1,l2=0.01,refit_every=10,buffer_len=120,shadow_rho=10"
P="trigger_delta=0.25,trigger_m=3,trigger_calib_size=120,trigger_window=30,trigger_min_calib=30,min_under_log=0.0"
S="trigger_delta=0.35,trigger_m=5,trigger_calib_size=240,trigger_window=90,trigger_min_calib=120,min_under_log=0.25"

CANDIDATES=(
  "name=LC0_direct_P,$COMMON,$P,shadow_window=0,k_drift=20,recovery_ttl=60,cooldown=3600,min_miss_gain=1,max_extra_prewarm_ratio=1.0"
  "name=LC1_direct_S,$COMMON,$S,shadow_window=0,k_drift=20,recovery_ttl=30,cooldown=7200,min_miss_gain=1,max_extra_prewarm_ratio=1.0"
  "name=LC2_S10_cost_P,$COMMON,$P,shadow_window=10,k_drift=20,recovery_ttl=30,cooldown=3600,min_miss_gain=1,max_extra_prewarm_ratio=0.25"
  "name=LC3_S10_bal_P,$COMMON,$P,shadow_window=10,k_drift=20,recovery_ttl=30,cooldown=3600,min_miss_gain=1,max_extra_prewarm_ratio=0.50"
  "name=LC4_S10_cost_S,$COMMON,$S,shadow_window=10,k_drift=20,recovery_ttl=30,cooldown=7200,min_miss_gain=1,max_extra_prewarm_ratio=0.25"
  "name=LC5_S10_K100_P,$COMMON,$P,shadow_window=10,k_drift=100,recovery_ttl=30,cooldown=3600,min_miss_gain=1,max_extra_prewarm_ratio=0.50"
  "name=LC6_S20_cost_P,$COMMON,$P,shadow_window=20,k_drift=20,recovery_ttl=60,cooldown=3600,min_miss_gain=1,max_extra_prewarm_ratio=0.25"
  "name=LC7_S20_bal_P,$COMMON,$P,shadow_window=20,k_drift=20,recovery_ttl=60,cooldown=3600,min_miss_gain=1,max_extra_prewarm_ratio=0.50"
  "name=LC8_S20_gain_P,$COMMON,$P,shadow_window=20,k_drift=20,recovery_ttl=60,cooldown=3600,min_miss_gain=5,max_extra_prewarm_ratio=0.50"
  "name=LC9_S20_bal_S,$COMMON,$S,shadow_window=20,k_drift=20,recovery_ttl=60,cooldown=7200,min_miss_gain=1,max_extra_prewarm_ratio=0.50"
  "name=LC10_S20_K100_P,$COMMON,$P,shadow_window=20,k_drift=100,recovery_ttl=60,cooldown=3600,min_miss_gain=1,max_extra_prewarm_ratio=0.50"
  "name=LC11_S20_K100_S,$COMMON,$S,shadow_window=20,k_drift=100,recovery_ttl=60,cooldown=7200,min_miss_gain=5,max_extra_prewarm_ratio=0.50"
  "name=LC12_S20_csr_P,$COMMON,$P,shadow_window=20,k_drift=20,recovery_ttl=60,cooldown=3600,min_miss_gain=1,max_extra_prewarm_ratio=1.00"
  "name=LC13_S10_csr_P,$COMMON,$P,shadow_window=10,k_drift=20,recovery_ttl=30,cooldown=3600,min_miss_gain=1,max_extra_prewarm_ratio=1.00"
)

candidate_args_all() {
  local out=()
  for cand in "${CANDIDATES[@]}"; do
    out+=(--candidate "$cand")
  done
  printf '%s\n' "${out[@]}"
}

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

  if [[ -n "$cand_file" ]]; then
    mapfile -t cand_args < <(candidate_args_from_file "$cand_file")
  else
    mapfile -t cand_args < <(candidate_args_all)
  fi

  if [[ -s "$manifest" && -d "$jobs_root" ]]; then
    echo "[$stage][$trace] export exists; reusing $jobs_root"
    return
  fi

  echo "[$stage][$trace] export start"
  PYTHONPATH="$ENV_PYTHONPATH" "$PY" -u scripts/revision_r31_drift_qualified_gate.py \
    --export \
    --trace "$trace" \
    --preset none \
    "${cand_args[@]}" \
    --rhos "$rhos" \
    --seeds "$seeds" \
    --jobs-root "$jobs_root" \
    --manifest-out "$manifest" \
    --device cuda \
    "${extra_args[@]}"
  echo "[$stage][$trace] export done"
}

prune_jobs_to_candidates() {
  local manifest="$1"
  local jobs_root="$2"
  "$PY" - "$manifest" "$jobs_root" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.load(open(sys.argv[1]))
names = {str(c["name"]) for c in manifest["candidates"]}
root = Path(sys.argv[2])
for path in sorted(root.glob("*/jobs.json")):
    meta = json.load(open(path))
    before = len(meta["jobs"])
    meta["jobs"] = [j for j in meta["jobs"] if str(j["method"]) in names]
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"pruned {path}: {before} -> {len(meta['jobs'])}")
PY
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
  local inputs=()
  case "$trace" in
    azure2021)
      inputs=("$AZ21_BASE" "$candidate_json")
      ;;
    azure2019)
      inputs=("$AZ19_BASE" "$candidate_json")
      ;;
    huawei)
      inputs=("$HW_BASE_MIXED" "$HW_BASE_SAT" "$candidate_json")
      ;;
    *)
      echo "unknown trace $trace" >&2
      exit 2
      ;;
  esac
  echo "[$stage][$trace] summarize start"
  PYTHONPATH="$ENV_PYTHONPATH" "$PY" -u scripts/revision_r31_drift_qualified_gate.py \
    --summarize \
    --inputs "${inputs[@]}" \
    --manifest "$manifest" \
    --prefix "$prefix"
  echo "[$stage][$trace] summarize done"
}

select_finalists() {
  local out_file="$1"
  "$PY" - "$out_file" "$RUNS/revision_r32_lifecycle_guarded_screen_azure2021_manifest.json" <<'PY'
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
    "azure2021": Path("results/tables/T_r32_lifecycle_guarded_screen_azure2021_rho10.csv"),
    "azure2019": Path("results/tables/T_r32_lifecycle_guarded_screen_azure2019_rho10.csv"),
    "huawei": Path("results/tables/T_r32_lifecycle_guarded_screen_huawei_rho10.csv"),
}
frames = []
for dataset, path in paths.items():
    df = pd.read_csv(path)
    df = df[df["method"].astype(str).str.startswith("LC")].copy()
    df["dataset"] = dataset
    frames.append(df)
all_rows = pd.concat(frames, ignore_index=True)
grouped = all_rows.groupby("method", dropna=False).agg(
    cells=("rho", "size"),
    ewma_nonworse=("delta_csr_pp_vs_ewma", lambda s: int((s <= 0.0).sum())),
    current_nonworse=("delta_csr_pp_vs_current", lambda s: int((s <= 0.0).sum())),
    mean_dcsr=("delta_csr_pp_vs_ewma", "mean"),
    max_dcsr=("delta_csr_pp_vs_ewma", "max"),
    max_dcost=("delta_cost_pct_vs_ewma", "max"),
    mean_dcost=("delta_cost_pct_vs_ewma", "mean"),
    max_current_dcsr=("delta_csr_pp_vs_current", "max"),
    mean_recovery=("route_recovery", "mean"),
    mean_recovery_learned=("route_recovery_learned", "mean"),
    mean_shadow_commits=("mean_shadow_commits_per_function", "mean"),
    mean_shadow_rejects=("mean_shadow_rejects_per_function", "mean"),
).reset_index()
grouped["strict_pass"] = (
    (grouped["max_dcost"] <= 3.0)
    & (grouped["max_dcsr"] <= 0.05)
    & (grouped["max_current_dcsr"] <= 0.01)
)
ranked = grouped.sort_values(
    by=[
        "strict_pass",
        "ewma_nonworse",
        "current_nonworse",
        "max_dcsr",
        "mean_dcsr",
        "max_dcost",
        "mean_recovery_learned",
    ],
    ascending=[False, False, False, True, True, True, True],
).reset_index(drop=True)
ranked.to_csv("results/tables/T_r32_lifecycle_guarded_screening_rank.csv", index=False)
finalists = ranked[ranked["strict_pass"]]["method"].tolist()[:5]
if len(finalists) < 3:
    for name in ranked["method"].tolist():
        if name not in finalists:
            finalists.append(name)
        if len(finalists) >= 5:
            break
selection = {
    "criteria": {
        "cost_cap_pct_vs_ewma": 3.0,
        "max_csr_regression_pp_vs_ewma": 0.05,
        "max_csr_regression_pp_vs_current": 0.01,
        "rank_order": [
            "strict_pass desc",
            "ewma_nonworse desc",
            "current_nonworse desc",
            "max_dcsr asc",
            "mean_dcsr asc",
            "max_dcost asc",
            "mean_recovery_learned asc",
        ],
    },
    "strict_pass_count": int(ranked["strict_pass"].sum()),
    "finalists": finalists,
}
Path("results/runs/revision_r32_lifecycle_guarded_screening_selection.json").write_text(
    json.dumps(selection, indent=1)
)

def raw_candidate(rec):
    keys = [
        "name", "reset_mode", "trigger_mode", "k_drift", "recovery_ttl",
        "stable_exit", "cooldown", "shadow_window", "min_miss_gain",
        "max_extra_prewarm_ratio", "shadow_rho", "trigger_alpha",
        "trigger_delta", "trigger_m", "trigger_calib_size",
        "trigger_window", "trigger_min_calib", "min_under_log", "l2",
        "refit_every", "buffer_len",
    ]
    return ",".join(f"{k}={rec[k]}" for k in keys if k in rec)

with open(out_file, "w") as f:
    for name in finalists:
        f.write(raw_candidate(cand_map[name]) + "\n")
print("selected finalists:", ",".join(finalists))
print("strict_pass_count:", int(ranked["strict_pass"].sum()))
PY
}

compact_full_results() {
  "$PY" <<'PY'
import json
from pathlib import Path

import pandas as pd

finalists_path = Path("results/runs/revision_r32_lifecycle_guarded_finalists.txt")
finalists = []
for line in finalists_path.read_text().splitlines():
    if not line.strip():
        continue
    parts = dict(part.split("=", 1) for part in line.split(",") if "=" in part)
    finalists.append(parts["name"])
paths = {
    "azure2021": Path("results/tables/T_r32_lifecycle_guarded_full_azure2021_fullrho.csv"),
    "azure2019": Path("results/tables/T_r32_lifecycle_guarded_full_azure2019_fullrho.csv"),
    "huawei": Path("results/tables/T_r32_lifecycle_guarded_full_huawei_fullrho.csv"),
}
rows = []
ci_rows = []
for dataset, path in paths.items():
    df = pd.read_csv(path)
    df = df[df["method"].isin(finalists)].copy()
    df["dataset"] = dataset
    rows.append(df)
    ci = df[df["rho"].astype(float) == 10.0][[
        "dataset", "split", "method", "rho", "delta_csr_pp_vs_ewma",
        "boot_ci95_lo_pp_vs_ewma", "boot_ci95_hi_pp_vs_ewma",
        "delta_cost_pct_vs_ewma", "delta_csr_pp_vs_current",
        "delta_cost_pct_vs_current", "route_recovery",
        "route_recovery_learned", "route_shadow_candidate",
        "mean_shadow_commits_per_function", "mean_shadow_rejects_per_function",
    ]]
    ci_rows.append(ci)
all_rows = pd.concat(rows, ignore_index=True)
compact_cols = [
    "dataset", "split", "method", "rho", "csr_pct",
    "delta_csr_pp_vs_ewma", "delta_cost_pct_vs_ewma",
    "delta_csr_pp_vs_current", "delta_cost_pct_vs_current",
    "route_recovery", "route_recovery_learned", "route_shadow_candidate",
    "mean_fires_per_function", "mean_shadow_commits_per_function",
    "mean_shadow_rejects_per_function", "mean_recovery_fits_per_function",
]
all_rows[compact_cols].to_csv(
    "results/tables/T_r32_lifecycle_guarded_cross_trace_compact_fullrho.csv",
    index=False,
)
pd.concat(ci_rows, ignore_index=True).to_csv(
    "results/tables/T_r32_lifecycle_guarded_finalist_ci_rho10.csv",
    index=False,
)
rank = all_rows.groupby("method", dropna=False).agg(
    cells=("rho", "size"),
    ewma_nonworse=("delta_csr_pp_vs_ewma", lambda s: int((s <= 0.0).sum())),
    current_nonworse=("delta_csr_pp_vs_current", lambda s: int((s <= 0.0).sum())),
    mean_dcsr=("delta_csr_pp_vs_ewma", "mean"),
    max_dcsr=("delta_csr_pp_vs_ewma", "max"),
    mean_dcost=("delta_cost_pct_vs_ewma", "mean"),
    max_dcost=("delta_cost_pct_vs_ewma", "max"),
    mean_recovery_learned=("route_recovery_learned", "mean"),
).reset_index().sort_values(
    ["ewma_nonworse", "current_nonworse", "max_dcsr", "mean_dcsr", "max_dcost"],
    ascending=[False, False, True, True, True],
)
rank.to_csv("results/tables/T_r32_lifecycle_guarded_finalist_rank_fullrho.csv", index=False)
summary = {
    "finalists": finalists,
    "rank_rows": json.loads(rank.to_json(orient="records")),
}
Path("results/runs/revision_r32_lifecycle_guarded_final_summary.json").write_text(
    json.dumps(summary, indent=1)
)
print(rank.to_string(index=False))
PY
}

echo "R32 lifecycle guarded run started at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
PYTHONPATH="$ENV_PYTHONPATH" "$PY" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

bash -n scripts/run_r32_lifecycle_guarded_experiments.sh
PYTHONPATH="$ENV_PYTHONPATH" "$PY" -m py_compile scripts/revision_r31_drift_qualified_gate.py scripts/des_runner_fast_parallel.py

SCREEN_AZ21_JOBS="$RUNS/des_jobs_r32_lifecycle_guarded_screen_azure2021"
SCREEN_AZ19_JOBS="$RUNS/des_jobs_r32_lifecycle_guarded_screen_azure2019"
SCREEN_HW_JOBS="$RUNS/des_jobs_r32_lifecycle_guarded_screen_huawei"
SCREEN_AZ21_MAN="$RUNS/revision_r32_lifecycle_guarded_screen_azure2021_manifest.json"
SCREEN_AZ19_MAN="$RUNS/revision_r32_lifecycle_guarded_screen_azure2019_manifest.json"
SCREEN_HW_MAN="$RUNS/revision_r32_lifecycle_guarded_screen_huawei_manifest.json"
SCREEN_AZ21_DES="$RUNS/revision_r32_lifecycle_guarded_screen_azure2021_fast_runs.json"
SCREEN_AZ19_DES="$RUNS/revision_r32_lifecycle_guarded_screen_azure2019_fast_runs.json"
SCREEN_HW_DES="$RUNS/revision_r32_lifecycle_guarded_screen_huawei_fast_runs.json"

run_export screen azure2021 "$RHOS_SCREEN" "$SEEDS_AZ21" "$SCREEN_AZ21_JOBS" "$SCREEN_AZ21_MAN"
prune_jobs_to_candidates "$SCREEN_AZ21_MAN" "$SCREEN_AZ21_JOBS"
run_des screen azure2021 "$SCREEN_AZ21_JOBS" "$SCREEN_AZ21_DES"
summarize_trace screen azure2021 "$SCREEN_AZ21_MAN" "$SCREEN_AZ21_DES" "T_r32_lifecycle_guarded_screen_azure2021"

run_export screen azure2019 "$RHOS_SCREEN" "$SEEDS_XTRACE" "$SCREEN_AZ19_JOBS" "$SCREEN_AZ19_MAN"
prune_jobs_to_candidates "$SCREEN_AZ19_MAN" "$SCREEN_AZ19_JOBS"
run_des screen azure2019 "$SCREEN_AZ19_JOBS" "$SCREEN_AZ19_DES"
summarize_trace screen azure2019 "$SCREEN_AZ19_MAN" "$SCREEN_AZ19_DES" "T_r32_lifecycle_guarded_screen_azure2019"

run_export screen huawei "$RHOS_SCREEN" "$SEEDS_XTRACE" "$SCREEN_HW_JOBS" "$SCREEN_HW_MAN"
prune_jobs_to_candidates "$SCREEN_HW_MAN" "$SCREEN_HW_JOBS"
run_des screen huawei "$SCREEN_HW_JOBS" "$SCREEN_HW_DES"
summarize_trace screen huawei "$SCREEN_HW_MAN" "$SCREEN_HW_DES" "T_r32_lifecycle_guarded_screen_huawei"

FINALISTS="$RUNS/revision_r32_lifecycle_guarded_finalists.txt"
select_finalists "$FINALISTS"

FULL_AZ21_JOBS="$RUNS/des_jobs_r32_lifecycle_guarded_full_azure2021"
FULL_AZ19_JOBS="$RUNS/des_jobs_r32_lifecycle_guarded_full_azure2019"
FULL_HW_JOBS="$RUNS/des_jobs_r32_lifecycle_guarded_full_huawei"
FULL_AZ21_MAN="$RUNS/revision_r32_lifecycle_guarded_full_azure2021_manifest.json"
FULL_AZ19_MAN="$RUNS/revision_r32_lifecycle_guarded_full_azure2019_manifest.json"
FULL_HW_MAN="$RUNS/revision_r32_lifecycle_guarded_full_huawei_manifest.json"
FULL_AZ21_DES="$RUNS/revision_r32_lifecycle_guarded_full_azure2021_fast_runs.json"
FULL_AZ19_DES="$RUNS/revision_r32_lifecycle_guarded_full_azure2019_fast_runs.json"
FULL_HW_DES="$RUNS/revision_r32_lifecycle_guarded_full_huawei_fast_runs.json"

run_export full azure2021 "$RHOS_FULL" "$SEEDS_AZ21" "$FULL_AZ21_JOBS" "$FULL_AZ21_MAN" "$FINALISTS"
prune_jobs_to_candidates "$FULL_AZ21_MAN" "$FULL_AZ21_JOBS"
run_des full azure2021 "$FULL_AZ21_JOBS" "$FULL_AZ21_DES"
summarize_trace full azure2021 "$FULL_AZ21_MAN" "$FULL_AZ21_DES" "T_r32_lifecycle_guarded_full_azure2021"

run_export full azure2019 "$RHOS_FULL" "$SEEDS_XTRACE" "$FULL_AZ19_JOBS" "$FULL_AZ19_MAN" "$FINALISTS"
prune_jobs_to_candidates "$FULL_AZ19_MAN" "$FULL_AZ19_JOBS"
run_des full azure2019 "$FULL_AZ19_JOBS" "$FULL_AZ19_DES"
summarize_trace full azure2019 "$FULL_AZ19_MAN" "$FULL_AZ19_DES" "T_r32_lifecycle_guarded_full_azure2019"

run_export full huawei "$RHOS_FULL" "$SEEDS_XTRACE" "$FULL_HW_JOBS" "$FULL_HW_MAN" "$FINALISTS"
prune_jobs_to_candidates "$FULL_HW_MAN" "$FULL_HW_JOBS"
run_des full huawei "$FULL_HW_JOBS" "$FULL_HW_DES"
summarize_trace full huawei "$FULL_HW_MAN" "$FULL_HW_DES" "T_r32_lifecycle_guarded_full_huawei"

compact_full_results
echo "R32 lifecycle guarded run finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
