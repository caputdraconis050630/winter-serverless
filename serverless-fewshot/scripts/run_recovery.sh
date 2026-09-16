#!/bin/bash
# Recovery chain after 2026-07-16 disk fault: rebuild 2019 pipeline end-to-end.
# Each step: run -> verify outputs readable -> sync. Aborts on first failure.
set -uo pipefail
cd /data/260715/serverless-fewshot
export PYTHONPATH=/data/260715/site-packages
DESPP=/data/260715/site-packages-des:.
LOG=results/logs
PY=python3.13

step() { echo ""; echo "===== [$(date '+%F %T')] $1 ====="; }
fail() { echo "!!!!! RECOVERY FAILED at: $1 (exit $2)"; exit 1; }

step "1/9 phase1b (2019 preprocess)"
$PY -u scripts/phase1b_azure2019.py > $LOG/phase1b_2019_rerun.log 2>&1 || fail phase1b $?
$PY - <<'PYEOF' || fail phase1b-verify $?
import numpy as np, json
d="data/processed_2019/"
m=json.load(open(d+"meta.json")); assert m["total_invocations"]==12495810846, m
f=np.load(d+"features.npy",mmap_mode="r"); c=np.load(d+"counts.npy",mmap_mode="r")
assert f.shape[0]==30000 and c.shape[0]==30000, (f.shape, c.shape)
assert abs(f[:100].sum())>0 and abs(f[-100:].sum())>0
print("phase1b verify OK", f.shape, c.shape)
PYEOF
sync

step "2/9 phase2 (tasks/splits)"
SF_DATA_DIR=processed_2019 $PY -u scripts/phase2_tasks.py > $LOG/phase2_2019_rerun.log 2>&1 || fail phase2 $?
$PY - <<'PYEOF' || fail phase2-verify $?
import numpy as np
s=np.load("data/processed_2019/splits.npz",allow_pickle=True)
info={k:len(np.atleast_1d(s[k])) for k in s.files}
b=open("results/tables/T2_drift_catalog.csv","rb").read(4096)
assert len(b)>0 and b.count(0)==0
print("phase2 verify OK", info)
PYEOF
sync

step "3/9 phase4_train seed0 + B5 + gate (NO-GO exit is expected)"
SF_DATA_DIR=processed_2019 $PY -u scripts/phase4_train.py > $LOG/phase4_2019_rerun.log 2>&1
echo "phase4_train exit=$? (1 = NO-GO, expected on 2019)"
$PY - <<'PYEOF' || fail phase4-verify $?
import json, torch
g=json.load(open("results/runs/go_nogo_gate.json")); print("gate:", g["decision"], g["ours_crps"], g["global_crps"])
for f in ["results/runs/best_anil_ridge_s2_s0.pt","results/runs/b5_global_s0.pt"]:
    torch.load(f, map_location="cpu", weights_only=False)
print("phase4 verify OK")
PYEOF
sync

step "4/9 phase4_multiseed (seeds 1,2)"
SF_DATA_DIR=processed_2019 $PY -u scripts/phase4_multiseed.py > $LOG/phase4_multiseed_2019_rerun.log 2>&1 || fail phase4_multiseed $?
$PY - <<'PYEOF' || fail multiseed-verify $?
import json, torch
m=json.load(open("results/runs/multiseed_training.json")); print("multiseed:", m)
for s in (1,2): torch.load(f"results/runs/best_anil_ridge_s2_s{s}.pt", map_location="cpu", weights_only=False)
print("multiseed verify OK")
PYEOF
sync

step "5/9 diag_gate_2019"
SF_DATA_DIR=processed_2019 $PY -u scripts/diag_gate_2019.py > $LOG/diag_gate_2019_rerun.log 2>&1 || fail diag $?
$PY -c "import json; json.load(open('results/runs/diag_gate2019_ksweep.json')); print('diag verify OK')" || fail diag-verify $?
sync

step "6/9 router_analysis 2019 + 2021"
SF_DATA_DIR=processed_2019 SF_TAG=2019 $PY -u scripts/router_analysis.py > $LOG/router_analysis_2019_rerun.log 2>&1 || fail router2019 $?
SF_DATA_DIR=processed SF_RUNS_DIR=results_azure2021/runs SF_TAG=azure2021 $PY -u scripts/router_analysis.py > $LOG/router_analysis_2021_rerun.log 2>&1 || fail router2021 $?
$PY - <<'PYEOF' || fail router-verify $?
import json
for t in ("2019","azure2021"):
    r=json.load(open(f"results/runs/router_analysis_{t}.json"))
    print(t, "router keys:", list(r)[:6])
print("router verify OK")
PYEOF
sync

step "7/9 DES export (jobs for S1/S2/S3)"
SF_DATA_DIR=processed_2019 $PY -u scripts/phase6_des_sampled.py --export-jobs results/runs/des_jobs_2019 > $LOG/des_export_2019_rerun.log 2>&1 || fail des-export $?
$PY - <<'PYEOF' || fail des-export-verify $?
import numpy as np, json, glob
jobs=sorted(glob.glob("results/runs/des_jobs_2019/**/*.npz", recursive=True))
assert len(jobs)>0, "no job npz found"
for f in jobs:
    d=np.load(f, allow_pickle=True); [d[k].shape if hasattr(d[k],'shape') else d[k] for k in d.files]
for f in glob.glob("results/runs/des_jobs_2019/**/*.json", recursive=True):
    json.load(open(f))
m=json.load(open("results/runs/des_sample_manifest.json"))
print("des export verify OK:", len(jobs), "npz files; manifest splits:", list(m))
PYEOF
sync

step "8/9 DES run (numba env)"
PYTHONPATH=$DESPP $PY -u scripts/des_runner_fast.py --jobs results/runs/des_jobs_2019 --out results/runs/sim_results_des_2019.json > $LOG/des_run_2019_rerun.log 2>&1 || fail des-run $?
$PY -c "import json; r=json.load(open('results/runs/sim_results_des_2019.json')); print('des run verify OK, entries:', len(r))" || fail des-run-verify $?
sync

step "9/9 DES stats (T3 table)"
$PY -u scripts/des_2019_stats.py > $LOG/des_stats_2019_rerun.log 2>&1 || fail des-stats $?
$PY - <<'PYEOF' || fail des-stats-verify $?
import json
s=json.load(open("results/runs/des_2019_stats.json"))
b=open("results/tables/T3_des_2019.csv","rb").read()
assert len(b)>0 and b.count(0)==0
print("des stats verify OK")
PYEOF
sync

echo ""
echo "===== [$(date '+%F %T')] RECOVERY CHAIN COMPLETE ====="
