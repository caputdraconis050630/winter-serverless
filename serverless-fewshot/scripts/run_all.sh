#!/usr/bin/env bash
# Full reproduction pipeline. Seeds are fixed inside each script.
# GPU required for phases 4/6/8; Kind+Knative cluster required for phase 7.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=/data/260715/site-packages:$PYTHONPATH
PY=python3.13

echo "== Phase 1: preprocessing =="
$PY scripts/phase1_preprocess.py

echo "== Phase 2: tasks & splits =="
$PY scripts/phase2_tasks.py

echo "== Phase 4: meta-training (seed 0) + GO gate =="
$PY scripts/phase4_train.py

echo "== Phase 4b: training seeds 1,2 =="
$PY scripts/phase4_multiseed.py

echo "== WP1 validation: simulator invariants =="
$PY scripts/validate_simulator.py

echo "== Phase 6: steady-state DES sweep (5 seeds x 3 splits) =="
$PY scripts/phase6_des.py --splits S1,S2,S3 --seeds 5

echo "== Phase 6b: hybrid gate arm =="
$PY scripts/phase65_gate.py

echo "== Phase 6c: paired statistics + decision gate =="
$PY scripts/phase6_stats.py

echo "== Phase 6d: model-seed robustness =="
$PY scripts/phase6_modelseeds.py

echo "== Phase 6.3: onboarding + drift + trigger benchmark =="
$PY scripts/phase63_onboarding_drift.py

echo "== Phase 7 (requires Kind+Knative; see testbed/) =="
echo "   $PY scripts/testbed_measure.py   # cold/warm CDFs -> T6, calibration"
echo "   $PY scripts/testbed_e2e.py       # A5 vs reactive end-to-end -> T6b"

echo "== Phase 8: head ablation (measured heatmap) =="
$PY scripts/phase8_heads.py

echo "== Phase 8b: scalability + statistics =="
$PY scripts/phase8_ablations.py

echo "== Phase 9: paper figures & tables =="
$PY scripts/phase9_paper.py

echo "ALL DONE. Figures: results/figures, Tables: results/tables"
