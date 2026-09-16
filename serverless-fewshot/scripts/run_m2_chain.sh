#!/bin/bash
# Chain the remaining M2 campaigns after the running holdout job finishes.
#   1. binned rho=10 pass  (pre-registration amendment K.2: S4/S5 windows)
#   2. M2-B 2021 age sweep (reviewer condition 1, second half)
cd /data/260715/serverless-fewshot
export PYTHONPATH=/data/260715/site-packages:.

while pgrep -f "revision_m2a_holdout_cohort.py --workers" >/dev/null; do sleep 60; done
echo "=== main holdout done $(date -u +%H:%M) ==="

SF_DATA_DIR=processed_2019 python3.13 -u scripts/revision_m2a_holdout_cohort.py \
    --rho10only --workers 8 > results/logs/m2a_binned.log 2>&1
echo "=== binned pass done $(date -u +%H:%M) rc=$? ==="

python3.13 -u scripts/revision_m2b_agesweep_2021.py \
    > results/logs/m2b_agesweep.log 2>&1
echo "=== m2b done $(date -u +%H:%M) rc=$? ==="
