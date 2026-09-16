#!/bin/bash
# Revision watcher v2 (2026-07-18): waits for the recovery chain
# (run_resume_89.sh PID 18636, steps 8-9 + followup) to exit, then runs the
# queued revision DES jobs with ONE RUNNER PER SPLIT-DIR IN PARALLEL
# (6 runners; each split dir is a self-contained jobs dir). Outputs are
# per-split JSONs merged at the end into the two canonical files:
#   revision_v1_hybridfull_2019.json  (V1 full hybrid histogram)
#   revision_e1_gates_2019.json       (E1 gated v3/v4)
set -uo pipefail
cd /data/260715/serverless-fewshot
DESPP=/data/260715/site-packages-des:.
PY=python3.13
LOG=results/logs

echo "[watcher] waiting for chain PID 18636 to exit ($(date '+%F %T'))"
while kill -0 18636 2>/dev/null; do sleep 120; done
echo "[watcher] chain exited ($(date '+%F %T')); resume log tail:"
tail -3 $LOG/run_resume_89.log 2>/dev/null || true

pids=()
# e1's S1/S3 are pre-split by gate method into e1a/e1b (4 jobs each) so the
# longest runner is ~4 jobs; e1/S2 is fast enough to stay whole.
for pair in v1:S1 v1:S2 v1:S3 e1:S2 e1a:S1 e1a:S3 e1b:S1 e1b:S3; do
    set="${pair%%:*}"; sp="${pair##*:}"
    jdir=$PWD/results/runs/des_jobs_2019_${set}/${sp}
    out=results/runs/revision_${set}_2019_${sp}.json
    [ -d "$jdir" ] || { echo "[watcher] missing $jdir"; continue; }
    # runner iterates subdirs of --jobs; wrap the single split in a root
    wrap=results/runs/split_jobs_${set}_${sp}
    mkdir -p "$wrap"
    ln -sfn "$jdir" "$wrap/${sp}"
    PYTHONPATH=$DESPP nice -n 5 $PY -u scripts/des_runner_fast.py \
        --jobs "$wrap" --out "$out" \
        > $LOG/revision_${set}_des_2019_${sp}.log 2>&1 &
    pids+=($!)
    echo "[watcher] launched ${set}/${sp} (pid ${pids[-1]})"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[watcher] all runners finished ($(date '+%F %T')), fail=$fail"

$PY - <<'PYEOF'
import json
merged = {"v1": ("revision_v1_hybridfull_2019.json", []),
          "e1": ("revision_e1_gates_2019.json", [])}
PARTS = [("v1", "v1", "S1"), ("v1", "v1", "S2"), ("v1", "v1", "S3"),
         ("e1", "e1", "S2"), ("e1", "e1a", "S1"), ("e1", "e1a", "S3"),
         ("e1", "e1b", "S1"), ("e1", "e1b", "S3")]
for s, tag, sp in PARTS:
    p = f"results/runs/revision_{tag}_2019_{sp}.json"
    try:
        part = json.load(open(p))
        rows = part["results"] if isinstance(part, dict) and "results" in part else part
        merged[s][1].extend(rows)
        print(f"[merge] {p}: {len(rows)} rows")
    except Exception as e:
        print(f"[merge] {p} FAILED: {e}")
for s, (name, rows) in merged.items():
    out = f"results/runs/{name}"
    json.dump(rows, open(out, "w"), default=float)
    blob = open(out, "rb").read()
    assert blob and blob.count(0) == 0
    json.load(open(out))
    print(f"[merge] saved+verified {out} ({len(rows)} rows)")
PYEOF
echo "[watcher] ALL DONE ($(date '+%F %T'))"
