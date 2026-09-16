#!/usr/bin/env bash
# Chain v2: gate the 4.3 h cohort run on the smoke run's QUALITY, not just on
# it having produced a file.
#
# The first smoke attempt exited 0 and wrote a result that was badly wrong:
# every arm identical, cold-classifier agreement 0.60, and the autoscaler
# silently left at Knative defaults because the webhook had rejected the
# requested grace period. "A file exists" is therefore not a pass condition.
# This gate checks the two things that were actually broken.
set -uo pipefail
cd /data/260715/serverless-fewshot
export PYTHONPATH=/data/260715/site-packages:/data/260715/site-packages-des
LOG=/tmp/claude-1000/-data-260715/283d29e5-4028-4fe3-bfc3-b75155cc6721/scratchpad
RUNS=results/runs

echo "[chain2] waiting for the smoke run to exit..."
while pgrep -f 'python3\.13 scripts/testbed_cohort\.py --minutes' >/dev/null; do
  sleep 15
done

if [ ! -f "$RUNS/testbed_cohort.json" ]; then
  echo "[chain2] ABORT: smoke run produced no result"; exit 1
fi

python3.13 - <<'PY'
import json, sys
d = json.load(open("results/runs/testbed_cohort.json"))
auto = d["constants"]["autoscaler"]
nf = d["n_functions"]
agree = min(a["classifier_agreement"] for a in d["arms"].values())
fails = sum(a["failed_requests"] for a in d["arms"].values())
print(f"[gate] autoscaler={auto!r} min_agreement={agree:.2f} failed_requests={fails}")
ok = True
if not auto.startswith("6s"):
    print("[gate] FAIL: autoscaler was not pinned tight; arms would blur"); ok = False
if agree < 0.90:
    print("[gate] FAIL: cold classifiers disagree; event timing still wrong"); ok = False
if fails:
    print("[gate] FAIL: some requests errored"); ok = False
# max-scale is 1, so a service cannot hold more than one warm pod: total
# pod-seconds can never exceed wall time x cohort size. The first fixed
# classifier still broke this (994 against a 720 ceiling) by counting
# draining pods, so check the arithmetic rather than trusting the code.
for name, a in d["arms"].items():
    ceil = a["wall_sec"] * nf * 1.05
    if a["pod_seconds"] > ceil:
        print(f"[gate] FAIL: {name} pod-seconds {a['pod_seconds']:.0f} "
              f"exceeds ceiling {ceil:.0f}")
        ok = False
if ok:
    print("[gate] arm CSR(event): " + ", ".join(
        f"{k}={v['csr_event']*100:.2f}%" for k, v in d["arms"].items()))
sys.exit(0 if ok else 2)
PY
if [ $? -ne 0 ]; then
  echo "[chain2] ABORT: smoke quality gate failed, not starting the 4 h run"
  exit 2
fi
echo "[chain2] smoke quality gate passed"
rm -f "$RUNS/testbed_cohort.json"

echo "[chain2] full run (10 funcs x 60 min x 4 arms)"
python3.13 scripts/testbed_cohort.py > "$LOG/e5.log" 2>&1
if [ ! -f "$RUNS/testbed_cohort.json" ]; then
  echo "[chain2] ABORT: full run failed, see e5.log"; exit 1
fi
echo "[chain2] E5 done."

echo "[chain2] DES mirror"
python3.13 scripts/testbed_cohort_desmirror.py > "$LOG/e6.log" 2>&1
if [ ! -f "$RUNS/testbed_cohort_desmirror.json" ]; then
  echo "[chain2] ABORT: mirror failed, see e6.log"; exit 1
fi
echo "[chain2] E6 done. all complete."
