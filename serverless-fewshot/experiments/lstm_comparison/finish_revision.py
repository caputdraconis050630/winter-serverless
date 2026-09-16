"""Finish the measured manuscript after all already-running jobs succeed.

This coordinator does not generate or modify experimental observations.
It requires every planned result, then runs the recorded validation and
document commands. Rendered pages still need a final visual review.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/"results/lstm_comparison_v1"
PAPER = ROOT.parent/"winter-paper"
LOGS = OUT/"finalization"
EXPECTED = ["initial_azure_primary", "initial_huawei", "initial_azure_evaluation", "continuous_48h"]
EXPECTED += [f"steady_{p}_{s}" for p in ("azure2021", "azure2019", "huawei")
             for s in (["h_mixed", "h_sparse", "h_saturated"] if p == "huawei" else ["S1", "S2", "S3"])]
EXPECTED += [f"drift_{p}_{s}" for p in ("azure2021", "azure2019", "huawei")
             for s in ("natural", "synthetic")]


def state(status, **values):
    record = dict(status=status, updated_at_utc=datetime.now(timezone.utc).isoformat(),
                  coordinator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  **values)
    temporary = LOGS/"state.json.tmp"
    temporary.write_text(json.dumps(record, indent=2)+"\n")
    temporary.replace(LOGS/"state.json")
    print(json.dumps(record), flush=True)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--azure-pid", type=int, required=True)
    parser.add_argument("--huawei-pid", type=int, required=True)
    parser.add_argument("--live-pid", type=int, required=True)
    args = parser.parse_args()
    LOGS.mkdir(exist_ok=True)
    lock = LOGS/"coordinator.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit("A finalization coordinator already owns the lock")
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        while True:
            missing = [n for n in EXPECTED if not (OUT/"cases"/n/"summary.json").exists()]
            live_missing = not (OUT/"strict_round/live/results.json").exists()
            if not missing and not live_missing:
                break
            dependencies = [(args.azure_pid, any(n.startswith("steady_azure2019_") for n in missing)),
                            (args.huawei_pid, "steady_huawei_h_saturated" in missing),
                            (args.live_pid, live_missing)]
            for pid, needed in dependencies:
                if needed and not alive(pid):
                    raise RuntimeError(f"Required experiment process {pid} exited before its results were complete")
            state("waiting_for_measurements", missing_cases=missing, primary_live_pending=live_missing)
            time.sleep(60)

        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH="/data/260715/site-packages-des:/data/260715/site-packages")

        def run(label, script, cwd):
            state("running", step=label)
            with (LOGS/(label+".log")).open("w") as log:
                process = subprocess.Popen([sys.executable, str(script)], cwd=cwd, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                while True:
                    try:
                        code = process.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        state("running", step=label, process=process.pid)
                if code:
                    raise RuntimeError(f"{label} failed with exit status {code}; see {log.name}")

        run("tests", HERE/"check_tests.py", ROOT)
        run("measurements", HERE/"verify_results.py", ROOT)
        assert json.loads((OUT/"verification/report.json").read_text())["status"] == "passed"
        run("figures_tables", PAPER/"audit/build_lstm_revision.py", PAPER)
        run("manuscript", PAPER/"audit/check_manuscript.py", PAPER)
        backup = LOGS/("previous_pdfs_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
        backup.mkdir()
        public_files = [PAPER/(name+".pdf") for name in ("main", "supplementary", "highlights")]
        public_files += [PAPER/"audit/build_manifest.json", PAPER/"audit/visual/checks.json"]
        for path in public_files:
            if path.exists():
                target = backup/path.relative_to(PAPER)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        try:
            run("pdf_build", PAPER/"audit/build_paper.py", PAPER)
            run("pdf_boundaries", PAPER/"audit/inspect_pdfs.py", PAPER)
        except Exception:
            for path in public_files:
                saved = backup/path.relative_to(PAPER)
                if saved.exists():
                    shutil.copy2(saved, path)
            raise
        run("korean_report", PAPER/"audit/write_lstm_report.py", PAPER)
        build = json.loads((PAPER/"audit/build_manifest.json").read_text())
        state("built_pending_visual_review", build_id=build["build_id"],
              pages={n:d["pages"] for n,d in build["documents"].items()},
              rendered_pages=str(PAPER/"audit/visual"/build["build_id"]))
    except Exception as exc:
        state("failed", error=str(exc))
        traceback.print_exc()
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
