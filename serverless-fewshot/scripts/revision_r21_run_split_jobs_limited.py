#!/usr/bin/env python3
"""Run an exported R21 DES job bundle as small independent jobs.

The standard DES runner executes all jobs in a bundle sequentially.  This
helper creates one split-method-rho bundle per exported job and runs those
bundles with bounded process-level parallelism.  It preserves resumability
because each child uses revision_r21_des_runner_resume.py and writes one JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def safe_name(split: str, method: str, rho: float) -> str:
    rho_s = f"{rho:g}".replace(".", "p")
    return f"{split}__{method}__rho{rho_s}"


def prepare(job_root: Path, split_root: Path) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    split_root.mkdir(parents=True, exist_ok=True)
    for split_dir in sorted(job_root.iterdir()):
        meta_path = split_dir / "jobs.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        for job in meta["jobs"]:
            name = safe_name(meta["split"], job["method"], float(job["rho"]))
            dst = split_root / name / meta["split"]
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(split_dir / "shared.npz", dst / "shared.npz")
            shutil.copy2(split_dir / job["file"], dst / job["file"])
            one = dict(meta)
            one["jobs"] = [job]
            with open(dst / "jobs.json", "w") as f:
                json.dump(one, f, indent=2)
            tasks.append((name, split_root / name))
    return tasks


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    data = json.load(open(path))
    rows = data.get("results", data) if isinstance(data, dict) else data
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-root", required=True)
    ap.add_argument("--split-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--log-root", required=True)
    ap.add_argument("--max-procs", type=int, default=6)
    ap.add_argument("--pythonpath", default="/data/260715/site-packages-des:.")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    job_root = Path(args.jobs_root)
    split_root = Path(args.split_root)
    out_root = Path(args.out_root)
    log_root = Path(args.log_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    tasks = prepare(job_root, split_root)
    pending = [(name, root) for name, root in tasks if count_rows(out_root / f"{name}.json") < 3]
    print(f"prepared {len(tasks)} split jobs; pending {len(pending)}", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = args.pythonpath
    active: list[tuple[str, subprocess.Popen, object]] = []
    completed = 0
    failures: list[str] = []

    while pending or active:
        while pending and len(active) < args.max_procs:
            name, root = pending.pop(0)
            out = out_root / f"{name}.json"
            log = open(log_root / f"{name}.log", "ab")
            cmd = [
                sys.executable,
                "scripts/revision_r21_des_runner_resume.py",
                "--jobs",
                str(root),
                "--out",
                str(out),
            ]
            proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append((name, proc, log))
            print(f"started {name}", flush=True)

        time.sleep(5)
        still: list[tuple[str, subprocess.Popen, object]] = []
        for name, proc, log in active:
            rc = proc.poll()
            if rc is None:
                still.append((name, proc, log))
                continue
            log.close()
            if rc == 0:
                completed += 1
                print(f"finished {name} rows={count_rows(out_root / f'{name}.json')}", flush=True)
            else:
                failures.append(name)
                print(f"failed {name} rc={rc}", flush=True)
        active = still

    if failures:
        print("failures: " + ", ".join(failures), file=sys.stderr)
        return 1
    print(f"completed {completed} split jobs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
