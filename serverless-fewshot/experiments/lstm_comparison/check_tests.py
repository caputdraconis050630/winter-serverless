"""Run the independent scientific checks in fresh Python processes."""
from datetime import datetime, timezone
import os
import re
import subprocess
import sys
import time

from common import HERE, OUT, ROOT, digest, write_json

destination = OUT / "verification"
destination.mkdir(exist_ok=True)
tests = [HERE / name for name in (
    "test_simulator.py", "test_model.py", "test_causality.py",
    "test_fast.py", "test_events_engine.py")]
tests.append(ROOT.parent / "winter-paper/audit/test_revision_checks.py")
rows = []
for path in tests:
    start = time.monotonic()
    result = subprocess.run([sys.executable, str(path)], cwd=path.parent,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    (destination/(path.stem+".log")).write_text(result.stdout)
    match = re.search(r"Ran (\d+) tests?", result.stdout)
    row = dict(test=str(path),sha256=digest(path),exit_code=result.returncode,
               tests=int(match.group(1)) if match else None,
               seconds=time.monotonic()-start,log_sha256=digest(destination/(path.stem+".log")))
    rows.append(row)
    print(path.name, "passed" if result.returncode == 0 else "FAILED", flush=True)
    if result.returncode:
        print(result.stdout,flush=True)
report = dict(status="passed" if all(r["exit_code"] == 0 for r in rows) else "failed",
              checked_at_utc=datetime.now(timezone.utc).isoformat(),
              source_sha256={str(p.relative_to(HERE)):digest(p) for p in HERE.rglob("*.py")},
              rows=rows,total_tests=sum(r["tests"] or 0 for r in rows))
write_json(destination/"tests.json",report)
if report["status"] != "passed":
    raise SystemExit(1)
