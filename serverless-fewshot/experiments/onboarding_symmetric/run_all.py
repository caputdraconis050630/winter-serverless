"""Logged, resumable execution; no evaluations until every selection is frozen."""
import subprocess
import sys
import time

from experiment import HERE, OUT, RHOS, digest, frozen_json, write_json


def run(name, script, *args):
    directory = OUT / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with open(directory / (name + ".log"), "a") as log:
        p = subprocess.Popen([sys.executable, str(HERE / script), *args], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line)
            log.flush()
            print(name, line.rstrip(), flush=True)
        code = p.wait()
    write_json(directory / (name + ".status.json"), dict(returncode=code, elapsed_seconds=time.monotonic()-start))
    if code:
        raise RuntimeError(f"Stage failed: {name}")


if __name__ == "__main__":
    run("prepare", "experiment.py", "prepare")
    run("tests", "test_followup.py")
    run("reporting_tests", "test_reporting.py")
    run("pooled", "pooled.py")
    for rho in RHOS:
        run(f"screen{rho:g}", "experiment.py", "screen", "--rho", str(rho))
        run(f"select{rho:g}", "experiment.py", "select", "--rho", str(rho))
    frozen_json(OUT / "evaluation_lock.json", {f"selection_rho{r:g}.json": digest(OUT / f"selection_rho{r:g}.json") for r in RHOS})
    frozen_json(OUT / "evaluation_code_lock.json", {name: digest(HERE / name) for name in ("experiment.py", "pooled.py")})
    for rho in RHOS:
        run(f"evaluate{rho:g}", "experiment.py", "evaluate", "--rho", str(rho))
    run("report", "report.py")
    run("forecast_diagnostics", "forecast_diagnostics.py")
    run("verify", "verify.py")
