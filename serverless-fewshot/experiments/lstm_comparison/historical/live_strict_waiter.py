"""Primary live replay: record every response and require HTTP 2xx success.

The first development replay used the archived request helper, which treats
HTTP 4xx responses as successful. This round is specified before reviewing
its final outcomes and retains individual responses for independent checks.
"""
from collections import Counter, defaultdict
import http.client
import json
import subprocess
import threading
import time

import live as experiment
from common import OUT, HERE, ROOT, digest, freeze, write_json

DIRECTORY = OUT / "strict_round"
DIRECTORY.mkdir(exist_ok=True)
REQUESTS = defaultdict(list)
LOCK = threading.Lock()
original_run_arm = experiment.live.run_arm
original_freeze = experiment.freeze


def invoke(service):
    started = time.time()
    status = None
    error = None
    connection = None
    try:
        connection = http.client.HTTPConnection(
            experiment.live.NODE_IP, experiment.live.PORT, timeout=60)
        connection.request("GET", "/", headers={
            "Host": f"{service}.{experiment.live.NS}.127.0.0.1.sslip.io"})
        response = connection.getresponse()
        response.read()
        status = response.status
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        if connection is not None:
            connection.close()
    record = dict(svc=service, t_start=started, latency=time.time()-started,
                  ok=status is not None and 200 <= status < 300,
                  http_status=status, error=error)
    with LOCK:
        REQUESTS[service].append(record)
    return record


def run_arm(arm, names, *args):
    result = original_run_arm(arm, names, *args)
    records = sorted([r for name in names for r in REQUESTS[name]],
                     key=lambda r: r["t_start"])
    assert len(records) == result["invocations"]
    assert sum(not r["ok"] for r in records) == result["failed_requests"]
    result["requests"] = records
    result["http_status_counts"] = dict(Counter(
        str(r["http_status"]) if r["http_status"] is not None else "transport_error"
        for r in records))
    return result


def freeze_protocol(path, contract):
    if path.name == "protocol.json":
        contract.update(success_definition="HTTP 200--299; all attempts retained",
                        individual_requests="timestamp, latency, status, error, function and trace minute",
                        primary_round="strict-status replay specified before development outcomes",
                        experiment_sha256=digest(HERE / "live.py"),
                        strict_wrapper_sha256=digest(__file__),
                        archived_helper_sha256=digest(ROOT / "scripts/testbed_cohort.py"),
                        lstm_checkpoint_sha256=digest(OUT / "models/s1_seed0/model.pt"))
    return original_freeze(path, contract)


def main():
    freeze(DIRECTORY / "plan.json", dict(
        role="primary live actuation comparison", minutes=60, arms=5,
        success="HTTP 2xx", raw_requests=True,
        reason="Archived helper accepts HTTP 4xx and does not retain response status.",
        selection="Protocol correction specified before the development replay finishes.",
        wrapper_sha256=digest(__file__)))
    # The development services must finish and disappear before this replay.
    while not (OUT / "live/results.json").exists():
        time.sleep(10)
    deadline = time.monotonic() + 900
    while True:
        raw = subprocess.check_output([
            "kubectl", "get", "ksvc,pods", "-n", "winter-lstm-eval-v2", "-o", "json"])
        if not json.loads(raw)["items"]:
            break
        if time.monotonic() > deadline:
            raise RuntimeError("Development services have not drained")
        time.sleep(5)
    experiment.OUT = DIRECTORY
    experiment.live.invoke = invoke
    experiment.live.run_arm = run_arm
    experiment.freeze = freeze_protocol
    print("Primary strict-status live replay begins", flush=True)
    experiment.main()


if __name__ == "__main__":
    main()
