"""Use the completed control branch's two workers for remaining Huawei cases.

Pause only the orchestrator while its existing four-worker child continues.
The finally block resumes it; completed stages are then skipped normally.
Measurement code and statistical settings remain frozen.
"""
import argparse
import os
from pathlib import Path
import signal
from datetime import datetime, timezone

from run_all import HERE, stage
from campaign import OUT
from common import read_json, write_json, digest


def main(pid):
    command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
    assert any(x.endswith(b'fixed_ewma/run_all.py') for x in command), command
    plan = read_json(OUT / 'execution_plan.json')
    assert plan['total_bulk_workers'] == 6
    for name, sha in plan['code_sha256'].items():
        assert digest(HERE / name) == sha, name
    assert read_json(OUT / 'logs/controls_analyze.status.json')['state'] == 'complete'
    assert read_json(OUT / 'logs/replay_steady_azure2019_S3.status.json')['state'] == 'running'
    names = ['steady_huawei_h_mixed', 'steady_huawei_h_sparse', 'steady_huawei_h_saturated']
    assert all(not (OUT / 'logs' / ('replay_' + name + '.status.json')).exists() for name in names)
    record = dict(orchestrator_pid=pid, worker_allocation={'azure2019_S3': 4, 'huawei': 2},
                  cases=names, measurement_code_unchanged=True, script_sha256=digest(__file__),
                  started_utc=datetime.now(timezone.utc).isoformat(), state='running')
    paused = False
    try:
        os.kill(pid, signal.SIGSTOP)
        paused = True
        write_json(OUT / 'parallel_completion.json', record)
        for name in names:
            stage('replay_' + name, [str(HERE / 'runner.py'), name, '--workers', '2'])
            stage('merge_' + name, [str(HERE / 'merge.py'), '--case', name])
        record['state'] = 'complete'
    except BaseException:
        record['state'] = 'failed'
        raise
    finally:
        if paused:
            os.kill(pid, signal.SIGCONT)
        record['orchestrator_resumed'] = paused
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(OUT / 'parallel_completion.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('orchestrator_pid', type=int)
    main(parser.parse_args().orchestrator_pid)
