"""Resume the last pool with six workers after Azure S3 has completed."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
import time

from run_all import HERE, stage
from campaign import OUT
from common import digest, read_json, write_json


def command_matches(pid, fragment):
    return fragment.encode() in (Path('/proc') / str(pid) / 'cmdline').read_bytes()


def process_stopped(pid):
    return (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1][0] == 'T'


def main(parent, coordinator, runner):
    assert command_matches(parent, 'experiments/fixed_ewma/run_all.py')
    assert command_matches(coordinator, 'experiments/fixed_ewma/finish_parallel.py')
    assert command_matches(runner, 'fixed_ewma/runner.py')
    assert command_matches(runner, 'steady_huawei_h_saturated')
    assert process_stopped(parent)
    plan = read_json(OUT / 'execution_plan.json')
    for name, sha in plan['code_sha256'].items():
        assert digest(HERE / name) == sha, name
    assert (OUT / 'replay/cases/steady_azure2019_S3/summary.json').exists()
    name = 'steady_huawei_h_saturated'
    case = OUT / 'replay/cases' / name
    while True:
        checkpoints = [p for p in (case / 'partial_states').glob('*.npz') if '.tmp.' not in p.name]
        completed = list((case / 'seed_functions').glob('*.npz'))
        if len(checkpoints) >= 2 or len(completed) >= 33:
            break
        assert command_matches(runner, 'fixed_ewma/runner.py')
        time.sleep(10)
    previous = read_json(OUT / 'parallel_completion.json')
    record = dict(state='running', reason='Azure S3 complete; resume last pool within original six-worker limit',
                  source=previous, old_workers=2, new_workers=6,
                  measurement_code_unchanged=True, script_sha256=digest(__file__),
                  started_utc=datetime.now(timezone.utc).isoformat())
    stopped = False
    try:
        # SIGTERM has no handler in this coordinator; it leaves the original
        # orchestrator paused. This process assumes responsibility for resuming it.
        os.kill(coordinator, signal.SIGTERM)
        stopped = True
        os.kill(runner, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while (Path('/proc') / str(runner)).exists():
            state = (Path('/proc') / str(runner) / 'stat').read_text().split(') ', 1)[1][0]
            if state == 'Z':
                break
            assert time.monotonic() < deadline, 'Old runner did not terminate'
            time.sleep(.1)
        assert process_stopped(parent)
        archive = OUT / 'worker_reallocation'
        archive.mkdir(exist_ok=True)
        shutil.copy2(OUT / 'logs' / ('replay_' + name + '.status.json'), archive / 'previous_stage_status.json')
        # Retain any interrupted temporary write outside the runner's input glob.
        for folder in ('partial_states', 'seed_functions', 'functions'):
            for p in (case / folder).glob('*.tmp.npz'):
                shutil.move(str(p), archive / (folder + '_' + p.name))
        record['checkpoint_sha256'] = {p.name: digest(p) for p in (case / 'partial_states').glob('*.npz')}
        record['completed_function_seeds_preserved'] = len(list((case / 'seed_functions').glob('*.npz')))
        write_json(OUT / 'worker_reallocation.json', record)
        print('Resuming with six workers', record['completed_function_seeds_preserved'], 'completed jobs;',
              len(record['checkpoint_sha256']), 'checkpoints', flush=True)
        stage('replay_' + name, [str(HERE / 'runner.py'), name, '--workers', '6'])
        stage('merge_' + name, [str(HERE / 'merge.py'), '--case', name])
        record['state'] = 'complete'
    except BaseException:
        record['state'] = 'failed'
        raise
    finally:
        if stopped:
            os.kill(parent, signal.SIGCONT)
        record['orchestrator_resumed'] = stopped
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(OUT / 'worker_reallocation.json', record)
        previous.update(state=record['state'], orchestrator_resumed=stopped,
                        continuation='worker_reallocation.json', finished_utc=record['finished_utc'])
        write_json(OUT / 'parallel_completion.json', previous)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parent', type=int)
    p.add_argument('coordinator', type=int)
    p.add_argument('runner', type=int)
    a = p.parse_args()
    main(a.parent, a.coordinator, a.runner)
