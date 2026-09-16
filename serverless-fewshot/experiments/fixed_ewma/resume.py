"""Resume the frozen measurements after later analysis tools were added."""
from concurrent.futures import ThreadPoolExecutor
from run_all import HERE,replays,controls
from campaign import OUT,ALL_CASES
from common import read_json,digest,write_json


def main():
    plan=read_json(OUT/'execution_plan.json')
    assert plan['ewma_alpha']==.3
    assert digest(OUT/'protocol.json')==plan['source_protocol_sha256']
    for name,sha in plan['code_sha256'].items():
        assert digest(HERE/name)==sha,name
    live=OUT/'strict_round/live/results.json'
    measured=read_json(live)
    assert len(measured['arms'])==5
    assert all(a['invocations']==686 for a in measured['arms'].values())
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(replays),pool.submit(controls)]
        for job in jobs:job.result()
    write_json(OUT/'measurement_status.json',dict(status='complete',ewma_alpha=.3,
        completed_cases=ALL_CASES,live_sha256=digest(live),
        note='Measurements complete; composite verification and paper integration run separately'))


if __name__=='__main__':main()
