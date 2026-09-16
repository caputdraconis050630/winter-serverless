"""Run all affected experiments, keeping bulk DES out of the live measurement."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from campaign import ROOT,OUT,ALL_CASES,INITIAL,DRIFT,STEADY
from common import digest,read_json,write_json,freeze

HERE=Path(__file__).resolve().parent
LOGS=OUT/'logs'


def stage(name,args):
    status=LOGS/(name+'.status.json')
    if status.exists() and read_json(status).get('returncode')==0:
        print('already complete',name,flush=True);return
    begin=time.time()
    write_json(status,dict(stage=name,state='running',command=args,started_utc=datetime.now(timezone.utc).isoformat()))
    print('START',name,flush=True)
    with (LOGS/(name+'.log')).open('a') as log:
        process=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    write_json(status,dict(stage=name,state='complete' if process.returncode==0 else 'failed',
                          returncode=process.returncode,elapsed_seconds=time.time()-begin,command=args))
    if process.returncode:raise RuntimeError(f'{name} failed; inspect {LOGS/(name+".log")}')
    print('DONE',name,round(time.time()-begin),flush=True)


def replays():
    order=INITIAL+['continuous_48h']+DRIFT+STEADY
    for name in order:
        stage('replay_'+name,[str(HERE/'runner.py'),name,'--workers','4'])
        stage('merge_'+name,[str(HERE/'merge.py'),'--case',name])


def controls():
    for name in ('calibrate','screen','select','evaluate','analyze'):
        stage('controls_'+name,[str(HERE/'controls.py'),name,'--workers','2'])


def main():
    LOGS.mkdir(parents=True,exist_ok=True)
    freeze(OUT/'execution_plan.json',dict(ewma_alpha=.3,cases=ALL_CASES,
        live_before_bulk_des=True,total_bulk_workers=6,
        code_sha256={p.name:digest(p) for p in HERE.glob('*.py')},
        source_protocol_sha256=digest(OUT/'protocol.json'),
        required=['19 affected DES cases','fixed-alpha controller selection and evaluation',
                  'five-arm strict HTTP live replay','paired analysis','verification','paper update']))
    start=time.monotonic()
    live=OUT/'strict_round/live/results.json'
    while not live.exists():
        if time.monotonic()-start>7200:raise RuntimeError('Live replay not complete after two hours; inspect live.log')
        print('Waiting for live measurement before bulk DES',round(time.monotonic()-start),'s',flush=True)
        time.sleep(30)
    measured=read_json(live)
    assert len(measured['arms'])==5
    assert all(a['invocations']==686 for a in measured['arms'].values())
    print('Live measurement complete; bulk replay and controller calibration start',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(replays),pool.submit(controls)]
        for future in futures:future.result()
    write_json(OUT/'measurement_status.json',dict(status='complete',ewma_alpha=.3,
        completed_cases=ALL_CASES,live_sha256=digest(live),
        note='Measurements complete; composite verification and paper integration run separately'))
    print('ALL FIXED-ALPHA MEASUREMENTS COMPLETE',flush=True)


if __name__=='__main__':main()
