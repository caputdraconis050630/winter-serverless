"""Checked, logged, resumable execution of every attribution-study stage."""
import argparse
import os
import subprocess
import sys
import time

from common import HERE,OUT,ROOT,write_json


STAGES=[
    ("prepare",["prepare.py"]),
    ("simulator_tests",["test_study.py"]),
    ("models",["models.py"]),
    ("policy_tests",["test_policies.py"]),
    ("analysis_tests",["test_analysis.py"]),
    ("bridge",["bridge.py"]),
    ("cheap_screen",["study.py","screen","--cheap-only","--workers","3"]),
    ("model_calibration",["study.py","calibrate_models","--workers","2"]),
    ("bounded_screen",["prune_screen.py"]),
    ("selection",["study.py","select","--workers","2"]),
    ("evaluation",["study.py","evaluate","--workers","3"]),
    ("representations",["study.py","representations","--workers","3"]),
    ("analysis",["analyze.py"]),
    ("verification",["verify.py"]),
]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--from-stage",choices=[s[0] for s in STAGES],default="prepare")
    args=ap.parse_args()
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    logs=OUT/"logs"; logs.mkdir(parents=True,exist_ok=True)
    active=False
    for name,command in STAGES:
        active=active or name==args.from_stage
        if not active:
            continue
        print("START",name,flush=True)
        start=time.time()
        with open(logs/(name+".log"),"a") as stream:
            result=subprocess.run([sys.executable,str(HERE/command[0]),*command[1:]],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        write_json(logs/(name+".status.json"),{"returncode":result.returncode,"elapsed_seconds":time.time()-start})
        if result.returncode:
            raise SystemExit(f"Failed {name}; see {logs/(name+'.log')}")
        print("DONE",name,flush=True)


if __name__=="__main__":
    main()
