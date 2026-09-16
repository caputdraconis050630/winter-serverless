"""Complete statistical verification and export only validated fixed-alpha evidence."""
from pathlib import Path
import shutil
import time
from datetime import datetime,timezone
from campaign import ROOT,OUT
from common import digest,read_json,write_json
from run_all import stage

HERE=Path(__file__).resolve().parent
PAPER=ROOT.parent/'winter-paper'
DEST=PAPER/'audit/fixed_ewma_2026-09-16'


def export():
    assert read_json(OUT/'measurement_status.json')['status']=='complete'
    assert read_json(OUT/'verification/report.json')['status']=='passed'
    paths={'protocol.json':OUT/'protocol.json',
        'verification.json':OUT/'verification/report.json',
        'nonshift.json':OUT/'analysis/nonshift.json',
        'nonshift_comparisons.csv':OUT/'analysis/nonshift_comparisons.csv',
        'drift/analysis.json':OUT/'analysis/drift.json',
        'drift/comparisons.csv':OUT/'analysis/comparisons.csv',
        'drift/operating_points.csv':OUT/'analysis/operating_points.csv',
        'drift/figure_curves.csv':OUT/'analysis/figure_curves.csv'}
    for name,source in paths.items():
        dest=DEST/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        assert digest(dest)==digest(source)
    write_json(DEST/'completion.json',dict(status='complete',ewma_alpha=.3,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        paper_sources={name:digest(source) for name,source in paths.items()},
        export_sha256=digest(__file__),
        note='Measurement and statistical evidence complete; manuscript integration checked separately'))


def main():
    start=time.monotonic()
    while not (OUT/'measurement_status.json').exists():
        failed=[p.name for p in (OUT/'logs').glob('*.status.json')
                if read_json(p).get('state')=='failed']
        if failed:raise RuntimeError('Measurement stage failed: '+str(failed))
        print('Awaiting all measurements',round(time.monotonic()-start),'seconds',flush=True)
        time.sleep(30)
    stage('paired_analysis',[str(HERE/'analyze.py')])
    stage('controls_verification',[str(HERE/'verify_controls.py')])
    stage('replay_verification',[str(HERE/'verify.py')])
    export()
    stage('reviewer_evidence',[str(PAPER/'audit/reviewer_evidence_2026_09_16.py')])
    stage('paper_figures_tables',[str(PAPER/'audit/build_lstm_revision.py')])
    write_json(OUT/'postprocess_status.json',dict(status='complete',ewma_alpha=.3,
        note='Measurements, paired statistics, independent verification and figure/table generation complete; prose and PDF review pending'))
    print('Verified evidence exported; prose and PDF review pending',flush=True)


if __name__=='__main__':main()
