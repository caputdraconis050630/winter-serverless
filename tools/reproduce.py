#!/usr/bin/env python3
"""Portable entry points for the fixed-alpha WINTER submission artifact."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import release_metadata

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'winter-paper'
EXP = ROOT / 'serverless-fewshot'
RESULTS = EXP / 'results/fixed_ewma_v1'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def output_directory(prefix):
    base = ROOT / '.repro-output'; base.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix + '-', dir=base))


def require_data():
    if not (RESULTS / 'cases').is_dir():
        raise SystemExit('Restore the matching DOI data archive beside this repository first. See docs/REPRODUCTION.md.')


def checksums(data=False):
    manifests = [ROOT / 'MANIFEST-code.sha256']
    if data:
        manifests.append(ROOT / 'MANIFEST-data.sha256')
    count = 0
    for manifest in manifests:
        if not manifest.is_file():
            raise SystemExit('Missing manifest: ' + str(manifest))
        for line in manifest.read_text().splitlines():
            digest, name = line.split('  ', 1)
            path = ROOT / name
            assert path.is_file(), name
            assert sha(path) == digest, name
            count += 1
            if count % 10000 == 0:
                print('Verified checksum entries:', count, flush=True)
    print('PASS: checksum entries:', count, flush=True)


def tests():
    destination = output_directory('tests')
    code = EXP / 'experiments/lstm_comparison'
    rows = []
    for name in ['test_simulator.py', 'test_model.py', 'test_causality.py',
                 'test_fast.py', 'test_events_engine.py']:
        run = subprocess.run([sys.executable, str(code / name)], cwd=code,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors='replace', env=os.environ)
        (destination / (name + '.log')).write_text(run.stdout)
        match = re.search(r'Ran (\d+) tests?', run.stdout)
        rows.append(dict(file=name, exit_code=run.returncode,
                         tests=int(match.group(1)) if match else None))
        print(name, 'passed' if run.returncode == 0 else 'FAILED', flush=True)
        if run.returncode:
            raise SystemExit(run.stdout)
    (destination / 'report.json').write_text(json.dumps(dict(status='passed', tests=rows), indent=2) + '\n')
    print('Report:', destination / 'report.json', flush=True)


def analysis_script(name, destination):
    """Only redirect the report directory; keep the numerical calculations."""
    source = PAPER / 'audit/submission_sentence_review_2026-09-16' / name
    text = source.read_text()
    marker = 'OUT = Path(__file__).resolve().parent' if name == 'numerical_review.py' else 'OUT=Path(__file__).resolve().parent'
    assert text.count(marker) == 1
    text = text.replace(marker, 'OUT = Path(' + repr(str(destination)) + ')')
    exec(compile(text, str(source), 'exec'), {'__file__': str(source), '__name__': '__main__'})


def verify(statistics=False):
    require_data()
    destination = output_directory('verify')
    expected = json.loads((RESULTS / 'verification/report.json').read_text())
    assert expected['status'] == 'passed' and expected['ewma_alpha'] == .3
    corrected_hashes = release_metadata.verify(ROOT, expected)
    cases = expected['expected_cases']; assert len(cases) == 19
    for name in cases:
        directory = RESULTS / 'cases' / name
        meta = json.loads((directory / 'case.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        assert sha(directory / 'data.npz') == meta['data_sha256'], name
        assert sha(directory / 'aggregate.npz') == summary['aggregate_sha256'], name
        for action, record in meta['actions'].items():
            assert sha(directory / record['file']) == record['sha256'], (name, action)
            assert not action.startswith(('EWMA_0.1', 'EWMA_0.5', 'WINTER_frozen'))
        for filename, digest in expected['cases'][name]['source_sha256'].items():
            digest = corrected_hashes.get((name, filename), digest)
            assert sha(directory / filename) == digest, (name, filename)
        contract = json.loads((directory / 'execution.json').read_text())
        assert sha(directory / 'case.json') == contract['case_sha256'], name
        print('Frozen inputs and policy arrays verified:', name, flush=True)
    analysis_script('table_review.py', destination)
    tables = json.loads((destination / 'table_checks.json').read_text())
    if tables['status'] != 'passed':
        raise SystemExit('Table verification failed; see ' + str(destination / 'table_checks.json'))
    if statistics:
        analysis_script('numerical_review.py', destination)
        numerical = json.loads((destination / 'numerical_checks.json').read_text())
        if numerical['status'] != 'passed':
            raise SystemExit('Numerical verification failed; see ' + str(destination / 'numerical_checks.json'))
    report = dict(status='passed', cases=len(cases), ewma_alpha=.3,
                  drift_metadata_corrections_verified=6,
                  input_and_action_hashes='passed', tables='passed',
                  independent_statistics='passed' if statistics else 'not requested',
                  simulation_rerun=False)
    (destination / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Report:', destination / 'verification.json', flush=True)


def build_paper(source=None):
    source = (source if source is not None else ROOT / 'submission').resolve()
    if not (source / 'main.tex').is_file():
        raise SystemExit('Provide the complete IEEE Access package with paper --source /path/to/ieee-access-20260922. See docs/REPRODUCTION.md.')
    main_text = (source / 'main.tex').read_text()
    document_class = re.search(r'^\s*\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}', main_text, re.MULTILINE)
    if document_class is None:
        raise SystemExit('No document class found in ' + str(source / 'main.tex'))
    access = document_class.group(1) == 'ieeeaccess'
    if not access and not document_class.group(1).startswith('cas-'):
        raise SystemExit('Unsupported manuscript class: ' + document_class.group(1))
    required = ['supplementary.tex'] + (['build.py', 'ieeeaccess.cls', 'IEEEtran.cls', 'IEEEtran.bst'] if access else [])
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise SystemExit('Incomplete manuscript package; missing: ' + ', '.join(missing))

    # Preserve nested author images and local font assets in the isolated build.
    suffixes = {'.tex', '.bib', '.bst', '.bbl', '.cls', '.sty', '.pdf',
                '.png', '.jpg', '.jpeg', '.pfb', '.tfm', '.map', '.fd'}
    excluded_dirs = {'.git', '.build', '.repro-output', '__pycache__'}
    outputs = {'main.pdf', 'supplementary.pdf', 'highlights.pdf'}
    sources = {}
    for path in source.rglob('*'):
        relative = path.relative_to(source)
        if any(part in excluded_dirs for part in relative.parts) or not path.is_file():
            continue
        if path.name not in outputs and (path.suffix.lower() in suffixes or path.name == 'build.py'):
            sources[relative] = sha(path)
    destination = output_directory('paper')
    for relative, digest in sources.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
        if sha(target) != digest:
            raise SystemExit('Source changed while staging: ' + str(relative))

    if access:
        commands = [[sys.executable, 'build.py']]
        documents = ['main', 'supplementary']
        log_directory = destination / '.build'
    else:
        commands = [['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], ['bibtex', 'main']]
        documents = ['main', 'supplementary']
        if (destination / 'highlights.tex').is_file():
            documents.append('highlights')
        for name in documents:
            count = {'main': 2, 'supplementary': 3, 'highlights': 1}[name]
            commands += [['pdflatex', '-interaction=nonstopmode', '-halt-on-error', name + '.tex']] * count
        log_directory = destination
    env = dict(os.environ, TEXINPUTS=str(destination) + os.pathsep,
               BIBINPUTS=str(destination) + os.pathsep, BSTINPUTS=str(destination) + os.pathsep)
    # pdfTeX also resolves local metrics, Type 1 fonts and font maps separately.
    for key in ('TFMFONTS', 'T1FONTS', 'TEXFONTMAPS'):
        env[key] = str(destination) + os.pathsep + env.get(key, '')
    for index, command in enumerate(commands):
        print('Building:', ' '.join(command), flush=True)
        run = subprocess.run(command, cwd=destination, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors='replace')
        (destination / f'command-{index:02d}.log').write_text(run.stdout)
        if run.returncode:
            raise SystemExit(run.stdout[-8000:])
    pages = {}
    for name in documents:
        log = (log_directory / (name + '.log')).read_text(errors='replace')
        markers = ['There were undefined references', 'multiply defined', 'Citation ',
                   'LaTeX Error:', 'Missing character:', 'Rerun to get cross-references right']
        if any(marker in log for marker in markers):
            raise SystemExit('Unresolved document issue; inspect ' + str(log_directory / (name + '.log')))
        match = re.search(r'Output written on .*?\((\d+) pages?', log, re.DOTALL)
        if match is None or not (destination / (name + '.pdf')).is_file():
            raise SystemExit('Missing completed PDF: ' + name)
        pages[name] = int(match.group(1))
    for relative, digest in sources.items():
        if sha(source / relative) != digest:
            raise SystemExit('Input package changed during build: ' + str(relative))
    report = dict(status='passed', format='ieee-access' if access else 'archived-cas',
                  source_package=source.name, pages=pages,
                  source_sha256={str(path): digest for path, digest in sorted(sources.items())},
                  pdf_sha256={name + '.pdf': sha(destination / (name + '.pdf')) for name in documents})
    (destination / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Built PDFs:', destination, 'pages:', pages, flush=True)


def replay(case, workers, output):
    require_data()
    source = RESULTS / 'cases' / case
    assert source.is_dir(), case
    destination = Path(output).resolve()
    if destination.exists():
        raise SystemExit('Choose a new output directory; the recorded results are not overwritten.')
    target = destination / 'cases' / case; target.mkdir(parents=True)
    metadata = json.loads((source / 'case.json').read_text())
    names = ['case.json', 'data.npz'] + [row['file'] for row in metadata['actions'].values()]
    if (source / 'lstm_forecasts.npz').is_file():
        names.append('lstm_forecasts.npz')
    for name in names:
        shutil.copy2(source / name, target / name)
    code = EXP / 'experiments/lstm_comparison'
    sys.path.insert(0, str(code)); sys.path.insert(1, str(EXP))
    common = importlib.import_module('common'); common.OUT = destination
    runner = importlib.import_module('run_seeds')
    runner.run(case, workers=workers)
    print('Fresh replay:', target, flush=True)


def figures():
    require_data()
    destination = output_directory('figures')
    shutil.copytree(PAPER, destination / 'winter-paper', ignore=shutil.ignore_patterns('__pycache__', 'build-*'))
    (destination / 'serverless-fewshot').symlink_to(EXP, target_is_directory=True)
    (destination / 'winter-paper/audit/final_revision_2026-09-16').mkdir(parents=True, exist_ok=True)
    retained_tables = {p.name for p in (PAPER / 'generated').glob('*.tex')}
    subprocess.run([sys.executable, 'audit/build_lstm_revision.py'], cwd=destination / 'winter-paper', check=True)
    for path in (destination / 'winter-paper/generated').glob('*.tex'):
        if path.name not in retained_tables:
            path.unlink()
    print('Regenerated figures and tables:', destination / 'winter-paper', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    checks = commands.add_parser('checksums'); checks.add_argument('--data', action='store_true')
    commands.add_parser('test')
    verify_parser = commands.add_parser('verify'); verify_parser.add_argument('--statistics', action='store_true')
    paper = commands.add_parser('paper')
    paper.add_argument('--source', type=Path, help='Complete manuscript source package; defaults to restored submission/.')
    commands.add_parser('figures')
    run = commands.add_parser('replay'); run.add_argument('--case', required=True)
    run.add_argument('--workers', type=int, default=4); run.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'checksums': checksums(args.data)
    elif args.command == 'test': tests()
    elif args.command == 'verify': verify(args.statistics)
    elif args.command == 'paper': build_paper(args.source)
    elif args.command == 'figures': figures()
    elif args.command == 'replay': replay(args.case, args.workers, args.output)


if __name__ == '__main__':
    main()
