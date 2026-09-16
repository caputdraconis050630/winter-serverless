"""Correct release descriptions while retaining measurement-time hash records."""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import shutil

RESULTS = Path('serverless-fewshot/results/fixed_ewma_v1')
LEDGER = Path('provenance/metadata-corrections.json')
OLD_PRIMARY = 'scheduled every 10 min, buffer 120, scalar ridge lambda .01; no onset input'
OLD_DIAGNOSTIC = 'same adapter frozen at known onset, diagnostic only'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def corrected_case(original, learned_source):
    assert original['surface'] == 'drift' and original['ewma_alpha'] == .3
    assert original['primary_adapter'] == OLD_PRIMARY
    assert original['diagnostic_adapter'] == OLD_DIAGNOSTIC
    assert learned_source['drift_reset'] is False
    assert learned_source['drift_onset_available_to_policy'] is False
    assert 'each tick' in learned_source['learned_loop']
    result = deepcopy(original)
    result['primary_adapter'] = learned_source['learned_loop']
    del result['diagnostic_adapter']
    for key in ['support_start', 'drift_reset', 'drift_onset_available_to_policy', 'output_selection']:
        result[key] = learned_source[key]
    result['gate'] = 'N=0 prototype; 0<N<100 EWMA0.3; N>=100 age<720 learned; otherwise EWMA0.3'
    result['no_handoff'] = 'same policy with no age boundary; zero-count and below-count routing unchanged'
    return result


def verify(root, expected):
    """Validate the exact descriptive change and return replacement file hashes."""
    ledger = read(root / LEDGER)
    assert ledger['version'] == '2026.09.16.1'
    names = {n for n in expected['expected_cases'] if n.startswith('drift_')}
    assert set(ledger['cases']) == names and len(names) == 6
    replacements = {}
    for name, entry in ledger['cases'].items():
        directory = root / RESULTS / 'cases' / name
        original_dir = root / 'provenance/metadata-before-correction' / name
        for filename in ['case.json', 'execution.json']:
            row = entry[filename]
            assert sha(original_dir / filename) == row['original_sha256']
            assert row['original_sha256'] == expected['cases'][name]['source_sha256'][filename]
            assert sha(directory / filename) == row['release_sha256']
            replacements[name, filename] = row['release_sha256']
        source = root / 'serverless-fewshot/results/winter_drift_v1/cases' / name / 'case.json'
        assert sha(source) == entry['learned_source_sha256']
        original = read(original_dir / 'case.json')
        assert read(directory / 'case.json') == corrected_case(original, read(source))
        execution = read(original_dir / 'execution.json')
        assert execution['case_sha256'] == sha(original_dir / 'case.json')
        execution['case_sha256'] = sha(directory / 'case.json')
        assert read(directory / 'execution.json') == execution
        for action, row in original['actions'].items():
            method = action.rsplit('__', 1)[0]
            if method == 'WINTER':
                assert row['measurement_source'].endswith('/winter_drift_v1/cases/' + name)
            elif method == 'WINTER_G':
                assert row['measurement_source'].endswith('/fixed_ewma_v1/replay/cases/' + name)
            assert method not in {'WINTER_frozen', 'WINTER_scheduled', 'EWMA_0.1', 'EWMA_0.5'}
    return replacements


def correct(root):
    expected = read(root / RESULTS / 'verification/report.json')
    if (root / LEDGER).exists():
        verify(root, expected)
        return
    ledger = dict(version='2026.09.16.1',
        reason='Final composed drift cases inherited obsolete scheduled/frozen adapter descriptions.',
        changes='Descriptive case fields and the enclosing case hash only; actions, outcomes and historical verification records are unchanged.',
        cases={})
    for name in expected['expected_cases']:
        if not name.startswith('drift_'):
            continue
        directory = root / RESULTS / 'cases' / name
        original_dir = root / 'provenance/metadata-before-correction' / name
        original_dir.mkdir(parents=True, exist_ok=False)
        entry = {}
        for filename in ['case.json', 'execution.json']:
            digest = sha(directory / filename)
            assert digest == expected['cases'][name]['source_sha256'][filename]
            shutil.copy2(directory / filename, original_dir / filename)
            entry[filename] = dict(original_sha256=digest)
        source = root / 'serverless-fewshot/results/winter_drift_v1/cases' / name / 'case.json'
        write(directory / 'case.json', corrected_case(read(original_dir / 'case.json'), read(source)))
        execution = read(original_dir / 'execution.json')
        execution['case_sha256'] = sha(directory / 'case.json')
        write(directory / 'execution.json', execution)
        for filename in ['case.json', 'execution.json']:
            entry[filename]['release_sha256'] = sha(directory / filename)
        entry['learned_source_sha256'] = sha(source)
        ledger['cases'][name] = entry
    write(root / LEDGER, ledger)
    verify(root, expected)
    print('Corrected and verified six final drift descriptions; retained original metadata and hashes.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    correct(parser.parse_args().root.resolve())
