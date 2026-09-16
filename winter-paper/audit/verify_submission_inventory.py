"""Read-only comparison of available experiment files to the handoff manifest.

Writes only the dedicated data_verification_2026-09-15 audit directory.
File identity does not establish scientific validity or historical run identity.
"""
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'winter-paper/audit/data_verification_2026-09-15'


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'winter-paper-handoff-20260915/provenance/FILES.csv'
    roots = {'serverless-fewshot', 'results', 'anil-fresh-start',
             'winter-artifact', 'review-audit',
             'winter-artifact-v1.2.4-fgcs-submission.tar.gz',
             'winter-artifact-v1.2.4-fgcs-submission.tar.gz.sha256'}
    totals, folders, byte_count = Counter(), Counter(), 0
    begin = time.monotonic()
    with source.open(newline='') as stream, (OUT / 'file_integrity.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, ['path', 'bytes', 'expected_sha256', 'actual_sha256', 'status'])
        writer.writeheader()
        rows = list(csv.DictReader(stream))
        for row in rows:
            rel = Path(row['path']).relative_to('workspace')
            if rel.parts[0] not in roots:
                continue
            p = ROOT / rel
            actual = ''
            if not p.is_file():
                status, size = 'missing', None
            else:
                size = p.stat().st_size
                h = hashlib.sha256()
                with p.open('rb') as f:
                    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                        h.update(block)
                actual = h.hexdigest()
                status = 'match' if actual == row['sha256'] else 'changed'
                byte_count += size
            writer.writerow(dict(path=str(rel), bytes=size, expected_sha256=row['sha256'], actual_sha256=actual, status=status))
            totals[status] += 1
            folders[rel.parts[0]] += 1
            if sum(totals.values()) % 5000 == 0:
                target.flush()
                print(dict(checked=sum(totals.values()), bytes=byte_count, status=dict(totals)), flush=True)
    result = dict(scope='All selected files listed in the handoff manifest; no experiment reruns',
                  manifest=str(source.relative_to(ROOT)),
                  manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  counts=dict(totals), files_by_root=dict(folders), bytes_hashed=byte_count,
                  seconds=time.monotonic()-begin)
    (OUT / 'file_integrity_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
