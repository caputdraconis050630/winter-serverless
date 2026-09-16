# -*- coding: utf-8 -*-
"""Merge per-split revision DES part files into the two canonical JSONs.
Idempotent; exits nonzero if any part is missing/invalid."""
import json, sys
ok = True
merged = {"v1": ("revision_v1_hybridfull_2019.json", []),
          "e1": ("revision_e1_gates_2019.json", [])}
PARTS = [("v1", "v1", "S1"), ("v1", "v1", "S2"), ("v1", "v1", "S3"),
         ("e1", "e1", "S2"), ("e1", "e1a", "S1"), ("e1", "e1a", "S3"),
         ("e1", "e1b", "S1"), ("e1", "e1b", "S3")]
for s, tag, sp in PARTS:
    p = f"results/runs/revision_{tag}_2019_{sp}.json"
    try:
        blob = open(p, "rb").read()
        assert blob and blob.count(0) == 0, "NUL bytes"
        part = json.loads(blob)
        rows = part["results"] if isinstance(part, dict) and "results" in part else part
        assert rows, "empty"
        merged[s][1].extend(rows)
        print(f"[merge] {p}: {len(rows)} rows")
    except Exception as e:
        print(f"[merge] {p} MISSING/INVALID: {e}")
        ok = False
if not ok:
    sys.exit(1)
for s, (name, rows) in merged.items():
    out = f"results/runs/{name}"
    json.dump(rows, open(out, "w"), default=float)
    blob = open(out, "rb").read()
    assert blob and blob.count(0) == 0
    json.load(open(out))
    print(f"[merge] saved+verified {out} ({len(rows)} rows)")
