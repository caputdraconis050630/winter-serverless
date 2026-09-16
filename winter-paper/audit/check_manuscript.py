"""Check source links, derived accounting, and deferred-asset preservation."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from build_nondrift_evidence import strict_al
import build_winter_g_evidence

ROOT = Path(__file__).resolve().parent.parent


def expand(path):
    text = path.read_text()
    for name in re.findall(r"\\input\{([^}]+)\}", text):
        source = ROOT / (name if name.endswith(".tex") else name + ".tex")
        assert source.is_file(), source
        text += "\n" + expand(source)
    for name in re.findall(r"\\tableinput\s+([^\s]+)", text):
        if not name.startswith("generated/"):
            continue
        assert (ROOT / name).is_file(), name
        text += "\n" + (ROOT / name).read_text()
    return text


docs = {name: expand(ROOT / (name + ".tex")) for name in ("main", "supplementary")}
labels = {name: re.findall(r"\\label\{([^}]+)\}", text) for name, text in docs.items()}
bib = set(re.findall(r"@\w+\{([^,]+)", (ROOT / "refs.bib").read_text()))
figures = {}
for name, text in docs.items():
    assert len(labels[name]) == len(set(labels[name])), (name, "duplicate labels")
    other, prefix = ("supplementary", "sup-") if name == "main" else ("main", "main-")
    known = set(labels[name]) | {prefix + x for x in labels[other]}
    references = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
    assert not references - known, (name, references - known)
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", text):
        assert set(group.split(",")) <= bib, group
    figs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)
    assert all((ROOT / f).is_file() for f in figs)
    assert "fig10_drift_response.pdf" not in figs
    figures[name] = figs
inventory = json.loads((ROOT / "audit/figure_manifest.json").read_text())
for name in docs:
    assert figures[name] == inventory[name], (name, "figure manifest mismatch", figures[name])
    assert not re.search(r"\[(?:insert|Insert|verified|TODO)\b", docs[name]), (name, "unresolved draft placeholder")
preserved = json.loads((ROOT / "audit/preservation_manifest.json").read_text())
for filename, digest in preserved.items():
    assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest, filename
drift = "fig10_drift_response.pdf"
data = json.loads((ROOT / "audit/onboarding_metrics.json").read_text())
for cohort in data.values():
    assert cohort["prototype_count"] == 16
    for rho, rows in cohort["by_rho"].items():
        base = rows["B4a_ewma"]
        for row in rows.values():
            np.testing.assert_allclose(row["cost_per_1k"], row["wm_per_1k"] + 150 * float(rho) * row["csr_pct"], rtol=1e-12)
            np.testing.assert_allclose(row["delta_csr_pp"], row["csr_pct"] - base["csr_pct"], atol=1e-12)
            assert 0 <= row["wilcoxon_holm_three_rhos"] <= 1
assert strict_al([1.] * 30)[0] is None
assert strict_al([1.] * 31)[0] == 0
assert strict_al([2.] * 35 + [1.] * 20)[0] is None
matched = json.loads((ROOT / "audit/matched_memory.json").read_text())
assert len(matched) == 12
assert all(row["interpolated_delta_csr_pp"] is None for row in matched)
figure_sources = json.loads((ROOT / "audit/revision_figure_sources.json").read_text())
assert figure_sources["measurement_status"] == "passed"
assert hashlib.sha256((ROOT / figure_sources["source"]).read_bytes()).hexdigest() == figure_sources["source_sha256"]
for filename, digest in figure_sources["outputs"].items():
    assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest, filename
verification_path = ROOT / figure_sources["verification"]
assert hashlib.sha256(verification_path.read_bytes()).hexdigest() == figure_sources["verification_sha256"]
verification = json.loads(verification_path.read_text())
assert verification["status"] == "passed" and not verification["missing_cases"]
assert verification["ewma_alpha"] == .3
for name, text in docs.items():
    assert set(re.findall(r"\\alpha\s*=\s*([0-9.]+)", text)) == {"0.3"}, (name, "inconsistent EWMA alpha")
assert len(verification["cases"]) == len(verification["expected_cases"]) == 19
assert verification["tests"]["status"] == verification["live"]["status"] == "passed"
new_data = json.loads((ROOT / figure_sources["source"]).read_text())
assert new_data["ewma_alpha"] == .3 and new_data["affected_replay_cases"] == 19
controls_record = new_data["additional_control_sources"]
controls_manifest_path = ROOT / controls_record["manifest"]
assert hashlib.sha256(controls_manifest_path.read_bytes()).hexdigest() == controls_record["sha256"]
controls_manifest = json.loads(controls_manifest_path.read_text())
for filename, digest in controls_manifest["sources"].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
for filename, digest in controls_manifest["outputs"].items():
    assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest, filename
controls_verification_path = next(Path(p) for p in controls_manifest["sources"]
                                  if p.endswith("/controls/verification.json"))
controls_verification = json.loads(controls_verification_path.read_text())
assert controls_verification["status"] == "passed" and controls_verification["ewma_alpha"] == .3
assert hashlib.sha256(controls_verification_path.read_bytes()).hexdigest() == verification["controls_verification_sha256"]
assert set(new_data["completed_cases"]) == set(verification["cases"])
for name, row in verification["cases"].items():
    assert row["status"] == "passed"
    assert new_data["source_manifests"][name] == row["source_sha256"]
    summary_path = ROOT / "audit/lstm_comparison" / (name + ".json")
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == row["source_sha256"]["summary.json"]
    methods = {x["method"] for x in json.loads(summary_path.read_text())["by_action"].values()}
    assert {x for x in methods if x.startswith("EWMA_")} == {"EWMA_0.3"}, name
gate = build_winter_g_evidence.verify()
assert gate["ewma_alpha"] == .3
gate_path = build_winter_g_evidence.MANIFEST
assert figure_sources["additional_replay_cases"] == gate["additional_replay_cases"] == 6
assert figure_sources["additional_gate_manifest_sha256"] == hashlib.sha256(gate_path.read_bytes()).hexdigest()
assert new_data["additional_gate_sources"]["sha256"] == figure_sources["additional_gate_manifest_sha256"]
assert all(figure_sources["outputs"][name] == digest for name, digest in gate["figures"].items())
assert gate["matched_winter_validation"] == figure_sources["matched_winter_validation"] == "passed"
assert gate["matched_winter_replay_cases"] == figure_sources["matched_winter_replay_cases"] == 6
assert new_data["matched_winter_sources"]["sha256"] == figure_sources["additional_gate_manifest_sha256"]
excluded_periodic = (
    r"\bscheduled\b", r"periodic[- ](?:adapter|update)",
    r"(?:10|ten)[- ]min(?:ute)?(?:s)? (?:adapter|update)",
    r"ssec:periodic-shift", r"fig:periodic-drift", r"stab:frozen-adapter",
    r"fig19_drift_comparison", r"lstm_drift_rows", r"final_frozen_adapter_rows",
)
for name, text in docs.items():
    for pattern in excluded_periodic:
        assert not re.search(pattern, text, re.I), (name, "excluded periodic configuration", pattern)
assert all(not row["case"].startswith("drift_") for row in
           json.loads((ROOT / "audit/lstm_comparison/timing_references.json").read_text()))
assert len((ROOT / "generated/lstm_reference_rows.tex").read_text().splitlines()) == 3
with (ROOT / "audit/lstm_comparison/all_results.csv").open(newline="") as file:
    expected_current = [row for row in csv.DictReader(file) if not row["case"].startswith("drift_")]
with (ROOT / "audit/lstm_comparison/current_results.csv").open(newline="") as file:
    current_rows = list(csv.DictReader(file))
assert current_rows == expected_current
assert len({row["case"] for row in current_rows}) == 13
for filename in ("supplement/periodic_shift.tex", "fig19_drift_comparison.pdf",
                 "generated/lstm_drift_rows.tex", "generated/final_frozen_adapter_rows.tex"):
    assert not (ROOT / filename).exists(), ("excluded submission asset", filename)
result = {"documents": {k: {"labels": len(labels[k]), "figures": figures[k]} for k in docs},
          "drift_pdf_sha256": hashlib.sha256((ROOT / drift).read_bytes()).hexdigest(),
          "derived_arithmetic_and_al_checks": "passed",
          "ewma_alpha": .3,
          "global_ewma_alpha_consistency": "passed",
          "additional_control_source_and_table_hashes": "passed",
          "additional_control_policy_files": controls_verification["evaluation_policy_files"],
          "resource_integration_validation": "passed for new comparison",
          "common_request_validation": "passed for new comparison",
          "matched_memory_available": 0,
          "new_replay_cases": len(verification["cases"]),
          "additional_replay_cases": gate["additional_replay_cases"],
          "additional_gate_validation": gate["status"],
          "matched_winter_replay_cases": gate["matched_winter_replay_cases"],
          "matched_winter_validation": gate["matched_winter_validation"],
          "excluded_periodic_configuration": "absent from both documents and submission assets",
          "current_nonshift_numerical_companion_cases": 13,
          "current_nonshift_numerical_companion_sha256": hashlib.sha256(
              (ROOT / "audit/lstm_comparison/current_results.csv").read_bytes()).hexdigest(),
          "additional_gate_manifest_sha256": figure_sources["additional_gate_manifest_sha256"],
          "logical_policy_function_seeds": verification["logical_policy_function_seeds"],
          "new_comparison_verification_sha256": figure_sources["verification_sha256"]}
(ROOT / "audit/source_checks.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
