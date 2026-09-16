"""Build a coupled document set in an isolated directory before publication."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import build_winter_g_evidence

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(build_dir):
    if build_dir.parent != ROOT:
        raise ValueError("Build directory must be a direct child of the paper directory")
    gate = build_winter_g_evidence.verify()
    gate_manifest_hash = sha(build_winter_g_evidence.MANIFEST)
    original_verification = ROOT / "audit/lstm_comparison/verification.json"
    if json.loads(original_verification.read_text())["status"] != "passed":
        raise RuntimeError("Original comparison measurements are not verified")
    build_dir.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="run-", dir=build_dir))
    sources = {}
    for source in ROOT.iterdir():
        if source.name in {"main.pdf", "supplementary.pdf", "highlights.pdf", "xstring.tex"}:
            continue
        if source.name in {"sections", "supplement", "generated", "thumbnails"} or (
            " 2." not in source.name and source.suffix in {".tex", ".cls", ".bib", ".bst", ".pdf"}
        ) or source.name == "cas-common.sty":
            (stage / source.name).symlink_to(source)
            if source.is_file() and source.suffix != ".pdf":
                sources[source.name] = sha(source)
    for folder in ("sections", "supplement", "generated"):
        for source in sorted((ROOT / folder).glob("*.tex")):
            sources[str(source.relative_to(ROOT))] = sha(source)
    for source in sorted((ROOT / "thumbnails").glob("cas-*")):
        if source.is_file():
            sources[str(source.relative_to(ROOT))] = sha(source)
    figures = {source.name: sha(source) for source in ROOT.glob("fig*.pdf")}
    env = dict(os.environ)
    # Only staged journal styles are visible; standard packages come from TeX Live.
    env.update(TEXINPUTS=str(stage) + os.pathsep,
               BIBINPUTS=str(stage) + os.pathsep, BSTINPUTS=str(stage) + os.pathsep)

    def run(command, name):
        result = subprocess.run(command, cwd=stage, env=env, text=True,
                                errors="replace", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (stage / (name + ".stdout")).write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(result.stdout[-10000:])

    def latex(name, phase):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-file-line-error", name + ".tex"], name + "-" + phase)

    latex("main", "initial")
    run(["bibtex", "main"], "bibtex")
    for index in range(4):
        latex("supplementary", str(index))
        latex("main", str(index))
    latex("highlights", "final")
    documents = {}
    for name in ("main", "supplementary", "highlights"):
        log = (stage / (name + ".log")).read_text(errors="replace")
        warnings = re.findall(r"(?:LaTeX Warning:|Package \S+ Warning:|Overfull)[^\n]*", log)
        fatal = ("There were undefined references", "multiply defined",
                 "Citation ", "Rerun to get cross-references right")
        if any(token in log for token in fatal):
            raise RuntimeError("Unresolved document references: " + name + "\n" + "\n".join(warnings))
        match = re.search(r"Output written on .*?\((\d+) pages?", log)
        documents[name] = {"pdf_sha256": sha(stage / (name + ".pdf")),
                           "pages": int(match.group(1)) if match else None,
                           "warnings": warnings}
    for filename, digest in {**sources, **figures}.items():
        if sha(ROOT / filename) != digest:
            raise RuntimeError("Source changed during compilation: " + filename)
    verification = ROOT / "audit/lstm_comparison/verification.json"
    validation = json.loads(verification.read_text())["status"] if verification.exists() else "pending"
    assert sha(build_winter_g_evidence.MANIFEST) == gate_manifest_hash
    build_winter_g_evidence.verify()
    report = {"build_id": stage.name, "built_at_utc": datetime.now(timezone.utc).isoformat(),
              "stage": str(stage.relative_to(ROOT)), "source_sha256": sources,
              "figure_sha256": figures,
              "documents": documents, "measurement_validation": validation,
              "ewma_alpha": gate["ewma_alpha"],
              "current_replay_cases": gate["affected_replay_cases"],
              "additional_gate_validation": gate["status"],
              "additional_replay_cases": gate["additional_replay_cases"],
              "matched_winter_replay_cases": gate["matched_winter_replay_cases"],
              "matched_winter_validation": gate["matched_winter_validation"],
              "additional_gate_manifest_sha256": gate_manifest_hash,
              "measurement_verification_sha256": sha(verification) if verification.exists() else None}
    # Validate every document before replacing any public output.
    for name in documents:
        staged_output = ROOT / ("." + name + ".pdf.tmp")
        shutil.copy2(stage / (name + ".pdf"), staged_output)
    for name in documents:
        os.replace(ROOT / ("." + name + ".pdf.tmp"), ROOT / (name + ".pdf"))
    (ROOT / "audit/build_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build-positioning")
    args = parser.parse_args()
    build((ROOT / args.build_dir).resolve())
