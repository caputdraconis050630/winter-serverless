"""Render every page and record text-boundary checks for visual review."""
import json
import hashlib
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
build = json.loads((ROOT / "audit/build_manifest.json").read_text())
OUT = ROOT / "audit/visual" / build["build_id"]
OUT.mkdir(parents=True, exist_ok=True)
report = {}
for name in ("main", "supplementary", "highlights"):
    source = ROOT / (name + ".pdf")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == build["documents"][name]["pdf_sha256"], (name, "not the built PDF")
    doc = fitz.open(source)
    pages = []
    issues = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
        filename = OUT / f"{name}-{i+1:02d}.png"
        pix.save(filename)
        picture = Image.open(filename).convert("RGB")
        picture.thumbnail((450, 600))
        tile = Image.new("RGB", (480, 635), "#eeeeee")
        tile.paste(picture, ((480 - picture.width) // 2, 25))
        ImageDraw.Draw(tile).text((12, 8), f"{name} page {i+1}", fill="black")
        pages.append(tile)
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    x0, y0, x1, y1 = s["bbox"]
                    if x0 < 0 or y0 < 0 or x1 > page.rect.width or y1 > page.rect.height:
                        issues.append({"page": i+1, "text": s["text"], "bbox": s["bbox"]})
        if "??" in page.get_text():
            issues.append({"page": i+1, "text": "unresolved reference marker"})
    for start in range(0, len(pages), 6):
        sheet = Image.new("RGB", (1440, 1270), "#cccccc")
        for j, tile in enumerate(pages[start:start+6]):
            sheet.paste(tile, ((j % 3) * 480, (j // 3) * 635))
        sheet.save(OUT / f"{name}-sheet-{start//6+1}.png")
    report[name] = {"pages": len(doc), "issues": issues,
                    "pdf_sha256": digest, "build_id": build["build_id"],
                    "render_dir": str(OUT.relative_to(ROOT))}
(OUT / "checks.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
(ROOT / "audit/visual/checks.json").write_text(json.dumps(report, indent=2))
if any(row["issues"] for row in report.values()):
    raise SystemExit("PDF inspection found unresolved references or text outside page boundaries")
