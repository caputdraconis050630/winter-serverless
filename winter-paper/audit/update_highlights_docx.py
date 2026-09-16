"""Synchronize the existing DOCX package using structured WordprocessingML."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent.parent
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W)
def q(name):
    return "{" + W + "}" + name

tex = (ROOT / "highlights.tex").read_text()
items = [s.replace(r"\%", "%") for s in re.findall(r"\\item (.+)", tex)]
assert len(items) == 4 and max(map(len, items)) <= 85
title = re.search(r"^\\noindent (.+)$", tex, re.MULTILINE).group(1)
path = ROOT / "highlights.docx"
with ZipFile(path) as source:
    members = {name: source.read(name) for name in source.namelist()}
document = ET.fromstring(members["word/document.xml"])
body = document.find(q("body"))
section = body.find(q("sectPr"))
body.clear()
for index, text in enumerate(["Highlights", title] + items):
    para = ET.SubElement(body, q("p"))
    if index == 0:
        props = ET.SubElement(para, q("pPr"))
        ET.SubElement(props, q("pStyle"), {q("val"): "Title"})
    run = ET.SubElement(para, q("r"))
    ET.SubElement(run, q("t")).text = ("\u2022 " if index >= 2 else "") + text
if section is not None:
    body.append(section)
members["word/document.xml"] = ET.tostring(document, encoding="utf-8", xml_declaration=True)
with ZipFile(path, "w", ZIP_DEFLATED) as target:
    for name, data in members.items():
        target.writestr(name, data)
with ZipFile(path) as target:
    written = ET.fromstring(target.read("word/document.xml"))
    assert [t.text for t in written.iter(q("t"))] == ["Highlights", title] + ["\u2022 " + s for s in items]
print("Highlights synchronized:", [len(s) for s in items])
