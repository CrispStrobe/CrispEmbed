#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned fixtures and bounded, model-free upstream Marker references.

This tool is test-only. It does not convert/download/load neural checkpoints.
The native runtime never imports it. See docs/marker-feature-coverage.md.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from unittest.mock import patch

MARKER_REVISION = "71dd5fd7e51ab24f8624b6ba2fcb6014b4e40ed4"
SURYA_AUDIT_REVISION = "9d5aadfebb07939755c24e8a80d0e61b3be80895"
FONT_NAMES = {
    "regular": "DejaVuSans.ttf",
    "bold": "DejaVuSans-Bold.ttf",
    "mono": "DejaVuSansMono.ttf",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fixtures(args) -> int:
    """Create deterministic fixtures; all text and geometry are authored here."""
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pdfencrypt import StandardEncryption
    import pypdfium2 as pdfium

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fonts = {role: args.font_dir / name for role, name in FONT_NAMES.items()}
    for role, path in fonts.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required fixture font is missing: {path.name}")
        pdfmetrics.registerFont(TTFont("Fixture-" + role, str(path)))
    if not args.font_notice.is_file():
        raise FileNotFoundError("Pass --font-notice for the embedded DejaVu font license")
    (out / "FONT-NOTICE.txt").write_bytes(args.font_notice.read_bytes())

    cases = []

    def make(name, password=None):
        encrypted = None
        if password:
            encrypted = StandardEncryption(
                password, ownerPassword="marker-fixture-owner", canPrint=1,
                canModify=0, canCopy=1, canAnnotate=1, strength=128,
            )
        c = canvas.Canvas(str(out / (name + ".pdf")), pagesize=(612, 792),
                          invariant=1, pageCompression=1, encrypt=encrypted)
        c.setTitle("Owned CrispEmbed Marker fixture: " + name)
        c.setAuthor("CrispEmbed test fixtures")
        return c

    def line(c, text, x=48, y=720, size=12, face="regular"):
        c.setFont("Fixture-" + face, size)
        c.setFillColorRGB(0.05, 0.07, 0.10)
        c.drawString(x, y, text)

    def finish(c, name, pages, **extras):
        c.save()
        case = {
            "id": name,
            "pdf": name + ".pdf",
            "page_count": len(pages),
            "text_lines_by_page": pages,
            "logical_order": [s for page in pages for s in page],
            "must_contain": [s for page in pages for s in page],
            **extras,
        }
        case["pdf_sha256"] = sha256(out / case["pdf"])
        cases.append(case)

    name = "simpledigital"
    c = make(name)
    rows = ["Native document test", "Hello from CrispEmbed.", "PDF text stays searchable."]
    line(c, rows[0], y=720, size=22, face="bold")
    line(c, rows[1], y=673)
    line(c, rows[2], y=650)
    finish(c, name, [rows], expected_heading=rows[0])

    name = "unicode"
    c = make(name)
    rows = [
        "Unicode and escaping",
        "Café naïve résumé - Ελληνικά - кириллица.",
        "Literal <script>alert(1)</script> & <tag> text.",
        "Markdown *stars* _under_ [brackets] $19.95 | pipe.",
    ]
    line(c, rows[0], y=720, size=22, face="bold")
    for i, text in enumerate(rows[1:]):
        line(c, text, y=672 - i * 29, size=12)
    finish(c, name, [rows], expected_heading=rows[0],
           html_forbidden_tags=["script", "tag"],
           exact_unicode=["Café", "naïve", "résumé", "Ελληνικά", "кириллица"])

    name = "structure"
    c = make(name)
    rows = [
        "Document structure", "1. Overview", "A short paragraph introduces the examples.",
        "- First owned list item", "- Second owned list item", "2. Code example",
        "if (left < right) {", '    print("safe & literal");', "}",
    ]
    for text, y, size, face in [
        (rows[0], 720, 22, "bold"), (rows[1], 666, 16, "bold"),
        (rows[2], 635, 12, "regular"), (rows[3], 601, 12, "regular"),
        (rows[4], 579, 12, "regular"), (rows[5], 532, 16, "bold"),
        (rows[6], 499, 11, "mono"), (rows[7], 482, 11, "mono"),
        (rows[8], 465, 11, "mono"),
    ]:
        line(c, text, y=y, size=size, face=face)
    finish(c, name, [rows], expected_headings=[rows[0], rows[1], rows[5]],
           expected_list_items=rows[3:5], expected_code="\n".join(rows[6:]))

    name = "multicolumn"
    c = make(name)
    title = "Two-column reading order"
    left = ["Left column", "Left one stays together.", "Left two follows one.", "Left three ends this column."]
    right = ["Right column", "Right one begins here.", "Right two follows one.", "Right three ends the page."]
    line(c, title, y=720, size=22, face="bold")
    # The content stream draws complete columns, preserving an explicit gold
    # order independent of either document processor's geometry heuristics.
    for x, texts in [(48, left), (326, right)]:
        for i, text in enumerate(texts):
            line(c, text, x=x, y=662 - i * 28,
                 size=15 if i == 0 else 11, face="bold" if i == 0 else "regular")
    finish(c, name, [[title] + left + right], expected_columns=[left, right],
           expected_heading=title)

    name = "table"
    c = make(name)
    title = "Aligned table fixture"
    grid = [["Item", "Quantity", "Price"], ["Alpha", "2", "$10"],
            ["Beta", "3", "$15"], ["Gamma", "4", "$20"]]
    line(c, title, y=720, size=22, face="bold")
    for r, cells in enumerate(grid):
        for x, text in zip([48, 250, 420], cells):
            line(c, text, x=x, y=657 - r * 32, size=12,
                 face="bold" if r == 0 else "regular")
    line(c, "All values are invented test data.", y=477)
    finish(c, name, [[title] + [v for row in grid for v in row] +
                     ["All values are invented test data."]], expected_table=grid,
           expected_heading=title)

    name = "multipage"
    c = make(name)
    pages = []
    body_pages = []
    for i in range(3):
        rows = ["Owned repeated running header", f"Chapter {i + 1}",
                f"Page {i + 1} has distinct body content.",
                "This sentence is part of the document body.", f"Page {i + 1} of 3"]
        line(c, rows[0], y=766, size=9)
        line(c, rows[1], y=710, size=22, face="bold")
        line(c, rows[2], y=664)
        line(c, rows[3], y=642)
        line(c, rows[4], y=26, size=9)
        pages.append(rows)
        body_pages.append(rows[1:4])
        c.showPage()
    finish(c, name, pages, body_lines_by_page=body_pages,
           must_contain=[s for page in body_pages for s in page],
           repeated_header="Owned repeated running header")

    name = "rotated"
    c = make(name)
    c.setPageRotation(90)
    rows = ["Rotated page", "Rotation preserves this sentence."]
    line(c, rows[0], x=80, y=300, size=22, face="bold")
    line(c, rows[1], x=80, y=260, size=12)
    finish(c, name, [rows], page_rotation_degrees=90)

    name = "encrypted"
    password = "marker-fixture"
    c = make(name, password=password)
    rows = ["Encrypted fixture", "The supplied test password opens this PDF."]
    line(c, rows[0], y=720, size=22, face="bold")
    line(c, rows[1], y=673)
    finish(c, name, [rows], password=password,
           password_notice="Public synthetic fixture password, not a credential.",
           upstream_reference="skip: the pinned PdfProvider does not accept a password")

    for case in cases:
        pdf = pdfium.PdfDocument(str(out / case["pdf"]), password=case.get("password"))
        try:
            case["pngs"] = []
            for i in range(len(pdf)):
                page = pdf[i]
                bitmap = page.render(scale=2)
                name = f"{case['id']}-page-{i + 1}.png"
                bitmap.to_pil().convert("RGB").save(out / name)
                bitmap.close()
                page.close()
                case["pngs"].append({"file": name, "sha256": sha256(out / name)})
        finally:
            pdf.close()

    write_json(out / "expected.json", {
        "schema_version": 1,
        "description": "Owned text/geometry gold, generated before reference/native outputs.",
        "license": "MIT for fixture text and generator; embedded fonts retain FONT-NOTICE.txt.",
        "font_files": {role: {"name": path.name, "sha256": sha256(path)}
                       for role, path in fonts.items()},
        "cases": cases,
    })
    print(json.dumps({"fixture_count": len(cases), "page_count": sum(x['page_count'] for x in cases),
                      "gold": str(out / "expected.json")}, indent=2))
    return 0


def normalized(text: str) -> str:
    return " ".join(text.split())


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def text_metrics(expected: str, actual: str) -> dict:
    a, b = normalized(expected), normalized(actual)
    aw, bw = a.split(), b.split()
    return {
        "normalization": "collapse Unicode whitespace only",
        "expected_characters": len(a), "actual_characters": len(b),
        "expected_words": len(aw), "actual_words": len(bw),
        "character_edits": edit_distance(a, b),
        "word_edits": edit_distance(aw, bw),
        "cer": edit_distance(a, b) / max(1, len(a)),
        "wer": edit_distance(aw, bw) / max(1, len(aw)),
        "exact_after_whitespace": a == b,
    }


class ForbiddenPredictor:
    def __init__(self, name: str, attempts: list):
        self.name, self.attempts = name, attempts
        self.disable_tqdm = True

    def __call__(self, *args, **kwargs):
        self.attempts.append(self.name)
        raise RuntimeError("Model invocation rejected: " + self.name)

    def __getattr__(self, name):
        def forbidden(*args, **kwargs):
            self.attempts.append(self.name + "." + name)
            raise RuntimeError("Model method rejected: " + self.name + "." + name)
        return forbidden


@contextmanager
def offline_guards(events: dict):
    """Deny network, checkpoint opens and inference-process launch in-process.

    The audit hook persists until this short-lived test process exits. It is
    not advertised as a general-purpose security sandbox.
    """
    state = {"block_processes": False}

    def reject_network(*args, **kwargs):
        events["network"].append("socket operation")
        raise RuntimeError("Network access rejected by model-free reference")

    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            name = os.fsdecode(args[0])
            if name.lower().endswith((".gguf", ".safetensors", ".pt", ".pth")):
                events["checkpoint"].append(Path(name).name)
                raise RuntimeError("Checkpoint access rejected by model-free reference")
        if state["block_processes"] and event in ("subprocess.Popen", "os.system", "os.fork"):
            events["process"].append(event)
            raise RuntimeError("Process launch rejected by model-free reference")

    sys.addaudithook(audit)
    with patch.object(socket.socket, "connect", reject_network), \
         patch.object(socket.socket, "connect_ex", reject_network), \
         patch.object(socket, "create_connection", reject_network), \
         patch.object(socket, "getaddrinfo", reject_network):
        yield state


def renderer_probe():
    """Construct owned schema input; execute the unchanged upstream renderers."""
    from marker.schema.blocks import Code, Equation, ListItem, SectionHeader, Table, Text
    from marker.schema.document import Document
    from marker.schema.groups.list import ListGroup
    from marker.schema.groups.page import PageGroup
    from marker.schema.polygon import PolygonBox
    from marker.schema.text.line import Line
    from marker.schema.text.span import Span

    page = PageGroup(page_id=0, polygon=PolygonBox.from_bbox([0, 0, 612, 792]),
                     children=[], structure=[], text_extraction_method="pdftext")
    document = Document(filepath="owned-renderer-probe", pages=[page])
    y = 40

    def add(cls, parent=None, **kwargs):
        nonlocal y
        block = page.add_full_block(cls(page_id=0,
            polygon=PolygonBox.from_bbox([40, y, 570, y + 24]), **kwargs))
        (page if parent is None else parent).add_structure(block)
        y += 28
        return block

    def text(cls, value, parent=None, **kwargs):
        block = add(cls, parent=parent, **kwargs)
        line = add(Line, parent=block)
        add(Span, parent=line, text=value, font="Fixture", font_weight=400,
            font_size=12, minimum_position=0, maximum_position=len(value), formats=["plain"])
        return block

    text(SectionHeader, "Renderer contract", heading_level=2)
    literal = 'Literal <script> & "quotes"; *stars* _under_ $5.'
    text(Text, literal)
    group = add(ListGroup)
    text(ListItem, "First item", parent=group)
    text(ListItem, "Second item", parent=group)
    code = 'if (left < right) {\n  return "safe & literal";\n}'
    add(Code, code=code)
    # These are trusted, owned schema literals, not raw OCR output. Preserve
    # original's handling of a literal pipe in a table cell for comparison.
    table_html = ("<table><tbody><tr><th>A</th><th>B</th></tr>"
                  "<tr><td>one | two</td><td>&lt;cell&gt;</td></tr></tbody></table>")
    add(Table, html=table_html)
    add(Equation, html='<math display="block">x^2 + y^2 = z^2</math>')
    gold = {
        "description": "Owned explicit schema, not layout/OCR inference",
        "heading": "Renderer contract", "literal": literal,
        "list_items": ["First item", "Second item"], "code": code,
        "table_html": table_html, "equation": "x^2 + y^2 = z^2",
        "top_level_types": ["SectionHeader", "Text", "ListGroup", "Code", "Table", "Equation"],
    }
    return document, gold


def render_outputs(document, config, output: Path) -> dict:
    from marker.renderers.html import HTMLRenderer
    from marker.renderers.json import JSONRenderer
    from marker.renderers.markdown import MarkdownRenderer

    output.mkdir(parents=True, exist_ok=False)
    markdown = MarkdownRenderer(config)(document)
    html = HTMLRenderer(config)(document)
    data = JSONRenderer(config)(document).model_dump(mode="json")
    (output / "document.md").write_text(markdown.markdown, encoding="utf-8")
    (output / "document.html").write_text(html.html, encoding="utf-8")
    write_json(output / "document.json", data)
    return {"markdown": markdown.markdown, "html": html.html, "json": data}


def capture(args) -> int:
    upstream, fixtures_dir, output = (p.resolve() for p in
                                      (args.upstream, args.fixtures, args.output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Reference output must be empty; preserve earlier attempts")
    output.mkdir(parents=True, exist_ok=True)
    receipt = {"schema_version": 1, "kind": "original-marker-model-free-reference",
               "status": "started", "cases": [], "neural_inference_tested": False}
    events = {"network": [], "checkpoint": [], "process": [], "predictor": []}
    started = time.monotonic()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream,
                                           text=True).strip()
        if revision != MARKER_REVISION:
            raise ValueError("Wrong Marker source revision: " + revision)
        dirty = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", "marker"],
                                        cwd=upstream, text=True).strip()
        if dirty:
            raise ValueError("Upstream Marker code has local edits: " + dirty)
        if not args.font.is_file():
            raise FileNotFoundError("An existing font is required; font downloading is forbidden")
        source_manifest = {str(p.relative_to(upstream)): sha256(p)
                           for p in sorted((upstream / "marker").rglob("*.py"))}
        write_json(output / "upstream-source-sha256.json", source_manifest)
        gold = json.loads((fixtures_dir / "expected.json").read_text())
        receipt.update({
            "marker_revision": revision,
            "surya_source_license_audit_revision": SURYA_AUDIT_REVISION,
            "surya_runtime_source": "published surya-ocr package; version and source hashes below",
            "fixture_gold_sha256": sha256(fixtures_dir / "expected.json"),
            "reference_script_sha256": sha256(Path(__file__)),
            "font": {"name": args.font.name, "sha256": sha256(args.font)},
            "dependencies": {name: importlib.metadata.version(name) for name in
                             ["surya-ocr", "pdftext", "pypdfium2", "torch", "torchvision",
                              "transformers", "pillow", "pydantic", "markdownify",
                              "beautifulsoup4", "reportlab"]},
        })
        os.environ.update({
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TORCH_DEVICE": "cpu",
            "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "",
            "SURYA_INFERENCE_AUTOSTART": "false", "FAST_LAYOUT_SERVER_AUTOSTART": "false",
            "OCR_ERROR_SERVER_AUTOSTART": "false", "DETECTOR_SERVER_AUTOSTART": "false",
        })
        sys.path.insert(0, str(upstream))
        with offline_guards(events) as guard_state:
            import torch
            torch.set_num_threads(1)
            from bs4 import BeautifulSoup
            from marker.converters.pdf import PdfConverter
            from marker.providers.pdf import PdfProvider
            from marker.settings import settings
            # Both upstream projects may use namespace-package roots without
            # __file__; authenticate a real imported module instead.
            if Path(sys.modules[PdfConverter.__module__].__file__).resolve() != upstream / "marker/converters/pdf.py":
                raise RuntimeError("Marker imported from an unexpected installation")
            surya_root = Path(sys.modules["surya.settings"].__file__).parent
            write_json(output / "surya-runtime-source-sha256.json", {
                str(p.relative_to(surya_root)): sha256(p)
                for p in sorted(surya_root.rglob("*.py"))
            })
            settings.FONT_PATH = str(args.font.resolve())
            settings.TORCH_DEVICE = "cpu"
            guard_state["block_processes"] = True

            checks = {}
            for key, call in [
                ("network", lambda: socket.create_connection(("127.0.0.1", 1))),
                ("checkpoint", lambda: open("forbidden-reference-selftest.gguf", "rb")),
                ("predictor", lambda: ForbiddenPredictor("selftest", events["predictor"])()),
                ("process", lambda: subprocess.run(["forbidden-reference-selftest"])),
            ]:
                try:
                    call()
                except RuntimeError:
                    checks[key] = len(events[key]) == 1
                else:
                    checks[key] = False
                events[key].clear()
            if not all(checks.values()):
                raise RuntimeError("A reference guard self-test failed")
            receipt["guard_selftests"] = checks

            config = {
                "disable_ocr": True, "force_layout_block": "Text", "mode": "fast",
                "use_llm": False, "pdftext_workers": 1, "disable_tqdm": True,
                "extract_images": False, "paginate_output": False,
            }
            receipt["configuration"] = config
            models = {name: ForbiddenPredictor(name, events["predictor"]) for name in
                      ["inference_manager", "layout_model", "fast_layout_model",
                       "recognition_model", "ocr_error_model"]}
            converter = PdfConverter(artifact_dict=models, config=config)
            for case in gold["cases"]:
                entry = {"id": case["id"], "pdf_sha256": sha256(fixtures_dir / case["pdf"])}
                if entry["pdf_sha256"] != case["pdf_sha256"]:
                    raise RuntimeError("Fixture changed after gold was frozen: " + case["id"])
                if case.get("password"):
                    entry.update({"status": "not-run", "reason": case["upstream_reference"]})
                    receipt["cases"].append(entry)
                    continue
                case_start = time.monotonic()
                path = str(fixtures_dir / case["pdf"])
                provider = PdfProvider(path, config)
                extraction = []
                for page_id in provider.page_range:
                    extraction.append({
                        "page_id": page_id,
                        "lines": [{"text": "".join(s.text for s in line.spans).strip(),
                                   "bbox": line.line.polygon.bbox,
                                   "spans": [{"text": s.text, "bbox": s.polygon.bbox,
                                              "font": s.font, "font_size": s.font_size,
                                              "formats": s.formats} for s in line.spans]}
                                  for line in provider.page_lines[page_id]],
                    })
                document = converter.build_document(path)
                actual = render_outputs(document, config, output / case["id"])
                write_json(output / case["id"] / "provider.json", extraction)
                expected_text = " ".join(case["logical_order"])
                extracted_text = " ".join(line["text"] for p in extraction for line in p["lines"])
                soup = BeautifulSoup(actual["html"], "html.parser")
                rendered_text = soup.get_text(" ", strip=True)
                content = {fragment: normalized(fragment) in normalized(rendered_text)
                           for fragment in case["must_contain"]}
                position, in_order = 0, True
                for fragment in case["logical_order"]:
                    found = normalized(rendered_text).find(normalized(fragment), position)
                    if found < 0:
                        in_order = False
                        break
                    position = found + len(normalized(fragment))
                safety = {tag: soup.find(tag) is None for tag in case.get("html_forbidden_tags", [])}
                entry.update({
                    "status": "captured", "elapsed_seconds": time.monotonic() - case_start,
                    "pages": len(document.pages),
                    "page_extraction_methods": [p.text_extraction_method for p in document.pages],
                    "provider_vs_authored_gold": text_metrics(expected_text, extracted_text),
                    "html_text_vs_authored_gold": text_metrics(expected_text, rendered_text),
                    "content_checks": content, "literal_html_checks": safety,
                    "authored_logical_order_preserved": in_order,
                    "output_sha256": {p.name: sha256(p) for p in sorted((output / case["id"]).iterdir())},
                    "upstream_page_stats": actual["json"]["metadata"]["page_stats"],
                })
                receipt["cases"].append(entry)

            document, probe_gold = renderer_probe()
            rendered = render_outputs(document, config, output / "renderer-probe")
            write_json(output / "renderer-probe" / "gold.json", probe_gold)
            soup = BeautifulSoup(rendered["html"], "html.parser")
            md = rendered["markdown"]
            probe_checks = {
                "heading_level_2": soup.h2 is not None and soup.h2.get_text(strip=True) == probe_gold["heading"],
                "literal_script_is_text": soup.find("script") is None and "&lt;script&gt;" in rendered["html"],
                "markdown_literal_asterisks": r"\*stars\*" in md,
                "markdown_literal_underscores": r"\_under\_" in md,
                "markdown_literal_dollar": r"\$5" in md,
                "list_items": [x.get_text(strip=True) for x in soup.select("li")] == probe_gold["list_items"],
                "code_escaped_in_html": soup.pre is not None and soup.pre.get_text().strip() == probe_gold["code"],
                "table_html_keeps_pipe": soup.table is not None and "one | two" in soup.table.get_text(" "),
                "equation_retained": probe_gold["equation"] in md,
                "json_document_pages": rendered["json"]["block_type"] == "Document" and len(rendered["json"]["children"]) == 1,
            }
            receipt["renderer_probe"] = {
                "checks": probe_checks,
                "markdown_table_pipe_behavior": "The original Markdown renderer replaces literal cell pipes with spaces; HTML retains them.",
                "output_sha256": {p.name: sha256(p) for p in sorted((output / "renderer-probe").iterdir())},
            }
            receipt["unexpected_attempts"] = events
            controlled = not any(events.values())
            captured = [c for c in receipt["cases"] if c["status"] == "captured"]
            content_ok = all(all(c["content_checks"].values()) and all(c["literal_html_checks"].values()) for c in captured)
            receipt["summary"] = {
                "captured_pdf_cases": len(captured), "captured_pdf_pages": sum(c["pages"] for c in captured),
                "neural_model_calls": len(events["predictor"]), "network_attempts": len(events["network"]),
                "checkpoint_open_attempts": len(events["checkpoint"]), "process_launch_attempts": len(events["process"]),
                "owned_content_checks_passed": content_ok,
                "renderer_checks_passed": sum(probe_checks.values()), "renderer_check_count": len(probe_checks),
                "forced_layout_order_matches": sum(c["authored_logical_order_preserved"] for c in captured),
                "scope": "Deterministic source behavior only; no learned layout/OCR or native parity claim.",
            }
            receipt["status"] = "pass" if controlled and content_ok and all(probe_checks.values()) else "captured-with-differences"
    except Exception as exc:
        receipt["status"] = "failed"
        message = str(exc)
        for path in [args.upstream, args.fixtures, args.output, args.font, Path(sys.prefix)]:
            message = message.replace(str(path.resolve()), "<" + path.name + ">")
        receipt["error"] = {"type": type(exc).__name__, "message": message}
        receipt["unexpected_attempts"] = events
    receipt["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "summary": receipt.get("summary"),
                      "error": receipt.get("error"), "receipt": str(output / "receipt.json")}, indent=2))
    return 0 if receipt["status"] == "pass" else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixtures", help="Generate owned digital PDF/image fixtures")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--font-dir", type=Path, required=True)
    fixture.add_argument("--font-notice", type=Path, required=True)
    fixture.set_defaults(run=fixtures)
    reference = commands.add_parser("capture", help="Run actual original Marker without models")
    reference.add_argument("--upstream", type=Path, required=True)
    reference.add_argument("--fixtures", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--font", type=Path, required=True)
    reference.set_defaults(run=capture)
    args = p.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
