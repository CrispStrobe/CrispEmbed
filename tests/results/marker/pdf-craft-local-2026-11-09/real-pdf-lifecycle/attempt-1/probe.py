#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned real-PDF text-mode lifecycle probe; no models or network requests.

Requires numpy and pypdfium2==5.10.1. Run with --repo CHECKOUT --lib LIBRARY
--pdfium REAL_PDFIUM_LIBRARY --out NEW_OUTPUT_DIRECTORY. Keep each attempt.
This qualifies this authored text fixture, not general PDF rendering/OCR quality.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


FORMATS = ("markdown", "html", "json", "chunks")
TEXT = ("First owned real PDF page.", "Second owned real PDF page.", "Third owned real PDF page.")
LABELS = ("iv", "章-2", "📖-3")
OPTIONAL = ("ARTIFACTS", "INCREMENTAL", "SOURCE_LINKS", "REFERENCES",
            "MULTILINGUAL_NORMALIZE", "REPETITION_CHECK")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned_pdf():
    # Standard fonts and uncompressed streams keep the fixture authored/readable.
    labels = (b"/PageLabels << /Nums [0 << /S /r /St 4 >> "
              b"1 << /S /D /St 2 /P <FEFF7AE0002D> >> "
              b"2 << /S /D /St 3 /P <FEFFD83DDCD6002D> >>] >>")
    objects = [b"<< /Type /Catalog /Pages 2 0 R " + labels + b" >>",
               b"<< /Type /Pages /Kids [4 0 R 6 0 R 8 0 R] /Count 3 >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, text in enumerate(TEXT):
        objects.append(("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] "
                        "/Resources << /Font << /F1 3 0 R >> >> /Contents "
                        f"{5 + 2 * i} 0 R >>").encode("ascii"))
        stream = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET\n".encode("ascii")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream")
    data, offsets = bytearray(b"%PDF-1.7\n"), [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend((f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
                 f"startxref\n{xref}\n%%EOF\n").encode())
    return bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "lib", "pdfium", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.repo, args.lib, args.pdfium = (p.resolve(strict=True) for p in (args.repo, args.lib, args.pdfium))
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo / "python"))
    from crispembed import CrispMarker

    rows, outputs = [], {}

    def check(name, condition):
        rows.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    def rejects(name, action, fragment):
        try:
            action()
        except RuntimeError as exc:
            check(name, fragment in str(exc))
        else:
            check(name, False)

    def gates(value):
        for suffix in OPTIONAL:
            key = "CRISPEMBED_MARKER_" + suffix
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def marker(**options):
        return CrispMarker(mode="text", lib_path=str(args.lib), n_threads=1, extract_images=False,
                           **dict({"pdfium_path": args.pdfium}, **options))

    def render_all(context, tag):
        rendered = {fmt: context.render(fmt) for fmt in FORMATS}
        for fmt, value in rendered.items():
            target = args.out / f"{tag}.{fmt}"
            target.write_text(value, encoding="utf-8")
            outputs[target.name] = sha(target)
        return rendered

    def walk(node):
        yield node
        for child in node.get("children") or []:
            yield from walk(child)

    def provenance(rendered, indices, tag):
        document = json.loads(rendered["json"])
        chunks = json.loads(rendered["chunks"])
        pages = document["children"]
        records = {item["id"]: item for item in document["source_records"]}
        check(tag + ": physical page indices and labels",
              [page["physical_page_index"] for page in pages] == indices and
              [page["printed_page_label"] for page in pages] == [LABELS[i] for i in indices] and
              all(page["page_label_source"] == "pdfium-page-label" and
                  page["page_label_status"] == "resolved" for page in pages))
        check(tag + ": physical source records",
              set(item["page"] for item in records.values()) == set(indices) and
              all(any(item["text"] == TEXT[i] and item["page"] == i for item in records.values()) for i in indices))
        valid = True
        mapped = 0
        for page, index in zip(pages, indices):
            for block in walk(page):
                if not block.get("text") or block.get("children"):
                    continue
                text, cursor = block["text"].encode("utf-8"), 0
                spans = block.get("source_ranges", [])
                valid &= bool(spans)
                for span in spans:
                    source = records.get(span["source_id"])
                    if source is None:
                        valid = False
                        continue
                    original = source["text"].encode("utf-8")
                    valid &= (source["page"] == index and span["transform"] == "copy" and
                              span["text_start"] == cursor and 0 <= span["source_start"] <= span["source_end"] <= len(original) and
                              text[span["text_start"]:span["text_end"]] == original[span["source_start"]:span["source_end"]])
                    cursor = span["text_end"]
                    mapped += 1
                valid &= cursor == len(text)
        check(tag + ": exact UTF-8 source-range reconstruction", valid and mapped >= len(indices))
        check(tag + ": retrieval chunks retain physical provenance",
              chunks["source_records"] == document["source_records"] and
              set(map(int, chunks["page_info"])) == set(indices) and
              set(p for chunk in chunks["assembled_chunks"] for p in chunk["pages"]) == set(indices))

    error, reference_info = None, {}
    source = args.out / "owned-three-page.pdf"
    source.write_bytes(owned_pdf())
    original = source.read_bytes()
    os.environ.update(CRISPEMBED_MARKER="1", CRISPEMBED_OFFLINE="1", CRISPEMBED_TELEOCR="0",
                      CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE="0", CRISPEMBED_ACCEPT_CC_BY_NC_LICENSE="0")
    try:
        # Separate ownership of PDFium's process-global initialization between
        # its Python helper and CrispEmbed's native provider.
        reference_code = """import json, sys, pypdfium2
with pypdfium2.PdfDocument(sys.argv[1]) as document:
    result = {'count': len(document), 'labels': [], 'texts': [], 'version': str(pypdfium2.PDFIUM_INFO)}
    for i in range(len(document)):
        result['labels'].append(document.get_page_label(i))
        with document[i] as page, page.get_textpage() as text:
            result['texts'].append(text.get_text_range().strip())
print(json.dumps(result))
"""
        reference = subprocess.run([sys.executable, "-B", "-c", reference_code, str(source)],
                                   capture_output=True, text=True, timeout=30, check=True)
        reference_info = json.loads(reference.stdout)
        check("real PDFium accepts three owned pages", reference_info["count"] == 3)
        check("real PDFium decodes printed labels", reference_info["labels"] == list(LABELS))
        for i, text in enumerate(reference_info["texts"]):
            check(f"authored page {i} text is exact", text == TEXT[i])
        gates(None)
        with marker() as legacy:
            legacy.convert(source)
            baseline = render_all(legacy, "gate-unset")
        for value in ("0", "FaLsE", "OfF", "nO"):
            gates(value)
            with marker() as legacy:
                legacy.convert(source)
                check(f"all four renderers preserve current gate-off output ({value})",
                      {fmt: legacy.render(fmt) for fmt in FORMATS} == baseline)
        gates("0")
        os.environ.update(CRISPEMBED_MARKER_ARTIFACTS="1", CRISPEMBED_MARKER_INCREMENTAL="1")
        with marker() as legacy:
            legacy.convert(source)
            check("artifact/lifecycle gates alone preserve legacy four-renderer output",
                  {fmt: legacy.render(fmt) for fmt in FORMATS} == baseline)
        os.environ.update(CRISPEMBED_MARKER_SOURCE_LINKS="1", CRISPEMBED_MARKER_REFERENCES="1")
        with marker() as legacy:
            legacy.convert(source)
            enabled = render_all(legacy, "enabled-legacy")
            provenance(enabled, [0, 1, 2], "legacy")
        raw, view = args.out / "resume.cemk", args.out / "processed.cemk"
        with marker() as first:
            state = first.start(source, raw)
            check("real PDF start creates durable empty checkpoint", state["completed_pages"] == 0 and raw.is_file())
            state = first.step(1)
            committed = raw.read_bytes()
            check("one-page budget commits only page zero", state["completed_pages"] == 1 and
                  state["next_page"] == 1 and TEXT[0] in first.render() and TEXT[1] not in first.render())
            first.cancel()
            check("cancellation preserves committed real PDF checkpoint",
                  first.step(64)["state"] == "cancelled" and raw.read_bytes() == committed)
        with marker() as resumed:
            state = resumed.start(source, raw, resume=True)
            check("fresh context resumes one committed page", state["completed_pages"] == 1 and not state["cancel_requested"])
            check("resumed one-page budget commits page one", resumed.step(1)["completed_pages"] == 2)
            state = resumed.step(64)
            check("resumed job completes three real PDF pages", state["state"] == "complete" and state["completed_pages"] == 3)
            complete = render_all(resumed, "incremental")
            for fmt in FORMATS:
                check(f"incremental {fmt} equals enabled legacy conversion", complete[fmt] == enabled[fmt])
            provenance(complete, [0, 1, 2], "incremental")
            resumed.save_extraction(view)
        with marker(first_page=1, last_page=2) as selected:
            selected.start(source, args.out / "selected.cemk")
            check("selected real pages preserve physical cursor", selected.step(1)["next_page"] == 2)
            selected.step(64)
            provenance(render_all(selected, "selected"), [1, 2], "selected")
        failed_raw = args.out / "identity-failure.cemk"
        with marker() as failed:
            failed.start(source, failed_raw)
            failed.step(1)
            committed = failed_raw.read_bytes()
            source.write_bytes(original + b"% owned identity change\n")
            rejects("changed real PDF source fails next step", lambda: failed.step(1), "identity changed")
            check("identity failure keeps committed page and checkpoint", failed.progress["completed_pages"] == 1 and
                  failed.progress["state"] == "error" and failed_raw.read_bytes() == committed and
                  TEXT[0] in failed.render() and TEXT[1] not in failed.render())
        with marker() as mismatch:
            rejects("changed source cannot resume original checkpoint",
                    lambda: mismatch.start(source, failed_raw, resume=True), "does not match")
            check("failed resume leaves durable checkpoint unchanged", failed_raw.read_bytes() == committed)
        source.write_bytes(original)
        with marker() as restored:
            check("restored source resumes original completed unit",
                  restored.start(source, failed_raw, resume=True)["completed_pages"] == 1)
            restored.step(64)
            check("restored source finishes with exact four-renderer output",
                  {fmt: restored.render(fmt) for fmt in FORMATS} == enabled)
        gates("0")
        os.environ["CRISPEMBED_MARKER_ARTIFACTS"] = "1"
        with marker(pdfium_path=args.out / "nonexistent-pdfium.so", dpi=150) as reader:
            reader.open_extraction(view)
            reopened = render_all(reader, "reopened")
            for fmt in FORMATS:
                check(f"processed {fmt} reopens without usable PDFium or matching processing config",
                      reopened[fmt] == complete[fmt])
            provenance(reopened, [0, 1, 2], "reopened")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for path, replacement in ((args.out, "<OUTPUT>"), (args.repo, "<REPOSITORY>"),
                                  (args.lib, "<NATIVE_LIBRARY>"), (args.pdfium, "<PDFIUM_LIBRARY>")):
            error = error.replace(str(path), replacement)
    finally:
        source.write_bytes(original)
    report = {
        "kind": "marker-owned-real-pdf-lifecycle", "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
        "probe_sha256": sha(__file__), "wrapper_sha256": sha(args.repo / "python/crispembed/_marker.py"),
        "library_sha256": sha(args.lib), "pdfium_sha256": sha(args.pdfium), "fixture_sha256": sha(source),
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pypdfium2")},
        "pdfium_version": reference_info.get("version"), "python_version": sys.version.split()[0],
        "model_weights_used": False, "general_pdf_or_ocr_quality_qualified": False,
        "fixture_page_count": 3, "expected_printed_labels": list(LABELS), "checks": rows,
        "output_sha256": outputs, "passed_count": sum(row["passed"] for row in rows), "total_count": len(rows),
        "passed": error is None and bool(rows) and all(row["passed"] for row in rows),
    }
    if error:
        report["error"] = error
    (args.out / "receipt.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{report['passed_count']}/{report['total_count']} owned real-PDF checks passed")
    if error:
        print(error, file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
