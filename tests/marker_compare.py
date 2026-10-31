#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent native text-mode comparison against owned gold and actual Marker.

No model inference is requested. Differences from original forced-text layout
are retained rather than hidden by a parity claim. Run after the native build.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from marker_reference import normalized, sha256, text_metrics, write_json


def walk_blocks(document):
    def visit(block):
        yield block
        for child in block.get("children") or []:
            yield from visit(child)
    for page in document["children"]:
        for block in page.get("children") or []:
            yield from visit(block)


def ordered_fragments(fragments, text):
    text, position = normalized(text), 0
    for fragment in fragments:
        found = text.find(normalized(fragment), position)
        if found < 0:
            return False
        position = found + len(normalized(fragment))
    return True


def geometry_checks(data):
    errors = []
    ids = []
    for page in data["children"]:
        width, height = page["bbox"][2:]
        def visit(block):
            ids.append(block["id"])
            box = block.get("bbox", [])
            if len(box) != 4 or not all(math.isfinite(x) for x in box):
                errors.append({"id": block["id"], "reason": "invalid bbox"})
            elif not (-0.001 <= box[0] <= box[2] <= width + 0.001 and
                      -0.001 <= box[1] <= box[3] <= height + 0.001):
                errors.append({"id": block["id"], "reason": "bbox outside page", "bbox": box})
            for child in block.get("children") or []:
                visit(child)
        for block in page.get("children") or []:
            visit(block)
    return {"all_bboxes_inside_pages": not errors, "unique_block_ids": len(ids) == len(set(ids)),
            "block_count": len(ids), "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--pdfium", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Preserve the earlier result; select an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"schema_version": 2, "kind": "independent-native-marker-text-comparison",
               "status": "started", "cases": [], "model_inference_tested": False}
    started = time.monotonic()
    try:
        gold_file = args.fixtures / "expected.json"
        reference_file = args.reference / "receipt.json"
        gold = json.loads(gold_file.read_text())
        original = json.loads(reference_file.read_text())
        environment = json.loads((args.reference / "environment.json").read_text())
        if original["status"] != "pass":
            raise ValueError("Original reference did not pass its declared controls")
        if sha256(gold_file) != original["fixture_gold_sha256"]:
            raise ValueError("Authored input gold changed after the original capture")
        if sha256(args.pdfium) != environment["pdfium_sha256"]:
            raise ValueError("Use the same PDFium binary as the captured original reference")
        source_files = ["src/marker.cpp", "src/marker_document.cpp", "src/marker_document.h",
                        "src/marker_backend.cpp", "src/marker_backend.h", "src/crispembed.h",
                        "python/crispembed/_marker.py", "CMakeLists.txt"]
        receipt.update({
            "fixture_gold_sha256": sha256(gold_file), "reference_receipt_sha256": sha256(reference_file),
            "reference_marker_revision": original["marker_revision"],
            "native_library_sha256": sha256(args.library),
            "pdfium_library_sha256": sha256(args.pdfium), "same_pdfium_binary": True,
            "pdfium_version": environment["pdfium_version"],
            "native_source_sha256": {name: sha256(args.repo / name) for name in source_files},
            "source_fingerprint_scope": "Workspace files observed at execution; binary hash is independent. Build correspondence requires the build receipt.",
            "comparison_script_sha256": sha256(Path(__file__)),
            "metric_helper_sha256": sha256(Path(__file__).with_name("marker_reference.py")),
            "configuration": {"mode": "text", "n_threads": 1, "dpi": 96,
                              "extract_images": False, "include_page_headers": False},
        })
        sys.path.insert(0, str((args.repo / "python").resolve()))
        from bs4 import BeautifulSoup
        from crispembed import CrispMarker
        os.environ["CRISPEMBED_MARKER"] = "1"
        original_cases = {c["id"]: c for c in original["cases"]}

        for case in gold["cases"]:
            start = time.monotonic()
            if sha256(args.fixtures / case["pdf"]) != case["pdf_sha256"]:
                raise ValueError("Fixture changed: " + case["id"])
            output = args.output / case["id"]
            output.mkdir()
            with CrispMarker(mode="text", n_threads=1, dpi=96, extract_images=False,
                             pdfium_path=args.pdfium.resolve(), lib_path=args.library.resolve(),
                             password=case.get("password", "")) as converter:
                raw_json = converter.convert(args.fixtures / case["pdf"], "json")
                html = converter.render("html")
                markdown = converter.render("markdown")
                chunks = converter.render("chunks")
                metadata = converter.metadata
                assets = converter.assets
            elapsed = time.monotonic() - start
            for name, content in [("document.json", raw_json), ("document.html", html),
                                  ("document.md", markdown), ("chunks.json", chunks)]:
                (output / name).write_text(content, encoding="utf-8")
            write_json(output / "metadata.json", metadata)
            data, chunks_data = json.loads(raw_json), json.loads(chunks)
            blocks = list(walk_blocks(data))
            types = Counter(b["block_type"] for b in blocks if not b.get("ignored"))
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)
            expected_lines = case.get("body_lines_by_page", case["text_lines_by_page"])
            expected = " ".join(line for page in expected_lines for line in page)
            # A declared bullet is represented by <li> structure in HTML rather
            # than a literal '-' text node. Keep the literal-input metrics too;
            # only the explicitly declared list-item markers are structural.
            list_gold = set(case.get("expected_list_items", []))
            def rendered_fragment(value):
                return re.sub(r"^\s*[-*•]\s+", "", value) if value in list_gold else value
            fragments = [rendered_fragment(line) for page in expected_lines for line in page]
            rendered_expected = " ".join(fragments)
            structural = {
                "page_count": len(data["children"]) == case["page_count"],
                "authored_order": ordered_fragments(fragments, text),
                "all_body_content": all(normalized(rendered_fragment(x)) in normalized(text) for x in case["must_contain"]),
                "rendered_text_exact": normalized(rendered_expected) == normalized(text),
                "native_provenance": metadata["schema"] == "crispembed.marker/1" and metadata["backend"] == "pdfium-text",
                "no_model_generation": metadata["generated_tokens"] == 0 and metadata["layout_generated_tokens"] == 0,
                "no_image_assets_requested": not assets,
                "chunks_have_blocks": bool(chunks_data.get("blocks")),
            }
            details = {}
            heading_gold = case.get("expected_headings", [case["expected_heading"]] if case.get("expected_heading") else [])
            if heading_gold:
                headings = [normalized(b["text"]) for b in blocks if b["block_type"] == "SectionHeader"]
                structural["authored_headings"] = all(normalized(x) in headings for x in heading_gold)
                details["heading_texts"] = headings
            if case.get("expected_list_items"):
                strip_bullet = lambda x: re.sub(r"^\s*[-*•]\s+", "", x)
                list_text = [normalized(b["text"]) for b in blocks if b["block_type"] == "ListItem"]
                structural["authored_list_items"] = all(normalized(strip_bullet(x)) in list_text for x in case["expected_list_items"])
                details["list_item_texts"] = list_text
            if case.get("expected_code"):
                code = [b["text"] for b in blocks if b["block_type"] == "Code"]
                structural["code_bytes_and_indentation"] = case["expected_code"] in code
                details["code_expected"] = case["expected_code"]
                details["code_actual"] = code
                details["code_source_lines"] = [b["lines"] for b in blocks if b["block_type"] == "Code"]
            if case.get("expected_table"):
                tables = [[[cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
                           for row in table.find_all("tr")] for table in soup.find_all("table")]
                structural["authored_table_grid"] = case["expected_table"] in tables
                details["table_grids"] = tables
            if case.get("expected_columns"):
                structural["columns_are_not_a_table"] = types.get("Table", 0) == 0
                structural["column_reading_order"] = ordered_fragments(
                    [x for column in case["expected_columns"] for x in column], text)
            if case.get("repeated_header"):
                structural["running_header_removed"] = case["repeated_header"] not in text
                structural["running_footers_removed"] = all(f"Page {i + 1} of 3" not in text for i in range(3))
            if case.get("html_forbidden_tags"):
                structural["literal_html_is_text"] = all(soup.find(tag) is None for tag in case["html_forbidden_tags"])
                structural["markdown_html_literals_escaped"] = all("<" + tag not in markdown for tag in case["html_forbidden_tags"])
            geometry = geometry_checks(data)
            structural["block_geometry"] = geometry["all_bboxes_inside_pages"] and geometry["unique_block_ids"]
            result = {
                "id": case["id"], "elapsed_seconds": elapsed, "block_types": dict(types),
                "owned_gold_profile": "body-only for declared running-header/footer fixture; all authored lines otherwise",
                "native_vs_owned_gold": text_metrics(expected, text),
                "rendered_gold_profile": "Same body gold; explicitly declared list bullet markers are represented by list structure. All other characters unchanged.",
                "native_vs_rendered_gold": text_metrics(rendered_expected, text),
                "structural_checks": structural, "geometry": geometry, "details": details,
                "metadata": metadata,
            }
            if original_cases[case["id"]]["status"] == "captured":
                original_html = (args.reference / case["id"] / "document.html").read_text()
                original_text = BeautifulSoup(original_html, "html.parser").get_text(" ", strip=True)
                result["native_vs_original_html_text"] = text_metrics(original_text, text)
                result["reference_note"] = (
                    "Original forced-text retains running marginalia; native intentionally removes repeated marginalia."
                    if case.get("repeated_header") else
                    "Original forced-text reverses heading/sentence order on this rotated page."
                    if case.get("page_rotation_degrees") else
                    "Text comparison only; native structure and Markdown serialization may differ."
                )
            else:
                result["native_vs_original_html_text"] = None
                result["reference_note"] = original_cases[case["id"]]["reason"]
            result["output_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir())}
            receipt["cases"].append(result)
        all_checks = [passed for c in receipt["cases"] for passed in c["structural_checks"].values()]
        receipt["summary"] = {
            "pdf_cases": len(receipt["cases"]),
            "checks_passed": sum(all_checks), "check_count": len(all_checks),
            "owned_gold_zero_cer_wer_cases": sum(c["native_vs_owned_gold"]["exact_after_whitespace"] for c in receipt["cases"]),
            "rendered_gold_zero_cer_wer_cases": sum(c["native_vs_rendered_gold"]["exact_after_whitespace"] for c in receipt["cases"]),
            "original_reference_zero_cer_wer_cases": sum(bool(c["native_vs_original_html_text"] and c["native_vs_original_html_text"]["exact_after_whitespace"]) for c in receipt["cases"]),
            "original_reference_cases": sum(c["native_vs_original_html_text"] is not None for c in receipt["cases"]),
        }
        receipt["status"] = "pass" if all(all_checks) else "fail"
    except Exception as exc:
        receipt["status"] = "error"
        message = str(exc)
        for path in [args.repo, args.library, args.pdfium, args.reference, args.fixtures, args.output]:
            message = message.replace(str(path.resolve()), "<" + path.name + ">")
        receipt["error"] = {"type": type(exc).__name__, "message": message}
    receipt["elapsed_seconds"] = time.monotonic() - started
    write_json(args.output / "receipt.json", receipt)
    (args.output / "marker_compare.executed.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status": receipt["status"], "summary": receipt.get("summary"),
                      "error": receipt.get("error"), "receipt": str(args.output / "receipt.json")}, indent=2))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
