#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent, model-free review of saved native Marker inference receipts."""
import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import zipfile


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def normal(text):
    return " ".join(text.split())


def edits(a, b):
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        next_row = [i]
        for j, right in enumerate(b, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1,
                                row[j - 1] + (left != right)))
        row = next_row
    return row[-1]


class ReviewHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text = []
        self.tables = []
        self.current_table = None
        self.current_row = None
        self.current_cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            assert self.current_table is None, "Nested table requires a different scorer"
            self.current_table = []
        elif tag == "tr" and self.current_table is not None:
            assert self.current_row is None
            self.current_row = []
        elif tag in ("td", "th") and self.current_row is not None:
            assert self.current_cell is None
            attributes = dict(attrs)
            self.current_cell = {"tag": tag, "data": [],
                                 "rowspan": int(attributes.get("rowspan", "1")),
                                 "colspan": int(attributes.get("colspan", "1"))}

    def handle_data(self, data):
        self.text.append(data)
        if self.current_cell is not None:
            self.current_cell["data"].append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.current_cell is not None:
            cell = self.current_cell
            cell["text"] = normal("".join(cell.pop("data")))
            self.current_row.append(cell)
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            assert self.current_cell is None
            self.current_table.append(self.current_row)
            self.current_row = None
        elif tag == "table" and self.current_table is not None:
            assert self.current_row is None
            self.tables.append(self.current_table)
            self.current_table = None


def walk(node):
    yield node
    for child in node.get("children") or []:
        yield from walk(child)


def geometry_ok(node, page_bbox):
    bb = node.get("bbox")
    if not isinstance(bb, list) or len(bb) != 4 or not all(
            isinstance(x, (int, float)) and math.isfinite(x) for x in bb):
        return False
    x0, y0, x1, y1 = bb
    return (page_bbox[0] - 0.01 <= x0 <= x1 <= page_bbox[2] + 0.01
            and page_bbox[1] - 0.01 <= y0 <= y1 <= page_bbox[3] + 0.01
            and all(len(point) == 2 and all(math.isfinite(x) for x in point)
                    and x0 - 0.01 <= point[0] <= x1 + 0.01
                    and y0 - 0.01 <= point[1] <= y1 + 0.01
                    for point in node.get("polygon", [])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result_root = root / "tests/results/marker"
    authored = json.loads((root / "tests/fixtures/marker/expected.json").read_text())
    authored = {case["id"]: case for case in authored["cases"]}
    synthetic = json.loads((root / "tests/fixtures/teleocr/synthetic-text.json").read_text())
    gold_by_id = {
        "synthetic-image": (synthetic["gold_text"], None),
        "digital-auto": ("\n".join(authored["simpledigital"]["logical_order"]), None),
        "digital-force-ocr": ("\n".join(authored["simpledigital"]["logical_order"]), None),
        "table-image": ("\n".join(authored["table"]["logical_order"]), authored["table"]["expected_table"]),
    }
    archives = {}
    for path in (result_root / "format-review-before").glob("*.zip"):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                archives[(name, sha_bytes(archive.read(name)))] = str(path.relative_to(root))

    panels = []
    for declared_path in sorted(result_root.glob("*/declared-panel.json")):
        directory = declared_path.parent
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            continue
        declared = json.loads(declared_path.read_text())
        if declared.get("kind") != "native-marker-real-model-panel":
            continue
        report = json.loads(summary_path.read_text())
        model_manifest = root / "tests/results/teleocr/model-manifests" / (report["model_file"] + ".json")
        weight_manifest = json.loads(model_manifest.read_text())
        source_checks = {}
        for source, digest in declared["source_sha256"].items():
            current = sha(root / source)
            archive = archives.get((source, digest))
            if source == "tests/marker_native.py":
                saved_runner = directory / "marker_native.executed.py"
                if saved_runner.exists() and sha(saved_runner) == digest:
                    archive = str(saved_runner.relative_to(root))
            source_checks[source] = {"declared": digest, "current": current,
                                      "current_matches": current == digest,
                                      "preserved_snapshot": archive,
                                      "source_available": current == digest or archive is not None}
        panel_checks = {
            "summary_preserves_every_declared_field": all(report[k] == v for k, v in declared.items()),
            "model_identity_matches_declaration_and_converter_manifest": report["model_sha256"] ==
                report["expected_model_sha256"] == weight_manifest["output_sha256"],
            "declared_source_snapshots_available": all(c["source_available"] for c in source_checks.values()),
            "frozen_runner_source_identified": source_checks["tests/marker_native.py"]["source_available"],
            "caps_and_backend_unchanged": report["max_tokens"] == report["max_layout_tokens"] == 1024
                and report["threads"] == 2 and report["backend"] == "CPU",
            "no_surya_or_numerical_parity_claim": report["surya_parity"] is False
                and report["teleocr_strict_numerical_qualification"] is False,
            "one_result_per_declared_case": [x["id"] for x in report["cases"]] ==
                [x["id"] for x in report["results"]],
            "summary_agrees_with_progress": report["results"] == json.loads((directory / "progress.json").read_text()),
        }
        cases = []
        for declared_case, receipt in zip(report["cases"], report["results"]):
            name = receipt["id"]
            doc = json.loads((directory / (name + ".json")).read_text())
            chunks = json.loads((directory / (name + ".chunks")).read_text())
            metadata = json.loads((directory / (name + ".meta.json")).read_text())
            html = ReviewHTML()
            html.feed((directory / (name + ".html")).read_text())
            html.close()
            expected, expected_grid = gold_by_id[name]
            actual = normal(" ".join(html.text))
            expected = normal(expected)
            char_edits, word_edits = edits(expected, actual), edits(expected.split(), actual.split())
            pages = doc["children"]
            nodes = [node for node in walk(doc) if node.get("block_type") not in ("Document", "Page")]
            identifiers = [node["id"] for node in walk(doc) if "id" in node]
            token_sum = sum(node["generated_tokens"] for node in nodes)
            layout_sum = sum(page["layout_generated_tokens"] for page in pages)
            warnings = [warning for node in walk(doc) for warning in node.get("warnings", [])]
            block_types = dict(Counter(node["block_type"] for node in nodes))
            checks = {
                "input_hash_matches_declaration": sha(root / declared_case["input"]) == declared_case["sha256"],
                "declared_gold_matches_authored_fixture": (declared_case["gold_text"], declared_case["table"]) == gold_by_id[name],
                "independent_visible_text_matches_gold": actual == expected,
                "independent_metrics_match_runner": receipt["quality"]["CER"] == char_edits / max(1, len(expected))
                    and receipt["quality"]["WER"] == word_edits / max(1, len(expected.split())),
                "runner_text_matches_saved_html": actual == receipt["actual_text"],
                "all_metadata_copies_match": doc["metadata"] == chunks["metadata"] == metadata == receipt["metadata"],
                "unique_document_ids": len(set(identifiers)) == len(identifiers),
                "finite_in_page_geometry": all(geometry_ok(node, page["bbox"])
                    for page in pages for node in walk(page)),
                "all_block_confidences_present_finite_in_range": all("confidence" in node and
                    math.isfinite(node["confidence"]) and 0 <= node["confidence"] <= 1 for node in nodes),
                "all_cap_fields_present_and_false": all("token_cap_reached" in node and
                    node["token_cap_reached"] is False for node in nodes) and
                    all("layout_token_cap_reached" in page and page["layout_token_cap_reached"] is False for page in pages),
                "token_and_cap_metadata_consistent": token_sum == metadata["generated_tokens"] and
                    layout_sum == metadata["layout_generated_tokens"] and
                    metadata["capped_blocks"] == metadata["layout_capped_pages"] == 0,
                "warnings_count_matches_all_document_nodes": len(warnings) == metadata["warning_count"],
                "backend_identifies_teleocr": metadata["backend"] == "crispembed-teleocr",
                "all_outputs_valid_utf8_without_literal_nul": all(b"\x00" not in (directory / (name + "." + suffix)).read_bytes()
                    and (directory / (name + "." + suffix)).read_bytes().decode("utf-8") is not None
                    for suffix in ("json", "html", "markdown", "chunks")),
                "chunk_ids_match_document_blocks": [node["id"] for node in chunks["blocks"]] == [node["id"] for node in nodes],
            }
            if name == "digital-auto":
                checks["auto_uses_embedded_pdf_text_without_recognition"] = token_sum == 0 and all(
                    node["source"].startswith("pdftext") for node in nodes)
            else:
                checks["native_recognition_executed"] = token_sum > 0 and all(
                    node["source"] == "teleocr-native" for node in nodes)
            if name == "digital-force-ocr":
                checks["force_ocr_mode_declared"] = report.get("mode") == "ocr"
            table = None
            if expected_grid is not None:
                grids = [[[cell["text"] for cell in row] for row in tab] for tab in html.tables]
                table = {"expected": expected_grid, "observed": grids,
                         "row_lengths": [[len(row) for row in tab] for tab in html.tables],
                         "cell_tags": [[cell["tag"] for row in tab for cell in row] for tab in html.tables],
                         "spans": [[[cell["rowspan"], cell["colspan"]] for row in tab for cell in row] for tab in html.tables]}
                checks["table_rows_columns_and_order_match_gold"] = grids == [expected_grid]
                checks["table_has_only_unit_spans"] = all(cell["rowspan"] == cell["colspan"] == 1
                    for tab in html.tables for row in tab for cell in row)
                table_nodes = [node for node in nodes if node["block_type"] == "Table"]
                table_html = ReviewHTML()
                for node in table_nodes:
                    table_html.feed(node["html"])
                checks["table_document_and_block_html_agree"] = table_html.tables == html.tables
            cases.append({"id": name, "checks": checks, "review_passed": all(checks.values()),
                          "input_sha256": declared_case["sha256"],
                          "visible_text": actual, "character_edits": char_edits, "reference_characters": len(expected),
                          "word_edits": word_edits, "reference_words": len(expected.split()),
                          "CER": char_edits / max(1, len(expected)), "WER": word_edits / max(1, len(expected.split())),
                          "block_types": block_types, "block_sources": [node["source"] for node in nodes],
                          "raw_layout": [page["raw_layout"] for page in pages],
                          "raw_block_text": [node["raw"] for node in nodes],
                          "block_bboxes": [node["bbox"] for node in nodes],
                          "confidences": [node["confidence"] for node in nodes],
                          "recognition_tokens": token_sum, "layout_tokens": layout_sum,
                          "warnings": warnings, "table": table,
                          "software_version": metadata["configuration"]["software_version"],
                          "seconds_reported": receipt["seconds"]})
        panels.append({"path": str(directory.relative_to(root)), "model_file": report["model_file"],
                       "model_sha256": report["model_sha256"], "library_sha256": report["library_sha256"],
                       "pdfium_sha256_recorded": report.get("pdfium_sha256"),
                       "model_manifest": str(model_manifest.relative_to(root)),
                       "model_manifest_sha256": sha(model_manifest),
                       "checks": panel_checks, "source_checks": source_checks, "cases": cases,
                       "review_passed": all(panel_checks.values()) and all(case["review_passed"] for case in cases),
                       "seconds_reported": report["seconds"], "peak_rss_kib_reported": report["peak_rss_kib"],
                       "output_sha256": {str(path.relative_to(root)): sha(path) for path in sorted(directory.iterdir()) if path.is_file()}})
    comparisons = []
    for quant in panels:
        if quant["model_file"] not in ("teleocr-q8_0.gguf", "teleocr-q4_k.gguf"):
            continue
        possible = [panel for panel in panels if panel["model_file"] == "teleocr-f16.gguf"
                    and panel["library_sha256"] == quant["library_sha256"]
                    and any(case["id"] == "synthetic-image" for case in panel["cases"])]
        if not possible:
            comparisons.append({"quantized_panel": quant["path"], "same_build_f16_available": False})
            continue
        reference = possible[-1]
        left = next(case for case in reference["cases"] if case["id"] == "synthetic-image")
        right = next(case for case in quant["cases"] if case["id"] == "synthetic-image")
        same_blocks = left["block_types"] == right["block_types"] and len(left["block_bboxes"]) == len(right["block_bboxes"])
        comparisons.append({"quantized_panel": quant["path"], "f16_panel": reference["path"],
                            "same_build_f16_available": True,
                            "same_input_sha256": left["input_sha256"] == right["input_sha256"],
                            "same_declared_source_hashes": {k: v["declared"] for k, v in reference["source_checks"].items()} ==
                                {k: v["declared"] for k, v in quant["source_checks"].items()},
                            "visible_text_exact": left["visible_text"] == right["visible_text"],
                            "raw_recognition_text_exact": left["raw_block_text"] == right["raw_block_text"],
                            "raw_layout_exact": left["raw_layout"] == right["raw_layout"],
                            "block_types_match": same_blocks,
                            "bbox_max_absolute_difference": max(abs(a-b)
                                for lb, rb in zip(left["block_bboxes"], right["block_bboxes"])
                                for a, b in zip(lb, rb)) if same_blocks else None,
                            "aggregate_block_confidence_max_absolute_difference": max(abs(a-b)
                                for a, b in zip(left["confidences"], right["confidences"])) if same_blocks else None,
                            "recognition_token_count_equal": left["recognition_tokens"] == right["recognition_tokens"],
                            "layout_token_count_equal": left["layout_tokens"] == right["layout_tokens"],
                            "token_ids_or_eos_compared": False,
                            "numerical_qualification": False})
    f16_repeats = []
    f16_synthetic = sorted([panel for panel in panels if panel["model_file"] == "teleocr-f16.gguf"
                     and any(case["id"] == "synthetic-image" for case in panel["cases"])],
                     key=lambda panel: (panel["path"] != "tests/results/marker/native-ocr-f16", panel["path"]))
    for earlier, later in zip(f16_synthetic, f16_synthetic[1:]):
        left = next(case for case in earlier["cases"] if case["id"] == "synthetic-image")
        right = next(case for case in later["cases"] if case["id"] == "synthetic-image")
        f16_repeats.append({"earlier_panel": earlier["path"], "later_panel": later["path"],
                            "same_library": earlier["library_sha256"] == later["library_sha256"],
                            "same_model": earlier["model_sha256"] == later["model_sha256"],
                            "same_input": left["input_sha256"] == right["input_sha256"],
                            "visible_text_exact": left["visible_text"] == right["visible_text"],
                            "raw_recognition_text_exact": left["raw_block_text"] == right["raw_block_text"],
                            "raw_layout_exact": left["raw_layout"] == right["raw_layout"],
                            "block_boxes_exact": left["block_bboxes"] == right["block_bboxes"],
                            "software_versions": [left["software_version"], right["software_version"]]})
    numerical_path = root / "tests/results/teleocr/final-matrix-peer-review.json"
    numerical = json.loads(numerical_path.read_text())
    final_build_path = result_root / "build/final-source.json"
    final_build = json.loads(final_build_path.read_text())
    final_library = final_build["binary_sha256"]["libcrispembed.so.0.17.14"]
    final_source_matches = {name: sha(root / name) == digest for name, digest in final_build["source_sha256"].items()}
    requirements = {
        "initial_full_f16": ("teleocr-f16.gguf", {"synthetic-image", "digital-auto", "table-image"}, False),
        "final_f16_synthetic": ("teleocr-f16.gguf", {"synthetic-image"}, True),
        "final_q8_synthetic": ("teleocr-q8_0.gguf", {"synthetic-image"}, True),
        "final_q4_synthetic": ("teleocr-q4_k.gguf", {"synthetic-image"}, True),
        "final_f16_force_pdf": ("teleocr-f16.gguf", {"digital-force-ocr"}, True),
    }
    present = {role: any(panel["model_file"] == model and {case["id"] for case in panel["cases"]} == case_ids
                         and (not final or panel["library_sha256"] == final_library) for panel in panels)
               for role, (model, case_ids, final) in requirements.items()}
    result = {"schema_version": 1, "kind": "independent-saved-native-marker-model-output-review",
              "review_date": "2026-10-31",
              "reviewer_source": "tests/results/marker/native-review.executed.py",
              "reviewer_source_sha256": sha(Path(__file__)),
              "reviewer_executed_models": False, "reviewer_rehashed_model_payloads": False,
              "identity_scope": "Runner's actual model hash is checked against its frozen declaration and converter manifest; source snapshots and every saved output are independently hashed.",
              "quality_protocol": "Independently parse saved HTML, collapse whitespace only, preserve case/punctuation/Unicode, compare against pre-existing authored fixture gold; table row/column sequence and unit spans checked separately.",
              "panels": panels, "same_build_synthetic_precision_comparisons": comparisons,
              "counts": {"saved_panels": len(panels),
                  "mode_build_precision_conversions": sum(len(panel["cases"]) for panel in panels),
                  "distinct_input_sha256": len({case["input_sha256"] for panel in panels for case in panel["cases"]}),
                  "distinct_model_sha256": len({panel["model_sha256"] for panel in panels}),
                  "distinct_native_library_sha256": len({panel["library_sha256"] for panel in panels}),
                  "independently_exact_visible_text_conversions": sum(case["character_edits"] == case["word_edits"] == 0
                      for panel in panels for case in panel["cases"]),
                  "independent_review_checks": sum(len(panel["checks"]) + sum(len(case["checks"]) for case in panel["cases"])
                      for panel in panels)},
              "same_model_synthetic_build_repeats": f16_repeats,
              "expected_review_roles_present": present,
              "review_complete": all(present.values()),
              "final_build_receipt": {"path": str(final_build_path.relative_to(root)), "sha256": sha(final_build_path),
                  "library_sha256": final_library, "current_source_matches": final_source_matches,
                  "all_current_sources_match": all(final_source_matches.values())},
              "inherited_numerical_review": {"path": str(numerical_path.relative_to(root)),
                  "sha256": sha(numerical_path), "scope": "Existing independent TeleOCR differential review; not re-executed by this document review.",
                  "thresholds": numerical["thresholds_verified_unchanged"], "counts": numerical["counts"],
                  "qualification": False},
              "all_review_checks_passed": bool(panels) and all(panel["review_passed"] for panel in panels),
              "review_passed": bool(panels) and all(panel["review_passed"] for panel in panels)
                  and all(present.values()) and all(final_source_matches.values()),
              "limits": [
                  "This evaluates a small owned synthetic panel, not representative document accuracy.",
                  "Digital-auto uses learned layout and PDF embedded text, with zero recognition tokens; it is not a PDF OCR accuracy test.",
                  "Synthetic text lines are all classified SectionHeader; the table title is classified Caption. Exact text does not establish learned label accuracy.",
                  "Table cell values, row/column sequence and no-span topology pass on one simple 4-by-3 table; header semantics and merged-cell inference are not qualified.",
                  "Geometry checks establish finite in-page coordinates, not box-IoU or layout accuracy.",
                  "Finite aggregate block confidences are not probability calibration or per-token confidence parity.",
                  "No caps are reported; this harness does not export token IDs or terminal tokens, so EOS and generation-token parity are not independently established here.",
                  "Q8_0 and Q4_K are synthetic-image-only panels; full table/digital panels are F16 only.",
                  "Existing TeleOCR intermediate-activation numerical gates remain failed; these document outputs do not change that qualification.",
                  "Original Surya models were not executed and no original-model parity is claimed.",
                  "Reported wall times and Linux peak RSS are single-run observations, not controlled performance benchmarks.",
                  "The initial full F16 manifest did not hash the PDFium binary or Python binding; the later runner records both. Its original source and outputs are preserved without retroactively changing that manifest.",
              ]}
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"panels": len(panels), "cases": sum(len(panel["cases"]) for panel in panels),
                      "review_complete": result["review_complete"], "review_passed": result["review_passed"],
                      "all_review_checks_passed": result["all_review_checks_passed"],
                      "failures": {panel["path"]: {"panel": [name for name, passed in panel["checks"].items() if not passed],
                          "cases": {case["id"]: [name for name, passed in case["checks"].items() if not passed] for case in panel["cases"] if not case["review_passed"]}}
                          for panel in panels if not panel["review_passed"]}}, indent=2))


if __name__ == "__main__":
    main()
