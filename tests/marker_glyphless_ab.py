#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay frozen searchable PDFs through three predeclared native font arms.

No OCR is run and no inputs are downloaded. The original negative quality
result remains separate; this driver retains every follow-up regression too.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

import marker_corpus_ab as common

ROOT = Path(__file__).resolve().parents[1]
GATE = "CRISPEMBED_MARKER_GLYPHLESS_TEXT"
ARMS = ("legacy", "font_only", "font_and_postprocess")
PAIRS = {"font_effect": ("legacy", "font_only"),
         "postprocessing_effect": ("font_only", "font_and_postprocess"),
         "total_effect": ("legacy", "font_and_postprocess")}


def renderer_equal(left, right, fmt):
    """Ignore exactly the declared per-run elapsed time, never all metadata."""
    if fmt not in ("json", "chunks"):
        return left == right
    a, b = json.loads(left), json.loads(right)
    for value in (a, b):
        value.get("metadata", {}).get("configuration", {}).pop("elapsed_seconds", None)
    return a == b


def line_records(document, include_monospace=False):
    """Count retained physical lines, including duplicates and ordered spans."""
    result = Counter()
    for page in document["children"]:
        for block in common.selected_blocks(page):
            for line in block.get("lines") or []:
                spans = []
                for source in line.get("spans") or []:
                    span = dict(source)
                    if not include_monospace:
                        span.pop("monospace", None)
                    spans.append(span)
                record = {"page": page["id"], "page_bbox": page["bbox"],
                          "line_bbox": line["bbox"], "spans": spans}
                result[json.dumps(record, sort_keys=True, ensure_ascii=False)] += 1
    return result


def style_warning_changes(before, after):
    removed, added = [], []
    for first, second in zip(before["children"], after["children"]):
        a, b = Counter(first.get("warnings") or []), Counter(second.get("warnings") or [])
        removed.extend((a - b).elements())
        added.extend((b - a).elements())
    # Match the reviewed native diagnostic exactly, including its positive count.
    permitted = lambda text: re.fullmatch(
        r"Glyphless OCR font style is unknown; artificial monospace styling ignored for [1-9][0-9]* characters", text)
    return {"removed": removed, "added": added,
            "passed": len(before["children"]) == len(after["children"]) and
                not removed and all(permitted(text) for text in added)}


def pair_comparison(row, left, right, protocol):
    pair = dict(row, arms={"off": row["arms"][left], "on": row["arms"][right]})
    compared = common.compare_page(pair, protocol)
    return {"nonregression": compared["nonregression"],
            "nonregression_passed": compared["nonregression_passed"],
            "absolute_quality": {left: compared["absolute_quality"]["off"],
                                 right: compared["absolute_quality"]["on"]},
            "absolute_quality_passed": {left: compared["absolute_quality_passed"]["off"],
                                        right: compared["absolute_quality_passed"]["on"]}}


def page_worker(args, protocol, page, pin):
    common.setup_environment(protocol)
    sys.path.insert(0, str(ROOT / "python"))
    from crispembed import CrispMarker
    scorer, layout = common.metric_modules(protocol)
    folder = args.out / page["id"]
    folder.mkdir(parents=True, exist_ok=False)
    pdf = args.inputs / pin["pdf"]
    row = {"id": page["id"], "source": page["source"], "language": page["lang"],
           "tags": page["tags"], "fresh_ocr": False, "pdf_sha256": common.sha(pdf),
           "image_dimensions": common.image_dimensions(page, pin["decoded_image_size"]),
           "arms": {}, "controls": {}}
    started = time.monotonic()
    try:
        if row["pdf_sha256"] != pin["sha256"] or pdf.stat().st_size != pin["bytes"]:
            raise ValueError("Retained searchable PDF differs from frozen follow-up pin")
        if common.sha(pdf.with_suffix(".txt")) != pin["ocr_text_sha256"]:
            raise ValueError("Retained producer OCR text differs from its frozen pin")
        producer_receipt = args.inputs / "receipt.json"
        if common.sha(producer_receipt) != protocol["inputs"]["producer_receipt_sha256"]:
            raise ValueError("Producer receipt changed between workers")
        original = next(item for item in json.loads(producer_receipt.read_text())["pages"] if item["id"] == page["id"])
        expected = "\n".join(block.get("text", "") for block in sorted(page["blocks"], key=lambda b: b.get("order", 0)))
        documents, outputs, chunk_sets = {}, {}, {}
        width, height = pin["decoded_image_size"]
        for arm in (*ARMS, "legacy_unset"):
            os.environ.update(protocol["arms"]["legacy" if arm == "legacy_unset" else arm])
            if arm == "legacy_unset":
                os.environ.pop(GATE, None)
            before = time.monotonic()
            with CrispMarker(mode="text", n_threads=1, dpi=144, include_page_headers=True,
                             extract_images=False, pdfium_path=args.pdfium, lib_path=str(args.lib)) as marker:
                marker.convert(pdf)
                converted = time.monotonic() - before
                rendered = {fmt: marker.render(fmt) for fmt in common.FORMATS}
                if arm != "legacy_unset":
                    before_save = time.monotonic()
                    marker.save_extraction(folder / (arm + ".cemk"))
                    saved = time.monotonic() - before_save
            outputs[arm] = rendered
            for fmt, text in rendered.items():
                (folder / (arm + "." + fmt)).write_text(text, encoding="utf-8")
            if arm == "legacy_unset":
                continue
            document = documents[arm] = json.loads(rendered["json"])
            chunk_sets[arm] = json.loads(rendered["chunks"])
            predicted = common.prediction_blocks(document, width, height)
            visible = common.VisibleText()
            visible.feed(rendered["html"])
            text = " ".join(visible.parts)
            p, g = " ".join(text.split()), " ".join(expected.split())
            valid_boxes = all(len(b["bbox"]) == 4 and all(math.isfinite(v) for v in b["bbox"]) and
                -1e-4 <= b["bbox"][0] <= b["bbox"][2] <= width + 1e-4 and
                -1e-4 <= b["bbox"][1] <= b["bbox"][3] <= height + 1e-4 for b in predicted)
            cfg = document["metadata"]["configuration"]
            row["arms"][arm] = {"text_metrics": scorer.score(text, expected),
                "raw_whitespace_metrics": {"cer": scorer.lev(p, g) / max(1, len(g)),
                    "wer": scorer.lev(p.split(), g.split()) / max(1, len(g.split()))},
                "layout_metrics": layout.score_page(page["blocks"], predicted),
                "prediction_blocks": predicted, "finite_in_bounds_geometry": bool(predicted) and valid_boxes,
                "processing_seconds": converted, "artifact_write_seconds": saved,
                "artifact_bytes": (folder / (arm + ".cemk")).stat().st_size,
                "native_leaf_types": dict(Counter(b["block_type"] for b in common.selected_blocks(document["children"][0]))),
                "features_observed": {"assembled_chunks": len(chunk_sets[arm].get("assembled_chunks", [])),
                    "normalized_blocks": int(cfg.get("normalized_blocks", "0")),
                    "continuations": sum(bool(n.get("continuation_target")) for n in common.walk(document)),
                    "resolved_references": int(cfg.get("resolved_footnote_references", "0"))},
                "output_sha256": {fmt: common.sha(folder / (arm + "." + fmt)) for fmt in common.FORMATS}}
        for fmt in common.FORMATS:
            prior = args.inputs / page["id"] / ("off." + fmt)
            if common.sha(prior) != original["arms"]["off"]["output_sha256"][fmt]:
                raise ValueError("Preserved legacy renderer differs from producer receipt")
            row["controls"]["legacy_preserved_" + fmt] = renderer_equal(prior.read_text(), outputs["legacy"][fmt], fmt)
            row["controls"]["unset_equals_false_" + fmt] = renderer_equal(outputs["legacy_unset"][fmt], outputs["legacy"][fmt], fmt)
        baseline_lines = line_records(documents["legacy"])
        row["retained_line_count"] = sum(baseline_lines.values())
        row["line_style_changes"] = {}
        for arm in ("font_only", "font_and_postprocess"):
            row["controls"][arm + "_line_source_geometry_exact"] = line_records(documents[arm]) == baseline_lines
            style_before, style_after = line_records(documents["legacy"], True), line_records(documents[arm], True)
            row["line_style_changes"][arm] = {"before_only": sum((style_before - style_after).values()),
                                             "after_only": sum((style_after - style_before).values())}
            row["controls"][arm + "_physical_page_geometry_exact"] = [
                (p["id"], p["bbox"]) for p in documents["legacy"]["children"]] == [
                (p["id"], p["bbox"]) for p in documents[arm]["children"]]
        row["style_warning_changes"] = style_warning_changes(documents["legacy"], documents["font_only"])
        row["controls"]["only_declared_style_warning_change"] = row["style_warning_changes"]["passed"]
        row["provenance"] = common.provenance_checks(documents["font_only"], documents["font_and_postprocess"],
                                                     chunk_sets["font_and_postprocess"])
        row["comparisons"] = {name: pair_comparison(row, *pair, protocol) for name, pair in PAIRS.items()}
        row["structural_passed"] = all(row["controls"].values()) and all(row["provenance"].values()) and all(
            value["finite_in_bounds_geometry"] for value in row["arms"].values())
        row["execution_passed"] = True
    except Exception as exc:
        row["execution_passed"] = False
        row["error"] = f"{type(exc).__name__}: {exc}".replace(str(args.out), "<OUTPUT>").replace(str(args.inputs), "<INPUTS>")
    row["elapsed_seconds"] = time.monotonic() - started
    row["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    common.write_json(folder / "receipt.json", row)
    return 0 if row["execution_passed"] else 1


def retrieval_worker(args, protocol):
    common.setup_environment(protocol)
    import numpy as np
    sys.path.insert(0, str(ROOT / "python"))
    from crispembed import CrispEmbed
    cfg = protocol["retrieval"]
    model = args.cache / cfg["filename"]
    if common.sha(model) != cfg["sha256"]:
        raise ValueError("Embedding model differs from frozen protocol")
    candidates, missing = {arm: [] for arm in ARMS}, []
    for query in cfg["queries"]:
        page = query["relevant_pages"][0]
        receipt = args.out / page / "receipt.json"
        if not receipt.is_file() or not json.loads(receipt.read_text()).get("execution_passed"):
            missing.append(page)
            continue
        for arm in ARMS:
            if arm == "font_and_postprocess":
                chunks = json.loads((args.out / page / (arm + ".chunks")).read_text())["assembled_chunks"]
            else:
                document = json.loads((args.out / page / (arm + ".json")).read_text())
                chunks = [{"id": b["id"], "text": b.get("text", "")} for b in common.selected_blocks(document["children"][0])]
            for chunk in chunks:
                windows = common.utf8_windows(chunk["text"], cfg["window_max_utf8_bytes"], cfg["window_overlap_characters"])
                if not common.windows_cover(chunk["text"], windows):
                    raise ValueError("Embedding windows lose or alter source text")
                for i, window in enumerate(windows):
                    if window["text"].strip():
                        candidates[arm].append(dict(window, id=f"{page}:{chunk['id']}:{i}", page=page, chunk_id=chunk["id"]))
    started = time.monotonic()
    encoder = CrispEmbed(str(model), n_threads=1, offline=True, auto_download=False, lib_path=str(args.lib))
    try:
        queries = np.asarray([encoder.encode(q["text"]) for q in cfg["queries"]], dtype=np.float32)
        vectors = {arm: np.asarray([encoder.encode(c["text"]) for c in candidates[arm]], dtype=np.float32).reshape((-1, 384))
                   for arm in ARMS}
    finally:
        del encoder
    for value in (queries, *vectors.values()):
        if not np.all(np.isfinite(value)) or (len(value) and not np.allclose(np.linalg.norm(value, axis=1), 1, atol=1e-4)):
            raise ValueError("Nonfinite, zero or unnormalized native embedding")
    np.savez_compressed(args.out / "retrieval-vectors.npz", queries=queries, **vectors)
    common.write_json(args.out / "retrieval-candidates.json", candidates)
    rows = []
    for i, query in enumerate(cfg["queries"]):
        row = dict(query, arms={})
        for arm in ARMS:
            page_scores = {}
            for candidate, score in zip(candidates[arm], queries[i] @ vectors[arm].T):
                page_scores[candidate["page"]] = max(page_scores.get(candidate["page"], -float("inf")), float(score))
            ranking = sorted(page_scores, key=lambda page: (-page_scores[page], page))
            ranks = [ranking.index(page) + 1 for page in query["relevant_pages"] if page in ranking]
            rank = min(ranks) if ranks else None
            row["arms"][arm] = {"rank": rank, "reciprocal_rank": 1 / rank if rank else 0.0,
                "recall_at_3": int(rank is not None and rank <= 3), "ranking": ranking, "page_scores": page_scores}
        row["nonregression"] = {name: row["arms"][right]["reciprocal_rank"] >= row["arms"][left]["reciprocal_rank"]
                                for name, (left, right) in PAIRS.items()}
        rows.append(row)
    metrics = {arm: {key: sum(row["arms"][arm][key] for row in rows) / len(rows)
                    for key in ("reciprocal_rank", "recall_at_3")} for arm in ARMS}
    result = {"kind": "native-minilm-glyphless-followup", "queries": rows, "metrics": metrics,
        "candidate_windows": {arm: len(candidates[arm]) for arm in ARMS}, "missing_pages": sorted(set(missing)),
        "complete": not missing, "full_text_window_coverage": True, "model_sha256": common.sha(model),
        "native_library_sha256": common.sha(args.lib), "elapsed_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "vectors_sha256": common.sha(args.out / "retrieval-vectors.npz"),
        "candidate_text_sha256": common.sha(args.out / "retrieval-candidates.json"),
        "nonregression_passed": {name: all(row["nonregression"][name] for row in rows) for name in PAIRS},
        "absolute_quality_passed": {arm: value["reciprocal_rank"] >= cfg["absolute_mrr_min"] and
            value["recall_at_3"] >= cfg["absolute_recall_at_3_min"] for arm, value in metrics.items()}}
    common.write_json(args.out / "retrieval.json", result)
    return 0


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "inputs", "cache", "lib", "pdfium", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--native-source", required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--repo", type=Path, default=ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--page-worker", help=argparse.SUPPRESS)
    parser.add_argument("--retrieval-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.native_source) or not re.fullmatch(r"[0-9a-f]{64}", args.native_sha256):
        raise ValueError("A complete native source commit and compiled SHA256 are required")
    ROOT = common.ROOT = args.repo.resolve(strict=True)
    for name in ("protocol", "inputs", "cache", "lib", "pdfium"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.out = args.out.resolve()
    protocol = json.loads(args.protocol.read_text())
    corpus_path = ROOT / protocol["corpus"]["path"]
    if common.sha(corpus_path) != protocol["corpus"]["sha256"]:
        raise ValueError("Frozen gold changed")
    if common.sha(ROOT / protocol["parent_protocol"]["path"]) != protocol["parent_protocol"]["sha256"]:
        raise ValueError("Original frozen protocol changed")
    if common.sha(common.__file__) != protocol["inputs"]["producer_driver_sha256"]:
        raise ValueError("Shared evaluation helper differs from the frozen producer")
    if common.sha(args.lib) != args.native_sha256 or common.sha(args.pdfium) != protocol["runtime"]["pdfium_library_sha256"]:
        raise ValueError("Candidate or PDFium library changed")
    corpus = json.loads(corpus_path.read_text())
    if args.page_worker:
        return page_worker(args, protocol, next(p for p in corpus["pages"] if p["id"] == args.page_worker),
                           next(p for p in protocol["inputs"]["pages"] if p["id"] == args.page_worker))
    if args.retrieval_worker:
        return retrieval_worker(args, protocol)
    if common.sha(args.inputs / "receipt.json") != protocol["inputs"]["producer_receipt_sha256"]:
        raise ValueError("Producer receipt differs from frozen input identity")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"kind": "glyphless-three-arm-frozen-pdf-replay", "fresh_ocr": False,
        "protocol_sha256": common.sha(args.protocol), "native_source_commit": args.native_source,
        "native_library_sha256": common.sha(args.lib), "pdfium_sha256": common.sha(args.pdfium),
        "script_sha256": common.sha(__file__), "shared_helper_sha256": common.sha(common.__file__),
        "producer_receipt_sha256": common.sha(args.inputs / "receipt.json"), "pages": [], "failed_or_missing_pages": []}
    common.write_json(args.out / "runtime-before-candidate.json", report)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--repo", str(ROOT),
               "--native-source", args.native_source, "--native-sha256", args.native_sha256]
    for name in ("protocol", "inputs", "cache", "lib", "pdfium", "out"):
        command += ["--" + name, str(getattr(args, name))]
    started = time.monotonic()
    for i, page in enumerate(corpus["pages"], 1):
        try:
            result = subprocess.run(command + ["--page-worker", page["id"]], capture_output=True, text=True, timeout=90)
            code, stdout, stderr = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            code, stdout, stderr = 124, "", "page replay timed out"
        folder = args.out / page["id"]
        folder.mkdir(exist_ok=True)
        (folder / "worker.stdout").write_text(stdout)
        (folder / "worker.stderr").write_text(stderr)
        path = folder / "receipt.json"
        row = json.loads(path.read_text()) if path.is_file() else {"id": page["id"], "error": "no page-worker receipt"}
        row["supervisor_exit_code"] = code
        row["execution_passed"] = row.get("execution_passed") is True and code == 0
        if not row["execution_passed"]:
            report["failed_or_missing_pages"].append({"id": page["id"], "reason": row.get("error", "worker failed")})
        report["pages"].append(row)
        common.write_json(args.out / "receipt.json", report)
        print(f"{i}/{len(corpus['pages'])} {page['id']} execution={row['execution_passed']}", flush=True)
    try:
        result = subprocess.run(command + ["--retrieval-worker"], capture_output=True, text=True, timeout=900)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        code, stdout, stderr = 124, "", "retrieval replay timed out"
    (args.out / "retrieval.stdout").write_text(stdout)
    (args.out / "retrieval.stderr").write_text(stderr)
    path = args.out / "retrieval.json"
    report["retrieval"] = json.loads(path.read_text()) if path.is_file() else {"complete": False, "error": "no retrieval receipt"}
    report["retrieval"]["supervisor_exit_code"] = code
    done = [row for row in report["pages"] if row["execution_passed"]]
    summary = {"expected_pages": len(corpus["pages"]), "executed_pages": len(done),
        "controls_passes": sum(all(row["controls"].values()) for row in done),
        "structural_passes": sum(row["structural_passed"] for row in done),
        "pages_without_spatial_gold": [row["id"] for row in done if not row["image_dimensions"]["spatial_qualification_available"]],
        "quality_nonregression_passes": {name: sum(row["comparisons"][name]["nonregression_passed"] for row in done) for name in PAIRS},
        "absolute_quality_passes": {arm: sum(next(value["absolute_quality_passed"][arm] for value in row["comparisons"].values()
            if arm in value["absolute_quality_passed"]) for row in done) for arm in ARMS},
        "elapsed_seconds": time.monotonic() - started, "arms": {}}
    for arm in ARMS:
        summary["arms"][arm] = {}
        for group in ("text_metrics", "raw_whitespace_metrics", "layout_metrics"):
            keys = done[0]["arms"][arm][group] if done else {}
            summary["arms"][arm][group] = {key: sum(values) / len(values) if values else None for key in keys
                for values in [[row["arms"][arm][group][key] for row in done if row["arms"][arm][group][key] is not None]]}
    report["summary"] = summary
    report["full_coverage"] = len(done) == len(corpus["pages"]) and not report["failed_or_missing_pages"]
    report["full_panel_spatial_qualification_available"] = report["full_coverage"] and not summary["pages_without_spatial_gold"]
    report["passed"] = (report["full_coverage"] and summary["structural_passes"] == len(done) and
        report["full_panel_spatial_qualification_available"] and
        all(value == len(done) for value in summary["quality_nonregression_passes"].values()) and
        summary["absolute_quality_passes"]["font_and_postprocess"] == len(done) and code == 0 and
        report["retrieval"].get("complete") is True and all(report["retrieval"].get("nonregression_passed", {}).values()) and
        report["retrieval"].get("absolute_quality_passed", {}).get("font_and_postprocess") is True)
    common.write_json(args.out / "receipt.json", report)
    print(json.dumps(summary, indent=2), flush=True)
    print("Full follow-up qualification:", report["passed"], flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
