#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Frozen licensed scans -> fresh Tesseract PDFs -> native Marker off/on -> retrieval.

The protocol declares inputs, metrics, thresholds and queries before execution.
Every missing input, timeout, negative quality result and retrieval miss remains
in the report. This lane does not execute or qualify TeleOCR or Surya models.
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FORMATS = ("json", "chunks", "markdown", "html")
TYPE_MAP = {"Text": "text", "TextInlineMath": "text", "SectionHeader": "title", "ListItem": "list",
            "ListGroup": "list", "PageHeader": "header", "PageFooter": "footer", "Footnote": "footnote",
            "Picture": "figure", "Figure": "figure", "Table": "table", "Equation": "text", "Code": "text"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def walk(node):
    yield node
    for child in node.get("children") or []:
        yield from walk(child)


def selected_blocks(page):
    def visit(block):
        if block.get("ignored"):
            return
        if block.get("children") and block["block_type"] not in ("Table", "Code", "Equation", "Figure", "Picture"):
            for child in block["children"]:
                yield from visit(child)
        else:
            yield block
    for block in page.get("children") or []:
        yield from visit(block)


def prediction_blocks(document, width, height):
    if len(document["children"]) != 1:
        raise ValueError("A frozen scan must produce exactly one PDF page")
    page = document["children"][0]
    dw, dh = page["bbox"][2:]
    if not (dw > 0 and dh > 0):
        raise ValueError("Invalid native page dimensions")
    result = []
    for order, block in enumerate(selected_blocks(page)):
        box = block["bbox"]
        result.append({"id": block["id"], "order": order,
                       "type": TYPE_MAP.get(block["block_type"], block["block_type"].lower()),
                       "text": block.get("text", ""), "html": block.get("html", ""),
                       "bbox": [box[0] * width / dw, box[1] * height / dh,
                                box[2] * width / dw, box[3] * height / dh]})
    return result


def image_dimensions(page, decoded_size):
    """Use pinned image bytes; never rescale a frozen spatial annotation."""
    stored = [page["image"]["width"], page["image"]["height"]]
    decoded = list(decoded_size)
    boxes = sum(block.get("bbox") is not None for block in page["blocks"])
    if decoded != stored and boxes:
        raise ValueError("Frozen image dimensions differ on a page with gold boxes")
    return {"stored": stored, "decoded": decoded, "match": stored == decoded,
            "gold_box_count": boxes, "gold_rescaled": False,
            "prediction_coordinate_space": "decoded image pixels",
            "spatial_qualification_available": boxes > 0,
            "layout_scoring_scope": "frozen boxes and text fallback" if boxes else
                "text fallback only; no gold boxes, so no spatial qualification"}


def ranges_valid(text, ranges, sources, assembled=False):
    encoded, cursor = text.encode("utf-8"), 0
    if encoded and not ranges:
        return False
    for span in ranges:
        begin, end = span["text_start"], span["text_end"]
        if begin != cursor or not (0 <= begin <= end <= len(encoded)):
            return False
        transform, source_id = span["transform"], span.get("source_id")
        if transform == "inserted" and assembled:
            if source_id is not None or span["source_start"] or span["source_end"] or encoded[begin:end] != b" ":
                return False
        else:
            if source_id not in sources:
                return False
            raw = sources[source_id]["text"].encode("utf-8")
            a, b = span["source_start"], span["source_end"]
            if not (0 <= a <= b <= len(raw)):
                return False
            if transform == "copy":
                if raw[a:b] != encoded[begin:end]:
                    return False
            elif transform == "whitespace":
                if any(byte not in b" \t\n\r" for byte in raw[a:b]) or encoded[begin:end] not in (b"", b" "):
                    return False
            elif transform == "dehyphenation" and assembled:
                if raw[a:b] != b"-" or begin != end:
                    return False
            else:
                return False
        cursor = end
    return cursor == len(encoded)


def provenance_checks(off, on, chunks):
    records = on.get("source_records", [])
    sources = {row["id"]: row for row in records}
    baseline = [(node.get("text"), node.get("bbox")) for node in walk(off) if node.get("text")]
    mapped = [node for node in walk(on) if node.get("source_ranges")]
    assembled = chunks.get("assembled_chunks", [])
    return {
        "unique_source_ids": bool(records) and len(sources) == len(records),
        "physical_source_identity": bool(records) and all(row["page"] == 0 for row in records),
        "original_text_geometry_retained": all((row["text"], row["bbox"]) in baseline for row in records),
        "block_source_ranges": bool(mapped) and all(ranges_valid(node["text"], node["source_ranges"], sources) for node in mapped),
        "assembled_source_ranges": bool(assembled) and all(ranges_valid(row["text"], row["source_ranges"], sources, True) for row in assembled),
        "assembled_physical_pages": bool(assembled) and all(row["pages"] == [0] for row in assembled),
        "source_catalog_matches_renderers": chunks.get("source_records") == records,
    }


def utf8_windows(text, maximum=240, overlap=32):
    """Bounded overlapping windows cover all input characters without truncation."""
    result, start = [], 0
    while start < len(text):
        end, used = start, 0
        while end < len(text) and used + len(text[end].encode("utf-8")) <= maximum:
            used += len(text[end].encode("utf-8"))
            end += 1
        if end == start:
            raise ValueError("Window budget cannot hold a Unicode scalar")
        result.append({"text": text[start:end], "start": len(text[:start].encode("utf-8")),
                       "end": len(text[:end].encode("utf-8"))})
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return result


def windows_cover(text, windows):
    encoded, covered = text.encode("utf-8"), 0
    for window in windows:
        if not (0 <= window["start"] <= covered < window["end"] <= len(encoded)):
            return False
        if window["text"].encode("utf-8") != encoded[window["start"]:window["end"]]:
            return False
        covered = window["end"]
    return covered == len(encoded)


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)


def metric_modules(protocol):
    cfg = protocol["scoring"]
    for kind in ("text", "layout"):
        if sha(ROOT / cfg[kind + "_scorer"]) != cfg[kind + "_scorer_sha256"]:
            raise ValueError("Frozen metric implementation changed")
    return (load_module("score", ROOT / cfg["text_scorer"]),
            load_module("marker_layout_score", ROOT / cfg["layout_scorer"]))


def setup_environment(protocol):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(CRISPEMBED_MARKER="1", CRISPEMBED_MARKER_ARTIFACTS="1", CRISPEMBED_OFFLINE="1",
                      CRISPEMBED_MARKER_INCREMENTAL="0", CRISPEMBED_TELEOCR="0",
                      CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE="0", CRISPEMBED_ACCEPT_CC_BY_NC_LICENSE="0")
    limit = protocol["resources"]["working_rss_budget_bytes"]
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def run_page(args, protocol, page):
    setup_environment(protocol)
    from PIL import Image
    sys.path.insert(0, str(ROOT / "python"))
    from crispembed import CrispMarker
    scorer, layout = metric_modules(protocol)
    out = args.out / page["id"]
    out.mkdir(parents=True, exist_ok=False)
    row = {"id": page["id"], "source": page["source"], "language": page["lang"], "tags": page["tags"],
           "fresh_ocr": True, "ocr_engine": "Tesseract", "native_ocr_engine": None,
           "image_expected_sha256": page["image"]["sha256"], "arms": {}}
    row["gold_annotation_provenance"] = {kind: {prefix: sum(str(block.get(kind, "")).startswith(prefix + ":")
        for block in page["blocks"]) for prefix in ("manual", "derived")} for kind in ("bbox_provenance", "type_provenance")}
    started = time.monotonic()
    try:
        image = args.cache / "images" / (page["id"] + ".image")
        if sha(image) != page["image"]["sha256"]:
            raise ValueError("Frozen image SHA differs")
        with Image.open(image) as decoded:
            width, height = decoded.size
        row["image_dimensions"] = image_dimensions(page, (width, height))
        if page["source"] == "cord-v2":
            lang = "eng"
        elif page["lang"] == "de":
            lang = "frk+deu" if "fraktur" in page["tags"] else "deu"
        else:
            lang = {"en": "eng", "fr": "fra"}[page["lang"]]
        row["ocr_language"] = lang
        row["traineddata_sha256"] = {name: sha(args.cache / "tessdata" / (name + ".traineddata")) for name in lang.split("+")}
        assets = json.loads(args.assets_report.read_text())
        if any(digest != assets["assets"][name + ".traineddata"]["sha256"]
               for name, digest in row["traineddata_sha256"].items()):
            raise ValueError("Pinned OCR weights changed after asset verification")
        prefix = out / "recognized"
        command = ["tesseract", str(image), str(prefix), "--tessdata-dir", str(args.cache / "tessdata"),
                   "-l", lang, "--oem", "1", "--psm", "3", "--dpi", "300",
                   "-c", "tessedit_create_pdf=1", "-c", "tessedit_create_txt=1", "-c", "tessedit_create_tsv=1"]
        before = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=protocol["ocr_provider"]["max_seconds_per_page"])
        row["ocr_seconds"] = time.monotonic() - before
        row["ocr_returncode"] = result.returncode
        (out / "ocr.stdout").write_text(result.stdout, encoding="utf-8")
        (out / "ocr.stderr").write_text(result.stderr, encoding="utf-8")
        if result.returncode or not prefix.with_suffix(".pdf").is_file():
            raise RuntimeError("Tesseract failed to produce the requested searchable PDF")
        expected = "\n".join(block.get("text", "") for block in sorted(page["blocks"], key=lambda b: b.get("order", 0)))
        row["ocr_text_metrics"] = scorer.score(prefix.with_suffix(".txt").read_text(encoding="utf-8"), expected)
        row["recognized_pdf_sha256"] = sha(prefix.with_suffix(".pdf"))
        documents, chunk_sets = {}, {}
        for arm, gates in protocol["arms"].items():
            os.environ.update(gates)
            before = time.monotonic()
            with CrispMarker(mode="text", n_threads=1, dpi=protocol["runtime"]["dpi"], include_page_headers=True,
                             extract_images=False, pdfium_path=args.pdfium, lib_path=str(args.lib)) as marker:
                marker.convert(prefix.with_suffix(".pdf"))
                converted = time.monotonic() - before
                rendered = {fmt: marker.render(fmt) for fmt in FORMATS}
                before_save = time.monotonic()
                marker.save_extraction(out / (arm + ".cemk"))
                saved = time.monotonic() - before_save
            for fmt, content in rendered.items():
                (out / (arm + "." + fmt)).write_text(content, encoding="utf-8")
            document = documents[arm] = json.loads(rendered["json"])
            chunk_sets[arm] = json.loads(rendered["chunks"])
            predicted = prediction_blocks(document, width, height)
            visible = VisibleText()
            visible.feed(rendered["html"])
            text = " ".join(visible.parts)
            p, g = " ".join(text.split()), " ".join(expected.split())
            raw = {"cer": scorer.lev(p, g) / max(1, len(g)),
                   "wer": scorer.lev(p.split(), g.split()) / max(1, len(g.split()))}
            boxes_ok = all(len(b["bbox"]) == 4 and all(math.isfinite(v) for v in b["bbox"]) and
                           -1e-4 <= b["bbox"][0] <= b["bbox"][2] <= width + 1e-4 and
                           -1e-4 <= b["bbox"][1] <= b["bbox"][3] <= height + 1e-4 for b in predicted)
            entry = {"text_metrics": scorer.score(text, expected), "raw_whitespace_metrics": raw,
                     "layout_metrics": layout.score_page(page["blocks"], predicted),
                     "finite_in_bounds_geometry": bool(predicted) and boxes_ok,
                     "processing_seconds": converted, "artifact_write_seconds": saved,
                     "artifact_bytes": (out / (arm + ".cemk")).stat().st_size,
                     "prediction_blocks": predicted, "output_sha256": {fmt: sha(out / (arm + "." + fmt)) for fmt in FORMATS}}
            row["arms"][arm] = entry
        row["provenance"] = provenance_checks(documents["off"], documents["on"], chunk_sets["on"])
        row["features_observed"] = {"assembled_chunks": len(chunk_sets["on"].get("assembled_chunks", [])),
                                    "continuations": sum(bool(n.get("continuation_target")) for n in walk(documents["on"])),
                                    "normalized_blocks": documents["on"]["metadata"]["configuration"].get("normalized_blocks", "0"),
                                    "resolved_references": documents["on"]["metadata"]["configuration"].get("resolved_footnote_references", "0")}
        row["execution_passed"] = True
    except Exception as exc:
        row["execution_passed"] = False
        row["error"] = f"{type(exc).__name__}: {exc}".replace(str(args.cache), "<CACHE>").replace(str(args.out), "<OUTPUT>")
    row["elapsed_seconds"] = time.monotonic() - started
    row["peak_self_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    row["peak_child_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    write_json(out / "receipt.json", row)
    return 0 if row["execution_passed"] else 1


def compare_page(row, protocol):
    off, on = row["arms"]["off"], row["arms"]["on"]
    checks = {key: on["text_metrics"][key] <= off["text_metrics"][key] for key in ("cer", "wer")}
    checks["word_recall"] = on["text_metrics"]["word_recall"] >= off["text_metrics"]["word_recall"]
    for key in ("block_f1", "type_accuracy", "table_content_f1", "reading_order_tau"):
        a, b = off["layout_metrics"][key], on["layout_metrics"][key]
        checks[key] = a is None or (b is not None and b >= a)
    floors = protocol["scoring"]["absolute_quality_per_page"]
    absolute = {}
    for arm, entry in row["arms"].items():
        t, layout = entry["text_metrics"], entry["layout_metrics"]
        quality = {"block_f1": layout["block_f1"] >= floors["block_f1_min"],
                   "type_accuracy": layout["type_accuracy"] is not None and layout["type_accuracy"] >= floors["type_accuracy_min"]}
        if row["source"] == "cord-v2":
            quality["partial_gold_word_recall"] = t["word_recall"] >= floors["partial_gold_word_recall_min"]
        else:
            quality.update(cer=t["cer"] <= floors["complete_gold_cer_max"], wer=t["wer"] <= floors["complete_gold_wer_max"])
        if layout["reading_order_tau"] is not None:
            quality["reading_order_tau"] = layout["reading_order_tau"] >= floors["matched_order_tau_min"]
        if layout["gold_tables"]:
            quality["table_content_f1"] = layout["table_content_f1"] >= floors["table_content_f1_min_when_table_gold_exists"]
        absolute[arm] = quality
    return {"nonregression": checks, "nonregression_passed": all(checks.values()),
            "absolute_quality": absolute, "absolute_quality_passed": {arm: all(v.values()) for arm, v in absolute.items()},
            "spatial_qualification_available": row["image_dimensions"]["spatial_qualification_available"],
            "structural_passed": all(row["provenance"].values()) and all(a["finite_in_bounds_geometry"] for a in row["arms"].values())}


def retrieval_worker(args, protocol):
    setup_environment(protocol)
    import numpy as np
    sys.path.insert(0, str(ROOT / "python"))
    from crispembed import CrispEmbed
    cfg = protocol["retrieval"]
    model = args.cache / cfg["filename"]
    if sha(model) != cfg["sha256"]:
        raise ValueError("Embedding weights differ from protocol")
    rows, candidates = [], {"off": [], "on": []}
    missing = []
    for query in cfg["queries"]:
        page = query["relevant_pages"][0]
        receipt = args.out / page / "receipt.json"
        if not receipt.is_file() or not json.loads(receipt.read_text()).get("execution_passed"):
            missing.append(page)
            continue
        for arm in candidates:
            document = json.loads((args.out / page / (arm + ".json")).read_text())
            if arm == "off":
                chunks = [{"id": b["id"], "text": b.get("text", "")} for b in selected_blocks(document["children"][0])]
            else:
                chunks = json.loads((args.out / page / "on.chunks").read_text())["assembled_chunks"]
            for chunk in chunks:
                windows = utf8_windows(chunk["text"], cfg["window_max_utf8_bytes"], cfg["window_overlap_characters"])
                if not windows_cover(chunk["text"], windows):
                    raise ValueError("Embedding windows lose or change source text")
                for index, window in enumerate(windows):
                    if window["text"].strip():
                        candidates[arm].append(dict(window, id=f"{page}:{chunk['id']}:{index}", page=page, chunk_id=chunk["id"]))
    started = time.monotonic()
    encoder = CrispEmbed(str(model), n_threads=1, offline=True, auto_download=False, lib_path=str(args.lib))
    try:
        query_vectors = []
        vectors = {arm: [] for arm in candidates}
        for query in cfg["queries"]:
            query_vectors.append(encoder.encode(query["text"]))
        for arm in candidates:
            for candidate in candidates[arm]:
                vectors[arm].append(encoder.encode(candidate["text"]))
    finally:
        # CrispEmbed owns its native context through __del__, not __enter__/exit.
        del encoder
    queries = np.asarray(query_vectors, dtype=np.float32)
    vectors = {arm: np.asarray(value, dtype=np.float32).reshape((-1, 384)) for arm, value in vectors.items()}
    for array in [queries, *vectors.values()]:
        if not np.all(np.isfinite(array)) or (len(array) and not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-4)):
            raise ValueError("Embedding vectors are nonfinite, zero or not normalized")
    np.savez_compressed(args.out / "retrieval-vectors.npz", queries=queries, off=vectors["off"], on=vectors["on"])
    write_json(args.out / "retrieval-candidates.json", candidates)
    for i, query in enumerate(cfg["queries"]):
        item = {"id": query["id"], "query": query["text"], "relevant_pages": query["relevant_pages"], "arms": {}}
        for arm in candidates:
            scores = queries[i] @ vectors[arm].T
            page_scores = {}
            for candidate, score in zip(candidates[arm], scores):
                page_scores[candidate["page"]] = max(page_scores.get(candidate["page"], -float("inf")), float(score))
            ranking = sorted(page_scores, key=lambda page: (-page_scores[page], page))
            ranks = [ranking.index(page) + 1 for page in query["relevant_pages"] if page in ranking]
            rank = min(ranks) if ranks else None
            item["arms"][arm] = {"rank": rank, "reciprocal_rank": 1 / rank if rank else 0.0,
                                  "recall_at_3": int(rank is not None and rank <= 3),
                                  "ranking": ranking, "page_scores": page_scores}
        item["nonregression"] = item["arms"]["on"]["reciprocal_rank"] >= item["arms"]["off"]["reciprocal_rank"]
        rows.append(item)
    metrics = {arm: {key: sum(row["arms"][arm][key] for row in rows) / len(rows)
                     for key in ("reciprocal_rank", "recall_at_3")} for arm in candidates}
    result = {"kind": "native-minilm-marker-gold-retrieval", "model_sha256": sha(model),
              "native_library_sha256": sha(args.lib), "queries": rows, "metrics": metrics,
              "candidate_windows": {arm: len(values) for arm, values in candidates.items()}, "missing_pages": sorted(set(missing)),
              "full_text_window_coverage": True,
              "elapsed_seconds": time.monotonic() - started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "vectors_sha256": sha(args.out / "retrieval-vectors.npz"),
              "candidate_text_sha256": sha(args.out / "retrieval-candidates.json"),
              "absolute_quality_passed": {arm: values["reciprocal_rank"] >= cfg["absolute_mrr_min"] and
                  values["recall_at_3"] >= cfg["absolute_recall_at_3_min"] for arm, values in metrics.items()},
              "nonregression_passed": all(row["nonregression"] for row in rows), "complete": not missing}
    write_json(args.out / "retrieval.json", result)
    return 0


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "cache", "assets_report", "lib", "pdfium", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--page-worker", help=argparse.SUPPRESS)
    parser.add_argument("--retrieval-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    ROOT = args.repo.resolve(strict=True)
    for name in ("protocol", "cache", "assets_report", "lib", "pdfium"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.out = args.out.resolve()
    protocol = json.loads(args.protocol.read_text())
    corpus = json.loads((ROOT / protocol["corpus"]["path"]).read_text())
    if sha(ROOT / protocol["corpus"]["path"]) != protocol["corpus"]["sha256"]:
        raise ValueError("Frozen corpus changed")
    if sha(args.lib) != protocol["runtime"]["native_library_sha256"] or sha(args.pdfium) != protocol["runtime"]["pdfium_library_sha256"]:
        raise ValueError("Runtime library differs from predeclared protocol")
    if args.page_worker:
        return run_page(args, protocol, next(page for page in corpus["pages"] if page["id"] == args.page_worker))
    if args.retrieval_worker:
        return retrieval_worker(args, protocol)
    args.out.mkdir(parents=True, exist_ok=False)
    assets = json.loads(args.assets_report.read_text())
    if assets["protocol_sha256"] != sha(args.protocol):
        raise ValueError("Asset receipt belongs to another protocol")
    if assets.get("total_pages") != len(corpus["pages"]):
        raise ValueError("Asset fetch has not completed")
    if not all(row["passed"] for row in assets["assets"].values()):
        raise ValueError("Required pinned model/support assets are unavailable")
    version = subprocess.check_output(["tesseract", "--version"], text=True).splitlines()[0]
    if version != "tesseract " + protocol["ocr_provider"]["version"]:
        raise ValueError("Tesseract version differs from predeclared protocol")
    report = {"kind": "marker-frozen-gold-off-on", "protocol_sha256": sha(args.protocol),
              "corpus_sha256": protocol["corpus"]["sha256"], "native_library_sha256": sha(args.lib),
              "native_source_commit": protocol["runtime"]["native_source_commit"], "script_sha256": sha(__file__),
              "pdfium_sha256": sha(args.pdfium), "asset_receipt_sha256": sha(args.assets_report),
              "tesseract_version": version, "fresh_ocr_provider": "Tesseract", "fresh_teleocr_inference": False,
              "tesseract_binary_sha256": sha(shutil.which("tesseract")),
              "address_space_limit_bytes_per_worker": protocol["resources"]["working_rss_budget_bytes"],
              "pages": [], "failed_or_missing_pages": []}
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--repo", str(ROOT)]
    for name in ("protocol", "cache", "assets_report", "lib", "pdfium", "out"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    started = time.monotonic()
    for index, page in enumerate(corpus["pages"], 1):
        if not assets["pages"].get(page["id"], {}).get("passed"):
            report["failed_or_missing_pages"].append({"id": page["id"], "reason": "frozen image unavailable"})
            continue
        try:
            result = subprocess.run(command + ["--page-worker", page["id"]], capture_output=True, text=True, timeout=180)
            code, stdout, stderr = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            code, stdout, stderr = 124, "", "page worker timed out"
        folder = args.out / page["id"]
        folder.mkdir(exist_ok=True)
        (folder / "worker.stdout").write_text(stdout)
        (folder / "worker.stderr").write_text(stderr)
        receipt = folder / "receipt.json"
        row = json.loads(receipt.read_text()) if receipt.is_file() else {"id": page["id"], "execution_passed": False, "error": "no worker report"}
        row["supervisor_exit_code"] = code
        row["execution_passed"] = row.get("execution_passed") is True and code == 0
        if row["execution_passed"]:
            row["comparison"] = compare_page(row, protocol)
        else:
            report["failed_or_missing_pages"].append({"id": page["id"], "reason": row.get("error", "worker failed")})
        report["pages"].append(row)
        write_json(args.out / "receipt.json", report)
        print(f"{index}/{len(corpus['pages'])} {page['id']} execution={row['execution_passed']}", flush=True)
    try:
        result = subprocess.run(command + ["--retrieval-worker"], capture_output=True, text=True, timeout=900)
        retrieval_code, retrieval_out, retrieval_err = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        retrieval_code, retrieval_out, retrieval_err = 124, "", "retrieval worker timed out"
    (args.out / "retrieval.stdout").write_text(retrieval_out)
    (args.out / "retrieval.stderr").write_text(retrieval_err)
    retrieval_file = args.out / "retrieval.json"
    report["retrieval"] = json.loads(retrieval_file.read_text()) if retrieval_file.is_file() else {"complete": False, "error": "no retrieval report"}
    report["retrieval"]["supervisor_exit_code"] = retrieval_code
    passed_rows = [row for row in report["pages"] if row["execution_passed"]]
    summary = {"expected_pages": len(corpus["pages"]), "executed_pages": len(passed_rows),
               "structural_passes": sum(row["comparison"]["structural_passed"] for row in passed_rows),
               "nonregression_passes": sum(row["comparison"]["nonregression_passed"] for row in passed_rows),
               "spatially_annotated_pages": sum(row["comparison"]["spatial_qualification_available"] for row in passed_rows),
               "pages_without_spatial_gold": [row["id"] for row in passed_rows
                   if not row["comparison"]["spatial_qualification_available"]],
               "absolute_quality_passes": {arm: sum(row["comparison"]["absolute_quality_passed"][arm] for row in passed_rows) for arm in ("off", "on")},
               "elapsed_seconds": time.monotonic() - started}
    for arm in ("off", "on"):
        summary[arm] = {}
        for group in ("text_metrics", "raw_whitespace_metrics", "layout_metrics"):
            keys = passed_rows[0]["arms"][arm][group] if passed_rows else {}
            summary[arm][group] = {key: sum(values) / len(values) if values else None for key in keys
                for values in [[row["arms"][arm][group][key] for row in passed_rows if row["arms"][arm][group][key] is not None]]}
    report["summary"] = summary
    report["summary"]["aggregation"] = "macro over executed pages; CORD CER/WER descriptive because its gold is partial"
    report["strata"] = {}
    for group in ("en", "de", "fr", "cord-v2"):
        selected = [row for row in passed_rows if (row["source"] if row["source"] == "cord-v2" else row["language"]) == group]
        report["strata"][group] = {"pages": len(selected), "arms": {arm: {
            key: sum(row["arms"][arm]["text_metrics"][key] for row in selected) / len(selected) if selected else None
            for key in ("cer", "cer_nospace", "wer", "word_recall")} for arm in ("off", "on")}}
    report["full_coverage"] = len(passed_rows) == len(corpus["pages"]) and not report["failed_or_missing_pages"]
    report["full_panel_nonregression"] = report["full_coverage"] and summary["nonregression_passes"] == len(passed_rows)
    report["full_panel_absolute_quality"] = report["full_coverage"] and summary["absolute_quality_passes"]["on"] == len(passed_rows)
    report["full_panel_spatial_qualification"] = (report["full_coverage"] and
        summary["spatially_annotated_pages"] == len(passed_rows))
    report["passed"] = (report["full_panel_nonregression"] and report["full_panel_absolute_quality"] and
        report["full_panel_spatial_qualification"] and
        summary["structural_passes"] == len(passed_rows) and retrieval_code == 0 and
        report["retrieval"].get("complete") is True and report["retrieval"].get("nonregression_passed") is True and
        report["retrieval"].get("absolute_quality_passed", {}).get("on") is True)
    write_json(args.out / "receipt.json", report)
    print(json.dumps(summary, indent=2), flush=True)
    print("Full declared qualification:", report["passed"], flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
