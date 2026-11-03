#!/usr/bin/env python3
"""Strict Xiaomi OTSL tables and bounded external-layout/native-region assembly.

The six OTSL tags and span semantics follow the Apache-2.0 upstream blueprint
SeerRay-Lab/Xiaomi-OCR-0, b1a9bb4c07060c453785e9151715ccda6b96f0b1,
postprocess/otsl.py (SHA-256 db46f5a05d9742a7bcfd3931428bdd9e8de845892b66fa4cf09bd21f83cf2b14).
Unlike its repair mode, this adapter rejects ragged rows and contradictory spans.
Cell content is literal escaped text. No Markdown-table fallback or OCR inference
occurs at import. PP-DocLayoutV3 remains an external original-model dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OTSL_SOURCE = {
    "repo": "SeerRay-Lab/Xiaomi-OCR-0",
    "revision": "b1a9bb4c07060c453785e9151715ccda6b96f0b1",
    "path": "postprocess/otsl.py",
    "sha256": "db46f5a05d9742a7bcfd3931428bdd9e8de845892b66fa4cf09bd21f83cf2b14",
    "git_blob": "1acbe769397246f9d26ee21b1122e2136d240c1f",
    "license": "Apache-2.0",
}
TAG = re.compile(r"<(?:fcel|ecel|lcel|ucel|xcel|nl)>")
RESERVED = re.compile(r"</?(?:[A-Za-z]+cel|nl)(?=[\s>/]|$)")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
LABELS = {
    "abstract": "text", "algorithm": "code", "aside_text": "text", "chart": "figure",
    "content": "text", "display_formula": "equation", "doc_title": "heading",
    "figure_title": "caption", "footer": "page_furniture", "footer_image": "page_furniture",
    "footnote": "footnote", "formula_number": "text", "header": "page_furniture",
    "header_image": "page_furniture", "image": "figure", "inline_formula": "equation",
    "number": "page_furniture", "paragraph_title": "heading", "reference": "text",
    "reference_content": "text", "seal": "text", "table": "table", "text": "text",
    "vertical_text": "text", "vision_footnote": "footnote",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


layout_policy = load_module("xiaomi0_document_layout", ROOT / "tools/xiaomi0_layout_contract.py")
policy = layout_policy.policy


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def otsl_table(source):
    """Return strict rectangular cells and HTML; never repair missing ownership."""
    require(isinstance(source, str) and 0 < len(source.encode("utf-8")) <= 1024 * 1024,
            "OTSL must be nonempty and at most one MiB")
    for match in RESERVED.finditer(source):
        require(TAG.match(source, match.start()), "Unsupported or malformed OTSL tag")
    matches = list(TAG.finditer(source))
    require(matches and not source[:matches[0].start()].strip(), "Text before the first OTSL cell")
    rows, row = [], []
    for index, match in enumerate(matches):
        token = match.group()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        payload = source[match.end():end]
        if token == "<nl>":
            require(row and not payload.strip(), "Empty OTSL row or text after its terminator")
            rows.append(row)
            row = []
        else:
            require(token == "<fcel>" or not payload.strip(), "Only fcel may carry text")
            row.append((token, payload.strip() if token == "<fcel>" else ""))
            require(len(row) <= 256, "OTSL column budget exceeded")
        require(len(rows) <= 256, "OTSL row budget exceeded")
    if row:
        rows.append(row)  # Upstream accepts an omitted final row terminator.
    require(rows and len(rows) <= 256, "Empty OTSL table or row budget exceeded")
    width = len(rows[0])
    require(width and all(len(items) == width for items in rows), "Ragged OTSL rows")
    require(len(rows) * width <= 16384, "OTSL cell budget exceeded")
    owners, cells = [[None] * width for _ in rows], {}
    for r, items in enumerate(rows):
        for c, (token, text) in enumerate(items):
            if token in {"<fcel>", "<ecel>"}:
                owner = (r, c)
                cells[owner] = {"row": r, "column": c, "text": text, "positions": []}
            elif token == "<lcel>":
                require(c > 0 and owners[r][c - 1] is not None, "lcel has no left origin")
                owner = owners[r][c - 1]
                require(owner[0] == r, "lcel below an origin requires xcel")
            elif token == "<ucel>":
                require(r > 0 and owners[r - 1][c] is not None, "ucel has no upper origin")
                owner = owners[r - 1][c]
                require(owner[1] == c, "ucel right of an origin requires xcel")
            else:  # <xcel>
                require(r > 0 and c > 0 and owners[r - 1][c] is not None
                        and owners[r - 1][c] == owners[r][c - 1], "xcel has conflicting/missing origins")
                owner = owners[r - 1][c]
                require(owner[0] < r and owner[1] < c, "xcel is not inside a rectangular span")
            owners[r][c] = owner
            cells[owner]["positions"].append((r, c))
    ordered = []
    for cell in cells.values():
        positions = cell.pop("positions")
        rs = max(r for r, _ in positions) - cell["row"] + 1
        cs = max(c for _, c in positions) - cell["column"] + 1
        require(len(positions) == rs * cs and all(
            owners[r][c] == (cell["row"], cell["column"])
            for r in range(cell["row"], cell["row"] + rs)
            for c in range(cell["column"], cell["column"] + cs)), "Nonrectangular or overlapping OTSL span")
        cell.update(rowspan=rs, colspan=cs)
        ordered.append(cell)
    parts = ["<table>"]
    for r in range(len(rows)):
        parts.append("<tr>")
        for cell in ordered:
            if cell["row"] != r:
                continue
            attrs = "".join(f' {name}="{cell[name]}"' for name in ("rowspan", "colspan") if cell[name] > 1)
            parts.append("<td" + attrs + ">" + html.escape(cell["text"], quote=False) + "</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return {"html": "".join(parts), "rows": len(rows), "columns": width, "cells": ordered,
            "text": " ".join(cell["text"] for cell in ordered), "source": OTSL_SOURCE.copy()}


def otsl_to_html(source):
    return otsl_table(source)["html"]


def crop_plan(layout, protocol):
    """Freeze every detected region and task cap before any recognition call."""
    require(protocol.get("schema") == "xiaomi0-region-protocol/v1", "Wrong Xiaomi region protocol")
    require(protocol.get("layout_record_sha256") == digest(layout), "Layout record differs from frozen protocol")
    require(layout.get("schema") == "xiaomi0-original-layout/v1"
            and layout.get("native_layout_implemented") is False
            and layout.get("document_qualified") is False, "Original external layout receipt required")
    maximum = protocol.get("max_regions")
    require(type(maximum) is int and 1 <= maximum <= 16, "Initial region budget must be 1..16")
    requests = layout_policy.region_requests(layout["regions"], layout["detector"], maximum=maximum)
    inp = protocol.get("input", {})
    require(isinstance(protocol.get("page_id"), str) and protocol["page_id"], "Page ID required")
    for name in ("source_sha256", "rendered_sha256"):
        require(isinstance(inp.get(name), str) and SHA256.fullmatch(inp[name]), "Frozen input hash required")
    require(inp["rendered_sha256"] == layout["detector"].get("image_sha256"), "Layout uses another rendered page")
    width, height = inp.get("width"), inp.get("height")
    require(type(width) is int and type(height) is int and width > 0 and height > 0
            and width * height <= 64 * 1024**2, "Invalid/big rendered page dimensions")
    rendering = inp.get("rendering", {})
    require(all(isinstance(rendering.get(k), str) and rendering[k] for k in ("tool", "version", "settings")),
            "Exact rendering recipe required")
    caps = protocol.get("caps")
    require(isinstance(caps, dict) and set(caps) == {"text", "table", "formula"}
            and all(type(v) is int and 1 <= v <= 256 for v in caps.values()), "Initial task caps must be 1..256")
    budget = protocol.get("max_total_new_tokens")
    require(type(budget) is int and 1 <= budget <= 2048, "Initial total generation budget must be 1..2048")
    recognition = protocol.get("recognition", {})
    require(recognition.get("precision") in {"f32", "f16"} and recognition.get("build_mode") == "portable",
            "Exact F32/F16 portable recognition profile required")
    for key in ("weights_sha256", "runner_sha256"):
        require(isinstance(recognition.get(key), str) and SHA256.fullmatch(recognition[key]), "Recognition artifact pin required")
    require(isinstance(recognition.get("runtime_revision"), str) and REVISION.fullmatch(recognition["runtime_revision"]),
            "Immutable native runtime revision required")
    total = 0
    for item in requests:
        item["id"] = "region-%03d" % item["index"]
        item["label_normalized"] = LABELS[item["label"]]
        item["pixel_bbox"] = [int(item["bbox_2d"][0] / 1000 * width), int(item["bbox_2d"][1] / 1000 * height),
                              int(item["bbox_2d"][2] / 1000 * width), int(item["bbox_2d"][3] / 1000 * height)]
        if item["ocr_required"]:
            x1, y1, x2, y2 = item["pixel_bbox"]
            require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, "Degenerate original region crop")
            item["max_new_tokens"] = caps[item["task"]]
            total += item["max_new_tokens"]
        else:
            item["max_new_tokens"] = None
    require(0 < total <= budget, "No OCR calls or total generation budget exceeded")
    return requests


def generation_call(request, row):
    """Derive completion from actual native IDs, never from text or a cap alone."""
    require(isinstance(row, dict) and type(row.get("cap")) is int
            and row["cap"] == request["max_new_tokens"], "Native cap differs from frozen task")
    ids, confidences = row.get("token_ids"), row.get("confidences")
    require(isinstance(ids, list) and 0 < len(ids) <= row["cap"]
            and all(type(x) is int and 0 <= x < 248320 for x in ids), "Invalid native generation IDs")
    require(248046 not in ids[:-1], "Native generation continues after EOS")
    require(isinstance(confidences, list) and len(confidences) == len(ids)
            and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in confidences),
            "Native confidence coverage differs")
    for key in ("wall_seconds", "init_seconds"):
        require(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] >= 0,
                "Actual native timing required")
    require(isinstance(row.get("raw_text_utf8_hex"), str), "Actual native text bytes required")
    try:
        text = bytes.fromhex(row["raw_text_utf8_hex"]).decode("utf-8")
    except (ValueError, UnicodeError) as error:
        raise ValueError("Malformed actual native text bytes") from error
    stop = row.get("stop_kind")
    require(type(stop) is int and stop in {1, 2, 3}, "Native generation stop is invalid")
    if stop == 1:
        require(ids[-1] == 248046 and row.get("stop_id") == 248046, "EOS lacks native-ID evidence")
        status = "eos"
    else:
        require(ids[-1] != 248046 and row.get("stop_id") is None, "Non-EOS stop contradicts actual IDs")
        if stop == 2:
            require(len(ids) == row["cap"], "Cap stop has the wrong generated length")
        status = "cap" if stop == 2 else "error"
    call = {"id": request["id"], "status": status, "prompt": request["prompt"],
            "max_new_tokens": request["max_new_tokens"], "generated_ids": ids,
            "eos_token_ids": [248046], "forced_eos_token_ids": [],
            "native_stop_kind": stop, "native_stop_reason": {1: "eos", 2: "max_tokens", 3: "context_limit"}[stop]}
    return call, text


def assemble(layout, protocol, capture, protocol_sha256):
    plan = crop_plan(layout, protocol)
    require(isinstance(protocol_sha256, str) and SHA256.fullmatch(protocol_sha256), "Protocol file hash required")
    require(capture.get("schema") == "xiaomi0-native-regions/v1"
            and capture.get("protocol_sha256") == protocol_sha256, "Capture protocol differs")
    require(capture.get("recognition") == protocol["recognition"], "Capture recognition provenance differs")
    expected = [x for x in plan if x["ocr_required"]]
    observations = capture.get("observations")
    require(isinstance(observations, list) and [x.get("id") for x in observations] == [x["id"] for x in expected],
            "Missing, duplicate, extra, or reordered native region calls")
    observed = {item["id"]: item for item in observations}
    blocks, tables, formulas, generation, raw, ordered, markup = [], [], [], [], [], [], []
    for request in plan:
        text, rendered = "", ""
        if request["ocr_required"]:
            item = observed[request["id"]]
            require(type(item.get("index")) is int and item["index"] == request["index"],
                    "Native observation region index differs")
            call, text = generation_call(request, item["native"])
            generation.append(call)
            raw.append(text)
            if request["task"] == "table":
                parsed = otsl_table(text)
                tables.append(parsed["html"])
                text, rendered = parsed["text"], parsed["html"]
            elif request["task"] == "formula":
                formulas.append(text)  # Literal LaTeX; no stripping or normalization.
                rendered = "<p>" + html.escape(text, quote=False) + "</p>"
            else:
                tag = "h2" if request["label_normalized"] == "heading" else "p"
                rendered = "<" + tag + ">" + html.escape(text, quote=False) + "</" + tag + ">"
            ordered.append(text)
            markup.append(rendered)
        blocks.append({"label": request["label_normalized"], "bbox": request["bbox"], "text": text})
    source = policy.evidence()
    detector = source["layout"]
    components = [
        {"name": "recognizer", "repo": policy.MODEL, "revision": policy.REVISION,
         "weights_sha256": protocol["recognition"]["weights_sha256"],
         "processor_sha256": source["source_files"]["processor_config.json"]["sha256"],
         "tokenizer_sha256": source["source_files"]["tokenizer.json"]["sha256"]},
        {"name": "external-original-layout", "repo": detector["model"], "revision": detector["revision"],
         "weights_sha256": detector["checkpoint_files"]["model.safetensors"]["sha256"],
         "processor_sha256": detector["metadata_files"]["preprocessor_config.json"],
         "tokenizer_sha256": digest({}), "tokenizer_manifest": {}},
    ]
    elapsed = sum(x["native"]["wall_seconds"] + x["native"]["init_seconds"] for x in observations) * 1000
    return {"schema": "crispembed-ocr-document-v1", "page_id": protocol["page_id"],
            "protocol_sha256": protocol_sha256, "input": protocol["input"],
            "document": {"raw_text": "\n\n".join(raw), "text": "\n\n".join(ordered),
                         "html": "<article>" + "\n".join(markup) + "</article>",
                         "blocks": blocks, "tables": tables, "formulas": formulas},
            "generation": generation,
            "run": {"engine": "xiaomi-ocr-0", "precision": protocol["recognition"]["precision"],
                    "backend": "cpu:external-original-layout+native-ggml-recognition",
                    "task": "region-document", "build_mode": "portable",
                    "runtime_revision": protocol["recognition"]["runtime_revision"],
                    "components": components, "expected_calls": [x["id"] for x in expected]},
            "performance": {"elapsed_ms": elapsed, "peak_rss_bytes": None,
                            "span": "sum of native region context initialization and recognition; external layout/crops/assembly excluded"},
            "adapter": {"schema": "xiaomi0-region-assembly/v1", "otsl_source": OTSL_SOURCE.copy(),
                        "table_policy": "strict rectangular spans; literal escaped text; no repair",
                        "assembly_policy": "all original detector regions in order; no upstream page-assembly heuristics",
                        "regions": plan, "native_layout_implemented": False, "vllm_qualified": False},
            "full_document_qualified": False, "publication_qualified": False}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def verify_native_artifacts(model_path, runner_path, build_receipt, recognition):
    """Bind supplied local artifacts to the predeclared source/precision run."""
    require(policy.file_sha256(model_path) == recognition["weights_sha256"], "Native GGUF bytes differ")
    require(policy.file_sha256(runner_path) == recognition["runner_sha256"], "Native runner bytes differ")
    manifest = json.loads(model_path.with_suffix(model_path.suffix + ".manifest.json").read_text())
    require(manifest.get("source_model") == policy.MODEL and manifest.get("source_revision") == policy.REVISION
            and manifest.get("architecture") == "qwen35_ocr"
            and manifest.get("dtype") == recognition["precision"]
            and manifest.get("bytes") == model_path.stat().st_size
            and manifest.get("checkpoint_tensor_count") == 488 and manifest.get("export_tensor_count") == 473
            and manifest.get("main_schema_sha256") == policy.MAIN_SCHEMA_SHA256,
            "Exact Xiaomi export manifest required")
    policy.check_profile(manifest.get("profile"))
    require(build_receipt.get("schema") == "xiaomi0-native-build/v1"
            and build_receipt.get("runtime_revision") == recognition["runtime_revision"]
            and build_receipt.get("build_mode") == "portable"
            and build_receipt.get("ggml_native") is False and build_receipt.get("crispembed_native") is False
            and build_receipt.get("binaries", {}).get("test-xiaomi0-run", {}).get("sha256") == recognition["runner_sha256"],
            "Actual portable Xiaomi native build receipt required")


def recognize_regions(args):
    """Hosted wrapper around actual native diagnostics, retaining raw IDs/stops.

    It consumes a prior original-detector receipt; it does not implement native
    PP-DocLayoutV3. The ordinary C API is usable for OCR but lacks token/stop
    introspection, so this evidence path deliberately uses the diagnostic runner.
    """
    require(os.environ.get("GITHUB_ACTIONS") == "true", "Native region inference runs on hosted CI only")
    require(os.environ.get("CRISPEMBED_BUILD_MODE", "portable") == "portable", "Portable native build required")
    layout = json.loads(args.layout.read_text())
    protocol = json.loads(args.protocol.read_text())
    plan = crop_plan(layout, protocol)
    require(policy.file_sha256(args.image) == protocol["input"]["rendered_sha256"], "Rendered image bytes differ")
    require(not args.output.exists() and not args.scratch.exists(), "Fresh region output/scratch directories required")
    out, scratch = args.output.resolve(), args.scratch.resolve()
    require(out != scratch and out not in scratch.parents and scratch not in out.parents,
            "Region output and model diagnostics scratch must be disjoint")
    out.mkdir(parents=True)
    scratch.mkdir(parents=True)
    hosted = load_module("xiaomi0_document_hosted", ROOT / "tools/ci-heavy/xiaomi0_reference.py")
    state = {"schema": "xiaomi0-native-regions/v1", "protocol_sha256": policy.file_sha256(args.protocol),
             "recognition": protocol["recognition"], "observations": [],
             "layout_record_sha256": digest(layout), "native_layout_implemented": False,
             "document_qualified": False}
    started = time.monotonic()
    try:
        state["resources"] = hosted.resource_snapshot(out, scratch)
        # Supplied weights are already present; reserve native residency and
        # bounded per-region diagnostic files, without a second model download.
        require(state["resources"]["memory_bytes"].get("MemAvailable", 0) >= 9 * 2**30
                and state["resources"]["scratch_free_bytes"] >= 4 * 2**30
                and state["resources"]["output_free_bytes"] >= 256 * 1024**2, "Insufficient region-run resources")
        verify_native_artifacts(args.model, args.native_runner, json.loads(args.build_receipt.read_text()),
                                protocol["recognition"])
        write_json(out / "crop-plan.json", plan)
        write_json(out / "recognition.json", state)
        from PIL import Image
        with Image.open(args.image) as source:
            require(source.size == (protocol["input"]["width"], protocol["input"]["height"]), "Rendered dimensions differ")
            image = source.convert("RGB")
        for request in plan:
            if not request["ocr_required"]:
                continue
            stem = request["id"]
            crop = scratch / (stem + ".png")
            image.crop(tuple(request["pixel_bbox"])).save(crop)
            prompt = scratch / (stem + ".prompt.txt")
            prompt.write_text(request["prompt"])
            logits = scratch / (stem + ".logits.f32")
            command = [args.native_runner.resolve(), args.model.resolve(), crop, request["max_new_tokens"],
                       prompt, logits, 2]
            result = hosted.run(command, out / (stem + ".log"), env=hosted.job_env(scratch), timeout=900)
            require(result.returncode == 0, "Native region recognition failed: " + stem)
            rows = [json.loads(line.removeprefix("XIAOMI0_RESULT ")) for line in result.stdout.splitlines()
                    if line.startswith("XIAOMI0_RESULT ")]
            require(len(rows) == 1, "Missing or duplicate actual native region output")
            generation_call(request, rows[0])  # Validate before accepting the raw observation.
            state["observations"].append({"id": stem, "index": request["index"], "native": rows[0],
                                          "crop_sha256": policy.file_sha256(crop),
                                          "crop_rgb_sha256": hashlib.sha256(image.crop(tuple(request["pixel_bbox"])).tobytes()).hexdigest()})
            state["elapsed_seconds"] = time.monotonic() - started
            write_json(out / "recognition.json", state)
            # This is document/output evidence, not a numerical-parity receipt.
            # Per-layer vectors are not uploaded; only remove these owned files.
            logits.unlink()
            Path(str(logits) + ".stages").unlink()
        document = assemble(layout, protocol, state, state["protocol_sha256"])
        write_json(out / "document.json", document)
    except Exception as error:
        state["failure"] = str(error)
        state["elapsed_seconds"] = time.monotonic() - started
        write_json(out / "recognition.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    table = modes.add_parser("table", help="Strict model-free OTSL-to-HTML conversion")
    table.add_argument("--input", required=True, type=Path)
    table.add_argument("--output", required=True, type=Path)
    assembly = modes.add_parser("assemble", help="Assemble already captured actual region outputs")
    execution = modes.add_parser("recognize", help="Hosted native region calls using a prior original layout receipt")
    for command in (assembly, execution):
        command.add_argument("--layout", type=Path, required=True)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    assembly.add_argument("--capture", type=Path, required=True)
    for name in ("image", "model", "native-runner", "build-receipt", "scratch"):
        execution.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "table":
        args.output.write_text(otsl_to_html(args.input.read_text()) + "\n")
    elif args.mode == "assemble":
        result = assemble(json.loads(args.layout.read_text()), json.loads(args.protocol.read_text()),
                          json.loads(args.capture.read_text()), policy.file_sha256(args.protocol))
        write_json(args.output, result)
    else:
        recognize_regions(args)


if __name__ == "__main__":
    main()
