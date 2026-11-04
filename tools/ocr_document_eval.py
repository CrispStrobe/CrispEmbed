#!/usr/bin/env python3
"""Versioned OCR document records and diagnostics; never an upstream benchmark score.

The frozen tools/ocr_quality.py contract is intentionally unchanged. This format
adds explicit missing annotations, ordered boxes, non-unit table spans, formulas,
generation evidence and provenance for new, separately declared protocols.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from html.parser import HTMLParser
from pathlib import Path

SCHEMA = "crispembed-ocr-document-v1"
POLICY = "crispembed-ocr-document-diagnostics-v1"
LABELS = {"text", "heading", "page_furniture", "list", "caption", "footnote",
          "equation", "code", "table", "form", "figure"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def distance(a, b):
    """Levenshtein distance with linear storage; no Unicode/case normalization."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def text_errors(expected, actual):
    chars, words = distance(expected, actual), distance(expected.split(), actual.split())
    return {"cer": chars / max(1, len(expected)),
            "wer": words / max(1, len(expected.split())),
            "character_edits": chars, "word_edits": words,
            "expected_characters": len(expected), "actual_characters": len(actual),
            "exact": expected == actual}


class Table(HTMLParser):
    """Explicit HTML table grammar, preserving positions, headers and spans.

    No browser repair, nested tables, scripts, style or silent attribute removal.
    A malformed table remains an error in the report, not an empty/perfect table.
    """
    inline = {"b", "strong", "i", "em", "sub", "sup", "span", "p", "div"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.cells, self.occupied = [], [], set()
        self.row, self.column, self.cell = -1, 0, None
        self.started, self.finished = False, False

    def handle_starttag(self, tag, attrs):
        parent = self.stack[-1] if self.stack else None
        if tag == "table":
            require(not self.started and parent is None, "nested/multiple table")
            self.started = True
        elif tag in {"thead", "tbody", "tfoot"}:
            require(parent == "table", "misplaced table section")
        elif tag == "tr":
            require(parent in {"table", "thead", "tbody", "tfoot"}, "misplaced row")
            self.row += 1
            self.column = 0
        elif tag in {"td", "th"}:
            require(parent == "tr" and self.cell is None, "misplaced cell")
            spans = {"rowspan": 1, "colspan": 1}
            seen = set()
            for key, value in attrs:
                require(key in spans and key not in seen and isinstance(value, str)
                        and re.fullmatch(r"[1-9][0-9]{0,3}", value), "invalid/duplicate span")
                spans[key] = int(value)
                seen.add(key)
            while (self.row, self.column) in self.occupied:
                self.column += 1
            rs, cs = spans["rowspan"], spans["colspan"]
            require(rs * cs <= 10000 and len(self.occupied) + rs * cs <= 100000,
                    "table exceeds bounded grammar")
            slots = {(r, c) for r in range(self.row, self.row + rs)
                     for c in range(self.column, self.column + cs)}
            require(not slots & self.occupied, "overlapping spans")
            self.occupied.update(slots)
            self.cell = {"row": self.row, "column": self.column, "rowspan": rs,
                         "colspan": cs, "header": tag == "th", "text": ""}
            self.column += cs
        elif tag in self.inline or tag == "br":
            require(self.cell is not None, "formatting outside a cell")
        else:
            raise ValueError("unsupported table tag: " + tag)
        require(not attrs or tag in {"td", "th"}, "unsupported HTML attributes")
        if tag == "br":
            self.cell["text"] += "\n"
        else:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        require(tag == "br", "unsupported self-closing tag")
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        require(bool(self.stack) and self.stack[-1] == tag, "unbalanced table HTML")
        self.stack.pop()
        if tag in {"td", "th"}:
            self.cell["text"] = " ".join(self.cell["text"].split())
            self.cells.append(self.cell)
            self.cell = None
        elif tag in {"p", "div"}:
            self.cell["text"] += "\n"
        elif tag == "table":
            self.finished = True

    def handle_data(self, data):
        if self.cell is None:
            require(not data.strip(), "text outside table cells")
        else:
            self.cell["text"] += data

    def handle_comment(self, data):
        raise ValueError("comments outside table grammar")

    def handle_decl(self, decl):
        raise ValueError("declarations outside table grammar")

    def handle_pi(self, data):
        raise ValueError("processing instruction outside table grammar")

    def unknown_decl(self, data):
        raise ValueError("declaration outside table grammar")

    def parse(self, source):
        # HTMLParser tolerates junk in an end tag before invoking our callback.
        # Reject that repair explicitly so malformed markup cannot score exact.
        closing_tags = re.findall(r"</[^>]*>", source)
        require(all(re.fullmatch(r"</[A-Za-z][A-Za-z0-9]*\s*>", tag)
                    for tag in closing_tags), "malformed closing tag")
        require("</" not in re.sub(r"</[^>]*>", "", source), "unterminated closing tag")
        self.feed(source)
        self.close()
        require(self.finished and not self.stack and bool(self.cells), "incomplete/empty table")
        require(max(r for r, _ in self.occupied) == self.row, "rowspan past last row")
        columns = max(c for _, c in self.occupied) + 1
        require(len(self.occupied) == (self.row + 1) * columns, "ragged table/span holes")
        return {"rows": self.row + 1, "columns": columns, "cells": self.cells}


def validate_document(doc):
    require(isinstance(doc, dict), "document must be an object")
    require(isinstance(doc.get("raw_text"), str) and isinstance(doc.get("text"), str),
            "raw_text and ordered text must be strings")
    require("html" in doc and (doc["html"] is None or isinstance(doc["html"], str)),
            "html must be a string or explicit null")
    for key in ("tables", "formulas"):
        require(key in doc and (doc[key] is None or
                isinstance(doc[key], list) and all(isinstance(x, str) for x in doc[key])),
                key + " must be an ordered string list or explicit null")
    require("blocks" in doc and (doc["blocks"] is None or isinstance(doc["blocks"], list)),
            "blocks must be an ordered list or explicit null")
    for block in doc["blocks"] or []:
        require(isinstance(block, dict) and block.get("label") in LABELS
                and isinstance(block.get("text"), str), "invalid block label/text")
        box = block.get("bbox")
        require("bbox" in block and (box is None or isinstance(box, list) and len(box) == 4
                and all(finite(x) and 0 <= x <= 1 for x in box)
                and box[0] < box[2] and box[1] < box[3]
                and (box[2] - box[0]) * (box[3] - box[1]) > 0), "invalid normalized bbox")


def completion(calls):
    """EOS is established from IDs, separately for every declared generation call."""
    require(isinstance(calls, list) and bool(calls), "generation calls must be nonempty")
    seen, states = set(), []
    for call in calls:
        require(isinstance(call, dict), "generation call must be an object")
        name, status = call.get("id"), call.get("status")
        require(isinstance(name, str) and name and name not in seen, "duplicate/missing call id")
        seen.add(name)
        require(status in {"eos", "cap", "forced_eos", "error", "unknown"}, "invalid generation status")
        require(isinstance(call.get("prompt"), str), "exact per-call prompt required")
        cap, ids, ends = call.get("max_new_tokens"), call.get("generated_ids"), call.get("eos_token_ids")
        require(type(cap) is int and cap > 0, "positive generation cap required")
        require(ids is None or isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids),
                "invalid generated token IDs")
        require(isinstance(ends, list) and all(type(x) is int and x >= 0 for x in ends),
                "EOS token IDs required")
        forced = call.get("forced_eos_token_ids")
        require(isinstance(forced, list) and all(type(x) is int and x >= 0 for x in forced),
                "explicit forced-EOS configuration required")
        if ids is not None:
            require(len(ids) <= cap, "generation exceeds declared cap")
            require(not any(x in ends for x in ids[:-1]), "tokens after EOS")
        if status == "eos":
            require(bool(ids) and ids[-1] in ends, "EOS lacks generated-ID evidence")
            require(not (len(ids) == cap and ids[-1] in forced), "forced EOS at cap is incomplete")
        if status == "cap":
            require(ids is not None and len(ids) == cap and ids[-1] not in ends,
                    "cap lacks generated-ID evidence")
        if status == "forced_eos":
            require(ids is not None and len(ids) == cap and ids[-1] in forced,
                    "forced EOS lacks cap/config/ID evidence")
        states.append(status)
    return {"complete": all(x == "eos" for x in states), "calls": len(states),
            "eos": states.count("eos"), "capped": states.count("cap"),
            "forced_eos": states.count("forced_eos"),
            "errors": states.count("error"), "unknown": states.count("unknown")}


def validate_record(record, *, gold=False):
    require(isinstance(record, dict) and record.get("schema") == SCHEMA, "wrong document schema")
    require(isinstance(record.get("page_id"), str) and record["page_id"], "page_id required")
    require(isinstance(record.get("protocol_sha256"), str)
            and SHA256.fullmatch(record["protocol_sha256"]), "frozen protocol SHA-256 required")
    inp = record.get("input", {})
    require(isinstance(inp, dict), "input must be an object")
    for key in ("source_sha256", "rendered_sha256"):
        require(isinstance(inp.get(key), str) and SHA256.fullmatch(inp[key]), "input " + key + " required")
    require(all(type(inp.get(k)) is int and inp[k] > 0 for k in ("width", "height")),
            "rendered dimensions required")
    render = inp.get("rendering", {})
    require(isinstance(render, dict) and all(isinstance(render.get(k), str) and render[k]
                for k in ("tool", "version", "settings")), "exact rendering recipe required")
    validate_document(record.get("document"))
    if gold:
        annotation = record.get("annotation", {})
        require(isinstance(annotation, dict), "annotation must be an object")
        require(all(isinstance(annotation.get(k), str) and annotation[k]
                    for k in ("source", "license", "method")), "annotation provenance required")
        return
    run = record.get("run", {})
    require(isinstance(run, dict), "run provenance required")
    for key in ("engine", "precision", "backend", "task", "build_mode"):
        require(isinstance(run.get(key), str) and run[key], "run " + key + " required")
    require(isinstance(run.get("runtime_revision"), str)
            and REVISION.fullmatch(run["runtime_revision"]), "immutable runtime revision required")
    components = run.get("components")
    require(isinstance(components, list) and bool(components), "all model components required")
    names = set()
    for component in components:
        require(isinstance(component, dict), "component must be an object")
        require(isinstance(component.get("name"), str) and component["name"]
                and component["name"] not in names, "duplicate/missing component name")
        names.add(component["name"])
        require(isinstance(component.get("repo"), str) and component["repo"], "component repo required")
        require(isinstance(component.get("revision"), str) and REVISION.fullmatch(component["revision"]),
                "immutable checkpoint revision required")
        for key in ("weights_sha256", "processor_sha256", "tokenizer_sha256"):
            require(isinstance(component.get(key), str) and SHA256.fullmatch(component[key]),
                    "component " + key + " required (hash explicit empty manifest if inapplicable)")
    completion(record.get("generation"))
    expected_calls = run.get("expected_calls")
    require(isinstance(expected_calls, list) and bool(expected_calls)
            and all(isinstance(x, str) and x for x in expected_calls)
            and len(set(expected_calls)) == len(expected_calls), "expected generation call IDs required")
    require(set(expected_calls) == {call["id"] for call in record["generation"]},
            "generation call coverage differs from pipeline trace")
    perf = record.get("performance", {})
    require(isinstance(perf, dict), "performance must be an object")
    for key in ("elapsed_ms", "peak_rss_bytes"):
        require(key in perf and (perf[key] is None or finite(perf[key]) and perf[key] >= 0),
                key + " must be nonnegative or explicit null")
    require(isinstance(perf.get("span"), str) and perf["span"], "timing span required")


def aligned(expected, actual, compare):
    if expected is None or actual is None:
        return {"available": False, "gold_available": expected is not None,
                "prediction_available": actual is not None, "exact": None}
    rows = []
    for index in range(max(len(expected), len(actual))):
        if index >= len(expected) or index >= len(actual):
            rows.append({"index": index, "exact": False,
                         "missing": index >= len(actual), "extra": index >= len(expected)})
        else:
            rows.append({"index": index, **compare(expected[index], actual[index])})
    values = [row["exact"] for row in rows]
    exact = False if any(v is False for v in values) else None if any(v is None for v in values) else True
    return {"available": True, "expected": len(expected), "actual": len(actual),
            "exact": exact, "items": rows}


def table_compare(expected, actual):
    parsed = {}
    for name, source in (("gold", expected), ("prediction", actual)):
        try:
            parsed[name] = Table().parse(source)
        except (ValueError, AssertionError) as exc:
            parsed[name + "_error"] = str(exc)
    return {**parsed, "exact": "gold" in parsed and "prediction" in parsed
            and parsed["gold"] == parsed["prediction"]}


def box_iou(a, b):
    if a is None or b is None:
        return None
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def block_compare(expected, actual):
    iou = box_iou(expected["bbox"], actual["bbox"])
    content_exact = expected["label"] == actual["label"] and expected["text"] == actual["text"]
    return {"label_exact": expected["label"] == actual["label"],
            "text": text_errors(expected["text"], actual["text"]),
            "position_indexed_iou": iou,
            "bbox_exact": None if iou is None else expected["bbox"] == actual["bbox"],
            "exact": False if not content_exact else None if iou is None else expected["bbox"] == actual["bbox"]}


def score(gold, prediction):
    validate_record(gold, gold=True)
    validate_record(prediction)
    for key in ("page_id", "protocol_sha256", "input"):
        require(gold[key] == prediction[key], "gold/prediction mismatch: " + key)
    expected, actual = gold["document"], prediction["document"]
    result = {"policy": POLICY, "page_id": prediction["page_id"],
              "gold_sha256": digest(gold), "prediction_sha256": digest(prediction),
              "completion": completion(prediction["generation"]),
              "raw": text_errors(expected["raw_text"], actual["raw_text"]),
              "ordered_text": text_errors(expected["text"], actual["text"]),
              "blocks": aligned(expected["blocks"], actual["blocks"], block_compare),
              "tables": aligned(expected["tables"], actual["tables"], table_compare),
              "formulas": aligned(expected["formulas"], actual["formulas"], text_errors),
              "performance": prediction["performance"], "model_qualified": False,
              "upstream_benchmark_score": None}
    dimensions = [result[k]["exact"] for k in ("ordered_text", "blocks", "tables", "formulas")]
    result["exact_complete_document"] = (False if not result["completion"]["complete"]
        or any(value is False for value in dimensions) else
        None if any(value is None for value in dimensions) else True)
    return result


def export_opb(record):
    """Export only observed capabilities in OmniParseBench's html/blocks shape.

    This is input to the pinned official scorer, not a substitute for running it.
    Completion and provenance stay in the original record alongside this output.
    """
    validate_record(record)
    doc, out = record["document"], {}
    if doc["html"] is not None:
        out["html"] = doc["html"]
    if doc["blocks"] is not None:
        require(all(b["bbox"] is not None for b in doc["blocks"]), "OPB blocks need actual boxes")
        out["blocks"] = [{k: b[k] for k in ("label", "bbox", "text")} for b in doc["blocks"]]
    require(bool(out), "no observed html/blocks capability to export")
    return out


def load(path):
    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant,
                      object_pairs_hook=unique_pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    check = subs.add_parser("validate")
    check.add_argument("record", type=Path)
    check.add_argument("--gold", action="store_true")
    compare = subs.add_parser("score")
    compare.add_argument("gold", type=Path)
    compare.add_argument("prediction", type=Path)
    export = subs.add_parser("export-opb")
    export.add_argument("record", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            record = load(args.record)
            validate_record(record, gold=args.gold)
            result = {"valid": True, "sha256": digest(record), "model_qualified": False}
        elif args.command == "score":
            result = score(load(args.gold), load(args.prediction))
        else:
            result = export_opb(load(args.record))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if args.command == "score":
            return 0 if result["exact_complete_document"] is True else 1
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"valid": False, "error": str(exc), "model_qualified": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
