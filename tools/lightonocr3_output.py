#!/usr/bin/env python3
"""Pure adapter for the pinned LightOnOCR-3 grounding grammar.

``parse_grounding(raw)`` returns blocks with label, normalized xyxy bbox, text
and continuation. It accepts upstream's inline markers and coordinate spacing.
Malformed/inverted/out-of-range boxes remain unlocated, never guessed or clipped.
Continuation blocks are preserved separately for document evaluation.
"""
from __future__ import annotations
import re

LABELS = frozenset({"text", "title", "section", "list", "header", "footer", "page_number",
                    "footnote", "caption", "formula", "code", "table", "image", "chart",
                    "header_image", "footer_image", "aside_text", "page_header", "page_footer", "page_footnote"})
# Same accepted label/spacing grammar as pinned lightonai/LightOnOCR grounding.py.
MARKER = re.compile(r"!\[([A-Za-z_][\w-]*?)(\+?)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]*")
MALFORMED = re.compile(r"!\[([A-Za-z_][\w-]*?)(\+?)\]\([^\n]*")


def parse_grounding(raw: str) -> dict:
    blocks, errors, unlocated = [], [], []
    pending = None
    body = []
    fence = None

    def finish():
        nonlocal body
        content = "\n".join(body).strip()
        if pending is None:
            if content:
                unlocated.append(content)
        else:
            blocks.append({**pending, "id": f"block-{len(blocks):04d}", "text": content})
        body = []

    for line_number, line in enumerate(raw.splitlines(), 1):
        fence_match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence_match:
            token = fence_match.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            body.append(line)
            continue
        if fence:
            body.append(line)
            continue
        matches = list(MARKER.finditer(line))
        if not matches:
            malformed = MALFORMED.search(line)
            if malformed and malformed.group(1) in LABELS:
                finish()
                pending = None
                errors.append({"line": line_number, "reason": "malformed_marker", "raw": line})
            body.append(line)
            continue
        start = 0
        for match in matches:
            body.append(line[start:match.start()])
            finish()
            pending = None
            coords = [int(v) for v in match.groups()[2:]]
            if any(c > 1000 for c in coords):
                reason = "out_of_range_bbox"
            elif coords[0] >= coords[2] or coords[1] >= coords[3]:
                reason = "empty_or_inverted_bbox"
            else:
                reason = None
            if reason:
                errors.append({"line": line_number, "reason": reason, "raw": match.group()})
                body.append(match.group())
            else:
                pending = {"label": match.group(1), "continuation": bool(match.group(2)),
                           "bbox": [c / 1000 for c in coords], "source_bbox": coords}
                if match.group(1) not in LABELS:
                    errors.append({"line": line_number, "reason": "unknown_label", "raw": match.group()})
            start = match.end()
        body.append(line[start:])
    finish()
    if fence:
        errors.append({"reason": "unclosed_code_fence"})
    return {"blocks": blocks, "unlocated_text": "\n\n".join(unlocated), "errors": errors}
