#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Document-pipeline and OTSL behavior adapted from teleocr-rs by its contributors,
# revision 957cdbabe80e86d652c5fcc938ed7fbcf1eb2bbe.
# See docs/teleocr-license.md and the accompanying Apache-2.0 license/notice.
"""Parse images/PDF pages with CrispEmbed's native TeleOCR GGML backend.

The Python layer handles page layout, crops and output formatting. All neural
inference uses the same C++ GGUF engine as the CLI, C API and server; no torch,
Transformers or executable Hugging Face model code is imported.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

LAYOUT_SIZE = 1036
BLOCK_TYPES = frozenset("text title table image code algorithm header footer page_number page_footnote "
                        "aside_text equation equation_block ref_text list phonetic table_caption image_caption "
                        "code_caption table_footnote image_footnote unknown seal char".split())
SKIP_EXTRACT = frozenset(("image", "list", "equation_block"))
PARATEXT = frozenset(("header", "footer", "page_number", "aside_text", "page_footnote", "unknown"))
LAYOUT_PROMPTS = {
    "detection": "\nAnalyze the image layout.",
    "segmentation": "\nMulti-point Layout Segmentation Analysis.",
}
TEXT_PROMPT = "\nPlease output the text content from the image."
PROMPTS = {
    "table": "\nThis is the image of a table. Please output the table in OTSL format.",
    "equation": "\nPlease write out the expression of the formula in the image using LaTeX format.",
    "code": "\nThe image contains a code snippet, please output the parsing result.",
    "seal": "\nSeal Recognition:",
    "char": "\nThis is a scientific figure. Please extract the table implied by this figure.",
}
_LAYOUT = re.compile(r"^<box:([\d\s]+)><label:(\w+)><([^>]+)>$")
_OTSL = re.compile(r"(<(?:fcel|ecel|lcel|ucel|xcel|nl)>)")


@dataclass
class Block:
    type: str
    bbox: list[float]
    polygon: Optional[list[float]] = None
    angle: Optional[int] = None
    raw: Optional[str] = None
    content: Optional[str] = None
    lang: Optional[str] = None
    generated_tokens: int = 0
    token_cap_reached: bool = False
    confidence: float = 0.0


def parse_layout(text: str, *, max_blocks: int = 512):
    """Parse normalized 0..1000 coordinates, preserving model reading order.

    Returns (blocks, warnings). Malformed/unknown/empty boxes are rejected;
    their diagnostics remain in the page JSON instead of being silently lost.
    """
    if max_blocks < 1:
        raise ValueError("max_blocks must be positive")
    blocks, warnings = [], []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        match = _LAYOUT.fullmatch(line)
        if not match:
            warnings.append(f"layout line {number}: invalid syntax")
            continue
        kind = match[2].lower()
        coords = [int(n) for n in match[1].split()]
        if kind not in BLOCK_TYPES:
            warnings.append(f"layout line {number}: unknown type {kind}")
            continue
        if len(coords) < 4 or len(coords) > 256 or len(coords) % 2 or any(n > 1000 for n in coords):
            warnings.append(f"layout line {number}: invalid coordinates")
            continue
        values = [n / 1000 for n in coords]
        box = [min(values[0::2]), min(values[1::2]), max(values[0::2]), max(values[1::2])]
        if box[0] >= box[2] or box[1] >= box[3]:
            warnings.append(f"layout line {number}: empty box")
            continue
        orientation = re.findall(r"\b(up|right|down|left)\b", match[3].lower())
        angle = {"up": 0, "right": 90, "down": 180, "left": 270}.get(orientation[0]) if orientation else None
        blocks.append(Block(kind, box, values if len(values) > 4 else None, angle))
        if len(blocks) > max_blocks:
            raise ValueError(f"layout exceeds the {max_blocks}-block limit")
    return blocks, warnings


def generation_params(kind: str):
    """Document-pipeline presets; these differ from model-card defaults."""
    frequency = 0.0 if kind in ("layout", "layout_seg") else 0.005 if kind in (
        "table", "char", "seal", "figure") else 0.05
    return {"repetition_penalty": 1.0, "presence_penalty": 1.0 if frequency else 0.0,
            "frequency_penalty": frequency, "no_repeat_ngram": 100}


def resize_by_need(image: Image.Image):
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("empty crop")
    if max(width, height) / min(width, height) > 50:
        new_width, new_height = (width, math.ceil(width / 50)) if width > height else (math.ceil(height / 50), height)
        canvas = Image.new("RGB", (new_width, new_height), "white")
        canvas.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
        image = canvas
    if min(image.size) < 28:
        scale = 28 / min(image.size)
        image = image.resize(tuple(math.ceil(n * scale) for n in image.size), Image.Resampling.BICUBIC)
    return image


def crop_block(page: Image.Image, block: Block):
    """Crop, apply a polygon mask at pixel centers, rotate upright, pad/upscale."""
    width, height = page.size
    x0, y0, x1, y1 = (int(block.bbox[0] * width), int(block.bbox[1] * height),
                      min(width, int(block.bbox[2] * width)), min(height, int(block.bbox[3] * height)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box is empty at the page's pixel resolution")
    crop = page.crop((x0, y0, x1, y1)).convert("RGB")
    if block.polygon:
        points = [(int(block.polygon[i] * width) - x0, int(block.polygon[i + 1] * height) - y0)
                  for i in range(0, len(block.polygon), 2)]
        pixels = np.asarray(crop).copy()
        xs = np.arange(crop.width, dtype=np.float64)[None, :] + 0.5
        # Chunk rows so a large crop does not require an unbounded float grid.
        for first in range(0, crop.height, 128):
            last = min(crop.height, first + 128)
            ys = np.arange(first, last, dtype=np.float64)[:, None] + 0.5
            inside = np.zeros((last - first, crop.width), dtype=bool)
            previous = points[-1]
            for point in points:
                xi, yi = point
                xj, yj = previous
                if yi != yj:
                    inside ^= ((yi > ys) != (yj > ys)) & (xs < (xj - xi) * (ys - yi) / (yj - yi) + xi)
                previous = point
            pixels[first:last][~inside] = 0
        crop = Image.fromarray(pixels)
    rotations = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}
    if block.angle in rotations:
        crop = crop.transpose(rotations[block.angle])
    return resize_by_need(crop)


class _TableHTML(HTMLParser):
    """Preserve table structure while escaping data and dropping unsafe attributes."""
    allowed = frozenset(("table", "thead", "tbody", "tfoot", "tr", "td", "th"))

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.allowed or len(self.stack) > 64:
            raise ValueError("unsupported HTML in table output")
        self.stack.append(tag)
        safe = []
        for key, value in attrs:
            if key in ("rowspan", "colspan") and value and value.isdecimal() and 1 <= int(value) <= 4096:
                safe.append(f' {key}="{int(value)}"')
        self.parts.append("<" + tag + "".join(safe) + ">")

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            raise ValueError("unbalanced HTML table")
        self.parts.append(f"</{tag}>")

    def handle_data(self, text):
        self.parts.append(html.escape(text))


def otsl_to_html(text: str):
    """Convert cells and horizontal/vertical spans to escaped HTML.

    Ragged rows are padded with empty cells. Orphan/overlapping continuation
    tokens are errors; the page parser retains their raw output with a warning.
    """
    text = text.strip()
    if text.startswith("<table"):
        parser = _TableHTML()
        parser.feed(text)
        parser.close()
        if parser.stack or not text.endswith("</table>"):
            raise ValueError("unbalanced HTML table")
        return "".join(parser.parts)
    rows, row = [], []
    parts = _OTSL.split(text)
    for index, part in enumerate(parts):
        if part == "<nl>":
            if row:
                rows.append(row)
                row = []
        elif part in ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>"):
            content = parts[index + 1].strip() if part == "<fcel>" and index + 1 < len(parts) else ""
            row.append((part, content))
    if row:
        rows.append(row)
    if not rows:
        if not text:
            return ""
        raise ValueError("table output contains no OTSL cells")
    width = max(map(len, rows))
    if width * len(rows) > 16384:
        raise ValueError("table exceeds the 16384-cell limit")
    rows = [r + [("<ecel>", "")] * (width - len(r)) for r in rows]
    owners = [[None] * width for _ in rows]
    cells = []
    for r, row in enumerate(rows):
        for c, (token, content) in enumerate(row):
            if token not in ("<fcel>", "<ecel>"):
                continue
            colspan = rowspan = 1
            while c + colspan < width and row[c + colspan][0] in ("<lcel>", "<xcel>"):
                colspan += 1
            while r + rowspan < len(rows) and rows[r + rowspan][c][0] in ("<ucel>", "<xcel>"):
                rowspan += 1
            cell_id = len(cells)
            cells.append((r, c, rowspan, colspan, content))
            for rr in range(r, r + rowspan):
                for cc in range(c, c + colspan):
                    if owners[rr][cc] is not None:
                        raise ValueError("overlapping OTSL cell spans")
                    owners[rr][cc] = cell_id
    if any(owner is None for row in owners for owner in row):
        raise ValueError("orphan OTSL continuation token")
    out = ["<table>"]
    for r, row in enumerate(owners):
        out.append("<tr>")
        for c, owner in enumerate(row):
            rr, cc, rowspan, colspan, content = cells[owner]
            if (r, c) != (rr, cc):
                continue
            attrs = (f' rowspan="{rowspan}"' if rowspan > 1 else "") + (f' colspan="{colspan}"' if colspan > 1 else "")
            out.append(f"<td{attrs}>{html.escape(content)}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def post_equation(text: str):
    text = re.sub(r"^(\$\$)?\s*\\[\[(]\s*", lambda m: m[1] or "", text)
    text = re.sub(r"\s*\\[\])]\s*(\$\$)?$", lambda m: m[1] or "", text)
    wrapped = "$$" + text.replace("$", "").strip() + "$$"
    match = re.fullmatch(r"\$\$(.*?)\s*(\([A-Za-z0-9][A-Za-z0-9.\-]*\))\s*\$\$", wrapped)
    if match and not match[1].rstrip().endswith("\\eqno"):
        return "$" + match[1].rstrip() + "$ " + match[2]
    return re.sub(r"(^|[^\\])tag\s*\{([^{}]*)\}(\s*\$\$\s*)$",
                  lambda m: m[1] + "\\tag{" + m[2] + "}" + m[3], wrapped)


def post_text(text: str):
    text = text.replace("$$", "$")
    positions = [m.start() for m in re.finditer(r"(?<!\\)\$", text)]
    if not positions or len(positions) % 2:
        return text
    out, last = [], 0
    for start, end in zip(positions[::2], positions[1::2]):
        out.append(text[last:start])
        if start and not text[start - 1].isspace():
            out.append(" ")
        out.append("$" + text[start + 1:end].strip() + "$")
        if end + 1 < len(text) and not text[end + 1].isspace():
            out.append(" ")
        last = end + 1
    out.append(text[last:])
    return "".join(out)


def split_code(text: str):
    lines = text.splitlines()
    lang = "txt"
    if lines and lines[0].startswith("```"):
        lang = lines.pop(0)[3:].strip() or lang
    if lines and lines[-1].strip() == "```":
        lines.pop()
    text = "\n".join(lines).strip()
    match = re.match(r"<_([^<>\n]+)_>", text)
    if match:
        lang, text = match[1].lower(), text[match.end():]
    # A language label must not be able to open a second Markdown construct.
    lang = re.sub(r"[^A-Za-z0-9_+#.\-]", "", lang)[:64] or "txt"
    return lang, text


def code_fence(text: str, lang: str = "txt"):
    runs = [len(m[0]) for m in re.finditer(r"`+", text)]
    fence = "`" * max(3, 1 + max(runs, default=0))
    return f"{fence}{lang}\n{text}\n{fence}"


def post_process(block: Block, warnings: list[str]):
    text = block.raw or ""
    if block.type == "equation":
        text = post_equation(text)
    elif block.type == "text":
        text = post_text(text)
    elif block.type == "table" or (block.type == "char" and ("<fcel>" in text or "<ecel>" in text)):
        try:
            text = otsl_to_html(text).replace("<table>", '<table border="1">', 1)
        except ValueError as exc:
            warnings.append(f"{block.type}: {exc}; retained raw output")
            text = code_fence(text, "text")
    elif block.type in ("code", "algorithm"):
        block.lang, text = split_code(text)
    elif block.type == "title":
        text = " ".join(text.split())
    block.content = text


def to_markdown(blocks: list[Block], paratext: bool = False):
    parts = []
    for block in blocks:
        if (not paratext and block.type in PARATEXT) or not block.content:
            continue
        content = block.content.strip()
        if not content or (block.type == "char" and len(content) <= 5):
            continue
        if block.type == "title":
            content = "# " + content
        elif block.type in ("code", "algorithm"):
            content = code_fence(content, block.lang or "txt")
        parts.append(content)
    return "\n\n".join(parts)


class TeleocrParser:
    def __init__(self, model, *, threads=2, lib_path=None, max_tokens=4096,
                 max_page_pixels=4_500_000, max_blocks=512):
        if max_tokens < 1 or max_blocks < 1 or max_page_pixels < LAYOUT_SIZE * LAYOUT_SIZE:
            raise ValueError("positive token/block limits and at least 1036*1036 page pixels are required")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
        from crispembed import CrispOcrModel
        self.ocr = CrispOcrModel(str(model), n_threads=threads, lib_path=lib_path)
        if self.ocr.get_generation_params() is None:
            self.close()
            raise ValueError("the model/native library does not expose TeleOCR generation controls")
        self.max_tokens, self.max_page_pixels, self.max_blocks = max_tokens, max_page_pixels, max_blocks

    def close(self):
        if getattr(self, "ocr", None) is not None:
            self.ocr.__del__()
            self.ocr = None

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def _generate(self, image, prompt, kind):
        if self.ocr is None:
            raise RuntimeError("parser is closed")
        if not self.ocr.set_generation_params(**generation_params(kind)) or not self.ocr.set_prompt(prompt):
            raise RuntimeError("native TeleOCR rejected the task settings")
        self.ocr.set_max_tokens(self.max_tokens)
        text = self.ocr.recognize(image)
        scores = self.ocr.confidences()
        if not len(scores):
            raise RuntimeError("native TeleOCR inference failed without token scores")
        return text, len(scores), float(scores.mean())

    def parse(self, page: Image.Image, *, mode="detection", paratext=False):
        if mode not in LAYOUT_PROMPTS:
            raise ValueError("mode must be detection or segmentation")
        page = page.convert("RGB")
        start = time.perf_counter()
        layout, layout_tokens, _ = self._generate(page.resize((LAYOUT_SIZE, LAYOUT_SIZE), Image.Resampling.BICUBIC),
                                                LAYOUT_PROMPTS[mode], "layout" if mode == "detection" else "layout_seg")
        layout_ms = (time.perf_counter() - start) * 1000
        blocks, warnings = parse_layout(layout, max_blocks=self.max_blocks)
        if not blocks:
            raise ValueError("TeleOCR returned no valid layout blocks: " + "; ".join(warnings[:5]))
        if layout_tokens >= self.max_tokens:
            warnings.append("layout reached the token cap; the page may be incomplete")
        source = page
        if page.width * page.height > self.max_page_pixels:
            scale = math.sqrt(self.max_page_pixels / (page.width * page.height))
            source = page.resize(tuple(max(1, math.floor(n * scale + 0.5)) for n in page.size), Image.Resampling.BICUBIC)
        start = time.perf_counter()
        total_tokens = layout_tokens
        for index, block in enumerate(blocks):
            if block.type in SKIP_EXTRACT:
                continue
            try:
                crop = crop_block(source, block)
            except ValueError as exc:
                warnings.append(f"block {index}: {exc}")
                continue
            block.raw, block.generated_tokens, block.confidence = self._generate(
                crop, PROMPTS.get(block.type, TEXT_PROMPT), block.type)
            total_tokens += block.generated_tokens
            block.token_cap_reached = block.generated_tokens >= self.max_tokens
            if block.token_cap_reached:
                warnings.append(f"block {index} ({block.type}) reached the token cap")
            post_process(block, warnings)
        return {"width": page.width, "height": page.height, "mode": mode, "layout_raw": layout,
                "blocks": [asdict(b) for b in blocks], "markdown": to_markdown(blocks, paratext),
                "warnings": warnings, "stats": {"layout_ms": layout_ms,
                "extract_ms": (time.perf_counter() - start) * 1000, "generated_tokens": total_tokens,
                "layout_token_cap_reached": layout_tokens >= self.max_tokens}}


def input_pages(path: Path, dpi: int):
    if path.suffix.lower() == ".pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError("PDF input needs the optional pypdfium2 package") from exc
        document = pdfium.PdfDocument(str(path))
        try:
            for index in range(len(document)):
                page = document[index]
                bitmap = page.render(scale=dpi / 72)
                try:
                    yield bitmap.to_pil().convert("RGB").copy()
                finally:
                    bitmap.close()
                    page.close()
        finally:
            document.close()
    else:
        with Image.open(path) as image:
            yield ImageOps.exif_transpose(image).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Markdown destination")
    parser.add_argument("--json", type=Path, help="Structured blocks, raw outputs and warnings")
    parser.add_argument("--mode", choices=tuple(LAYOUT_PROMPTS), default="detection")
    parser.add_argument("--paratext", action="store_true", help="Keep headers/footers/page numbers")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-blocks", type=int, default=512)
    parser.add_argument("--max-page-pixels", type=int, default=4_500_000)
    parser.add_argument("--pdf-dpi", type=int, default=200)
    args = parser.parse_args()
    if not 36 <= args.pdf_dpi <= 600:
        parser.error("--pdf-dpi must be between 36 and 600")
    protected = {args.input.resolve(), args.model.resolve()}
    if args.lib:
        protected.add(args.lib.resolve())
    if args.output.resolve() in protected or (args.json and args.json.resolve() in protected):
        parser.error("output paths must not overwrite the input, model or native library")
    if args.json and args.json.resolve() == args.output.resolve():
        parser.error("Markdown and JSON destinations must be different files")
    pages = []
    with TeleocrParser(args.model, threads=args.threads, lib_path=str(args.lib.resolve()) if args.lib else None,
                       max_tokens=args.max_tokens, max_page_pixels=args.max_page_pixels,
                       max_blocks=args.max_blocks) as engine:
        for number, image in enumerate(input_pages(args.input, args.pdf_dpi), 1):
            page = engine.parse(image, mode=args.mode, paratext=args.paratext)
            page["page"] = number
            pages.append(page)
            for warning in page["warnings"]:
                print(f"page {number}: {warning}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n---\n\n".join(p["markdown"] for p in pages) + "\n", encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"pages": pages}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"teleocr-parse: {exc}", file=sys.stderr)
        raise SystemExit(1)
