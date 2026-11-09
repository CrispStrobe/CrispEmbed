#!/usr/bin/env python3
"""Hosted representation experiment. Never a production compatibility gate.

Compare preserved native output and standards-valid scalar/cluster encodings.
Every ToUnicode entry describes its actual glyph. Visual painting order does
not change the supplied logical source. No reader result is substituted for
another, and every PDF, parser trace and failed control remains available.
"""

import argparse
from collections import Counter
import ctypes as C
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject,
                           FloatObject, NameObject, NumberObject)

import check_ocr_pdfa
import test_ocr_render_artifacts as fixtures


class Sequence(C.Structure):
    _fields_ = [("encoding", C.c_uint32), ("buffer", C.c_void_p), ("length", C.c_size_t)]


class Run(C.Structure):
    _fields_ = [("offset", C.c_size_t), ("length", C.c_size_t), ("level", C.c_uint8)]


def visual_cells(lib, text, height):
    signatures = {
        "SBAlgorithmCreate": ([C.POINTER(Sequence)], C.c_void_p),
        "SBAlgorithmCreateParagraph": ([C.c_void_p, C.c_size_t, C.c_size_t, C.c_uint8], C.c_void_p),
        "SBParagraphCreateLine": ([C.c_void_p, C.c_size_t, C.c_size_t], C.c_void_p),
        "SBLineGetRunCount": ([C.c_void_p], C.c_size_t),
        "SBLineGetRunsPtr": ([C.c_void_p], C.POINTER(Run)),
        "SBCodepointGetGeneralCategory": ([C.c_uint32], C.c_uint8),
        "SBLineRelease": ([C.c_void_p], None),
        "SBParagraphRelease": ([C.c_void_p], None),
        "SBAlgorithmRelease": ([C.c_void_p], None),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result
    rows = re.findall(r"[^\r\n\u2028\u2029]*(?:\r\n|[\r\n\u2028\u2029]|$)", text)
    rows = [row for row in rows if row]
    result, source_offset = [], 0
    for row_number, row in enumerate(rows):
        values = (C.c_uint32 * len(row))(*map(ord, row))
        sequence = Sequence(2, C.cast(values, C.c_void_p), len(row))
        algorithm = lib.SBAlgorithmCreate(C.byref(sequence))
        paragraph = lib.SBAlgorithmCreateParagraph(algorithm, 0, len(row), 0xfe)
        line = lib.SBParagraphCreateLine(paragraph, 0, len(row))
        if not algorithm or not paragraph or not line:
            raise RuntimeError("Could not resolve diagnostic visual positions")
        try:
            runs = lib.SBLineGetRunsPtr(line)
            positions = [(0.0, 0.0)] * len(row)
            cells = 0
            for index in range(lib.SBLineGetRunCount(line)):
                run = runs[index]
                clusters = []
                for source in range(run.offset, run.offset + run.length):
                    category = lib.SBCodepointGetGeneralCategory(ord(row[source]))
                    if category in (0x1a, 0x1b) or row[source] in "\r\n\u2028\u2029":
                        continue
                    if 6 <= category <= 8 and clusters:
                        clusters[-1].append(source)
                    else:
                        clusters.append([source])
                if run.level & 1:
                    clusters.reverse()
                for cell, cluster in enumerate(clusters):
                    for index, source in enumerate(cluster):
                        part = len(cluster) - index - 1 if run.level & 1 else index
                        positions[source] = (cells + cell + part / len(cluster),
                                             cells + cell + (part + 1) / len(cluster))
                cells += len(clusters)
            for index, (left, right) in enumerate(positions):
                result.append({"source": source_offset + index, "row": row_number,
                               "x": 10 + left * 300 / max(1, cells),
                               "width": (right - left) * 300 / max(1, cells),
                               "y": 210 - (row_number + 1) * height / len(rows),
                               "height": height / len(rows)})
            source_offset += len(row)
        finally:
            lib.SBLineRelease(line)
            lib.SBParagraphRelease(paragraph)
            lib.SBAlgorithmRelease(algorithm)
    assert source_offset == len(text)
    return result


def stream_object(writer, data, extra=None):
    stream = DecodedStreamObject()
    stream.set_data(data)
    if extra:
        stream.update(extra)
    return writer._add_object(stream)


def scalar_pdf(source, destination, text, positions, visual, actual, clusters, lib):
    writer = PdfWriter(clone_from=source)
    page = writer.pages[0]
    resources = page["/Resources"]
    original_font = next(iter(resources["/Font"].values())).get_object()
    original_descendant = original_font["/DescendantFonts"][0].get_object()
    original_descriptor = original_descendant["/FontDescriptor"].get_object()
    groups = []
    for index, position in enumerate(positions):
        category = lib.SBCodepointGetGeneralCategory(ord(text[index]))
        if clusters and groups and 6 <= category <= 8 and len(groups[-1]) < 32:
            groups[-1].append(index)
        else:
            groups.append([index])
    glyphs = []
    for group in groups:
        selected = [positions[i] for i in group]
        x = min(p["x"] for p in selected)
        right = max(p["x"] + p["width"] for p in selected)
        unit = dict(selected[0], x=x, width=right - x,
                    text="".join(text[i] for i in group))
        unit["metric"] = max(1 if unit["width"] else 0, round(unit["width"] / unit["height"] * 64))
        assert unit["metric"] <= 32767
        glyphs.append(unit)
    if visual:
        glyphs.sort(key=lambda item: (item["row"], item["x"], item["source"]))
    assert Counter("".join(g["text"] for g in glyphs)) == Counter(text)
    widths = sorted({g["metric"] for g in glyphs})
    names = [".notdef"] + [f"width{width}" for width in widths]
    program = TTFont(BytesIO(original_descriptor["/FontFile2"].get_data()))
    # Decode tables against their original glyph order before replacing it.
    # Lazy hmtx decoding otherwise indexes the newly shortened order.
    program.ensureDecompiled()
    notdef = program["glyf"][".notdef"]
    program.setGlyphOrder(names)
    program["glyf"].glyphs = {".notdef": notdef}
    program["glyf"].glyphOrder = names
    program["hmtx"].metrics = {".notdef": (64, 0)}
    for width in widths:
        pen = TTGlyphPen(None)
        pen.moveTo((0, 0)); pen.lineTo((width, 0))
        pen.lineTo((width, 64)); pen.lineTo((0, 64)); pen.closePath()
        name = f"width{width}"
        program["glyf"][name] = pen.glyph()
        program["hmtx"].metrics[name] = (width, 0)
    # CID fonts use the PDF CIDToGIDMap and may omit a font-internal cmap.
    if "cmap" in program:
        for table in program["cmap"].tables:
            table.cmap = {}
    program["head"].unitsPerEm = 64
    program["hhea"].numberOfHMetrics = len(names)
    program["maxp"].numGlyphs = len(names)
    encoded_font = BytesIO(); program.save(encoded_font); program.close()
    font_bytes = encoded_font.getvalue()
    descriptor = DictionaryObject(original_descriptor)
    descriptor[NameObject("/FontFile2")] = stream_object(writer, font_bytes,
        {NameObject("/Length1"): NumberObject(len(font_bytes))})
    descriptor[NameObject("/FontBBox")] = ArrayObject([NumberObject(0), NumberObject(0),
        FloatObject(max(widths) * 1000 / 64), NumberObject(1000)])
    descendant = DictionaryObject(original_descendant)
    descendant[NameObject("/FontDescriptor")] = writer._add_object(descriptor)
    cid_map = b"\0\0" + b"".join((widths.index(g["metric"]) + 1).to_bytes(2, "big") for g in glyphs)
    descendant[NameObject("/CIDToGIDMap")] = stream_object(writer, cid_map)
    descendant[NameObject("/W")] = ArrayObject([NumberObject(1), ArrayObject([
        FloatObject(g["metric"] * 1000 / 64) for g in glyphs])])
    mapping = ("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
               "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
               "/CMapName /CrispDiagnostic def\n/CMapType 2 def\n"
               "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n")
    for first in range(0, len(glyphs), 100):
        batch = glyphs[first:first + 100]
        mapping += f"{len(batch)} beginbfchar\n"
        for index, glyph in enumerate(batch, first + 1):
            mapping += f"<{index:04X}> <{glyph['text'].encode('utf-16-be').hex().upper()}>\n"
        mapping += "endbfchar\n"
    mapping += "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    font = DictionaryObject(original_font)
    font[NameObject("/DescendantFonts")] = ArrayObject([writer._add_object(descendant)])
    font[NameObject("/ToUnicode")] = stream_object(writer, mapping.encode())
    resources[NameObject("/Font")] = DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    # Keep the native image/page transform exactly, replacing only invisible text.
    content = b"q\n1 0 0 1 0 0 cm\nq\n320 0 0 240 0 0 cm\n/Im1 Do\nQ\n"
    if actual:
        value = "FEFF" + text.encode("utf-16-be").hex().upper()
        content += f"/Span << /ActualText <{value}> >> BDC\n".encode()
    content += b"BT\n3 Tr\n"
    previous_height = None
    for index, glyph in enumerate(glyphs, 1):
        # Stable visual runs must not flush a reader's RTL buffer with
        # redundant Tf operators. Keep logical-emission controls unchanged.
        if not visual or previous_height != glyph['height']:
            content += f"/F1 {glyph['height']:.12f} Tf\n".encode()
            previous_height = glyph['height']
        content += (f"1 0 0 1 {glyph['x']:.12f} {glyph['y']:.12f} Tm\n"
                    f"<{index:04X}> Tj\n").encode()
    content += b"ET\n" + (b"EMC\n" if actual else b"") + b"Q\n"
    page[NameObject("/Contents")] = stream_object(writer, content)
    with destination.open("wb") as output:
        writer.write(output)
    return [{k: v for k, v in g.items() if k != "metric"} for g in glyphs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--verapdf", required=True)
    options = parser.parse_args()
    fixtures.OPTIONS = SimpleNamespace(library=options.library, artifacts=str(options.artifacts))
    fixtures.ArtifactTests.setUpClass()
    case = fixtures.ArtifactTests()
    root = options.artifacts
    reports = root / "verapdf"; reports.mkdir()
    receipt = {"purpose": "Representation experiment, not production acceptance",
               "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "cases": []}
    controls = dict.fromkeys([0x061c, 0x200e, 0x200f, *range(0x202a, 0x202f), *range(0x2066, 0x206a)])
    normal = lambda text: " ".join(text.translate(controls).split())
    pixels = hashlib.sha256(Image.open(root / "rgb.png").convert("RGB").tobytes()).hexdigest()
    sources = [("arabic", "العربية"), ("hebrew", "שלום"), ("mixed-rtl", "العربية 123 ABC"),
               ("mixed-ltr", "Invoice 123: العربية / שלום (45) café e\u0301"),
               ("marks", "مَرْحَبًا שָׁלוֹם ١٢٣ / 42"), ("isolate", "Report \u2067العربية 123\u2069: ABC"),
               ("override", "\u202eABCD\u202c 123"), ("supplementary", "\U0001e900\U0001e922 123 ABC"),
               ("multiline", "العربية 123\nשלום ABC\r\nMixed العربية\u2028סיום\u2029Final ABC"),
               ("mapping-boundary", "א" * 180 + " " + "\U0001e900" * 100),
               ("combining-boundary", "מ" + "\u05b7" * 270 + " 123 ABC"),
               ("ltr", "Reviewed café 中文 😀 e\u0301")]
    variants = ("native-run", "run-fallback", "scalar-logical", "scalar-visual", "cluster-visual",
                "scalar-visual-actual", "scalar-logical-actual")
    for name, text in sources:
        height = 1 if name.endswith("boundary") else 80 if name == "multiline" else 30
        base = case.render("base-" + name + ".pdf", [{"image": root / "rgb.png",
                           "regions": [(text, 10, 30, 300, height)]}], pdfa=True, bidi=True)
        positions = visual_cells(case.lib, text, height)
        for variant in variants:
            path = root / (variant + "-" + name + ".pdf")
            record = {"case": name, "variant": variant, "source": text}
            receipt["cases"].append(record)
            try:
                if variant in ("native-run", "run-fallback"):
                    content = base.read_bytes()
                    path.write_bytes(content if variant == "native-run" else content.replace(b"/ActualText", b"/StoredText"))
                else:
                    record["painted_glyphs"] = scalar_pdf(base, path, text, positions,
                        "visual" in variant, variant.endswith("actual"), variant == "cluster-visual", case.lib)
                searches = [query for query in ("العربية", "שלום", "ABC", "\U0001e900\U0001e922", "😀") if query in text]
                record["readers"] = {engine: case.probe_pdf(path, engine, searches
                    if engine in ("PyMuPDF", "PDFium") else ()) for engine in ("PyMuPDF", "Poppler", "pypdf", "PDFium")}
                record["logical_matches"] = {engine: probe.get("exit_code") == 0 and
                    normal(probe.get("extracted", "")) == normal(text) for engine, probe in record["readers"].items()}
                record["pixels_exact"] = record["readers"]["PyMuPDF"].get("pixel_sha256") == pixels
                record["search_matches"] = {engine: all(record["readers"][engine].get("search", {}).get(query)
                    for query in searches) for engine in ("PyMuPDF", "PDFium")}
                record["verapdf"] = check_ocr_pdfa.validate(options.verapdf, path, reports / path.stem)
                record["all_checks_passed"] = (record["pixels_exact"] and record["verapdf"]["compliant"] and
                    all(record["logical_matches"].values()) and all(record["search_matches"].values()))
            except Exception as error:
                import traceback
                record["error"] = str(error)
                record["traceback"] = traceback.format_exc()
                record["all_checks_passed"] = False
            (root / "strategy-diagnostic.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    receipt["complete_matches_out_of_12"] = {name: sum(row["all_checks_passed"] for row in receipt["cases"]
        if row["variant"] == name) for name in variants}
    (root / "strategy-diagnostic.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(json.dumps(receipt["complete_matches_out_of_12"], indent=2))
    fixtures.ArtifactTests.tearDownClass()
    return 0 if any(count == len(sources) for count in receipt["complete_matches_out_of_12"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
