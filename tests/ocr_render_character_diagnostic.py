#!/usr/bin/env python3
"""Hosted-only comparison of per-scalar PDF encodings; never production code.

Keep the reviewed renderer frozen. Re-encode its invisible layer at the same
Unicode 17 visual cells and record every reader, pixel and conformance result.
An experiment that improves one CLI must not hide damage in another reader.
"""

import argparse
import ctypes as C
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

from fontTools.ttLib import TTFont
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, ContentStream, DecodedStreamObject,
                           DictionaryObject, FloatObject, NameObject, NumberObject)

import check_ocr_pdfa
import test_ocr_render_artifacts as fixtures


class Sequence(C.Structure):
    _fields_ = [("encoding", C.c_uint32), ("buffer", C.c_void_p), ("length", C.c_size_t)]


class Run(C.Structure):
    _fields_ = [("offset", C.c_size_t), ("length", C.c_size_t), ("level", C.c_uint8)]


def visual_cells(lib, text):
    """Use the exact vendored UAX #9 implementation; do not reverse strings."""
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
    values = (C.c_uint32 * len(text))(*map(ord, text))
    sequence = Sequence(2, C.cast(values, C.c_void_p), len(text))
    algorithm = lib.SBAlgorithmCreate(C.byref(sequence))
    paragraph = lib.SBAlgorithmCreateParagraph(algorithm, 0, len(text), 0xfe)
    line = lib.SBParagraphCreateLine(paragraph, 0, len(text))
    if not algorithm or not paragraph or not line:
        raise RuntimeError("Could not resolve diagnostic visual positions")
    try:
        runs = lib.SBLineGetRunsPtr(line)
        positions = [(0.0, 0.0)] * len(text)
        cells = 0
        for index in range(lib.SBLineGetRunCount(line)):
            run = runs[index]
            clusters = []
            for source in range(run.offset, run.offset + run.length):
                category = lib.SBCodepointGetGeneralCategory(ord(text[source]))
                if category in (0x1a, 0x1b):
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
        return [(10 + left * 300 / cells, (right - left) * 300 / cells) for left, right in positions]
    finally:
        lib.SBLineRelease(line)
        lib.SBParagraphRelease(paragraph)
        lib.SBAlgorithmRelease(algorithm)


def scalar_pdf(source, destination, text, positions, alternate, em_scale):
    writer = PdfWriter(clone_from=source)
    page = writer.pages[0]
    fonts = page["/Resources"]["/Font"]
    mappings = {}
    for font_name, reference in list(fonts.items()):
        font = reference.get_object()
        for cid, encoded in re.findall(rb"<([0-9A-F]+)>\s*<([0-9A-F]+)>", font["/ToUnicode"].get_data()):
            value = bytes.fromhex(encoded.decode()).decode("utf-16-be")
            if len(value) == 1 and int(cid, 16):
                mappings.setdefault(value, (str(font_name), cid.decode()))
        descendant = font["/DescendantFonts"][0].get_object()
        # All original single-scalar CIDs use geometric glyph 1. Unused run
        # CIDs remain mapped, but no longer need their variable PDF widths.
        descendant.pop("/W", None)
        descendant[NameObject("/DW")] = FloatObject(1000 / em_scale)
        if em_scale != 1:
            descriptor = descendant["/FontDescriptor"]
            font_file = descriptor["/FontFile2"]
            with TTFont(BytesIO(font_file.get_data())) as program:
                program["head"].unitsPerEm *= em_scale
                assert program["head"].unitsPerEm <= 16384
                stream = BytesIO()
                program.save(stream)
            embedded = DecodedStreamObject()
            embedded.set_data(stream.getvalue())
            embedded[NameObject("/Length1")] = NumberObject(len(stream.getvalue()))
            descriptor[NameObject("/FontFile2")] = writer._add_object(embedded)
            descriptor[NameObject("/FontBBox")] = ArrayObject([
                FloatObject(float(value) / em_scale) for value in descriptor["/FontBBox"]])
            for key in ("/Ascent", "/Descent", "/CapHeight", "/StemV", "/MissingWidth"):
                descriptor[NameObject(key)] = FloatObject(float(descriptor[key]) / em_scale)
        if alternate:
            # Distinct PDF font objects with identical mappings/metrics force
            # real font changes without changing the encoded Unicode or cells.
            fonts[NameObject(str(font_name) + "Alt")] = writer._add_object(DictionaryObject(font))
    original = page.get_contents().get_data()
    prefix = original.split(b"/Span", 1)[0]
    content = bytearray(prefix)
    for index, (character, (x, width)) in enumerate(zip(text, positions)):
        font, cid = mappings[character]
        if alternate and index & 1:
            font += "Alt"
        actual = "FEFF" + character.encode("utf-16-be").hex().upper()
        content.extend((f"/Span << /ActualText <{actual}> >> BDC\nBT\n3 Tr\n"
                        f"{font} {30 * em_scale} Tf\n{width / 30:.12f} 0 0 1 {x:.12f} 180 Tm\n"
                        f"<{cid}> Tj\nET\nEMC\n").encode())
    content.extend(b"Q\n")
    stream = DecodedStreamObject()
    stream.set_data(bytes(content))
    page[NameObject("/Contents")] = writer._add_object(stream)
    with destination.open("wb") as output:
        writer.write(output)


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
    reports = root / "verapdf"
    reports.mkdir()
    receipt = {"purpose": "Per-scalar encoding experiment; not production acceptance", "cases": []}
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    receipt["source_sha"] = source_sha
    controls = dict.fromkeys([0x061c, 0x200e, 0x200f, *range(0x202a, 0x202f), *range(0x2066, 0x206a)])
    normal = lambda text: " ".join(text.translate(controls).split())
    expected_pixels = hashlib.sha256(Image.open(root / "rgb.png").convert("RGB").tobytes()).hexdigest()
    sources = [("arabic", "العربية"), ("mixed", "العربية 123 ABC"),
               ("marks", "مَرْحَبًا שָׁלוֹם ١٢٣ / 42"), ("isolate", "Report \u2067العربية 123\u2069: ABC"),
               ("supplementary", "\U0001e900\U0001e922 123 ABC"),
               ("repeated", "מ" + "\u05b7" * 70 + " אאא 123 ABC")]
    variants = [("one-font", False, 1), ("alternating-fonts", True, 1),
                ("alternating-wide-em", True, 256)]
    for name, text in sources:
        base = case.render("base-" + name + ".pdf", [{"image": root / "rgb.png",
                           "regions": [(text, 10, 30, 300, 30)]}], pdfa=True, bidi=True)
        positions = visual_cells(case.lib, text)
        for variant, alternate, em_scale in variants:
            path = root / (variant + "-" + name + ".pdf")
            record = {"case": name, "variant": variant, "source": text, "visual_cells": positions}
            receipt["cases"].append(record)
            try:
                scalar_pdf(base, path, text, positions, alternate, em_scale)
                reader = PdfReader(path, strict=True)
                logical = "".join(str(operands[1]["/ActualText"])
                                  for operands, op in ContentStream(reader.pages[0]["/Contents"], reader).operations
                                  if op == b"BDC" and "/ActualText" in operands[1])
                record["actual_text_exact"] = logical == text
                searches = [query for query in ("العربية", "ABC") if query in text]
                record["readers"] = {engine: case.probe_pdf(path, engine, searches
                                           if engine in ("PyMuPDF", "PDFium") else ())
                                     for engine in ("PyMuPDF", "Poppler", "pypdf", "PDFium")}
                record["logical_matches"] = {engine: probe.get("exit_code") == 0 and
                                               normal(probe.get("extracted", "")) == normal(text)
                                              for engine, probe in record["readers"].items()}
                record["pixels_exact"] = record["readers"]["PyMuPDF"].get("pixel_sha256") == expected_pixels
                record["search_matches"] = {engine: all(record["readers"][engine].get("search", {}).get(query)
                                                        for query in searches)
                                            for engine in ("PyMuPDF", "PDFium")}
                record["verapdf"] = check_ocr_pdfa.validate(options.verapdf, path, reports / path.stem)
                record["all_checks_passed"] = (record["actual_text_exact"] and record["pixels_exact"] and
                                                record["verapdf"]["compliant"] and
                                                all(record["logical_matches"].values()) and
                                                all(record["search_matches"].values()))
            except Exception as error:
                record["error"] = str(error)
                record["all_checks_passed"] = False
            (root / "character-diagnostic.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    summary = {name: sum(row["all_checks_passed"] for row in receipt["cases"] if row["variant"] == name)
               for name, _, _ in variants}
    receipt["all_reader_conformance_matches_out_of_6"] = summary
    (root / "character-diagnostic.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))
    fixtures.ArtifactTests.tearDownClass()
    return 0 if any(count == len(sources) for count in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
