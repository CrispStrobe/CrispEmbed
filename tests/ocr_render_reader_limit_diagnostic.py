#!/usr/bin/env python3
"""Retain strict supplementary search checks and an independent producer control.

This diagnostic is separate from writer/archive/exact-copy qualification. A
failed full-scalar search remains a failed assertion. Lone-surrogate queries
only diagnose the reader's internal representation; they never enter a PDF.
The unmodified independent producer output is retained even if its CMap has
malformed destinations, and only corrected UTF-16BE output is a valid control.
"""

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import subprocess
import traceback

import fitz
import pypdfium2 as pdfium
import pypdfium2_raw as raw
import pypdfium2.version as version
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_control(root, font):
    text = "Independent café 😀 e\u0301"
    original = root / "independent-producer.pdf"
    corrected = root / "independent-producer-corrected-cmap.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_font(fontname="ReaderControl", fontfile=str(font))
        page.insert_text((40, 80), text, fontname="ReaderControl", fontsize=18)
        document.save(original)

    writer = PdfWriter(clone_from=original)
    corrections = []
    for page in writer.pages:
        for font_ref in page["/Resources"]["/Font"].values():
            resource = font_ref.get_object()
            mapping = resource["/ToUnicode"].get_data().decode("ascii")
            changed = []
            for line in mapping.splitlines(keepends=True):
                # CID sources are 16-bit. Match only the destination after one
                # bfchar source or two bfrange source endpoints, not arbitrary
                # hex tokens in the font or content stream.
                match = re.fullmatch(r"(\s*<[0-9a-fA-F]{4}>\s+(?:<[0-9a-fA-F]{4}>\s+)?)"
                                     r"<([0-9a-fA-F]{5,6})>(\s*)", line)
                if match:
                    scalar = int(match[2], 16)
                    if not 0x10000 <= scalar <= 0x10ffff:
                        raise AssertionError("Independent producer has an unsupported malformed destination")
                    replacement = chr(scalar).encode("utf-16-be").hex()
                    corrections.append({"from": "<" + match[2] + ">", "to": "<" + replacement + ">"})
                    line = match[1] + "<" + replacement + ">" + match[3]
                changed.append(line)
            updated = "".join(changed)
            section = None
            for line in updated.splitlines():
                if "beginbfchar" in line:
                    section = "char"
                elif "beginbfrange" in line:
                    section = "range"
                elif "endbf" in line:
                    section = None
                elif section:
                    tokens = re.findall(r"<([0-9a-fA-F]+)>", line)
                    if not tokens:
                        continue
                    destination = bytes.fromhex(tokens[1 if section == "char" else 2])
                    destination.decode("utf-16-be", errors="strict")
                    if section == "range":
                        first, last = int(tokens[0], 16), int(tokens[1], 16)
                        for offset in range(last - first + 1):
                            (int.from_bytes(destination, "big") + offset).to_bytes(len(destination), "big").decode(
                                "utf-16-be", errors="strict")
            if updated != mapping:
                stream = DecodedStreamObject()
                stream.set_data(updated.encode("ascii"))
                resource[NameObject("/ToUnicode")] = writer._add_object(stream)
    writer.write(corrected)

    original_reader, corrected_reader = PdfReader(original), PdfReader(corrected)
    assert original_reader.pages[0].get_contents().get_data() == corrected_reader.pages[0].get_contents().get_data()
    assert original_reader.pages[0].mediabox == corrected_reader.pages[0].mediabox
    programs = []
    for reader in (original_reader, corrected_reader):
        resource = next(iter(reader.pages[0]["/Resources"]["/Font"].values())).get_object()
        descendant = resource["/DescendantFonts"][0].get_object()
        program = descendant["/FontDescriptor"]["/FontFile2"].get_data()
        programs.append(hashlib.sha256(program).hexdigest())
    assert programs[0] == programs[1], "Independent glyph program changed during CMap repair"
    return corrected, text, {
        "producer": "PyMuPDF " + fitz.VersionBind,
        "font_file": font.name, "font_sha256": sha256(font),
        "font_program_sha256": programs[0], "content_and_media_box_unchanged": True,
        "all_corrected_cmap_destinations_valid_utf16be": True,
        "original_pdf": original.name, "original_sha256": sha256(original),
        "corrected_pdf": corrected.name, "corrected_sha256": sha256(corrected),
        "corrected_destinations": corrections,
        "original_is_valid_unicode_control": not corrections,
        "correction_scope": "Only malformed scalar destinations become UTF-16BE; pypdf serializes the object graph",
    }


def probe(path, expected):
    record = {"file": path.name, "sha256": sha256(path), "expected": expected, "raw_search": []}
    with fitz.open(path) as document:
        record["mupdf_copy"] = document[0].get_text().strip()
        record["mupdf_supplementary_search"] = [list(rect) for rect in document[0].search_for("😀")]
    with pdfium.PdfDocument(path) as document:
        page = document[0]
        try:
            text = page.get_textpage()
            try:
                record["pdfium_copy"] = text.get_text_bounded(errors="strict").strip()
                record["character_unicode"] = [int(raw.FPDFText_GetUnicode(text.raw, i))
                                                for i in range(text.count_chars())]
                for needle in ("café", "e\u0301", "😀", "\ud83d", "\ude00"):
                    encoded = needle.encode("utf-16-le", errors="surrogatepass")
                    units = [int.from_bytes(encoded[i:i + 2], "little") for i in range(0, len(encoded), 2)]
                    buffer = (C.c_ushort * (len(units) + 1))(*units, 0)
                    for flags in (0, 1, 2):
                        handle = raw.FPDFText_FindStart(text.raw, buffer, flags, 0)
                        if not handle:
                            raise RuntimeError("PDFium did not create a raw search handle")
                        try:
                            found = bool(raw.FPDFText_FindNext(handle))
                            record["raw_search"].append({
                                "needle": needle, "utf16_units": units, "flags": flags, "found": found,
                                "index": int(raw.FPDFText_GetSchResultIndex(handle)) if found else None,
                                "count": int(raw.FPDFText_GetSchCount(handle)) if found else None,
                                "diagnostic_lone_surrogate": len(needle) == 1 and 0xd800 <= ord(needle) <= 0xdfff,
                            })
                        finally:
                            raw.FPDFText_FindClose(handle)
            finally:
                text.close()
        finally:
            page.close()
    record["copy_exact"] = record["mupdf_copy"] == expected and record["pdfium_copy"] == expected
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-pdf", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binaries = sorted(Path(raw.__file__).parent.glob("*pdfium*.so"))
    if len(binaries) != 1:
        raise RuntimeError("Expected exactly one identifiable PDFium shared library")
    report = {
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scope": "Strict reader supplementary-search diagnostic, separate from writer/archive/exact-copy acceptance",
        "versions": {"pymupdf": fitz.VersionBind, "mupdf": fitz.VersionFitz,
                     "wrapper": str(version.PYPDFIUM_INFO), "pdfium": str(version.PDFIUM_INFO),
                     "pdfium_flags": list(version.PDFIUM_INFO.flags)},
        "binary_sha256": {path.name: sha256(path) for path in binaries},
        "cases": [], "failures": [],
    }
    try:
        independent, text, provenance = independent_control(args.output, args.font)
        report["independent_producer"] = provenance
        cases = [(args.native_pdf, "Reviewed archive café 中文 😀 e\u0301"), (independent, text)]
        for path, expected in cases:
            result = probe(path, expected)
            report["cases"].append(result)
            if not result["copy_exact"]:
                report["failures"].append({"file": path.name, "check": "exact-copy"})
            if not result["mupdf_supplementary_search"]:
                report["failures"].append({"file": path.name, "check": "MuPDF supplementary search"})
            for query in result["raw_search"]:
                if query["flags"] == 0 and not query["diagnostic_lone_surrogate"] and not query["found"]:
                    report["failures"].append({"file": path.name, "check": "PDFium raw search", "needle": query["needle"]})
    except Exception:
        report["exception"] = traceback.format_exc()
        raise
    finally:
        # ensure_ascii also safely preserves the deliberately invalid query
        # units in JSON; those queries never change either valid PDF.
        (args.output / "supplementary-search.json").write_text(json.dumps(report, indent=2, ensure_ascii=True))
    if report["failures"]:
        raise AssertionError(json.dumps(report["failures"], ensure_ascii=True))
    print("Exact copies and full supplementary searches passed both producer controls")


if __name__ == "__main__":
    main()
