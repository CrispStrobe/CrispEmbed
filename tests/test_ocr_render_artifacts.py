#!/usr/bin/env python3
"""Open, extract and rasterize real model-free native OCR exports.

Run with --library PATH to a matching native library. --artifacts DIR retains
the generated PDFs, XML, source images and raster previews for independent QA.
Requires Pillow, pypdf, PyMuPDF and the Poppler pdftotext/pdftoppm executables.
"""

import argparse
import ctypes as C
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

import fitz
from PIL import Image, ImageChops, ImageDraw, ImageStat
from pypdf import PdfReader


class Word(C.Structure):
    _fields_ = [("text", C.c_char_p), ("x", C.c_int), ("y", C.c_int),
                ("w", C.c_int), ("h", C.c_int), ("confidence", C.c_float)]


class Line(C.Structure):
    _fields_ = [("words", C.POINTER(Word)), ("n_words", C.c_int),
                ("x", C.c_int), ("y", C.c_int), ("w", C.c_int), ("h", C.c_int)]


class Page(C.Structure):
    _fields_ = [("lines", C.POINTER(Line)), ("n_lines", C.c_int),
                ("width", C.c_int), ("height", C.c_int), ("image_path", C.c_char_p)]


def native_library(path):
    lib = C.CDLL(str(Path(path).resolve()))
    signatures = {
        "ocr_render_create": ([C.c_int], C.c_void_p),
        "crispembed_ocr_render_capabilities": ([], C.c_uint32),
        "ocr_render_set_pdfa": ([C.c_void_p, C.c_int], None),
        "ocr_render_begin": ([C.c_void_p], None),
        "ocr_render_set_page_size_points": ([C.c_void_p, C.c_double, C.c_double], None),
        "ocr_render_add_page": ([C.c_void_p, C.POINTER(Page)], None),
        "ocr_render_end": ([C.c_void_p], None),
        "ocr_render_error": ([C.c_void_p], C.c_char_p),
        "ocr_render_output": ([C.c_void_p], C.c_void_p),
        "ocr_render_output_size": ([C.c_void_p], C.c_int),
        "ocr_render_free": ([C.c_void_p], None),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    return lib


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = native_library(OPTIONS.library)
        cls.temporary = tempfile.TemporaryDirectory(prefix="ocr-render-")
        cls.root = Path(OPTIONS.artifacts or cls.temporary.name)
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.receipt = {"tests": [], "capabilities": cls.lib.crispembed_ocr_render_capabilities()}
        source = Image.new("RGB", (320, 240), "white")
        draw = ImageDraw.Draw(source)
        draw.rectangle((0, 0, 31, 31), fill=(230, 20, 30))
        draw.rectangle((288, 0, 319, 31), fill=(20, 210, 40))
        draw.rectangle((0, 208, 31, 239), fill=(30, 50, 220))
        draw.rectangle((288, 208, 319, 239), fill=(230, 180, 20))
        draw.text((70, 100), "Native OCR export fixture", fill=(40, 40, 40))
        source.save(cls.root / "rgb.png")
        source.convert("L").save(cls.root / "gray.png")
        source.save(cls.root / "rgb.jpg", quality=95, subsampling=0)
        alpha = source.convert("RGBA")
        alpha.putalpha(128)
        alpha.save(cls.root / "alpha.png")
        source.quantize(colors=16).save(cls.root / "palette.png")

    @classmethod
    def tearDownClass(cls):
        (cls.root / "validation.json").write_text(json.dumps(cls.receipt, indent=2, ensure_ascii=False))
        cls.temporary.cleanup()

    def render(self, name, pages, fmt=3, pdfa=False, expected_error=None):
        lib = self.lib
        renderer = lib.ocr_render_create(fmt)
        self.assertTrue(renderer)
        try:
            lib.ocr_render_set_pdfa(renderer, int(pdfa))
            lib.ocr_render_begin(renderer)
            for page in pages:
                records = page.get("regions", [("Image text", 40, 40, 240, 24)])
                words = (Word * len(records))()
                lines = (Line * len(records))()
                for i, (text, x, y, width, height) in enumerate(records):
                    text = text.encode("utf-8") if isinstance(text, str) else text
                    words[i] = Word(text, x, y, width, height, 0.9)
                    lines[i] = Line(C.pointer(words[i]), 1, x, y, width, height)
                image = page.get("image")
                image = str(image).encode() if image else None
                native = Page(lines, len(lines), page.get("width", 320), page.get("height", 240), image)
                if "points" in page:
                    lib.ocr_render_set_page_size_points(renderer, *page["points"])
                lib.ocr_render_add_page(renderer, C.byref(native))
            lib.ocr_render_end(renderer)
            error = lib.ocr_render_error(renderer)
            size = lib.ocr_render_output_size(renderer)
            if expected_error is not None:
                self.assertIsNotNone(error)
                self.assertIn(expected_error, error.decode())
                self.assertEqual(size, 0)
                return None
            self.assertIsNone(error, error.decode() if error else "")
            data = C.string_at(lib.ocr_render_output(renderer), size)
            output = self.root / name
            output.write_bytes(data)
            return output
        finally:
            lib.ocr_render_free(renderer)

    def read_pdf(self, path):
        reader = PdfReader(path, strict=True)
        document = fitz.open(path)
        self.assertFalse(document.is_repaired, "PDF reader had to repair a malformed file")
        self.assertEqual(len(reader.pages), len(document))
        return reader, document

    def poppler_text(self, path):
        result = subprocess.run(["pdftotext", "-raw", "-enc", "UTF-8", str(path), "-"],
                                check=True, capture_output=True)
        self.assertEqual(result.stderr, b"", result.stderr.decode(errors="replace"))
        return result.stdout.decode("utf-8")

    def test_raster_images_and_physical_size(self):
        for filename in ("rgb.png", "gray.png", "alpha.png", "palette.png", "rgb.jpg"):
            with self.subTest(image=filename):
                path = self.render(filename + ".pdf", [{"image": self.root / filename, "points": (160, 120)}])
                reader, document = self.read_pdf(path)
                self.assertEqual(tuple(float(x) for x in reader.pages[0].mediabox), (0, 0, 160, 120))
                image_ref = reader.pages[0]["/Resources"]["/XObject"]["/Im1"].get_object()
                self.assertEqual(image_ref["/ColorSpace"], "/DeviceRGB")
                self.assertEqual(len(image_ref.get_data()), 320 * 240 * 3)
                actual = Image.frombytes("RGB", (320, 240), document[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).samples)
                expected_rgba = Image.open(self.root / filename).convert("RGBA")
                expected = Image.alpha_composite(Image.new("RGBA", expected_rgba.size, "white"), expected_rgba).convert("RGB")
                maximum = max(high for low, high in ImageChops.difference(actual, expected).getextrema())
                mean = sum(ImageStat.Stat(ImageChops.difference(actual, expected)).mean) / 3
                self.assertLessEqual(maximum, 5 if filename.endswith("jpg") else 0)
                self.assertLess(mean, 0.6 if filename.endswith("jpg") else 0.001)
                actual.save(self.root / (filename + "-render.png"))
                # Selection bounds must scale with the same physical page transform.
                words = document[0].get_text("words")
                self.assertTrue(words)
                self.assertTrue(all(19.9 <= w[0] <= w[2] <= 140.1 and 19.9 <= w[1] <= w[3] <= 32.1 for w in words), words)
                self.assertIn("Image text", self.poppler_text(path))
                document.close()
                self.receipt["tests"].append({"image": filename, "max_pixel_error": maximum,
                                              "mean_pixel_error": mean, "points": [160, 120]})
        path = self.root / "rgb.png.pdf"
        result = subprocess.run(["pdftoppm", "-r", "144", "-singlefile", "-png", str(path),
                                 str(self.root / "poppler-raster")], capture_output=True, check=True)
        self.assertEqual(result.stderr, b"")
        poppler = Image.open(self.root / "poppler-raster.png").convert("RGB")
        self.assertEqual(poppler.size, (320, 240))
        expected = Image.open(self.root / "rgb.png")
        # Poppler applies edge resampling even at an integral image scale.
        # The decoded stream and MuPDF rendering above are byte exact; here
        # require a small whole-image error plus unchanged interior markers.
        self.assertLess(sum(ImageStat.Stat(ImageChops.difference(poppler, expected)).mean) / 3, 1.0)
        for point in ((10, 10), (310, 10), (10, 230), (310, 230), (150, 50)):
            self.assertEqual(poppler.getpixel(point), expected.getpixel(point))

    def test_unicode_multiline_and_long_corrections(self):
        unicode_text = "café € Ελληνικά 中文 😀 e\u0301"
        long_text = " ".join(["naïve 文 😀"] * 700) + " COMPLETE_SUFFIX_終"
        multiline = "First corrected line\nSecond corrected line"
        path = self.render("unicode-long.pdf", [{"image": self.root / "rgb.png", "regions": [
            (unicode_text, 20, 35, 280, 20), (long_text, 20, 65, 280, 30),
            (multiline, 20, 150, 280, 60)]}])
        reader, document = self.read_pdf(path)
        for source, extracted in (("PyMuPDF", document[0].get_text()), ("Poppler", self.poppler_text(path)),
                                  ("pypdf", reader.pages[0].extract_text())):
            self.assertIn(unicode_text, extracted)
            self.assertTrue(long_text in " ".join(extracted.split()), source + " lost corrected paragraph text")
            self.assertIn(multiline, extracted)
        self.assertTrue(document[0].search_for("中文"))
        self.assertTrue(document[0].search_for("COMPLETE_SUFFIX_終"))
        for font in reader.pages[0]["/Resources"]["/Font"].values():
            font = font.get_object()
            descendant = font["/DescendantFonts"][0].get_object()
            self.assertIn("/ToUnicode", font)
            self.assertIn("/CIDToGIDMap", descendant)
            self.assertIn("/FontFile2", descendant["/FontDescriptor"])
        self.assertNotIn(b"pdfaid:conformance", path.read_bytes())
        document.close()
        self.receipt["tests"].append({"unicode": unicode_text, "long_text_bytes": len(long_text.encode()),
                                      "independent_extractors": ["PyMuPDF", "Poppler", "pypdf"]})

    def test_large_page_tree_and_per_page_sizes(self):
        pages = [{"image": self.root / "rgb.png", "points": (160 + i, 120 + i),
                  "regions": [(f"Page {i + 1} complete", 20, 50, 280, 30)]} for i in range(125)]
        path = self.render("125-pages.pdf", pages)
        reader, document = self.read_pdf(path)
        self.assertEqual(len(document), 125)
        self.assertEqual(len(reader.trailer["/Root"]["/Pages"]["/Kids"]), 125)
        text = self.poppler_text(path)
        for i, page in enumerate(reader.pages):
            self.assertIn(f"Page {i + 1} complete", text)
            self.assertEqual(float(page.mediabox.width), 160 + i)
            self.assertEqual(float(page.mediabox.height), 120 + i)
        document.close()
        self.receipt["tests"].append({"page_count": 125, "page_tree": "parsed_without_repair"})

    def test_unicode_font_subset_boundary(self):
        # Private-use scalars exercise more than 65535 distinct CIDs without
        # involving shaping, missing display fonts or script direction rules.
        scalars = [cp for cp in range(0xF0000, 0x100008) if cp not in (0xFFFFE, 0xFFFFF)]
        text = "".join(chr(cp) for cp in scalars)
        path = self.render("unicode-subsets.pdf", [{"regions": [(text, 10, 10, 300, 220)]}])
        reader, document = self.read_pdf(path)
        self.assertEqual(len(reader.pages[0]["/Resources"]["/Font"]), 2)
        for source, extracted in (("PyMuPDF", document[0].get_text()), ("Poppler", self.poppler_text(path)),
                                  ("pypdf", reader.pages[0].extract_text())):
            self.assertTrue(extracted.replace("\n", "").replace("\f", "") == text,
                            source + " lost characters at the CID subset boundary")
        document.close()
        self.receipt["tests"].append({"unique_scalars": len(scalars), "font_subsets": 2})

    def test_text_only_pdf_is_safe_for_legacy_string_callers(self):
        text = "Text-only café 中文 😀"
        path = self.render("text-only.pdf", [{"regions": [(text, 10, 30, 300, 30)]}])
        data = path.read_bytes()
        self.assertTrue(data.isascii())
        self.assertNotIn(b"\0", data)
        reader, document = self.read_pdf(path)
        self.assertIn(text, document[0].get_text())
        self.assertIn(text, self.poppler_text(path))
        self.assertIn(text, reader.pages[0].extract_text())
        document.close()

    def test_long_xml_and_invalid_controls(self):
        text = ("文 & <tag> \"quoted\" 😀\n\t\r" * 400) + "FINAL_終"
        for fmt, suffix in ((1, "hocr"), (2, "alto")):
            path = self.render("long." + suffix, [{"regions": [(text, 0, 0, 320, 240)]}], fmt=fmt)
            root = ET.fromstring(path.read_bytes())
            values = ([node.attrib["CONTENT"] for node in root.iter() if "CONTENT" in node.attrib]
                      if fmt == 2 else [node.text for node in root.iter() if node.attrib.get("class") == "ocrx_word"])
            self.assertEqual(values, [text])
            path = self.render("controls." + suffix, [{"regions": [(b"A\x01B\xffC", 0, 0, 320, 240)]}], fmt=fmt)
            root = ET.fromstring(path.read_bytes())
            self.assertIn("A\ufffdB\ufffdC", path.read_text())
        self.receipt["tests"].append({"xml_bytes": len(text.encode()), "controls": "replacement-decoded"})

    def test_error_contract(self):
        self.assertEqual(self.lib.crispembed_ocr_render_capabilities() & 15, 7)
        self.render("unsupported.pdf", [{}], pdfa=True, expected_error="PDF/A")
        self.render("missing.pdf", [{"image": self.root / "missing.png"}], expected_error="Could not read")
        self.render("mismatch.pdf", [{"image": self.root / "rgb.png", "width": 640}], expected_error="dimensions")
        self.render("outside.pdf", [{"regions": [("outside", 310, 0, 50, 20)]}], expected_error="bounds")
        self.render("empty.pdf", [], expected_error="at least one page")
        self.render("invalid-size.pdf", [{"points": (float("nan"), 120)}], expected_error="finite")
        self.render("huge-size.pdf", [{"points": (20000, 120)}], expected_error="14400")
        self.render("tiny-size.pdf", [{"points": (0.000001, 120)}], expected_error="0.01")

    def test_bidirectional_pdf_fails_without_corrupting_other_formats(self):
        for text in ("العربية 123 ABC", "שלום 123", "\u202eoverride\u202c", "\u2067isolate\u2069", "\U0001e900"):
            with self.subTest(text=text):
                page = {"regions": [(text, 10, 30, 300, 30)]}
                self.render("unsupported-rtl.pdf", [page], expected_error="right-to-left")
                alto = self.render("rtl.alto", [page], fmt=2)
                root = ET.fromstring(alto.read_bytes())
                self.assertEqual([node.attrib["CONTENT"] for node in root.iter() if "CONTENT" in node.attrib], [text])
                plain = self.render("rtl.txt", [page], fmt=0)
                self.assertEqual(plain.read_text(), text + "\n")
        self.receipt["tests"].append({"bidi_pdf": "explicit_unsupported_error", "bidi_alto_text": "exact"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--artifacts")
    OPTIONS, remaining = parser.parse_known_args()
    unittest.main(argv=[__file__, *remaining])
