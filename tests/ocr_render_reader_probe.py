#!/usr/bin/env python3
"""Read one generated PDF in an isolated process and retain crash progress.

Native reader failures must not erase other cases or prevent archival
validation. The parent keeps this receipt, stdout, stderr and return code.
"""

import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("PyMuPDF", "Poppler", "pypdf", "PDFium"), required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search", action="append", default=[])
    options = parser.parse_args()
    faulthandler.enable()
    record = {"engine": options.engine, "file": options.pdf.name, "status": "running"}

    def progress(stage):
        record["stage"] = stage
        temporary = options.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        temporary.replace(options.output)

    try:
        progress("open")
        if options.engine == "PyMuPDF":
            import fitz
            with fitz.open(options.pdf) as document:
                record["repaired"] = document.is_repaired
                record["pages"] = len(document)
                progress("extract_text")
                record["extracted"] = document[0].get_text()
                record["search"] = {}
                for query in options.search:
                    progress("search:" + query)
                    record["search"][query] = [list(rect) for rect in document[0].search_for(query)]
                progress("rasterize")
                pixmap = document[0].get_pixmap(alpha=False)
                record["pixel_sha256"] = hashlib.sha256(pixmap.samples).hexdigest()
                record["pixel_size"] = [pixmap.width, pixmap.height]
        elif options.engine == "pypdf":
            from pypdf import PdfReader
            reader = PdfReader(options.pdf, strict=True)
            record["pages"] = len(reader.pages)
            progress("extract_text")
            record["extracted"] = reader.pages[0].extract_text()
        elif options.engine == "PDFium":
            import pypdfium2 as pdfium
            with pdfium.PdfDocument(options.pdf) as document:
                record["pages"] = len(document)
                with document[0] as page, page.get_textpage() as text:
                    progress("extract_text")
                    # GetText is UCS-2-limited; the bounded API preserves
                    # supplementary Unicode and exposes PDFium's actual text.
                    record["extracted"] = text.get_text_bounded(errors="strict")
                    record["search"] = {}
                    for query in options.search:
                        progress("search:" + query)
                        with text.search(query) as search:
                            match = search.get_next()
                            record["search"][query] = list(match) if match else []
        else:
            progress("extract_text")
            result = subprocess.run(["pdftotext", "-raw", "-enc", "UTF-8", str(options.pdf), "-"],
                                    capture_output=True, timeout=60)
            record["reader_exit_code"] = result.returncode
            record["reader_stderr"] = result.stderr.decode("utf-8", errors="replace")
            record["extracted"] = result.stdout.decode("utf-8")
            progress("extract_raw_words")
            words = subprocess.run(["pdftotext", "-raw", "-bbox", "-enc", "UTF-8", str(options.pdf), "-"],
                                   capture_output=True, timeout=60)
            record["word_api_exit_code"] = words.returncode
            record["word_api_stderr"] = words.stderr.decode("utf-8", errors="replace")
            if not words.returncode:
                root = ET.fromstring(words.stdout)
                record["raw_words"] = [{"text": node.text or "", "bounds": node.attrib}
                                       for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "word"]
            if result.returncode or result.stderr:
                raise RuntimeError("Poppler failed or reported a parsing warning")
        record["status"] = "passed"
        progress("complete")
    except Exception:
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
        progress(record["stage"])
        raise


if __name__ == "__main__":
    main()
