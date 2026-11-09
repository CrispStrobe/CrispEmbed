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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("PyMuPDF", "Poppler", "pypdf"), required=True)
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
        else:
            progress("extract_text")
            result = subprocess.run(["pdftotext", "-raw", "-enc", "UTF-8", str(options.pdf), "-"],
                                    capture_output=True, timeout=60)
            record["reader_exit_code"] = result.returncode
            record["reader_stderr"] = result.stderr.decode("utf-8", errors="replace")
            record["extracted"] = result.stdout.decode("utf-8")
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
