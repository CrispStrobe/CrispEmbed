"""Actual current Python wrapper against an explicit older native library."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--pdfium", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    with args.library.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual == args.library_sha256, "verify pinned old library before loading"
    sys.path.insert(0, str(args.python_source.resolve()))
    from crispembed import CrispMarker
    import crispembed._marker as wrapper

    rows, failures = [], []

    def check(value, name):
        rows.append({"name": name, "passed": bool(value)})
        if not value:
            raise AssertionError(name)

    legacy = ("default_config", "init", "free", "convert", "convert_raw", "render", "metadata", "error",
              "asset_count", "asset_name", "asset_data")
    extensions = ("save_extraction", "open_extraction", "start", "step", "cancel", "progress")
    lib = ctypes.CDLL(str(args.library))
    check(all(hasattr(lib, "crispembed_marker_" + name) for name in legacy), "old release exports every legacy Marker symbol")
    missing = {}
    for name in extensions:
        missing[name] = not hasattr(lib, "crispembed_marker_" + name)
        check(missing[name], "old release lacks " + name)

    with tempfile.TemporaryDirectory(prefix="marker-python-old-") as temporary:
        target = Path(temporary) / "unavailable.cemk"
        with CrispMarker(mode="text", lib_path=str(args.library), pdfium_path=str(args.pdfium), extract_images=False) as marker:
            initial = marker.convert(args.pdf)
            formats = ("markdown", "html", "json", "chunks")
            required = ("Native document test", "Hello from CrispEmbed.", "PDF text stays searchable.")
            check(all(text in initial for text in required), "current Python wrapper converts the real owned PDF with v0.17.15")
            rendered = {fmt: marker.render(fmt) for fmt in formats}
            visible = VisibleText()
            visible.feed(rendered["html"])
            document = json.loads(rendered["json"])
            chunks = json.loads(rendered["chunks"])
            document_text = tuple(block["text"] for page in document["children"] for block in page["children"])
            chunk_text = tuple(block["text"] for block in chunks["blocks"])
            check(all(text in "".join(visible.parts) for text in required)
                  and document_text == required and chunk_text == required,
                  "legacy Markdown, visible HTML and JSON/chunk text match the owned gold")
            check(marker.metadata["page_count"] == 1 and marker.assets == {}, "legacy metadata and asset access remain usable")
            operations = {
                "save_extraction": lambda: marker.save_extraction(target),
                "open_extraction": lambda: marker.open_extraction(target),
                "start": lambda: marker.start(args.pdf, target),
                "step": lambda: marker.step(1),
                "cancel": marker.cancel,
                "progress": lambda: marker.progress,
            }
            for name, operation in operations.items():
                before = {fmt: marker.render(fmt) for fmt in formats}
                try:
                    operation()
                except RuntimeError as error:
                    cause = error.__cause__
                    missing_symbol = ("does not provide Marker extraction artifacts/lifecycle" in str(error)
                                      and isinstance(cause, AttributeError)
                                      and "crispembed_marker_" + name in str(cause))
                    failures.append({"method": name, "error_type": type(error).__name__, "message": str(error),
                                     "cause_type": type(cause).__name__, "missing_symbol": "crispembed_marker_" + name})
                    check(missing_symbol, name + " reports the unavailable native symbol")
                else:
                    check(False, name + " unexpectedly succeeded on the older library")
                check(not target.exists(), name + " creates no artifact or checkpoint")
                check({fmt: marker.render(fmt) for fmt in formats} == before,
                      name + " failure preserves all four existing renders")
                check(marker.convert(args.pdf) == initial, name + " failure permits later exact legacy conversion")
    print(json.dumps({"client": "Python", "suite": "actual-v0.17.15-compatibility", "passed": True,
                      "checks": rows, "total": len(rows), "missing_symbols": missing,
                      "expected_unavailable_errors": failures, "library_sha256": actual,
                      "wrapper_sha256": hashlib.sha256(Path(wrapper.__file__).read_bytes()).hexdigest(),
                      "model_weights_used": False, "real_pdfium_used": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
