#!/usr/bin/env python3
"""Compile exact production Rust renderer/FFI source against a real native DLL.

This hosted model-free gate avoids rebuilding unrelated inference backends.
The wrapper is extracted without rewriting, with SHA-256 for every source
slice and generated translation unit. The paired application gate separately
compiles the complete crate. Tests themselves live in the Rust contract file.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    safe = (root / "crispembed/src/lib.rs").read_text()
    ffi = (root / "crispembed-sys/src/lib.rs").read_text()
    result_start = safe.index("/// A single detected + recognized text region.")
    result_end = safe.index("\nimpl OcrPipeline", result_start)
    api_start = safe.index("/// One page of OCR regions for")
    api_end = safe.index("// SAFMN super-resolution", api_start)
    api_end = safe.rfind("// ---------------------------------------------------------------------------", api_start, api_end)
    ffi_start = ffi.index("/// `ocr_render_word`")
    ffi_end = ffi.index("\n}", ffi.index("pub fn ocr_render_free", ffi_start)) + 2
    slices = {"OcrResult": safe[result_start:result_end], "renderer": safe[api_start:api_end],
              "ffi": ffi[ffi_start:ffi_end]}
    assert slices["ffi"].count('extern "C" {') == 1
    source = ('#![allow(dead_code)]\nuse std::ffi::CStr;\nextern crate self as crispembed;\nmod crispembed_sys {\n'
              'use std::ffi::{c_char, c_int, c_float, c_void};\n' +
              slices["ffi"].replace('extern "C" {', '#[link(name = "ocr_render_test")]\nextern "C" {') +
              "\n}\n" + slices["OcrResult"] + slices["renderer"] +
              (root / "crispembed/tests/ocr_writer_contract.rs").read_text())
    build = options.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    translation = build / "ocr-writer-production.rs"
    translation.write_text(source)
    receipt = {"source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               "rustc": subprocess.check_output(["rustc", "--version", "--verbose"], text=True).strip(),
               "source_slices_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in slices.items()},
               "translation_unit_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "scope": "Exact production wrapper/FFI against model-free native library; not the complete inference crate"}
    (build / "artifacts" / "rust-source.json").write_text(json.dumps(receipt, indent=2))
    subprocess.run(["rustc", "--edition=2021", "--test", "-D", "warnings", str(translation),
                    "-L", "native=" + str(build), "-C", "link-arg=-Wl,-rpath,$ORIGIN",
                    "-o", str(build / "test-ocr-writer")], check=True)


if __name__ == "__main__":
    main()
