#!/usr/bin/env python3
"""Contract checks for manifest artifact normalization."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ocr_engine_benchmark", ROOT / "tests/ocr_engine_benchmark.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> int:
    assert MODULE.artifact_filename("legacy.gguf") == "legacy.gguf"
    assert MODULE.artifact_filename({"file": "detector.gguf", "repo": "cstr/example"}) == "detector.gguf"
    assert MODULE.artifact_filename(None) is None
    assert MODULE.pipeline_engine({"name": "ppocrv6-small", "engine": "ppocrv6"}) == "ppocrv6"
    assert MODULE.pipeline_engine({"name": "custom", "engine": "dbnet", "pipeline_engine": "tesseract"}) == "tesseract"
    assert not MODULE.runtime_failed({"timed_out": False, "returncode": 0}, "normal completion")
    assert MODULE.runtime_failed({"timed_out": False, "returncode": 0}, "ocr_pipeline: failed to load detection model")
    assert MODULE.runtime_failed({"timed_out": False, "returncode": 1}, "")
    print("OCR engine benchmark manifest contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
