#!/usr/bin/env python3
"""Contract test for the independent EasyOCR page-reference runner."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_easyocr_reference_page.py"
COMPARE = ROOT / "tools" / "compare_easyocr_manifests.py"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = {
            "schema": "crispembed.easyocr.postprocess.v1",
            "mode": "lines",
            "records": [{
                "index": 0, "line": 0, "text": "hello", "confidence": 0.9,
                "detector_confidence": 0.0, "box": [10, 20, 40, 12],
                "crop": [8, 18, 44, 16], "normalized_box": [50, 200, 250, 320],
            }],
        }
        reference = tmp / "reference.json"
        native = tmp / "native.json"
        reference.write_text(json.dumps({**source, "detector_confidence_source": "unavailable"}))
        native.write_text(json.dumps({**source, "records": [dict(source["records"][0], detector_confidence=0.8)]}))
        result = subprocess.run([
            sys.executable, str(COMPARE), "--reference", str(reference), "--native", str(native),
            "--ignore-detector-confidence",
        ], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    print("easyocr reference page contract: PASS")


if __name__ == "__main__":
    main()
