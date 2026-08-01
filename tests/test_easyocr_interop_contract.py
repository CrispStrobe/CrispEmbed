#!/usr/bin/env python3
"""Exercise the weight-free EasyOCR -> native/LayoutLM handoff contract.

This is intentionally a deterministic contract test, not a quality claim for
the recognizer.  Real EasyOCR page manifests remain the live acceptance gate.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POSTPROCESS = ROOT / "tools" / "easyocr_postprocess_reference.py"
LAYOUTLM = ROOT / "tools" / "validate_layoutlm_handoff.py"


def run_json(command: list[str]) -> dict:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def main() -> None:
    source = {
        "image": "german-page.png",
        "width": 1000,
        "height": 600,
        "items": [
            {"box": {"x": 620, "y": 110, "w": 110, "h": 30}, "text": "Welt", "confidence": 0.81},
            {"box": {"x": 100, "y": 108, "w": 120, "h": 32}, "text": "Guten", "confidence": 0.93},
            {"box": [[100, 210], [280, 210], [280, 246], [100, 246]], "text": "zweite", "confidence": 0.77},
        ],
    }
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        source_path = directory / "easyocr.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        words_path = directory / "words.json"
        lines_path = directory / "lines.json"
        for mode, output in (("words", words_path), ("lines", lines_path)):
            subprocess.run(
                [sys.executable, str(POSTPROCESS), "--input", str(source_path), "--output", str(output), "--mode", mode],
                check=True,
                capture_output=True,
                text=True,
            )
        words = json.loads(words_path.read_text(encoding="utf-8"))
        lines = json.loads(lines_path.read_text(encoding="utf-8"))
        assert [record["text"] for record in words["records"]] == ["Guten", "Welt", "zweite"]
        assert [record["line"] for record in words["records"]] == [0, 0, 1]
        assert [record["text"] for record in lines["records"]] == ["Guten Welt", "zweite"]
        assert lines["records"][0]["box"] == [100.0, 108.0, 630.0, 32.0]
        assert words["records"][0]["crop"] == [98, 106, 124, 36]
        assert words["records"][0]["normalized_box"] == [100, 180, 220, 233]

        handoff = run_json([sys.executable, str(LAYOUTLM), "--manifest", str(words_path)])
        assert handoff["apply_ocr"] is False
        assert handoff["processor_args"]["words"] == ["Guten", "Welt", "zweite"]
        assert handoff["processor_args"]["boxes"] == [
            [100, 180, 220, 233],
            [620, 183, 730, 233],
            [100, 350, 280, 410],
        ]
        assert [item["index"] for item in handoff["sidecar"]] == [0, 1, 2]
    print("easyocr-interop-contract PASS words=3 lines=2 layoutlm_apply_ocr=False")


if __name__ == "__main__":
    main()
