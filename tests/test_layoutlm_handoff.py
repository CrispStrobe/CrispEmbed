#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "validate_layoutlm_handoff.py"


def main():
    manifest = {
        "schema": "crispembed.easyocr.postprocess.v1",
        "image": "fixture.png",
        "width": 200,
        "height": 100,
        "mode": "words",
        "records": [
            {"index": 0, "text": "hello", "confidence": 0.9, "box": [10, 20, 40, 12], "normalized_box": [50, 200, 250, 320]},
            {"index": 1, "text": "world", "confidence": 0.8, "box": [100, 20, 30, 12], "normalized_box": [500, 200, 650, 320]},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tmp / "manifest.json"
        source.write_text(json.dumps(manifest), encoding="utf-8")
        result = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(source)], text=True,
                                capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    handoff = json.loads(result.stdout)
    assert handoff["apply_ocr"] is False
    assert handoff["processor_args"] == {
        "words": ["hello", "world"],
        "boxes": [[50, 200, 250, 320], [500, 200, 650, 320]],
    }
    assert handoff["sidecar"][1]["confidence"] == 0.8
    print("layoutlm-handoff PASS apply_ocr=False words=2")


if __name__ == "__main__":
    main()
