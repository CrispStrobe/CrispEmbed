#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = ROOT / "tools" / "compare_easyocr_manifests.py"


def main():
    manifest = {
        "schema": "crispembed.easyocr.postprocess.v1",
        "mode": "words",
        "records": [{
            "index": 0, "line": 0, "text": "hello", "confidence": 0.9,
            "detector_confidence": 0.8, "box": [10, 20, 40, 12], "crop": [8, 18, 44, 16],
            "normalized_box": [50, 200, 250, 320],
        }],
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        reference, native, broken = tmp / "reference.json", tmp / "native.json", tmp / "broken.json"
        reference.write_text(json.dumps(manifest), encoding="utf-8")
        native.write_text(json.dumps(manifest), encoding="utf-8")
        broken_manifest = dict(manifest)
        broken_manifest["records"] = [dict(manifest["records"][0], text="hullo")]
        broken.write_text(json.dumps(broken_manifest), encoding="utf-8")
        passed = subprocess.run([sys.executable, str(COMPARATOR), "--reference", str(reference), "--native", str(native)])
        failed = subprocess.run([sys.executable, str(COMPARATOR), "--reference", str(reference), "--native", str(broken)])
    assert passed.returncode == 0
    assert failed.returncode == 1
    print("easyocr-manifest-compare PASS and mismatch detection PASS")


if __name__ == "__main__":
    main()
