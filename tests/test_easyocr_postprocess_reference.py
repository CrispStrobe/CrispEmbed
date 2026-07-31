#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "easyocr_postprocess_reference.py"


def main():
    source = {
        "image": "fixture.png",
        "width": 200,
        "height": 100,
        "items": [
            {"box": [[100, 20], [130, 20], [130, 32], [100, 32]], "text": "world", "confidence": 0.8},
            {"box": {"x": 10, "y": 21, "w": 40, "h": 12}, "text": "hello", "confidence": 0.9},
            {"box": [15, 65, 60, 75], "text": "next", "confidence": 0.7},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source_path, lines_path, words_path = tmp / "source.json", tmp / "lines.json", tmp / "words.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        for mode, output in (("lines", lines_path), ("words", words_path)):
            subprocess.run([sys.executable, str(SCRIPT), "--input", str(source_path), "--output", str(output),
                            "--mode", mode], check=True)
        lines = json.loads(lines_path.read_text(encoding="utf-8"))
        words = json.loads(words_path.read_text(encoding="utf-8"))
    assert lines["text"] == "hello world\nnext"
    assert len(lines["records"]) == 2
    assert lines["records"][0]["crop"] == [8, 18, 124, 17]
    assert lines["records"][0]["normalized_box"] == [50, 200, 650, 330]
    assert len(words["records"]) == 3
    assert [r["text"] for r in words["records"]] == ["hello", "world", "next"]
    assert words["records"][0]["normalized_box"] == [50, 210, 250, 330]
    print("easyocr-postprocess-reference PASS lines=2 words=3")


if __name__ == "__main__":
    main()
