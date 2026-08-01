#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "compare_easyocr_tsv.py"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        manifest = tmp / "native.json"
        tsv = tmp / "tesseract.tsv"
        manifest.write_text(json.dumps({"records": [{"text": "Hello", "line": 0, "box": [10, 20, 30, 12]}]}))
        tsv.write_text(
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t11\t21\t30\t12\t95\tHello\n"
        )
        result = subprocess.run(
            [sys.executable, str(TOOL), "--native", str(manifest), "--tesseract-tsv", str(tsv)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or "PASS" not in result.stdout:
            print(result.stdout, result.stderr, file=sys.stderr)
            return 1
    print("easyocr-tsv-compare self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
