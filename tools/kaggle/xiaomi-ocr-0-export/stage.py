#!/usr/bin/env python3
"""Stage a self-contained, hash-recorded Kaggle GPU conversion control.

Usage: python stage.py NEW_SCRATCH_DIRECTORY
The target must not exist. The staged runner embeds the exact current files,
including uncommitted changes; it never reads the source checkout on Kaggle.
"""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCES = ("models/convert-qwen35-ocr-to-gguf.py", "models/qwen35_xiaomi_profile.py",
           "tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json")
PLACEHOLDER = "__CRISPEMBED_SOURCE_BUNDLE__"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]).resolve()
    if target.exists():
        raise SystemExit("Target already exists; choose a new scratch directory")
    files = {}
    for name in SOURCES:
        content = (REPO / name).read_bytes()
        files[name] = {"sha256": hashlib.sha256(content).hexdigest(),
                       "content": base64.b64encode(content).decode("ascii")}
    runner = (HERE / "runner.py").read_bytes()
    package = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
               "runner_template_sha256": hashlib.sha256(runner).hexdigest(), "files": files}
    bundle = base64.b64encode(zlib.compress(json.dumps(package, sort_keys=True).encode(), 9)).decode("ascii")
    text = runner.decode().replace(PLACEHOLDER, bundle, 1)
    if text.count(PLACEHOLDER) != 1:
        raise ValueError("Unexpected source bundle placeholders")
    compile(text, "staged-runner.py", "exec")
    target.mkdir(parents=True)
    (target / "runner.py").write_text(text)
    (target / "kernel-metadata.json").write_bytes((HERE / "kernel-metadata.json").read_bytes())
    print(json.dumps({"base_commit": package["base_commit"],
                      "runner_template_sha256": package["runner_template_sha256"],
                      "staged_runner_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "files_sha256": {name: row["sha256"] for name, row in files.items()}}, indent=2))


if __name__ == "__main__":
    main()
