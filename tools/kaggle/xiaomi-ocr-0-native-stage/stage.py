#!/usr/bin/env python3
"""Stage pinned base plus an explicit, hash-checked dirty-worktree overlay."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = "15adb672f71f8470a23db1284dafa102515adae9"
GGML = "a06fc0bb46b73cc10011fc90ae4054aee8438aeb"
SOURCES = (
    "CMakeLists.txt", "src/qwen35_frontend.h", "src/qwen35_ocr.cpp",
    "src/qwen35_tokenizer.cpp", "src/xiaomi_profile_pin.h",
    "tests/test_xiaomi_ocr.cpp", "tests/test_xiaomi_patch_embed.cpp",
    "tests/test_xiaomi_patch_embed_cuda.cpp", "models/verify-gguf-export.py",
    "tests/results/xiaomi-ocr-0/kaggle-export-v2-manifest.json",
)
MARKER = "__CRISPEMBED_SOURCE_BUNDLE__"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage.py NEW_DIRECTORY")
    target = Path(sys.argv[1]).resolve()
    if target.exists():
        raise ValueError("staging directory already exists")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != BASE:
        raise ValueError("unexpected base commit")
    gitlink = subprocess.check_output(["git", "rev-parse", "HEAD:ggml"], cwd=ROOT, text=True).strip()
    if gitlink != GGML:
        raise ValueError("unexpected ggml gitlink")
    entries = {}
    for name in SOURCES:
        data = (ROOT / name).read_bytes()
        entries[name] = {"sha256": hashlib.sha256(data).hexdigest(),
                         "content": base64.b64encode(data).decode("ascii")}
    template = (HERE / "runner.py").read_bytes()
    package = {"base_commit": BASE, "ggml_commit": GGML,
               "runner_template_sha256": hashlib.sha256(template).hexdigest(), "files": entries}
    encoded = base64.b64encode(zlib.compress(json.dumps(package, sort_keys=True).encode(), 9)).decode()
    text = template.decode().replace(MARKER, encoded, 1)
    if text.count(MARKER) != 1:
        raise ValueError("unexpected marker count")
    compile(text, "staged-runner.py", "exec")
    target.mkdir(parents=True)
    (target / "runner.py").write_text(text)
    (target / "kernel-metadata.json").write_bytes((HERE / "kernel-metadata.json").read_bytes())
    print(json.dumps({"base_commit": BASE, "ggml_commit": GGML,
                      "runner_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "files_sha256": {name: value["sha256"] for name, value in entries.items()}}, indent=2))


if __name__ == "__main__":
    main()
