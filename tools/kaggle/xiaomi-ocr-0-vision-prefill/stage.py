#!/usr/bin/env python3
"""Stage the pinned base plus a hash-checked overlay into one Kaggle script.

usage: stage.py NEW_DIRECTORY

Freezes the exact bytes this audit executes: the base commit, the ggml gitlink,
the test-only CMakeLists/target file, the test source and the prefill source
files that are still dirty in the worktree. Refuses an existing target.
"""
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = "15adb672f71f8470a23db1284dafa102515adae9"
GGML = "a06fc0bb46b73cc10011fc90ae4054aee8438aeb"
# Repo-relative destination -> local source. Only files this audit executes.
SOURCES = {
    "audit/CMakeLists.txt": HERE / "CMakeLists.txt",
    "tests/test_xiaomi_vision_prefill.cpp": ROOT / "tests/test_xiaomi_vision_prefill.cpp",
    # Dirty vision chain from the same worktree; the pristine versions of the
    # remaining vision/norm/linear sources come from the pinned base checkout.
    "src/qwen35_vision.cpp": ROOT / "src/qwen35_vision.cpp",
    "src/qwen35_vision.h": ROOT / "src/qwen35_vision.h",
    "src/qwen35_image_bounds.h": ROOT / "src/qwen35_image_bounds.h",
}
MARKER = "__CRISPEMBED_SOURCE_BUNDLE__"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage.py NEW_DIRECTORY")
    target = Path(sys.argv[1]).resolve()
    if target.exists():
        raise ValueError("staging directory already exists")
    for expression, expected in (("HEAD", BASE), ("HEAD:ggml", GGML)):
        actual = subprocess.check_output(["git", "rev-parse", expression], cwd=ROOT, text=True).strip()
        if actual != expected:
            raise ValueError("revision mismatch for " + expression)
    entries = {}
    for destination, source in SOURCES.items():
        data = Path(source).read_bytes()
        entries[destination] = {"sha256": hashlib.sha256(data).hexdigest(),
                                "content": base64.b64encode(data).decode("ascii")}
    template = (HERE / "runner.py").read_bytes()
    package = {"base_commit": BASE, "ggml_commit": GGML,
               "runner_template_sha256": hashlib.sha256(template).hexdigest(), "files": entries}
    encoded = base64.b64encode(zlib.compress(json.dumps(package, sort_keys=True).encode(), 9)).decode()
    text = template.decode().replace(MARKER, encoded, 1)
    if text.count(MARKER) != 1:
        raise ValueError("unexpected marker count")
    compile(text, "staged-runner.py", "exec")
    account = os.environ.get("KAGGLE_USERNAME")
    if not account:
        config = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))) / "kaggle.json"
        account = json.loads(config.read_text())["username"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", account):
        raise ValueError("invalid Kaggle account name")
    metadata = json.loads((HERE / "kernel-metadata.json").read_text().replace("${KAGGLE_ACCOUNT}", account))
    target.mkdir(parents=True)
    (target / "runner.py").write_text(text)
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    closure = {"base_commit": BASE, "ggml_commit": GGML,
               "runner_template_sha256": package["runner_template_sha256"],
               "staged_runner_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "files_sha256": {name: entry["sha256"] for name, entry in entries.items()}}
    (target / "staged-closure.json").write_text(json.dumps(closure, indent=2) + "\n")
    print(json.dumps(closure, indent=2))


if __name__ == "__main__":
    main()
