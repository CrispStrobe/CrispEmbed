#!/usr/bin/env python3
"""Stage the Surya OCR 2.5 convert/quantize Kaggle kernel.

Renders the exact commit into ``kernel.py`` (Kaggle script kernels upload only
the code file, so the kernel clones the repository itself), freezes the runner
bytes and the pinned model digests, and refuses to stage over an existing
directory or with a leftover placeholder.

usage: python3 stage.py NEW_DIRECTORY [--commit <40-hex>]
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PINNED = {
    "model": "datalab-to/surya-ocr-2.5",
    "revision": "00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3",
    "checkpoint_sha256": "6940baca2d16a20b15b7302cac7253246c78c95469920fcdb064963f48b5cb17",
    "checkpoint_bytes": 1378772208,
    "profile": "surya25",
}
FILES = ("kernel.py", "run.py", "kernel-metadata.json")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def head_commit():
    finished = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True)
    return finished.stdout.strip()


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    target = Path(argv[1]).resolve()
    commit = None
    if "--commit" in argv:
        commit = argv[argv.index("--commit") + 1]
    commit = commit or head_commit()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit(f"stage: not a 40-hex commit: {commit!r}")
    if target.exists():
        raise SystemExit("stage: staging directory already exists")

    kernel_text = (HERE / "kernel.py").read_text()
    rendered = kernel_text.replace("@COMMIT@", commit)
    if "@COMMIT@" in rendered:
        raise SystemExit("stage: unrendered @COMMIT@ remains")
    for name in ("@VARIANT@", "@PIN@"):
        if name in rendered:
            raise SystemExit(f"stage: unhandled placeholder {name}")
    compile(rendered, "staged-kernel.py", "exec")
    runner = (HERE / "run.py").read_bytes()
    compile(runner.decode(), "staged-run.py", "exec")

    target.mkdir(parents=True)
    (target / "kernel.py").write_text(rendered)
    (target / "run.py").write_bytes(runner)
    (target / "kernel-metadata.json").write_bytes((HERE / "kernel-metadata.json").read_bytes())
    manifest = {
        "script_version": "surya25-kaggle-quant-v1",
        "commit": commit,
        "pinned": PINNED,
        "arms": ["f16", "q8_0", "q4_k"],
        "anchor_arm": "f16",
        "files_sha256": {name: digest((target / name).read_bytes()) for name in FILES},
    }
    (target / "source-closure.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
