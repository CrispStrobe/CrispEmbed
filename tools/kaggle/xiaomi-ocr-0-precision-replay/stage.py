#!/usr/bin/env python3
"""Stage the self-contained Xiaomi precision-replay kernel.

The runner is a single file with no repository source dependency: it downloads
the pinned checkpoint, reads the two mounted GGUF inputs directly, and imports
the pinned upstream transformers module. Staging therefore freezes the exact
runner bytes and the pinned artifact digests, and nothing else.

usage: python3 stage.py NEW_DIRECTORY
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE_COMMIT = "15adb672f71f8470a23db1284dafa102515adae9"
PINNED = {
    "model_repo": "xiaomi-research/Xiaomi-OCR-0",
    "revision": "bbd8cf8d21f256b3644b44cb860f3a7cecb93c30",
    "checkpoint_sha256": "d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348",
    "export_sha256": "1759b8da2a6fa2651de9cacdb5e90fe0551d7ab8b1fe783198f7339389451235",
    "reference_sha256": "2a58fc3fc2a4ae2d603ff600371cf71fa1f54e40ebd260d7680bfa4c1f868334",
    "transformers": "5.8.1",
    "tokenizers": "0.22.2",
    "gguf": "0.19.0",
    "safetensors": "0.8.0",
}


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]).resolve()
    if target.exists():
        raise ValueError("staging directory already exists")
    runner = (HERE / "runner.py").read_bytes()
    compile(runner.decode(), "staged-runner.py", "exec")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                          capture_output=True).stdout.strip() or None
    manifest = {
        "script_version_line": [line for line in runner.decode().splitlines()
                                if line.startswith("SCRIPT_VERSION")][0],
        "base_commit": BASE_COMMIT,
        "staging_head": head,
        "pinned": PINNED,
        "runner_template_sha256": hashlib.sha256(runner).hexdigest(),
        "staged_runner_sha256": hashlib.sha256(runner).hexdigest(),
        "files_sha256": {"runner.py": hashlib.sha256(runner).hexdigest()},
    }
    target.mkdir(parents=True)
    (target / "runner.py").write_bytes(runner)
    (target / "kernel-metadata.json").write_bytes((HERE / "kernel-metadata.json").read_bytes())
    (target / "source-closure.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
