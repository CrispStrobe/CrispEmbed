#!/usr/bin/env python3
"""Stage the CUDA audit Kaggle kernel at an exact pushed commit.

Kaggle script kernels upload only kernel.py, which clones CrispEmbed and runs
tools/kaggle/cuda-audit/run.py from that checkout. This renders the commit,
refuses an unpushed commit or a leftover placeholder, and writes a new
directory for tools/kaggle/kpush.py.

usage: python3 stage.py NEW_DIRECTORY --batch NAME [--commit <40-hex>]
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True).stdout.strip()


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    target = Path(argv[1]).resolve()
    commit = argv[argv.index("--commit") + 1] if "--commit" in argv else git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit(f"stage: not a 40-hex commit: {commit!r}")
    if not git("branch", "-r", "--contains", commit):
        raise SystemExit("stage: commit is not on any pushed branch; the kernel clones from GitHub")
    batch = argv[argv.index("--batch") + 1] if "--batch" in argv else ""
    if not re.fullmatch(r"[a-z0-9-]+", batch):
        raise SystemExit("stage: --batch NAME required")
    if target.exists():
        raise SystemExit("stage: staging directory already exists")
    rendered = (HERE / "kernel.py").read_text().replace("@COMMIT@", commit).replace("@BATCH@", batch)
    if "@" + "COMMIT@" in rendered or "@" + "BATCH@" in rendered:
        raise SystemExit("stage: unrendered placeholder")
    compile(rendered, "staged-kernel.py", "exec")
    target.mkdir(parents=True)
    (target / "kernel.py").write_text(rendered)
    (target / "kernel-metadata.json").write_bytes((HERE / "kernel-metadata.json").read_bytes())
    manifest = {"script_version": "cuda-audit-kernel-v1", "commit": commit, "batch": batch,
                "run_py_sha256": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest(),
                "files_sha256": {n: hashlib.sha256((target / n).read_bytes()).hexdigest()
                                 for n in ("kernel.py", "kernel-metadata.json")}}
    (target / "source-closure.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
