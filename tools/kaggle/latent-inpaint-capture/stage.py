#!/usr/bin/env python3
"""Stage the latent-inpaint original-capture Kaggle kernel for one model.

usage: python3 stage.py NEW_DIRECTORY moebius|pixelhacker [--commit <40-hex>]
Renders @COMMIT@/@MODEL@, refuses an unpushed commit, then push the new
directory with tools/kaggle/kpush.py.
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
    if len(argv) < 3 or argv[2] not in ("moebius", "pixelhacker"):
        raise SystemExit(__doc__)
    target, model = Path(argv[1]).resolve(), argv[2]
    commit = argv[argv.index("--commit") + 1] if "--commit" in argv else git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit(f"stage: not a 40-hex commit: {commit!r}")
    if not git("branch", "-r", "--contains", commit):
        raise SystemExit("stage: commit is not on any pushed branch; the kernel clones from GitHub")
    if target.exists():
        raise SystemExit("stage: staging directory already exists")
    target.mkdir(parents=True)
    for name in ("kernel.py", "kernel-metadata.json"):
        text = (HERE / name).read_text().replace("@COMMIT@", commit).replace("@MODEL@", model)
        if "@COMMIT@" in text or "@MODEL@" in text:
            raise SystemExit("stage: unrendered placeholder")
        (target / name).write_text(text)
    compile((target / "kernel.py").read_text(), "staged-kernel.py", "exec")
    manifest = {"script_version": "latent-capture-v1", "model": model, "commit": commit,
                "run_py_sha256": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()}
    (target / "source-closure.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
