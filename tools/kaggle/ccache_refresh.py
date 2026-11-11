#!/usr/bin/env python3
"""Refresh the ${KAGGLE_ACCOUNT}/crispembed-ccache seed dataset from a finished seed run.

    set -a; . <private env file>; set +a
    python3 tools/kaggle/kpush.py <rendered crispembed-ccache-seed dir>   # (see docs/kaggle-ccache.md)
    python3 tools/kaggle/ccache_refresh.py            # after the kernel completes

Downloads ONLY ccache.tar from the kernel output (the kernel keeps its cache on
/kaggle/temp, so the output holds a handful of files), stages it alone and
creates the dataset on first use or adds a new version. Kaggle auto-extracts
the tar on upload, so it mounts as <slug>/.ccache/..., which
kaggle_harness._warm_ccache_from_dataset() consumes. The account comes from
KAGGLE_ACCOUNT; nothing account-specific is committed.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

account = os.environ.get("KAGGLE_ACCOUNT", "").strip()
if not account:
    sys.exit("set KAGGLE_ACCOUNT")
kernel = os.environ.get("CCACHE_SEED_KERNEL", f"{account}/crispembed-ccache-seed")
slug = os.environ.get("CCACHE_DATASET", "crispembed-ccache")
env = dict(os.environ)
if env.get("KAGGLE_TOKEN") and not env.get("KAGGLE_API_TOKEN"):
    env["KAGGLE_API_TOKEN"] = env["KAGGLE_TOKEN"]


def kaggle(*args):
    return subprocess.run([sys.executable, "-m", "kaggle", *args], env=env, text=True, capture_output=True)


with tempfile.TemporaryDirectory(prefix="ccache-refresh-") as tmp:
    out, stage = Path(tmp) / "out", Path(tmp) / "stage"
    out.mkdir()
    stage.mkdir()
    r = kaggle("kernels", "output", kernel, "-p", str(out), "--file-pattern", r"^(ccache\.tar|summary\.json)$")
    tar = out / "ccache.tar"
    if not tar.is_file():
        sys.exit("no ccache.tar in the kernel output:\n" + r.stdout + r.stderr)
    if tar.stat().st_size < 50 << 20:
        sys.exit(f"ccache.tar is only {tar.stat().st_size} bytes; refusing to publish a near-empty seed")
    if (out / "summary.json").is_file():
        print(json.dumps(json.loads((out / "summary.json").read_text()).get("builds", {}), indent=1))
    tar.rename(stage / "ccache.tar")
    (stage / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": f"{account}/{slug}", "licenses": [{"name": "other"}]}))
    exists = kaggle("datasets", "status", f"{account}/{slug}").returncode == 0
    if exists:
        r = kaggle("datasets", "version", "-p", str(stage), "-m", f"seed from {kernel}", "--dir-mode", "tar")
    else:
        r = kaggle("datasets", "create", "-p", str(stage), "--dir-mode", "tar")
    print(r.stdout, r.stderr)
    sys.exit(r.returncode)
