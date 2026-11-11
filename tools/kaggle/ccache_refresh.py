#!/usr/bin/env python3
"""Version the ${KAGGLE_ACCOUNT}/crispembed-ccache seed dataset from a finished kernel.

    set -a; . <private env file>; set +a
    python3 tools/kaggle/ccache_refresh.py [--from-kernel <owner/slug or slug>] [--wait]
    python3 tools/kaggle/kpush.py <kernel dir> --refresh-ccache   # push, wait, then this

Any build kernel that imports kaggle_harness from its CrispEmbed clone writes
ccache-export.json (the seed decision) and, only when the seed went stale,
ONE ccache.tar. This downloads just those files (plus summary.json), exits 0
without touching the dataset when the kernel judged the seed still warm,
refuses a near-empty tar, and otherwise creates the private dataset or adds a
version. Kaggle auto-extracts the tar on upload, so it mounts as
<slug>/.ccache/..., which kaggle_harness._warm_ccache_from_dataset() consumes.
The account comes from KAGGLE_ACCOUNT; nothing account-specific is committed.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MIN_TAR_BYTES = 50 << 20
DONE = ("complete", "error", "cancel_requested", "cancel_acknowledged")


def kaggle_env():
    env = dict(os.environ)
    if env.get("KAGGLE_TOKEN") and not env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_API_TOKEN"] = env["KAGGLE_TOKEN"]
    return env


def kaggle(*args):
    return subprocess.run([sys.executable, "-m", "kaggle", *args], env=kaggle_env(), text=True, capture_output=True)


def wait_for_kernel(kernel, poll=60, timeout=6 * 3600):
    """Block until the kernel leaves queued/running; return its final status word."""
    t0 = time.time()
    while True:
        r = kaggle("kernels", "status", kernel)
        text = (r.stdout + r.stderr).lower()
        m = re.search(r'status "(?:kernelworkerstatus\.)?(\w+)', text)
        word = m.group(1) if m else ""
        if word in DONE:
            print(f"kernel {kernel}: {word}", flush=True)
            return word
        if time.time() - t0 > timeout:
            sys.exit(f"timed out waiting for {kernel}: {text.strip()}")
        print(f"waiting for {kernel}: {word or text.strip()[-80:]} ({time.time() - t0:.0f}s)", flush=True)
        time.sleep(poll)


def refresh(kernel, account, slug, dry_run=False):
    with tempfile.TemporaryDirectory(prefix="ccache-refresh-") as tmp:
        out, stage = Path(tmp) / "out", Path(tmp) / "stage"
        out.mkdir()
        stage.mkdir()
        r = kaggle("kernels", "output", kernel, "-p", str(out), "--file-pattern",
                   r"^(ccache\.tar|ccache-export\.json|summary\.json)$")
        decision = {}
        if (out / "ccache-export.json").is_file():
            decision = json.loads((out / "ccache-export.json").read_text())
            print("seed decision:", json.dumps({k: decision.get(k) for k in
                  ("export", "reasons", "hits", "misses", "hit_rate", "age_commits", "age_days")}), flush=True)
        if (out / "summary.json").is_file():
            builds = json.loads((out / "summary.json").read_text()).get("builds", {})
            print(json.dumps({k: {x: v.get(x) for x in ("rc", "total_s", "hits", "misses", "hit_rate")}
                              for k, v in builds.items()}, indent=1), flush=True)
        tar = out / "ccache.tar"
        if not tar.is_file():
            if decision and not decision.get("export"):
                print("seed is current (kernel did not export); dataset unchanged")
                return 0
            print("no ccache.tar and no seed decision in the kernel output:\n" + r.stdout + r.stderr)
            return 1
        if tar.stat().st_size < MIN_TAR_BYTES:
            print(f"ccache.tar is only {tar.stat().st_size} bytes; refusing to publish a near-empty seed")
            return 1
        print(f"ccache.tar {tar.stat().st_size / 1e6:.0f} MB", flush=True)
        if dry_run:
            return 0
        tar.rename(stage / "ccache.tar")
        (stage / "dataset-metadata.json").write_text(json.dumps(
            {"title": slug, "id": f"{account}/{slug}", "licenses": [{"name": "other"}]}))
        exists = kaggle("datasets", "status", f"{account}/{slug}").returncode == 0
        note = f"seed from {kernel}: " + "; ".join(decision.get("reasons") or ["manual"])
        if exists:
            r = kaggle("datasets", "version", "-p", str(stage), "-m", note[:250], "--dir-mode", "tar")
        else:
            r = kaggle("datasets", "create", "-p", str(stage), "--dir-mode", "tar")
        print(r.stdout, r.stderr)
        return r.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-kernel", help="kernel ref (owner/slug or slug); default <account>/crispembed-ccache-seed")
    ap.add_argument("--wait", action="store_true", help="poll until the kernel finishes first")
    ap.add_argument("--dataset", default=os.environ.get("CCACHE_DATASET", "crispembed-ccache"))
    ap.add_argument("--dry-run", action="store_true", help="download and check, do not version")
    a = ap.parse_args(argv)
    account = os.environ.get("KAGGLE_ACCOUNT", "").strip()
    if not account:
        sys.exit("set KAGGLE_ACCOUNT")
    kernel = a.from_kernel or os.environ.get("CCACHE_SEED_KERNEL") or "crispembed-ccache-seed"
    if "/" not in kernel:
        kernel = f"{account}/{kernel}"
    if a.wait and wait_for_kernel(kernel) != "complete":
        print("kernel did not complete; still checking its output for a seed", flush=True)
    return refresh(kernel, account, a.dataset, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
