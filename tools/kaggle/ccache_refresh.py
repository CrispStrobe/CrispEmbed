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
from kpush import service_result

MIN_TAR_BYTES = 50 << 20
DONE = ("complete", "error", "cancel_requested", "cancel_acknowledged")


def explicit_dataset_missing(result):
    """Accept a canonical service 404, not an identifier or incidental phrase."""
    canonical = re.compile(
        r"(?:HTTP(?:Error)?[: ]*|HTTP/\d(?:\.\d)?\s+)?404(?:\s*[-:]\s*|\s+)"
        r"(?:Client Error:\s*)?Not Found"
        r"(?:\s+for url:\s+https://(?:www\.)?kaggle\.com/api/v1/datasets/status/[^\s]+)?\.?",
        re.I,
    )
    for line in (result.stdout, result.stderr, *result.stdout.splitlines(), *result.stderr.splitlines()):
        if canonical.fullmatch(line.strip()):
            return True
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        error = payload.get("error", payload)
        if not isinstance(error, dict):
            continue
        code = error.get("status", error.get("statusCode", error.get("code")))
        message = error.get("message")
        if code in (404, "404") and isinstance(message, str) and message.strip().lower() in (
                "not found", "dataset not found", "dataset does not exist"):
            return True
    return False


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
        if r.returncode or re.search(r"push error|error:", text, re.I):
            raise SystemExit(service_result(r, "ccache-refresh: status query"))
        if word in DONE:
            print(f"ccache-refresh: kernel status {word}.", flush=True)
            return word
        if time.time() - t0 > timeout:
            sys.exit("ccache-refresh: timed out waiting for kernel completion")
        status = word if word in ("queued", "running", "pending") else "unrecognized"
        print(f"ccache-refresh: waiting ({status}; {time.time() - t0:.0f}s).", flush=True)
        time.sleep(poll)


def refresh(kernel, account, slug, dry_run=False):
    with tempfile.TemporaryDirectory(prefix="ccache-refresh-") as tmp:
        out, stage = Path(tmp) / "out", Path(tmp) / "stage"
        out.mkdir()
        stage.mkdir()
        r = kaggle("kernels", "output", kernel, "-p", str(out), "--file-pattern",
                   r"^(ccache\.tar|ccache-export\.json|summary\.json)$")
        rc = service_result(r, "ccache-refresh: output retrieval")
        if rc:
            return rc
        decision = {}
        if (out / "ccache-export.json").is_file():
            decision = json.loads((out / "ccache-export.json").read_text())
            if not isinstance(decision, dict):
                print("ccache-refresh: invalid seed decision.", file=sys.stderr)
                return 1
            print("ccache-refresh: seed decision received.", flush=True)
        if (out / "summary.json").is_file():
            summary = json.loads((out / "summary.json").read_text())
            if not isinstance(summary, dict):
                print("ccache-refresh: invalid build summary.", file=sys.stderr)
                return 1
            builds = summary.get("builds", {})
            if not isinstance(builds, dict) or any(not isinstance(v, dict) for v in builds.values()):
                print("ccache-refresh: invalid build summary.", file=sys.stderr)
                return 1
            print("ccache-refresh: build summary received.", flush=True)
        tar = out / "ccache.tar"
        if not tar.is_file():
            if decision and not decision.get("export"):
                print("seed is current (kernel did not export); dataset unchanged")
                return 0
            print("ccache-refresh: required cache artifact or seed decision is missing.", file=sys.stderr)
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
        status = kaggle("datasets", "status", f"{account}/{slug}")
        status_text = status.stdout + status.stderr
        if re.search(r"unauthorized|forbidden|authentication failed|permission denied|"
                     r"timed out|connection (?:error|failed)|network (?:error|failed)|rate limit|too many requests|"
                     r"(?:HTTP(?:Error)?|status(?: code)?)[^\n]{0,16}\b(?:40[13]|429|5\d\d)\b|"
                     r"^\s*(?:40[13]|429|5\d\d)(?:\s|[-:]|$)", status_text, re.I | re.M):
            # A contradictory permission/network error never admits creation,
            # even if the CLI exits zero or also mentions a missing resource.
            return service_result(status, "ccache-refresh: dataset status", force_failure=True)
        if explicit_dataset_missing(status):
            # Some CLI versions report a missing dataset while exiting zero.
            exists = False
        elif status.returncode or re.search(r"push error|error:", status_text, re.I):
            # Only a confirmed missing dataset admits creation. Authentication,
            # network and ambiguous failures must not trigger an upload attempt.
            return service_result(status, "ccache-refresh: dataset status")
        else:
            exists = True
        reasons = decision.get("reasons") or ["manual"]
        if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
            print("ccache-refresh: invalid seed reasons.", file=sys.stderr)
            return 1
        note = "Refresh compiler cache seed"
        if exists:
            r = kaggle("datasets", "version", "-p", str(stage), "-m", note[:250], "--dir-mode", "tar")
        else:
            r = kaggle("datasets", "create", "-p", str(stage), "--dir-mode", "tar")
        return service_result(r, "ccache-refresh: dataset update")


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
    try:
        if a.wait and wait_for_kernel(kernel) != "complete":
            print("kernel did not complete; still checking its output for a seed", flush=True)
        return refresh(kernel, account, a.dataset, a.dry_run)
    except (OSError, ValueError, TypeError):
        print("ccache-refresh: invalid cache metadata or local I/O failure.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
