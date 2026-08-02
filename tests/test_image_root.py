#!/usr/bin/env python3
"""Assert that crispembed-server's --image-root actually confines request paths.

POLICY.md §4 tells deployers to set --image-root whenever the port is not
loopback-only, on the grounds that endpoints read images by server-side path
and /face turns any readable file into a biometric template. That promise is
only worth making if the confinement resists the obvious escapes, so this
exercises them against a live server:

  in-root          -> served
  absolute outside -> refused
  ../ traversal    -> refused
  symlink out      -> refused
  sibling prefix   -> refused   (/srv/scansEVIL vs /srv/scans — the bug a
                                 naive startswith() check would ship)

Refusal is deliberately indistinguishable from "no image path" in the HTTP
response, so an unauthenticated caller cannot probe for file existence; the
reason goes to the server's stderr instead, and this asserts on both.

Usage:
    python tests/test_image_root.py --det-model yunet.gguf [--build-dir build]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port: int, path: str, payload: dict, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode())
        except Exception:
            return {"error": f"http {e.code}"}


def wait_ready(proc: subprocess.Popen, port: int, deadline_s: float = 90.0) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-model", help="face detection GGUF (e.g. yunet.gguf)")
    ap.add_argument("--build-dir", default=str(ROOT / "build"))
    args = ap.parse_args()

    server = Path(args.build_dir) / "crispembed-server"
    if not server.exists():
        print(f"SKIP: {server} not built; confinement was NOT exercised.")
        return 0

    det = args.det_model or os.environ.get("CRISPEMBED_DET_MODEL")
    if not det and os.environ.get("CRISPEMBED_GGUF_DIR"):
        cand = Path(os.environ["CRISPEMBED_GGUF_DIR"]) / "yunet.gguf"
        if cand.exists():
            det = str(cand)
    if not det or not Path(det).exists():
        print("SKIP: image-root test needs a detection GGUF.\n"
              "      Pass --det-model, set CRISPEMBED_DET_MODEL, or put yunet.gguf\n"
              "      in CRISPEMBED_GGUF_DIR. Confinement was NOT exercised.")
        return 0

    sample = ROOT / "tests" / "regression" / "images" / "face.png"
    if not sample.exists():
        print(f"SKIP: fixture {sample} missing; confinement was NOT exercised.")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="crispembed_imgroot_"))
    try:
        root = tmp / "root"
        outside = tmp / "outside"
        sibling = tmp / "rootEVIL"  # shares the string prefix of `root`
        for d in (root, outside, sibling):
            d.mkdir()
        shutil.copy(sample, root / "ok.png")
        shutil.copy(sample, outside / "secret.png")
        shutil.copy(sample, sibling / "x.png")
        os.symlink(outside / "secret.png", root / "link.png")

        port = free_port()
        proc = subprocess.Popen(
            [str(server), "--det", det, "--image-root", str(root),
             "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "CRISPEMBED_ACCEPT_BIOMETRIC": "1"},
        )
        try:
            if not wait_ready(proc, port):
                out, err = proc.communicate(timeout=10)
                print("SKIP: server did not start; confinement was NOT exercised.")
                print((err or out or "")[-800:])
                return 0

            cases = [
                ("in-root path served", str(root / "ok.png"), True),
                ("absolute path outside root refused", str(outside / "secret.png"), False),
                ("../ traversal refused", str(root / ".." / "outside" / "secret.png"), False),
                ("symlink out of root refused", str(root / "link.png"), False),
                ("sibling prefix dir refused", str(sibling / "x.png"), False),
            ]

            failures = 0
            for label, image, should_serve in cases:
                body = post(port, "/detect", {"image": image})
                served = "faces" in body
                ok = served == should_serve
                failures += not ok
                print(f"  [{'ok' if ok else 'FAIL'}] {label}: "
                      f"{'SERVED' if served else 'REFUSED'}")
        finally:
            proc.terminate()
            try:
                _, err = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                _, err = proc.communicate()

        # The rejections must be attributable in the server log, or a deployer
        # has no way to tell confinement from a malformed request.
        logged = (err or "").count("rejected image path outside --image-root")
        expected_rejections = sum(1 for _, _, serve in cases if not serve)
        if logged != expected_rejections:
            print(f"  [FAIL] server logged {logged} rejections, expected {expected_rejections}")
            failures += 1
        else:
            print(f"  [ok] all {logged} rejections logged with a reason")

        if failures:
            print(f"\nFAIL: {failures} image-root check(s) failed.")
            return 1
        print("\nPASS: --image-root confines request paths (POLICY.md §4).")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
