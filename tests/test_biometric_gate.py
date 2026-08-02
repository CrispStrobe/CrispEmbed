#!/usr/bin/env python3
"""Regression test for the biometric acknowledgement gate.

POLICY.md §4 tells integrators that loading a face *recognition* model
requires an explicit acknowledgement, and that detection is never gated.
That sentence is the one a reader relies on, and before this test nothing
checked it: the gate lived only in the CLI for a while, and every binding
was open. A refactor of ``crispembed_face_init()`` could remove it again
without breaking anything else.

What is asserted here is the contract as documented, not an implementation
detail:

  * a detection model loads with no acknowledgement (a box is not a template)
  * a recognition model is refused without one
  * ``CRISPEMBED_ACCEPT_BIOMETRIC=1`` satisfies the gate
  * ``crispembed.accept_biometric_use()`` satisfies the gate
  * a *renamed* recognition model is still refused — the gate keys on the
    model's declared type, not on its filename
  * the CLI refuses non-interactively and proceeds with ``--accept-biometric``

The acknowledgement is a process-wide latch that cannot be cleared, so each
case runs in its own subprocess.

This is a speed bump and an audit trail, not a security control — the check
is trivially removable from MIT-licensed code. The test exists so that its
removal is deliberate rather than accidental.

Models (a detection and a recognition GGUF) are found via, in order:
``--det-model`` / ``--rec-model``, ``CRISPEMBED_DET_MODEL`` /
``CRISPEMBED_REC_MODEL``, or a ``yunet.gguf`` / ``sface.gguf`` in
``CRISPEMBED_GGUF_DIR``. Without them the test reports an explicit skip
rather than silently claiming to have run.

Usage:
    python tests/test_biometric_gate.py
    python tests/test_biometric_gate.py --det-model yunet.gguf --rec-model sface.gguf
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"


def _resolve(explicit: str | None, env_var: str, default_name: str) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    from_env = os.environ.get(env_var)
    if from_env and Path(from_env).is_file():
        return Path(from_env)
    gguf_dir = os.environ.get("CRISPEMBED_GGUF_DIR")
    if gguf_dir:
        p = Path(gguf_dir) / default_name
        if p.is_file():
            return p
    return None


def _run_py(snippet: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter with the repo bindings importable."""
    full_env = dict(os.environ)
    full_env.pop("CRISPEMBED_ACCEPT_BIOMETRIC", None)
    full_env.update(env)
    full_env["PYTHONPATH"] = os.pathsep.join(
        [str(PYTHON_DIR), full_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", snippet],
        check=False,
        text=True,
        capture_output=True,
    env=full_env,
    )


_LOAD = """
from crispembed._binding import CrispFace
try:
    CrispFace({path!r})
    print("LOADED")
except Exception:
    print("RAISED")
"""


def _load_result(model: Path, env: dict[str, str], pre: str = "") -> str:
    """Return LOADED / REFUSED, distinguishing a gate refusal from any other error.

    A bare "the constructor raised" is not enough: a missing library or a
    corrupt GGUF would also raise, and scoring that as REFUSED would let a
    removed gate pass this test. The gate's own stderr banner is the signal.
    """
    snippet = pre + _LOAD.format(path=str(model))
    proc = _run_py(snippet, env)
    outcome = None
    for line in reversed(proc.stdout.splitlines()):
        if line.strip() in ("LOADED", "RAISED"):
            outcome = line.strip()
            break
    gate_banner = "is a FACE RECOGNITION model" in proc.stderr
    if outcome == "LOADED":
        return "LOADED"
    if outcome == "RAISED":
        return "REFUSED" if gate_banner else "RAISED-NOT-BY-GATE"
    return f"ERROR(rc={proc.returncode}): {proc.stderr.strip()[-200:]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-model")
    ap.add_argument("--rec-model")
    ap.add_argument("--build-dir", default=str(ROOT / "build"))
    args = ap.parse_args()

    det = _resolve(args.det_model, "CRISPEMBED_DET_MODEL", "yunet.gguf")
    rec = _resolve(args.rec_model, "CRISPEMBED_REC_MODEL", "sface.gguf")
    if not det or not rec:
        print(
            "SKIP: biometric gate test needs a detection and a recognition GGUF.\n"
            "      Pass --det-model/--rec-model, set CRISPEMBED_DET_MODEL/"
            "CRISPEMBED_REC_MODEL,\n"
            "      or put yunet.gguf and sface.gguf in CRISPEMBED_GGUF_DIR.\n"
            "      The gate was NOT exercised."
        )
        return 0

    # A build that cannot be loaded at all is a missing prerequisite, not a
    # failed gate. Say so plainly rather than reporting every case as broken.
    probe = _run_py("import crispembed._binding as b; b._load_library()\nprint('LIB-OK')", {})
    if "LIB-OK" not in probe.stdout:
        print(
            "SKIP: libcrispembed could not be loaded; the gate was NOT exercised.\n"
            f"      {probe.stderr.strip()[-300:]}"
        )
        return 0

    failures: list[str] = []

    def check(name: str, got: str, want: str) -> None:
        ok = got == want
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {got}")
        if not ok:
            failures.append(f"{name}: expected {want}, got {got}")

    print("Python binding:")
    check("detection loads unacknowledged", _load_result(det, {}), "LOADED")
    check("recognition refused unacknowledged", _load_result(rec, {}), "REFUSED")
    check(
        "recognition loads with CRISPEMBED_ACCEPT_BIOMETRIC=1",
        _load_result(rec, {"CRISPEMBED_ACCEPT_BIOMETRIC": "1"}),
        "LOADED",
    )
    check(
        "recognition loads after accept_biometric_use()",
        _load_result(rec, {}, pre="import crispembed\ncrispembed.accept_biometric_use()\n"),
        "LOADED",
    )

    # The gate must key on the model's declared type, not its filename:
    # renaming a recognition model must not get it past the check.
    with tempfile.TemporaryDirectory() as td:
        disguised = Path(td) / "definitely_a_detector.gguf"
        shutil.copy(rec, disguised)
        check(
            "renamed recognition model still refused",
            _load_result(disguised, {}),
            "REFUSED",
        )

    cli = Path(args.build_dir) / "crispembed"
    if not cli.is_file():
        print(f"\nSKIP: CLI not built at {cli}; CLI cases not exercised.")
        return _report(failures)

    try:
        from PIL import Image
    except ImportError:
        print("\nSKIP: PIL unavailable; CLI cases not exercised.")
        return _report(failures)

    print("CLI:")
    with tempfile.TemporaryDirectory() as td:
        img = Path(td) / "blank.png"
        Image.new("RGB", (320, 320), (128, 128, 128)).save(img)
        env = dict(os.environ)
        env.pop("CRISPEMBED_ACCEPT_BIOMETRIC", None)

        def run_cli(extra: list[str], model: Path) -> str:
            """Return REFUSED/LOADED based on the gate's own message.

            Deliberately not keyed on the exit code alone: the CLI can exit
            non-zero for reasons that have nothing to do with the gate (a
            model that finds no faces, a backend hiccup), and reading those
            as "refused" would both mask a removed gate and produce flaky
            failures. The refusal text is what the gate itself emits.
            """
            proc = subprocess.run(
                [str(cli), "-m", str(model), "--detect", str(img), *extra],
                check=False,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                env=env,
            )
            refused = "refusing to run a face recognition model" in proc.stderr
            if refused and proc.returncode == 0:
                return "REFUSED-BUT-EXIT-0"  # fails closed in message only
            return "REFUSED" if refused else "LOADED"

        # Non-interactive stdin: the CLI must fail closed rather than hang or
        # silently proceed.
        check("recognition refused non-interactively", run_cli([], rec), "REFUSED")
        check(
            "recognition proceeds with --accept-biometric",
            run_cli(["--accept-biometric"], rec),
            "LOADED",
        )
        check("detection ungated", run_cli([], det), "LOADED")

    return _report(failures)


def _report(failures: list[str]) -> int:
    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        print("\nThe biometric gate does not behave as POLICY.md §4 describes.")
        return 1
    print("PASS: biometric gate matches POLICY.md §4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
