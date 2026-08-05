#!/usr/bin/env python3
"""Guard: every vendored tools/kaggle/*/kaggle_harness.py resolves HF tokens
correctly on BOTH Kaggle dataset mount layouts and fails fast when asked to.

Why this exists (F9b, 2026-08-05): CrispEmbed kernels clone CrispEmbed, so a
fix in CrispASR's canonical harness never reaches them — 15 vendored copies
went stale, and the one in crispembed-imatrix-quant cost a full 21-minute
Kaggle run whose every upload 401'd because the worker mounted the token
dataset only under the LONG layout (/kaggle/input/datasets/<owner>/<slug>/)
that the stale resolver never scanned (T19-E3 run 1). This test re-checks the
behavioral contract against EVERY copy on every CI run, so the next stale
re-vendor fails here instead of on a burned kernel.

Hermetic: stdlib only, no network, no Kaggle, no pytest. Each scenario builds
a fake input tree in a tempdir and points the copy at it via the
KAGGLE_INPUT_ROOT seam (mirrors CrispASR tests/test_kaggle_harness_token.py,
which remains the deeper 13-case suite for the canonical file).
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_ENABLE_HF_TRANSFER")
TOK_SHORT = "hf_short_dummy_token_0123456789"
TOK_LONG = "hf_long_dummy_token_0123456789"

checks = 0
failures = 0


def check(cond: bool, what: str) -> None:
    global checks, failures
    checks += 1
    if not cond:
        failures += 1
        print(f"FAIL {what}")


def load_copy(path: Path):
    """Freshly import one harness copy with a clean HF environment."""
    for var in TOKEN_ENV_VARS:
        os.environ.pop(var, None)
    spec = importlib.util.spec_from_file_location("kh_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kh_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def put_token(root: Path, rel: str, token: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(token + "\n")


def run_scenarios(copy: Path) -> None:
    name = copy.parent.name

    # Signature: fail-fast must be available (F9's require= parameter).
    mod = load_copy(copy)
    sig = inspect.signature(mod.resolve_hf_token)
    check("require" in sig.parameters, f"{name}: resolve_hf_token has require=")

    def with_root(builder):
        """Fresh tempdir root + fresh module import per scenario (the resolver
        exports HF_TOKEN on success, which would leak into the next case)."""
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name) / "kaggle" / "input"
        root.mkdir(parents=True)
        builder(root)
        os.environ["KAGGLE_INPUT_ROOT"] = str(root)
        m = load_copy(copy)
        return tmp, m

    # Short (classic) layout resolves.
    tmp, m = with_root(lambda r: put_token(r, "crispasr-hf-token/hf_token.txt", TOK_SHORT))
    check(m.resolve_hf_token() == TOK_SHORT, f"{name}: short layout resolves")
    tmp.cleanup()

    # Long (nested datasets/<owner>/<slug>) layout resolves — the T19-E3 defect.
    tmp, m = with_root(lambda r: put_token(r, "datasets/chr1s4/crispasr-hf-token/hf_token.txt", TOK_LONG))
    check(m.resolve_hf_token() == TOK_LONG, f"{name}: LONG layout resolves (the t19 defect)")
    tmp.cleanup()

    # Both mounted: short wins (deterministic precedence).
    def both(r):
        put_token(r, "crispasr-hf-token/hf_token.txt", TOK_SHORT)
        put_token(r, "datasets/chr1s4/crispasr-hf-token/hf_token.txt", TOK_LONG)
    tmp, m = with_root(both)
    check(m.resolve_hf_token() == TOK_SHORT, f"{name}: short preferred when both mounted")
    tmp.cleanup()

    # No token anywhere: falsy, and require=True aborts up front (SystemExit),
    # i.e. BEFORE any compute/upload could run.
    tmp, m = with_root(lambda r: None)
    check(not m.resolve_hf_token(), f"{name}: no token -> falsy")
    try:
        m.resolve_hf_token(require=True)
        check(False, f"{name}: require=True raises on missing token")
    except SystemExit:
        check(True, f"{name}: require=True raises on missing token")
    tmp.cleanup()

    # Env var wins over a mounted dataset.
    tmp, m = with_root(lambda r: put_token(r, "crispasr-hf-token/hf_token.txt", TOK_SHORT))
    os.environ["HF_TOKEN"] = "hf_env_dummy_token_0123456789"
    check(m.resolve_hf_token() == "hf_env_dummy_token_0123456789", f"{name}: env HF_TOKEN wins")
    os.environ.pop("HF_TOKEN", None)
    tmp.cleanup()


def main() -> int:
    copies = sorted((REPO / "tools" / "kaggle").glob("*/kaggle_harness.py"))
    check(len(copies) > 0, "at least one vendored copy found")
    print(f"vendored kaggle_harness.py copies: {len(copies)}")
    for copy in copies:
        run_scenarios(copy)
    print(f"vendored-kaggle-harness: {checks} checks, {failures} failure(s) across {len(copies)} copies")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
