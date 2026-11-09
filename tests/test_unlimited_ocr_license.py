#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the native Jina/CC-BY-NC gate without model weights or a network.

Run against the built shared library::

    python tests/test_unlimited_ocr_license.py --lib build/libcrispembed.so \
        --report build/unlimited-ocr-license.json

Each fixture contains only GGUF metadata. Restricted fixtures must stop at the
license check; acknowledged and permissive controls must reach the distinct
zero-tensor check. Both checks precede model context/backend/weight allocation.
The actual public resolver is used for existing local paths and cached aliases;
direct cases call the public OCR loader without the resolver. Each probe runs
in a subprocess, so a native crash, assertion or timeout is a failure.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ACK_ENV = "CRISPEMBED_ACCEPT_CC_BY_NC_LICENSE"
LICENSE_ERROR = "Set CRISPEMBED_ACCEPT_CC_BY_NC_LICENSE=1 to acknowledge"
TYPE_ERROR = "unlimited_ocr: malformed license/identity metadata:"
TENSOR_ERROR = "unlimited_ocr: GGUF contains no model tensors"
OFF_VALUES = (
    ("unset", None), ("empty", ""), ("zero", "0"),
    ("false", "false"), ("off", "off"), ("no", "no"),
    ("mixed-false", "FaLsE"), ("mixed-off", "OfF"), ("mixed-no", "nO"),
)
JINA = {
    "general.name": "jina-ocr-v1",
    "general.source": "https://huggingface.co/jinaai/jina-ocr-v1",
    "general.license": "cc-by-nc-4.0",
}
UNLIMITED = {
    "general.name": "unlimited-ocr",
    "general.source": "https://huggingface.co/baidu/Unlimited-OCR",
    "general.license": "MIT",
}
IDENTITY_FIELDS = (
    "general.name", "general.source", "general.url", "general.source.url",
    "general.source.huggingface.repository",
)
LICENSE_FIELDS = ("general.license", "general.license.name", "general.license.link")


@dataclass(frozen=True)
class Case:
    name: str
    metadata: dict[str, str | int]
    expected: str = LICENSE_ERROR
    acknowledgement: str | None = None
    # "direct" bypasses resolution; "local" resolves the existing file path;
    # all other values are real registry names/filenames resolved from cache.
    surface: str = "direct"
    download_acknowledgement: str | None = None


def cases():
    for surface in ("direct", "local", "jina-ocr-v1", "jina-ocr-v1-q4_k.gguf", "jina-ocr-v1-q8"):
        for label, value in OFF_VALUES:
            yield Case(f"{surface}-ack-{label}", JINA, acknowledgement=value, surface=surface)
        yield Case(f"{surface}-acknowledged", JINA, TENSOR_ERROR, "1", surface)
    yield Case("ack-mixed-true", JINA, TENSOR_ERROR, "TrUe")
    yield Case("download-acceptance-is-not-native-ack", JINA, download_acknowledgement="all")

    # Every retained identity field independently prevents a misleading MIT
    # tag or a renamed filename from suppressing the Jina acknowledgement.
    for key in IDENTITY_FIELDS:
        metadata = dict(UNLIMITED)
        metadata[key] = ("JiNa OCR v1 (Q8_0)" if key == "general.name" else
                         "https://huggingface.co/jinaai/jina-ocr-v1/resolve/revision/model.gguf")
        yield Case(f"identity-{key}-overrides-mit", metadata)
        yield Case(f"identity-{key}-acknowledged", metadata, TENSOR_ERROR, "1")
    yield Case("name-only-missing-license", {"general.name": "jina-ocr-v1"})
    yield Case("rehost-source-only", {"general.source": "https://huggingface.co/cstr/jina-ocr-v1-GGUF"})
    yield Case("mirror-claim-still-restricted", {
        "general.name": "unlimited-ocr", "general.license": "MIT",
        "general.source": "https://example.invalid/mirror/jina-ocr-v1/model.gguf",
    })

    # License identity survives renaming. Normalized words cover SPDX labels,
    # customary display names and the canonical Creative Commons license URI.
    for i, value in enumerate((
        "cc-by-nc-4.0", "CC-BY-NC-4.0", "CC BY NC 4.0", "cc_by_nc_4_0",
        "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0", "MIT OR cc-by-nc-4.0",
        "https://creativecommons.org/licenses/by-nc/4.0/",
        "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode",
        "Creative Commons Attribution-NonCommercial 4.0 International",
        "Creative Commons Attribution Non Commercial 4.0 International",
    )):
        metadata = dict(UNLIMITED, **{"general.license": value})
        yield Case(f"renamed-noncommercial-{i}", metadata)
    for key in LICENSE_FIELDS:
        metadata = {"general.name": "renamed-export", "general.license": "other", key: "cc-by-nc-4.0"}
        yield Case(f"license-field-{key}", metadata)
        yield Case(f"license-field-{key}-acknowledged", metadata, TENSOR_ERROR, "1")

    # Non-string fields must fail closed before any getter with a type
    # assertion. The acknowledgement permits loading, not malformed metadata.
    for key in (*IDENTITY_FIELDS, *LICENSE_FIELDS):
        for label, value in (("unset", None), ("accepted", "1")):
            metadata = dict(UNLIMITED)
            metadata[key] = 7
            yield Case(f"malformed-{key}-{label}", metadata, TYPE_ERROR, value)
    yield Case("license-before-hparams", dict(JINA, **{"unlimited_ocr.sam.num_heads": 0}))

    # The shared engine must retain permissive Unlimited and legacy metadata
    # behavior. These fixtures prove that no acknowledgement is requested;
    # they deliberately contain no tensor headers, so cannot load a model.
    for name, metadata in (
        ("permissive-unlimited-mit", UNLIMITED),
        ("permissive-unlimited2-mit", {
            "general.name": "unlimited-ocr2", "general.license": "MIT",
            "general.source": "https://huggingface.co/baidu/Unlimited-OCR-2", "unlimited_ocr.variant": 2,
        }),
        ("permissive-apache", dict(UNLIMITED, **{"general.license": "Apache-2.0"})),
        ("permissive-cc-by", dict(UNLIMITED, **{"general.license": "cc-by-4.0"})),
        ("legacy-missing-identity-license", {}),
        ("different-version-token", {"general.name": "jina-ocr-v10", "general.license": "MIT"}),
        ("different-name-token", {"general.name": "notjina-ocr-v1", "general.license": "MIT"}),
    ):
        yield Case(name, metadata, TENSOR_ERROR)


def write_fixture(path: Path, metadata: dict[str, str | int]) -> None:
    import gguf
    writer = gguf.GGUFWriter(path, "unlimited_ocr")
    try:
        for key, value in metadata.items():
            kind = gguf.GGUFValueType.STRING if isinstance(value, str) else gguf.GGUFValueType.UINT32
            writer.add_key_value(key, value, kind)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
    finally:
        writer.close()


def probe(library: str, model: str, surface: str) -> int:
    lib = ctypes.CDLL(library)
    lib.crispembed_ocr_model_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_ocr_model_init.restype = ctypes.c_void_p
    lib.crispembed_ocr_model_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_ocr_model_free.restype = None
    path = os.fsencode(model)
    if surface != "direct":
        lib.crispembed_set_offline.argtypes = [ctypes.c_int]
        lib.crispembed_set_offline.restype = None
        lib.crispembed_set_offline(1)
        lib.crispembed_resolve_model.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.crispembed_resolve_model.restype = ctypes.c_char_p
        query = path if surface == "local" else os.fsencode(surface)
        resolved = lib.crispembed_resolve_model(query, 0)
        if not resolved or Path(os.fsdecode(resolved)).resolve() != Path(model).resolve():
            print("resolver did not return the expected fixture", file=sys.stderr)
            return 1
        path = resolved
        print("resolved expected fixture")
    ctx = lib.crispembed_ocr_model_init(path, 1)
    if ctx:
        lib.crispembed_ocr_model_free(ctx)
        print("metadata-only fixture unexpectedly loaded", file=sys.stderr)
        return 1
    print("native loader returned null")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    library = args.lib.resolve(strict=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="unlimited-ocr-license-") as tmp:
        for i, case in enumerate(cases()):
            cache = Path(tmp) / str(i) / "cache"
            cache.mkdir(parents=True)
            filename = "jina-ocr-v1-q8_0.gguf" if case.surface == "jina-ocr-v1-q8" else "jina-ocr-v1-q4_k.gguf"
            model = (cache / filename if case.surface not in ("direct", "local") else
                     cache.parent / "renamed-model.gguf")
            write_fixture(model, case.metadata)
            env = dict(os.environ, CRISPEMBED_CACHE_DIR=str(cache), CRISPEMBED_OFFLINE="1",
                       HF_HUB_OFFLINE="1", CRISPEMBED_RAM_GUARD_AVAILABLE_MB="4096")
            for key, value in ((ACK_ENV, case.acknowledgement),
                               ("CRISPEMBED_ACCEPT_LICENSE", case.download_acknowledgement)):
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
            argv = [sys.executable, str(Path(__file__).resolve()), "--probe", str(library), str(model), case.surface]
            row = {"name": case.name, "surface": case.surface,
                   "license_acknowledgement": case.acknowledgement,
                   "download_acknowledgement": case.download_acknowledgement,
                   "expected_diagnostic": case.expected,
                   "fixture_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
            try:
                result = subprocess.run(argv, capture_output=True, text=True, timeout=30, env=env)
                forbidden = [message for message in (LICENSE_ERROR, TYPE_ERROR, TENSOR_ERROR)
                             if message != case.expected]
                passed = (result.returncode == 0 and case.expected in result.stderr and
                          not any(message in result.stderr for message in forbidden) and
                          "native loader returned null" in result.stdout and
                          (case.surface == "direct" or "resolved expected fixture" in result.stdout))
                row.update(passed=passed, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            except subprocess.TimeoutExpired:
                row.update(passed=False, error="native load timed out")
            rows.append(row)
            print(f"{'PASS' if row['passed'] else 'FAIL'} {case.name}", flush=True)
    report = {
        "kind": "unlimited-ocr-native-noncommercial-license-gate",
        "model_weights_used": False,
        "model_parity_claimed": False,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "checks": rows,
        "passed": all(row["passed"] for row in rows),
        "passed_count": sum(row["passed"] for row in rows),
        "total_count": len(rows),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['passed_count']}/{report['total_count']} native license checks passed")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--probe":
        raise SystemExit(probe(sys.argv[2], sys.argv[3], sys.argv[4]))
    raise SystemExit(main())
