#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reject malformed TeleOCR metadata through the public native loader.

Run after building the shared library::

    python tests/test_teleocr_loader.py --lib build/libcrispembed.so \
        --report teleocr-loader.json

The fixtures contain metadata/tensor headers only, so no model weights or
network access are needed. Each load runs in a subprocess: an assertion, crash,
or signal in the native parser is a test failure, even if the process prints
the expected error.
The correctly typed metadata control must reach the model-profile check; every
negative must be rejected by its own earlier, specific validation check.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


GATE_ERROR = "TeleOCR requires CRISPEMBED_TELEOCR=1 and a CrispEmbed TeleOCR GGUF v1"
TYPE_ERROR = "TeleOCR: malformed metadata array or conflicting architecture prefix"
PROFILE_ERROR = "TeleOCR: unsupported or inconsistent model metadata"
TENSOR_TYPE_ERROR = "TeleOCR: incompatible tensor type or rank"


def probe(library: str, model: str) -> int:
    lib = ctypes.CDLL(library)
    lib.crispembed_ocr_model_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_ocr_model_init.restype = ctypes.c_void_p
    lib.crispembed_ocr_model_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_ocr_model_free.restype = None
    ctx = lib.crispembed_ocr_model_init(os.fsencode(model), 1)
    if ctx:
        lib.crispembed_ocr_model_free(ctx)
        print("metadata-only fixture unexpectedly loaded", file=sys.stderr)
        return 1
    print("native loader returned null")
    return 0


def cases():
    """Return authored invalid metadata, plus one correctly typed control."""
    import gguf
    kind = gguf.GGUFValueType
    # These are the array contracts that the shared Qwen loader reads. The
    # remaining model-profile metadata is deliberately absent: even the control
    # cannot allocate a backend or weights.
    base = {
        "teleocr.gguf_version": (1, kind.UINT32, None),
        "qwen2vl.vision.fullatt_block_indexes": ([7, 15, 23, 31], kind.ARRAY, kind.UINT32),
        "qwen2vl.vision.image_mean": ([0.48145466, 0.4578275, 0.40821073], kind.ARRAY, kind.FLOAT32),
        "qwen2vl.vision.image_std": ([0.26862954, 0.26130258, 0.27577711], kind.ARRAY, kind.FLOAT32),
        "qwen2vl.rope_sections": ([16, 24, 24], kind.ARRAY, kind.UINT32),
    }
    yield "typed-array-control", base, "1", PROFILE_ERROR
    for label, gate in (("unset", None), ("empty", ""), ("zero", "0")):
        yield f"gate-{label}", base, gate, GATE_ERROR
    # Follow the repository-wide core_env contract: any nonempty value other
    # than the literal "0" enables a gate. "false" is therefore truthy.
    yield "gate-nonempty-false-is-on", base, "false", PROFILE_ERROR
    for name, replacement in (
        ("missing-format-version", None),
        ("unsupported-format-version", (2, kind.UINT32, None)),
        ("wrong-format-version-type", ("1", kind.STRING, None)),
    ):
        metadata = base.copy()
        if replacement is None:
            del metadata["teleocr.gguf_version"]
        else:
            metadata["teleocr.gguf_version"] = replacement
        yield name, metadata, "1", GATE_ERROR
    for key, (values, value_type, subtype) in base.items():
        if value_type != kind.ARRAY:
            continue
        for suffix, replacement in (
            ("missing", None),
            ("scalar", (1, kind.UINT32, None)),
            ("wrong-count", (values[:-1], kind.ARRAY, subtype)),
            ("wrong-element-type", ([str(x) for x in values], kind.ARRAY, kind.STRING)),
        ):
            metadata = base.copy()
            if replacement is None:
                del metadata[key]
            else:
                metadata[key] = replacement
            yield f"{key}-{suffix}", metadata, "1", TYPE_ERROR
    metadata = dict(base, **{"qwen2vl.vision.layer_norm_eps": ("0.000001", kind.STRING, None)})
    yield "wrong-optional-norm-epsilon-type", metadata, "1", TYPE_ERROR
    metadata = dict(base, **{"qwen3vl.hidden_size": (1024, kind.UINT32, None)})
    yield "conflicting-qwen3vl-prefix", metadata, "1", TYPE_ERROR


def tensor_cases():
    """Headers isolate dtype rejection before profile/backend allocation."""
    import gguf
    dtype = gguf.GGMLQuantizationType
    base = next(cases())[1]
    for tensor_name, width in (
        ("v.blk.0.norm1.weight", 1280),
        ("l.blk.0.attn_q_norm.weight", 128),
        ("v.blk.0.attn_qkv.bias", 3840),
    ):
        label = tensor_name.replace(".", "-")
        yield f"{label}-f32-control", base, "1", PROFILE_ERROR, [(tensor_name, (width,), dtype.F32)]
        for kind in (dtype.F16, dtype.Q8_0):
            yield f"{label}-{kind.name}-rejected", base, "1", TENSOR_TYPE_ERROR, [(tensor_name, (width,), kind)]
    for kind in (dtype.F32, dtype.F16, dtype.BF16, dtype.Q8_0, dtype.Q4_K):
        yield f"matrix-{kind.name}-control", base, "1", PROFILE_ERROR, [
            ("l.blk.0.attn_q.weight", (2048, 1024), kind)
        ]
    for kind in (dtype.I32, dtype.F64):
        yield f"matrix-{kind.name}-rejected", base, "1", TENSOR_TYPE_ERROR, [
            ("l.blk.0.attn_q.weight", (2048, 1024), kind)
        ]
    yield "unflattened-patch-rank-rejected", base, "1", TENSOR_TYPE_ERROR, [
        ("v.patch_embed.weight", (1280, 6, 196), dtype.F32)
    ]


def write_fixture(path: Path, metadata, tensors=()):
    import gguf
    import numpy as np
    writer = gguf.GGUFWriter(path, "teleocr")
    try:
        for key, (value, value_type, subtype) in metadata.items():
            writer.add_key_value(key, value, value_type, sub_type=subtype)
        for name, shape, dtype in tensors:
            block_size, type_size = gguf.GGML_QUANT_SIZES[dtype]
            tensor_bytes = int(np.prod(shape)) // block_size * type_size
            writer.add_tensor_info(name, shape, np.dtype(np.float32), tensor_bytes, raw_dtype=dtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
    finally:
        writer.close()
    if tensors:
        # GGUF metadata parsing seeks to the aligned data-section boundary,
        # even with no_alloc=True. No tensor payload is read by this test.
        with path.open("ab") as stream:
            stream.write(b"\x00" * (-stream.tell() % writer.data_alignment))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    library = args.lib.resolve(strict=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="teleocr-loader-") as tmp:
        fixtures = [(*case, ()) for case in cases()] + list(tensor_cases())
        for name, metadata, gate, expected, tensors in fixtures:
            model = Path(tmp) / (name + ".gguf")
            write_fixture(model, metadata, tensors)
            env = os.environ.copy()
            if gate is None:
                env.pop("CRISPEMBED_TELEOCR", None)
            else:
                env["CRISPEMBED_TELEOCR"] = gate
            argv = [sys.executable, str(Path(__file__).resolve()), "--probe", str(library), str(model)]
            try:
                result = subprocess.run(argv, capture_output=True, text=True, timeout=30, env=env)
                row = {"name": name, "gate": gate,
                       "passed": result.returncode == 0 and expected in result.stderr and
                                 "native loader returned null" in result.stdout,
                       "returncode": result.returncode, "expected_diagnostic": expected,
                       "stdout": result.stdout, "stderr": result.stderr}
            except subprocess.TimeoutExpired:
                row = {"name": name, "gate": gate, "passed": False, "error": "native load timed out"}
            row["fixture_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
            rows.append(row)
            print(f"{'PASS' if row['passed'] else 'FAIL'} {name}", flush=True)
    report = {
        "kind": "teleocr-native-loader-metadata-validation",
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
    print(f"{report['passed_count']}/{report['total_count']} native metadata checks passed")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--probe":
        raise SystemExit(probe(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())
