#!/usr/bin/env python3
"""Validate and census a Big-LaMa lama-v2 GGUF variant against its F32 source.

INTERNAL RESEARCH ONLY. Parses GGUF v3 headers with the standard library,
checks every tensor against the lama-v2 policy written by crispembed-quantize
(tools/quantize.cpp lama_tensor_plan) and prints a JSON census: per-type counts
and bytes, flattened-storage counts, file size and SHA-256.

    python tools/lama_quant_census.py --source big-lama-f32.gguf \
        --variant big-lama-q8_0.gguf --requested q8_0 [--output census.json]
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

TYPES = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 12: "q4_k", 13: "q5_k",
         14: "q6_k"}
BLOCK = {"f32": (1, 4), "f16": (1, 2), "q4_0": (32, 18), "q4_1": (32, 20), "q5_0": (32, 22), "q5_1": (32, 24),
         "q8_0": (32, 34), "q4_k": (256, 144), "q5_k": (256, 176), "q6_k": (256, 210)}
CRITICAL = {"generator.model.1.ffc.convl2l.weight", "generator.model.34.weight"}
FALLBACK = {"q4_k": "q4_0", "q5_k": "q5_0", "q6_k": "q8_0"}


def read_gguf(path):
    with open(path, "rb") as f:
        def u(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

        def string():
            return f.read(u("Q")).decode("utf-8")

        def value(kind):
            scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
            if kind in scalar:
                return u(scalar[kind])
            if kind == 8:
                return string()
            if kind == 9:
                inner, count = u("I"), u("Q")
                return [value(inner) for _ in range(count)]
            raise ValueError(f"unknown GGUF value type {kind}")

        if f.read(4) != b"GGUF" or u("I") != 3:
            raise ValueError("expected GGUF v3")
        n_tensors, n_kv = u("Q"), u("Q")
        metadata = {}
        for _ in range(n_kv):
            key = string()
            metadata[key] = value(u("I"))
        tensors = {}
        for _ in range(n_tensors):
            name = string()
            dims = [u("Q") for _ in range(u("I"))]
            kind, offset = u("I"), u("Q")
            tensors[name] = {"ne": dims + [1] * (4 - len(dims)), "type": TYPES.get(kind, f"type{kind}"),
                             "offset": offset}
    return metadata, tensors


def expected_plan(name, ne, requested):
    """Mirror of tools/quantize.cpp lama_tensor_plan, policy lama-v2 (no --tensor-type overrides)."""
    if (ne[1] == 1 and ne[2] == 1 and ne[3] == 1) or name in CRITICAL:
        return "f32", list(ne)
    if requested == "f16":
        return "f16", list(ne)
    count, natural = math.prod(ne), ne[0] * ne[1] * ne[2]
    kind = requested
    if natural % BLOCK[kind][0] == 0:
        return kind, [natural, count // natural, 1, 1]
    if BLOCK[kind][0] == 256 and count % 256 == 0:
        return kind, [256, count // 256, 1, 1]
    kind = FALLBACK.get(kind, "q8_0")
    return kind, [natural, count // natural, 1, 1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def census(source, variant, requested):
    source_meta, source_tensors = read_gguf(source)
    meta, tensors = read_gguf(variant)
    errors = []
    if meta.get("lama.quantization.policy") != "lama-v2":
        errors.append("missing lama.quantization.policy=lama-v2")
    for key, value in source_meta.items():
        if key.startswith("lama.") and meta.get(key) != value:
            errors.append(f"metadata changed: {key}")
    if set(tensors) != set(source_tensors):
        errors.append("tensor inventory differs")
    counts, nbytes, flattened, matmul_rows = {}, {}, 0, 0
    for name, row in tensors.items():
        if name not in source_tensors:
            continue
        kind, shape = expected_plan(name, source_tensors[name]["ne"], requested)
        if row["type"] != kind or row["ne"] != shape:
            errors.append(f"{name}: {row['type']}{row['ne']} != planned {kind}{shape}")
        block, size = BLOCK.get(row["type"], (1, 0))
        counts[row["type"]] = counts.get(row["type"], 0) + 1
        nbytes[row["type"]] = nbytes.get(row["type"], 0) + math.prod(row["ne"]) // block * size
        flattened += row["ne"] != source_tensors[name]["ne"]
        matmul_rows += row["type"] != "f32" and (row["ne"] == source_tensors[name]["ne"] or
                                                row["ne"][0] == math.prod(source_tensors[name]["ne"][:3]))
    return {"variant": Path(variant).name, "requested": requested, "bytes": Path(variant).stat().st_size,
            "sha256": sha256(variant), "source_sha256": sha256(source), "tensor_count": len(tensors),
            "type_counts": counts, "type_payload_bytes": nbytes, "flattened_storage_tensors": flattened, "patch_row_kernels": matmul_rows,
            "policy_conformant": not errors, "errors": errors[:20]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--requested", required=True, choices=sorted(set(BLOCK) - {"f32"}))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = census(args.source, args.variant, args.requested)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not report["policy_conformant"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
