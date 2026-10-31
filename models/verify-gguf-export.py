#!/usr/bin/env python3
"""Verify a GGUF export against its converter manifest, independent of gguf-py.

Reads the container header directly and requires, for every manifest record:
the same tensor set, the same element count, an F32 payload whose SHA-256 equals
the manifest digest, and shapes equal up to ggml's axis reversal (ggml stores
ne[0] first, so a row-major source matrix appears with reversed extents).

Usage: python models/verify-gguf-export.py MODEL.gguf MODEL.gguf.manifest.json
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
          7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
ALIGNMENT = 32
HASH_CHUNK_BYTES = 1024 * 1024


class Reader:
    def __init__(self, path):
        self.stream = Path(path).open("rb")
        if self.stream.read(4) != b"GGUF":
            raise ValueError("Not a GGUF container")
        self.version, = struct.unpack("<I", self.stream.read(4))
        if self.version not in (2, 3):
            raise ValueError("Unsupported GGUF version")
        count, kv_count = struct.unpack("<QQ", self.stream.read(16))
        self.metadata = {}
        for _ in range(kv_count):
            key = self.string()
            self.metadata[key] = self.value(struct.unpack("<I", self.stream.read(4))[0])
        self.tensors = []
        for _ in range(count):
            name = self.string()
            axes, = struct.unpack("<I", self.stream.read(4))
            shape = [struct.unpack("<Q", self.stream.read(8))[0] for _ in range(axes)]
            kind, offset = struct.unpack("<IQ", self.stream.read(12))
            self.tensors.append((name, shape, kind, offset))
        self.data = (self.stream.tell() + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    def string(self):
        length, = struct.unpack("<Q", self.stream.read(8))
        return self.stream.read(length).decode("utf-8")

    def value(self, kind):
        if kind in SCALAR:
            return struct.unpack(SCALAR[kind], self.stream.read(struct.calcsize(SCALAR[kind])))[0]
        if kind == 8:
            return self.string()
        if kind == 9:
            element, count = struct.unpack("<IQ", self.stream.read(12))
            if element not in SCALAR and element != 8:
                raise ValueError(f"Unsupported GGUF array element type {element}")
            return [self.value(element) for _ in range(count)]
        raise ValueError(f"Unsupported GGUF value type {kind}")

    def payload_sha256(self, shape, offset):
        count = 1
        for axis in shape:
            count *= axis
        self.stream.seek(self.data + offset)
        remaining = count * 4
        digest = hashlib.sha256()
        while remaining:
            chunk = self.stream.read(min(remaining, HASH_CHUNK_BYTES))
            if not chunk:
                raise ValueError("truncated GGUF tensor payload")
            digest.update(chunk)
            remaining -= len(chunk)
        return digest.hexdigest()

    def close(self):
        self.stream.close()


def verify(gguf_path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    reader = Reader(gguf_path)
    try:
        records = {row["name"]: row for row in manifest["records"]}
        tensors = {name: (shape, kind, offset) for name, shape, kind, offset in reader.tensors}
        if len(records) != len(manifest["records"]) or len(tensors) != len(reader.tensors):
            raise ValueError("duplicate tensor name in manifest or GGUF")
        if len(records) != manifest["export_tensor_count"]:
            raise ValueError("Manifest record count disagrees with its own total")
        if set(records) != set(tensors):
            raise ValueError(f"Tensor set differs: missing={sorted(set(records) - set(tensors))}, "
                             f"extra={sorted(set(tensors) - set(records))}")
        for name, record in records.items():
            shape, kind, offset = tensors[name]
            if kind != 0:
                raise ValueError(f"{name}: expected F32 payload, found type {kind}")
            if record["dtype"] != "float32":
                raise ValueError(f"{name}: manifest dtype is {record['dtype']}")
            elements = 1
            for axis in shape:
                elements *= axis
            if elements != int(math.prod(record["shape"])):
                raise ValueError(f"{name}: element count differs from the manifest shape")
            if shape != list(reversed(record["shape"])):
                raise ValueError(f"{name}: GGUF shape {shape} is not the reversed source shape {record['shape']}")
            digest = reader.payload_sha256(shape, offset)
            if digest != record["sha256"]:
                raise ValueError(f"{name}: payload digest differs")
    finally:
        reader.close()
    return {"tensors": len(records), "bytes": Path(gguf_path).stat().st_size,
            "reversed_axes": sum(1 for name, shape, _, _ in reader.tensors
                                 if shape != records[name]["shape"]),
            "metadata": {key: value for key, value in reader.metadata.items()
                         if key.startswith("qwen35_ocr.") and key in
                         ("qwen35_ocr.format_version", "qwen35_ocr.profile",
                          "qwen35_ocr.stop_token_ids", "qwen35_ocr.primary_eos_id",
                          "qwen35_ocr.checkpoint_sha256", "qwen35_ocr.excluded_source_tensors")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf")
    parser.add_argument("manifest")
    args = parser.parse_args()
    report = verify(args.gguf, args.manifest)
    report["excluded_source_tensor_count"] = len(
        report["metadata"].get("qwen35_ocr.excluded_source_tensors", []))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
