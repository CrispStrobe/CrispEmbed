#!/usr/bin/env python3
"""Frozen latent-inpainting identities, inert Torch ZIP reader and streamed GGUF.

Adapted from CrispEmbed's prepare_migan_inpaint and migan_reference_contract.
No framework or network import occurs here. Checkpoint SHA-256 is checked before
pickle metadata is interpreted. Only a fixed allow-list of harmless tensor
metadata constructors is accepted; storage data is streamed from the ZIP.
"""

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import re
import struct
import sys
import zipfile

from migan_reference_contract import sha256

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/latent-inpaint-source-contract.json"
NATIVE_SCHEMA = ROOT / "src/latent_inpaint_schema.h"
MODELS = ("moebius", "pixelhacker", "sdxl_vae")
CASES = ("subtitle_bar", "edge_touch", "center", "empty")
RESOLUTION = 512
MAX_PICKLE_BYTES = 1024 * 1024
MAX_ARCHIVE_BYTES = 4_500_000_000
MAX_MEMBERS = 5000
BLOCK_BYTES = 256 * 1024
DTYPE_BYTES = {"float32": 4, "float16": 2, "int64": 8}
TENSOR_NAME_ENCODING = "sha256-56-v1"


def gguf_tensor_name(original):
    """Fit the pinned ggml 64-byte name field without losing source identity.

    Source/schema/reference names remain unchanged. The physical GGUF name is
    deterministic, and the receipt retains both spellings. The complete census
    is collision-checked before any output is opened; a short source name that
    happens to equal another tensor's derived alias is also rejected.
    """
    if not isinstance(original, str) or not original or "\0" in original:
        raise ValueError("invalid original tensor name")
    raw = original.encode("utf-8")
    return original if len(raw) < 64 else "t." + hashlib.sha256(raw).hexdigest()[:56]


def _tensor_name_map(rows, metadata):
    encoding = metadata.get("crispembed.tensor_name_encoding")
    if encoding not in (None, TENSOR_NAME_ENCODING):
        raise ValueError("unsupported GGUF tensor name encoding")
    names, physical = {}, set()
    for row in rows:
        original = row["name"]
        derived = gguf_tensor_name(original)
        name = derived if encoding else original
        if original in names or name in physical:
            raise ValueError("tensor name alias collision or duplicate source name")
        # An omitted inert buffer is not a GGUF tensor. Legacy callers such as
        # LaMa keep their existing receipt fields and short generator names.
        if row["export_shape"] is not None and len(name.encode("utf-8")) >= 64:
            raise ValueError("long tensor name requires explicit GGUF name encoding")
        names[original] = name
        physical.add(name)
    return names


def load_contract():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("schema") != "crispembed-latent-inpaint-source-v1" or contract.get("version") != 1:
        raise ValueError("unsupported latent-inpainting source contract")
    if set(contract["models"]) != set(MODELS):
        raise ValueError("model inventory differs")
    for model in MODELS:
        record = contract["models"][model]
        names = [row["name"] for row in record["state"]]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("tensor schema must be uniquely sorted")
        for row in record["state"]:
            if row["dtype"] not in DTYPE_BYTES:
                raise ValueError("unsupported original tensor dtype")
            if row["export_shape"] != export_shape(row["name"], row["shape"], row["dtype"]):
                raise ValueError("unapproved tensor-layout transformation")
        if record["original_tensor_count"] != len(names):
            raise ValueError("original tensor count differs")
        if record["export_tensor_count"] != sum(row["export_shape"] is not None for row in record["state"]):
            raise ValueError("export tensor count differs")
        _tensor_name_map(record["state"], {"crispembed.tensor_name_encoding": TENSOR_NAME_ENCODING})
    return contract


def export_shape(name, shape, dtype):
    shape = list(shape)
    if any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("invalid exported tensor shape")
    if dtype == "int64":
        if not name.endswith(".num_batches_tracked") or shape:
            raise ValueError("only scalar inference-inert BN tracking counters may be omitted")
        return None
    if len(shape) == 5:
        if not name.endswith(".attn1.pos_conv.weight") or shape[1:3] != [1, 1] or shape[3:] != [15, 15]:
            raise ValueError("unsupported rank-five tensor")
        return shape[:2] + shape[3:]
    if len(shape) > 4:
        raise ValueError("invalid exported tensor shape")
    return shape or [1]


def model_record(model, contract=None):
    if model not in MODELS:
        raise ValueError("unsupported latent-inpainting model")
    return (contract or load_contract())["models"][model]


def verify_checkpoint(path, record):
    path = Path(path)
    expected = record["checkpoint"]
    if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise ValueError("checkpoint byte count or SHA-256 differs from the pinned official artifact")


def verify_source(source_dir, model, contract=None):
    contract = contract or load_contract()
    record = contract["source"][model_record(model, contract)["source_model"]]
    for name, expected in record["files"].items():
        path = Path(source_dir) / name
        if path.is_symlink() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"original source identity differs: {name}")
    return record


class _FloatStorage:
    pass


class _HalfStorage:
    pass


class _LongStorage:
    pass


STORAGE_TYPES = {_FloatStorage: "float32", _HalfStorage: "float16", _LongStorage: "int64"}


@dataclass(frozen=True)
class Storage:
    key: str
    count: int
    dtype: str


@dataclass(frozen=True)
class Tensor:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple


def _rebuild_tensor(storage, offset, shape, stride, *extras):
    if (not isinstance(storage, Storage) or type(offset) is not int or offset < 0
            or not isinstance(shape, (tuple, list)) or not isinstance(stride, (tuple, list))
            or len(shape) != len(stride) or len(shape) > 5
            or any(type(n) is not int or n <= 0 for n in shape)
            or any(type(n) is not int or n < 0 for n in stride)
            or len(extras) not in (0, 2, 3)):
        raise ValueError("unsupported tensor metadata")
    if extras and (type(extras[0]) is not bool or not isinstance(extras[1], (dict, OrderedDict)) or extras[1]):
        raise ValueError("unsupported tensor gradient/hook metadata")
    if len(extras) == 3 and extras[2] not in (None, {}):
        raise ValueError("unsupported conjugate/negative tensor metadata")
    return Tensor(storage, offset, tuple(shape), tuple(stride))


class StateReader(pickle.Unpickler):
    """Allow-list parser: no arbitrary pickle callable, import or class is used."""

    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch", "FloatStorage"): _FloatStorage,
            ("torch", "HalfStorage"): _HalfStorage,
            ("torch", "LongStorage"): _LongStorage,
            ("torch._utils", "_rebuild_tensor"): _rebuild_tensor,
            ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor,
        }
        if (module, name) not in allowed:
            raise ValueError(f"unsupported pickle global: {module}.{name}")
        return allowed[(module, name)]

    def persistent_load(self, value):
        if (not isinstance(value, tuple) or len(value) != 5 or value[0] != "storage"
                or value[1] not in STORAGE_TYPES or not isinstance(value[2], str)
                or re.fullmatch(r"[0-9]+", value[2]) is None
                or not isinstance(value[3], str) or len(value[3]) > 128
                or type(value[4]) is not int or not 0 < value[4] <= MAX_ARCHIVE_BYTES // DTYPE_BYTES[STORAGE_TYPES[value[1]]]):
            raise ValueError("unsupported persistent storage")
        return Storage(value[2], value[4], STORAGE_TYPES[value[1]])


def read_state_pickle(payload):
    if len(payload) > MAX_PICKLE_BYTES:
        raise ValueError("checkpoint pickle exceeds the bounded metadata size")
    stream = io.BytesIO(payload)
    state = StateReader(stream).load()
    if stream.read(1):
        raise ValueError("trailing pickle data")
    if not isinstance(state, (dict, OrderedDict)) or any(not isinstance(name, str) or not isinstance(t, Tensor) for name, t in state.items()):
        raise ValueError("expected a plain tensor state dictionary")
    return state


def validate_state(state, rows):
    expected = {row["name"]: row for row in rows}
    if len(expected) != len(rows) or set(state) != set(expected):
        raise ValueError("checkpoint tensor names differ from the frozen census")
    for name, tensor in state.items():
        row = expected[name]
        observed = {"shape": list(tensor.shape), "stride": list(tensor.stride), "offset": tensor.offset,
                    "storage": tensor.storage.key, "storage_count": tensor.storage.count, "dtype": tensor.storage.dtype}
        if any(observed[key] != row[key] for key in observed):
            raise ValueError(f"checkpoint tensor shape/type/storage differs: {name}")
        count = 1
        for dimension, stride in zip(reversed(tensor.shape), reversed(tensor.stride)):
            if dimension > 1 and stride != count:
                raise ValueError("the pinned profiles require contiguous tensor storage")
            count *= dimension
        if tensor.offset != 0 or count != tensor.storage.count:
            raise ValueError("the pinned profiles require complete unaliased storage")
    if len({t.storage.key for t in state.values()}) != len(state):
        raise ValueError("shared checkpoint storage is not supported by this fixed profile")


class Checkpoint:
    """Bounded, validated PyTorch ZIP; actual payload reads remain streamed."""

    def __init__(self, path, record, verify=True):
        if verify:
            verify_checkpoint(path, record)
        self.archive = zipfile.ZipFile(path)
        try:
            members = self.archive.infolist()
            names = [item.filename for item in members]
            if (len(names) != len(set(names)) or len(names) > MAX_MEMBERS
                    or sum(item.file_size for item in members) > MAX_ARCHIVE_BYTES
                    or any(item.compress_type != zipfile.ZIP_STORED for item in members)):
                raise ValueError("unsupported checkpoint ZIP inventory/compression")
            pickles = [name for name in names if name.endswith("/data.pkl")]
            if len(pickles) != 1 or self.archive.getinfo(pickles[0]).file_size > MAX_PICKLE_BYTES:
                raise ValueError("expected exactly one bounded PyTorch data.pkl")
            self.prefix = pickles[0][:-len("data.pkl")]
            if not re.fullmatch(r"[A-Za-z0-9_-]+/", self.prefix):
                raise ValueError("unsupported checkpoint ZIP prefix")
            if self.prefix + "byteorder" in names and self.archive.read(self.prefix + "byteorder") != b"little":
                raise ValueError("only little-endian original checkpoints are supported")
            payload = self.archive.read(pickles[0])
            cp = record["checkpoint"]
            if len(payload) != cp["pickle_bytes"] or hashlib.sha256(payload).hexdigest() != cp["pickle_sha256"]:
                raise ValueError("checkpoint metadata differs from the frozen original census")
            self.state = read_state_pickle(payload)
            validate_state(self.state, record["state"])
            expected_members = {self.prefix + "data/" + t.storage.key for t in self.state.values()}
            actual_members = {name for name in names if name.startswith(self.prefix + "data/")}
            if actual_members != expected_members:
                raise ValueError("checkpoint storage member inventory differs")
            for tensor in self.state.values():
                expected_size = tensor.storage.count * DTYPE_BYTES[tensor.storage.dtype]
                if self.archive.getinfo(self.prefix + "data/" + tensor.storage.key).file_size != expected_size:
                    raise ValueError("checkpoint storage byte count differs")
        except BaseException:
            self.archive.close()
            raise

    def close(self):
        self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def f32_blocks(self, name):
        tensor = self.state[name]
        if tensor.storage.dtype not in ("float32", "float16"):
            raise ValueError("non-floating tensor cannot be exported as a weight")
        size = DTYPE_BYTES[tensor.storage.dtype]
        with self.archive.open(self.prefix + "data/" + tensor.storage.key) as stream:
            while True:
                block = stream.read(BLOCK_BYTES)
                if not block:
                    break
                if len(block) % size:
                    raise ValueError("truncated checkpoint storage")
                fmt = "<f" if size == 4 else "<e"
                values = [item[0] for item in struct.iter_unpack(fmt, block)]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"nonfinite checkpoint tensor: {name}")
                yield block if size == 4 else struct.pack("<" + "f" * len(values), *values)

    def validate_omitted(self, rows):
        omitted = []
        for row in rows:
            if row["export_shape"] is not None:
                continue
            tensor = self.state[row["name"]]
            raw = self.archive.read(self.prefix + "data/" + tensor.storage.key)
            if len(raw) != 8 or struct.unpack("<q", raw)[0] < 0:
                raise ValueError("invalid omitted BN batch counter")
            omitted.append({"name": row["name"], "dtype": "int64", "shape": [],
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "reason": "BatchNorm evaluation uses running_mean and running_var, not this training counter"})
        return omitted


def gguf_metadata(model, contract=None):
    contract = contract or load_contract()
    record = model_record(model, contract)
    source = contract["source"][record["source_model"]]
    return {
        "general.architecture": model,
        "general.name": model + "-places2-experimental-f32" if model != "sdxl_vae" else "sdxl-vae-experimental-f32",
        "general.file_type": 0,
        "general.license": record["license"],
        "crispembed.tensor_name_encoding": TENSOR_NAME_ENCODING,
        "latent_inpaint.schema": 1,
        "latent_inpaint.component": record["component"],
        "latent_inpaint.model": model,
        "latent_inpaint.image_size": RESOLUTION,
        "latent_inpaint.source.commit": source["commit"],
        "latent_inpaint.checkpoint.sha256": record["checkpoint"]["sha256"],
        "latent_inpaint.checkpoint.revision": record["checkpoint"]["revision"],
        "latent_inpaint.contract.sha256": sha256(CONTRACT),
        "latent_inpaint.tensor.layout": "original contiguous; GGUF dimensions reversed; pos_conv singleton depth squeezed; BN counters omitted",
        "latent_inpaint.input.contract": "RGB8;binary-mask8;255=fill;fixed512",
    }


def _string(stream, value):
    raw = value.encode("utf-8")
    stream.write(struct.pack("<Q", len(raw)))
    stream.write(raw)


def write_streamed_gguf(path, checkpoint, rows, metadata):
    """Write one F32 tensor at a time; no multi-GB in-memory state dictionary."""
    names = _tensor_name_map(rows, metadata)
    encoded = metadata.get("crispembed.tensor_name_encoding") == TENSOR_NAME_ENCODING
    exported = [row for row in rows if row["export_shape"] is not None]
    records = []
    with Path(path).open("xb") as stream:
        stream.write(struct.pack("<IIQQ", 0x46554747, 3, len(exported), len(metadata)))
        for key, value in metadata.items():
            _string(stream, key)
            if isinstance(value, str):
                stream.write(struct.pack("<I", 8))
                _string(stream, value)
            elif type(value) is bool:
                stream.write(struct.pack("<IB", 7, int(value)))
            elif type(value) is int and 0 <= value < 2**32:
                stream.write(struct.pack("<II", 4, value))
            else:
                raise ValueError("unsupported GGUF metadata value")
        offset = 0
        for row in exported:
            shape = row["export_shape"]
            size = math.prod(shape) * 4
            _string(stream, names[row["name"]])
            stream.write(struct.pack("<I", len(shape)))
            stream.write(struct.pack("<" + "Q" * len(shape), *reversed(shape)))
            stream.write(struct.pack("<IQ", 0, offset))
            offset += (size + 31) & ~31
        stream.write(b"\0" * (-stream.tell() % 32))
        for row in exported:
            digest, count = hashlib.sha256(), 0
            for block in checkpoint.f32_blocks(row["name"]):
                stream.write(block)
                digest.update(block)
                count += len(block)
            if count != math.prod(row["export_shape"]) * 4:
                raise ValueError("exported tensor byte count differs")
            stream.write(b"\0" * (-count % 32))
            record = {"name": names[row["name"]], "shape": row["export_shape"], "source_shape": row["shape"],
                      "source_dtype": row["dtype"], "dtype": "float32", "bytes": count,
                      "sha256": digest.hexdigest(), "transformation": row["export"]}
            if encoded:
                record["original_name"] = row["name"]
            records.append(record)
    return records


def read_gguf_index(path):
    """Small strict header reader; payloads are verified separately in chunks."""
    with Path(path).open("rb") as stream:
        def unpack(fmt):
            raw = stream.read(struct.calcsize(fmt))
            if len(raw) != struct.calcsize(fmt):
                raise ValueError("truncated GGUF header")
            return struct.unpack(fmt, raw)

        def string():
            size = unpack("<Q")[0]
            if size > MAX_PICKLE_BYTES:
                raise ValueError("oversized GGUF string")
            raw = stream.read(size)
            if len(raw) != size:
                raise ValueError("truncated GGUF string")
            return raw.decode("utf-8")

        magic, version, count, nmeta = unpack("<IIQQ")
        if (magic, version) != (0x46554747, 3) or count > MAX_MEMBERS or nmeta > 100:
            raise ValueError("unsupported GGUF header")
        metadata = {}
        for _ in range(nmeta):
            key, kind = string(), unpack("<I")[0]
            if key in metadata or kind not in (4, 7, 8):
                raise ValueError("duplicate or unsupported GGUF metadata")
            if kind == 7:
                value = unpack("<B")[0]
                if value not in (0, 1):
                    raise ValueError("invalid GGUF boolean")
                metadata[key] = bool(value)
            else:
                metadata[key] = string() if kind == 8 else unpack("<I")[0]
        rows, names = [], set()
        expected_offset = 0
        for _ in range(count):
            name, ndim = string(), unpack("<I")[0]
            if name in names or not 1 <= ndim <= 4:
                raise ValueError("duplicate GGUF tensor or invalid rank")
            names.add(name)
            dims = unpack("<" + "Q" * ndim)
            kind, offset = unpack("<IQ")
            if kind != 0 or offset != expected_offset or any(n <= 0 for n in dims):
                raise ValueError("non-F32/noncontiguous/invalid GGUF tensor")
            size = math.prod(dims) * 4
            expected_offset += (size + 31) & ~31
            rows.append({"name": name, "shape": list(reversed(dims)), "offset": offset, "bytes": size})
        data_offset = (stream.tell() + 31) & ~31
    if Path(path).stat().st_size != data_offset + expected_offset:
        raise ValueError("GGUF size or trailing bytes differ")
    return metadata, rows, data_offset


def verify_gguf(path, metadata, records):
    encoding = metadata.get("crispembed.tensor_name_encoding")
    if encoding not in (None, TENSOR_NAME_ENCODING):
        raise ValueError("unsupported GGUF tensor name encoding")
    original_names, physical_names = set(), set()
    for record in records:
        original = record.get("original_name") if encoding else record["name"]
        derived = gguf_tensor_name(original)
        expected_name = derived if encoding else original
        if (record["name"] != expected_name or len(expected_name.encode("utf-8")) >= 64
                or original in original_names or expected_name in physical_names):
            raise ValueError("GGUF tensor alias/source identity differs or collides")
        original_names.add(original)
        physical_names.add(expected_name)
    actual_meta, actual, start = read_gguf_index(path)
    if (actual_meta != metadata or len(actual) != len(records)
            or any(type(actual_meta[key]) is not type(value) for key, value in metadata.items())):
        raise ValueError("GGUF metadata or tensor count differs")
    with Path(path).open("rb") as stream:
        for observed, expected in zip(actual, records):
            if any(observed[key] != expected[key] for key in ("name", "shape", "bytes")):
                raise ValueError("GGUF tensor index differs")
            stream.seek(start + observed["offset"])
            digest, remaining = hashlib.sha256(), observed["bytes"]
            while remaining:
                block = stream.read(min(BLOCK_BYTES, remaining))
                if not block:
                    raise ValueError("truncated GGUF payload")
                digest.update(block)
                remaining -= len(block)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError("GGUF tensor readback hash differs")


def fixture(name):
    if name not in CASES:
        raise ValueError("unknown latent-inpainting fixture")
    rgb, mask = bytearray(RESOLUTION * RESOLUTION * 3), bytearray(RESOLUTION * RESOLUTION)
    for y in range(RESOLUTION):
        for x in range(RESOLUTION):
            index = y * RESOLUTION + x
            rgb[3 * index:3 * index + 3] = bytes(((3 * x + y) % 256, (x + 2 * y) % 256,
                                               (x // 16 * 23 + y // 16 * 11) % 256))
            hole = ((name == "subtitle_bar" and 48 <= x < 464 and 416 <= y < 480)
                    or (name == "edge_touch" and x < 70 and 96 <= y < 416)
                    or (name == "center" and 176 <= x < 336 and 176 <= y < 336))
            mask[index] = 255 if hole else 0
            if name == "subtitle_bar" and hole:
                glyph = 432 <= y < 462 and 72 <= x < 440 and x % 26 < 16 and (y % 14 < 8 or x % 26 < 4)
                rgb[3 * index:3 * index + 3] = bytes((235, 235, 235) if glyph else (24, 24, 24))
    return bytes(rgb), bytes(mask)


def fixture_identity(name):
    rgb, mask = fixture(name)
    return {"id": name, "width": RESOLUTION, "height": RESOLUTION,
            "rgb_sha256": hashlib.sha256(rgb).hexdigest(), "mask_sha256": hashlib.sha256(mask).hexdigest(),
            "fill_pixels": mask.count(255)}


def native_schema_text():
    contract = load_contract()
    lines = ["// Generated from the independently frozen original checkpoint metadata census.",
             "// Regenerate with: python tools/latent_inpaint_contract.py --write-native-schema",
             "// Inert BN batch counters are excluded; Conv3d singleton depth is explicitly squeezed.",
             "// clang-format off",
             "#pragma once", '#include "latent_inpaint_ops.h"', "#include <cstdint>", "#include <stdexcept>", "",
             "namespace latent_inpaint {", "",
             'inline const char * source_contract_sha256() { return "' + sha256(CONTRACT) + '"; }', ""]
    for function, field in (("source_commit", "source"), ("checkpoint_sha256", "sha256"),
                            ("checkpoint_revision", "revision"),
                            ("license", "license"), ("checkpoint_bytes", "bytes")):
        result_type = "uint64_t" if field == "bytes" else "const char *"
        lines.append("inline " + result_type + " " + function + "(const std::string & model) {")
        for model in MODELS:
            record = contract["models"][model]
            if field == "source":
                value = contract["source"][record["source_model"]]["commit"]
            elif field == "license":
                value = record[field]
            else:
                value = record["checkpoint"][field]
            literal = str(value) + "ULL" if field == "bytes" else '"' + value + '"'
            lines.append('    if (model == "' + model + '") return ' + literal + ';')
        lines.extend(['    throw std::invalid_argument("unsupported latent inpainting model");', "}", ""])
    lines.extend(['inline const char * tensor_layout() { return "' + gguf_metadata("moebius", contract)["latent_inpaint.tensor.layout"] + '"; }', ""])
    lines.extend(['inline const char * tensor_name_encoding() { return "' + TENSOR_NAME_ENCODING + '"; }', ""])
    lines.extend(['inline const char * input_contract() { return "' + gguf_metadata("moebius", contract)["latent_inpaint.input.contract"] + '"; }', ""])
    for model in MODELS:
        lines.extend(["inline ShapeMap " + model + "_schema() {", "    return {"])
        for row in contract["models"][model]["state"]:
            if row["export_shape"] is not None:
                shape = ", ".join(str(n) for n in row["export_shape"])
                lines.append('        {"' + row["name"] + '", {' + shape + '}},')
        lines.extend(["    };", "}", ""])
    lines.extend(['inline ShapeMap denoiser_schema(const std::string & model) {',
                  '    if (model == "moebius") return moebius_schema();',
                  '    if (model == "pixelhacker") return pixelhacker_schema();',
                  '    throw std::invalid_argument("unsupported latent inpainting model");', "}",
                  "inline ShapeMap vae_schema() { return sdxl_vae_schema(); }", "",
                  "} // namespace latent_inpaint", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-native-schema", action="store_true")
    args = parser.parse_args()
    if args.write_native_schema:
        NATIVE_SCHEMA.write_text(native_schema_text(), encoding="utf-8")
    else:
        contract = load_contract()
        print(json.dumps({"contract_sha256": sha256(CONTRACT), "models": {
            model: {"original": contract["models"][model]["original_tensor_count"],
                    "export": contract["models"][model]["export_tensor_count"],
                    "checkpoint_sha256": contract["models"][model]["checkpoint"]["sha256"]} for model in MODELS}}, indent=2))


if __name__ == "__main__":
    main()
