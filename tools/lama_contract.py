#!/usr/bin/env python3
"""Pinned Big-LaMa archive, inert Lightning metadata and streamed GGUF helpers.

The accepted input is the exact ZIP linked by the original LaMa README. Its
complete byte count and SHA-256 are verified before ZIP or pickle parsing.
No Torch, Lightning, OmegaConf, NumPy, upstream code or network is imported.
The small allow-list maps checkpoint object records to inert local holders.
"""

import argparse
from collections import OrderedDict, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import zipfile

from latent_inpaint_contract import (BLOCK_BYTES, Checkpoint as TensorStream, DTYPE_BYTES,
                                    MAX_PICKLE_BYTES, StateReader, Tensor, sha256,
                                    validate_state, verify_gguf, write_streamed_gguf)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/lama-source-contract.json"
NATIVE_SCHEMA = ROOT / "src/lama_schema.h"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 4096
LAYOUT = "original contiguous; GGUF dimensions reversed; BN counters omitted; evaluation-only state omitted"
INPUT_CONTRACT = "RGB8;binary-mask8;255=fill;maskedRGB01+mask;symmetric-pad8;reflect-conv"
WEIGHT_LICENSE_STATUS = "Apache-2.0-maintainer-comment; dedicated-weight-licence-text-absent"


def export_shape(name, shape, dtype):
    """Only generator F32 weights and inference-inert BN counters are accepted."""
    if not name.startswith("generator.model."):
        return None
    if dtype == "int64" and name.endswith(".num_batches_tracked") and shape == []:
        return None
    if dtype != "float32" or len(shape) not in (1, 4) or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("unsupported Big-LaMa generator tensor layout")
    return list(shape)


def load_contract():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("schema") != "crispembed-lama-source-v1" or contract.get("version") != 1:
        raise ValueError("unsupported Big-LaMa source contract")
    rows = contract["state"]
    names = [row["name"] for row in rows]
    if names != sorted(set(names)) or contract["original_tensor_count"] != len(rows):
        raise ValueError("Big-LaMa state census is not uniquely sorted")
    for row in rows:
        if row["dtype"] not in ("float32", "int64"):
            raise ValueError("unsupported Big-LaMa source dtype")
        if row["export_shape"] != export_shape(row["name"], row["shape"], row["dtype"]):
            raise ValueError("unapproved Big-LaMa state transformation")
        if not row["name"].startswith(("generator.model.", "val_evaluator.", "test_evaluator.")):
            raise ValueError("unexpected Big-LaMa state namespace")
    selected = [row for row in rows if row["name"].startswith("generator.")]
    if (contract["generator_tensor_count"] != len(selected)
            or contract["export_tensor_count"] != sum(row["export_shape"] is not None for row in rows)
            or contract["omitted_generator_counter_count"] != sum(row["dtype"] == "int64" for row in selected)
            or contract["export_tensor_bytes"] != sum(math.prod(row["export_shape"]) * 4 for row in rows
                                                     if row["export_shape"] is not None)):
        raise ValueError("Big-LaMa tensor census totals differ")
    return contract


def generator_rows(contract=None):
    return [row for row in (contract or load_contract())["state"] if row["name"].startswith("generator.")]


def verify_source(source_dir, contract=None):
    source = (contract or load_contract())["source"]
    for relative, expected in source["files"].items():
        path = Path(source_dir) / relative
        if path.is_symlink() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("original Big-LaMa source identity differs: " + relative)
    return source


def verify_archive(path, contract=None):
    path = Path(path)
    expected = (contract or load_contract())["checkpoint"]
    if (path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_ARCHIVE_BYTES
            or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]):
        raise ValueError("official Big-LaMa archive byte count or SHA-256 differs")


class _InertRecord:
    """Stores a BUILD record without setting executable or original attributes."""
    def __setstate__(self, record):
        if not isinstance(record, dict):
            raise ValueError("unsupported inert configuration object state")
        self.record = record


class _ModelCheckpoint:
    pass


class _TypingAny:
    pass


class LightningReader(StateReader):
    """No original Lightning/OmegaConf class, import or method is executed."""
    def __init__(self, stream):
        super().__init__(stream)
        self.storages = {}

    def find_class(self, module, name):
        if module == "__builtin__" and name in ("dict", "list", "long"):
            return {"dict": dict, "list": list, "long": int}[name]
        if (module, name) == ("collections", "defaultdict"):
            return defaultdict
        if (module, name) == ("pytorch_lightning.callbacks.model_checkpoint", "ModelCheckpoint"):
            return _ModelCheckpoint
        if (module, name) == ("typing", "Any"):
            return _TypingAny
        if (module, name) in {
            ("omegaconf.base", "ContainerMetadata"), ("omegaconf.base", "Metadata"),
            ("omegaconf.dictconfig", "DictConfig"), ("omegaconf.listconfig", "ListConfig"),
            ("omegaconf.nodes", "AnyNode"),
        }:
            return _InertRecord
        return super().find_class(module, name)

    def persistent_load(self, value):
        storage = super().persistent_load(value)
        if storage.dtype not in ("float32", "int64") or storage.count * DTYPE_BYTES[storage.dtype] > MAX_ARCHIVE_BYTES:
            raise ValueError("unsupported Big-LaMa storage")
        previous = self.storages.setdefault(storage.key, storage)
        if previous != storage:
            raise ValueError("conflicting aliased Big-LaMa storage")
        return storage


def read_lightning_pickle(payload):
    if len(payload) > MAX_PICKLE_BYTES:
        raise ValueError("Big-LaMa pickle exceeds bounded metadata size")
    stream = io.BytesIO(payload)
    reader = LightningReader(stream)
    root = reader.load()
    if stream.read(1) or not isinstance(root, (dict, OrderedDict)):
        raise ValueError("invalid Lightning checkpoint root or trailing pickle bytes")
    return root, reader.storages


def _tensor_row(name, tensor):
    if not isinstance(tensor, Tensor):
        raise ValueError("Big-LaMa state must contain only tensor records")
    return {"name": name, "shape": list(tensor.shape), "stride": list(tensor.stride),
            "offset": tensor.offset, "storage": tensor.storage.key,
            "storage_count": tensor.storage.count, "dtype": tensor.storage.dtype}


def auxiliary_tensors(root):
    """Collect inert non-state tensors, including the saved validation score."""
    result, seen = {}, set()

    def walk(value, path):
        if isinstance(value, Tensor):
            result[path] = value
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str):
                    label = key
                elif key is _ModelCheckpoint:
                    label = "<ModelCheckpoint>"
                elif type(key) is int:
                    label = str(key)
                else:
                    raise ValueError("unexpected inert configuration dictionary key")
                walk(child, path + "." + label if path else label)
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                walk(child, path + "." + str(index))
        elif isinstance(value, _InertRecord):
            walk(getattr(value, "record", {}), path + ".<record>")
        elif value is not None and type(value) not in (bool, int, float, str, bytes, type):
            raise ValueError("unexpected inert configuration value")

    for key, value in root.items():
        if key != "state_dict":
            walk(value, key)
    return result


def validate_lightning(root, storages, contract):
    cp = contract["checkpoint"]
    if set(root) != set(cp["root_keys"]):
        raise ValueError("Lightning checkpoint root keys differ")
    for name, expected in cp["root_primitive_values"].items():
        if type(root[name]) is not type(expected) or root[name] != expected:
            raise ValueError("Lightning checkpoint primitive metadata differs: " + name)
    state = root.get("state_dict")
    if not isinstance(state, (dict, OrderedDict)) or set(state) != {row["name"] for row in contract["state"]}:
        raise ValueError("Big-LaMa full state names differ from the frozen census")
    for row in contract["state"]:
        actual = _tensor_row(row["name"], state[row["name"]])
        if any(actual[key] != row[key] for key in actual):
            raise ValueError("Big-LaMa state shape/type/storage differs: " + row["name"])
    selected = {name: tensor for name, tensor in state.items() if name.startswith("generator.")}
    validate_state(selected, generator_rows(contract))
    auxiliary = auxiliary_tensors(root)
    observed_auxiliary = [_tensor_row(name, tensor) for name, tensor in sorted(auxiliary.items())]
    if observed_auxiliary != cp["auxiliary_state"]:
        raise ValueError("Big-LaMa auxiliary tensor census differs")
    expected_storages = {}
    for row in contract["state"] + cp["auxiliary_state"]:
        key = row["storage"]
        value = {"count": row["storage_count"], "dtype": row["dtype"]}
        if key in expected_storages and expected_storages[key] != value:
            raise ValueError("conflicting frozen state storage")
        expected_storages[key] = value
    actual_storages = {key: {"count": value.count, "dtype": value.dtype} for key, value in storages.items()}
    if actual_storages != expected_storages:
        raise ValueError("Big-LaMa checkpoint storage census differs")
    return state, selected, auxiliary


class Checkpoint(TensorStream):
    """Verified official ZIP, safely extracted to a private temporary file.

    ``state`` contains all 989 generator records, including BN counters.
    ``all_state`` additionally contains the explicitly excluded evaluation
    records. ``inner_sha256`` is an observed extraction digest, not a claim of
    an independently published inner-checkpoint pin. The outer archive is the
    pinned cryptographic input identity.
    """

    def __init__(self, official_zip_path, contract=None, work_dir=None):
        self.contract = contract or load_contract()
        self.archive = None
        self._temporary = None
        verify_archive(official_zip_path, self.contract)
        cp = self.contract["checkpoint"]
        try:
            self._temporary = tempfile.TemporaryDirectory(prefix=".lama-checkpoint-", dir=work_dir)
            self.inner_path = Path(self._temporary.name) / "checkpoint.bin"
            with zipfile.ZipFile(official_zip_path) as outer:
                members = outer.infolist()
                expected = {row["name"]: row for row in cp["archive_members"]}
                if len(members) != len(expected) or {item.filename for item in members} != set(expected):
                    raise ValueError("official Big-LaMa ZIP member inventory differs")
                for item in members:
                    row = expected[item.filename]
                    if (item.file_size != row["bytes"] or item.compress_size != row["compressed_bytes"]
                            or item.compress_type != row["method"] or item.flag_bits != row["flags"]
                            or format(item.CRC, "08x") != row["crc32"] or item.header_offset != row["local_offset"]
                            or item.flag_bits & 1):
                        raise ValueError("official Big-LaMa ZIP member properties differ")
                self.config_bytes = outer.read(cp["config"]["path"])
                if (len(self.config_bytes) != cp["config"]["bytes"]
                        or hashlib.sha256(self.config_bytes).hexdigest() != cp["config"]["sha256"]):
                    raise ValueError("Big-LaMa original configuration differs")
                count, digest = 0, hashlib.sha256()
                with outer.open(cp["member"]["path"]) as source, self.inner_path.open("xb") as target:
                    while True:
                        block = source.read(BLOCK_BYTES)
                        if not block:
                            break
                        count += len(block)
                        if count > cp["member"]["bytes"] or count > MAX_ARCHIVE_BYTES:
                            raise ValueError("Big-LaMa extracted checkpoint exceeds its fixed size")
                        target.write(block)
                        digest.update(block)
                if count != cp["member"]["bytes"]:
                    raise ValueError("Big-LaMa extracted checkpoint is truncated")
                self.inner_sha256, self.inner_bytes = digest.hexdigest(), count
            self.archive = zipfile.ZipFile(self.inner_path)
            members = self.archive.infolist()
            names = [item.filename for item in members]
            if (len(names) != len(set(names)) or len(names) > MAX_MEMBERS
                    or sum(item.file_size for item in members) > MAX_ARCHIVE_BYTES
                    or any(item.compress_type != zipfile.ZIP_STORED or item.flag_bits & 1 for item in members)):
                raise ValueError("unsupported inner Big-LaMa checkpoint ZIP")
            self.prefix = cp["torch_zip_prefix"]
            pickle_name = self.prefix + "data.pkl"
            if self.archive.getinfo(pickle_name).file_size != cp["pickle_bytes"]:
                raise ValueError("Big-LaMa pickle byte count differs")
            payload = self.archive.read(pickle_name)
            if hashlib.sha256(payload).hexdigest() != cp["pickle_sha256"]:
                raise ValueError("Big-LaMa pickle digest differs")
            root, storages = read_lightning_pickle(payload)
            self.all_state, self.state, self.auxiliary = validate_lightning(root, storages, self.contract)
            expected_data = {self.prefix + "data/" + key for key in storages}
            allowed_meta = {pickle_name, self.prefix + "version", self.prefix + "byteorder",
                            self.prefix + ".data/serialization_id"}
            if ({name for name in names if name.startswith(self.prefix + "data/")} != expected_data
                    or set(names) - expected_data - allowed_meta):
                raise ValueError("Big-LaMa inner storage member inventory differs")
            if self.prefix + "byteorder" in names and self.archive.read(self.prefix + "byteorder") != b"little":
                raise ValueError("only little-endian Big-LaMa storage is supported")
            for key, storage in storages.items():
                if self.archive.getinfo(self.prefix + "data/" + key).file_size != storage.count * DTYPE_BYTES[storage.dtype]:
                    raise ValueError("Big-LaMa storage byte count differs")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.archive is not None:
            self.archive.close()
            self.archive = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def raw_blocks(self, name):
        tensor = self.state[name]
        with self.archive.open(self.prefix + "data/" + tensor.storage.key) as stream:
            while True:
                block = stream.read(BLOCK_BYTES)
                if not block:
                    break
                yield block

    def read_counter(self, name):
        tensor = self.state[name]
        if tensor.storage.dtype != "int64" or tensor.shape or not name.endswith(".num_batches_tracked"):
            raise ValueError("only scalar generator BatchNorm counters can be read as integers")
        raw = b"".join(self.raw_blocks(name))
        if len(raw) != 8 or struct.unpack("<q", raw)[0] < 0:
            raise ValueError("invalid Big-LaMa BatchNorm training counter")
        return struct.unpack("<q", raw)[0]

    def validate_omitted(self, rows=None):
        return super().validate_omitted(rows or generator_rows(self.contract))


def gguf_metadata(contract=None):
    contract = contract or load_contract()
    cp = contract["checkpoint"]
    return {
        "general.architecture": "big_lama",
        "general.name": "big-lama-places2-experimental-f32",
        "general.file_type": 0,
        "general.license": "Apache-2.0",
        "lama.schema": 1,
        "lama.source.commit": contract["source"]["commit"],
        "lama.contract.sha256": sha256(CONTRACT),
        "lama.checkpoint.sha256": cp["sha256"],
        "lama.checkpoint.revision": cp["revision"],
        "lama.checkpoint.container": "official-zip",
        "lama.checkpoint.member": cp["member"]["path"],
        "lama.tensor.layout": LAYOUT,
        "lama.input.contract": INPUT_CONTRACT,
        "lama.image.min_dimension": 9,
        "lama.image.max_dimension": 1024,
        "lama.generator.blocks": 18,
        "lama.lfu": False,
        "lama.fft.norm": "ortho",
        "lama.weights.license.status": WEIGHT_LICENSE_STATUS,
    }


def native_schema_text():
    contract = load_contract()
    cp = contract["checkpoint"]
    strings = {
        "source_commit": contract["source"]["commit"],
        "source_contract_sha256": sha256(CONTRACT),
        "checkpoint_sha256": cp["sha256"],
        "checkpoint_revision": cp["revision"],
        "checkpoint_container": "official-zip",
        "checkpoint_member": cp["member"]["path"],
        "tensor_layout": LAYOUT,
        "input_contract": INPUT_CONTRACT,
        "weight_license_status": WEIGHT_LICENSE_STATUS,
    }
    lines = ["// Generated from the pinned official archive's independently frozen tensor census.",
             "// checkpoint_sha256 identifies the complete official ZIP, not an inferred inner-file digest.",
             "// Regenerate with: python tools/lama_contract.py --write-native-schema",
             "// clang-format off", "#pragma once", '#include "latent_inpaint_ops.h"', "#include <cstdint>",
             "", "namespace lama { namespace detail {", ""]
    for name, value in strings.items():
        lines.append('inline const char * ' + name + '() { return "' + value + '"; }')
    lines.extend(["inline uint64_t checkpoint_bytes() { return " + str(cp["bytes"]) + "ULL; }", "",
                  "inline latent_inpaint::ShapeMap schema() {", "    return {"])
    for row in generator_rows(contract):
        if row["export_shape"] is not None:
            shape = ", ".join(str(n) for n in row["export_shape"])
            lines.append('        {"' + row["name"] + '", {' + shape + '}},')
    lines.extend(["    };", "}", "", "} } // namespace lama::detail", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-native-schema", action="store_true")
    args = parser.parse_args()
    if args.write_native_schema:
        NATIVE_SCHEMA.write_text(native_schema_text(), encoding="utf-8")
    else:
        contract = load_contract()
        print(json.dumps({"source_contract_sha256": sha256(CONTRACT),
                          "archive_sha256": contract["checkpoint"]["sha256"],
                          "original_tensor_count": contract["original_tensor_count"],
                          "generator_tensor_count": contract["generator_tensor_count"],
                          "export_tensor_count": contract["export_tensor_count"],
                          "export_tensor_bytes": contract["export_tensor_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
