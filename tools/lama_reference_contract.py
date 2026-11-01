#!/usr/bin/env python3
"""Big-LaMa CPU/F32 schema, owned fixtures and safe checkpoint byte reader.

This module imports no tensor framework and never downloads checkpoint weights.
The supported generator is the 18-block, 75%-global, LFU-disabled profile from
the pinned upstream big-lama training configuration. A conversion records model
identity; it does not establish checkpoint licensing or native quality parity.
"""

import array
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import pickle
import pickletools
import re
import struct
import sys
import zipfile

from migan_reference_contract import read_f32_gguf, sha256, write_f32_gguf

SOURCE_COMMIT = "786f5936b27fb3dacd2b1ad799e4de968ea697e7"
PROFILE = "big-lama-18-075-v1"
REFERENCE_TORCH_VERSION = "2.8.0"
MAX_CHECKPOINT_BYTES = 2 * 1024**3
MAX_PICKLE_BYTES = 16 * 1024**2
MAX_STORAGE_BYTES = 512 * 1024**2
MAX_PICKLE_OPS = 500000
MAX_MEMO_INDEX = 262143
MAX_METADATA_NODES = 200000
MAX_METADATA_DEPTH = 128
# LaMa's pinned trainer saves an OmegaConf configuration and uses callback types
# as Lightning checkpoint dictionary keys. These classes are NEVER imported.
# Their serialized metadata is retained inertly and discarded after selecting
# generator tensors. Primitive type GLOBALs are inert too: real bytes(n), for
# example, would allocate before a post-load resource check could reject n.
INERT_METADATA_GLOBALS = (
    ("omegaconf.dictconfig", "DictConfig"), ("omegaconf.listconfig", "ListConfig"),
    ("omegaconf.base", "Metadata"), ("omegaconf.base", "ContainerMetadata"),
    ("omegaconf.nodes", "AnyNode"), ("omegaconf.nodes", "StringNode"),
    ("omegaconf.nodes", "IntegerNode"), ("omegaconf.nodes", "FloatNode"),
    ("omegaconf.nodes", "BooleanNode"), ("typing", "Any"), ("typing", "Dict"), ("typing", "List"),
    ("pytorch_lightning.callbacks.model_checkpoint", "ModelCheckpoint"),
    ("collections", "defaultdict"),
) + tuple(("builtins", name) for name in ("dict", "list", "tuple", "str", "int", "float", "bool", "bytes", "object"))


class _InertMetadata:
    """Storage-only stand-in: no original constructor, method or import runs."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self._arguments = args
        self._keywords = kwargs

    def __setstate__(self, state):
        self._serialized_state = state


class _InertDefaultDict(dict):
    """Accept metadata SETITEMS but never invoke a serialized default factory."""

    def __init__(self, factory=None):
        super().__init__()
        self._serialized_factory = factory

    def __setstate__(self, state):
        self._serialized_state = state


INERT_METADATA_TYPES = {
    (module, name): type(name, ((_InertDefaultDict if (module, name) == ("collections", "defaultdict")
                               else _InertMetadata),), {"__module__": module})
    for module, name in INERT_METADATA_GLOBALS
}
SOURCE_FILES = {
    "LICENSE": "4ceeeac5a802e86c413c22b16cce8e9a22027b0250c97e6f8ac97c14cf0542c0",
    "configs/training/big-lama.yaml": "2ef2fb8e7cea812070f6c66d9ac162f1a8a6caffdafae213ecd5436ecdcb9947",
    "configs/prediction/default.yaml": "e7733ceadd0be2210008f97b786772d10e866a6b5f7cd272d3ec5a2160e01e2c",
    "saicinpainting/training/modules/ffc.py": "1647ce037032e533c777fd1d486c61d3ecc0eee7603b8cb96b23d0deffcaf308",
    "saicinpainting/training/modules/base.py": "526016ea45cd40b17b1bfa812c3be1ae227b735de684cd77382a5788eb5ac4aa",
    "saicinpainting/training/modules/spatial_transform.py": "4f479a0c9ba94bd46b197507e8e8cdcc570e84ec29f5a577ade63e542adb43b5",
    "saicinpainting/training/modules/squeeze_excitation.py": "933c53adb2477b99c58ab325b26b124507a06b8c097ed2dd213c28f5841839fe",
    "saicinpainting/training/modules/depthwise_sep_conv.py": "9aa49550f3d34f1682cabbcc92f89d5f131ff0e89ebeefbfde396a85153c5b22",
    "saicinpainting/training/modules/multidilated_conv.py": "26929410919ebe0e024bd65d318978d14bdefcfeef94d079ae9c4d20189f2780",
    "saicinpainting/utils.py": "0a399137ac692a8c0eac6b8f1befacdd9d8f3443397d8a0b000845010c2b0cfc",
    "saicinpainting/evaluation/data.py": "ae2d1959b40eeb494bead932a891c6aba718297738bd01664e556611b9ac636a",
    "saicinpainting/training/trainers/default.py": "85e4a9ffaf21d725a48cf6dc066de3224c2168cea51c19696197e2a2f33f24ef",
    "bin/predict.py": "4e40c5eece914d30ea16c3aafc0699f46e07c924c98f3aeecc6dcb8642b53cf8",
}


def expected_state_shapes(include_counters=False):
    """Original PyTorch axis order, with only generator. prefix removed."""
    shapes = {}

    def conv(name, ic, oc, kernel=1, bias=False, transpose=False):
        shapes[name + ".weight"] = ((ic, oc) if transpose else (oc, ic)) + (kernel, kernel)
        if bias:
            shapes[name + ".bias"] = (oc,)

    def bn(name, channels):
        for field in ("weight", "bias", "running_mean", "running_var"):
            shapes[name + "." + field] = (channels,)
        if include_counters:
            shapes[name + ".num_batches_tracked"] = ()

    def ffc(name, il, ig, ol, og, kernel):
        if il and ol:
            conv(name + ".ffc.convl2l", il, ol, kernel)
        if il and og:
            conv(name + ".ffc.convl2g", il, og, kernel)
        if ig and ol:
            conv(name + ".ffc.convg2l", ig, ol, kernel)
        if ig and og:
            spectral = name + ".ffc.convg2g"
            conv(spectral + ".conv1.0", ig, og // 2)
            bn(spectral + ".conv1.1", og // 2)
            conv(spectral + ".fu.conv_layer", og, og)
            bn(spectral + ".fu.bn", og)
            conv(spectral + ".conv2", og // 2, og)
        if ol:
            bn(name + ".bn_l", ol)
        if og:
            bn(name + ".bn_g", og)

    ffc("model.1", 4, 0, 64, 0, 7)
    ffc("model.2", 64, 0, 128, 0, 3)
    ffc("model.3", 128, 0, 256, 0, 3)
    ffc("model.4", 256, 0, 128, 384, 3)
    for block in range(5, 23):
        for layer in ("conv1", "conv2"):
            ffc(f"model.{block}.{layer}", 128, 384, 128, 384, 3)
    for index, ic, oc in ((24, 512, 256), (27, 256, 128), (30, 128, 64)):
        conv(f"model.{index}", ic, oc, 3, True, True)
        bn(f"model.{index + 1}", oc)
    conv("model.34", 64, 3, 7, True)
    return shapes


def expected_stage_names():
    """Required observer subset in original forward-execution order."""
    names = ["generator.input"]

    def ffc(name, global_input, global_output):
        names.append(name + ".ffc.convl2l.output")
        if global_input:
            names.append(name + ".ffc.convg2l.output")
        if global_output:
            names.append(name + ".ffc.convl2g.output")
        if global_input:
            spectral = name + ".ffc.convg2g"
            names.extend(spectral + suffix for suffix in
                         (".conv1.0.output", ".conv1.1.output", ".conv1.2.output"))
            names.extend(spectral + ".fu" + suffix for suffix in
                         (".input", ".rfft", ".conv_layer.output", ".bn.output", ".relu.output", ".output"))
            names.extend((spectral + ".conv2.output", spectral + ".output"))
        names.append(name + ".bn_l.output")
        if global_output:
            names.append(name + ".bn_g.output")
        names.append(name + ".output.local")
        if global_output:
            names.append(name + ".output.global")

    for index in (1, 2, 3):
        ffc(f"model.{index}", False, False)
    ffc("model.4", False, True)
    for block in range(5, 23):
        for layer in ("conv1", "conv2"):
            ffc(f"model.{block}.{layer}", True, True)
        names += [f"model.{block}.output.local", f"model.{block}.output.global"]
    names += [f"model.{i}.output" for i in range(23, 33)]
    names += ["model.34.output", "model.35.output", "generator.output"]
    return names


def validate_state_shapes(shapes, include_counters=False):
    expected = expected_state_shapes(include_counters)
    if set(shapes) != set(expected):
        raise ValueError(f"generator keys differ: missing={sorted(set(expected) - set(shapes))}, "
                         f"extra={sorted(set(shapes) - set(expected))}")
    for name, shape in shapes.items():
        if tuple(shape) != expected[name]:
            raise ValueError(f"generator shape differs: {name}: {tuple(shape)} != {expected[name]}")


def canonical_generator_config():
    return {
        "kind": "ffc_resnet", "input_nc": 4, "output_nc": 3, "ngf": 64,
        "n_downsampling": 3, "n_blocks": 18, "add_out_act": "sigmoid",
        "init_conv_kwargs": {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
        "downsample_conv_kwargs": {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
        "resnet_conv_kwargs": {"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False},
    }


def validate_config(config):
    """Reject architecture changes, including silent PyTorch default changes."""
    if not isinstance(config, dict) or not isinstance(config.get("generator"), dict):
        raise ValueError("config must contain the original generator mapping")
    generator = config["generator"]
    expected = canonical_generator_config()
    optional = {"padding_type": "reflect", "max_features": 1024, "out_ffc": False,
                "spatial_transform_layers": None, "spatial_transform_kwargs": {}, "out_ffc_kwargs": {}}
    if set(generator) - set(expected) - set(optional):
        raise ValueError("unsupported generator options: " + ", ".join(sorted(set(generator) - set(expected) - set(optional))))

    def resolve(value, seen=()):
        if not isinstance(value, str) or not value.startswith("${"):
            return value
        if not value.endswith("}") or value in seen or len(seen) > 8:
            raise ValueError("unsupported/cyclic generator interpolation")
        path = value[2:-1].split(".")
        if len(path) != 3 or path[0] != "generator" or path[1] not in expected:
            raise ValueError("unsupported generator interpolation")
        try:
            target = generator[path[1]][path[2]]
        except (KeyError, TypeError):
            raise ValueError("unresolved generator interpolation") from None
        return resolve(target, seen + (value,))

    def equal(actual, target):
        actual = resolve(actual)
        if isinstance(target, bool):
            return type(actual) is bool and actual == target
        if isinstance(target, (int, float)):
            return type(actual) in (int, float) and actual == target
        return type(actual) is type(target) and actual == target

    for key, target in expected.items():
        actual = generator.get(key)
        if isinstance(target, dict):
            if not isinstance(actual, dict) or set(actual) != set(target):
                raise ValueError(f"unsupported generator configuration: {key}")
            if any(not equal(actual[field], value) for field, value in target.items()):
                raise ValueError(f"unsupported generator configuration: {key}")
        elif not equal(actual, target):
            raise ValueError(f"unsupported generator configuration: {key}")
    for key, target in optional.items():
        if key in generator and not equal(generator[key], target):
            raise ValueError(f"unsupported generator configuration: {key}")
    training = config.get("training_model", {})
    if not isinstance(training, dict) or training.get("concat_mask", True) is not True:
        raise ValueError("LaMa requires training_model.concat_mask=true")
    return expected


def load_config(path):
    return load_config_with_sha256(path)[0]


def load_config_with_sha256(path):
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("config exceeds 1 MiB")
    text = raw.decode("utf-8")
    if path.suffix.lower() == ".json":
        config = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("YAML config requires PyYAML; JSON config needs no dependency") from exc
        config = yaml.safe_load(text)
    return validate_config(config), hashlib.sha256(raw).hexdigest()


def validate_sha256(value, label="SHA-256"):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def validate_reference_torch_version(value):
    if not isinstance(value, str) or value.split("+", 1)[0] != REFERENCE_TORCH_VERSION:
        raise ValueError("original LaMa reference requires PyTorch 2.8.0; install torch==2.8.0 from the CPU wheel index")


def model_metadata(checkpoint_sha256, config_sha256):
    return {
        "general.architecture": "lama", "general.name": "Big-LaMa-18-075-experimental-F32",
        "general.file_type": 0, "lama.variant": "big-lama", "lama.schema.version": 1,
        "lama.profile": PROFILE, "lama.source.commit": SOURCE_COMMIT,
        "lama.checkpoint.sha256": validate_sha256(checkpoint_sha256, "checkpoint SHA-256"),
        "lama.config.sha256": validate_sha256(config_sha256, "config SHA-256"),
        "lama.source.license": "Apache-2.0", "lama.checkpoint.license": "user-supplied;not-verified",
        "lama.input.contract": "RGB8;binary-mask8;255=fill;symmetric-pad8",
        "lama.tensor.layout": "original contiguous OIHW/IOHW; GGUF dimensions reversed",
        "lama.fft.norm": "ortho",
    }


class _FloatStorage:
    pass


class _LongStorage:
    pass


@dataclass(frozen=True)
class _Storage:
    key: str
    count: int
    itemsize: int

    def __setstate__(self, state):
        raise ValueError("BUILD on checkpoint storage records is unsupported")


@dataclass(frozen=True)
class _Tensor:
    storage: _Storage
    offset: int
    shape: tuple
    stride: tuple

    def __setstate__(self, state):
        raise ValueError("BUILD on checkpoint tensor records is unsupported")


def _rebuild_tensor(storage, offset, shape, stride, *extras):
    if (not isinstance(storage, _Storage) or type(offset) is not int or offset < 0
            or not isinstance(shape, (tuple, list)) or not isinstance(stride, (tuple, list))
            or len(shape) > 4 or len(shape) != len(stride)
            or any(type(n) is not int or n <= 0 for n in shape)
            or any(type(n) is not int or n < 0 for n in stride)
            or len(extras) not in (0, 2, 3)):
        raise ValueError("unsupported checkpoint tensor metadata")
    if extras and (type(extras[0]) is not bool or not isinstance(extras[1], (dict, OrderedDict)) or extras[1]):
        raise ValueError("unsupported checkpoint gradient/hook metadata")
    if len(extras) == 3 and extras[2] not in (None, {}):
        raise ValueError("conjugate/negative tensor metadata is unsupported")
    tensor = _Tensor(storage, offset, tuple(shape), tuple(stride))
    # Validate at REDUCE time too: an invalid rebuilt object could otherwise be
    # popped from the pickle stack and disappear before the post-load walk.
    validate_tensor_metadata(tensor)
    return tensor


class _StateReader(pickle.Unpickler):
    """Whitelist inert tensor metadata; never import or execute pickle globals."""

    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch", "FloatStorage"): _FloatStorage,
            ("torch", "LongStorage"): _LongStorage,
            ("torch._utils", "_rebuild_tensor"): _rebuild_tensor,
            ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor,
        }
        if (module, name) in allowed:
            return allowed[(module, name)]
        legacy_names = {("__builtin__", "unicode"): ("builtins", "str"),
                        ("__builtin__", "long"): ("builtins", "int")}
        qualified = legacy_names.get((module, name), ("builtins" if module == "__builtin__" else module, name))
        if qualified in INERT_METADATA_TYPES:
            if not hasattr(self, "inert_globals"):
                self.inert_globals = set()
            self.inert_globals.add(".".join(qualified))
            return INERT_METADATA_TYPES[qualified]
        raise ValueError(f"unsupported pickle global: {module}.{name}")

    def persistent_load(self, value):
        if (not isinstance(value, tuple) or len(value) != 5 or value[0] != "storage"
                or value[1] not in (_FloatStorage, _LongStorage)
                or not isinstance(value[2], str) or re.fullmatch(r"[0-9]+", value[2]) is None
                or not isinstance(value[3], str) or len(value[3]) > 128
                or type(value[4]) is not int or value[4] <= 0):
            raise ValueError("unsupported persistent checkpoint storage")
        itemsize = 4 if value[1] is _FloatStorage else 8
        if value[4] * itemsize > MAX_STORAGE_BYTES:
            raise ValueError("checkpoint storage exceeds the 512 MiB bound")
        if not hasattr(self, "storages"):
            self.storages = {}
        description = (value[4], itemsize)
        if value[2] in self.storages and self.storages[value[2]] != description:
            raise ValueError("checkpoint storage key has inconsistent count/type")
        self.storages[value[2]] = description
        return _Storage(value[2], value[4], itemsize)


def _tensor_bytes(tensor, raw):
    # BUILD can mutate instance dictionaries even on frozen dataclasses. Recheck
    # every field after the entire pickle has been interpreted, before using it
    # for byte counts, strides or archive member selection.
    validate_tensor_metadata(tensor)
    if len(raw) != tensor.storage.count * tensor.storage.itemsize:
        raise ValueError("checkpoint storage byte count differs")
    count = math.prod(tensor.shape)
    last = tensor.offset + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride))
    if last >= tensor.storage.count or count * tensor.storage.itemsize > MAX_STORAGE_BYTES:
        raise ValueError("checkpoint tensor view exceeds storage")
    contiguous, expected = True, 1
    for dimension, stride in zip(reversed(tensor.shape), reversed(tensor.stride)):
        if dimension > 1 and stride != expected:
            contiguous = False
        expected *= dimension
    size = tensor.storage.itemsize
    if contiguous:
        return raw[size * tensor.offset:size * (tensor.offset + count)]
    result = bytearray(count * size)
    for index, coordinates in enumerate(itertools.product(*(range(n) for n in tensor.shape))):
        offset = tensor.offset + sum(i * s for i, s in zip(coordinates, tensor.stride))
        result[size * index:size * (index + 1)] = raw[size * offset:size * (offset + 1)]
    return bytes(result)


def validate_tensor_metadata(tensor):
    if not isinstance(tensor, _Tensor) or not isinstance(getattr(tensor, "storage", None), _Storage):
        raise ValueError("invalid checkpoint tensor/storage record")
    storage = tensor.storage
    if (not isinstance(getattr(storage, "key", None), str) or len(storage.key) > 128
            or re.fullmatch(r"[0-9]+", storage.key) is None
            or type(getattr(storage, "count", None)) is not int or storage.count <= 0
            or type(getattr(storage, "itemsize", None)) is not int or storage.itemsize not in (4, 8)
            or storage.count * storage.itemsize > MAX_STORAGE_BYTES):
        raise ValueError("invalid checkpoint storage metadata")
    if (type(getattr(tensor, "offset", None)) is not int or tensor.offset < 0
            or not isinstance(getattr(tensor, "shape", None), tuple)
            or not isinstance(getattr(tensor, "stride", None), tuple)
            or len(tensor.shape) > 4 or len(tensor.shape) != len(tensor.stride)
            or any(type(n) is not int or n <= 0 for n in tensor.shape)
            or any(type(n) is not int or n < 0 for n in tensor.stride)):
        raise ValueError("invalid checkpoint tensor metadata")
    last = tensor.offset + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride))
    if last >= storage.count or math.prod(tensor.shape) * storage.itemsize > MAX_STORAGE_BYTES:
        raise ValueError("checkpoint tensor view exceeds storage")


def validate_pickle_program(raw):
    """Bound memo allocation before the C unpickler can reserve memory.

    Extension opcodes bypass find_class via Python's global extension cache, so
    they are never accepted. The protocol-2/4/5 forms used by torch.save remain
    supported; no legacy text persistent-ID or application globals are needed.
    """
    memo_highwater = -1
    for index, (opcode, argument, _) in enumerate(pickletools.genops(raw)):
        if index >= MAX_PICKLE_OPS:
            raise ValueError("checkpoint pickle opcode count exceeds bound")
        if opcode.name in ("EXT1", "EXT2", "EXT4", "PERSID", "INST", "OBJ"):
            raise ValueError(f"unsupported checkpoint pickle opcode: {opcode.name}")
        if opcode.name in ("BINPUT", "LONG_BINPUT", "PUT", "BINGET", "LONG_BINGET", "GET"):
            if type(argument) is not int or not 0 <= argument <= MAX_MEMO_INDEX:
                raise ValueError("checkpoint pickle memo index exceeds bound")
            if opcode.name in ("BINPUT", "LONG_BINPUT", "PUT"):
                memo_highwater = max(memo_highwater, argument)
        if opcode.name == "MEMOIZE":
            memo_highwater += 1
            if memo_highwater > MAX_MEMO_INDEX:
                raise ValueError("checkpoint pickle memo index exceeds bound")
        if opcode.name == "FRAME" and argument > MAX_PICKLE_BYTES:
            raise ValueError("checkpoint pickle frame exceeds bound")


def validate_metadata_graph(root, tensor_type=None):
    """Bound discarded training metadata, including cycles and BUILD state."""
    pending, seen, tensors = [(root, 0)], set(), []
    known_types = set(INERT_METADATA_TYPES.values())
    while pending:
        value, depth = pending.pop()
        if depth > MAX_METADATA_DEPTH:
            raise ValueError("checkpoint metadata depth exceeds bound")
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if len(seen) > MAX_METADATA_NODES:
            raise ValueError("checkpoint metadata node count exceeds bound")
        if value is None or type(value) in (bool, float):
            continue
        if type(value) is int:
            if value.bit_length() > 128:
                raise ValueError("checkpoint metadata integer exceeds bound")
            continue
        if type(value) in (str, bytes):
            if len(value) > 1024 * 1024:
                raise ValueError("checkpoint metadata text/bytes exceeds 1 MiB")
            continue
        if isinstance(value, _Tensor):
            validate_tensor_metadata(value)
            tensors.append(value)
            continue
        if tensor_type is not None and isinstance(value, tensor_type):
            if value.numel() * value.element_size() > MAX_STORAGE_BYTES:
                raise ValueError("checkpoint tensor exceeds storage bound")
            continue
        if isinstance(value, type) and value in known_types:
            continue
        if isinstance(value, _InertMetadata):
            children = list(vars(value).values())
        elif isinstance(value, (dict, OrderedDict)):
            if len(value) > 100000:
                raise ValueError("checkpoint metadata mapping exceeds bound")
            children = list(value.keys()) + list(value.values())
            if hasattr(value, "__dict__"):
                children += list(vars(value).values())
        elif type(value) in (tuple, list, set, frozenset):
            if len(value) > 100000:
                raise ValueError("checkpoint metadata sequence exceeds bound")
            children = list(value)
        else:
            raise ValueError("unsupported checkpoint metadata object")
        pending.extend((child, depth + 1) for child in children)
    return tensors


def validate_float_bytes(name, raw):
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f"nonfinite model tensor: {name}")
    if name.endswith(".running_var") and any(x < 0 for x in values):
        raise ValueError(f"negative BatchNorm running variance: {name}")


def extract_tensors(checkpoint, checkpoint_sha256, shapes=None, provenance=None):
    """Read F32 generator bytes and omit verified inference-inert BN counters.

    `shapes` exists for small authored parser tests; production uses the complete
    fixed profile. Non-generator Lightning state is never exported or executed.
    """
    checkpoint = Path(checkpoint)
    validate_sha256(checkpoint_sha256, "checkpoint SHA-256")
    with checkpoint.open("rb") as stream:
        tensors, counters = _extract_open_checkpoint(stream, checkpoint_sha256, shapes, provenance)
        # Rehash the SAME open file description after extraction. Path swaps
        # cannot change the parsed file, and concurrent in-place writes fail.
        stream.seek(0)
        if _stream_sha256(stream) != checkpoint_sha256:
            raise ValueError("checkpoint changed during extraction")
    return tensors, counters


def _stream_sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _extract_open_checkpoint(stream, checkpoint_sha256, shapes, provenance):
    stream.seek(0, 2)
    checkpoint_bytes = stream.tell()
    if not 0 < checkpoint_bytes <= MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint size must be in 1..2 GiB")
    stream.seek(0)
    if _stream_sha256(stream) != checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 differs")
    stream.seek(0)
    expected = expected_state_shapes(True) if shapes is None else shapes
    with zipfile.ZipFile(stream) as archive:
        members = archive.infolist()
        names = [row.filename for row in members]
        if (len(names) != len(set(names)) or len(names) > 10000
                or sum(row.file_size for row in members) > MAX_CHECKPOINT_BYTES
                or any(row.flag_bits & 1 for row in members)):
            raise ValueError("unsupported checkpoint ZIP member inventory")
        candidates = [name for name in names if name.endswith("/data.pkl")]
        if len(candidates) != 1 or archive.getinfo(candidates[0]).file_size > MAX_PICKLE_BYTES:
            raise ValueError("expected one bounded PyTorch ZIP data.pkl")
        prefix = candidates[0][:-len("data.pkl")]
        if prefix + "byteorder" in names:
            if archive.getinfo(prefix + "byteorder").file_size != 6 or archive.read(prefix + "byteorder") != b"little":
                raise ValueError("only little-endian checkpoints are supported")
        pickle_bytes = archive.read(candidates[0])
        validate_pickle_program(pickle_bytes)
        pickle_stream = io.BytesIO(pickle_bytes)
        reader = _StateReader(pickle_stream)
        state = reader.load()
        if pickle_stream.read(1):
            raise ValueError("trailing checkpoint pickle data")
        validate_metadata_graph(state)
        # Include even persistent storages later POPped from the pickle stack;
        # PyTorch's loader would still execute their initial storage requests.
        for key, (count, itemsize) in getattr(reader, "storages", {}).items():
            declared_size = count * itemsize
            member = prefix + "data/" + key
            if member not in names or archive.getinfo(member).file_size != declared_size:
                raise ValueError("checkpoint storage is missing or has inconsistent declared size")
        if provenance is not None:
            provenance["inert_metadata_globals"] = sorted(getattr(reader, "inert_globals", set()))
            provenance["metadata_globals_executed"] = False
            provenance["checkpoint_bytes"] = checkpoint_bytes
        if not isinstance(state, (dict, OrderedDict)):
            raise ValueError("expected a state dictionary or Lightning checkpoint")
        if "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, (dict, OrderedDict)) or any(not isinstance(k, str) for k in state):
            raise ValueError("checkpoint state must have string keys")
        prefixed = any(name.startswith("generator.") for name in state)
        if prefixed:
            state = {name[len("generator."):]: value for name, value in state.items()
                     if name.startswith("generator.")}
        if set(state) != set(expected):
            raise ValueError(f"generator keys differ: missing={sorted(set(expected) - set(state))}, "
                             f"extra={sorted(set(state) - set(expected))}")
        tensors, counters = [], []
        for name, shape in sorted(expected.items()):
            tensor = state[name]
            validate_tensor_metadata(tensor)
            counter = name.endswith(".num_batches_tracked")
            if (not isinstance(tensor, _Tensor) or tensor.shape != tuple(shape)
                    or tensor.storage.itemsize != (8 if counter else 4)):
                raise ValueError(f"checkpoint tensor shape/type differs: {name}")
            member = prefix + "data/" + tensor.storage.key
            if member not in names or archive.getinfo(member).file_size != tensor.storage.count * tensor.storage.itemsize:
                raise ValueError(f"checkpoint storage is missing or has wrong size: {name}")
            raw = _tensor_bytes(tensor, archive.read(member))
            if counter:
                if shape != () or struct.unpack("<q", raw)[0] < 0:
                    raise ValueError(f"invalid inference-inert BatchNorm counter: {name}")
                counters.append({"name": name, "value": struct.unpack("<q", raw)[0]})
            else:
                validate_float_bytes(name, raw)
                tensors.append((name, tuple(shape), raw))
    return tensors, counters


CASES = ("subtitle", "edge", "padded", "empty", "full")


def fixture(name):
    if name not in CASES:
        raise ValueError(f"unknown LaMa fixture: {name}")
    width, height = (37, 29) if name == "padded" else (40, 32)
    rgb, mask = bytearray(width * height * 3), bytearray(width * height)
    for y in range(height):
        for x in range(width):
            index = y * width + x
            rgb[3 * index:3 * index + 3] = bytes(((7 * x + y) % 256, (x + 9 * y) % 256, (x * 3 + y * 5) % 256))
            fill = {"subtitle": 4 <= x < width - 4 and height - 10 <= y < height - 3,
                    "edge": x < 5 and y >= 4, "padded": x >= width - 7 and y >= height - 9,
                    "empty": False, "full": True}[name]
            mask[index] = 255 if fill else 0
            if name == "subtitle" and fill:
                glyph = x % 5 < 3 and y % 4 < 2
                rgb[3 * index:3 * index + 3] = bytes((235, 235, 235) if glyph else (22, 22, 22))
    return width, height, bytes(rgb), bytes(mask)


def fixture_identity(name):
    width, height, rgb, mask = fixture(name)
    return {"id": name, "width": width, "height": height,
            "padded_width": (width + 7) // 8 * 8, "padded_height": (height + 7) // 8 * 8,
            "rgb_sha256": hashlib.sha256(rgb).hexdigest(), "mask_sha256": hashlib.sha256(mask).hexdigest(),
            "fill_pixels": mask.count(255)}


def validate_input(rgb, mask, width, height):
    if type(width) is not int or type(height) is not int or not 16 <= width <= 4096 or not 16 <= height <= 4096:
        raise ValueError("reference dimensions must be integers in 16..4096")
    if len(rgb) != width * height * 3 or len(mask) != width * height:
        raise ValueError("expected tightly packed RGB8 and same-size mask8")
    if any(x not in (0, 255) for x in mask):
        raise ValueError("mask must contain only 0 and 255")


def verify_source(source_dir):
    source_dir = Path(source_dir)
    for relative, expected in SOURCE_FILES.items():
        if sha256(source_dir / relative) != expected:
            raise ValueError(f"original LaMa source differs: {relative}")
    return dict(SOURCE_FILES)
