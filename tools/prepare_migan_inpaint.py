#!/usr/bin/env python3
"""Reproduce the reviewed MI-GAN F32 GGUF using only Python's standard library.

This is a byte replication tool for ONE previously reviewed checkpoint, not a
general PyTorch converter or a new model qualification. It never imports torch,
executes the upstream generator, or relaxes the original-reference protocol.
Every checkpoint, tensor and final GGUF byte is checked against frozen evidence.
"""

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
from pathlib import Path
import pickle
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile

from migan_reference_contract import expected_state_shapes, read_f32_gguf, sha256, write_f32_gguf

ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "tests/results/migan/original-reference.json"
CONTRACT_PATH = ROOT / "docs/migan-reference-contract.json"
REVIEW_SHA256 = "ed43228a14627efef8e22fbd7b9e41c95cc7ab5e247a77f789cae6e34533460c"
CONTRACT_SHA256 = "ffbba862956246feb69dd1c118e7db65e2ee788d4a3ba7c8cb7f6986c698fb4b"
CHECKPOINT_SHA256 = "8b82b2e82fc8e5a2e1f06827594aac1a5d66a9ca41e24199ddd969847788097f"
CHECKPOINT_BYTES = 25231129
GGUF_SHA256 = "b8fcefc3de7e04f96867b9d15bd84a2cf7e0c444d2925644a7e8b15081809c8b"
GGUF_BYTES = 25185504
MODEL_NAME = "migan-256-places2-f32-experimental.gguf"
SOURCE_COMMIT = "2b793c5ece43f4253e32d4afc257120a5deed6f5"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_PICKLE_BYTES = 1024 * 1024


def load_review():
    """Load only the independently reviewed, immutable original receipt."""
    if sha256(REVIEW_PATH) != REVIEW_SHA256 or sha256(CONTRACT_PATH) != CONTRACT_SHA256:
        raise ValueError("the frozen MI-GAN review or contract differs; use the matching source checkout")
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    conversion = review["conversion"]
    if (review["schema"] != "crispembed-migan-original-independent-review-v1"
            or review["service_conclusion"] != "success"
            or conversion["checkpoint_sha256"] != CHECKPOINT_SHA256
            or conversion["gguf_sha256"] != GGUF_SHA256
            or conversion["gguf_bytes"] != GGUF_BYTES
            or conversion["all_tensor_bytes_exact"] is not True
            or contract["original_source"]["commit"] != SOURCE_COMMIT):
        raise ValueError("the frozen MI-GAN identities do not agree")
    return review, contract


def verify_file(path, expected_bytes, expected_sha256, label):
    path = Path(path)
    if path.stat().st_size != expected_bytes or sha256(path) != expected_sha256:
        raise ValueError(f"{label} byte count or SHA-256 differs from the reviewed artifact")


def download(url, destination, expected_bytes, expected_sha256):
    """Bound a public download and check its identity before making it usable."""
    destination = Path(destination)
    count = 0
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "CrispEmbed-MIGAN-replication/1"})
    try:
        with destination.open("xb") as output:
            with urllib.request.urlopen(request, timeout=60) as response:
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("download returned an HTML confirmation/error page")
                while True:
                    block = response.read(min(1024 * 1024, expected_bytes - count + 1))
                    if not block:
                        break
                    count += len(block)
                    if count > expected_bytes:
                        raise ValueError("download exceeds the reviewed byte count")
                    digest.update(block)
                    output.write(block)
        if count != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ValueError("download byte count or SHA-256 differs from the reviewed artifact")
    except FileExistsError:
        raise
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


class _FloatStorage:
    """A harmless token; no torch code is imported or executed."""


@dataclass(frozen=True)
class _Storage:
    key: str
    count: int


@dataclass(frozen=True)
class _Tensor:
    storage: _Storage
    offset: int
    shape: tuple
    stride: tuple


def _rebuild_tensor(storage, offset, shape, stride, *extras):
    if (not isinstance(storage, _Storage) or type(offset) is not int or offset < 0
            or not isinstance(shape, (tuple, list)) or not isinstance(stride, (tuple, list))
            or not 0 <= len(shape) <= 4 or len(shape) != len(stride)
            or any(type(n) is not int or n <= 0 for n in shape)
            or any(type(n) is not int or n < 0 for n in stride)
            or len(extras) not in (0, 2, 3)):
        raise ValueError("unsupported tensor metadata in the MI-GAN checkpoint")
    if extras and (type(extras[0]) is not bool or not isinstance(extras[1], (dict, OrderedDict)) or extras[1]):
        raise ValueError("unsupported tensor gradient/hook metadata")
    if len(extras) == 3 and extras[2] not in (None, {}):
        raise ValueError("tensor conjugate/negative metadata is not supported")
    return _Tensor(storage, offset, tuple(shape), tuple(stride))


class _StateReader(pickle.Unpickler):
    """Read only the tensor/storage metadata forms used by the pinned state dict.

    Persistent storages are inert records, and tensor rebuilds return metadata.
    No imported class, callable from the pickle, CUDA code or filesystem hook is
    permitted. The production caller verifies the entire checkpoint hash first.
    """

    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch", "FloatStorage"): _FloatStorage,
            ("torch._utils", "_rebuild_tensor"): _rebuild_tensor,
            ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor,
        }
        if (module, name) not in allowed:
            raise ValueError(f"unsupported pickle global: {module}.{name}")
        return allowed[(module, name)]

    def persistent_load(self, value):
        if (not isinstance(value, tuple) or len(value) != 5 or value[0] != "storage"
                or value[1] is not _FloatStorage or not isinstance(value[2], str)
                or re.fullmatch(r"[0-9]+", value[2]) is None
                or not isinstance(value[3], str) or len(value[3]) > 128
                or type(value[4]) is not int or not 0 < value[4] <= MAX_ARCHIVE_BYTES // 4):
            raise ValueError("unsupported persistent storage in the MI-GAN checkpoint")
        return _Storage(value[2], value[4])


def _tensor_bytes(tensor, raw):
    if len(raw) != tensor.storage.count * 4:
        raise ValueError("checkpoint storage byte count differs")
    last = tensor.offset + sum((dimension - 1) * stride
                               for dimension, stride in zip(tensor.shape, tensor.stride))
    count = 1
    for dimension in tensor.shape:
        count *= dimension
    if last >= tensor.storage.count or count > MAX_ARCHIVE_BYTES // 4:
        raise ValueError("tensor view is outside its checkpoint storage")
    contiguous, expected = True, 1
    for dimension, stride in zip(reversed(tensor.shape), reversed(tensor.stride)):
        if dimension > 1 and stride != expected:
            contiguous = False
        expected *= dimension
    if contiguous:
        return raw[4 * tensor.offset:4 * (tensor.offset + count)]
    result = bytearray(count * 4)
    # Support a transposed/strided F32 view without importing a tensor framework.
    for index, coordinates in enumerate(itertools.product(*(range(n) for n in tensor.shape))):
        offset = tensor.offset + sum(i * s for i, s in zip(coordinates, tensor.stride))
        result[4 * index:4 * index + 4] = raw[4 * offset:4 * offset + 4]
    return bytes(result)


def extract_tensors(checkpoint, expected_rows):
    """Return original-order F32 bytes, verified against reviewed tensor hashes.

    This helper accepts tiny authored archives in unit tests. The user-facing
    prepare() path must hash the complete real checkpoint before calling it.
    """
    expected = {row["name"]: row for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("duplicate reviewed tensor name")
    with zipfile.ZipFile(checkpoint) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if (len(names) != len(set(names)) or len(names) > 1024
                or sum(item.file_size for item in members) > MAX_ARCHIVE_BYTES):
            raise ValueError("unsupported checkpoint ZIP member inventory")
        pickles = [name for name in names if name.endswith("/data.pkl")]
        if len(pickles) != 1 or archive.getinfo(pickles[0]).file_size > MAX_PICKLE_BYTES:
            raise ValueError("expected exactly one bounded PyTorch ZIP data.pkl")
        prefix = pickles[0][:-len("data.pkl")]
        if prefix + "byteorder" in names and archive.read(prefix + "byteorder") != b"little":
            raise ValueError("only the reviewed little-endian F32 checkpoint is supported")
        stream = io.BytesIO(archive.read(pickles[0]))
        state = _StateReader(stream).load()
        if stream.read(1):
            raise ValueError("trailing pickle data in checkpoint")
        if not isinstance(state, (dict, OrderedDict)) or set(state) != set(expected):
            raise ValueError("checkpoint tensor names differ from the reviewed inventory")
        storage_cache = {}
        tensors = []
        for name in sorted(state):
            tensor, row = state[name], expected[name]
            if (not isinstance(tensor, _Tensor) or tensor.shape != tuple(row["shape"])
                    or row["dtype"] != "float32"):
                raise ValueError(f"checkpoint tensor shape/type differs: {name}")
            key = tensor.storage.key
            if key not in storage_cache:
                storage_cache[key] = archive.read(prefix + "data/" + key)
            raw = _tensor_bytes(tensor, storage_cache[key])
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError(f"checkpoint tensor bytes differ: {name}")
            tensors.append((name, tensor.shape, raw))
    return tensors


def metadata():
    # Preserve exact key insertion order, strings and integer types from the
    # already-reviewed converter: the final GGUF must match its full-file SHA.
    return {
        "general.architecture": "migan",
        "general.name": "MI-GAN-256-Places2-experimental-F32",
        "general.file_type": 0,
        "general.license": "MIT",
        "migan.image_size": 256,
        "migan.source.commit": SOURCE_COMMIT,
        "migan.checkpoint.sha256": CHECKPOINT_SHA256,
        "migan.input.contract": "RGB8;binary-mask8;255=fill;fixed256",
        "migan.tensor.layout": "original contiguous OIHW; GGUF dimensions reversed; scalar=[1]",
    }


def prepare(output_dir, checkpoint=None, download_checkpoint=False, source_dir=None):
    if (checkpoint is None) == (not download_checkpoint):
        raise ValueError("choose exactly one of --checkpoint or --download")
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("output directory already exists; choose a new directory")
    if not output_dir.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    review, contract = load_review()
    rows = review["producer"]["state"]
    if {row["name"]: tuple(row["shape"]) for row in rows} != expected_state_shapes():
        raise ValueError("reviewed tensor schema differs from the fixed256 model")

    with tempfile.TemporaryDirectory(prefix=".migan-prepare-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        if download_checkpoint:
            checkpoint = stage / contract["checkpoint"]["filename"]
            download(contract["checkpoint"]["download_url"], checkpoint, CHECKPOINT_BYTES, CHECKPOINT_SHA256)
        checkpoint = Path(checkpoint)
        # Hash before any pickle metadata is interpreted, including a local file.
        verify_file(checkpoint, CHECKPOINT_BYTES, CHECKPOINT_SHA256, "checkpoint")
        tensors = extract_tensors(checkpoint, rows)
        model = stage / MODEL_NAME
        write_f32_gguf(model, tensors, metadata())
        verify_file(model, GGUF_BYTES, GGUF_SHA256, "reproduced GGUF")
        actual_metadata, actual = read_f32_gguf(model)
        if actual_metadata != metadata() or actual != {
                name: (shape or (1,), raw) for name, shape, raw in tensors}:
            raise ValueError("reproduced GGUF metadata or tensor readback differs")

        notices = stage / "licenses"
        notices.mkdir()
        for filename in ("LICENSE", "LICENSE-WEIGHTS"):
            record = contract["original_source"]["files"][filename]
            target = notices / filename
            if source_dir is None:
                url = f"https://raw.githubusercontent.com/Picsart-AI-Research/MI-GAN/{SOURCE_COMMIT}/{filename}"
                download(url, target, record["bytes"], record["sha256"])
            else:
                source = Path(source_dir) / filename
                verify_file(source, record["bytes"], record["sha256"], filename)
                shutil.copyfile(source, target)
        (notices / "SOURCE.txt").write_text(
            f"MI-GAN: https://github.com/Picsart-AI-Research/MI-GAN/tree/{SOURCE_COMMIT}\n"
            "See the preserved source and weight MIT notices for copyright and licensing.\n"
            "This file reproduces the fixed256 F32 checkpoint already reviewed by CrispEmbed.\n",
            encoding="utf-8")
        receipt = {
            "schema": "crispembed-migan-byte-replication-v1",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "gguf_sha256": GGUF_SHA256,
            "gguf_bytes": GGUF_BYTES,
            "tensor_count": len(tensors),
            "all_reviewed_tensor_bytes_exact": True,
            "original_review_sha256": REVIEW_SHA256,
            "reference_contract_sha256": CONTRACT_SHA256,
            "source_commit": SOURCE_COMMIT,
            "framework_imported": False,
            "inference_performed": False,
            "new_quality_or_backend_qualification": False,
        }
        (stage / "replication.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        # Keep the downloaded checkpoint transient; only GGUF/notices/receipt
        # belong in the result. Publish only after every check succeeds.
        if download_checkpoint:
            checkpoint.unlink()
        output_dir.mkdir(exist_ok=False)
        try:
            for path in stage.iterdir():
                shutil.move(str(path), str(output_dir / path.name))
        except BaseException:
            shutil.rmtree(output_dir)
            raise
    return output_dir / MODEL_NAME


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--checkpoint", type=Path, help="Existing official migan_256_places2.pt; hash checked")
    choice.add_argument("--download", action="store_true", help="Download the exact official 25 MB checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for GGUF, notices and receipt")
    parser.add_argument("--source-dir", type=Path, help="Pinned MI-GAN checkout for offline LICENSE/LICENSE-WEIGHTS")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.output_dir, args.checkpoint, args.download, args.source_dir)
    except (OSError, ValueError, KeyError, pickle.UnpicklingError, EOFError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(result)
    print(f"SHA-256 {GGUF_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
