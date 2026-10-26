#!/usr/bin/env python3
"""Dependency-free MI-GAN 256 reference identity, fixtures and F32 GGUF I/O.

This module performs no network access or framework imports. Tensor names and
OIHW/CHW contiguous bytes are preserved; GGUF dimensions reverse NumPy axes.
"""

import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/migan-reference-contract.json"
RESOLUTION = 256
CASES = ("subtitle_bar", "edge_touch", "empty", "full", "center", "one_pixel", "checkerboard", "thin_strip")
FULL_STAGE_CASES = CASES[:2]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract():
    return json.loads(CONTRACT.read_text())


def validate_input(rgb, fill_mask, width=256, height=256):
    if (width, height) != (RESOLUTION, RESOLUTION):
        raise ValueError("MI-GAN reference requires exactly 256x256")
    if len(rgb) != width * height * 3 or len(fill_mask) != width * height:
        raise ValueError("expected RGB8 and same-size single-channel mask")
    if any(value not in (0, 255) for value in fill_mask):
        raise ValueError("mask must be binary: 255=fill, 0=known")


def fixture(name):
    """Repository-authored integer fixtures; no corpus, font or RNG dependency."""
    if name not in CASES:
        raise ValueError(f"unknown fixture: {name}")
    rgb = bytearray(256 * 256 * 3)
    mask = bytearray(256 * 256)
    for y in range(256):
        for x in range(256):
            i = y * 256 + x
            rgb[3 * i:3 * i + 3] = bytes(((3 * x + y) % 256, (x + 2 * y) % 256, (x // 16 * 23 + y // 16 * 11) % 256))
            hole = {
                "subtitle_bar": 24 <= x < 232 and 208 <= y < 240,
                "edge_touch": x < 35 and 48 <= y < 208,
                "empty": False,
                "full": True,
                "center": 88 <= x < 168 and 88 <= y < 168,
                "one_pixel": x == 128 and y == 128,
                "checkerboard": ((x // 32 + y // 32) % 2) == 0,
                "thin_strip": 250 <= y < 253,
            }[name]
            mask[i] = 255 if hole else 0
            # Owned burned-in block glyphs over a dark subtitle panel.
            if name == "subtitle_bar" and 24 <= x < 232 and 208 <= y < 240:
                glyph = 216 <= y < 231 and 36 <= x < 220 and (x % 13 < 8) and (y % 7 < 4 or x % 13 < 2)
                rgb[3 * i:3 * i + 3] = bytes((235, 235, 235) if glyph else (24, 24, 24))
    validate_input(rgb, mask)
    return bytes(rgb), bytes(mask)


def fixture_identity(name):
    rgb, mask = fixture(name)
    return {"id": name, "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
            "fill_mask_sha256": hashlib.sha256(mask).hexdigest(),
            "fill_pixels": mask.count(255), "full_stage_capture": name in FULL_STAGE_CASES}


def expected_state_shapes():
    """Independent exact key/shape schema for the pinned standalone Generator."""
    shapes = {}

    def conv(prefix, ic, oc, kernel=1, depthwise=False, bias=False):
        shapes[prefix + ".weight"] = (oc, 1 if depthwise else ic, kernel, kernel)
        if bias:
            shapes[prefix + ".bias"] = (oc,)

    def up(prefix, channels, resolution):
        shapes[prefix + ".filter_const"] = (1, 1, resolution, resolution)
        conv(prefix + ".filter", channels, channels, 4, True)

    def sep(prefix, ic, oc, resolution, down=False, upwards=False, noise=False):
        conv(prefix + ".conv1", ic, ic, 3, True, True)
        conv(prefix + ".conv2", ic, oc)
        if down:
            conv(prefix + ".downsample.filter", ic, ic, 4, True)
        if upwards:
            up(prefix + ".upsample", oc, resolution)
        if noise:
            shapes[prefix + ".noise_const"] = (resolution, resolution)
            shapes[prefix + ".noise_strength"] = ()

    channels = lambda res: min(32768 // res, 512)
    for res in (256, 128, 64, 32, 16, 8, 4):
        prefix = f"encoder.b{res}"
        ic, oc = channels(res), channels(max(res // 2, 4))
        if res == 256:
            conv(prefix + ".fromrgb", 4, ic, bias=True)
        sep(prefix + ".conv1", ic, ic, res)
        sep(prefix + ".conv2", ic, oc, res, down=res > 4)
    for res in (4, 8, 16, 32, 64, 128, 256):
        prefix = f"synthesis.b{res}"
        ic, oc = channels(max(res // 2, 4)), channels(res)
        sep(prefix + ".conv1", ic, oc, res, upwards=res > 4, noise=res > 4)
        sep(prefix + ".conv2", oc, oc, res, noise=res > 4)
        conv(prefix + ".torgb", oc, 3, bias=True)
        if res > 4:
            up(prefix + ".upsample", 3, res)
    return shapes


def validate_state_shapes(shapes):
    expected = expected_state_shapes()
    if set(shapes) != set(expected):
        raise ValueError(f"state keys differ: missing={sorted(set(expected)-set(shapes))}, extra={sorted(set(shapes)-set(expected))}")
    for name, shape in shapes.items():
        if tuple(shape) != expected[name]:
            raise ValueError(f"state shape differs: {name}: {shape} != {expected[name]}")


def _string(stream, value):
    data = value.encode("utf-8")
    stream.write(struct.pack("<Q", len(data)) + data)


def write_f32_gguf(path, tensors, metadata):
    """tensors: [(name, NumPy-order shape, little-endian F32 bytes)]."""
    tensors = list(tensors)
    if len({name for name, _, _ in tensors}) != len(tensors):
        raise ValueError("duplicate tensor name")
    with open(path, "wb") as stream:
        stream.write(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata)))
        for key, value in metadata.items():
            _string(stream, key)
            if isinstance(value, str):
                stream.write(struct.pack("<I", 8))
                _string(stream, value)
            elif type(value) is int and 0 <= value < 2**32:
                stream.write(struct.pack("<II", 4, value))
            else:
                raise ValueError(f"unsupported GGUF metadata: {key}")
        offset = 0
        for name, shape, data in tensors:
            shape = tuple(shape) or (1,)
            count = 1
            for dim in shape:
                if dim <= 0:
                    raise ValueError("invalid tensor dimension")
                count *= dim
            if len(data) != count * 4:
                raise ValueError(f"F32 byte count differs: {name}")
            _string(stream, name)
            stream.write(struct.pack("<I", len(shape)))
            stream.write(struct.pack("<" + "Q" * len(shape), *reversed(shape)))
            stream.write(struct.pack("<IQ", 0, offset))
            offset += (len(data) + 31) & ~31
        stream.write(b"\0" * (-stream.tell() % 32))
        for _, _, data in tensors:
            stream.write(data)
            stream.write(b"\0" * (-len(data) % 32))


def read_f32_gguf(path):
    """Strict small reader used for independent exported-byte verification."""
    with open(path, "rb") as stream:
        def unpack(fmt):
            return struct.unpack(fmt, stream.read(struct.calcsize(fmt)))

        def string():
            return stream.read(unpack("<Q")[0]).decode("utf-8")

        magic, version, count, kv_count = unpack("<IIQQ")
        if (magic, version) != (0x46554747, 3):
            raise ValueError("not GGUF v3")
        metadata = {}
        for _ in range(kv_count):
            key, kind = string(), unpack("<I")[0]
            if key in metadata or kind not in (4, 8):
                raise ValueError("unsupported or duplicate metadata")
            metadata[key] = string() if kind == 8 else unpack("<I")[0]
        entries = []
        for _ in range(count):
            name, ndim = string(), unpack("<I")[0]
            if not 1 <= ndim <= 4:
                raise ValueError("invalid GGUF dimension count")
            dims = unpack("<" + "Q" * ndim)
            kind, offset = unpack("<IQ")
            if kind != 0 or offset % 32 or any(dim == 0 for dim in dims):
                raise ValueError("non-F32 or unaligned tensor")
            entries.append((name, tuple(reversed(dims)), offset))
        start = (stream.tell() + 31) & ~31
        tensors = {}
        for name, shape, offset in entries:
            count = 1
            for dim in shape:
                count *= dim
            stream.seek(start + offset)
            data = stream.read(count * 4)
            if len(data) != count * 4 or name in tensors:
                raise ValueError("truncated or duplicate GGUF tensor")
            tensors[name] = (shape, data)
    return metadata, tensors
