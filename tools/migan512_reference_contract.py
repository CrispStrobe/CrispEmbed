#!/usr/bin/env python3
"""Separate fixed512 contract; the frozen fixed256 contract stays unchanged.

Only stdlib is imported. Original model execution lives in dump_migan512_reference.
"""
import hashlib
import json
from pathlib import Path

from migan_reference_contract import sha256, read_f32_gguf, write_f32_gguf

RESOLUTION = 512
CASES = ("subtitle_bar", "edge_touch", "empty", "full", "center", "one_pixel", "checkerboard", "thin_strip")
FULL_STAGE_CASES = CASES[:2]
SOURCE_COMMIT = "2b793c5ece43f4253e32d4afc257120a5deed6f5"
CHECKPOINT_BYTES = 29553256
CHECKPOINT_FILE_ID = "1D_YCuCgo20S2256sqpedsmENNm2WMtVY"
CONTRACT = Path(__file__).resolve().parents[1] / "docs/migan512-reference-contract.json"


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_input(rgb, fill_mask, width=512, height=512):
    if (width, height) != (RESOLUTION, RESOLUTION):
        raise ValueError("MI-GAN512 reference requires exactly 512x512")
    if len(rgb) != width * height * 3 or len(fill_mask) != width * height:
        raise ValueError("expected RGB8 and same-size single-channel mask")
    if any(value not in (0, 255) for value in fill_mask):
        raise ValueError("mask must be binary: 255=fill, 0=known")


def fixture(name):
    if name not in CASES:
        raise ValueError(f"unknown fixture: {name}")
    rgb, mask = bytearray(512 * 512 * 3), bytearray(512 * 512)
    for y in range(512):
        for x in range(512):
            i = y * 512 + x
            rgb[3 * i:3 * i + 3] = bytes(((3 * x + y) % 256, (x + 2 * y) % 256,
                                         (x // 16 * 23 + y // 16 * 11) % 256))
            hole = {
                "subtitle_bar": 48 <= x < 464 and 416 <= y < 480,
                "edge_touch": x < 70 and 96 <= y < 416,
                "empty": False, "full": True,
                "center": 176 <= x < 336 and 176 <= y < 336,
                "one_pixel": x == 256 and y == 256,
                "checkerboard": ((x // 64 + y // 64) % 2) == 0,
                "thin_strip": 500 <= y < 503,
            }[name]
            mask[i] = 255 if hole else 0
            if name == "subtitle_bar" and hole:
                glyph = 432 <= y < 462 and 72 <= x < 440 and (x % 26 < 16) and (y % 14 < 8 or x % 26 < 4)
                rgb[3 * i:3 * i + 3] = bytes((235, 235, 235) if glyph else (24, 24, 24))
    return bytes(rgb), bytes(mask)


def fixture_identity(name):
    rgb, mask = fixture(name)
    return {"id": name, "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
            "fill_mask_sha256": hashlib.sha256(mask).hexdigest(), "fill_pixels": mask.count(255),
            "full_stage_capture": name in FULL_STAGE_CASES}


def expected_state_shapes():
    # Build from the unchanged original256 schema, removing the256 fromRGB and
    # adding exactly the b512 levels declared by Generator(resolution=512).
    from migan_reference_contract import expected_state_shapes as shapes256
    shapes = shapes256()
    del shapes["encoder.b256.fromrgb.weight"]
    del shapes["encoder.b256.fromrgb.bias"]
    shapes.update({
        "encoder.b512.fromrgb.weight": (64, 4, 1, 1),
        "encoder.b512.fromrgb.bias": (64,),
        "encoder.b512.conv1.conv1.weight": (64, 1, 3, 3),
        "encoder.b512.conv1.conv1.bias": (64,),
        "encoder.b512.conv1.conv2.weight": (64, 64, 1, 1),
        "encoder.b512.conv2.conv1.weight": (64, 1, 3, 3),
        "encoder.b512.conv2.conv1.bias": (64,),
        "encoder.b512.conv2.conv2.weight": (128, 64, 1, 1),
        "encoder.b512.conv2.downsample.filter.weight": (64, 1, 4, 4),
        "synthesis.b512.conv1.conv1.weight": (128, 1, 3, 3),
        "synthesis.b512.conv1.conv1.bias": (128,),
        "synthesis.b512.conv1.conv2.weight": (64, 128, 1, 1),
        "synthesis.b512.conv1.upsample.filter_const": (1, 1, 512, 512),
        "synthesis.b512.conv1.upsample.filter.weight": (64, 1, 4, 4),
        "synthesis.b512.conv1.noise_const": (512, 512),
        "synthesis.b512.conv1.noise_strength": (),
        "synthesis.b512.conv2.conv1.weight": (64, 1, 3, 3),
        "synthesis.b512.conv2.conv1.bias": (64,),
        "synthesis.b512.conv2.conv2.weight": (64, 64, 1, 1),
        "synthesis.b512.conv2.noise_const": (512, 512),
        "synthesis.b512.conv2.noise_strength": (),
        "synthesis.b512.torgb.weight": (3, 64, 1, 1),
        "synthesis.b512.torgb.bias": (3,),
        "synthesis.b512.upsample.filter_const": (1, 1, 512, 512),
        "synthesis.b512.upsample.filter.weight": (3, 1, 4, 4),
    })
    return shapes


def validate_state_shapes(shapes):
    expected = expected_state_shapes()
    if set(shapes) != set(expected):
        raise ValueError(f"state keys differ: missing={sorted(set(expected)-set(shapes))}, extra={sorted(set(shapes)-set(expected))}")
    for name, shape in shapes.items():
        if tuple(shape) != expected[name]:
            raise ValueError(f"state shape differs: {name}: {shape} != {expected[name]}")


def load_contract():
    return json.loads(CONTRACT.read_text())


def validate_capture(report_path, checkpoint_sha256):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    if report.get("schema") != "crispembed-migan512-original-v1" or report.get("complete") is not True:
        raise ValueError("a complete original512 capture must precede conversion")
    if report.get("checkpoint_sha256") != checkpoint_sha256 or not valid_digest(checkpoint_sha256):
        raise ValueError("original512 checkpoint identity differs")
    if report.get("source_commit") != SOURCE_COMMIT or report.get("contract_sha256") != sha256(CONTRACT):
        raise ValueError("original512 source/contract identity differs")
    if [row.get("id") for row in report.get("cases", [])] != list(CASES):
        raise ValueError("original512 cases differ")
    for row in report["cases"]:
        if any(row.get(k) != v for k, v in fixture_identity(row["id"]).items()):
            raise ValueError("original512 fixture identity differs")
        if row.get("known_pixels_bit_exact") is not True:
            raise ValueError("original512 known pixels differ")
        if row["id"] in FULL_STAGE_CASES and row.get("hook_neutrality_bit_exact") is not True:
            raise ValueError("original512 hooks changed ordinary output")
        directory = report_path.parent / row["id"]
        names = set()
        stage_names = [stage["stage"] for stage in row["stages"]]
        required = {"generator.input", "known_mask", "generator.output"}
        if row["id"] in FULL_STAGE_CASES:
            required.update({"encoder.b512.fromrgb.output", "encoder.b512.output.0", "encoder.b512.output.1",
                             "synthesis.b512.output.0", "synthesis.b512.output.1"})
        if len(stage_names) != len(set(stage_names)) or not required.issubset(stage_names):
            raise ValueError("original512 mandatory/unique stages differ")
        for stage in row["stages"]:
            filename = stage["file"]
            if Path(filename).name != filename or filename in names or sha256(directory / filename) != stage["sha256"]:
                raise ValueError("original512 stage payload differs")
            names.add(filename)
        for filename, key in (("generated.png", "generated_png_sha256"), ("composited.png", "composited_png_sha256"),
                              ("reference.gguf", "reference_gguf_sha256"), ("input.rgb8", "input_rgb8_sha256"),
                              ("fill-mask.u8", "fill_mask_u8_sha256"), ("composited.rgb8", "composited_rgb8_sha256")):
            if sha256(directory / filename) != row[key]:
                raise ValueError("original512 image/reference identity differs")
    if report.get("rejected_controls") != ["nonbinary", "mask_size", "rgb_channels", "wrong_resolution"]:
        raise ValueError("original512 negative controls differ")
    return report
