#!/usr/bin/env python3
"""Exercise the actual CLI implementation with the explicitly test-only driver.

The driver substitutes deterministic model/detector calls; PNG decoding, mask
geometry, argument parsing, preflight, batch ordering and output writes are real.
No numerical or model-quality claim follows from these tests.
"""

import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
FILL = bytes((241, 31, 151))
BINARY = None
REAL_BINARY = None


def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def png_bytes(width, height, pixels, channels=3):
    assert len(pixels) == width * height * channels
    rows = b"".join(b"\0" + pixels[y * width * channels:(y + 1) * width * channels] for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2 if channels == 3 else 0, 0, 0, 0)
    return PNG_SIGNATURE + png_chunk(b"IHDR", header) + png_chunk(b"IDAT", zlib.compress(rows)) + png_chunk(b"IEND", b"")


def decode_png(data):
    """Small independent decoder for the noninterlaced RGB8 outputs under test."""
    if data[:8] != PNG_SIGNATURE:
        raise AssertionError("output is not PNG (stdout may be contaminated)")
    offset, compressed, metadata = 8, bytearray(), []
    width = height = None
    ended = False
    while offset < len(data):
        if len(data) - offset < 12:
            raise AssertionError("truncated PNG chunk")
        length = struct.unpack_from(">I", data, offset)[0]
        if length > len(data) - offset - 12:
            raise AssertionError("truncated PNG chunk payload")
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != crc:
            raise AssertionError("invalid output PNG CRC")
        if width is None and kind != b"IHDR":
            raise AssertionError("IHDR must be the first PNG chunk")
        offset += length + 12
        if kind == b"IHDR":
            if width is not None or length != 13:
                raise AssertionError("invalid or duplicate IHDR")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            if width <= 0 or height <= 0 or (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
                raise AssertionError("expected noninterlaced RGB8 PNG")
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"iTXt":
            metadata.append(payload)
        elif kind == b"IEND":
            if length != 0 or offset != len(data):
                raise AssertionError("invalid PNG end or trailing bytes")
            ended = True
            break
    if not ended or not compressed:
        raise AssertionError("missing PNG image data or IEND")
    raw = zlib.decompress(compressed)
    stride = width * 3
    if len(raw) != (stride + 1) * height:
        raise AssertionError("unexpected output pixel length")
    previous = bytearray(stride)
    output = bytearray()
    for y in range(height):
        begin = y * (stride + 1)
        filter_type = raw[begin]
        row = bytearray(raw[begin + 1:begin + 1 + stride])
        for x in range(stride):
            left = row[x - 3] if x >= 3 else 0
            above = previous[x]
            corner = previous[x - 3] if x >= 3 else 0
            if filter_type == 0:
                prediction = 0
            elif filter_type == 1:
                prediction = left
            elif filter_type == 2:
                prediction = above
            elif filter_type == 3:
                prediction = (left + above) // 2
            elif filter_type == 4:
                p = left + above - corner
                distances = (abs(p - left), abs(p - above), abs(p - corner))
                prediction = (left, above, corner)[distances.index(min(distances))]
            else:
                raise AssertionError("unsupported PNG filter")
            row[x] = (row[x] + prediction) & 255
        output.extend(row)
        previous = row
    return width, height, bytes(output), b"\n".join(metadata)


def minimal_gguf(architecture):
    def string(value):
        value = value.encode("utf-8")
        return struct.pack("<Q", len(value)) + value
    data = (struct.pack("<4sIQQ", b"GGUF", 3, 0, 1) + string("general.architecture")
            + struct.pack("<I", 8) + string(architecture))
    return data + bytes((-len(data)) % 32)


class InpaintCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="inpaint-cli-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.width, self.height = 9, 7
        self.rgb = bytes((i * 17 + 11) % 233 for i in range(self.width * self.height * 3))
        self.mask = bytearray(self.width * self.height)
        for index, value in ((0, 127), (1, 128), (12, 255), (31, 255), (62, 255)):
            self.mask[index] = value
        self.image_path = self.root / "frame.png"
        self.mask_path = self.root / "mask.png"
        self.model_path = self.root / "model.gguf"
        self.detector_path = self.root / "dbnet.gguf"
        self.image_path.write_bytes(png_bytes(self.width, self.height, self.rgb))
        self.mask_path.write_bytes(png_bytes(self.width, self.height, self.mask, 1))
        self.model_path.write_bytes(b"explicit test model, never a real MI-GAN")
        self.detector_path.write_bytes(minimal_gguf("dbnet"))

    def run_cli(self, args, env=None, stdin=None, binary=None, cwd=None):
        environment = os.environ.copy()
        for key in list(environment):
            if key.startswith("INPAINT_TEST_") or key in (
                    "CRISPEMBED_IMAGE_FORMAT", "CRISPEMBED_C2PA_CERT", "CRISPEMBED_C2PA_KEY"):
                environment.pop(key, None)
        environment["CRISPEMBED_MIGAN"] = "1"
        for key, value in (env or {}).items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        return subprocess.run([str(binary or BINARY), *map(str, args)], cwd=cwd or self.root,
                              env=environment, input=stdin, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=30, check=False)

    def single(self, extra=(), **kwargs):
        return self.run_cli(["--inpaint-model", self.model_path, "--inpaint", self.image_path,
                             "--mask", self.mask_path, *extra], **kwargs)

    def expected(self, selected):
        output = bytearray(self.rgb)
        for i in selected:
            output[3 * i:3 * i + 3] = FILL
        return bytes(output)

    def assert_success_png(self, result, expected):
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        w, h, rgb, metadata = decode_png(result.stdout)
        self.assertEqual((w, h), (self.width, self.height))
        self.assertEqual(rgb, expected)
        self.assertIn(b"engine=migan-inpaint", metadata)
        self.assertEqual(result.stderr.count(b"INPAINT_TEST_LOAD"), 1)

    def test_help_without_gate_or_files(self):
        result = self.run_cli(["--inpaint-help"], env={"CRISPEMBED_MIGAN": None})
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"--mask-from-text", result.stdout)
        self.assertNotIn(b"INPAINT_TEST_LOAD", result.stderr)

    def test_unset_empty_and_zero_gate(self):
        for value in (None, "", "0"):
            with self.subTest(value=value):
                result = self.single(env={"CRISPEMBED_MIGAN": value})
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b"")
                self.assertIn(b"CRISPEMBED_MIGAN=1", result.stderr)
                self.assertNotIn(b"INPAINT_TEST_LOAD", result.stderr)

    def test_threshold_polarity_png_and_known_bytes(self):
        self.assert_success_png(self.single(), self.expected(i for i, value in enumerate(self.mask) if value >= 128))

    def test_empty_mask(self):
        self.mask_path.write_bytes(png_bytes(self.width, self.height, bytes(len(self.mask)), 1))
        self.assert_success_png(self.single(), self.rgb)

    def test_dilation_then_region(self):
        mask = bytearray(len(self.mask))
        mask[3 * self.width + 4] = 255
        self.mask_path.write_bytes(png_bytes(self.width, self.height, mask, 1))
        result = self.single(["--mask-dilate", "2", "--mask-region", "3,2,3,3"])
        selected = [y * self.width + x for y in range(2, 5) for x in range(3, 6)]
        self.assert_success_png(result, self.expected(selected))

    def test_invalid_options_before_model_load(self):
        cases = [
            ["--threads", "65"], ["--threads", "2junk"], ["--mask-dilate", "-1"],
            ["--mask-region", "0,0,0,3"], ["--mask-region", "0,0,2,2,2"],
            ["--mask-region", "8,0,2,2"], ["--inpaint-dir", self.root],
            ["--mask", self.mask_path], ["--inpaint-temporal"], ["--mask-from-text"],
            ["--inpaint-output"], ["--inpaint-overwrite"],
        ]
        for extra in cases:
            with self.subTest(extra=extra):
                result = self.single(extra)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn(b"INPAINT_TEST_LOAD", result.stderr)
                self.assertEqual(result.stdout, b"")

    def test_mask_dimensions_preflight(self):
        self.mask_path.write_bytes(png_bytes(2, 2, bytes(4), 1))
        result = self.single()
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"dimensions must match", result.stderr)
        self.assertNotIn(b"INPAINT_TEST_LOAD", result.stderr)

    def test_non_png_format_is_not_mislabeled(self):
        result = self.single(env={"CRISPEMBED_IMAGE_FORMAT": "ppm"})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")

    def test_existing_output_requires_explicit_overwrite(self):
        path = self.root / "output.png"
        path.write_bytes(b"previous output")
        result = self.single(["--inpaint-output", path])
        self.assertEqual(result.returncode, 2)
        self.assertEqual(path.read_bytes(), b"previous output")
        result = self.single(["--inpaint-output", path, "--inpaint-overwrite"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(decode_png(path.read_bytes())[2], self.expected(i for i, v in enumerate(self.mask) if v >= 128))

    def test_protect_input_mask_and_model(self):
        for path in (self.image_path, self.mask_path, self.model_path):
            with self.subTest(path=path.name):
                before = path.read_bytes()
                result = self.single(["--inpaint-output", path, "--inpaint-overwrite"])
                self.assertEqual(result.returncode, 2)
                self.assertEqual(path.read_bytes(), before)

    def test_inference_failure_preserves_previous_output(self):
        path = self.root / "output.png"
        path.write_bytes(b"previous output")
        result = self.single(["--inpaint-output", path, "--inpaint-overwrite"],
                             env={"INPAINT_TEST_FAIL_AT": "1"})
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"frame 1", result.stderr)
        self.assertEqual(path.read_bytes(), b"previous output")
        self.assertEqual(result.stdout, b"")

    def test_decode_failure_never_creates_output(self):
        self.mask_path.write_bytes(self.mask_path.read_bytes()[:33])
        path = self.root / "output.png"
        result = self.single(["--inpaint-output", path])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(path.exists())

    def test_directory_order_and_one_loaded_context(self):
        source, masks, output = (self.root / name for name in ("frames", "masks", "results"))
        source.mkdir()
        masks.mkdir()
        for name in ("0002.png", "0001.png", "0003.PNG"):
            (source / name).write_bytes(self.image_path.read_bytes())
            (masks / name).write_bytes(self.mask_path.read_bytes())
        (source / "ignore.txt").write_text("not a frame", encoding="utf-8")
        result = self.run_cli(["--inpaint-model", self.model_path, "--inpaint-dir", source,
                               "--mask-dir", masks, "--output-dir", output])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.count(b"INPAINT_TEST_LOAD"), 1)
        self.assertIn(b"INPAINT_TEST_FREE frames=3", result.stderr)
        self.assertLess(result.stderr.index(b"0001.png"), result.stderr.index(b"0002.png"))
        self.assertEqual(sorted(p.name for p in output.iterdir()), ["0001.png", "0002.png", "0003.PNG"])
        self.assertEqual(result.stdout, b"")

    def test_batch_preflights_later_masks_before_first_write(self):
        manifest = self.root / "frames.tsv"
        manifest.write_text("frame.png\tmask.png\tfirst.png\nframe.png\tmissing.png\tsecond.png\n", encoding="utf-8")
        result = self.run_cli(["--inpaint-model", self.model_path, "--inpaint-list", manifest])
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / "first.png").exists())
        self.assertNotIn(b"INPAINT_TEST_LOAD", result.stderr)

    def test_manifest_paths_relative_to_manifest(self):
        manifest = self.root / "frames.tsv"
        manifest.write_text("# Frames\r\nframe.png\tmask.png\tout one.png\r\nframe.png\tmask.png\tout two.png\n", encoding="utf-8")
        other = self.root / "elsewhere"
        other.mkdir()
        result = self.run_cli(["--inpaint-model", self.model_path, "--inpaint-list", manifest], cwd=other)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "out one.png").exists())
        self.assertTrue((self.root / "out two.png").exists())
        self.assertEqual(result.stderr.count(b"INPAINT_TEST_LOAD"), 1)

    def test_stdin_list(self):
        result = self.run_cli(["--inpaint-model", self.model_path, "--inpaint-list", "-"],
                              stdin=b"frame.png\tmask.png\tout.png\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "out.png").exists())
        self.assertEqual(result.stdout, b"")

    def test_duplicate_and_cross_frame_input_outputs_reject(self):
        for rows in ("frame.png\tmask.png\tout.png\nframe.png\tmask.png\tout.png\n",
                     "frame.png\tmask.png\tout.png\nframe.png\tmask.png\tframe.png\n"):
            with self.subTest(rows=rows):
                result = self.run_cli(["--inpaint-model", self.model_path, "--inpaint-list", "-",
                                       "--inpaint-overwrite"], stdin=rows.encode())
                self.assertEqual(result.returncode, 2)
                self.assertFalse((self.root / "out.png").exists())

    def test_bad_manifest_rows(self):
        for row in (b"frame.png mask.png out.png\n", b"frame.png\tmask.png\t-\n",
                    b"frame.png\tmask.png\tout.png\textra\n", b"frame.png\t\tout.png\n"):
            with self.subTest(row=row):
                result = self.run_cli(["--inpaint-model", self.model_path, "--inpaint-list", "-"], stdin=row)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn(b"INPAINT_TEST_LOAD", result.stderr)

    def test_batch_fails_at_exact_frame_and_retains_completed_output(self):
        rows = b"frame.png\tmask.png\tfirst.png\nframe.png\tmask.png\tsecond.png\nframe.png\tmask.png\tthird.png\n"
        result = self.run_cli(["--inpaint-model", self.model_path, "--inpaint-list", "-"], stdin=rows,
                              env={"INPAINT_TEST_FAIL_AT": "2"})
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"frame 2", result.stderr)
        self.assertTrue((self.root / "first.png").exists())
        self.assertFalse((self.root / "second.png").exists())
        self.assertFalse((self.root / "third.png").exists())
        self.assertEqual(result.stderr.count(b"INPAINT_TEST_LOAD"), 1)

    def text_mask(self, extra=(), **kwargs):
        return self.run_cli(["--inpaint-model", self.model_path, "--inpaint", self.image_path,
                             "--mask-from-text", "--mask-text-model", self.detector_path, *extra], **kwargs)

    def test_text_quadrilateral_and_region(self):
        result = self.text_mask(["--mask-dilate", "0", "--mask-region", "3,3,3,2"])
        self.assert_success_png(result, self.expected(y * self.width + x for y in (3, 4) for x in (3, 4)))

    def test_text_default_dilation(self):
        result = self.text_mask()
        self.assert_success_png(result, self.expected(y * self.width + x for y in range(7) for x in range(8)))

    def test_no_text_is_successful_passthrough(self):
        self.assert_success_png(self.text_mask(env={"INPAINT_TEST_NO_TEXT": "1"}), self.rgb)

    def test_detector_failure_and_nonfinite_output_are_errors(self):
        for name in ("INPAINT_TEST_DETECT_FAIL", "INPAINT_TEST_DETECT_NAN"):
            with self.subTest(name=name):
                result = self.text_mask(env={name: "1"})
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, b"")

    def test_wrong_detector_architecture_rejects(self):
        self.detector_path.write_bytes(minimal_gguf("surya"))
        result = self.text_mask()
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"requires a DBNet", result.stderr)
        self.assertNotIn(b"INPAINT_TEST_LOAD", result.stderr)

    def test_text_mask_list_sentinel(self):
        result = self.run_cli(["--inpaint-model", self.model_path, "--inpaint-list", "-",
                               "--mask-from-text", "--mask-text-model", self.detector_path],
                              stdin=b"frame.png\t-\ttext.png\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "text.png").exists())

    def test_real_binary_routing_and_model_rejection(self):
        if REAL_BINARY is None:
            self.skipTest("--real-binary not supplied")
        help_result = self.run_cli(["--inpaint-help"], binary=REAL_BINARY, env={"CRISPEMBED_MIGAN": None})
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(b"--inpaint-list", help_result.stdout)
        result = self.single(binary=REAL_BINARY)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(b"SHA-256 mismatch", result.stderr)
        self.assertEqual(result.stdout, b"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--real-binary", type=Path)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve()
    REAL_BINARY = args.real_binary.resolve() if args.real_binary else None
    unittest.main(argv=[__file__, *remaining])
