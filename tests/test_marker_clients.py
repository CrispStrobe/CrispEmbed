#!/usr/bin/env python3
"""Model-free regressions for Marker image input and saved bundle ownership."""
import ast
import ctypes
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from crispembed._marker import CrispMarker  # noqa: E402


class CaptureNative:
    def crispembed_marker_convert_raw(self, context, pixels, width, height, channels):
        self.image = (width, height, channels, ctypes.string_at(pixels, width * height * channels))
        return 1


class StubMarker(CrispMarker):
    """Keep the actual client code; replace only the native inference calls."""
    def __init__(self, assets=None):
        self._ctx, self._max_pixels, self._lock = 1, 1000, threading.RLock()
        self._lib = CaptureNative()
        self._assets = assets or {}

    def render(self, output_format="markdown"):
        return "new document\n"

    @property
    def metadata(self):
        return {"backend": "test-only-native-stub"}

    @property
    def assets(self):
        return self._assets

    def close(self):
        self._ctx = None


class MarkerClientTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="marker-client-", dir=ROOT / "tests/results/marker")
        self.folder = Path(self.temporary.name)
        self.output = self.folder / "document.md"
        self.meta = self.folder / "document_meta.json"

    def tearDown(self):
        self.temporary.cleanup()

    def sentinel_output(self):
        self.output.write_bytes(b"existing document")
        self.meta.write_bytes(b"existing metadata")

    def assert_output_unchanged(self):
        self.assertEqual(self.output.read_bytes(), b"existing document")
        self.assertEqual(self.meta.read_bytes(), b"existing metadata")

    def roundtrip_png(self, image, **options):
        encoded = io.BytesIO()
        image.save(encoded, format="PNG", **options)
        encoded.seek(0)
        return Image.open(encoded)

    def test_palette_png_transparency_reaches_native_alpha(self):
        image = Image.new("P", (2, 1))
        image.putpalette([0, 0, 0, 12, 34, 56] + [0] * 762)
        image.putdata([0, 1])
        decoded = self.roundtrip_png(image, transparency=0)
        self.assertEqual(decoded.mode, "P")
        marker = StubMarker()
        marker.convert_image(decoded)
        self.assertEqual(marker._lib.image, (2, 1, 4, bytes([0, 0, 0, 0, 12, 34, 56, 255])))

    def test_rgb_png_trns_reaches_native_alpha(self):
        image = Image.new("RGB", (2, 1))
        image.putdata([(12, 34, 56), (0, 0, 0)])
        decoded = self.roundtrip_png(image, transparency=(12, 34, 56))
        self.assertEqual(decoded.mode, "RGB")
        marker = StubMarker()
        marker.convert_image(decoded)
        self.assertEqual(marker._lib.image, (2, 1, 4, bytes([12, 34, 56, 0, 0, 0, 0, 255])))

    def test_opaque_pil_is_rgba_and_numpy_rgb_is_unchanged(self):
        marker = StubMarker()
        marker.convert_image(Image.new("RGB", (1, 1), (1, 2, 3)))
        self.assertEqual(marker._lib.image, (1, 1, 4, bytes([1, 2, 3, 255])))
        marker.convert_image(np.array([[[4, 5, 6]]], dtype=np.uint8))
        self.assertEqual(marker._lib.image, (1, 1, 3, bytes([4, 5, 6])))

    def test_file_convert_method_is_byte_unchanged(self):
        def method(path):
            source = path.read_text()
            tree = ast.parse(source)
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CrispMarker")
            fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "convert")
            return ast.get_source_segment(source, fn).encode("utf-8")
        self.assertEqual(method(ROOT / "python/crispembed/_marker.py"),
                         method(ROOT / "tests/results/marker/client-review-before/_marker.py"))

    def test_conflicting_asset_leaves_all_existing_files_unchanged(self):
        self.sentinel_output()
        asset = self.folder / "page_0_figure_0.png"
        asset.write_bytes(b"old figure")
        with self.assertRaisesRegex(RuntimeError, "separate directory per document"):
            StubMarker({asset.name: b"new figure"}).save(self.output)
        self.assert_output_unchanged()
        self.assertEqual(asset.read_bytes(), b"old figure")

    def test_all_assets_are_preflighted_before_creating_any(self):
        self.sentinel_output()
        conflict = self.folder / "second.png"
        conflict.write_bytes(b"previous")
        with self.assertRaises(RuntimeError):
            StubMarker({"first.png": b"new", "second.png": b"different"}).save(self.output)
        self.assertFalse((self.folder / "first.png").exists())
        self.assert_output_unchanged()

    def test_identical_asset_reused_without_timestamp_change(self):
        asset = self.folder / "figure.png"
        asset.write_bytes(b"same\0binary\xff")
        os.utime(asset, ns=(1234567890000000000, 1234567890000000000))
        before = asset.stat()
        StubMarker({asset.name: asset.read_bytes()}).save(self.output)
        self.assertEqual(self.output.read_text(), "new document\n")
        self.assertEqual(asset.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(asset.stat().st_ino, before.st_ino)
        self.assertEqual(asset.read_bytes(), b"same\0binary\xff")

    def test_new_bundle_creates_parent_and_preserves_binary_asset(self):
        path = self.folder / "new" / "folder" / "document.md"
        StubMarker({"figure.png": b"\0\xff\n"}).save(path)
        self.assertEqual(path.read_text(), "new document\n")
        self.assertEqual((path.parent / "figure.png").read_bytes(), b"\0\xff\n")
        self.assertIn("test-only-native-stub", (path.parent / "document_meta.json").read_text())

    def test_unsafe_asset_names_rejected_before_writes(self):
        self.sentinel_output()
        for name in ("../escape.png", "/absolute.png", "back\\slash.png", "nul\0.png", "new\n.png", "", ".", "..", "NUL.png", "LPT1.png", "dot."):
            with self.subTest(name=repr(name)), self.assertRaises(RuntimeError):
                StubMarker({"first.png": b"new", name: b"bad"}).save(self.output)
            self.assertFalse((self.folder / "first.png").exists())
            self.assert_output_unchanged()

    def test_asset_must_not_alias_output_or_metadata(self):
        self.sentinel_output()
        for name in (self.output.name, self.meta.name):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "collides"):
                StubMarker({name: b"existing document"}).save(self.output)
            self.assert_output_unchanged()

    def test_output_symlink_is_rejected_without_touching_target(self):
        target = self.folder / "target.md"
        target.write_bytes(b"outside target")
        self.output.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            StubMarker({"new.png": b"new"}).save(self.output)
        self.assertEqual(target.read_bytes(), b"outside target")
        self.assertFalse(self.meta.exists())
        self.assertFalse((self.folder / "new.png").exists())

    def test_metadata_symlink_is_rejected_before_primary_write(self):
        self.output.write_bytes(b"existing document")
        target = self.folder / "target.json"
        target.write_bytes(b"outside metadata")
        self.meta.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            StubMarker().save(self.output)
        self.assertEqual(self.output.read_bytes(), b"existing document")
        self.assertEqual(target.read_bytes(), b"outside metadata")

    def test_asset_symlink_rejected_even_if_bytes_match(self):
        self.sentinel_output()
        target = self.folder / "outside.png"
        target.write_bytes(b"same")
        (self.folder / "figure.png").symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            StubMarker({"figure.png": b"same"}).save(self.output)
        self.assert_output_unchanged()
        self.assertEqual(target.read_bytes(), b"same")

    def test_parent_symlink_and_nonregular_output_rejected(self):
        real = self.folder / "real"
        real.mkdir()
        link = self.folder / "link"
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            StubMarker().save(link / "document.md")
        self.assertEqual(list(real.iterdir()), [])
        with self.assertRaisesRegex(RuntimeError, "regular file"):
            StubMarker().save(real)

    def test_asset_hardlink_to_output_is_rejected(self):
        self.sentinel_output()
        asset = self.folder / "figure.png"
        os.link(self.output, asset)
        with self.assertRaisesRegex(RuntimeError, "collides"):
            StubMarker({asset.name: self.output.read_bytes()}).save(self.output)
        self.assert_output_unchanged()

    def test_metadata_hardlink_to_output_is_rejected(self):
        self.output.write_bytes(b"existing document")
        os.link(self.output, self.meta)
        with self.assertRaisesRegex(RuntimeError, "destinations collide"):
            StubMarker().save(self.output)
        self.assertEqual(self.output.read_bytes(), b"existing document")
        self.assertEqual(self.meta.read_bytes(), b"existing document")

    def test_duplicate_asset_hardlinks_rejected_before_writes(self):
        self.sentinel_output()
        first = self.folder / "first.png"
        first.write_bytes(b"same")
        os.link(first, self.folder / "second.png")
        with self.assertRaisesRegex(RuntimeError, "destinations collide"):
            StubMarker({"first.png": b"same", "second.png": b"same"}).save(self.output)
        self.assert_output_unchanged()


if __name__ == "__main__":
    unittest.main(verbosity=2)
