#!/usr/bin/env python3
"""Bounded photo-fixture integrity controls; no framework/model execution."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("migan_photo", ROOT / "tools/dump_migan_photo_reference.py")
PHOTO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PHOTO)


class PhotoFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "photos"
        PHOTO.prepare(256, self.directory)
        self.manifest = self.directory / "fixtures.json"

    def rewrite(self, edit):
        value = json.loads(self.manifest.read_text())
        edit(value)
        PHOTO.write_json(self.manifest, value)

    def test_pinned_photos_and_masks_are_accepted(self):
        value = PHOTO.load_fixtures(self.manifest)
        self.assertEqual([case["fill_pixels"] for case in value["cases"]], [6144, 4096])
        self.assertTrue(all(case["license"] == "Public-Domain" for case in value["cases"]))

    def test_second_resolution_masks_have_exact_integer_area(self):
        for size in (256, 512):
            self.assertEqual(PHOTO.mask_bytes(size, "bar").count(255), size * size * 3 // 32)
            self.assertEqual(PHOTO.mask_bytes(size, "center").count(255), size * size // 16)
            self.assertEqual(set(PHOTO.mask_bytes(size, "bar")), {0, 255})

    def test_self_declared_hash_cannot_replace_public_photo(self):
        rgb = self.directory / PHOTO.SOURCES[0]["name"] / "input.rgb8"
        data = bytearray(rgb.read_bytes())
        data[0] ^= 1
        rgb.write_bytes(data)
        self.rewrite(lambda doc: doc["cases"][0].update(rgb_sha256=PHOTO.digest(rgb)))
        with self.assertRaisesRegex(ValueError, "crop differs"):
            PHOTO.load_fixtures(self.manifest)

    def test_self_declared_hash_cannot_replace_mask_geometry(self):
        mask = self.directory / PHOTO.SOURCES[0]["name"] / "fill-mask.u8"
        data = bytearray(mask.read_bytes())
        data[0] = 255
        mask.write_bytes(data)
        self.rewrite(lambda doc: doc["cases"][0].update(mask_sha256=PHOTO.digest(mask)))
        with self.assertRaisesRegex(ValueError, "mask differs"):
            PHOTO.load_fixtures(self.manifest)

    def test_crop_and_provenance_edits_are_rejected(self):
        original = self.manifest.read_bytes()
        for fields in ({"crop_xyxy": [0, 0, 256, 256]}, {"license": "unknown"},
                       {"source_page": "https://example.invalid"}):
            self.manifest.write_bytes(original)
            self.rewrite(lambda doc: doc["cases"][0].update(fields))
            with self.assertRaises(ValueError):
                PHOTO.load_fixtures(self.manifest)

    def test_unknown_or_nonbinary_geometry_is_rejected(self):
        for size, kind in ((255, "bar"), (1024, "bar"), (256, "freehand")):
            with self.assertRaises(ValueError):
                PHOTO.mask_bytes(size, kind)


if __name__ == "__main__":
    unittest.main()
