#!/usr/bin/env python3
"""Independent tiny PNG vectors for the inpainting CLI's pixel test oracle."""

from pathlib import Path
import struct
import sys
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inpaint_cli import decode_png, png_bytes


def chunk(kind, payload):
    return (len(payload).to_bytes(4, "big") + kind + payload
            + (zlib.crc32(kind + payload) & 0xFFFFFFFF).to_bytes(4, "big"))


def authored_png(width, height, filtered_rows, extra=b""):
    header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes((8, 2, 0, 0, 0))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + extra
            + chunk(b"IDAT", zlib.compress(filtered_rows)) + chunk(b"IEND", b""))


class InpaintPngOracleTests(unittest.TestCase):
    def test_five_png_filter_vectors_decode_to_known_pixels(self):
        # All encoded bytes are fixed, manually derived PNG filter examples.
        # This does not implement a second filter loop that can share a bug.
        first = (10, 20, 30, 50, 70, 90, 130, 170, 210)
        second = (15, 25, 35, 60, 80, 100, 140, 180, 220)
        encoded = [
            (first, second),
            ((10, 20, 30, 40, 50, 60, 80, 100, 120), (15, 25, 35, 45, 55, 65, 80, 100, 120)),
            (first, (5, 5, 5, 10, 10, 10, 10, 10, 10)),
            ((10, 20, 30, 45, 60, 75, 105, 135, 165), (10, 15, 20, 28, 33, 38, 45, 55, 65)),
            ((10, 20, 30, 40, 50, 60, 80, 100, 120), (5, 5, 5, 10, 10, 10, 10, 10, 10)),
        ]
        for filter_type, (row0, row1) in enumerate(encoded):
            with self.subTest(filter_type=filter_type):
                raw = bytes((filter_type, *row0, filter_type, *row1))
                self.assertEqual(decode_png(authored_png(3, 2, raw))[:3], (3, 2, bytes(first + second)))

    def test_paeth_corner_prediction_and_modulo_wrap(self):
        raw = bytes((4, 60, 60, 60, 40, 40, 40, 4, 196, 196, 196, 30, 30, 30))
        expected = bytes((60, 60, 60, 100, 100, 100, 0, 0, 0, 90, 90, 90))
        self.assertEqual(decode_png(authored_png(2, 2, raw))[:3], (2, 2, expected))

    def test_fixture_rgb_and_grayscale_encoding_have_expected_wire_data(self):
        rgb = bytes((0, 1, 2, 250, 251, 252))
        self.assertEqual(decode_png(png_bytes(2, 1, rgb))[:3], (2, 1, rgb))
        gray = png_bytes(3, 1, bytes((0, 127, 255)), 1)
        self.assertEqual(struct.unpack(">IIBBBBB", gray[16:29]), (3, 1, 8, 0, 0, 0, 0))
        idat_size = int.from_bytes(gray[33:37], "big")
        self.assertEqual(gray[37:41], b"IDAT")
        self.assertEqual(zlib.decompress(gray[41:41 + idat_size]), bytes((0, 0, 127, 255)))

    def test_metadata_and_multiple_idat_chunks(self):
        provenance = b"CrispEmbed\0\0\0\0\0engine=migan-inpaint\n"
        raw = b"\0\1\2\3"
        png = authored_png(1, 1, raw, chunk(b"iTXt", provenance))
        self.assertEqual(decode_png(png), (1, 1, b"\1\2\3", provenance))
        compressed = zlib.compress(raw)
        header = png[:33]
        multi = (header + chunk(b"IDAT", compressed[:3]) + chunk(b"IDAT", compressed[3:])
                 + chunk(b"IEND", b""))
        self.assertEqual(decode_png(multi)[:3], (1, 1, b"\1\2\3"))

    def test_missing_end_truncation_contamination_and_crc_fail(self):
        png = authored_png(1, 1, b"\0\1\2\3")
        wrong_crc = bytearray(png)
        wrong_crc[29] ^= 1
        bad = (png[:-12], png[:7], png[:35], png[:-1], png + b"log output", bytes(wrong_crc),
               b"log output\n" + png)
        for index, value in enumerate(bad):
            with self.subTest(index=index), self.assertRaises(AssertionError):
                decode_png(value)

    def test_invalid_filter_and_wrong_pixel_count_fail(self):
        for rows in (b"\5\1\2\3", b"\0\1\2", b"\0\1\2\3\4"):
            with self.subTest(rows=rows), self.assertRaises(AssertionError):
                decode_png(authored_png(1, 1, rows))


if __name__ == "__main__":
    unittest.main()
