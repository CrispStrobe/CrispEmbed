#!/usr/bin/env python3
"""Stdlib-only controls for the hash-pinned MI-GAN byte replication helper."""

import hashlib
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_migan_inpaint as prepare


def _text(value):
    encoded = value.encode("utf-8")
    return b"X" + struct.pack("<I", len(encoded)) + encoded


def _integer(value):
    return b"J" + struct.pack("<i", value)


def _tuple(values):
    return b"(" + b"".join(_integer(value) for value in values) + b"t"


def _state_pickle(name, count, shape, stride, offset=0, global_name="_rebuild_tensor_v2"):
    # A real protocol-2 tensor metadata pickle assembled without importing torch
    # or any mock tensor framework. The storage is a BINPERSID record.
    storage = (b"(" + _text("storage") + b"ctorch\nFloatStorage\n" + _text("0")
               + _text("cpu") + _integer(count) + b"tQ")
    tensor = (b"ctorch._utils\n" + global_name.encode() + b"\n(" + storage
              + _integer(offset) + _tuple(shape) + _tuple(stride)
              + b"\x89ccollections\nOrderedDict\n)RtR")
    return b"\x80\x02}" + _text(name) + tensor + b"s."


def _archive(path, raw, shape, stride, offset=0, pickle_data=None, byteorder=b"little"):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle_data if pickle_data is not None else
                         _state_pickle("weight", len(raw) // 4, shape, stride, offset))
        archive.writestr("archive/data/0", raw)
        if byteorder is not None:
            archive.writestr("archive/byteorder", byteorder)


def _row(shape, raw):
    return {"name": "weight", "shape": list(shape), "dtype": "float32",
            "sha256": hashlib.sha256(raw).hexdigest()}


class DownloadResponse(io.BytesIO):
    def __init__(self, body, content_type="application/octet-stream"):
        super().__init__(body)
        self.headers = {"Content-Type": content_type}


class PrepareMiganTest(unittest.TestCase):
    def test_review_is_exact_and_contains_all_tensor_bytes(self):
        review, contract = prepare.load_review()
        self.assertEqual(len(review["producer"]["state"]), 154)
        self.assertEqual(review["conversion"]["gguf_sha256"], prepare.GGUF_SHA256)
        self.assertEqual(contract["checkpoint"]["declared_bytes"], prepare.CHECKPOINT_BYTES)
        self.assertEqual(review["producer"]["source_commit"], prepare.SOURCE_COMMIT)

    def test_modified_review_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            path.write_text("{}")
            with mock.patch.object(prepare, "REVIEW_PATH", path):
                with self.assertRaisesRegex(ValueError, "frozen"):
                    prepare.load_review()

    def test_metadata_preserves_original_converter_types_and_order(self):
        metadata = prepare.metadata()
        self.assertEqual(list(metadata), [
            "general.architecture", "general.name", "general.file_type", "general.license", "migan.image_size",
            "migan.source.commit", "migan.checkpoint.sha256", "migan.input.contract", "migan.tensor.layout"])
        self.assertIs(type(metadata["general.file_type"]), int)
        self.assertEqual(metadata["migan.input.contract"], "RGB8;binary-mask8;255=fill;fixed256")

    def test_contiguous_storage_and_scalar_have_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            for shape, stride, values in (((2, 3), (3, 1), [1, 2, 3, 4, 5, 6]), ((), (), [0.125])):
                with self.subTest(shape=shape):
                    raw = struct.pack("<" + "f" * len(values), *values)
                    _archive(path, raw, shape, stride)
                    self.assertEqual(prepare.extract_tensors(path, [_row(shape, raw)]), [("weight", shape, raw)])

    def test_transposed_view_reproduces_contiguous_logical_tensor_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            storage = struct.pack("<4f", 1, 2, 3, 4)
            expected = struct.pack("<4f", 1, 3, 2, 4)
            _archive(path, storage, (2, 2), (1, 2))
            self.assertEqual(prepare.extract_tensors(path, [_row((2, 2), expected)]),
                             [("weight", (2, 2), expected)])

    def test_view_offset_and_singleton_stride_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            storage = struct.pack("<4f", -1, 2, 3, -4)
            expected = struct.pack("<2f", 2, 3)
            _archive(path, storage, (1, 2), (99, 1), offset=1)
            self.assertEqual(prepare.extract_tensors(path, [_row((1, 2), expected)]),
                             [("weight", (1, 2), expected)])

    def test_old_torch_archive_without_byteorder_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            raw = struct.pack("<f", 0.5)
            _archive(path, raw, (1,), (1,), byteorder=None)
            self.assertEqual(prepare.extract_tensors(path, [_row((1,), raw)]), [("weight", (1,), raw)])

    def test_ordered_state_dict_with_module_metadata(self):
        raw = struct.pack("<f", 0.5)
        basic = _state_pickle("weight", 1, (1,), (1,))
        # torch.nn.Module.state_dict() adds its _metadata attribute via BUILD.
        ordered = (b"\x80\x02ccollections\nOrderedDict\n)R" + basic[3:-1]
                   + b"}" + _text("_metadata") + b"}" + _text("") + b"}" + _text("version")
                   + _integer(1) + b"sssb.")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            _archive(path, raw, (1,), (1,), pickle_data=ordered)
            self.assertEqual(prepare.extract_tensors(path, [_row((1,), raw)]), [("weight", (1,), raw)])

    def test_foreign_pickle_globals_are_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            _archive(path, b"\0" * 4, (1,), (1,), pickle_data=b"\x80\x02cos\nsystem\n.")
            with self.assertRaisesRegex(ValueError, "unsupported pickle global: os.system"):
                prepare.extract_tensors(path, [])

    def test_wrong_dtype_byteorder_and_trailing_pickle_are_rejected(self):
        raw = struct.pack("<f", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            _archive(path, raw, (1,), (1,), byteorder=b"big")
            with self.assertRaisesRegex(ValueError, "little-endian"):
                prepare.extract_tensors(path, [_row((1,), raw)])
            _archive(path, raw, (1,), (1,))
            bad = _row((1,), raw)
            bad["dtype"] = "float16"
            with self.assertRaisesRegex(ValueError, "shape/type"):
                prepare.extract_tensors(path, [bad])
            _archive(path, raw, (1,), (1,), pickle_data=_state_pickle("weight", 1, (1,), (1,)) + b"extra")
            with self.assertRaisesRegex(ValueError, "trailing pickle"):
                prepare.extract_tensors(path, [_row((1,), raw)])

    def test_changed_tensor_payload_shape_inventory_and_view_are_rejected(self):
        raw = struct.pack("<f", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            _archive(path, raw, (1,), (1,))
            with self.assertRaisesRegex(ValueError, "tensor bytes"):
                prepare.extract_tensors(path, [_row((1,), struct.pack("<f", 2))])
            with self.assertRaisesRegex(ValueError, "shape/type"):
                prepare.extract_tensors(path, [_row((2,), raw)])
            with self.assertRaisesRegex(ValueError, "tensor names"):
                prepare.extract_tensors(path, [])
            with self.assertRaisesRegex(ValueError, "duplicate reviewed"):
                prepare.extract_tensors(path, [_row((1,), raw)] * 2)
            _archive(path, raw, (1,), (1,), offset=1)
            with self.assertRaisesRegex(ValueError, "outside"):
                prepare.extract_tensors(path, [_row((1,), raw)])

    def test_verified_download_and_failed_download_cleanup(self):
        raw = b"tiny authored payload"
        expected = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "download.pt"
            with mock.patch.object(prepare.urllib.request, "urlopen", return_value=DownloadResponse(raw)):
                prepare.download("https://example.invalid/model", path, len(raw), expected)
            self.assertEqual(path.read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                prepare.download("https://example.invalid/model", path, len(raw), expected)
            self.assertEqual(path.read_bytes(), raw)
            path.unlink()
            for response, size, digest, message in (
                    (DownloadResponse(raw + b"extra"), len(raw), expected, "exceeds"),
                    (DownloadResponse(raw[:-1]), len(raw), expected, "SHA-256 differs"),
                    (DownloadResponse(raw), len(raw), "0" * 64, "SHA-256 differs"),
                    (DownloadResponse(b"<html>", "text/html"), len(raw), expected, "HTML")):
                with mock.patch.object(prepare.urllib.request, "urlopen", return_value=response):
                    with self.assertRaisesRegex(ValueError, message):
                        prepare.download("https://example.invalid/model", path, size, digest)
                self.assertFalse(path.exists())

    def test_local_checkpoint_is_checked_before_metadata_and_does_not_publish_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "wrong.pt"
            checkpoint.write_bytes(b"not the reviewed checkpoint")
            destination = Path(directory) / "model"
            with mock.patch.object(prepare, "extract_tensors") as extract:
                with self.assertRaisesRegex(ValueError, "checkpoint byte count or SHA-256"):
                    prepare.prepare(destination, checkpoint=checkpoint)
                extract.assert_not_called()
            self.assertFalse(destination.exists())
            self.assertEqual(checkpoint.read_bytes(), b"not the reviewed checkpoint")
            self.assertEqual(list(Path(directory).iterdir()), [checkpoint])

    def test_existing_output_and_conflicting_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            sentinel = output / "keep"
            sentinel.write_text("keep")
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare.prepare(output, checkpoint="unused")
            self.assertEqual(sentinel.read_text(), "keep")
            for checkpoint, download in ((None, False), ("unused", True)):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    prepare.prepare(Path(directory) / "new", checkpoint, download)

    def test_no_framework_import(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)


if __name__ == "__main__":
    unittest.main()
