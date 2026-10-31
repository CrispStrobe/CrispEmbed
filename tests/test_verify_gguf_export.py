"""Synthetic GGUF fixtures for the manifest verifier."""
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "models/verify-gguf-export.py"


def load():
    spec = importlib.util.spec_from_file_location("gguf_verifier", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_gguf(path, tensors, metadata):
    """tensors: list of (name, gguf_shape, little-endian f32 bytes)."""
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), len(metadata))
    for key, value in metadata.items():
        blob += struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 4) + struct.pack("<I", value)
    payload = b""
    offsets = []
    for name, shape, data in tensors:
        offsets.append((len(payload) + 31) // 32 * 32)
        payload += b"\0" * ((-len(payload)) % 32) + data
    header = blob
    for (name, shape, _), offset in zip(tensors, offsets):
        header += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", len(shape))
        header += b"".join(struct.pack("<Q", axis) for axis in shape)
        header += struct.pack("<IQ", 0, offset)
    header += b"\0" * ((-len(header)) % 32)
    Path(path).write_bytes(header + payload)


class GgufExportVerifierTest(unittest.TestCase):
    def test_payload_hash_uses_bounded_reads_and_rejects_truncation(self):
        verifier = load()
        class Tracked(io.BytesIO):
            max_read = 0
            def read(self, size: int | None = -1):
                self.max_read = max(self.max_read, size if size is not None else -1)
                return super().read(size)
        payload = b"abcd" * (verifier.HASH_CHUNK_BYTES // 4 + 5)
        reader = verifier.Reader.__new__(verifier.Reader)
        reader.data = 0
        reader.stream = Tracked(payload)
        self.assertEqual(reader.payload_sha256([len(payload) // 4], 0), hashlib.sha256(payload).hexdigest())
        self.assertLessEqual(reader.stream.max_read, verifier.HASH_CHUNK_BYTES)
        with self.assertRaisesRegex(ValueError, "truncated"):
            reader.payload_sha256([len(payload) // 4 + 1], 0)

    def test_matching_manifest_verifies(self):
        verifier = load()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            payload = b"".join(struct.pack("<f", value) for value in (1.0, 2.0, 3.0, 4.0))
            write_gguf(base / "model.gguf", [("t", [2, 2], payload)],
                       {"qwen35_ocr.profile": 0, "qwen35_ocr.primary_eos_id": 248046})
            manifest = {"export_tensor_count": 1, "records": [
                {"name": "t", "shape": [2, 2], "dtype": "float32",
                 "sha256": hashlib.sha256(payload).hexdigest()}]}
            (base / "model.gguf.manifest.json").write_text(json.dumps(manifest))
            report = verifier.verify(base / "model.gguf", base / "model.gguf.manifest.json")
            self.assertEqual(report["tensors"], 1)
            self.assertEqual(report["metadata"]["qwen35_ocr.primary_eos_id"], 248046)

    def test_reversed_axes_are_accepted_but_not_relabelled(self):
        verifier = load()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            payload = b"".join(struct.pack("<f", value) for value in range(6))
            write_gguf(base / "model.gguf", [("t", [3, 2], payload)], {})
            manifest = {"export_tensor_count": 1, "records": [
                {"name": "t", "shape": [2, 3], "dtype": "float32",
                 "sha256": hashlib.sha256(payload).hexdigest()}]}
            (base / "model.gguf.manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verifier.verify(base / "model.gguf", base / "model.gguf.manifest.json")["reversed_axes"], 1)

    def test_incorrect_axis_permutation_is_rejected(self):
        verifier = load()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            payload = b"".join(struct.pack("<f", value) for value in range(24))
            write_gguf(base / "model.gguf", [("t", [2, 4, 3], payload)], {})
            manifest = {"export_tensor_count": 1, "records": [
                {"name": "t", "shape": [2, 3, 4], "dtype": "float32",
                 "sha256": hashlib.sha256(payload).hexdigest()}]}
            (base / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "shape"):
                verifier.verify(base / "model.gguf", base / "manifest.json")

    def test_duplicate_tensor_names_are_rejected(self):
        verifier = load()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            payload = struct.pack("<f", 2.0)
            write_gguf(base / "model.gguf", [("t", [1], payload), ("t", [1], payload)], {})
            manifest = {"export_tensor_count": 1, "records": [
                {"name": "t", "shape": [1], "dtype": "float32",
                 "sha256": hashlib.sha256(payload).hexdigest()}]}
            (base / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                verifier.verify(base / "model.gguf", base / "manifest.json")

    def test_tampered_payload_missing_tensor_and_wrong_type_are_rejected(self):
        verifier = load()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            payload = b"".join(struct.pack("<f", value) for value in (1.0, 2.0))
            write_gguf(base / "model.gguf", [("t", [2], payload)], {})
            good = {"export_tensor_count": 1, "records": [
                {"name": "t", "shape": [2], "dtype": "float32",
                 "sha256": hashlib.sha256(payload).hexdigest()}]}
            cases = {
                "digest differs": {"export_tensor_count": 1, "records": [
                    {"name": "t", "shape": [2], "dtype": "float32", "sha256": "0" * 64}]},
                "Tensor set differs": {"export_tensor_count": 0, "records": []},
                "record count": {"export_tensor_count": 2, "records": good["records"]},
                "element count": {"export_tensor_count": 1, "records": [
                    {"name": "t", "shape": [4], "dtype": "float32",
                     "sha256": hashlib.sha256(payload).hexdigest()}]},
            }
            for message, manifest in cases.items():
                with self.subTest(message=message):
                    (base / "model.gguf.manifest.json").write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, message):
                        verifier.verify(base / "model.gguf", base / "model.gguf.manifest.json")
            bad = dict(good["records"][0], dtype="float16")
            (base / "model.gguf.manifest.json").write_text(
                json.dumps({"export_tensor_count": 1, "records": [bad]}))
            with self.assertRaisesRegex(ValueError, "manifest dtype"):
                verifier.verify(base / "model.gguf", base / "model.gguf.manifest.json")


if __name__ == "__main__":
    unittest.main()
