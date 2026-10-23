"""Small archive-integrity controls; no NumPy, model imports or execution."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

PATH = Path(__file__).resolve().parents[1] / 'tools/ci-heavy/ovisocr2_vllm_review.py'
SPEC = importlib.util.spec_from_file_location('ovis_review', PATH)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def npy(dtype='<i8', shape=(2,), payload=None):
    header = repr({'descr': dtype, 'fortran_order': False, 'shape': shape}).encode() + b'\n'
    payload = struct.pack('<qq', 1, 2) if payload is None else payload
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + payload


class ArchiveControls(unittest.TestCase):
    def test_payload_hash_and_shape_come_from_retained_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'array.npz'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('input_ids.npy', npy())
            self.assertEqual(review.npz_records(path), {'input_ids': {
                'shape': [2], 'dtype': 'int64',
                'sha256': hashlib.sha256(struct.pack('<qq', 1, 2)).hexdigest()}})

    def test_reject_truncation_object_dtype_and_unsafe_member(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'array.npz'
            for name, data in [('input.npy', npy(payload=b'')),
                               ('input.npy', npy(dtype='|O')),
                               ('../input.npy', npy()), ('input.npy', b'bad-header')]:
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr(name, data)
                with self.assertRaises(ValueError):
                    review.npz_records(path)

    def test_failed_real_original_attempt_cannot_unlock_native_model(self):
        root = PATH.parents[2]
        failure = json.loads((root / 'tests/results/qwen35-ocr/initial-ovisocr2-vllm-cuda-link-failure.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'ovisocr2-vllm-reference.json').write_text(json.dumps(failure['producer']))
            (path / 'ovisocr2-vllm-worker.json').write_text(json.dumps(failure['worker']))
            with self.assertRaisesRegex(ValueError, 'execution/precision/source'):
                review.review(path, {}, 'unused', 'unused')


if __name__ == '__main__':
    unittest.main()
