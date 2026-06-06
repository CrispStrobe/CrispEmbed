#!/usr/bin/env python3
"""Unit tests for the CrispFace and CrispFacePipeline Python wrappers.

Environment variables (all optional — tests that need them are skipped
if unset):

    CRISPEMBED_LIB          Path to libcrispembed.{so,dylib,dll}
    CRISPEMBED_DET_MODEL    Path to a face-detection GGUF (e.g. yunet.gguf)
    CRISPEMBED_REC_MODEL    Path to a face-recognition GGUF (e.g. auraface-v1.gguf)
    CRISPEMBED_TEST_IMAGE   Path to an image with at least one face

Usage:
    python tests/test_face_python.py
    pytest tests/test_face_python.py -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

LIB = os.environ.get("CRISPEMBED_LIB")
DET_MODEL = os.environ.get("CRISPEMBED_DET_MODEL")
REC_MODEL = os.environ.get("CRISPEMBED_REC_MODEL")
TEST_IMAGE = os.environ.get("CRISPEMBED_TEST_IMAGE")

HAVE_MODELS = DET_MODEL and REC_MODEL and TEST_IMAGE
skip_reason = "Set CRISPEMBED_DET_MODEL, CRISPEMBED_REC_MODEL, CRISPEMBED_TEST_IMAGE"


def _need_models(fn):
    return unittest.skipUnless(HAVE_MODELS, skip_reason)(fn)


class TestCrispFaceDetection(unittest.TestCase):
    """Tests for CrispFace in detection mode."""

    @classmethod
    def setUpClass(cls):
        if not HAVE_MODELS:
            return
        from crispembed import CrispFace
        cls.det = CrispFace(DET_MODEL, n_threads=2, lib_path=LIB)

    @_need_models
    def test_detect_returns_faces(self):
        faces = self.det.detect(TEST_IMAGE, conf=0.3)
        self.assertGreater(len(faces), 0, "Expected at least one face")

    @_need_models
    def test_detection_fields(self):
        faces = self.det.detect(TEST_IMAGE, conf=0.3)
        f = faces[0]
        for key in ("x", "y", "w", "h", "confidence", "landmarks"):
            self.assertIn(key, f, f"Missing key: {key}")
        self.assertGreater(f["confidence"], 0.0)
        self.assertGreater(f["w"], 0.0)
        self.assertGreater(f["h"], 0.0)
        self.assertEqual(len(f["landmarks"]), 10)

    @_need_models
    def test_detect_no_faces(self):
        # Create a tiny blank image — should find no faces
        try:
            import struct
            # Minimal 2x2 white BMP
            bmp = _make_blank_bmp(2, 2)
            with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp:
                tmp.write(bmp)
                tmp_path = tmp.name
            faces = self.det.detect(tmp_path, conf=0.9)
            self.assertEqual(len(faces), 0)
        finally:
            os.unlink(tmp_path)

    @_need_models
    def test_detector_dim_is_zero(self):
        self.assertEqual(self.det.dim, 0)

    @_need_models
    def test_detector_model_type(self):
        self.assertIn(self.det.model_type, ("detection", "scrfd", "yunet", ""))


class TestCrispFaceRecognition(unittest.TestCase):
    """Tests for CrispFace in recognition mode."""

    @classmethod
    def setUpClass(cls):
        if not HAVE_MODELS:
            return
        from crispembed import CrispFace
        cls.det = CrispFace(DET_MODEL, n_threads=2, lib_path=LIB)
        cls.rec = CrispFace(REC_MODEL, n_threads=2, lib_path=LIB)

    @_need_models
    def test_encode_returns_embedding(self):
        faces = self.det.detect(TEST_IMAGE, conf=0.3)
        self.assertGreater(len(faces), 0)
        emb = self.rec.encode(TEST_IMAGE, faces[0]["landmarks"])
        self.assertEqual(emb.ndim, 1)
        self.assertGreater(emb.shape[0], 0)

    @_need_models
    def test_embedding_is_l2_normalized(self):
        faces = self.det.detect(TEST_IMAGE, conf=0.3)
        emb = self.rec.encode(TEST_IMAGE, faces[0]["landmarks"])
        norm = float(np.linalg.norm(emb))
        self.assertAlmostEqual(norm, 1.0, places=2)

    @_need_models
    def test_recognizer_dim_positive(self):
        self.assertGreater(self.rec.dim, 0)

    @_need_models
    def test_encode_bad_landmarks_raises(self):
        with self.assertRaises(ValueError):
            self.rec.encode(TEST_IMAGE, [0.0] * 8)  # need 10


class TestCrispFacePipeline(unittest.TestCase):
    """Tests for the joint detect+embed pipeline."""

    @classmethod
    def setUpClass(cls):
        if not HAVE_MODELS:
            return
        from crispembed import CrispFacePipeline
        cls.pipe = CrispFacePipeline(DET_MODEL, REC_MODEL, n_threads=2, lib_path=LIB)

    @_need_models
    def test_pipeline_returns_results(self):
        results = self.pipe.run(TEST_IMAGE, conf=0.3)
        self.assertGreater(len(results), 0)

    @_need_models
    def test_pipeline_result_structure(self):
        results = self.pipe.run(TEST_IMAGE, conf=0.3)
        r = results[0]
        self.assertIn("det", r)
        self.assertIn("embedding", r)
        self.assertIsInstance(r["embedding"], np.ndarray)
        self.assertGreater(r["embedding"].shape[0], 0)

    @_need_models
    def test_pipeline_embedding_normalized(self):
        results = self.pipe.run(TEST_IMAGE, conf=0.3)
        emb = results[0]["embedding"]
        norm = float(np.linalg.norm(emb))
        self.assertAlmostEqual(norm, 1.0, places=2)

    @_need_models
    def test_match_same_face_high_cos(self):
        results = self.pipe.run(TEST_IMAGE, conf=0.3)
        if len(results) < 1:
            self.skipTest("Need at least 1 face")
        emb = results[0]["embedding"]
        cos = self.pipe.match(emb, emb)
        self.assertGreater(cos, 0.99, "Same embedding should have cos~1.0")

    @_need_models
    def test_match_static_method(self):
        from crispembed import CrispFacePipeline
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        cos = CrispFacePipeline.match(a, b)
        self.assertAlmostEqual(cos, 0.0, places=3)


class TestCrispFaceFromRegistry(unittest.TestCase):
    """Tests for the from_registry() class method."""

    def test_from_registry_import(self):
        from crispembed import CrispFace, CrispFacePipeline
        self.assertTrue(hasattr(CrispFace, "from_registry"))
        self.assertTrue(hasattr(CrispFacePipeline, "from_registry"))


# ---- helpers ----

def _make_blank_bmp(w: int, h: int) -> bytes:
    """Create a minimal white BMP image in memory."""
    import struct
    row_size = (w * 3 + 3) & ~3
    img_size = row_size * h
    file_size = 54 + img_size
    header = struct.pack(
        "<2sIHHI IHHI HH IIIIII",
        b"BM", file_size, 0, 0, 54,
        40, w, h, 1, 24,
        0, img_size, 2835, 2835, 0, 0,
    )
    pixels = b"\xff\xff\xff" * w
    padding = b"\x00" * (row_size - w * 3)
    return header + (pixels + padding) * h


if __name__ == "__main__":
    unittest.main()
