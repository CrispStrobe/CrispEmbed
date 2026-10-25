"""Pure OvisOCR3 original quality preflight and immutable ZIP controls."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ovisocr3_quality_run_test", ROOT / "tools/ci-heavy/ovisocr3_quality_run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class QualityRunControls(unittest.TestCase):
    def test_reference_absence_rejects_before_framework(self):
        with self.assertRaisesRegex(ValueError, "Reviewed original OvisOCR3 F32 module reference is absent"):
            run.require_original_reference()
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_immutable_prepared_zip_exact_members_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "prepared.zip"
            cases = []
            members = {"prepared.json": b"", "a.png": b"a", "b.png": b"b", "c.png": b"c"}
            for name in ("a", "b", "c"):
                cases.append({"id": name, "file": name + ".png",
                              "png_sha256": hashlib.sha256(members[name + ".png"]).hexdigest()})
            prepared = {"source_commit": "source", "cases": cases}
            members["prepared.json"] = json.dumps(prepared).encode()
            with zipfile.ZipFile(archive, "w") as zipped:
                for name, data in members.items():
                    zipped.writestr(name, data)
            preservation = {"bytes": archive.stat().st_size, "sha256": run.sha(archive),
                            "source": {"case_inputs": cases}}
            prior = {"artifact_report_sha256": hashlib.sha256(members["prepared.json"]).hexdigest(),
                     "source_commit": "source"}
            actual, unpacked = run.verify_prepared_archive(archive, preservation, prior)
            self.assertEqual(actual, prepared)
            self.assertEqual(unpacked, members)
            with self.assertRaises(ValueError):
                run.verify_prepared_archive(archive, dict(preservation, sha256="0" * 64), prior)
            bad = Path(directory) / "traversal.zip"
            with zipfile.ZipFile(bad, "w") as zipped:
                zipped.writestr("../prepared.json", members["prepared.json"])
                for name in ("a.png", "b.png", "c.png"):
                    zipped.writestr(name, members[name])
            with self.assertRaises(ValueError):
                run.verify_prepared_archive(bad, dict(preservation, bytes=bad.stat().st_size,
                                                       sha256=run.sha(bad)), prior)


if __name__ == "__main__":
    unittest.main()
