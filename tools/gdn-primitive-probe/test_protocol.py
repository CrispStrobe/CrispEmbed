"""Model-free admission checks for the fixed source-fed operator experiment."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gdn_worker", ROOT / "tools/kaggle/lighton4b_gdn_source_primitives.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
class ProtocolControls(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).parent / "protocol-v1.json").read_text())
    def test_exact_protocol(self):
        worker.admit(self.cfg)
    def test_mutations_rejected(self):
        changes = [("tokens", [4]), ("eps", 1e-5), ("threads", 8),
                   ("fused_gdn_ch", False), ("auto_fgdn", True),
                   ("total_timeout_seconds", 0), ("model_forward_allowed", True),
                   ("upstream_commit", "0" * 40), ("minimum_available_memory_bytes", 1),
                   ("physical_result_sha256", "0" * 64)]
        for key, value in changes:
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg); cfg[key] = value
                with self.assertRaises(RuntimeError): worker.admit(cfg)
        cfg = copy.deepcopy(self.cfg); cfg["tensors"][0]["source_offset"] += 1
        with self.assertRaises(RuntimeError): worker.admit(cfg)
if __name__ == "__main__":
    unittest.main()
