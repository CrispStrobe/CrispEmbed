#!/usr/bin/env python3
"""Small stdlib controls for hosted artifact routing and immutable fixtures."""
import importlib.util
from pathlib import Path
import re
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from lama_reference_contract import CASES, identity

spec = importlib.util.spec_from_file_location("lama_hosted", ROOT / "tools/ci-heavy/lama_inpaint.py")
hosted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hosted)


class HostedContracts(unittest.TestCase):
    def test_cli_target_routes_to_real_output_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            actual = root / "bin/crispembed"
            actual.write_bytes(b"fixture")
            self.assertEqual(hosted.binary_path(root, "crispembed-cli"), actual)
            (root / "other").mkdir()
            (root / "other/crispembed").write_bytes(b"duplicate")
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                hosted.binary_path(root, "crispembed-cli")

    def test_missing_executable_is_not_reported_as_built(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                hosted.binary_path(directory, "test-lama-diff")

    def test_each_production_observer_leak_fails(self):
        tokens = ("lama::run_reference(", "lama::set_observer(")
        hosted.assert_observer_symbols("unrelated symbols", False)
        for text in (*tokens, "\n".join(tokens)):
            with self.assertRaises(AssertionError):
                hosted.assert_observer_symbols(text, False)
        for text in ("", *tokens):
            with self.assertRaises(AssertionError):
                hosted.assert_observer_symbols(text, True)
        hosted.assert_observer_symbols("\n".join(tokens), True)

    @unittest.skipUnless(hosted.os.name == "posix", "hosted process groups use POSIX")
    def test_timeout_kills_descendants_after_parent_already_exited(self):
        process = Mock(pid=12345)
        process.wait.return_value = -signal.SIGTERM
        with patch.object(hosted.os, "killpg") as kill:
            hosted.terminate_tree(process)
        self.assertEqual(kill.call_args_list[0].args, (12345, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (12345, signal.SIGKILL))

    def test_native_fixture_hashes_match_frozen_owned_inputs(self):
        source = (ROOT / "tests/test_lama_diff.cpp").read_text(encoding="utf-8")
        records = re.findall(r'\{\s*"([a-z_]+)"\s*,\s*\{\s*(\d+)\s*,\s*(\d+)\s*,\s*'
                             r'"([0-9a-f]{64})"\s*,\s*"([0-9a-f]{64})"\s*\}\s*\}', source)
        self.assertEqual(len(records), len(CASES))
        observed = {name: (int(width), int(height), rgb, mask) for name, width, height, rgb, mask in records}
        expected = {}
        for name in CASES:
            row = identity(name)
            expected[name] = (row["width"], row["height"], row["rgb_sha256"], row["mask_sha256"])
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
