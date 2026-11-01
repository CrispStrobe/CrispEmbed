#!/usr/bin/env python3
"""Model-free output/checkpoint admission controls for the native benchmark."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


BINARY = None


class BenchmarkOutputControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if BINARY is None:
            raise unittest.SkipTest("run this control with --binary pointing to test-inpaint-benchmark")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="inpaint-benchmark-control-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "input.rgb8").write_bytes(bytes(12))
        (self.root / "mask.u8").write_bytes(bytes((255, 0, 0, 0)))
        self.output = self.root / "final.rgb8"
        self.checkpoint = self.root / "cold.rgb8"

    def invoke(self, output=None, checkpoint=None):
        command = [str(BINARY), "--engine", "lama", "--model", "unused.gguf",
                   "--model-sha256", "0" * 64, "--rgb", "input.rgb8", "--mask", "mask.u8",
                   "--width", "2", "--height", "2", "--output", str(output or self.output),
                   "--checkpoint-output", str(checkpoint or self.checkpoint), "--runs", "1",
                   "--threads", "1", "--backend", "cpu"]
        env = {name: value for name, value in os.environ.items()
               if not name.startswith(("CRISPEMBED_INPAINT_", "INPAINT_TEST_"))}
        env["CRISPEMBED_LAMA"] = "0"
        result = subprocess.run(command, cwd=self.root, env=env, capture_output=True,
                                text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("INPAINT_RUN ", result.stdout)
        return result.stderr

    def test_distinct_new_paths_reach_closed_model_gate(self):
        error = self.invoke()
        self.assertIn("requires CRISPEMBED_LAMA=1", error)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.checkpoint.exists())

    def test_existing_final_output_is_preserved(self):
        self.output.write_bytes(b"existing final image")
        self.assertIn("output must be a new file", self.invoke())
        self.assertEqual(self.output.read_bytes(), b"existing final image")
        self.assertFalse(self.checkpoint.exists())

    def test_existing_checkpoint_is_preserved(self):
        self.checkpoint.write_bytes(b"existing cold image")
        self.assertIn("checkpoint output must be a new file", self.invoke())
        self.assertEqual(self.checkpoint.read_bytes(), b"existing cold image")
        self.assertFalse(self.output.exists())

    def test_checkpoint_dangling_symlink_is_rejected(self):
        target = self.root / "must-not-be-created.rgb8"
        try:
            self.checkpoint.symlink_to(target)
        except (OSError, NotImplementedError) as error:
            self.skipTest("symbolic links unavailable: " + str(error))
        self.assertIn("checkpoint output must be a new file", self.invoke())
        self.assertTrue(self.checkpoint.is_symlink())
        self.assertFalse(target.exists())

    def test_same_checkpoint_and_final_path_is_rejected(self):
        self.assertIn("must be different paths", self.invoke(checkpoint=self.output))
        self.assertFalse(self.output.exists())

    def test_dotdot_alias_of_final_path_is_rejected(self):
        (self.root / "subdirectory").mkdir()
        alias = self.root / "subdirectory" / ".." / self.output.name
        self.assertIn("must be different paths", self.invoke(checkpoint=alias))
        self.assertFalse(self.output.exists())

    def test_parent_symlink_alias_of_final_path_is_rejected(self):
        directory = self.root / "real"
        directory.mkdir()
        alias = self.root / "alias"
        try:
            alias.symlink_to(directory, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest("symbolic links unavailable: " + str(error))
        self.assertIn("must be different paths", self.invoke(
            output=directory / "same.rgb8", checkpoint=alias / "same.rgb8"))
        self.assertFalse((directory / "same.rgb8").exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    BINARY = args.binary.resolve(strict=True)
    unittest.main(argv=[__file__, *remainder])
