"""Weight-free tests for the hosted composed-vision stage closure and gates."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tools/kaggle/xiaomi-ocr-0-vision-prefill"
RUNNER, STAGE = DIR / "runner.py", DIR / "stage.py"
CPP = ROOT / "tests/test_xiaomi_vision_prefill.cpp"
EXPECTED = ["prefill.vision." + name for name in
            ("patch_embed", "position", "embedded", "rotary_cos", "rotary_sin")]
EXPECTED += [f"prefill.vision.layer.{i}" for i in range(12)]
EXPECTED += ["prefill.vision.merger.norm", "prefill.vision.merger.up",
             "prefill.vision.merger.activation", "prefill.vision.merger"]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle(runner, files, base=None, ggml=None):
    package = {"base_commit": base or runner.BASE, "ggml_commit": ggml or runner.GGML,
               "runner_template_sha256": "0" * 64, "files": files}
    return base64.b64encode(zlib.compress(json.dumps(package).encode())).decode()


def stage_line(name, max_abs, cos_min, passed):
    return "STAGE " + json.dumps({"name": name, "n": 100, "pass": passed, "cos_min": cos_min,
                                  "cos_mean": 1.0, "cos_global": 1.0, "max_abs": max_abs,
                                  "mean_abs": 0.0, "rms": 0.0, "native_norm": 1.0,
                                  "reference_norm": 1.0})


class VisionPrefillTest(unittest.TestCase):
    def test_cpu_audit_has_its_required_sources_and_kernel_inputs(self):
        metadata = json.loads((DIR / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["enable_gpu"], "false")
        self.assertEqual(len(metadata["kernel_sources"]), 2)
        self.assertTrue(any("export-current-v2" in name for name in metadata["kernel_sources"]))
        cmake = (DIR / "CMakeLists.txt").read_text()
        self.assertNotIn("src/qwen35_weight_cache.cpp", cmake)
        source = RUNNER.read_text()
        self.assertIn('("src", "tests", "ggml")', source)
        self.assertNotIn("torch.cuda.is_available()", source)

    def test_unstaged_template_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, \
                self.assertRaisesRegex(ValueError, "not staged"):
            load(RUNNER, "vision_runner").unpack(Path(directory))

    def test_closure_requires_exact_names_digests_and_revisions(self):
        runner = load(RUNNER, "vision_runner")
        content = b"source bytes"
        files = {"src/test.cpp": {"content": base64.b64encode(content).decode(),
                                  "sha256": hashlib.sha256(content).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "ALLOWED", set(files)):
            root = Path(directory)
            with patch.object(runner, "BUNDLE", bundle(runner, files)):
                closure = runner.unpack(root)
            self.assertEqual((root / "src/test.cpp").read_bytes(), content)
            self.assertEqual(closure["files_sha256"]["src/test.cpp"], files["src/test.cpp"]["sha256"])
            files["src/test.cpp"]["content"] = base64.b64encode(b"tampered").decode()
            with patch.object(runner, "BUNDLE", bundle(runner, files)), \
                    self.assertRaisesRegex(ValueError, "digest differs"):
                runner.unpack(root)
            files["src/test.cpp"]["content"] = base64.b64encode(content).decode()
            files["extra"] = files["src/test.cpp"]
            with patch.object(runner, "BUNDLE", bundle(runner, files)), \
                    self.assertRaisesRegex(ValueError, "closure differs"):
                runner.unpack(root)
            del files["extra"]
            with patch.object(runner, "BUNDLE", bundle(runner, files, ggml="0" * 40)), \
                    self.assertRaisesRegex(ValueError, "closure differs"):
                runner.unpack(root)
            with patch.object(runner, "BUNDLE", bundle(runner, files, base="1" * 40)), \
                    self.assertRaisesRegex(ValueError, "closure differs"):
                runner.unpack(root)

    def test_thresholds_are_literal_and_shared_with_the_stage(self):
        runner = load(RUNNER, "vision_runner")
        stage = load(STAGE, "vision_stage")
        self.assertEqual((runner.MAX_ABS, runner.COS_MIN), (0.002, 0.99999))
        self.assertEqual(runner.REPORT["thresholds"],
                         {"max_abs": 0.002, "cos_min": 0.99999, "relaxed_for_dtype": False})
        self.assertEqual(set(stage.SOURCES), runner.ALLOWED)
        self.assertEqual(runner.EXPECTED, [name.removeprefix("prefill.vision.") for name in EXPECTED])

    def test_stage_script_pins_base_gitlink_and_executed_sources(self):
        stage = load(STAGE, "vision_stage")
        self.assertEqual((stage.BASE, stage.GGML),
                         ("15adb672f71f8470a23db1284dafa102515adae9",
                          "a06fc0bb46b73cc10011fc90ae4054aee8438aeb"))
        self.assertIn("tests/test_xiaomi_vision_prefill.cpp", stage.SOURCES)
        self.assertIn("audit/CMakeLists.txt", stage.SOURCES)
        # Only the test-only audit CMakeLists ships; the repository build file does not.
        self.assertNotIn("CMakeLists.txt", stage.SOURCES)
        for destination, source in stage.SOURCES.items():
            self.assertTrue(Path(source).is_file(), destination)
        self.assertIn("add_subdirectory(ggml EXCLUDE_FROM_ALL)",
                      (DIR / "CMakeLists.txt").read_text())

    def test_runner_threshold_checker_rejects_a_relaxed_verdict(self):
        runner = load(RUNNER, "vision_runner")
        rows = runner.parse_stages("\n".join(
            [stage_line(name, 0.001, 1.0, True) for name in EXPECTED[:-1]] +
            [stage_line(EXPECTED[-1], 0.0031, 0.999999, True)]))
        self.assertEqual(len(rows), 21)
        last = rows[-1]
        self.assertGreater(last["max_abs"], runner.MAX_ABS)
        # A producer that calls a relaxed verdict true must not survive the audit.
        strict = last["max_abs"] <= runner.MAX_ABS and last["cos_min"] >= runner.COS_MIN
        self.assertNotEqual(strict, last["pass"])
        rows = runner.parse_stages(stage_line("prefill.vision.merger", 0.0, 0.9999, True))
        self.assertFalse(rows[0]["cos_min"] >= runner.COS_MIN)

    def test_cpp_stage_covers_every_named_stage_and_never_uses_env_threshold(self):
        source = CPP.read_text()
        for name in ("patch_embed", "position", "embedded", "rotary_cos", "rotary_sin",
                     "merger.norm", "merger.up", "merger.activation"):
            self.assertIn(f'"{name}"', source)
        self.assertIn("static constexpr float kMaxAbs = 0.002f, kCosMin = 0.99999f;", source)
        # The verdict must be the literal comparison against those constants.
        self.assertIn("report.max_abs <= kMaxAbs && report.cos_min >= kCosMin", source)
        # Strip comments: the header explains the hazard by naming it.
        code = "\n".join(line.split("//", 1)[0] for line in source.splitlines())
        # No call may reach Report::is_pass(), which honours CRISPEMBED_DIFF_COS_THRESHOLD.
        self.assertNotIn(".is_pass(", code)
        self.assertNotIn("is_pass_global(", code)
        self.assertNotIn("CRISPEMBED_DIFF_COS_THRESHOLD", code)
        self.assertIn("kXiaomiPatchLimit", source)
        self.assertIn("hidden.size() != 12", source)
        self.assertIn("NEGATIVE_CONTROL rejected=1", source)


if __name__ == "__main__":
    unittest.main()
