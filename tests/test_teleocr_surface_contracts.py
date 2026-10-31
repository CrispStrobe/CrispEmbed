#!/usr/bin/env python3
"""Weight-free checks that the TeleOCR selector reaches every public surface.

This complements test_teleocr_public.py, which executes the actual model. It
guards ABI numbering and the CLI/server routing lists that have drifted in past
integrations; it does not assert model correctness from source spelling.
"""

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


class TeleocrSurfaceContracts(unittest.TestCase):
    def test_append_only_abi(self):
        src = read("src/crispembed.cpp")
        body = re.search(r"static ocr_orchestrator::engine map_engine\(int e\).*?\n}\n", src, re.S)[0]
        pairs = {int(i): name for i, name in re.findall(r"case (\d+):\s*return E::(\w+);", body)}
        self.assertEqual(pairs[23], "minicpm_ocr")
        self.assertEqual(pairs[24], "teleocr")
        self.assertEqual(pairs[25], "mineru25_pro")
        self.assertEqual(len(pairs), 26)
        self.assertRegex(read("src/crispembed.h"), r"#define CRISPEMBED_OCR_ENGINE_TELEOCR 24\b")
        self.assertRegex(src, r'arch == "teleocr"\) return OCR_MODEL_QWEN2VL;')

    def test_cli_and_server_route_as_vlm(self):
        for path in ("examples/cli/main.cpp", "examples/server/server.cpp"):
            with self.subTest(path=path):
                src = read(path)
                self.assertRegex(src, r'if \(n == "teleocr"\) return 24;')
                expr = re.search(r"const bool is_vlm =\s*(.*?);", src, re.S)[1]
                self.assertIn("eid == 24", expr)
                self.assertIn('core_env::on("CRISPEMBED_TELEOCR")', src)
                self.assertIn('"teleocr.gguf_version", 0) == 1', src)
        # This startup cap is also the state restored after a per-request cap.
        server = read("examples/server/server.cpp")
        self.assertIn("if (ocr_model_ctx && ocr_max_tokens > 0)", server)

    def test_model_identity_and_request_reset(self):
        src = read("src/ocr_orchestrator.cpp")
        block = src.split("case engine::teleocr: {", 1)[1].split("case engine::qwen3vl:", 1)[0]
        self.assertIn("st.model_a != ctx->teleocr_model_path", block)
        self.assertIn("qwen2vl_ocr_free(ctx->teleocr)", block)
        self.assertIn("ctx->teleocr_default_prompt.c_str()", block)
        self.assertIn("ctx->teleocr_default_max_tokens", block)
        self.assertIn("st.params.vlm_max_tokens", block)
        self.assertIn("st.params.vlm_prompt", block)

    def test_gpu_requires_separate_opt_in(self):
        src = read("src/qwen2vl_ocr.cpp")
        self.assertTrue('if (!core_env::on("CRISPEMBED_TELEOCR_GPU")) force_cpu = true;' in src,
                        "TeleOCR must require a separate GPU opt-in")
        self.assertTrue('bool force_cpu = (getenv("QWEN2VL_OCR_FORCE_CPU")' in src,
                        "preserve the existing force-CPU override before applying TeleOCR policy")

    def test_binding_constants(self):
        module = ast.parse(read("python/crispembed/_binding.py"))
        values = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in module.body
                  if isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id == "OCR_ENGINE_TELEOCR"}
        self.assertEqual(values["OCR_ENGINE_TELEOCR"], 24)
        self.assertIn('"OCR_ENGINE_TELEOCR"', read("python/crispembed/__init__.py"))
        self.assertIn("CRISPEMBED_OCR_ENGINE_TELEOCR: c_int = 24", read("crispembed-sys/src/lib.rs"))
        self.assertIn("OCR_ENGINE_TELEOCR: i32 = crispembed_sys::CRISPEMBED_OCR_ENGINE_TELEOCR",
                      read("crispembed/src/lib.rs"))
        self.assertIn("const int crispOcrEngineTeleocr = 24;",
                      read("flutter/crispembed/lib/src/crispembed.dart"))


if __name__ == "__main__":
    unittest.main()
