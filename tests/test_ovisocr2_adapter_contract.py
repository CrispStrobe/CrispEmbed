"""Dependency-free gate and ABI checks for the explicit Ovis format2 selector."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
GATES = ("CRISPEMBED_QWEN35_OCR", "CRISPEMBED_OVISOCR2_FRONTEND",
         "CRISPEMBED_OVISOCR2_INFERENCE")


class OvisAdapterContract(unittest.TestCase):
    def test_engine_is_append_only_and_public_int_is_21(self):
        enum = (ROOT / "src/ocr_orchestrator.h").read_text().split("enum class engine {")[1].split("};")[0]
        self.assertLess(enum.index("qwen35_ocr,"), enum.index("ovisocr2,"))
        cxx = (ROOT / "src/crispembed.cpp").read_text()
        self.assertRegex(cxx, r"case 21:\s*return E::ovisocr2;")
        self.assertIn("21=ovisocr2", (ROOT / "src/crispembed.h").read_text())
        self.assertIn("21=ovisocr2", (ROOT / "crispembed-sys/src/lib.rs").read_text())
        self.assertIn("21=ovisocr2", (ROOT / "crispembed/src/lib.rs").read_text())

    def test_pipeline_selector_rejects_other_profiles_and_missing_gates(self):
        source = (ROOT / "src/ocr_orchestrator.cpp").read_text()
        case = source.split("case engine::ovisocr2: {", 1)[1].split("case engine::lightonocr:", 1)[0]
        for gate in GATES:
            self.assertIn(f'core_env::on("{gate}")', case)
        self.assertIn('kv_u32(meta, "qwen35_ocr.format_version", 0) == 2', case)
        self.assertIn('kv_str(meta, "qwen35_ocr.profile", "") == "ovisocr2"', case)
        self.assertIn("!st.params.vlm_prompt.empty()", case)
        self.assertIn("ctx->ovis", case)
        self.assertIn("ctx->qwen35", case)

    def test_cli_and_server_require_explicit_model_and_no_default_alias(self):
        cli = (ROOT / "examples/cli/main.cpp").read_text()
        server = (ROOT / "examples/server/server.cpp").read_text()
        for source in (cli, server):
            self.assertRegex(source, r'if \(n == "ovisocr2"\) return 21;')
            for gate in GATES:
                self.assertIn(f'core_env::on("{gate}")', source)
            self.assertIn("eid == 21", source)
        self.assertIn("ocr_rec_path.empty()", cli)
        self.assertIn("ocr_rec_model_path.empty()", server)
        registry = (ROOT / "examples/cli/model_mgr.cpp").read_text()
        self.assertNotRegex(registry, r'\{\s*"ovisocr2"\s*,')


if __name__ == "__main__":
    unittest.main()
