"""Pure source and phase gates; no model or renderer imports."""

import importlib.util
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'ovis_quality_run', ROOT / 'tools/ci-heavy/ovisocr2_quality_run.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class QualityDriverControls(unittest.TestCase):
    def test_original_readme_ast_prompt_filter_and_cleanup(self):
        source = '''```python
class OvisOCR2Parser:
    def __init__(self):
        prompt = 'Original ' + 'prompt'
    def _clean_truncated_repeats(self, text: str) -> str:
        return text.removesuffix('!')
    def parse(self, filter_imgtags=True):
        text = 'placeholder'
        if filter_imgtags:
            text = "\\n\\n".join(block for block in text.split("\\n\\n")
                               if not block.strip().startswith('<img src="images/bbox_'))
```'''
        prompt, cleaner, filter_fn = driver.original_readme_functions(source)
        self.assertEqual(prompt, 'Original prompt')
        self.assertEqual(driver.process_readme_output(' hello! ', cleaner, filter_fn), 'hello')
        self.assertEqual(driver.process_readme_output('A\n\n<img src="images/bbox_1" />\n\nB',
                                                       cleaner, filter_fn), 'A\n\nB')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            driver.original_readme_functions('```python\nprompt = input()\n```')

    def test_missing_baseline_receipt_blocks_all_phases(self):
        missing = ROOT / 'tests/results/qwen35-ocr/ovisocr2-model-audit.json'
        if missing.exists():
            self.skipTest('A real receipt now exists; its content is reviewed by the hosted gate')
        for phase in driver.PHASES:
            with self.assertRaisesRegex(ValueError, 'baseline receipt absent'):
                driver.require_baseline(phase)
        with self.assertRaises(ValueError):
            driver.require_baseline('unsupported')

    def test_no_framework_import(self):
        for name in ('torch', 'transformers', 'gguf', 'PIL', 'vllm'):
            self.assertNotIn(name, sys.modules)


if __name__ == '__main__':
    unittest.main()
