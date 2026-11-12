"""Stdlib-only execution of the real wrapper method against controlled C outputs.

Native lazy-load status and decoded/model behavior have separate hosted gates.
"""
import ast
import ctypes
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT / 'python/crispembed/_binding.py').read_text())
cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'CrispOcrOrchestrator')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
namespace = {'ctypes': ctypes}
exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual orchestrator run>', 'exec'), namespace)
run = namespace['run']


def native(answer, diagnostic=b'image exceeds native patch bound', has_diagnostic=True):
    def call(ctx, path, count, text, confidence):
        ctypes.cast(count, ctypes.POINTER(ctypes.c_int))[0] = 0
        ctypes.cast(text, ctypes.POINTER(ctypes.c_char_p))[0] = answer
        ctypes.cast(confidence, ctypes.POINTER(ctypes.c_float))[0] = 0.0
        return None
    lib = types.SimpleNamespace(crispembed_ocr_pipeline_run=call)
    if has_diagnostic:
        lib.crispembed_ocr_pipeline_last_error = lambda ctx: diagnostic
    return types.SimpleNamespace(_ctx=19, _lib=lib)


class PipelineFailureControls(unittest.TestCase):
    def test_native_failure_exposes_reason(self):
        with self.assertRaisesRegex(RuntimeError, 'image exceeds native patch bound'):
            run(native(None), 'page.png')

    def test_valid_blank_is_success(self):
        result = run(native(b''), 'blank.png')
        self.assertEqual(result['text'], '')
        self.assertEqual(result['n_regions'], 0)

    def test_older_runtime_without_accessor(self):
        with self.assertRaisesRegex(RuntimeError, 'OCR pipeline recognition failed'):
            run(native(None, has_diagnostic=False), 'page.png')
        self.assertEqual(run(native(b'', has_diagnostic=False), 'blank.png')['text'], '')

    def test_next_success_does_not_reuse_old_error(self):
        owner = native(None)
        with self.assertRaises(RuntimeError):
            run(owner, 'large.png')
        owner._lib = native(b'Hello OCR\n\n12345')._lib
        self.assertEqual(run(owner, 'small.png')['text'], 'Hello OCR\n\n12345')


if __name__ == '__main__':
    unittest.main()
