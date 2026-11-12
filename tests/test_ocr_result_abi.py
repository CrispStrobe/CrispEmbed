"""Stdlib-only production-mirror/array controls against the authoritative C header.

The separate hosted native-memory fixture also exercises the full production
module, signature setup and constructor without any model download/inference.
"""
import ast
import ctypes as C
from pathlib import Path
import re
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / 'python/crispembed/_binding.py'
module = ast.parse(BINDING.read_text())
node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == '_CrispOcrResult')
namespace = {'ctypes': C}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(BINDING), 'exec'), namespace)
Result = namespace['_CrispOcrResult']
body = re.search(r'typedef struct crispembed_ocr_result\s*\{(.*?)\}\s*crispembed_ocr_result;',
                 (ROOT / 'src/crispembed.h').read_text(), re.S).group(1)
body = re.sub(r'//[^\n]*', '', body)
ctype = {'float': C.c_float, 'int': C.c_int, 'const char *': C.c_char_p}
fields = []
for raw in body.split(';'):
    if not raw.strip():
        continue
    match = re.fullmatch(r'\s*(const\s+char\s*\*|float|int)\s*(\w+(?:\s*,\s*\w+)*)\s*', raw)
    if not match:
        raise AssertionError('C result declaration changed: update the ABI test parser')
    kind = 'const char *' if '*' in match[1] else match[1]
    fields.extend((name.strip(), ctype[kind]) for name in match[2].split(','))
NativeResult = type('HeaderResult', (C.Structure,), {'_fields_': fields})
owner = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'CrispOcrOrchestrator')
run_node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
exec(compile(ast.Module(body=[run_node], type_ignores=[]), str(BINDING), 'exec'), namespace)
run = namespace['run']


class OcrResultAbiControls(unittest.TestCase):
    def test_fields_match_authoritative_header(self):
        self.assertEqual(Result._fields_, NativeResult._fields_)

    def test_size_and_offsets_match_header_contract(self):
        self.assertEqual(C.sizeof(Result), C.sizeof(NativeResult))
        for name, _ in fields:
            self.assertEqual(getattr(Result, name).offset, getattr(NativeResult, name).offset)
        if C.sizeof(C.c_void_p) == 8:
            # Independently measured by the exact-header native C ABI control.
            self.assertEqual(C.sizeof(Result), 48)
            self.assertEqual(Result.text.offset, 24)
            self.assertEqual(Result.orientation_angle.offset, 40)
            self.assertEqual(Result.orientation_confidence.offset, 44)

    def test_real_wrapper_reads_three_native_layout_records(self):
        self.assertEqual(C.sizeof(Result), C.sizeof(NativeResult), 'stop before unsafe pointer indexing')
        native = (NativeResult * 3)()
        buffers = [C.create_string_buffer(text) for text in [b'first line', b'second line', b'third line']]
        for i, row in enumerate(native):
            row.text = C.cast(buffers[i], C.c_char_p)
            row.text_len = len(buffers[i].value)
            row.x, row.y, row.w, row.h = i + 1, 10 + i, 100, 20
            row.confidence = .75
            row.orientation_corrected = int(i > 0)
            row.orientation_angle = i * 90
            row.orientation_confidence = .25 * (i + 1)
        pointer = C.cast(native, C.POINTER(Result))
        def call(ctx, image, count, text, confidence):
            C.cast(count, C.POINTER(C.c_int))[0] = 3
            C.cast(text, C.POINTER(C.c_char_p))[0] = b'first line\nsecond line\nthird line'
            C.cast(confidence, C.POINTER(C.c_float))[0] = .75
            return pointer
        instance = types.SimpleNamespace(_ctx=19, _lib=types.SimpleNamespace(crispembed_ocr_pipeline_run=call))
        result = run(instance, 'owned-fixture.png')
        self.assertEqual([row['text'] for row in result['regions']], ['first line', 'second line', 'third line'])
        self.assertEqual([row['x'] for row in result['regions']], [1, 2, 3])
        self.assertEqual([pointer[i].orientation_angle for i in range(3)], [0, 90, 180])
        for i in range(3):
            self.assertAlmostEqual(pointer[i].orientation_confidence, .25 * (i + 1))
        buffers[1].value = b'changed'
        self.assertEqual(result['regions'][1]['text'], 'second line')

    def test_old_mirror_is_rejected_before_array_dereference(self):
        OldResult = type('OldResult', (C.Structure,), {'_fields_': Result._fields_[:-2]})
        self.assertNotEqual(C.sizeof(OldResult), C.sizeof(NativeResult))
        if C.sizeof(C.c_void_p) == 8:
            self.assertEqual(C.sizeof(OldResult), 40)


if __name__ == '__main__':
    unittest.main()
