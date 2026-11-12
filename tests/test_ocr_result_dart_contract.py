"""Declaration-level C/Dart ABI guard; native-memory Dart execution is separate."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DartResultContract(unittest.TestCase):
    def test_struct_mirrors_complete_header_record(self):
        header = (ROOT / 'src/crispembed.h').read_text()
        body = re.search(r'typedef struct crispembed_ocr_result\s*\{(.*?)\}\s*crispembed_ocr_result;', header, re.S)[1]
        body = re.sub(r'//[^\n]*', '', body)
        expected = []
        for raw in body.split(';'):
            if not raw.strip():
                continue
            match = re.fullmatch(r'\s*(const\s+char\s*\*|float|int)\s*(\w+(?:\s*,\s*\w+)*)\s*', raw)
            self.assertIsNotNone(match)
            kind = 'pointer' if '*' in match[1] else match[1]
            expected.extend((name.strip(), kind) for name in match[2].split(','))
        source = (ROOT / 'flutter/crispembed/lib/src/crispembed.dart').read_text()
        record = re.search(r'final class _NativeOcrResult extends Struct\s*\{(.*?)\n\}', source, re.S)[1]
        actual = []
        for annotation, kind, name in re.findall(r'(?:(@(?:Float|Int32)\(\))\s*)?external\s+(double|int|Pointer<Utf8>)\s+(\w+)\s*;', record):
            cname = re.sub(r'([A-Z])', lambda m: '_' + m[1].lower(), name)
            ckind = {'@Float()': 'float', '@Int32()': 'int', '': 'pointer'}[annotation]
            self.assertEqual(kind, {'float': 'double', 'int': 'int', 'pointer': 'Pointer<Utf8>'}[ckind])
            actual.append((cname, ckind))
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
