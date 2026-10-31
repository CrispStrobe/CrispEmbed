"""Weight-free Ovis direct-F16 regression stage and frozen gates."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / 'tools/kaggle/ovisocr2-direct-f16-regression'


def runner():
    spec = importlib.util.spec_from_file_location('ovis_direct', LANE / 'run.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot import runner')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class OvisDirect(unittest.TestCase):
    def test_private_stage_carries_exact_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'stage'
            command = [sys.executable, str(LANE / 'stage.py'), str(out), '--commit', 'a' * 40]
            manifest = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
            tree = ast.parse((out / 'kernel.py').read_text())
            values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                      if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                      and n.targets[0].id in ('SOURCES_B64', 'SOURCES_SHA256')}
            raw = base64.b64decode(values['SOURCES_B64'], validate=True)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), values['SOURCES_SHA256'])
            self.assertEqual(manifest['bundle_sha256'], values['SOURCES_SHA256'])
            for name, entry in json.loads(raw).items():
                data = base64.b64decode(entry['base64'], validate=True)
                self.assertEqual(data, (ROOT / name).read_bytes())
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'])
            metadata = json.loads((out / 'kernel-metadata.json').read_text())
            self.assertEqual(metadata['is_private'], 'true')
            self.assertEqual(metadata['dataset_sources'], [])
            self.assertIn('OVIS_BUILD=BUILD', (out / 'kernel.py').read_text())

    def test_receipt_and_expected_files_pinned(self):
        r = runner()
        for name, pin in r.SOURCE_PINS.items():
            self.assertEqual(r.sha(ROOT / name), pin)
        receipt = json.loads((ROOT / r.RECEIPT).read_text())
        self.assertEqual(r.canonical(receipt['producer']), r.PRODUCER_SHA)
        self.assertEqual(receipt['independent_review']['producer_canonical_sha256'], r.PRODUCER_SHA)
        self.assertTrue(all(case in receipt['producer']['native_cases']['f16'] for case in r.CASES))

    def test_exact_comparison_rejects_each_changed_field(self):
        r = runner()
        original = {'token_ids': [3, 248046], 'raw_text': 'text', 'stop_id': 248046}
        self.assertTrue(all(r.exact(original, original).values()))
        for field, bad in (('token_ids', [4, 248046]), ('raw_text', 'wrong'), ('stop_id', 248044)):
            self.assertFalse(all(r.exact({**original, field: bad}, original).values()))
        with self.assertRaises(ValueError):
            r.exact({}, original)


if __name__ == '__main__':
    unittest.main()
