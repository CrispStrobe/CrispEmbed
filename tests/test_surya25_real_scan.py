"""Weight-free real-scan staging, frozen crops and comparison checks."""
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

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / 'tools/kaggle/surya25-real-scan'


def runner():
    spec = importlib.util.spec_from_file_location('real_scan', LANE / 'run.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot import real-scan runner')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class RealScan(unittest.TestCase):
    def test_source_closed_private_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / 'stage'
            command = [sys.executable, str(LANE / 'stage.py'), str(stage), '--commit', 'a' * 40]
            manifest = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
            tree = ast.parse((stage / 'kernel.py').read_text())
            values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                      if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                      and n.targets[0].id in ('SOURCES_B64', 'SOURCES_SHA256')}
            raw = base64.b64decode(values['SOURCES_B64'], validate=True)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), values['SOURCES_SHA256'])
            self.assertEqual(manifest['bundle_sha256'], values['SOURCES_SHA256'])
            for path, entry in json.loads(raw).items():
                data = base64.b64decode(entry['base64'], validate=True)
                self.assertEqual(data, (ROOT / path).read_bytes())
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'])
            metadata = json.loads((stage / 'kernel-metadata.json').read_text())
            self.assertEqual(metadata['is_private'], 'true')
            self.assertEqual(metadata['enable_gpu'], 'false')
            self.assertEqual(metadata['dataset_sources'], [])
            self.assertEqual(metadata['model_sources'], [])
            source = (stage / 'kernel.py').read_text()
            self.assertIn("SURYA25_NATIVE=BUILD + '/test-surya25-run'", source)

    def test_fixture_crop_and_lossless_roundtrip(self):
        module = runner()
        with tempfile.TemporaryDirectory() as tmp:
            for case in module.CASES.values():
                source = ROOT / case['path']
                self.assertEqual(module.digest(source), case['source_sha256'])
                with Image.open(source) as image:
                    crop = image.convert('RGB').crop(tuple(case['box']))
                self.assertEqual(hashlib.sha256(crop.tobytes()).hexdigest(), case['rgb_sha256'])
                png = module.canonical_png(crop)
                self.assertEqual(hashlib.sha256(png).hexdigest(), case['png_sha256'])
                path = Path(tmp) / 'crop.png'
                path.write_bytes(png)
                with Image.open(path) as reopened:
                    self.assertEqual(reopened.convert('RGB').tobytes(), crop.tobytes())

    def test_comparison_reports_divergence(self):
        module = runner()
        ref = {'token_ids': [3, 2], 'text': 'hello', 'stop': 'EOS', 'confidences': [.8, .9]}
        same = module.compare(ref, ref)
        self.assertTrue(same['ids_exact'] and same['text_exact'] and same['stop_exact'] and same['both_reached_eos'])
        alt = {**ref, 'token_ids': [4, 2], 'text': 'bye'}
        comp = module.compare(ref, alt)
        self.assertFalse(comp['ids_exact'])
        self.assertEqual(comp['first_divergent_step'], 0)
        self.assertEqual((comp['reference_token'], comp['arm_token']), (3, 4))


if __name__ == '__main__':
    unittest.main()
