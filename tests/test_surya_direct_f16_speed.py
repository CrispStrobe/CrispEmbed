"""Hermetic speed-lane staging and direct/off dispatch controls."""
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
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / 'tools/kaggle/surya-direct-f16-speed'
QUANT = ROOT / 'tools/kaggle/surya-ocr-25-quant/run.py'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class SpeedLane(unittest.TestCase):
    def test_stage_embeds_all_exact_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'stage'
            run = subprocess.run([sys.executable, str(LANE / 'stage.py'), str(dest), '--commit',
                                  '9498e67e3558a9a0b0ec20f93c40698b2434204e'],
                                 capture_output=True, text=True, check=True)
            manifest = json.loads(run.stdout)
            tree = ast.parse((dest / 'kernel.py').read_text())
            values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign) and len(node.targets) == 1
                      and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id in ('COMMIT', 'SOURCES_B64', 'SOURCES_SHA256')}
            bundle_bytes = base64.b64decode(values['SOURCES_B64'], validate=True)
            self.assertEqual(hashlib.sha256(bundle_bytes).hexdigest(), values['SOURCES_SHA256'])
            self.assertEqual(manifest['bundle_sha256'], values['SOURCES_SHA256'])
            for path, entry in json.loads(bundle_bytes).items():
                data = base64.b64decode(entry['base64'], validate=True)
                self.assertEqual(data, (ROOT / path).read_bytes())
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'])
            metadata = json.loads((dest / 'kernel-metadata.json').read_text())
            self.assertEqual(metadata['enable_gpu'], 'false')
            self.assertEqual(metadata['dataset_sources'], [])
            self.assertEqual(metadata['model_sources'], [])

    def test_stage_rejects_wrong_base_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'stage'
            run = subprocess.run([sys.executable, str(LANE / 'stage.py'), str(dest), '--commit', 'a' * 40],
                                 capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertFalse(dest.exists())

    def test_native_profile_gate_and_results(self):
        runner = module('speed_lane', LANE / 'run.py')
        quant = module('quant_helpers', QUANT)
        output = {'raw_text_utf8_hex': '6869', 'token_ids': [3, 2], 'confidences': [.5, .99],
                  'mean_confidence': .745, 'stop_kind': 1, 'init_seconds': .3, 'wall_seconds': 2}
        profile = 'PROFILE_MATRIX phase=decode tensor=test calls=5 source_bytes=4 f32_bytes=8 materialize_ns=1\n'
        profile += 'PROFILE_COMPLETE active=1 dropped=0 events=2 matrices=1\n'
        completed = subprocess.CompletedProcess([], 0, profile + 'SURYA25_RESULT ' + json.dumps(output), '')
        with mock.patch.object(runner.subprocess, 'run', return_value=completed) as call:
            result = runner.native(quant, Path('/a'), Path('/b'), 'direct', profile=True)
        self.assertEqual(call.call_args.kwargs['env']['CRISPEMBED_QWEN35_DIRECT_F16'], '1')
        self.assertEqual(call.call_args.kwargs['env']['CRISPEMBED_QWEN35_PROFILE'], '1')
        self.assertEqual(result['profile']['matrix_calls'], 5)
        self.assertTrue(result['profile']['complete'])
        self.assertEqual(result['text'], 'hi')
        with mock.patch.object(runner.subprocess, 'run', return_value=completed) as call:
            runner.native(quant, Path('/a'), Path('/b'), 'off')
        self.assertEqual(call.call_args.kwargs['env']['CRISPEMBED_QWEN35_DIRECT_F16'], '0')
        self.assertEqual(call.call_args.kwargs['env']['CRISPEMBED_QWEN35_PROFILE'], '0')

    def test_exact_gate_rejects_wrong_stop_or_tokens(self):
        runner = module('speed_lane_compare', LANE / 'run.py')
        quant = module('quant_helpers_compare', QUANT)
        original = {'token_ids': [3, 2], 'text': 'hi', 'stop_eos': True}
        candidate = {'token_ids': [3, 2], 'text': 'hi', 'stop_kind': 1}
        self.assertTrue(runner.valid_comparison(runner.compare(quant, original, candidate)))
        for bad in ({**candidate, 'stop_kind': 2}, {**candidate, 'token_ids': [4, 2]},
                    {**candidate, 'text': 'different'}):
            self.assertFalse(runner.valid_comparison(runner.compare(quant, original, bad)))


if __name__ == '__main__':
    unittest.main()
