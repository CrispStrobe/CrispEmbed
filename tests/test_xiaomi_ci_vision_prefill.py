"""Weight-free contracts for the GitHub-hosted Xiaomi vision audit."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/ci-heavy/xiaomi_vision_prefill.py'


def load():
    spec = importlib.util.spec_from_file_location('xiaomi_ci_vision', SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisionCiTests(unittest.TestCase):
    def test_missing_operator_sources_fails_closed_before_retrieval(self):
        module = load()
        with patch.object(module, 'download') as download:
            with self.assertRaisesRegex(ValueError, 'approved HTTPS'):
                module.retrieve_inputs(None, None, Path('/unused'), {})
        download.assert_not_called()

    def test_mismatch_is_exit_one_not_setup_failure(self):
        module = load()
        rows = []
        for name, shape in module.EXPECTED.items():
            failed = name.endswith('patch_embed')
            rows.append({'name': name, 'shape': shape, 'n': shape[0] * shape[1],
                         'pass': not failed, 'max_abs': 0.003 if failed else 0.001,
                         'mean_abs': 0.0001, 'rms': 0.0002, 'cos_min': 0.999999,
                         'cos_mean': 1.0, 'cos_global': 1.0, 'native_norm': 1.0,
                         'reference_norm': 1.0, 'input_fnv1a': '0123456789abcdef'})
        log = 'AUDIT_BACKEND CPU threads=4\nNEGATIVE_CONTROL rejected=1 name=patch max_abs=1\n' + ''.join(
            'STAGE ' + json.dumps(row) + '\n' for row in rows)
        report = {}
        self.assertEqual(module.evaluate(log, 1, report), 1)
        self.assertEqual(report['first_failing_stage'], rows[0]['name'])
        self.assertEqual(len(report['stages']), 21)
        with self.assertRaisesRegex(ValueError, 'exit'):
            module.evaluate(log, 2, {})
        with self.assertRaisesRegex(ValueError, 'exit'):
            module.evaluate(log, 0, {})

    def test_hash_mismatch_removes_download(self):
        module = load()
        import io
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.gguf'
            with patch.object(module, 'open_url', return_value=io.BytesIO(b'wrong')):
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    module.download('https://example.org/model', path, '0' * 64, 100)
            self.assertFalse(path.exists())

    def test_download_accepts_matching_bytes(self):
        module = load()
        import hashlib
        import io
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'reference.gguf'
            with patch.object(module, 'open_url', return_value=io.BytesIO(b'fixture')):
                result = module.download('https://example.org/ref', path, hashlib.sha256(b'fixture').hexdigest(), 100)
            self.assertEqual(path.read_bytes(), b'fixture')
            self.assertEqual(result['bytes'], 7)

    def test_download_mocks_are_called_with_distinct_pins(self):
        module = load()
        with tempfile.TemporaryDirectory() as temp, patch.object(module, 'download') as download:
            module.retrieve_inputs('https://example.org/model', 'https://example.org/ref', Path(temp), {})
            self.assertEqual([c.args[2] for c in download.call_args_list], [module.MODEL_SHA, module.REF_SHA])

    def test_main_records_strict_mismatch_and_exits_one(self):
        module = load()
        import sys
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(module.os.environ, {'HEAVY_OUT': temp + '/out', 'HEAVY_SCRATCH': temp + '/scratch'}), \
             patch.object(sys, 'argv', ['script']), patch.object(module, 'run', return_value=1):
            self.assertEqual(module.main(), 1)
            report = json.loads((Path(temp) / 'out/report.json').read_text())
            self.assertEqual(report['status'], 'strict_mismatch')
            self.assertEqual(report['exit'], 1)

    def test_main_redacts_retrieval_exception(self):
        module = load()
        import sys
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(module.os.environ, {'HEAVY_OUT': temp + '/out', 'HEAVY_SCRATCH': temp + '/scratch'}), \
             patch.object(sys, 'argv', ['script']), \
             patch.object(module, 'run', side_effect=ValueError('https://secret.example/?token=secret')):
            self.assertEqual(module.main(), 2)
            report = (Path(temp) / 'out/report.json').read_text()
            self.assertNotIn('secret', report)
            self.assertEqual(json.loads(report)['status'], 'setup_failure')

    def test_base_dependencies_are_checked_against_pinned_revision(self):
        module = load()
        calls = []
        with patch.object(module, 'git', side_effect=lambda args, cwd, env: calls.append(args) or ''):
            module.verify_base_dependencies({})
        self.assertEqual(calls[0][:5], ['diff', '--quiet', module.BASE, 'HEAD', '--'])
        self.assertIn('src/qwen35_linear.cpp', calls[0])
        self.assertIn('src/crispembed_diff.h', calls[0])
        self.assertNotIn('src/qwen35_vision.cpp', calls[0])

    def test_standalone_cmake_uses_cache_implementation_without_phantom_file(self):
        module = load()
        original = (ROOT / 'tools/kaggle/xiaomi-ocr-0-vision-prefill/CMakeLists.txt').read_text()
        transformed = module.prepare_cmake(original)
        self.assertNotIn('src/qwen35_weight_cache.cpp', transformed)
        self.assertIn('    src/qwen35_linear.cpp\n', transformed)
        self.assertIn('converted_weight_cache::get(', (ROOT / 'src/qwen35_linear.cpp').read_text())
        self.assertEqual(original, transformed)
        with self.assertRaisesRegex(ValueError, 'CMake'):
            module.prepare_cmake(original + '\nsrc/qwen35_weight_cache.cpp')

    def test_configure_failure_is_reported_as_setup_not_mismatch(self):
        module = load()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = {'commands': []}
            args = type('Args', (), {'model_url': 'https://example.org/model',
                                     'reference_url': 'https://example.org/ref'})()
            failed = type('Result', (), {'returncode': 1})()
            with patch.object(module, 'require_hosted'), patch.object(module, 'source_closure'), \
                 patch.object(module, 'retrieve_inputs', return_value=(root / 'm', root / 'r')), \
                 patch.object(module.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 16 * 1024**3})()), \
                 patch.object(module, 'command', return_value=failed):
                with self.assertRaisesRegex(ValueError, 'configure failed'):
                    module.run(args, root, root, report)

    def test_hosted_guard_and_url_secrets(self):
        module = load()
        with patch.dict(module.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'GitHub-hosted'):
                module.require_hosted()
        for url in ('http://example.org/a', 'https://u:p@example.org/a',
                    'https://example.org/a?token=secret', 'https://example.org/a#secret'):
            with self.assertRaises(ValueError):
                module.check_url(url)


if __name__ == '__main__':
    unittest.main()
