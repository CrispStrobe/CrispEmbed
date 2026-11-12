"""Model-free book trace admission controls; no builds or inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import source_patch


class Controls(unittest.TestCase):
    def test_independent_patch_configuration(self):
        configured = source_patch.configured()
        self.assertEqual(source_patch.base.ENV, 'CRISPEMBED_TEST_UPSTREAM_TEXT_FORM_TRACE_DIR')
        self.assertIn('(n==0||n==31)', source_patch.base.HELPER)
        self.assertIn('(n==0||n==405)', configured.HELPER)
        self.assertIn('16u<<20', configured.HELPER)
        self.assertIn('256u<<20', configured.HELPER)
        self.assertEqual(configured.STAGES, source_patch.base.STAGES)
        self.assertEqual(len(set(configured.STAGES)), len(configured.STAGES))

    def test_drift_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['src/llama-context.cpp', 'common/sampling.cpp']:
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_text('wrong')
            with self.assertRaisesRegex(ValueError, 'source drift'):
                source_patch.patch(root)
            self.assertEqual((root / 'src/llama-context.cpp').read_text(), 'wrong')
            self.assertEqual((root / 'common/sampling.cpp').read_text(), 'wrong')

    def test_response_guards(self):
        spec = importlib.util.spec_from_file_location('book_worker', Path(__file__).with_name('worker.py'))
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        frozen = dict(tokens=[2]*653+[248046], content='book', completion_probabilities=[{}]*654,
                      stop_type='eos', tokens_evaluated=256, truncated=False)
        worker.validate_response(frozen, frozen)
        for key, value in [('tokens', frozen['tokens'][:-1]), ('content', 'changed'),
                           ('stop_type', 'limit'), ('truncated', True), ('tokens_evaluated', 255)]:
            altered = dict(frozen, **{key: value})
            with self.assertRaises(RuntimeError):
                worker.validate_response(altered, frozen)
        cfg = json.loads(Path(__file__).with_name('protocol.json').read_text())
        worker.admit(cfg)
        for key, value in [('same_prefix_tokens', 404), ('full_f32_model_allowed', True),
                           ('capture_sample_counts', [0, 92]), ('request_cap', 512)]:
            with self.assertRaises(RuntimeError):
                worker.admit(dict(cfg, **{key: value}))


if __name__ == '__main__':
    unittest.main()
