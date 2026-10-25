"""Small source-only controls for the undispatched OvisOCR3 T4 recipe."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'tools/kaggle/ovisocr3-vllm-reference'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecipeControls(unittest.TestCase):
    def test_unreviewed_metadata_fails_before_work(self):
        runner = load('ovisocr3_gpu_runner', DIR / 'ovisocr3_vllm_reference.py')
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            runner.configured()
        self.assertEqual(runner.METADATA_RUN, 0)
        self.assertEqual(runner.MODEL, 'ATH-MaaS/OvisOCR3')
        self.assertEqual(runner.CHECKPOINT_BYTES, 1706030528)
        self.assertEqual(runner.CHECKPOINT_SHA,
                         'aa2311c6b22b07cf3afe9aac6e8bcb02a2b19dd8671b2a91d0b144071577e600')

    def test_metadata_file_and_kernel_identity(self):
        metadata = json.loads((DIR / 'kernel-metadata.json').read_text())
        self.assertEqual(metadata['code_file'], 'ovisocr3_vllm_reference.py')
        self.assertEqual(metadata['enable_gpu'], 'true')
        self.assertEqual(metadata['model_sources'], [])
        runner = load('ovisocr3_gpu_runner_identity', DIR / 'ovisocr3_vllm_reference.py')
        self.assertEqual(runner.SOURCE_EOS, [248044, 248046])
        self.assertEqual(runner.REVISION, '5f6ce6932efee9ec10c30746db55bd780214b7fd')

    def test_selected_score_required_each_step(self):
        import os
        prior = os.environ.get('OVIS_OUT')
        os.environ['OVIS_OUT'] = '/tmp'
        try:
            worker = load('ovisocr3_gpu_worker', DIR / 'worker.py')
        finally:
            if prior is None:
                del os.environ['OVIS_OUT']
            else:
                os.environ['OVIS_OUT'] = prior
        score = lambda value: SimpleNamespace(logprob=value, rank=1, decoded_token='x')
        result = SimpleNamespace(token_ids=[42, 43],
                                 logprobs=[{42: score(-0.25), 3: score(-1.5)}, {43: score(-0.1)}])
        rows = worker.logprob_steps(result)
        self.assertEqual([row['selected_token_id'] for row in rows], [42, 43])
        self.assertEqual(rows[0]['selected_logprob'], -0.25)
        result.logprobs[1] = {44: score(-0.1)}
        with self.assertRaisesRegex(ValueError, 'selected'):
            worker.logprob_steps(result)
        result.logprobs[1] = {43: score(float('nan'))}
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            worker.logprob_steps(result)


if __name__ == '__main__':
    unittest.main()
