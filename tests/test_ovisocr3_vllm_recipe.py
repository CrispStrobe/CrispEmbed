"""Small source-only controls for the undispatched OvisOCR3 T4 recipe."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'tools/kaggle/ovisocr3-vllm-reference'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecipeControls(unittest.TestCase):
    def test_reviewed_metadata_pins_and_missing_review_fail_before_work(self):
        runner = load('ovisocr3_gpu_runner', DIR / 'ovisocr3_vllm_reference.py')
        runner.configured()
        self.assertEqual(runner.METADATA_RUN, 40172525903)
        self.assertEqual(runner.PROOF_COMMIT, 'd6f954c58f402c686949a69fdec2f76acdf60079')
        runner.METADATA_RUN = 0
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            runner.configured()
        self.assertEqual(runner.MODEL, 'ATH-MaaS/OvisOCR3')
        self.assertEqual(runner.CHECKPOINT_BYTES, 1706030528)
        self.assertEqual(runner.CHECKPOINT_SHA,
                         'aa2311c6b22b07cf3afe9aac6e8bcb02a2b19dd8671b2a91d0b144071577e600')
        self.assertEqual(runner.WORKER_SHA, runner.digest(DIR / 'worker.py'))

    def test_metadata_file_and_kernel_identity(self):
        metadata = json.loads((DIR / 'kernel-metadata.json').read_text())
        self.assertEqual(metadata['code_file'], 'ovisocr3_vllm_reference.py')
        self.assertEqual(metadata['enable_gpu'], 'true')
        self.assertEqual(metadata['model_sources'], [])
        runner = load('ovisocr3_gpu_runner_identity', DIR / 'ovisocr3_vllm_reference.py')
        self.assertEqual(runner.SOURCE_EOS, [248044, 248046])
        self.assertEqual(runner.REVISION, '5f6ce6932efee9ec10c30746db55bd780214b7fd')
        # The pinned OvisOCR3 README passes a list even for one image.
        self.assertIn("'multi_modal_data': {'image': [image.convert('RGB')]}",
                      (DIR / 'worker.py').read_text())

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

    def test_first_gpu_only_no_redraw_with_multiple_visible(self):
        runner = load('ovisocr3_gpu_runner_selection', DIR / 'ovisocr3_vllm_reference.py')
        runner.save = lambda report: None
        two = 'Tesla T4, 7.5, 580.178.04, 15109\nTesla T4, 7.5, 580.178.04, 15109\n'
        with mock.patch.object(runner.subprocess, 'check_output', return_value=two):
            report = {}
            runner.gpu_gate(report)
        self.assertEqual(report['hardware']['selected_physical_index'], 0)
        self.assertEqual(report['hardware']['visible_gpu_count'], 2)
        wrong_first = 'Tesla P100, 6.0, 580.178.04, 16280\n' + two.splitlines()[0] + '\n'
        with mock.patch.object(runner.subprocess, 'check_output', return_value=wrong_first):
            with self.assertRaisesRegex(ValueError, 'not pinned T4'):
                runner.gpu_gate({})


if __name__ == '__main__':
    unittest.main()
