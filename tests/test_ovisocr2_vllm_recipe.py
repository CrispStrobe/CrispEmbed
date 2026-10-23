"""Source-only guards for the hosted original OvisOCR2 vLLM recipe."""

import ast
import importlib.util
import json
from pathlib import Path
import urllib.request
import unittest
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/kaggle/ovisocr2-vllm-reference/ovisocr2_vllm_reference.py'
spec = importlib.util.spec_from_file_location('ovisocr2_vllm_recipe', SCRIPT)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class RecipeControls(unittest.TestCase):
    def test_isolated_pip_does_not_depend_on_ensurepip_or_modify_base_environment(self):
        with mock.patch.object(recipe.venv, 'EnvBuilder') as builder, \
                mock.patch.object(recipe, 'run') as run, mock.patch.object(recipe, 'save'):
            report = {}
            python = recipe.isolated_environment(report)
            builder.assert_called_once_with(with_pip=False, clear=True)
            builder.return_value.create.assert_called_once_with(recipe.VENV)
            argv = run.call_args.args[1]
            self.assertEqual(argv[:3], [recipe.sys.executable, '-m', 'pip'])
            self.assertEqual(argv[3:5], ['--python', python])
            self.assertEqual(argv[-1], 'pip==' + recipe.ISOLATED_PIP_VERSION)
            self.assertEqual(report['isolated_pip_version'], recipe.ISOLATED_PIP_VERSION)

    def test_cuda_link_uses_real_driver_and_task_owned_directory_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            driver = root / 'libcuda.so.580.178.04'
            driver.write_bytes(b'controlled-driver-fixture')
            output = ' libcuda.so.1 (libc6,x86-64) => ' + str(driver) + '\n'
            original = {'LIBRARY_PATH': '/existing', 'OTHER': 'unchanged'}
            with mock.patch.object(recipe, 'SCRATCH', root), \
                    mock.patch.object(recipe.subprocess, 'check_output', return_value=output), \
                    mock.patch.object(recipe, 'save'):
                report = {}
                env = recipe.cuda_link_environment(report, original)
                link = root / 'cuda-driver-link/libcuda.so'
                self.assertTrue(link.is_symlink())
                self.assertEqual(link.resolve(), driver)
                self.assertEqual(env['LIBRARY_PATH'], str(link.parent) + ':/existing')
                self.assertEqual(original['LIBRARY_PATH'], '/existing')
                self.assertEqual(env['OTHER'], 'unchanged')
                self.assertFalse(report['cuda_driver_link']['system_files_modified'])
                self.assertEqual(report['cuda_driver_link']['sha256'], recipe.digest(driver))
                recipe.cuda_link_environment({}, {})  # repeat replaces only own symlink
            for bad in ('', ' libcuda.so (libc6) => ' + str(driver),
                        ' libcuda.so.1 (libc6) => ' + str(root / 'missing')):
                with mock.patch.object(recipe.subprocess, 'check_output', return_value=bad):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        recipe.cuda_link_environment({}, {})

    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads((ROOT / 'tests/results/qwen35-ocr/ovisocr2-metadata.json').read_text())

    def test_real_reviewed_producer_proof(self):
        report = {}
        with mock.patch.object(recipe, 'get_json', return_value=self.evidence), mock.patch.object(recipe, 'save'):
            self.assertEqual(recipe.metadata_proof(report), self.evidence)
            altered = json.loads(json.dumps(self.evidence))
            altered['producer']['tokenizer_eos'] = 248044
            with mock.patch.object(recipe, 'get_json', return_value=altered):
                with self.assertRaises(ValueError):
                    recipe.metadata_proof({})

    def test_frontend_requires_exact_successful_commit(self):
        run = {'id': recipe.FRONTEND_RUN, 'head_sha': recipe.FRONTEND_HEAD,
               'status': 'completed', 'conclusion': 'success',
               'path': '.github/workflows/ovisocr2-metadata.yml'}
        with mock.patch.object(recipe, 'get_json', return_value=run):
            self.assertEqual(recipe.service_gate(recipe.FRONTEND_RUN, recipe.FRONTEND_HEAD,
                                                 'frontend')['head_sha'], recipe.FRONTEND_HEAD)
        for change in ({'head_sha': recipe.FRONTEND_HEAD[:-1] + '0'},
                       {'conclusion': 'failure'}, {'status': 'queued'},
                       {'path': 'another-workflow.yml'}):
            with mock.patch.object(recipe, 'get_json', return_value=dict(run, **change)):
                with self.assertRaises(ValueError):
                    recipe.service_gate(recipe.FRONTEND_RUN, recipe.FRONTEND_HEAD, 'frontend')

    def test_driver_gate_precedes_gpu_and_installation(self):
        source = SCRIPT.read_text()
        self.assertLess(source.index("service_gate(FRONTEND_RUN"), source.index('        gpu_gate(report)'))
        self.assertLess(source.index('        gpu_gate(report)'), source.index('        python = isolated_environment(report)'))
        self.assertEqual(recipe.DRIVER_MIN, (575, 51, 3))
        self.assertIn('+cu129', recipe.WHEEL_NAME)
        self.assertEqual(len(recipe.WHEEL_SHA), 64)
        worker = ROOT / 'tools/kaggle/ovisocr2-vllm-reference/worker.py'
        self.assertEqual(recipe.digest(worker), recipe.WORKER_SHA)
        self.assertIn(recipe.WORKER_COMMIT, recipe.WORKER_URL)

    def test_cross_host_artifact_redirect_strips_authorization(self):
        request = urllib.request.Request('https://api.github.com/artifact',
                                         headers={'Authorization': 'Bearer private'})
        handler = recipe.StripCrossHostAuth()
        redirected = handler.redirect_request(request, None, 302, 'Found', {},
                                              'https://blob.example.invalid/zip')
        self.assertFalse(redirected.has_header('Authorization'))
        same_host = handler.redirect_request(request, None, 302, 'Found', {},
                                             'https://api.github.com/elsewhere')
        self.assertTrue(same_host.has_header('Authorization'))

    def test_deployed_kernel_uses_pinned_public_mirror_without_credential_dataset(self):
        metadata = json.loads((ROOT / 'tools/kaggle/ovisocr2-vllm-reference/kernel-metadata.json').read_text())
        self.assertEqual(metadata['dataset_sources'], [])
        self.assertEqual(metadata['id'], '${KAGGLE_ACCOUNT}/crispembed-ovisocr2-pinned-vllm-reference')
        self.assertEqual(metadata['code_file'], 'ovisocr2_vllm_reference.py')
        self.assertEqual(recipe.MIRROR_SHA, self.evidence['artifact']['zip_sha256'])
        self.assertEqual(recipe.ORIGINAL_ARTIFACT_ID, self.evidence['artifact']['id'])
        self.assertIn(recipe.MIRROR_REVISION, recipe.MIRROR_URL)
        self.assertEqual(recipe.MIRROR_BYTES, 524058)

    def test_import_does_not_load_model_or_cuda_frameworks(self):
        import sys
        for name in ('torch', 'transformers', 'vllm', 'numpy', 'PIL', 'huggingface_hub'):
            self.assertNotIn(name, sys.modules)

    def test_worker_retains_original_llm_precision_and_gdn_arguments(self):
        worker = ROOT / 'tools/kaggle/ovisocr2-vllm-reference/worker.py'
        tree = ast.parse(worker.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'LLM']
        self.assertEqual(len(calls), 1)
        kwargs = {key.arg: key.value for key in calls[0].keywords}
        self.assertNotIn('dtype', kwargs)
        self.assertEqual(ast.literal_eval(kwargs['tensor_parallel_size']), 1)
        self.assertEqual(ast.literal_eval(kwargs['gpu_memory_utilization']), 0.8)
        self.assertEqual(ast.literal_eval(kwargs['gdn_prefill_backend']), 'triton')
        self.assertIn("images_kwargs={'min_pixels': 448 * 448, 'max_pixels': 2880 * 2880}",
                      worker.read_text())


if __name__ == '__main__':
    unittest.main()
