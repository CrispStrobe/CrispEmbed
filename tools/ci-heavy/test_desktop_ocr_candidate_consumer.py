"""Source-only checks for the bounded manual consumer; no compiler or model use."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import tomllib
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('consumer', Path(__file__).with_name('desktop_ocr_candidate_consumer.py'))
consumer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consumer)


class CandidateConsumerTests(unittest.TestCase):
    def service(self):
        return dict(head_sha=consumer.WORKFLOW_SOURCE, path='.github/workflows/release.yml',
                    event='workflow_dispatch', status='completed', conclusion='success')

    def test_existing_producer_and_selectable_desktop_matrix(self):
        all_targets = consumer.plan('all', self.service())['include']
        self.assertEqual(len(all_targets), 4)
        self.assertEqual({item['artifact'] for item in all_targets},
                         {'linux-x86_64', 'linux-arm64', 'macos-arm64', 'windows-x86_64'})
        for item in all_targets:
            self.assertEqual(consumer.plan(item['artifact'], self.service()), {'include': [item]})
        self.assertEqual(consumer.PRODUCER_RUN, 43302121233)

    def test_other_producer_failed_run_or_non_desktop_target_rejected(self):
        for key, value in [('head_sha', 'a' * 40), ('event', 'push'), ('status', 'queued'),
                           ('conclusion', 'failure'), ('path', '.github/workflows/release-wasm.yml')]:
            service = self.service()
            service[key] = value
            with self.assertRaises(ValueError):
                consumer.plan('linux-x86_64', service)
        with self.assertRaises(ValueError):
            consumer.plan('android-arm64', self.service())

    def test_generated_image_is_complete_small_24_bit_bmp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'white.bmp'
            consumer.write_image(path)
            data = path.read_bytes()
            self.assertEqual(data[:2], b'BM')
            self.assertEqual(struct.unpack_from('<I', data, 2)[0], len(data))
            self.assertEqual(struct.unpack_from('<ii', data, 18), (32, 32))
            self.assertEqual(data[54:], b'\xff' * (32 * 32 * 3))

    def test_execution_report_cannot_be_missing_false_or_duplicated(self):
        report = {key: True for key in consumer.NATIVE_KEYS}
        line = consumer.MARKER + json.dumps(report)
        self.assertEqual(consumer.parse_control('native log\n' + line, consumer.NATIVE_KEYS), report)
        for text in ['', line + '\n' + line, consumer.MARKER + '{}',
                     consumer.MARKER + json.dumps({key: False for key in report})]:
            with self.assertRaises(ValueError):
                consumer.parse_control(text, consumer.NATIVE_KEYS)

    def test_rust_harness_preserves_locked_dependency_versions_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'rust'
            before = (ROOT / 'Cargo.lock').read_bytes()
            consumer.rust_project(ROOT, project)
            dependencies = tomllib.loads((project / 'Cargo.toml').read_text())['dependencies']
            self.assertEqual(dependencies['crispembed']['path'], (ROOT / 'crispembed').as_posix())
            self.assertEqual(dependencies['crispembed']['version'], '=0.17.19')
            locked = tomllib.loads((project / 'Cargo.lock').read_text())['package']
            original = tomllib.loads(before.decode())['package']
            self.assertEqual(locked[:-1], original)
            self.assertEqual(locked[-1]['dependencies'], ['crispembed'])
            self.assertEqual((ROOT / 'Cargo.lock').read_bytes(), before)

    def test_rust_link_proof_requires_the_verified_prebuilt_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'debug/build/crispembed-sys-fixture/output'
            output.parent.mkdir(parents=True)
            package = root / 'verified-package'
            output.write_text('cargo:LIB_DIR=' + str(package) + '\n')
            self.assertEqual(consumer.prebuilt_metadata(root, package), output)
            output.write_text('cargo:LIB_DIR=' + str(root / 'source-build') + '\n')
            with self.assertRaises(ValueError):
                consumer.prebuilt_metadata(root, package)

    def test_actual_native_and_rust_sources_do_not_substitute_mocked_results(self):
        self.assertIn('#include "crispembed.h"', consumer.NATIVE_SOURCE)
        self.assertIn('crispembed_ocr_pipeline_run(ctx,path', consumer.NATIVE_SOURCE)
        self.assertIn('crispembed_ocr_pipeline_review_json(ctx', consumer.NATIVE_SOURCE)
        self.assertIn('pipeline(&[failed.clone()]).run(&args[1]).is_err()', consumer.RUST_SOURCE)
        self.assertIn('pipeline(&[empty]).run(&args[1]).unwrap()', consumer.RUST_SOURCE)
        self.assertIn('result.full_text.is_empty() && result.regions.is_empty()', consumer.RUST_SOURCE)

    def test_workflow_is_manual_read_only_and_does_not_produce_native_assets(self):
        workflow = (ROOT / '.github/workflows/desktop-ocr-candidate-consumer.yml').read_text()
        for forbidden in ('  push:', '  pull_request:', 'concurrency:', 'contents: write', 'actions: write',
                          'gh workflow run', 'cmake --build', 'release.yml --ref', 'release-wasm'):
            self.assertNotIn(forbidden, workflow)
        self.assertIn('default: linux-x86_64', workflow)
        self.assertIn('ref: ' + consumer.WORKFLOW_SOURCE, workflow)
        self.assertIn('ref: ' + consumer.RUNTIME_SOURCE, workflow)
        self.assertEqual(workflow.count('submodules: recursive'), 2)
        self.assertIn("PYTHONDONTWRITEBYTECODE: '1'", workflow)

    def test_windows_checkout_preserves_raw_git_bytes_without_weakening_validation(self):
        workflow = (ROOT / '.github/workflows/desktop-ocr-candidate-consumer.yml').read_text()
        step = workflow.split('      - name: Preserve committed source bytes on Windows\n', 1)[1].split('      - name:', 1)[0]
        self.assertIn("if: runner.os == 'Windows'", step)
        self.assertLess(workflow.index('Preserve committed source bytes on Windows'), workflow.index('Check out this consumer controller'))
        script = '\n'.join(line[10:] for line in step.split('        run: |\n', 1)[1].splitlines())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'repo'
            repo.mkdir()
            env = dict(os.environ, GIT_CONFIG_GLOBAL=str(root / 'gitconfig'))
            def git(*args):
                return subprocess.check_output(['git', '-C', str(repo), *args], env=env, stderr=subprocess.DEVNULL)
            git('init', '-q')
            git('config', '--global', 'core.autocrlf', 'true')
            source = repo / 'control.py'
            source.write_bytes(b'print("source identity")\n')
            (repo / '.gitattributes').write_text('*.raw -text\n')
            retained = repo / 'capture.raw'
            retained.write_bytes(b'retained\r\nbytes\r\n')
            git('add', '.')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
            committed = git('show', 'HEAD:control.py')
            source.unlink()
            git('checkout', '--', 'control.py')
            self.assertIn(b'\r\n', source.read_bytes())
            self.assertNotEqual(source.read_bytes(), committed)
            self.assertEqual(git('status', '--porcelain'), b'')
            subprocess.run(['bash', '-c', script], env=env, check=True)
            source.unlink()
            retained.unlink()
            git('checkout', '--', 'control.py', 'capture.raw')
            self.assertEqual(source.read_bytes(), committed)
            self.assertEqual(retained.read_bytes(), b'retained\r\nbytes\r\n')
            self.assertEqual(git('status', '--porcelain'), b'')
            source.write_bytes(committed + b'# changed\n')
            self.assertNotEqual(source.read_bytes(), committed)

    def test_harness_keeps_inputs_fixed_and_limits_unexecuted_qualification(self):
        source = Path(consumer.__file__).read_text()
        self.assertIn('validator_env = dict(env, GITHUB_SHA=WORKFLOW_SOURCE)', source)
        self.assertIn('controller_source_commit=controller', source)
        self.assertIn("'--desktop-candidate'", source)
        self.assertIn("'--runtime-source-commit', RUNTIME_SOURCE", source)
        self.assertIn("'cargo', 'run', '--quiet', '--locked'", source)
        self.assertIn('app_boundary_executed=False', source)
        self.assertIn('positive_blank_model_executed=False', source)
        self.assertIn('decoded_ocr_executed=False', source)
        self.assertIn('gpu_execution_qualified=False, qualification=False', source)

    def test_local_execution_rejected_before_any_job_or_build(self):
        with patch.dict(os.environ, {'GITHUB_ACTIONS': 'false'}), self.assertRaisesRegex(ValueError, 'Hosted'):
            consumer.consume('linux-x86_64', ROOT, ROOT, ROOT)

    def header_member(self, data):
        return {'kind': 'file', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    def test_header_binding_accepts_only_exact_bytes_or_windows_whole_file_transform(self):
        source = b'// UTF-8: \xe2\x96\x81\nint fixture(void);\n'
        exact = consumer.header_binding(source, source, self.header_member(source), False)
        self.assertEqual(exact['transformation'], 'exact-bytes')
        windows = source.replace(b'\n', b'\r\n')
        proof = consumer.header_binding(windows, source, self.header_member(windows), True)
        self.assertEqual(proof['transformation'], 'windows-git-whole-file-lf-to-crlf')
        self.assertEqual(proof['packaged_sha256'], self.header_member(windows)['sha256'])
        self.assertEqual(proof['committed_source_sha256'], self.header_member(source)['sha256'])
        self.assertNotEqual(proof['packaged_sha256'], proof['committed_source_sha256'])
        with self.assertRaisesRegex(ValueError, 'exact runtime source'):
            consumer.header_binding(windows, source, self.header_member(windows), False)

    def test_header_binding_rejects_mixed_endings_lone_cr_and_every_other_edit(self):
        source = b'// source\nint fixture(void);\n'
        windows = source.replace(b'\n', b'\r\n')
        for altered in [windows.replace(b'\r\n', b'\n', 1), windows.replace(b'\r\n', b'\r', 1),
                        windows.replace(b'int', b'long'), windows + b'\r\n', windows[:-1],
                        windows.replace(b'\r\n', b'\r\r\n'), b'\xef\xbb\xbf' + windows]:
            # Even a raw member hash matching these bytes cannot waive source binding.
            with self.assertRaisesRegex(ValueError, 'exact runtime source'):
                consumer.header_binding(altered, source, self.header_member(altered), True)
        mixed_source = b'// source\r\nint fixture(void);\n'
        with self.assertRaisesRegex(ValueError, 'exact runtime source'):
            consumer.header_binding(windows, mixed_source, self.header_member(windows), True)

    def test_header_binding_keeps_raw_manifest_hash_and_size_checks_strict(self):
        source = b'int fixture(void);\n'
        windows = source.replace(b'\n', b'\r\n')
        for member in [self.header_member(source), {**self.header_member(windows), 'kind': 'symlink'},
                       {**self.header_member(windows), 'bytes': len(windows) + 1}]:
            with self.assertRaisesRegex(ValueError, 'verified raw manifest member'):
                consumer.header_binding(windows, source, member, True)

    def test_header_verifier_preserves_downloaded_manifest_and_extracted_member_binding(self):
        source = b'int fixture(void);\n'
        windows = source.replace(b'\n', b'\r\n')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, package, output = (root / name for name in ('runtime', 'package', 'output'))
            (runtime / 'src').mkdir(parents=True)
            (runtime / 'src/crispembed.h').write_bytes(source)
            (package / 'include').mkdir(parents=True)
            header = package / 'include/crispembed.h'
            header.write_bytes(windows)
            output.mkdir()
            archive = 'crispembed-windows-x86_64.zip'
            manifest = {'files': [{'name': archive, 'members': [dict(self.header_member(windows), path='include/crispembed.h')]}]}
            path = output / 'manifest.zip'
            with zipfile.ZipFile(path, 'w') as bundle:
                bundle.writestr('successor-native-manifest.json', json.dumps(manifest))
            receipt = {'manifest_artifact': {'digest': 'sha256:' + consumer.digest(path)}, 'archives': [{'name': archive}]}
            with patch.object(consumer.subprocess, 'check_output', return_value=source):
                proof = consumer.verify_header(package, runtime, output, receipt, True)
                self.assertEqual(proof['transformation'], 'windows-git-whole-file-lf-to-crlf')
                header.write_bytes(source)
                with self.assertRaisesRegex(ValueError, 'verified raw manifest member'):
                    consumer.verify_header(package, runtime, output, receipt, True)
                header.write_bytes(windows)
                path.write_bytes(path.read_bytes() + b'altered')
                with self.assertRaisesRegex(ValueError, 'manifest artifact bytes changed'):
                    consumer.verify_header(package, runtime, output, receipt, True)


if __name__ == '__main__':
    unittest.main()
