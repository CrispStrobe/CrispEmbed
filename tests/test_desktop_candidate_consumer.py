"""Offline producer-service fixtures with separate real runtime/tooling commits."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
import successor_assets as consumer
sys.path.insert(0, str(ROOT / 'tools'))
import successor_asset_manifest as manifest_tool


class DesktopConsumerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        files = {'VERSION': '1.0.0\n', 'ggml': 'fixture gitlink', 'src/crispembed.h': 'runtime header',
                 'crispembed/src/lib.rs': 'runtime binding', 'tools/successor_asset_manifest.py': 'old generator',
                 'vendor/native.h': 'vendor input', 'cmake/native.cmake': 'build input', '.gitignore': 'ignored-input.h\n',
                 '.github/workflows/release.yml': 'old recipe', 'tools/ci-heavy/successor_assets.py': 'old consumer'}
        for path, content in files.items():
            target = self.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        self.commit('runtime')
        self.runtime = self.git('rev-parse', 'HEAD')
        self.runtime_checkout = self.root / 'runtime'
        self.git('worktree', 'add', '-q', '--detach', str(self.runtime_checkout), self.runtime)
        for path in ('tools/successor_asset_manifest.py', '.github/workflows/release.yml', 'tools/ci-heavy/successor_assets.py'):
            (self.repo / path).write_bytes((ROOT / path).read_bytes())
        self.commit('workflow')
        self.workflow = self.git('rev-parse', 'HEAD')
        self.payload = self.root / 'payload'
        self.payload.mkdir()
        for name in manifest_tool.DESKTOP_REQUIRED:
            if name.endswith('.zip'):
                with zipfile.ZipFile(self.payload / name, 'w') as archive:
                    archive.writestr('fixture.txt', 'fixture')
            else:
                with tarfile.open(self.payload / name, 'w:gz') as archive:
                    item = tarfile.TarInfo('fixture.txt')
                    item.size = 7
                    archive.addfile(item, io.BytesIO(b'fixture'))
        self.manifest = manifest_tool.build(self.payload, 'native', self.runtime_checkout,
                                           scope='desktop', workflow_source_root=self.repo)
        self.service = {'head_sha': self.workflow, 'path': '.github/workflows/release.yml',
                        'status': 'completed', 'conclusion': 'success', 'event': 'workflow_dispatch'}

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True).strip()

    def commit(self, message):
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', message)

    def artifact(self, run, name, destination, limit):
        with zipfile.ZipFile(destination, 'w') as archive:
            if name == 'successor-native-manifest':
                archive.writestr('successor-native-manifest.json', json.dumps(self.manifest))
            else:
                archive.write(self.payload / 'crispembed-linux-x86_64.tar.gz', 'crispembed-linux-x86_64.tar.gz')
        return {'id': 1, 'name': name, 'size_in_bytes': destination.stat().st_size,
                'digest': 'sha256:' + manifest_tool.sha(destination)}

    def prepare(self, **kwargs):
        options = dict(desktop_candidate=True, runtime_source_commit=self.runtime, runtime_source_root=self.runtime_checkout)
        options.update(kwargs)
        with mock.patch.object(consumer, 'ROOT', self.repo), mock.patch.object(consumer, 'api', return_value=self.service), \
             mock.patch.object(consumer, 'artifact', side_effect=self.artifact), \
             mock.patch.dict(os.environ, {'GITHUB_ACTIONS': 'true', 'GITHUB_SHA': self.git('rev-parse', 'HEAD')}):
            return consumer.prepare(1, 'native', 'linux-x86_64', self.root / 'out', **options)

    def test_dual_sources_validate_and_receipt_preserves_both_identities(self):
        receipt = self.prepare()
        self.assertEqual(receipt['source_commit'], self.runtime)
        self.assertEqual(receipt['workflow_source_commit'], self.workflow)
        self.assertNotEqual(self.runtime, self.workflow)
        self.assertTrue(receipt['runtime_source_closure']['runtime_tree'])
        self.assertTrue(receipt['runtime_source_closure']['workflow_tree'])
        self.assertFalse(receipt['qualification'])

    def test_legacy_validation_does_not_silently_accept_desktop_scope(self):
        with self.assertRaisesRegex(ValueError, 'explicit desktop-candidate'):
            self.prepare(desktop_candidate=False)

    def test_expected_runtime_pin_is_required(self):
        with self.assertRaisesRegex(ValueError, 'expected desktop runtime'):
            self.prepare(runtime_source_commit=None)

    def test_manifest_cannot_substitute_workflow_head_for_runtime_source(self):
        self.manifest['source_commit'] = self.workflow
        with self.assertRaisesRegex(ValueError, 'expected immutable source'):
            self.prepare()

    def test_spoofed_workflow_identity_rejected(self):
        self.manifest['workflow_source_commit'] = self.runtime
        with self.assertRaisesRegex(ValueError, 'workflow identity'):
            self.prepare()

    def test_wrong_generator_hash_rejected(self):
        self.manifest['generator_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Manifest/source closure'):
            self.prepare()

    def test_runtime_binding_drift_rejected(self):
        (self.runtime_checkout / 'crispembed/src/lib.rs').write_text('changed binding')
        with self.assertRaisesRegex(ValueError, 'runtime complete source tree'):
            self.prepare()

    def test_workflow_recipe_drift_rejected(self):
        (self.repo / '.github/workflows/release.yml').write_text('changed recipe')
        self.commit('unreviewed recipe drift')
        with self.assertRaisesRegex(ValueError, 'workflow checkout identity'):
            self.prepare()

    def test_separate_runtime_checkout_is_required(self):
        with self.assertRaisesRegex(ValueError, 'Separate exact desktop runtime'):
            self.prepare(runtime_source_root=None)

    def test_vendor_and_build_inputs_are_checked_without_path_allowlist(self):
        for relative in ('vendor/native.h', 'cmake/native.cmake'):
            path = self.runtime_checkout / relative
            original = path.read_bytes()
            path.write_text('changed build input')
            with mock.patch.object(consumer, 'ROOT', self.repo), self.assertRaisesRegex(ValueError, 'runtime complete source tree'):
                consumer.desktop_source_closure(self.runtime, self.workflow, self.runtime_checkout)
            path.write_bytes(original)

    def test_new_untracked_and_ignored_inputs_are_rejected(self):
        for relative in ('vendor/future-input.h', 'ignored-input.h'):
            path = self.runtime_checkout / relative
            path.write_text('injected build input')
            with mock.patch.object(consumer, 'ROOT', self.repo), self.assertRaisesRegex(ValueError, 'runtime complete source tree'):
                consumer.desktop_source_closure(self.runtime, self.workflow, self.runtime_checkout)
            path.unlink()

    def test_desktop_execution_revalidates_source_before_loading_assets(self):
        self.prepare()
        (self.runtime_checkout / 'vendor/native.h').write_text('changed after validation')
        with mock.patch.object(consumer, 'ROOT', self.repo), self.assertRaisesRegex(ValueError, 'runtime complete source tree'):
            consumer.desktop(self.root / 'out', runtime_source_root=self.runtime_checkout)

    def add_runtime_submodule(self):
        dependency = self.root / 'dependency'
        dependency.mkdir()
        subprocess.run(['git', 'init', '-q', str(dependency)], check=True)
        (dependency / 'native.h').write_text('dependency input')
        (dependency / '.gitignore').write_text('ignored.h\n')
        subprocess.run(['git', '-C', str(dependency), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(dependency), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'dependency'], check=True)
        subprocess.run(['git', '-C', str(self.runtime_checkout), '-c', 'protocol.file.allow=always', 'submodule', 'add', '-q', str(dependency), 'dependency'], check=True)
        subprocess.run(['git', '-C', str(self.runtime_checkout), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qam', 'pin dependency'], check=True)
        self.runtime = subprocess.check_output(['git', '-C', str(self.runtime_checkout), 'rev-parse', 'HEAD'], text=True).strip()

    def test_ignored_submodule_inputs_are_rejected(self):
        self.add_runtime_submodule()
        with mock.patch.object(consumer, 'ROOT', self.repo):
            consumer.desktop_source_closure(self.runtime, self.workflow, self.runtime_checkout)
            (self.runtime_checkout / 'dependency/ignored.h').write_text('injected ignored dependency')
            with self.assertRaisesRegex(ValueError, 'submodule source tree'):
                consumer.desktop_source_closure(self.runtime, self.workflow, self.runtime_checkout)

    def test_uninitialized_submodule_is_rejected(self):
        self.add_runtime_submodule()
        subprocess.run(['git', '-C', str(self.runtime_checkout), 'submodule', 'deinit', '-q', '-f', '--', 'dependency'], check=True)
        with mock.patch.object(consumer, 'ROOT', self.repo), self.assertRaisesRegex(ValueError, 'submodule identity'):
            consumer.desktop_source_closure(self.runtime, self.workflow, self.runtime_checkout)

    def test_incomplete_desktop_inventory_rejected(self):
        self.manifest['files'].pop()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.prepare()


if __name__ == '__main__':
    unittest.main()
