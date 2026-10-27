#!/usr/bin/env python3
"""Mocked bootstrap negatives; no hardware, framework or payload access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import types
import shutil
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/kaggle/josev2b-original/bootstrap.py'


class BootstrapTests(unittest.TestCase):
    def load(self, directory):
        spec = importlib.util.spec_from_file_location('josev_bootstrap_test', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.OUT = Path(directory) / 'out'
        module.SCRATCH = Path(directory) / 'scratch'
        return module

    def reject(self, hardware, python, error):
        with tempfile.TemporaryDirectory() as directory:
            module = self.load(directory)
            with patch.object(module.subprocess, 'check_output', return_value=hardware), \
                 patch.object(module.sys, 'version_info', python), \
                 patch.object(module.sys, 'version', 'mock actual Python version'), \
                 patch.object(module, 'run', side_effect=AssertionError('must not install or access payload')):
                with self.assertRaisesRegex(ValueError, error):
                    module.main()
            receipt = json.loads((module.OUT / 'bootstrap.json').read_text())
            self.assertFalse(receipt['conclusive'])
            self.assertFalse(receipt['retry_permitted'])
            self.assertFalse(receipt['model_payload_accessed'])
            self.assertEqual(receipt['commands'], [])
            self.assertEqual(receipt['python'], 'mock actual Python version')
            self.assertTrue(receipt['hardware'])
            self.assertFalse(module.SCRATCH.exists())
            return receipt

    def test_unsupported_gpu_rejects_before_wrong_python(self):
        receipt = self.reject('Tesla P100, 6.0, 16384, 16000, 580.65.06\n', (3, 11), 'unsupported GPU')
        self.assertIn('no retry', receipt['error'])

    def test_supported_gpu_wrong_python_is_recorded_before_install(self):
        self.reject('Tesla T4, 7.5, 15360, 15000, 580.65.06\n', (3, 11), 'requires Python3.13')

    def test_old_driver_rejects_before_python_or_install(self):
        self.reject('Tesla T4, 7.5, 15360, 15000, 570.00.00\n', (3, 11), 'driver is below')

    def test_python313_still_requires_disk_before_install(self):
        with tempfile.TemporaryDirectory() as directory:
            module = self.load(directory)
            with patch.object(module.subprocess, 'check_output', return_value='Tesla T4, 7.5, 15360, 15000, 580.65.06\n'), \
                 patch.object(module.sys, 'version_info', (3, 13)), \
                 patch.object(module.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=10 * 2**30)), \
                 patch.object(module, 'run', side_effect=AssertionError('must not install')):
                with self.assertRaisesRegex(ValueError, 'scratch'):
                    module.main()
            self.assertFalse(module.SCRATCH.exists())
            self.assertFalse(json.loads((module.OUT / 'bootstrap.json').read_text())['conclusive'])

    def test_missing_ensurepip_uses_existing_isolated_bootstrap_route(self):
        with tempfile.TemporaryDirectory() as directory:
            module = self.load(directory)
            module.OUT.mkdir()
            with patch.object(module.venv, 'EnvBuilder') as factory, patch.object(module, 'run') as execute:
                python = module.isolated_environment({'PIP_USER':'0'})
            self.assertFalse(factory.call_args.kwargs['with_pip'])
            self.assertFalse(factory.call_args.kwargs['system_site_packages'])
            factory.return_value.create.assert_called_once_with(module.SCRATCH/'venv')
            args=execute.call_args.args[1]
            self.assertEqual(args[:5],[module.sys.executable,'-m','pip','--python',python])
            self.assertEqual(args[-1],module.PIP_BOOTSTRAP['url'])
            self.assertEqual(module.PIP_BOOTSTRAP['url'].split('#sha256=')[1],module.PIP_BOOTSTRAP['sha256'])
            self.assertFalse(module.REPORT['model_payload_accessed'])

    def test_complete_lock_is_verified_and_original_framework_pins_stay_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            module=self.load(directory);module.OUT.mkdir();repo=Path(directory)/'repo'
            cfg=json.loads((ROOT/'docs/josev2b-source-contract.json').read_text())
            for name in (cfg['environment_lock']['path'],cfg['environment_lock']['requirements_path']):
                target=repo/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
            path=module.locked_requirements(repo,cfg)
            self.assertEqual(path,repo/cfg['environment_lock']['requirements_path'])
            self.assertEqual(module.digest(path),cfg['environment_lock']['requirements_sha256'])
            self.assertEqual(module.REPORT['environment_lock']['packages'],57)
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'SHA'):module.locked_requirements(repo,cfg)
            self.assertFalse(module.REPORT['model_payload_accessed'])

    def installer_fixture(self,module):
        module.OUT.mkdir();output=module.OUT/'wheel-audit-install';output.mkdir()
        cfg=json.loads((ROOT/'docs/josev2b-source-contract.json').read_text());module.REPORT['environment_lock']=cfg['environment_lock']
        lock=(ROOT/cfg['environment_lock']['path']).read_bytes();(module.OUT/'environment-lock.json').write_bytes(lock)
        proof=dict(passed=True,completed=True,source=dict(head=module.WORKER_SOURCE),
            packages=[dict(name=r['name'],version=r['version'],audited=True,installed=True,archive=dict(passed=True)) for r in json.loads(lock)['packages']],pip_check=True,
            framework_imported=False,model_tokenizer_accessed=False,retry_permitted=False,
            installed_environment=dict(package_count=57,all_wheel_metadata_hashes_exact=True,exact_versions=True,
                owning_top_level_metadata_only=True,environment_lock_sha256=cfg['environment_lock']['sha256']))
        (output/'audit.json').write_text(json.dumps(proof));return output,proof

    def test_installer_requires_audited_gpu_cli_and_complete_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            module=self.load(directory);self.installer_fixture(module);repo=Path(directory)/'repo'
            with patch.object(module,'run') as execute:module.install_locked('/venv/bin/python',repo,{'PIP_USER':'0'})
            args=execute.call_args.args[1]
            self.assertEqual(args[:2],['/venv/bin/python',repo/'tools/ci-heavy/josev2b_gpu_install.py'])
            self.assertIn('--reviewed-source',args);self.assertIn('--gpu-preflight',args)
            self.assertEqual(execute.call_args.kwargs['env']['CRISPEMBED_JOSEV_GPU_REFERENCE_SETUP'],'1')
            self.assertEqual(execute.call_args.kwargs['timeout'],1500)
            self.assertFalse(module.REPORT['model_payload_accessed']);self.assertTrue(module.REPORT['wheel_audit_install']['passed'])

    def test_partial_or_unaudited_installer_cannot_reach_framework_barrier(self):
        for field in ('partial','duplicate','unaudited','owner','pipcheck','framework','source'):
            with tempfile.TemporaryDirectory() as directory:
                module=self.load(directory);out,proof=self.installer_fixture(module)
                if field=='partial':proof['packages'].pop()
                elif field=='duplicate':proof['packages'][1]=proof['packages'][0].copy()
                elif field=='unaudited':proof['packages'][0]['audited']=False
                elif field=='owner':proof['installed_environment']['all_wheel_metadata_hashes_exact']=False
                elif field=='pipcheck':proof['pip_check']=False
                elif field=='framework':proof['framework_imported']=True
                else:proof['source']['head']='0'*40
                (out/'audit.json').write_text(json.dumps(proof))
                with patch.object(module,'run'),self.assertRaisesRegex(ValueError,'barrier rejected'):
                    module.install_locked('/venv/bin/python',Path(directory)/'repo',{})

    def test_worker_commit_and_contract_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            module = self.load(directory)
            contract = subprocess.check_output(['git', 'show', module.WORKER_SOURCE + ':docs/josev2b-source-contract.json'], cwd=ROOT)
            self.assertEqual(hashlib.sha256(contract).hexdigest(), module.CONTRACT_SHA)
            self.assertEqual(json.loads(contract)['runtime']['python_minor'], '3.13')
            self.assertEqual(json.loads(contract)['runtime']['tokenizers'], '0.22.2')
            self.assertEqual(json.loads(contract)['environment_lock']['packages'],57)
            for wheel in module.WHEELS.values():
                self.assertIn('-cp313-cp313-manylinux_2_28_x86_64.whl', wheel['filename'])
                self.assertEqual(wheel['url'].split('#sha256=')[1], wheel['sha256'])


if __name__ == '__main__':
    unittest.main()
