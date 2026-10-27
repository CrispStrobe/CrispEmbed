#!/usr/bin/env python3
"""Synthetic wheels and mocked subprocesses only; no packages/network/GPU."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gpu_install',ROOT/'tools/ci-heavy/josev2b_gpu_install.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
DATA=b'Metadata-Version: 2.4\nName: demo\nVersion: 1\nRequires-Python: >=3.10\nRequires-Dist: dep>=2\n\nDescription.\n'


def fixture():
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        z.writestr('demo-1.dist-info/METADATA',DATA)
        z.writestr('demo/_vendor/other-1.dist-info/METADATA',DATA.replace(b'Name: demo',b'Name: other'))
        z.writestr('demo-1.dist-info/RECORD','demo-1.dist-info/METADATA,,\ndemo/_vendor/other-1.dist-info/METADATA,,\n')
    payload=stream.getvalue()
    return payload,dict(name='demo',version='1',filename='demo-1-py3-none-any.whl',
        url='https://files.pythonhosted.org/packages/a/demo-1-py3-none-any.whl',
        bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
        core_metadata_bytes=len(DATA),core_metadata_sha256=hashlib.sha256(DATA).hexdigest(),
        requires_python='>=3.10',requires_dist=['dep>=2'])


class Controls(unittest.TestCase):
    def execute(self,*,sidecar=DATA,bad_hash=False,pip_failure=False):
        payload,row=fixture()
        if bad_hash:row['sha256']='0'*64
        events=[];result={}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';scratch=root/'scratch';out.mkdir();scratch.mkdir()
            def transfer(r,target,*,sidecar:bool,**kw):
                events.append('sidecar' if sidecar else 'wheel')
                data=self.sidecar if sidecar else payload;target.write_bytes(data)
                return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            def run(args,**kwargs):
                events.append('install')
                for flag in ('--no-deps','--no-index','--require-hashes','--no-cache-dir'):self.assertIn(flag,args)
                requirement=Path(args[-1]).read_text();self.assertTrue(requirement.startswith('file:///'))
                self.assertIn(' --hash=sha256:'+row['sha256'],requirement)
                self.assertNotIn('https://',requirement)
                self.assertTrue(result['audited']);self.assertTrue(result['archive']['owner_equals_sidecar'])
                if pip_failure:raise subprocess.CalledProcessError(1,args)
            def owner(r,prefix):events.append('installed-owner');return dict(owner_metadata_sha256=r['core_metadata_sha256'])
            self.sidecar=sidecar
            with patch.object(m.audit,'transfer',side_effect=transfer),patch.object(m.subprocess,'run',side_effect=run),patch.object(m,'installed_owner',side_effect=owner):
                error=None
                try:m.one(row,out,scratch,root/'venv',time.monotonic()+60,dict(max_metadata_bytes=1048576,max_metadata_bytes_per_wheel=8388608),result,lambda:None)
                except Exception as exc:error=exc
            self.assertEqual(list(scratch.iterdir()),[])
            self.assertFalse(any(p.suffix=='.whl' for p in out.rglob('*')))
            return events,result,error

    def test_audited_archive_before_local_install_and_installed_owner(self):
        events,result,error=self.execute()
        self.assertIsNone(error);self.assertEqual(events,['sidecar','wheel','install','installed-owner'])
        self.assertTrue(result['audited']);self.assertTrue(result['installed'])
        self.assertEqual(result['archive']['legacy_selector_candidate_count'],2)

    def test_sidecar_byte_difference_cannot_install(self):
        events,result,error=self.execute(sidecar=DATA.replace(b'Description.',b'Other.'))
        self.assertIsInstance(error,ValueError);self.assertEqual(events,['sidecar','wheel'])
        self.assertFalse(result['audited']);self.assertFalse(result['installed'])
        self.assertTrue(all(result['archive']['dependency_fields_exact'].values()))

    def test_payload_rehashed_immediately_before_install(self):
        events,result,error=self.execute(bad_hash=True)
        self.assertIsInstance(error,ValueError);self.assertIn('changed before installation',str(error))
        self.assertEqual(events,['sidecar','wheel']);self.assertFalse(result['installed'])

    def test_install_failure_retains_audit_and_never_checks_installed_owner(self):
        events,result,error=self.execute(pip_failure=True)
        self.assertIsInstance(error,subprocess.CalledProcessError)
        self.assertEqual(events,['sidecar','wheel','install']);self.assertTrue(result['audited']);self.assertFalse(result['installed'])

    def test_gpu_proof_rejects_unsupported_before_any_payload(self):
        for row in (['Tesla P100','6.0','16384','15000','580.178.04'],['T4','7.5','15360','15000','570.0.0'],['T4','7.5','15360','10000','580.178.04']):
            with self.assertRaises(ValueError):m.hardware([row])
        self.assertEqual(m.hardware([[' T4 ','7.5','14912','14000','580.178.04']])[0][0],'T4')
        args=type('Args',(),{})()
        with patch.dict(m.os.environ,{},clear=True),patch.object(m.audit,'transfer',side_effect=AssertionError('network forbidden')):
            with self.assertRaisesRegex(ValueError,'Hosted original GPU task only'):m.guard(args)

    def test_fixed_model_lock_and_cpu_audit_unmodified(self):
        cfg=(ROOT/'docs/josev2b-source-contract.json').read_bytes()
        self.assertEqual(hashlib.sha256(cfg).hexdigest(),m.CONTRACT_SHA)
        lock=json.loads((ROOT/'docs/josev2b-environment-lock.json').read_text())
        self.assertEqual(len(lock['packages']),57);self.assertEqual(sum(r['bytes'] for r in lock['packages']),2790174915)
        old=subprocess.check_output(['git','show','e74df24f:tools/ci-heavy/josev2b_wheel_metadata.py'],cwd=ROOT)
        self.assertEqual(old,(ROOT/'tools/ci-heavy/josev2b_wheel_metadata.py').read_bytes())
        text=(ROOT/'tools/ci-heavy/josev2b_gpu_install.py').read_text()
        self.assertLess(text.index('guard(args);save()'),text.index("for row in lock['packages']"))
        self.assertIn('import_runtime=False',text);self.assertNotIn('import torch',text)

if __name__=='__main__':unittest.main()
