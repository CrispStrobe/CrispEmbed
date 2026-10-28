"""No payload/framework fixtures: provider boundaries and unchanged math controls."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/moedern-scifact')]
import contract_gha_v1 as C
spec=importlib.util.spec_from_file_location('gha_adapter',ROOT/'tools/ci-heavy/moedern_scifact_gha_v1.py')
G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)


def fake_environment():
    return dict(GITHUB_ACTIONS='true',RUNNER_ENVIRONMENT='github-hosted',RUNNER_OS='Linux',RUNNER_ARCH='X64',
        GITHUB_REPOSITORY='CrispStrobe/CrispEmbed',GITHUB_EVENT_NAME='workflow_dispatch',GITHUB_RUN_ID='123',
        GITHUB_RUN_ATTEMPT='1',GITHUB_SHA='a'*40,
        GITHUB_WORKFLOW_REF='CrispStrobe/CrispEmbed/.github/workflows/moedern-scifact-gha-adapter-v1.yml@refs/heads/test',
        GH_TOKEN='synthetic-artifact-read',
        KAGGLE_API_TOKEN='synthetic-token',KAGGLE_ACCOUNT='synthetic-owner')


class AdapterControls(unittest.TestCase):
    def test_genuine_service_required_without_kaggle_spoof_or_missing_secrets(self):
        G.environment(fake_environment())
        for key,bad in [('RUNNER_ENVIRONMENT','self-hosted'),('GITHUB_ACTIONS','false'),('GITHUB_SHA','main'),
                        ('GITHUB_REPOSITORY','other/project'),('KAGGLE_API_TOKEN',''),('KAGGLE_ACCOUNT','')]:
            env=fake_environment();env[key]=bad
            with self.assertRaises(ValueError):G.environment(env)
        env=dict(KAGGLE_KERNEL_RUN_TYPE='Batch')
        with self.assertRaises(ValueError):G.environment(env)

    def test_resource_floor_not_reduced_to_fit_standard_runner(self):
        G.limits(14*2**30,24*2**30,2)
        for ram,disk,cpu in [(14*2**30-1,24*2**30,2),(14*2**30,24*2**30-1,2),(14*2**30,24*2**30,1)]:
            with self.assertRaises(ValueError):G.limits(ram,disk,cpu)

    def test_all_inference_functions_ast_equal_corrected_worker(self):
        original=ast.parse((ROOT/'tools/moedern-scifact/worker_v1.py').read_text())
        updated=ast.parse((ROOT/'tools/moedern-scifact/worker_gha_v1.py').read_text())
        defs=lambda tree:{r.name:ast.dump(r,include_attributes=False)for r in tree.body if isinstance(r,ast.FunctionDef)}
        a,b=defs(original),defs(updated)
        self.assertEqual(set(a),set(b))
        self.assertEqual({k:v for k,v in a.items()if k!='resources'}, {k:v for k,v in b.items()if k!='resources'})
        oldbody=ast.dump(next(r for r in original.body if isinstance(r,ast.If)),include_attributes=False)
        newbody=ast.dump(next(r for r in updated.body if isinstance(r,ast.If)),include_attributes=False)
        self.assertEqual(oldbody,newbody)

    def test_frozen_original_worker_contract_and_helpers_byte_identical(self):
        cfg=C.declaration()
        for name in ['tools/moedern-scifact/worker_v1.py','tools/moedern-scifact/contract_native_v1.py',
                     'docs/moedern-scifact-native-q4k-v1.json','tools/moedern-scifact/original_inputs_v1.py',
                     'tools/moedern-scifact/compare_v1.py','tools/moedern-scifact/authored_v1.py']:
            old=subprocess.check_output(['git','show',C.BASE+':'+name],cwd=ROOT)
            self.assertEqual(old,(ROOT/name).read_bytes())
            self.assertEqual(hashlib.sha256(old).hexdigest(),cfg['protected_sha256'][name])

    def test_get_status_and_every_paginated_list_use_exact_version(self):
        calls=[];pin=dict(kernel_slug='original',version_label='v1',source='a'*40,
                        submitted_script_sha256=hashlib.sha256(('worker '+'a'*40).encode()).hexdigest())
        class Request:
            def __str__(self):return json.dumps({'versionLabel':self.version_label})
        class Client:
            def get_kernel(self,q):
                calls.append(q);return json.dumps(dict(blob=dict(source='worker '+'a'*40)))
            def get_kernel_session_status(self,q):calls.append(q);return types.SimpleNamespace(status='COMPLETE')
            def list_kernel_session_output(self,q):
                calls.append(q);later=hasattr(q,'page_token')
                return types.SimpleNamespace(files=[types.SimpleNamespace(file_name='b'if later else'a',url='https://invalid/')],next_page_token=None if later else'next')
        files,proof=G.indexed_outputs(Client(),Request,Request,Request,'synthetic-owner',pin)
        self.assertEqual(set(files),{'a','b'});self.assertEqual(proof['pages'],2)
        self.assertEqual(len(calls),4)
        self.assertTrue(all(q.version_label=='v1'and q.kernel_slug=='original'for q in calls))
        self.assertNotIn('synthetic-owner',json.dumps(proof))

    def test_duplicate_members_and_wrong_submitted_source_reject(self):
        pin=dict(kernel_slug='original',version_label='v1',source='a'*40,
                 submitted_script_sha256=hashlib.sha256(('worker '+'a'*40).encode()).hexdigest())
        class Request:
            def __str__(self):return json.dumps({'versionLabel':self.version_label})
        class Client:
            def get_kernel(self,q):return json.dumps(dict(blob=dict(source='worker '+'a'*40)))
            def get_kernel_session_status(self,q):return types.SimpleNamespace(status='COMPLETE')
            def list_kernel_session_output(self,q):return types.SimpleNamespace(files=[types.SimpleNamespace(file_name='same',url='https://invalid/')]*2,next_page_token=None)
        with self.assertRaisesRegex(ValueError,'Duplicate'):G.indexed_outputs(Client(),Request,Request,Request,'private',pin)
        pin['submitted_script_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'source differs'):G.indexed_outputs(Client(),Request,Request,Request,'private',pin)

    def test_bounded_stream_fullhash_and_extent_without_network(self):
        import io
        class Response(io.BytesIO):headers={'Content-Length':'8'}
        with tempfile.TemporaryDirectory()as folder:
            pin=dict(bytes=8,sha256=hashlib.sha256(b'12345678').hexdigest())
            with patch.object(G.urllib.request,'urlopen',return_value=Response(b'12345678')):
                self.assertEqual(G.stream('https://invalid/',Path(folder)/'good',pin),pin)
            with patch.object(G.urllib.request,'urlopen',return_value=Response(b'1234567!')):
                with self.assertRaisesRegex(ValueError,'SHA'):G.stream('https://invalid/',Path(folder)/'bad',pin)
            with patch.object(G.urllib.request,'urlopen',return_value=Response(b'123456789')):
                with self.assertRaisesRegex(ValueError,'transfer rejected'):G.stream('https://invalid/',Path(folder)/'extra',pin)

    def test_entrypoint_imports_before_any_optional_packages_or_payloads(self):
        process=subprocess.run([sys.executable,'-I',str(ROOT/'tools/moedern-scifact/worker_gha_v1.py'),'--help'],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertIn('--variant',process.stdout)
        self.assertFalse(any(k in sys.modules for k in ['numpy','torch','kaggle']))

    def test_standalone_workflow_gates_before_packages_fetch_and_models(self):
        text=(ROOT/'.github/workflows/moedern-scifact-gha-adapter-v1.yml').read_text()
        self.assertNotIn('workflow_call:',text)
        self.assertLess(text.index('gha_v1.py admit'),text.index('pip install'))
        self.assertLess(text.index('gha_v1.py fetch'),text.index('gha_v1.py run'))
        self.assertIn('retention-days: 90',text)
        self.assertIn('timeout-minutes: 360',text)
        self.assertIn('KAGGLE_API_TOKEN: ${{ secrets.KAGGLE_TOKEN }}',text)
        self.assertEqual(C.declaration()['enabled_variants'],['1.5M'])


if __name__=='__main__':unittest.main()
