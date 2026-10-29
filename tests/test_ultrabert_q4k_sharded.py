#!/usr/bin/env python3
"""Tiny stdlib checks for durable public-vector checkpoint admission."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import types
import os
import contextlib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sharded', ROOT / 'tools/portfolio/ultrabert_q4k_sharded.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.binding = dict(model='a'*64, source='b'*40, inputs=['first', 'second'])
        self.rows = [dict(id='d:1', input_sha256='c'*64), dict(id='d:2', input_sha256='d'*64)]
        self.raw = struct.pack('<768f', *[float(i)/768 for i in range(768)])
        self.norm = struct.pack('<768f', *[float(i)/1536 for i in range(768)])

    def append(self):
        self.assertEqual(m.resume_prefix(self.out, self.binding, self.rows), 0)
        m.commit_row(self.out, dict(index=0, id='d:1', input_sha256='c'*64), self.raw, self.norm)

    def test_original_striding_complete_disjoint_5483(self):
        shards = [m.indices(i) for i in range(8)]
        self.assertEqual(sorted(x for shard in shards for x in shard), list(range(5483)))
        self.assertEqual([len(x) for x in shards], [686, 686, 686, 685, 685, 685, 685, 685])
        for bad in [-1, 8, True]:
            with self.assertRaises(ValueError): m.indices(bad)

    def test_committed_prefix_and_uncommitted_tails(self):
        self.append()
        # Crash after one next-vector write and before its journal commit.
        with (self.out/'raw.f32').open('ab') as f: f.write(self.norm)
        self.assertEqual(m.resume_prefix(self.out, self.binding, self.rows), 1)
        self.assertEqual((self.out/'raw.f32').read_bytes(), self.raw)
        self.assertEqual((self.out/'normalized.f32').read_bytes(), self.norm)
        self.assertEqual(json.loads((self.out/'rows.jsonl').read_text())['raw_sha256'], hashlib.sha256(self.raw).hexdigest())

    def test_changed_committed_vector_rejected(self):
        self.append(); (self.out/'raw.f32').write_bytes(self.norm)
        with self.assertRaisesRegex(ValueError, 'committed vector'): m.resume_prefix(self.out, self.binding, self.rows)

    def test_changed_model_source_or_input_rejected(self):
        self.append()
        with self.assertRaisesRegex(ValueError, 'binding'): m.resume_prefix(self.out, dict(self.binding, model='e'*64), self.rows)
        changed = [dict(self.rows[0], input_sha256='f'*64), self.rows[1]]
        with self.assertRaisesRegex(ValueError, 'order/input'): m.resume_prefix(self.out, self.binding, changed)

    def test_torn_extra_out_of_order_and_truncated_rejected(self):
        self.append(); journal=self.out/'rows.jsonl'; saved=journal.read_bytes()
        journal.write_bytes(saved+b'{')
        with self.assertRaisesRegex(ValueError, 'torn'): m.resume_prefix(self.out, self.binding, self.rows)
        journal.write_bytes(saved+saved+saved)
        with self.assertRaisesRegex(ValueError, 'extra'): m.resume_prefix(self.out, self.binding, self.rows)
        record=json.loads(saved); record['index']=1; journal.write_text(json.dumps(record)+'\n')
        with self.assertRaisesRegex(ValueError, 'order/input'): m.resume_prefix(self.out, self.binding, self.rows)
        journal.write_bytes(saved); (self.out/'normalized.f32').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'committed vector'): m.resume_prefix(self.out, self.binding, self.rows)

    def test_unbound_stream_and_extra_manifest_file_rejected(self):
        (self.out/'raw.f32').write_bytes(self.raw)
        with self.assertRaisesRegex(ValueError, 'unbound'): m.resume_prefix(self.out, self.binding, self.rows)
        m.write(self.out/'manifest.json', m.manifest(self.out)); m.validate_manifest(self.out)
        (self.out/'unexpected').write_bytes(b'not an authenticated member')
        with self.assertRaisesRegex(ValueError, 'manifest'): m.validate_manifest(self.out)

    def test_pending_dependencies_reject_before_payload_imports(self):
        self.assertIsNone(m.read(m.PROTOCOL)['reviewed_dependencies']['freeze'])
        with self.assertRaisesRegex(ValueError, 'pending reviewed freeze'): m.approval('freeze')
        p=m.read(m.PROTOCOL)
        self.assertFalse(p['activation']['reviewed'])
        self.assertEqual(p['activation']['shards'], [0])
        self.assertEqual(p['activation']['recipe'], 'q4k-last-four-ffn')
        self.assertFalse(p['q8_pending']['activation_authorized'])
        self.assertEqual(p['q8_pending']['required_standard_file_type'], 7)

    def test_owned_closure_matches_and_old_recipe_scope(self):
        p=m.read(m.PROTOCOL)
        for path, digest in p['owned_sha256'].items(): self.assertEqual(m.sha(ROOT/path), digest, path)
        self.assertEqual(p['runtime_source'], m.RUNTIME)
        self.assertEqual(p['retained_v2']['version_label'], 'v2')
        self.assertEqual(len(p['retained_v2']['indexed_members']), 72)
        worker=(ROOT/'tools/portfolio/ultrabert_q4k_sharded.py').read_text()
        self.assertNotIn('old.native(',worker)
        self.assertNotIn('old.run(',worker)
        self.assertIn("lib.crispembed_encode_tokens_raw(ctx",worker)
        self.assertIn("lib.crispembed_encode(ctx",worker)
        self.assertIn("old.recipes()[FIRST_RECIPE]",worker)
        self.assertIn("types == {'F32': 206, 'Q4_K': 12}",worker)

    def test_manifest_mutation_and_exact_old_metrics_gate(self):
        (self.out/'one').write_bytes(b'one');m.write(self.out/'manifest.json',m.manifest(self.out));m.validate_manifest(self.out)
        (self.out/'one').write_bytes(b'two')
        with self.assertRaises(ValueError):m.validate_manifest(self.out)
        metrics={k:{f:1. for f in ['max_absolute','mean_absolute','rms','min_row_cosine','original_l2','native_l2']}for k in ['raw','normalized']}
        m.metric_agreement(metrics,metrics)
        wrong=json.loads(json.dumps(metrics));wrong['raw']['rms']=1.001
        with self.assertRaisesRegex(ValueError,'metric disagreement'):m.metric_agreement(wrong,metrics)


class ExactServiceTests(unittest.TestCase):
    def test_versioned_get_status_and_every_page(self):
        calls=[]
        class Request:
            def __str__(self):return json.dumps(dict(versionLabel=self.version_label))
        class Body:
            def __str__(self):return json.dumps(dict(blob=dict(source=m.RUNTIME)))
        class Service:
            def get_kernel(self,q):calls.append(('get',q.version_label));return Body()
            def get_kernel_session_status(self,q):calls.append(('status',q.version_label));return types.SimpleNamespace(status='ERROR')
            def list_kernel_session_output(self,q):
                calls.append(('list',q.version_label,getattr(q,'page_token',None)))
                if not getattr(q,'page_token',None):return types.SimpleNamespace(files=[types.SimpleNamespace(file_name='first',url='https://fixture.invalid/one')],next_page_token='second')
                return types.SimpleNamespace(files=[types.SimpleNamespace(file_name='second',url='https://fixture.invalid/two')],next_page_token=None)
        service=Service()
        class API:
            def authenticate(self):pass
            def build_kaggle_client(self):return contextlib.nullcontext(types.SimpleNamespace(kernels=types.SimpleNamespace(kernels_api_client=service)))
        module=types.ModuleType('kaggle.api.kaggle_api_extended');module.KaggleApi=API
        typesmod=types.SimpleNamespace(ApiGetKernelRequest=Request,ApiGetKernelSessionStatusRequest=Request,ApiListKernelSessionOutputRequest=Request)
        pin=dict(kernel_slug='frozen-fixture',version_label='v2',source=m.RUNTIME,submitted_script_sha256=hashlib.sha256(m.RUNTIME.encode()).hexdigest())
        with mock.patch.dict(os.environ,{'KAGGLE_ACCOUNT':'fixture','KAGGLE_API_TOKEN':'test-only'}), mock.patch.dict('sys.modules',{'kaggle.api.kaggle_api_extended':module}), mock.patch.object(m.importlib,'import_module',return_value=typesmod):
            files,proof=m.kaggle_outputs(pin)
            self.assertEqual(set(files),{'first','second'});self.assertEqual(proof['pages'],2)
            self.assertEqual(calls,[('get','v2'),('status','v2'),('list','v2',None),('list','v2','second')])
            with self.assertRaisesRegex(ValueError,'submitted v2 source differs'):
                m.kaggle_outputs(dict(pin,submitted_script_sha256='0'*64))

    def test_frozen_source_not_changed_and_quota_not_overloaded(self):
        import subprocess
        paths=['tools/portfolio/ultrabert_q4k.py','src/ultrabert_profile.h','models/convert-ultrabert-to-gguf.py']
        for path in paths:
            old=subprocess.check_output(['git','show',m.RUNTIME+':'+path],cwd=ROOT)
            self.assertEqual((ROOT/path).read_bytes(),old,path)
        workflow=(ROOT/'.github/workflows/ultrabert-q4k-sharded-v1.yml').read_text()
        self.assertNotIn('matrix:',workflow)
        self.assertIn('if: always()',workflow)
        self.assertIn('timeout-minutes: 360',workflow)
        self.assertIn('LD_LIBRARY_PATH=',workflow)
        self.assertNotIn('ovisocr2-metadata',workflow)


if __name__ == '__main__': unittest.main()
