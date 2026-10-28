"""Bounded stdlib-only replay admission, byte-adapter and collection controls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/moedern-scifact')]
import contract_native_v1 as contract
import diff_adapter_v1 as adapter
from original_inputs_v1 import expected_tokens,region,locate
from providers_v1 import Redirect
import compare_v1


class Controls(unittest.TestCase):
    def test_all_reviewed_original_pins(self):
        cfg=contract.validate(json.loads(contract.PATH.read_text()))
        for variant in('1.5M','27M','115M'):
            pin=contract.admit(cfg,variant);p=ROOT/pin['public_report']
            self.assertEqual(p.stat().st_size,pin['report_bytes']);self.assertEqual(contract.sha(p),pin['report_sha256'])
            r=json.loads(p.read_text());self.assertTrue(r['completed']);self.assertEqual(r['rows'],5483)
            self.assertEqual(r['source'],pin['source']);self.assertEqual(set(r['arrays']),{'all.f32','selected.f32','scores.npy','data.json','rows.json','rankings.json'})

    def test_pending_and_unversioned_reject(self):
        cfg=json.loads(contract.PATH.read_text())
        for key,value in(('reviewed',False),('version_label',''),('report_sha256',None),('report_bytes',0)):
            bad=copy.deepcopy(cfg);bad['originals']['115M'][key]=value
            with self.assertRaisesRegex(ValueError,'pending'):contract.admit(bad,'115M')
        with self.assertRaises(ValueError):contract.admit(cfg,'latest')

    def test_gates_and_complete_closure_cannot_weaken(self):
        cfg=json.loads(contract.PATH.read_text())
        for key in cfg['gates']:
            bad=copy.deepcopy(cfg);bad['gates'][key]=0
            with self.assertRaises(ValueError):contract.validate(bad)
        for name in('src/moedern_text_api.cpp','tools/quantize.cpp','tools/moedern-colbert/public_clients.py'):
            bad=copy.deepcopy(cfg);bad['protected_sha256'].pop(name)
            with self.assertRaises(ValueError):contract.validate(bad)
        bad=copy.deepcopy(cfg);bad['owned_sha256'].pop('tools/moedern-scifact/capture_v1.py')
        with self.assertRaises(ValueError):contract.validate(bad)

    def test_derived_expansion_is_not_scoring_mask(self):
        q=dict(role='query',ids=[2,30000,4,4],supplied_mask=[1,1,0,0],effective_mask=[1]*4,scoring_mask=[1]*4)
        self.assertEqual(expected_tokens(q)['expansion_positions'],[0,0,1,1])
        d=dict(q,role='document',effective_mask=q['supplied_mask'],scoring_mask=[0,1,0,0])
        self.assertEqual(expected_tokens(d)['expansion_positions'],[])

    def test_original_unique_hash_not_basename(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);(root/'a').mkdir();(root/'b').mkdir()
            (root/'a/original.json').write_text('auth');(root/'b/original.json').write_text('fake')
            pin=dict(report_bytes=4,report_sha256=contract.sha(root/'a/original.json'))
            self.assertEqual(locate(root,pin),root/'a/original.json')
            (root/'b/original.json').write_text('auth')
            with self.assertRaises(ValueError):locate(root,pin)

    def test_span_hash_truncation_and_offset(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t)/'data';p.write_bytes(b'firstactual-last')
            self.assertEqual(region(p,5,6),hashlib.sha256(b'actual').hexdigest())
            with self.assertRaisesRegex(ValueError,'truncated'):region(p,5,100)

    def test_lossless_adapter_payload_and_axes(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);p=root/'stream.f32'
            payload=struct.pack('<128f',-0.,1.25,*([0.]*126));p.write_bytes(b'x'*512+payload+b'z'*512)
            rows={name:dict(path=p,offset=512,bytes=512,shape=[1,128])for name in('normalized_all','normalized_selected')}
            result=adapter.write(root/'probe.gguf',rows)
            spec=importlib.util.spec_from_file_location('old_reader',ROOT/'tools/moedern-native-v2/diff_adapter.py')
            old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
            seen=old.read_gguf(root/'probe.gguf')
            self.assertEqual(set(seen),set(rows))
            for name,r in seen.items():
                self.assertEqual(r['shape'],[1,128]);self.assertEqual(r['sha256'],hashlib.sha256(payload).hexdigest())
                self.assertEqual(result['tensors'][name]['span_sha256'],r['sha256'])
            self.assertEqual((root/'probe.gguf').read_bytes().count(payload),2)

    def test_adapter_rejects_wrong_semantics(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t)/'stream';p.write_bytes(bytes(1024));valid=dict(path=p,offset=0,bytes=512,shape=[1,128])
            for change in(dict(shape=[128,1]),dict(offset=1),dict(bytes=True),dict(shape=[True,128]),dict(bytes=1536),dict(shape=[adapter.MAX_ROWS+1,128])):
                with self.assertRaises(ValueError):adapter.bounds(dict(valid,**change))
            link=Path(t)/'link';link.symlink_to(p)
            with self.assertRaises(ValueError):adapter.bounds(dict(valid,path=link))

    def test_eight_partitions_cover_every_case_once(self):
        ranges=compare_v1.partition(5483)
        self.assertEqual(len(ranges),8);self.assertEqual([i for a,b in ranges for i in range(a,b)],list(range(5483)))
        self.assertLessEqual(max(b-a for a,b in ranges),686)
        with self.assertRaises(ValueError):compare_v1.partition(5483,7)

    def test_ref_collection_numeric_failure_retained(self):
        metric=dict(shape=[1,128],elements=128,max_absolute=1.,mean_absolute=.01,mean_error=.01,rms=.1,min_row_cosine=0.,
                    original_l2=1.,native_l2=1.,original_min=0.,original_max=1.,original_abs_min=0.,original_abs_max=1.,
                    native_min=0.,native_max=1.,native_abs_min=0.,native_abs_max=1.,passed=False)
        independent={n:copy.deepcopy(metric)for n in('normalized_all','normalized_selected')}
        rows=[dict(name=n,ggml_shape=[128,1],row_axis=0,found=True,**{k:v for k,v in metric.items()if k not in('shape',)})for n in independent]
        report=dict(schema='crispembed-moedern-scifact-ref-v1',max_absolute_budget=.002,minimum_row_cosine=.99999,status='numeric_rejection',stages=rows)
        with tempfile.TemporaryDirectory()as t:
            p=Path(t)/'ref.json';p.write_text(json.dumps(report));r=adapter.collect(1,p,independent)
            self.assertEqual(r['exit'],1);self.assertTrue(r['all_metrics_agree']);self.assertFalse(independent['normalized_all']['passed'])
            with self.assertRaises(ValueError):adapter.collect(2,p,independent)
            report['stages'].pop();p.write_text(json.dumps(report))
            with self.assertRaises(ValueError):adapter.collect(1,p,independent)

    def test_auth_removed_on_signed_redirect(self):
        import urllib.request
        r=urllib.request.Request('https://api.github.com/input',headers={'Authorization':'Bearer placeholder'})
        out=Redirect().redirect_request(r,None,302,'moved',{},'https://elsewhere.example/file')
        self.assertIsNone(out.get_header('Authorization'))

    def test_kernel_placeholders_and_lock_dependency_closure(self):
        source=(ROOT/'tools/kaggle/moedern-scifact-native-q4k/kernel.py').read_text()
        self.assertIn("VARIANT='@VARIANT@'",source);self.assertIn('timeout=9*3600',source)
        self.assertLess(source.index("run('source-closure'"),source.index("run('install-locked-tools'"))
        lock=(ROOT/'tools/kaggle/moedern-scifact-native-q4k/requirements.lock').read_text().splitlines()
        self.assertEqual({line.split(' @ ')[0]for line in lock},{'pip','numpy','gguf','PyYAML','tqdm'})
        for line in lock:self.assertRegex(line,r'\.whl --hash=sha256:[0-9a-f]{64}$')
        metadata=json.loads((ROOT/'tools/kaggle/moedern-scifact-native-q4k/kernel-metadata.json').read_text())
        self.assertEqual(metadata['enable_gpu'],'false');self.assertEqual(metadata['kernel_sources'],[])


if __name__=='__main__':unittest.main()
