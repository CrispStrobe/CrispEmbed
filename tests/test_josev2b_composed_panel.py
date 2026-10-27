"""Frozen source coverage and fail-closed mapping controls; stdlib only."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import importlib.util
import os
import subprocess
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import josev2b_composed_panel as panel

class Coverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((ROOT/'tests/results/josev2b/original-reference.json').read_text())

    def native(self, case):
        result={};counts={}
        for r in self.report['arrays']:
            if r['case']!=case or r['stage'] in panel.UNMAPPED: continue
            n=counts.get(r['stage'],0);counts[r['stage']]=n+1
            result[(r['stage'],n)]={'shape':r['shape']}
        return result

    def test_frozen_251_declares_exact_227_mapped_and_24_unmapped(self):
        counts=[];unmapped=[]
        for case in self.report['cases']:
            a,b=panel.coverage(self.report,case['id'],self.native(case['id']))
            counts.append(len(a));unmapped+=b
        self.assertEqual(counts,[68,59,50,50]);self.assertEqual(sum(counts),227)
        self.assertEqual(len(unmapped),24)
        self.assertEqual(sum(r['dtype']=='int64' for r in unmapped),12)
        self.assertEqual(sum(r['dtype']=='float32' for r in unmapped),12)

    def test_missing_unreferenced_and_wrong_shape_never_pass(self):
        good=self.native('mixed-de')
        mutations=[]
        missing=copy.deepcopy(good);missing.pop(('head/field/0',0));mutations.append(missing)
        extra=copy.deepcopy(good);extra[('head/fabricated',0)]={'shape':[512]};mutations.append(extra)
        wrong=copy.deepcopy(good);wrong[('backbone/final_norm',0)]['shape']=[319,2048];mutations.append(wrong)
        duplicate=copy.deepcopy(good);duplicate[('backbone/final_norm',1)]={'shape':[1,319,2048]};mutations.append(duplicate)
        for actual in mutations:
            with self.assertRaises(ValueError):panel.coverage(self.report,'mixed-de',actual)

    def test_repeated_head_inputs_have_separate_original_targets(self):
        mapped,_=panel.coverage(self.report,'mixed-de',self.native('mixed-de'))
        rows=[r for r in mapped if r['stage']=='head/option_lexical_projection/input/0']
        self.assertEqual([r['occurrence'] for r in rows],[0,1,2])
        self.assertEqual([r['shape'] for r in rows],[[2,2048],[2,2048],[3,2048]])
        self.assertEqual(len({r['sha256'] for r in rows}),3)

class DependencyBarrier(unittest.TestCase):
    def test_pending_review_rejects_before_api_or_payload(self):
        spec=importlib.util.spec_from_file_location('composed_hosted',ROOT/'tools/ci-heavy/josev2b_composed.py')
        hosted=importlib.util.module_from_spec(spec);spec.loader.exec_module(hosted)
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        with mock.patch.object(hosted,'verify',return_value={}), mock.patch.object(hosted,'api') as remote, \
             mock.patch.object(hosted.json,'loads',return_value={'schema':'crispembed-josev-composed-dependencies-v1','reviewed':False}), \
             mock.patch.dict(os.environ,{'GITHUB_ACTIONS':'true','GITHUB_SHA':head}):
            with self.assertRaisesRegex(ValueError,'dependency review pending'):hosted.policy()
            remote.assert_not_called()

class ArtifactMembers(unittest.TestCase):
    def test_real_selective_extractor_uses_explicit_producer_member(self):
        import io,zipfile,tempfile,hashlib,types
        spec=importlib.util.spec_from_file_location('composed_extract',ROOT/'tools/ci-heavy/josev2b_composed.py')
        hosted=importlib.util.module_from_spec(spec);spec.loader.exec_module(hosted)
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            z.writestr('producer/component.gguf',b'owned-test-only-bytes')
            z.writestr('other/probe',b'never-extract-this-test-only-binary')
        payload=stream.getvalue();digest=hashlib.sha256(payload).hexdigest()
        pin=dict(run=1,source='a'*40,id=2,name='test-owned',bytes=len(payload),sha256=digest)
        run=dict(head_sha='a'*40,event='workflow_dispatch',run_attempt=1,status='completed',conclusion='success',path='.github/workflows/ovisocr2-metadata.yml')
        artifact=dict(workflow_run={'id':1,'head_sha':'a'*40},name='test-owned',size_in_bytes=len(payload),expired=False,digest='sha256:'+digest,id=2,expires_at='test-owned')
        def download(*args,**kwargs):
            kwargs['stdout'].write(payload)
            return types.SimpleNamespace(returncode=0)
        for member,accepted in [('component.gguf',False),('producer/component.gguf',True)]:
            with tempfile.TemporaryDirectory() as temp, mock.patch.object(hosted,'api',side_effect=[run,artifact]), \
                 mock.patch.object(hosted.subprocess,'run',side_effect=download):
                if not accepted:
                    with self.assertRaisesRegex(ValueError,'declared artifact member absent'):
                        hosted.fetch(pin,Path(temp),{member})
                else:
                    out,_=hosted.fetch(pin,Path(temp),{member})
                    self.assertEqual((out/member).read_bytes(),b'owned-test-only-bytes')
                    self.assertFalse((out/'other/probe').exists())

if __name__=='__main__':unittest.main()
