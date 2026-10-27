"""Source/fixture controls only. No packages, model payloads, or builds."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('packed_worker',ROOT/'tools/ci-heavy/josev2b_packed.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


class Controls(unittest.TestCase):
    def test_runtime_exact_new_source_and_reused_original_helpers(self):
        cfg=w.load(ROOT/w.PROTOCOL);self.assertEqual(w.sha(ROOT/w.PROTOCOL),w.PROTOCOL_SHA)
        self.assertEqual(cfg['pack_profile_sha256'],w.packer.PROFILE_SHA)
        self.assertEqual(cfg['runtime_source'],'0f0b1e5955a8791855fdba04d018b536b6a879f8')
        for path,obj in cfg['runtime_objects'].items():
            self.assertEqual(w.git('rev-parse',cfg['runtime_source']+':'+path),obj)
        for path,digest in cfg['reference_helpers'].items():
            self.assertEqual(hashlib.sha256(subprocess.check_output(['git','show',cfg['reference_source']+':'+path],cwd=ROOT)).hexdigest(),digest)
        self.assertEqual(cfg['expected'],dict(cases=4,compared_stages=227,unmapped=24,max_absolute=.002,min_row_cosine=.99999,logit_absolute=.002,probability_absolute=.0001,ordinal_absolute=.001,complete_typed_responses_exact=True))

    def test_composed_success_pin_has_exact_service_identity_and_report_hash(self):
        pin=w.packer.profile()['composed_review'];self.assertTrue(pin['reviewed'])
        self.assertEqual(pin['run'],40449265616);self.assertEqual(pin['source'],'ed23f1aeb5f448b80a89b567592986bffb951de9')
        self.assertEqual(pin['receipt_sha256'],'45689505bcfd6de6e32a3c353ac4a502f0c82826227b2100788e9ac9b15d0fd7')
        self.assertEqual(pin['review_artifact']['id'],12710133899)

    def test_explicit_tokenizer_lca_and_all_source_notices(self):
        cfg=w.load(ROOT/w.PROTOCOL)
        self.assertEqual(cfg['component_members']['tokenizer'],'josev-tokenizer-schema/josev2b-tokenizer.gguf')
        self.assertEqual(len(cfg['notices']),8);self.assertIn('LICENSE',cfg['notices']);self.assertIn('README.md',cfg['notices'])
        self.assertEqual(cfg['gates']['CRISPEMBED_JOSEV_PACKED'],'1')
        self.assertEqual(len(cfg['gates']),7)

    def test_failed_source_service_rejects_before_any_archive_download(self):
        pin=dict(run=1,source='a'*40,id=2,name='fixed',bytes=123,sha256='b'*64)
        run=dict(head_sha='a'*40,path='.github/workflows/ovisocr2-metadata.yml',event='workflow_dispatch',run_attempt=1,status='completed',conclusion='failure')
        with mock.patch.object(w,'api',return_value=run),mock.patch.object(w.subprocess,'run')as network:
            with self.assertRaisesRegex(ValueError,'producer'):w.fetch(pin,Path('unused'))
            network.assert_not_called()
        run['conclusion']='success'
        artifact=dict(workflow_run=dict(id=1,head_sha='a'*40),name='wrong',size_in_bytes=123,digest='sha256:'+'b'*64,expired=False)
        with mock.patch.object(w,'api',side_effect=[run,artifact]),mock.patch.object(w.subprocess,'run')as network:
            with self.assertRaisesRegex(ValueError,'identity'):w.fetch(pin,Path('unused'))
            network.assert_not_called()

    def test_complete_archive_manifest_rejects_missing_or_extra_members(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'review').mkdir();(root/'original').mkdir()
            w.write(root/'review/artifact.json',dict(completed=True,arrays=251,reference_sha256='51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090',files={}))
            (root/'original/extra').write_text('unexpected')
            with self.assertRaisesRegex(ValueError,'coverage'):w.validate_head_archive(root)

    def test_original_control_and_packing_order_preserves_no_injected_inputs(self):
        text=(ROOT/'tools/ci-heavy/josev2b_packed.py').read_text();tree=ast.parse(text)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run');body=ast.get_source_segment(text,fn)
        self.assertLess(body.index('packer.reviewed_composition'),body.index('runtime_checkout'))
        self.assertLess(body.index('panel.ref_controls'),body.index("for kind in ('tokenizer','backbone','head')"))
        self.assertLess(body.index('packer.pack('),body.index("for label,variant in [('plain'"))
        self.assertIn("panel.compare(reference,freeze,original,outputs",body)
        self.assertNotIn('np.load',body);self.assertNotIn('hidden=',body)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)

    def test_workflow_keeps_heavy_work_hosted_and_artifacts_separate(self):
        text=(ROOT/'.github/workflows/josev2b-packed.yml').read_text()
        self.assertIn('timeout-minutes: 60',text);self.assertIn('numpy==2.2.6',text)
        self.assertIn('submodules: false',text);self.assertIn('fetch-depth: 0',text)
        self.assertIn('josev-packed-f32-asset-',text);self.assertIn('josev-packed-traces-',text)
        self.assertIn('.gguf.partial',text);self.assertIn('retention-days: 90',text)
        self.assertNotIn('pip install torch',text);self.assertNotIn('transformers==',text)


if __name__=='__main__':unittest.main()
