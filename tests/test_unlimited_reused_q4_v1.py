"""Only stdlib/source controls. Actual Ref and model runs remain hosted."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/unlimited-reused-q4-v1'))
import lane as c
spec=importlib.util.spec_from_file_location('new_last1_replay',ROOT/'tools/unlimited-reused-q4-v1/replay.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
spec=importlib.util.spec_from_file_location('new_last1_fetch',ROOT/'tools/unlimited-reused-q4-v1/fetch.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

class Controls(unittest.TestCase):
    def fixture(self):
        stages={k:dict(shape=[1,1,4],type='f32',bytes=16,sha256=hashlib.sha256(struct.pack('<4f',0.,1.,2.,3.)).hexdigest())for k in ['layer_'+str(i)for i in range(58)]}
        row=dict(ids=[7,8],text='complete\ntext',output_count=2,token_cap_reached=False,eos_observed=False,
            stop_contract='source-derived EOS/page guard; generated IDs exclude EOS',command=dict(stdout_sha256=hashlib.sha256(b'complete\ntext\n').hexdigest()),vision=dict(bytes=5120,sha256='f'*64),stages=stages)
        current={m:copy.deepcopy(row)for m in ('plain','ordinary','observed','repeat')}
        return current,copy.deepcopy(current),dict(ids=row['ids'],text=row['text'])
    def test_complete_f16_control(self):
        a,b,ref=self.fixture();self.assertTrue(r.barrier(a,b,ref)['passed'])
    def test_f16_any_mode_text_ids_cap_stop_or_stage_changes_reject(self):
        for mode in ('plain','ordinary','observed','repeat'):
            for field,value in [('ids',[7,9]),('text','other'),('token_cap_reached',True),('eos_observed',True),('stop_contract','invented')]:
                a,b,ref=self.fixture();a[mode][field]=value
                with self.assertRaises(ValueError):r.barrier(a,b,ref)
        for mode in ('observed','repeat'):
            for field,value in [('shape',[2,1,2]),('type','i32'),('bytes',12),('sha256','0'*64)]:
                a,b,ref=self.fixture();a[mode]['stages']['layer_3'][field]=value
                with self.assertRaises(ValueError):r.barrier(a,b,ref)
            a,b,ref=self.fixture();a[mode]['vision']['sha256']='0'*64
            with self.assertRaises(ValueError):r.barrier(a,b,ref)
    def test_raw_stdout_change_cannot_hide_behind_normalized_text(self):
        a,b,ref=self.fixture();a['ordinary']['command']['stdout_sha256']=hashlib.sha256(b'complete\r\ntext\n').hexdigest()
        with self.assertRaises(ValueError):r.barrier(a,b,ref)

    def test_signed_zero_is_not_byte_neutral(self):
        a,b,ref=self.fixture();a['repeat']['stages']['layer_3']['sha256']=hashlib.sha256(struct.pack('<4f',-0.,1.,2.,3.)).hexdigest()
        with self.assertRaises(ValueError):r.barrier(a,b,ref)
    def test_import_blocker_covers_framework_array_and_image_before_544(self):
        block=f.NoPayloadImports()
        for name in ('numpy','numpy.core','torch','tensorflow','transformers','PIL','PIL.Image'):
            with self.assertRaises(ImportError):block.find_spec(name)
        self.assertIsNone(block.find_spec('json'))
    def test_image_import_is_blocked_before_authentication_without_execution(self):
        import importlib
        with tempfile.TemporaryDirectory()as d:
            directory=Path(d);package=directory/'PIL';package.mkdir();sentinel=directory/'executed'
            (package/'__init__.py').write_text('from pathlib import Path;Path('+repr(str(sentinel))+').touch()')
            block=f.NoPayloadImports();sys.meta_path.insert(0,block);sys.path.insert(0,d)
            try:
                with self.assertRaises(ImportError):importlib.import_module('PIL')
                self.assertFalse(sentinel.exists())
            finally:sys.path.remove(d);sys.meta_path.remove(block)

    def test_png_whole_sha_and_bounded_response(self):
        payload=b'authored-image';pin=dict(maximum_bytes=100,sha256=hashlib.sha256(payload).hexdigest())
        class Response(io.BytesIO):
            status=200;headers={'Content-Length':str(len(payload)),'Content-Encoding':'identity'}
        with tempfile.TemporaryDirectory()as d,patch.object(f.urllib.request,'urlopen',return_value=Response(payload)):
            proof=f.image('https://example.org/frozen',Path(d)/'input.png',pin);self.assertEqual(proof['sha256'],pin['sha256'])
        for data,headers,status in [(b'wrong',{'Content-Length':'5'},200),(payload,{'Content-Length':'101'},200),(payload,{'Content-Length':str(len(payload))},206)]:
            response=io.BytesIO(data);response.headers=headers;response.status=status
            with tempfile.TemporaryDirectory()as d,patch.object(f.urllib.request,'urlopen',return_value=response):
                with self.assertRaises(ValueError):f.image('https://example.org/frozen',Path(d)/'input.png',pin)
    def test_process_helper_exact_reviewed_source(self):
        self.assertEqual((ROOT/'tools/unlimited-reused-q4-v1/process.py').read_bytes(),subprocess.check_output(['git','show','765c758fe:tools/ci_process.py'],cwd=ROOT))

    def test_declared_capture_order_covers58_without_altering_observations(self):
        order=c.stage_order();self.assertEqual(len(order),58);self.assertEqual(len(set(order)),58)
        self.assertEqual(order[0],'layer_0_chunk_0');self.assertEqual(order[-2:],['final_norm','logits'])
        self.assertLess(order.index('layer_11_chunk_0_gate'),order.index('layer_11_chunk_0'))
        self.assertEqual(sum(k.endswith('_ids')for k in order),11)

    def test_exact_source_closure(self):
        cfg=c.declaration()
        for group in ('protected_sha256','owned_sha256'):
            for name,pin in cfg[group].items():self.assertEqual(c.digest(ROOT/name),pin,name)
        self.assertEqual(cfg['counts']['all_authenticated_files'],544);self.assertEqual(cfg['counts']['f16_forwards'],16)
        self.assertEqual(cfg['counts']['q4_forwards'],12);self.assertEqual(cfg['counts']['projection_executions'],16)
        self.assertEqual(cfg['counts']['original_forwards'],0);self.assertEqual(cfg['counts']['quantizer_runs'],0)
    def test_old_files_and_runtime_remain_exact_1aa(self):
        for name in ('panel.py','contract.py','retained_fetch_gha.py','retained_v3_gha.py','overlay.py','adapter.py','build_profile.py'):
            p='tools/unlimited-ocr2-q4k/'+name
            self.assertEqual((ROOT/p).read_bytes(),subprocess.check_output(['git','show','1aaee225783828162883a6ae062446036cd2f043:'+p],cwd=ROOT))
        self.assertEqual(subprocess.check_output(['git','rev-parse','HEAD:src'],cwd=ROOT),subprocess.check_output(['git','rev-parse',c.frozen.BASE+':src'],cwd=ROOT))
    def test_low_level_provider_not_old_guard_controller(self):
        text=(ROOT/'tools/unlimited-reused-q4-v1/fetch.py').read_text()
        for name in ('delivery.output_urls','delivery.download','delivery.safe_name','structural.audit'):self.assertIn(name,text)
        self.assertNotIn('delivery.main(',text);self.assertNotIn('delivery.source_guard(',text);self.assertNotIn('source_guard=',text)
        self.assertLess(text.index("len(result['images'])==4"),text.index('sys.meta_path.remove(block)'))
        self.assertNotIn('render(',text)
    def test_no_quantizer_or_original_dispatch_and_only_last1(self):
        text=(ROOT/'tools/unlimited-reused-q4-v1/run.py').read_text();replay=(ROOT/'tools/unlimited-reused-q4-v1/replay.py').read_text()
        self.assertNotIn('crispembed-quantize',text);self.assertNotIn('overrides(',text+replay);self.assertNotIn('range(8, 12)',replay)
        self.assertIn("report.get('f16_all_four_strict_retained_barrier_passed')is True",replay)
        self.assertLess(replay.index("report.update(f16_all_four_strict_retained_barrier_passed=True"),replay.index("for mode in ('plain','observed','repeat')"))
        self.assertIn("old.projection_inputs(expected,11",replay)
        self.assertIn('candidate_preserved=True',replay)
    def test_workflow_exact_source_and_outputs_always(self):
        text=(ROOT/'.github/workflows/unlimited-reused-q4-first-panel-v1.yml').read_text()
        self.assertIn('ref: ${{ inputs.source_ref }}',text);self.assertIn('fetch-depth: 0',text);self.assertIn('submodules: recursive',text)
        self.assertLess(text.index('/fetch.py --source'),text.index('/run.py --source'))
        self.assertEqual(text.count('if: always()'),3);self.assertEqual(text.count('retention-days: 90'),3)
        self.assertIn('secrets.KAGGLE_TOKEN',text);self.assertIn('timeout-minutes: 360',text)
    def test_source_and_help_load_no_optional_packages(self):
        for p in(ROOT/'tools/unlimited-reused-q4-v1').glob('*.py'):ast.parse(p.read_text())
        for name in ('numpy','torch','tensorflow','transformers','PIL','gguf'):self.assertNotIn(name,sys.modules)
        result=subprocess.run([sys.executable,'-I',ROOT/'tools/unlimited-reused-q4-v1/replay.py','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--budget-seconds',result.stdout)

if __name__=='__main__':unittest.main()
