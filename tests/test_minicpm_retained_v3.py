"""Stdlib controls for immutable23/resume7 and bounded canonical retention."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools/minicpm-retained-v3'),str(ROOT/'tools/minicpm-ocr'),str(ROOT/'tools')]
import common as C
import retention as R
import graph_adapter as A


def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools/minicpm-retained-v3'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class Controls(unittest.TestCase):
    def test_exact_old23_and_missing_seven_never_replay_completed_arm(self):
        catalog=json.loads(C.CATALOG.read_text());missing=C.remaining(catalog)
        self.assertEqual(len(missing),7);self.assertEqual(missing[0],('tall','q4_k',1))
        self.assertTrue(all(row[0]=='tiny'for row in missing[1:]));self.assertEqual(sum(r['strict_passed']for r in catalog['cases']),8)
        catalog['cases'][0]['id']='tiny'
        with self.assertRaises(ValueError):C.remaining(catalog)

    def test_modes_keep_old_budgets_and_analysis_zero_models_forwards(self):
        a=C.declaration('analysis');b=C.declaration('continue')
        self.assertEqual(a['native_forwards'],0);self.assertFalse(a['model_access'])
        self.assertEqual(a['budgets'],b['budgets']);self.assertEqual(b['budgets']['max_absolute'],.002)
        self.assertFalse(a['original_rerun']);self.assertFalse(b['original_rerun'])
        self.assertEqual(a['working_output_cap_bytes'],12*2**30);self.assertEqual(a['working_reserve_bytes'],4*2**30)

    def test_atomic_receipt_preserves_prior_valid_json_when_new_write_fails(self):
        with tempfile.TemporaryDirectory()as folder:
            target=Path(folder)/'receipt.json';C.atomic(target,{'old':True});old=target.read_bytes()
            with patch.object(C.os,'replace',side_effect=OSError('synthetic exhausted volume')):
                with self.assertRaises(OSError):C.atomic(target,{'new':True})
            self.assertEqual(target.read_bytes(),old)
            self.assertEqual(list(Path(folder).iterdir()),[target])
            with patch.object(C.json,'dump',side_effect=ValueError('nonfinite synthetic receipt')):
                with self.assertRaises(ValueError):C.atomic(target,{'new':True})
            self.assertEqual(target.read_bytes(),old)

    def test_working_quota_and_reserve_separate_from_scratch(self):
        C.room(6*2**30,2*2**30,6*2**30)
        for current,planned,free in [(11*2**30,2*2**30,10*2**30),(6*2**30,2*2**30,6*2**30-1),
                                     (0,2*2**30+1,10*2**30)]:
            with self.assertRaises(ValueError):C.room(current,planned,free)

    def test_only_known_duplicate_adapters_can_be_omitted(self):
        for name in ['original_0.f32','native_25.f32','prefill/original.gguf','projected/comparison/native.gguf',
                     'projected/layer_23.f32','probe-input.gguf']:
            self.assertTrue(C.duplicate(name));self.assertFalse(C.canonical(name))
        for name in ['dump/model_merger.npy','dump/ids.npy','projected/projection_0.control.f32',
                     'projected/projection_186.native.f32']:
            self.assertTrue(C.canonical(name));self.assertFalse(C.duplicate(name))
        self.assertFalse(C.duplicate('other-project/model.gguf'));self.assertFalse(C.canonical('checkpoint/model.safetensors'))

    def test_streaming_gguf_reconstruction_matches_real_lossless_writer(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);data=root/'raw.f32';data.write_bytes(struct.pack('<6f',-0.,1.,2.,3.,4.,5.))
            digest=C.sha(data);spec=dict(path=data,shape=[3,2],sha256=digest)
            A.write_gguf(root/'adapter.gguf',[dict(name='stage_0',**spec),dict(name='stage_1',**spec)])
            measured=R.gguf_digest([(name,dict(path=data,shape=[3,2],raw_sha256=digest,kind='raw'))for name in ['stage_0','stage_1']])
            self.assertEqual(measured['sha256'],C.sha(root/'adapter.gguf'));self.assertEqual(measured['bytes'],(root/'adapter.gguf').stat().st_size)
            data.write_bytes(struct.pack('<6f',0.,1.,2.,3.,4.,5.))
            with self.assertRaisesRegex(ValueError,'payload SHA'):R.gguf_digest([('stage_0',dict(path=data,shape=[3,2],raw_sha256=digest,kind='raw'))])

    def test_only_actual_frozen_batch1_concat_shapes_align(self):
        a=load('retained_analysis','analyze.py')
        self.assertEqual(a.aligned_shape([1,12,1152],[4,8],1152),[12,1152])
        self.assertEqual(a.aligned_shape([3,1024],[1,2],1024,True),[3,1024])
        for shape in [[2,8,1152],[1,13,1152],[12,1024]]:
            with self.assertRaises(ValueError):a.aligned_shape(shape,[4,8],1152)
        parts=[dict(w=28,h=28,tokens=1,overview=True),dict(w=28,h=56,tokens=2,overview=False)]
        self.assertEqual(a.pixel_plan(parts,[[2,2],[4,2]]),[[2,2],[4,2]])
        with self.assertRaises(ValueError):a.pixel_plan(parts,[[2,2],[2,4]])

    def test_exact_original_cli_gates_acceptance_and_limits_copied(self):
        old=ast.parse((ROOT/'tools/minicpm-ocr/graph_panel.py').read_text());new=ast.parse((ROOT/'tools/minicpm-retained-v3/continue.py').read_text())
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='run_case')
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='run_case')
        def selected(function):
            out=[]
            for node in ast.walk(function):
                if isinstance(node,(ast.Assign,ast.AugAssign)):
                    text=ast.unparse(node.targets[0]if isinstance(node,ast.Assign)else node.target)
                    if text=='env'or text=="row['passed']":out.append(ast.dump(node,include_attributes=False))
                if isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=='captured':
                    # CLI argument order/gates retained; probe path is scratch-only.
                    if ast.unparse(node.args[0]).startswith('[str(args.cli)'):out.append(ast.dump(node.args[0],include_attributes=False))
            return out
        self.assertEqual(selected(a),selected(b));self.assertGreaterEqual(len(selected(a)),4)
        self.assertNotIn('graph_panel.compare_pairs=',(ROOT/'tools/minicpm-retained-v3/continue.py').read_text())

    def test_all_entrypoints_help_reject_optional_framework_imports(self):
        code="""import builtins,runpy,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0]in{'numpy','torch','transformers','gguf','kaggle'}:raise RuntimeError('optional import before admission:'+name)
 return original(name,*args,**kwargs)
builtins.__import__=guard
target=sys.argv[1];sys.argv=[target,'--help'];runpy.run_path(target,run_name='__main__')
"""
        for name in ['common.py','fetch.py','build.py','analyze.py','continue.py','launch.py']:
            result=subprocess.run([sys.executable,'-I','-c',code,str(ROOT/'tools/minicpm-retained-v3'/name)],capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,name+':'+result.stderr)
        self.assertFalse(any(n in sys.modules for n in ['numpy','torch','transformers','gguf','kaggle']))

    def test_exact_version_failed_retained_run_is_distinct_from_original_success(self):
        import types,contextlib
        fetch=load('retained_fetch','fetch.py');calls=[]
        class Request:
            def __str__(self):return json.dumps({'versionLabel':self.version_label})
        text='source '+'a'*40
        pin=dict(kernel_slug='frozen',version_label='v3',source='a'*40,status='ERROR',submitted_script_sha256=hashlib.sha256(text.encode()).hexdigest())
        class Client:
            def get_kernel(self,q):calls.append(q);return json.dumps({'blob':{'source':text}})
            def get_kernel_session_status(self,q):calls.append(q);return types.SimpleNamespace(status='ERROR')
            def list_kernel_session_output(self,q):calls.append(q);return types.SimpleNamespace(files=[],next_page_token=None)
        api=types.SimpleNamespace(build_kaggle_client=lambda:contextlib.nullcontext(types.SimpleNamespace(kernels=types.SimpleNamespace(kernels_api_client=Client()))))
        module=types.SimpleNamespace(ApiGetKernelRequest=Request,ApiListKernelSessionOutputRequest=Request,ApiGetKernelSessionStatusRequest=Request)
        with patch.object(fetch.importlib,'import_module',return_value=module),patch.dict(fetch.os.environ,{'KAGGLE_ACCOUNT':'private-fixture'}):
            files,proof=fetch.client_index(api,pin)
            self.assertEqual(proof['status'],'ERROR');self.assertEqual(len(calls),3)
            self.assertTrue(all(q.version_label=='v3'for q in calls));self.assertNotIn('private-fixture',json.dumps(proof))
            pin['status']='COMPLETE'
            with self.assertRaises(ValueError):fetch.client_index(api,pin)

    def test_analysis_workflow_never_reconstructs_or_accesses_models(self):
        a=(ROOT/'.github/workflows/minicpm-graph-retained-analysis-v1.yml').read_text()
        self.assertNotIn('build.py models',a);self.assertNotIn('continue.py',a)
        self.assertIn('analyze.py',a);self.assertNotIn('workflow_call:',a)
        b=(ROOT/'.github/workflows/minicpm-graph-seven-arm-v1.yml').read_text()
        self.assertIn('build.py models',b);self.assertIn('continue.py',b)
        self.assertIn('retention-days: 90',a);self.assertIn('retention-days: 90',b)


if __name__=='__main__':unittest.main()
