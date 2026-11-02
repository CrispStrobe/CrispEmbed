"""Tiny stdlib adapters/source fixtures; no native execution or NumPy import."""
import copy
import ast
import os
import shutil
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/minicpm-ocr'))
import graph_adapter as a
import graph_panel as p
import build_profile as profile


class Adapter(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def source(self,name='input.f32',bits=(0,0x80000000,0x3f800000,0xbf800000),shape=(2,2)):
        path=self.root/name;path.write_bytes(struct.pack('<'+'I'*len(bits),*bits))
        return p.spec(path,shape)
    def test_exact_bits_shape_and_signed_zero(self):
        source=self.source();target=self.root/'adapter.gguf'
        receipt=a.write_gguf(target,[dict(name='node',**source)])
        self.assertEqual(a.read_gguf(target)['node']['sha256'],source['sha256'])
        self.assertEqual(receipt['tensors']['node']['shape'],[2,2])
        self.assertIn(struct.pack('<IIII',0,0x80000000,0x3f800000,0xbf800000),target.read_bytes())
    def test_altered_payload_sha_nonfinite_shape_and_duplicate_reject(self):
        source=self.source();Path(source['path']).write_bytes(b'0'*16)
        with self.assertRaisesRegex(ValueError,'SHA256'):a.source_bytes(source)
        source=self.source('nan.f32',(0x7fc00000,0,0,0))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):a.source_bytes(source)
        for dims in ([0],[300001],[4096,300000],[1,1,1,1,1]):
            with self.assertRaises(ValueError):a.shape(dims)
        source=self.source('valid.f32')
        with self.assertRaises(ValueError):a.write_gguf(self.root/'duplicate.gguf',[dict(name='same',**source)]*2)
    def test_wrong_shape_and_truncated_readback(self):
        left=self.source();right=dict(left,shape=[4])
        with self.assertRaisesRegex(ValueError,'shapes differ'):a.write_pair(self.root/'a',self.root/'b',[dict(name='stage',original=left,native=right)])
        path=self.root/'valid.gguf';a.write_gguf(path,[dict(name='s',**left)]);path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):a.read_gguf(path)
    def test_vocabulary_axis_is_explicit_and_payload_bound_remains(self):
        self.assertEqual(a.shape([1,248100]),(1,248100))
        self.assertEqual(a.shape([2372,1024]),(2372,1024))
        with self.assertRaisesRegex(ValueError,'element bound'):a.shape([248100,1024])


class BuildProfile(unittest.TestCase):
    def fixture(self, directory):
        root,other=directory/'root',directory/'probe';root.mkdir();other.mkdir()
        values={k:'OFF' for k in profile.CACHE_KEYS}
        values.update(GGML_LLAMAFILE='ON',GGML_CPU_REPACK='ON',GGML_AVX='ON',GGML_AVX2='ON',GGML_FMA='ON',GGML_F16C='ON')
        for folder,target in [(root,'crispembed'),(other,'minicpm-graph-probe')]:
            (folder/'CMakeCache.txt').write_text(''.join(f'{k}:BOOL={v}\n' for k,v in values.items()))
            rows=[]
            for name in profile.QWEN:
                rows.append(dict(file='/src/'+name,arguments=['c++','-mavx2','-mfma','-mf16c','-ffp-contract=off','-o',f'CMakeFiles/{target}.dir/{name}.o']))
            for name in ['ggml-cpu.c','ggml-cpu.cpp']:
                rows.append(dict(file='/ggml/'+name,arguments=['cc','-mavx2','-mfma','-mf16c','-DGGML_USE_LLAMAFILE','-DGGML_USE_CPU_REPACK','-o',f'CMakeFiles/ggml-cpu.dir/{name}.o']))
            (folder/'compile_commands.json').write_text(json.dumps(rows))
        return root,other
    def test_exact_actual_flags_and_backend_profile(self):
        with tempfile.TemporaryDirectory() as d:
            root,other=self.fixture(Path(d));result=profile.verify(root,other)
            self.assertTrue(result['passed']);self.assertEqual(len(result['implementation']),6);self.assertEqual(len(result['backend']),2)
    def test_backend_cache_actual_isa_and_contraction_mismatches_reject(self):
        for kind in ['llamafile','openmp','isa','contraction','backend-definitions']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root,other=self.fixture(Path(d))
                if kind in ('llamafile','openmp'):
                    path=other/'CMakeCache.txt';s=path.read_text()
                    s=s.replace('GGML_LLAMAFILE:BOOL=ON','GGML_LLAMAFILE:BOOL=OFF') if kind=='llamafile' else s.replace('GGML_OPENMP:BOOL=OFF','GGML_OPENMP:BOOL=ON')
                    path.write_text(s)
                else:
                    path=other/'compile_commands.json';rows=json.loads(path.read_text())
                    if kind=='isa':rows[0]['arguments'].remove('-mfma')
                    elif kind=='contraction':rows[0]['arguments'].remove('-ffp-contract=off')
                    else:rows[-1]['arguments'].remove('-DGGML_USE_LLAMAFILE')
                    path.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):profile.verify(root,other)
    def test_exact_root_helper_and_profile_before_model_conversion(self):
        root=(ROOT/'CMakeLists.txt').read_text();standalone=(ROOT/'tests/minicpm-graph/CMakeLists.txt').read_text()
        begin=root.index('function(crispembed_apply_arch_flags target)');end=root.index('endfunction()',begin)+len('endfunction()')
        self.assertIn(root[begin:end],standalone)
        self.assertIn('set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)',standalone)
        self.assertIn('set(GGML_OPENMP OFF CACHE BOOL "" FORCE)',standalone)
        kernel=(ROOT/'tools/kaggle/minicpm-ocr-graph/kernel.py').read_text()
        self.assertLess(kernel.index("profile=verify("),kernel.index("checkpoint=reference.parent.parent"))
        self.assertIn("'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'",kernel)
        for step in ("run(['/kaggle/temp/b/test-ultrabert-profile'])", "run(['/kaggle/temp/jq/josev-quant-controls'])"):
            self.assertLess(kernel.index(step),kernel.index('checkpoint=reference.parent.parent'))


class FailureEvidence(unittest.TestCase):
    def test_timeout_preserves_real_child_output(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            result=p.captured([sys.executable,'-S','-c',"import sys,time;print('started',flush=True);print('before-timeout',file=sys.stderr,flush=True);time.sleep(3)"],folder,'native',os.environ,1.0)
            self.assertTrue(result.timed_out);self.assertNotEqual(result.returncode,0)
            self.assertIn(b'started',result.stdout);self.assertIn(b'before-timeout',result.stderr)
            self.assertEqual(result.stdout,(folder/'native.stdout').read_bytes())
    def test_later_structural_rejection_keeps_decoded_results(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);row=dict(id='line',dtype='f16',graph=1,ids=[1,248046],text='kept',confidences=[.5,.8],passed=True)
            p.write(folder/'partial-case.json',row)
            actual=p.recover_case(folder,'line','f16',1,ValueError('later projection failure'))
            for key in ('ids','text','confidences'):self.assertEqual(actual[key],row[key])
            self.assertFalse(actual['passed']);self.assertIn('later projection',actual['error'])
            with self.assertRaises(ValueError):p.recover_case(folder,'tiny','f16',1,'wrong-case')
    def test_fresh_scratch_parent_created_before_resource_observation(self):
        source=ast.parse((ROOT/'tools/kaggle/minicpm-ocr-graph/kernel.py').read_text())
        function=next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name=='resources')
        module=ast.Module(body=[function],type_ignores=[]);namespace=dict(Path=Path,shutil=shutil,os=os)
        exec(compile(ast.fix_missing_locations(module),'<isolated-resource-helper>','exec'),namespace)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);scratch=folder/'new-parent'/'scratch';meminfo=folder/'meminfo'
            meminfo.write_text('MemAvailable: 12345678 kB\n')
            self.assertFalse(scratch.exists())
            result=namespace['resources'](scratch,meminfo)
            self.assertTrue(scratch.is_dir());self.assertEqual(result['mem_available'],12345678*1024)
            self.assertGreater(result['disk_free'],0)

    def test_actual_timing_parser_and_error_metrics(self):
        text='  [time] minicpm_ocr.prefill 123.4ms tokens=256\n  [time] minicpm_ocr.decode steps=2 total=3.0ms mean=1.5ms p50=2.0ms max=2.0ms list=1.0,2.0\n  [time] minicpm_ocr.ids n=3 list=1,2,248046\n'
        actual=p.parse_timing(text)
        self.assertEqual(actual['prefill_ms'],123.4);self.assertEqual(actual['ids'],[1,2,248046])
        self.assertEqual(actual['decode']['steady_mean_ms'],2.0)
        self.assertEqual(p.first_difference([1,2],[1,3]),1);self.assertEqual(p.first_difference([1],[1,2]),1)
        self.assertEqual(p.edits('Grüße','Grüxe'),1);self.assertEqual(p.edits('one two'.split(),'one three'.split()),1)


class Recipe(unittest.TestCase):
    def fixture(self):
        result={}
        for i in range(24):
            prefix=f'l.layers.{i}.'
            for family in ('mlp.gate_proj','mlp.up_proj','mlp.down_proj'):
                result[prefix+family+'.weight']=dict(type=1,shape=[1024,1024],bytes=1,sha256='a'*64)
            families=('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj') if i%4==3 else ('linear_attn.in_proj_qkv','linear_attn.in_proj_z','linear_attn.in_proj_a','linear_attn.in_proj_b','linear_attn.out_proj')
            for family in families:result[prefix+family+'.weight']=dict(type=1,shape=[1024,1024],bytes=1,sha256='a'*64)
        result.update({f'v.protected.{i}':dict(type=1,shape=[14,14,3,1152],bytes=1,sha256='b'*64) for i in range(593)})
        candidate=copy.deepcopy(result)
        for name in candidate:
            if p.SELECTED.fullmatch(name):candidate[name].update(type=12,bytes=2,sha256='c'*64)
        return result,candidate
    def test_exact_186_selected_and_593_protected(self):
        ref,native=self.fixture();x=p.protected_q4(ref,native)
        self.assertEqual(len(x['selected']),186);self.assertEqual(x['protected'],593)
    def test_protected_mutation_wrong_quant_missing_matrix_rejected(self):
        ref,native=self.fixture();name=next(k for k in ref if not p.SELECTED.fullmatch(k));native[name]['sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'protected'):p.protected_q4(ref,native)
        ref,native=self.fixture();name=next(k for k in ref if p.SELECTED.fullmatch(k));native[name]['type']=8
        with self.assertRaisesRegex(ValueError,'Q4_K'):p.protected_q4(ref,native)
        ref,native=self.fixture();native.pop(name)
        with self.assertRaisesRegex(ValueError,'names'):p.protected_q4(ref,native)
    def test_q8_patch_requires_f16_four_axes_and_exact_source_bits(self):
        name='v.emb.patch_embedding.weight'
        ref={name:dict(type=1,shape=[14,14,3,1152],bytes=1354752,sha256='a'*64)}
        self.assertTrue(p.protected_q8_patch(ref,copy.deepcopy(ref))['exact_f16_source'])
        for field,value in [('type',0),('shape',[588,1152]),('sha256','b'*64),('bytes',2709504)]:
            bad=copy.deepcopy(ref);bad[name][field]=value
            with self.assertRaises(ValueError):p.protected_q8_patch(ref,bad)

    def test_actual_ref_api_schema_fixed_budgets_and_no_model_import(self):
        self.assertNotIn('numpy',sys.modules);self.assertNotIn('torch',sys.modules)
        s=(ROOT/'tools/minicpm-ocr/graph_panel.py').read_text()
        self.assertIn("collect(run.returncode,folder/'diff.json',independent)",s)
        self.assertIn('aeb1756dcbcfb72d72c13cf5a8d8131bb90e8d8884cf07c70723cc1fe4a64c32',s)
        self.assertIn("all_protected_byte_exact=True",s)
        self.assertIn("len(report['cases'])==30",s)
        self.assertIn("for case in ref['cases']:\n            for dtype,graph in",s)
        self.assertEqual(json.loads((ROOT/'docs/minicpm-ocr-graph-v1.json').read_text())['execution_order'],'case-major-with-declared-six-arm-order')
        probe=(ROOT/'tests/minicpm-graph/probe.cpp').read_text()
        self.assertIn('journal.calls == 187',probe)
        self.assertIn('cache value neutrality',probe)
        self.assertIn('core_util::clean_exit(',probe)
        diff=(ROOT/'tests/minicpm-graph/diff.cpp').read_text()
        self.assertIn('double(r.max_abs) <= 0.002 && double(r.cos_min) >= 0.99999',diff)
        self.assertIn('return failed ? 1 : 0',diff)
    def test_observer_only_in_separate_build(self):
        s=(ROOT/'src/qwen35_graph_decoder.cpp').read_text()
        block=s[s.index('void graph_decoder::observe('):s.index('bool graph_decoder::project(')]
        self.assertIn('#endif',block)
        cmake=(ROOT/'tests/minicpm-graph/CMakeLists.txt').read_text()
        self.assertIn('target_compile_definitions(minicpm-graph-probe PRIVATE CRISPEMBED_MINICPM_GRAPH_AUDIT=1)',cmake)
        self.assertNotIn('target_compile_definitions(minicpm-graph-controls',cmake)


if __name__=='__main__':unittest.main()
