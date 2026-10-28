"""Tiny stdlib adapters/source fixtures; no native execution or NumPy import."""
import copy
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
    def test_actual_ref_api_schema_fixed_budgets_and_no_model_import(self):
        self.assertNotIn('numpy',sys.modules);self.assertNotIn('torch',sys.modules)
        s=(ROOT/'tools/minicpm-ocr/graph_panel.py').read_text()
        self.assertIn("collect(run.returncode,folder/'diff.json',independent)",s)
        self.assertIn('aeb1756dcbcfb72d72c13cf5a8d8131bb90e8d8884cf07c70723cc1fe4a64c32',s)
        self.assertIn("all_protected_byte_exact=True",s)
        self.assertIn("len(report['cases'])==30",s)
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
