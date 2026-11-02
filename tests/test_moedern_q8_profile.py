#!/usr/bin/env python3
"""Stdlib source/planning guards; actual C/ggml controls execute hosted only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('q8recipe',ROOT/'tools/moedern-colbert/quantize_q8_critical.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


class Controls(unittest.TestCase):
    def test_genuine_q8_and_critical_shapes(self):
        for width in [128,640,1024]:
            for name in ['attn_q','attn_k','attn_v','attn_output','ffn_up_exps','ffn_gate_exps','ffn_down_exps','ffn_up_shexp','ffn_gate_shexp','ffn_down_shexp','conv_in','conv_out']:
                p=w.plan('blk.0.'+name+'.weight',[width,2,8])
                self.assertEqual((p['type'],p['type_id'],p['bytes']),('Q8_0',8,width*2*8//32*34))
                self.assertEqual(p['shape'],[width,2,8])
        for name in ['token_embd.hash.weight','token_embd.proj.weight','dense_head.weight','blk.0.ffn_gate_inp.weight','blk.0.attn_norm.weight','blk.0.conv_dw.weight','token_embd.norm.bias']:
            self.assertEqual(w.plan(name,[128,2])['type'],'F32',name)

    def test_no_padding_fallback_or_noninteger_axes(self):
        for shape in [[16,2],[33,2]]:
            with self.assertRaisesRegex(ValueError,'no fallback'):w.plan('blk.0.attn_q.weight',shape)
        for shape in [[],[True],[128.],[0],[-1],[30003],[128]*5]:
            with self.assertRaises(ValueError):w.plan('owned.weight',shape)
        self.assertEqual(w.plan('dense_head.weight',[16,1])['shape'],[16,1])

    def test_standard7_exact_integer_not_tensor_type8(self):
        self.assertTrue(w.q8_file_type(7))
        for bad in [True,False,7.,'7',0,1,2,8,12,14,None]:self.assertFalse(w.q8_file_type(bad))

    def test_actual_C_command_existing_keys_and_all_exclusions(self):
        rows=[w.plan('blk.0.attn_q.weight',[128,2]),w.plan('dense_head.weight',[128,2]),w.plan('token_embd.hash.weight',[16,2])]
        args=w.command('/quantizer','/source','/out',rows)
        self.assertEqual(args[:4],['/quantizer','/source','/out','q8_0'])
        self.assertEqual(args.count('--metadata-string'),2)
        self.assertIn('moedern.packed.format='+w.FORMAT,args);self.assertIn('modernbert_hash_moe_v10.quant_recipe='+w.RECIPE,args)
        self.assertIn('dense_head.weight=f32',args);self.assertIn('token_embd.hash.weight=f32',args)
        self.assertNotIn('blk.0.attn_q.weight=f32',args)

    def test_metadata_typed_and_raw_signedzero_identity(self):
        def kv(kind,value,raw):return dict(kind=kind,value=value,raw_sha256=raw)
        a={'metadata':{'moedern.packed.format':kv(8,'old','a'),'modernbert_hash_moe_v10.quant_recipe':kv(8,'f32','b'),'critical-f64':kv(12,-0.,'minuszero')}}
        b={'metadata':{'moedern.packed.format':kv(8,w.FORMAT,'c'),'modernbert_hash_moe_v10.quant_recipe':kv(8,w.RECIPE,'d'),'critical-f64':kv(12,-0.,'minuszero'),'general.file_type':kv(4,7,'e'),'general.quantization_version':kv(4,2,'f')}}
        w.metadata_readback(a,b)
        changed=json.loads(json.dumps(b));changed['metadata']['critical-f64']['raw_sha256']='pluszero'
        with self.assertRaisesRegex(ValueError,'raw metadata'):w.metadata_readback(a,changed)
        for value in [12,14,8]:
            changed=json.loads(json.dumps(b));changed['metadata']['general.file_type']['value']=value
            with self.assertRaisesRegex(ValueError,'standard Q8'):w.metadata_readback(a,changed)
        changed=json.loads(json.dumps(b));changed['metadata']['general.file_type']['kind']=8
        with self.assertRaises(ValueError):w.metadata_readback(a,changed)
        changed=json.loads(json.dumps(b));changed['metadata']['extra']=kv(8,'x','g')
        with self.assertRaisesRegex(ValueError,'key set'):w.metadata_readback(a,changed)

    def test_all_three_exact_semantic_role_counts(self):
        # Independently derive the published schemas rather than trusting counts
        # or treating any non-F32 tensor as a successful quantization.
        for variant,hidden,hashdim,expert_dim,dilations in [
            ('1.5M',128,16,128,[0,0]),('27M',640,80,640,[0,0]),
            ('115M',1024,192,640,[1,1,0,2,1,0,1,0,2,0])]:
            dims={'token_embd.hash.weight':[hashdim,30002],'token_embd.proj.weight':[hashdim,hidden],
                  'token_embd.norm.weight':[hidden],'token_embd.norm.bias':[hidden],
                  'output_norm.weight':[hidden],'dense_head.weight':[hidden,128]}
            for i,d in enumerate(dilations):
                prefix='blk.'+str(i)+'.'
                if d:
                    dims.update({prefix+n:[hidden]for n in ['conv_norm.weight','conv_gate_b_norm.weight','conv_gate_c_norm.weight']})
                    dims.update({prefix+'conv_in.weight':[hidden,3*hidden],prefix+'conv_out.weight':[hidden,hidden],prefix+'conv_dw.weight':[hidden,3]})
                else:
                    dims.update({prefix+n:[hidden]for n in ['attn_norm.weight','ffn_norm.weight']})
                    # All three blueprints have KV width exactly hidden/2.
                    dims.update({prefix+'attn_q.weight':[hidden,hidden],prefix+'attn_output.weight':[hidden,hidden],prefix+'attn_k.weight':[hidden,hidden//2],prefix+'attn_v.weight':[hidden,hidden//2],prefix+'ffn_gate_inp.weight':[hidden,8]})
                    for name in ['ffn_up','ffn_gate']:
                        dims[prefix+name+'_exps.weight']=[hidden,expert_dim,8];dims[prefix+name+'_shexp.weight']=[hidden,expert_dim]
                    dims[prefix+'ffn_down_exps.weight']=[expert_dim,hidden,8];dims[prefix+'ffn_down_shexp.weight']=[expert_dim,hidden]
            rows=[w.plan(n,d)for n,d in dims.items()];cfg=w.PROFILES[variant]
            self.assertEqual(len(rows),cfg['tensors']);self.assertEqual(sum(r['semantic_parameters']for r in rows),cfg['parameters'])
            self.assertEqual(sum(r['type']=='Q8_0'for r in rows),w.Q8_COUNTS[variant])

    def test_exact_source_closure_and_preserved_old_paths(self):
        p=json.loads((ROOT/'docs/moedern-q8-f32-critical-v1.json').read_text())
        for path,digest in p['owned_sha256'].items():self.assertEqual(w.sha(ROOT/path),digest,path)
        for path,obj in p['protected_git_objects'].items():self.assertEqual(subprocess.check_output(['git','rev-parse',p['base']+':'+path],cwd=ROOT,text=True).strip(),obj,path)
        for path in ['src/moedern_text_engine.cpp','src/moedern_colbert.cpp','src/moedern_q4k_profile.h','src/moedern_packed_profile.h','tools/moedern-colbert/quantize_gguf.py','tools/moedern-colbert/quantize_q4k_padded.py']:
            self.assertEqual((ROOT/path).read_bytes(),subprocess.check_output(['git','show',p['base']+':'+path],cwd=ROOT),path)
        before=subprocess.check_output(['git','show',p['base']+':src/moedern_family_engine.cpp'],cwd=ROOT,text=True)
        after=(ROOT/'src/moedern_family_engine.cpp').read_text()
        # Entire existing fast graph, including expert weighting/router math, is exact.
        a=before[before.index('floats encode_fast('):before.index('bool encode_tokens(')]
        b=after[after.index('floats encode_fast('):after.index('bool encode_tokens(')]
        self.assertEqual(a,b)
        self.assertFalse(p['model_qualified']);self.assertFalse(p['dispatch_authorized'])

    def test_dedicated_1p5_route_and_no_legacy_gate_weakening(self):
        text=(ROOT/'src/moedern_text.cpp').read_text();front=(ROOT/'src/moedern_text_frontend.cpp').read_text();engine=(ROOT/'src/moedern_family_engine.cpp').read_text()
        self.assertRegex(text,r'named_q8\s*\? pending->tokenizer\.load_packed_q8')
        self.assertRegex(text,r'named_q8\s*\? moedern_family::init_q8')
        self.assertIn('if (q8_pin) result.family_revision_ = q8_pin->revision',front)
        self.assertIn('Named Q8_0 profile requires explicit Q8_0 initializer',engine)
        self.assertRegex(engine,r'GGML_TYPE_Q8_0\s*:\s*GGML_TYPE_F32')
        self.assertIn('actual_q8 == (p.layers == 10 ? 52 : 20)',engine)
        self.assertIn('Q4_K_PADDED OFF',engine)
        self.assertIn('core_env::on("CRISPEMBED_MOEDERN_Q8_0")',engine)


if __name__=='__main__':unittest.main()
