#!/usr/bin/env python3
"""Tiny stdlib/source controls only; no model/framework/network/compiler work."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/moedern-sdpa'))
import sdpa_contract as c
from diff_adapter import write_pair,read_gguf


class Controls(unittest.TestCase):
    def fixtures(self):
        cfg=c.protocol();base={k:None for k in ('source_contract_sha256','panel_sha256','versions',
                'checkpoint_files','inventory','loaded_tensor_sha256','pipeline')}
        old=dict(base,completed=True,github_sha=cfg['old_source'],stage_arrays=[])
        observed=dict(base,completed=False,github_sha=cfg['observer_source'],stage_arrays=[])
        for case,pin in cfg['cases'].items():
            old['stage_arrays'] += copy.deepcopy(list(pin['common'].values())+list(pin['old_outputs'].values()))
            observed['stage_arrays'] += copy.deepcopy(list(pin['common'].values())+list(pin['observer_outputs'].values()))
        return cfg,old,observed

    def test_complete_both_history_and_shared_inputs_never_new_gold(self):
        cfg,old,observed=self.fixtures();proof=c.historical_contract(old,observed,cfg)
        self.assertTrue(proof['common_inputs_exact']);self.assertTrue(proof['observer_continuity_remains_rejected'])
        self.assertEqual(set(cfg['cases']),{'q-de','d-de'})
        self.assertFalse(cfg['qualification']);self.assertFalse(cfg['new_gold']);self.assertFalse(cfg['native_outputs'])

    def test_changed_shape_hash_missing_output_or_promoted_failure_reject(self):
        for mutation in ('shape','hash','output','duplicate','completed','source'):
            cfg,old,new=self.fixtures()
            if mutation=='shape':new['stage_arrays'][0]['shape']=[32,128]
            if mutation=='hash':new['stage_arrays'][0]['sha256']='a'*64
            if mutation=='output':new['stage_arrays'].pop()
            if mutation=='duplicate':new['stage_arrays'].append(new['stage_arrays'][0])
            if mutation=='completed':new['completed']=True
            if mutation=='source':new['github_sha']='0'*40
            with self.assertRaises((ValueError,KeyError)):c.historical_contract(old,new,cfg)

    def test_payload_authentication_rejects_same_extent_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=struct.pack('<2f',1.,2.);(root/'a.f32').write_bytes(data)
            report={'stage_arrays':[dict(case='q',stage='input',path='a.f32',shape=[1,2],dtype='f32',bytes=8,sha256=c.sha(data))]}
            c.checked_arrays(root,report)
            (root/'a.f32').write_bytes(struct.pack('<2f',1.,3.))
            with self.assertRaisesRegex(ValueError,'hash'):c.checked_arrays(root,report)

    def test_actual_kernel_evidence_no_silent_backend_fallback(self):
        math=['aten::scaled_dot_product_attention','aten::_scaled_dot_product_attention_math']
        flash=['aten::scaled_dot_product_attention','aten::_scaled_dot_product_flash_attention_for_cpu']
        self.assertEqual(c.kernel_evidence('math',math),sorted(math))
        self.assertEqual(c.kernel_evidence('cpu-flash',flash),sorted(flash))
        for backend,names in [('default',[]),('math',flash),('cpu-flash',math),('cpu-flash',['aten::_scaled_dot_product_flash_attention'])]:
            with self.assertRaises(ValueError):c.kernel_evidence(backend,names)

    def test_direct_call_observer_keeps_objects_and_restores_after_error(self):
        token=object();mask=object();events=[]
        def original(value,*,attn_mask):
            self.assertIs(value,token);self.assertIs(attn_mask,mask)
            return token
        module=SimpleNamespace(sdpa=original)
        def observer(*args,**kwargs):events.append((args,kwargs));return original(*args,**kwargs)
        with c.replace_call(module,'sdpa',observer):self.assertIs(module.sdpa(token,attn_mask=mask),token)
        self.assertIs(module.sdpa,original);self.assertIs(events[0][0][0],token)
        with self.assertRaisesRegex(ValueError,'observer'):
            with c.replace_call(module,'sdpa',lambda *a,**kw:(_ for _ in ()).throw(ValueError('observer'))):
                module.sdpa(token,attn_mask=mask)
        self.assertIs(module.sdpa,original)

    def test_profile_or_observer_drift_including_signed_zero_rejects(self):
        a=struct.pack('<2f',0.,1.);c.exact_control(a,a,a,a)
        for bad in (struct.pack('<2f',-0.,1.),struct.pack('<2f',0.,1.000001)):
            with self.assertRaises(ValueError):c.exact_control(a,a,bad)

    def test_actual_layout_records_keep_broadcast_strides_without_claiming_history(self):
        tensor=SimpleNamespace(shape=(1,2,32,64),stride=lambda:(2048,0,64,1),storage_offset=lambda:0,
            is_contiguous=lambda:False,dtype='torch.float32',device='cpu',data_ptr=lambda:128)
        row=c.tensor_layout(tensor)
        self.assertEqual(row['stride'],[2048,0,64,1]);self.assertFalse(row['contiguous'])
        self.assertEqual(row['pointer_alignment_mod64'],0)
        self.assertIn('historical strides were not captured',row['provenance'])

    def test_row_metrics_and_lossless_adapters_do_not_flatten_shapes(self):
        a=struct.pack('<6f',1.,0.,0.,1.,1.,0.);b=struct.pack('<6f',1.,0.,0.,1.,0.,1.)
        m=c.metrics(a,b,[3,2]);self.assertFalse(m['passed']);self.assertEqual(m['min_row_cosine'],0.)
        self.assertAlmostEqual(m['rms'],(2/6)**.5);self.assertEqual(m['original_l2'],m['native_l2'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a').write_bytes(a);(root/'b').write_bytes(b)
            def spec(p):return dict(path=root/p,sha256=c.sha((root/p).read_bytes()),shape=[3,2])
            pairs=[dict(name='stage',original=spec('a'),native=spec('b'))]
            write_pair(root/'a.gguf',root/'b.gguf',pairs)
            self.assertEqual(read_gguf(root/'a.gguf')['stage']['sha256'],c.sha(a))
            pairs[0]['native']['shape']=[1,3,2]
            with self.assertRaisesRegex(ValueError,'shapes differ'):write_pair(root/'c.gguf',root/'d.gguf',pairs)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):c.metrics(struct.pack('<f',float('nan')),struct.pack('<f',0.),[1])

    def test_historical_source_files_and_gates_stay_byte_identical(self):
        cfg=c.protocol()
        for name,sha in cfg['protected'].items():
            self.assertEqual(c.sha(subprocess.check_output(['git','show',cfg['base_source']+':'+name],cwd=ROOT)),sha)
            self.assertEqual(c.sha((ROOT/name).read_bytes()),sha)
        self.assertEqual(cfg['gates']['max_absolute'],.002);self.assertEqual(cfg['gates']['minimum_row_cosine'],.99999)

    def test_original_first_barrier_and_standalone_actual_ref_source(self):
        path=ROOT/'tools/ci-heavy/moedern_sdpa_reproducibility.py';tree=ast.parse(path.read_text())
        top=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(n) or 'numpy' in ast.unparse(n) for n in top))
        source=path.read_text();self.assertLess(source.index('retained(scratch)\n    checkpoint'),source.index('    import numpy as np'))
        self.assertIn('backends=[SDPBackend.MATH if',source);self.assertIn('profile.key_averages()',source)
        cpp=(ROOT/'tests/moedern-sdpa/diff.cpp').read_text();self.assertIn('original.compare(name, b.first, b.second, 0)',cpp)
        self.assertIn('core_util::clean_exit',cpp);self.assertNotIn('r.is_pass()',cpp)
        cmake=(ROOT/'tests/moedern-sdpa/CMakeLists.txt').read_text();self.assertNotIn('add_subdirectory',cmake)

    def test_dispatcher_extends_existing_phase_union_and_workflow_is_bounded(self):
        base=subprocess.check_output(['git','show',c.protocol()['base_source']+':.github/workflows/ovisocr2-metadata.yml'],cwd=ROOT,text=True)
        now=(ROOT/'.github/workflows/ovisocr2-metadata.yml').read_text()
        for text in (base,now):self.assertIn('moedern-original-1p5m',text)
        import re
        parse=lambda text:set(x.strip() for x in re.search(r'options: \[([^]]+)\]',text).group(1).split(','))
        self.assertTrue(parse(base)<=parse(now));self.assertIn('moedern-sdpa-reproducibility',parse(now))
        workflow=(ROOT/'.github/workflows/moedern-sdpa-reproducibility.yml').read_text()
        self.assertIn('timeout-minutes: 20',workflow);self.assertIn('if: always()',workflow)
        self.assertLess(workflow.index('--phase prepare'),workflow.index('pip install torch'))


if __name__=='__main__':unittest.main()
