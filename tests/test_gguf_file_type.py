#!/usr/bin/env python3
"""Bounded authored GGUF migration fixtures; no model/framework/compiler."""
import copy
import importlib.util
import os
from unittest.mock import patch
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import gguf_file_type_canonicalize as c
import gguf_file_type_contract as contract
import gguf_file_type_public_bridge as public


def fixture(path,*,file_type=12,tensor_type=12,duplicate=False):
    text=lambda s:struct.pack('<Q',len(s.encode()))+s.encode()
    rows=[('general.architecture',8,'authored_fixture'),('profile',8,'source-frozen-v1'),
          ('general.file_type',4,file_type),('general.quantization_version',4,2)]
    if duplicate:rows.append(('general.file_type',4,file_type))
    b=b'GGUF'+struct.pack('<IQQ',3,2,len(rows))
    for key,kind,value in rows:b+=text(key)+struct.pack('<I',kind)+(text(value)if kind==8 else struct.pack('<I',value))
    payload=struct.pack('<ee',1.,0.)+bytes(140)
    b+=text('matrix')+struct.pack('<IQQIQ',2,256,1,tensor_type,0)
    b+=text('norm')+struct.pack('<IQIQ',1,1,0,160)
    path.write_bytes(b+bytes((-len(b))%32)+payload+bytes(16)+struct.pack('<f',1.))


def descriptor(source,report):
    p=c.parse(source)
    with source.open('rb')as f:rows=[dict(name=n,semantic_shape=r['stored_shape'],stored_shape=r['stored_shape'],type=r['type'],bytes=r['bytes'],sha256=c.span(f,r['data_offset'],r['bytes']))for n,r in p['tensors'].items()]
    receipt=dict(output_sha256=c.sha(source),output_bytes=source.stat().st_size,recipe='authored')
    report.write_text(json.dumps(receipt)+'\n')
    return dict(schema='crispembed-gguf-file-type-input-v1',reviewed=True,family='josev',recipe='authored',producer_source='a'*40,
        producer_report=dict(bytes=report.stat().st_size,sha256=c.sha(report)),source_model=dict(bytes=source.stat().st_size,sha256=c.sha(source)),
        report_model_sha256_path=['output_sha256'],report_model_bytes_path=['output_bytes'],report_recipe_path=['recipe'],
        architecture='authored_fixture',profile_markers={'profile':'source-frozen-v1'},inventory=rows)


class Controls(unittest.TestCase):
    def authored(self):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup);root=Path(td.name)
        source=root/'source.gguf';out=root/'canonical.gguf';report=root/'producer.json';fixture(source)
        return source,out,report,descriptor(source,report)

    def test_exact_four_byte_repair_and_immutable_source(self):
        source,out,report,cfg=self.authored();before=source.read_bytes()
        result=c.canonicalize(source,out,cfg,report);after=out.read_bytes();offset=result['allowed_changes'][0]['offset']
        self.assertEqual(before,source.read_bytes());self.assertEqual(len(before),len(after))
        self.assertEqual(after,before[:offset]+struct.pack('<I',14)+before[offset+4:])
        self.assertEqual(result['schema'],c.SCHEMA);self.assertTrue(result['payloads_byte_exact'])
        self.assertFalse(result['numerical_qualified']or result['publication_qualified'])
        self.assertEqual(c.parse(source)['tensors'],c.parse(out)['tensors'])

    def test_identity_and_recipe_mismatch_before_copy(self):
        for field in ('source','report','recipe','review','profile','inventory'):
            source,out,report,cfg=self.authored()
            if field=='source':cfg['source_model']['sha256']='0'*64
            elif field=='report':cfg['producer_report']['sha256']='0'*64
            elif field=='recipe':cfg['recipe']='other'
            elif field=='review':cfg['reviewed']=False
            elif field=='profile':cfg['profile_markers']['profile']='other'
            else:cfg['inventory'][0]['sha256']='0'*64
            with self.assertRaises(ValueError):c.canonicalize(source,out,cfg,report)
            self.assertFalse(out.exists()or out.with_name(out.name+'.partial').exists())

    def test_legacy14_is_not_blindly_remapped(self):
        source,out,report,cfg=self.authored();fixture(source,file_type=14);cfg=descriptor(source,report)
        with self.assertRaisesRegex(ValueError,'legacy Q4'):c.canonicalize(source,out,cfg,report)
        self.assertFalse(out.exists())

    def test_duplicate_key_unknown_type_shape_and_truncation(self):
        source,out,report,cfg=self.authored()
        fixture(source,duplicate=True)
        with self.assertRaisesRegex(ValueError,'duplicate'):c.parse(source)
        fixture(source,tensor_type=14)
        with self.assertRaisesRegex(ValueError,'storage'):c.parse(source)
        fixture(source);source.write_bytes(source.read_bytes()[:-1])
        with self.assertRaises(ValueError):c.parse(source)

    def test_destination_existing_alias_and_symlink_rejected(self):
        for mode in ('same','existing','hardlink','symlink'):
            source,out,report,cfg=self.authored();before=source.read_bytes()
            if mode=='same':out=source
            elif mode=='existing':out.write_bytes(b'owned-other-output')
            elif mode=='hardlink':out.hardlink_to(source)
            else:out.symlink_to(source)
            with self.assertRaisesRegex(ValueError,'fresh destination'):c.canonicalize(source,out,cfg,report)
            self.assertEqual(source.read_bytes(),before)

    def test_changed_descriptor_axes_type_extent_reject(self):
        for field,value in [('stored_shape',[512,1]),('type',0),('bytes',288)]:
            source,out,report,cfg=self.authored();cfg['inventory'][0][field]=value
            with self.assertRaisesRegex(ValueError,'descriptor'):c.canonicalize(source,out,cfg,report)

    def test_source_closure_and_six_baseline_deltas_are_explicit(self):
        cfg=contract.validate(json.loads(contract.PATH.read_text()))
        self.assertEqual(set(cfg['preserved_moedern_delta']),contract.MOE)
        for name in ('src/qwen35_decoder.cpp','tests/josev-q4k/CMakeLists.txt'):
            bad=copy.deepcopy(cfg);bad['protected_sha256'].pop(name)
            with self.assertRaisesRegex(ValueError,'closure'):contract.validate(bad)
        bad=copy.deepcopy(cfg);bad['existing_delta'].pop('src/ultrabert_profile.h')
        with self.assertRaisesRegex(ValueError,'deltas'):contract.validate(bad)

    def test_actual_capi_pair_logic_gate_recovery_and_cleanup_with_fake_functions(self):
        source,out,report,cfg=self.authored();library=source;closed='b'*40
        freeze={'records':[{'request':{'authored':n}}for n in range(4)]}
        gates=('CRISPEMBED_JOSEV_QUANT','CRISPEMBED_QWEN35_QUANT')
        class Function:
            def __init__(self,fn):self.fn=fn
            def __call__(self,*args):return self.fn(*args)
        class Library:
            def __init__(self,drift=False):
                self.active={};self.freed=[];self.next=0;self.drift=drift
                self.crispembed_systemone_init=Function(self.init)
                self.crispembed_systemone_free=Function(self.free)
                self.crispembed_systemone_run_json=Function(self.run)
                self.crispembed_systemone_error=Function(lambda _:b'authored error')
            def init(self,path,threads):
                if any(os.environ.get(g)in(None,'','0')for g in gates):return None
                self.next+=1;self.active[self.next]=path;return self.next
            def free(self,ctx):self.freed.append(ctx);self.active.pop(ctx)
            def run(self,ctx,request):
                if request is None or any(os.environ.get(g)in(None,'','0')for g in gates):return None
                return b'{"answer":2}'if self.drift and self.active[ctx].endswith(b'canonical')else b'{"answer":1}'
        for drift in (False,True):
            lib=Library(drift)
            with patch.dict(os.environ,{g:'1'for g in gates},clear=True),patch.object(public.ctypes,'CDLL',return_value=lib),patch.object(public.subprocess,'check_output',return_value='plain'):
                if drift:
                    with self.assertRaisesRegex(ValueError,'response bytes'):public.compare(library,dict(source=closed,sha256=c.sha(library)),closed,'legacy','canonical',freeze)
                else:
                    result=public.compare(library,dict(source=closed,sha256=c.sha(library)),closed,'legacy','canonical',freeze)
                    self.assertTrue(result['all_public_capi_response_bytes_exact'])
                    self.assertEqual(len(result['arms']['canonical']['gate_controls']),6)
                self.assertFalse(lib.active);self.assertEqual(len(lib.freed),2)
                self.assertTrue(all(os.environ[g]=='1'for g in gates))

    def test_bridge_authenticates_before_forwards_and_keeps_debt(self):
        source=(ROOT/'tools/ci-heavy/gguf_file_type_bridge.py').read_text()
        self.assertLess(source.index('canonicalizer.descriptor('),source.index("control=execute('f32'"))
        self.assertLess(source.index("control=execute('f32'"),source.index('canonicalizer.canonicalize('))
        self.assertIn('old_numeric_results_replaced=False',source)
        self.assertIn('all_public_capi_response_bytes_exact=True',source)
        self.assertIn("comparison['compared_stages']==227",source)
        methods_source=(ROOT/'tools/ci-heavy/gguf_file_type_controls.py').read_text()
        self.assertIn('actual_cli_methods=13',methods_source)

    def test_writer_has_explicit_standard_and_legacy_policy(self):
        source=(ROOT/'tools/quantize.cpp').read_text();header=(ROOT/'src/core/gguf_file_type.h').read_text()
        self.assertIn('--legacy-file-type-header',source);self.assertIn('core_gguf_file_type::standard(ftype)',source)
        self.assertIn('constexpr uint32_t q4_k_s = 14',header);self.assertIn('constexpr uint32_t legacy_q4_k = 12',header)
        for name in ('josev2b_packed_profile.h','ultrabert_profile.h','moedern_q4k_profile.h'):
            self.assertIn('core_gguf_file_type::', (ROOT/'src'/name).read_text())
        self.assertNotIn('numpy',sys.modules);self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
