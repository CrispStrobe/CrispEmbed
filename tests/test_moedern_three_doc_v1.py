#!/usr/bin/env python3
"""Small source, SDK and authored retained-span controls; no optional packages."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/moedern-three-doc'))
import common as c
import prepare as p


class Controls(unittest.TestCase):
    def authored(self):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup);root=Path(td.name);a=root/'old';b=root/'native';out=root/'out';a.mkdir();b.mkdir();out.mkdir()
        documents=[dict(id=k[2:],text='Frozen authored text '+str(i))for i,k in enumerate(c.KEYS)]
        documents += [dict(id='other-'+str(i),text='Authored')for i in range(5180)]
        queries=[dict(id=str(i),text='q',relevant=[])for i in range(300)]
        data=dict(documents=documents,queries=queries)
        selected=[];original=[];native=[];all_bytes=b'';sel_bytes=b''
        for i,key in enumerate(c.KEYS):
            ids=[2,30001,3];all_blob=struct.pack('<384f',*([.125+i*.01]*384));sel_blob=all_blob[512:]
            row=dict(key=key,id=key[2:],role='document',ids=ids,supplied_ids=ids,supplied_mask=[1,1,1],effective_mask=[1,1,1],scoring_mask=[0,1,1],
                all_offset=len(all_bytes),all_bytes=len(all_blob),all_sha256=hashlib.sha256(all_blob).hexdigest(),
                selected_offset=len(sel_bytes),selected_bytes=len(sel_blob),selected_sha256=hashlib.sha256(sel_blob).hexdigest())
            original.append(row);native.append(dict(row,completed=True,typed_exact=True,expansion_positions=[]));all_bytes+=all_blob;sel_bytes+=sel_blob
        original += [dict(key='authored-'+str(i))for i in range(5480)]
        native += [dict(key='authored-'+str(i))for i in range(5480)]
        (a/'data.json').write_text(json.dumps(data));(a/'rows.json').write_text(json.dumps(original));(b/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in native))
        for folder in (a,b):(folder/'all.f32').write_bytes(all_bytes);(folder/'selected.f32').write_bytes(sel_bytes)
        def pins(folder):return {x.name:dict(bytes=x.stat().st_size,sha256=c.sha(x))for x in folder.iterdir()}
        cfg=dict(original=dict(members=pins(a),service={}),native=dict(members=pins(b),service={}),failing_shapes={k:[3,128]for k in c.KEYS})
        return a,b,out,cfg

    def test_three_packet_exact_bytes_ids_masks_and_order(self):
        a,b,out,cfg=self.authored();packet=p.packet(a,b,out,cfg)
        self.assertEqual([r['key']for r in packet['cases']],c.KEYS)
        self.assertEqual(len(packet['cases']),3)
        for row in packet['cases']:
            for side in ('original','native'):
                for pin in row['anchors'][side].values():
                    path=out/pin['path'];self.assertEqual(c.sha(path),pin['sha256']);self.assertEqual(path.stat().st_size,pin['bytes'])
        self.assertNotIn('numpy',sys.modules);self.assertNotIn('torch',sys.modules)

    def test_wrong_whole_sha_and_changed_selected_mask_reject(self):
        for fault in ('sha','mask','missing','duplicate'):
            a,b,out,cfg=self.authored()
            if fault=='sha':cfg['original']['members']['all.f32']['sha256']='0'*64
            else:
                rows=[json.loads(s)for s in(b/'rows.jsonl').read_text().splitlines()]
                if fault=='mask':rows[0]['scoring_mask']=[1,1,1]
                elif fault=='missing':rows[0]['key']='gone'
                else:rows[1]['key']=rows[0]['key']
                (b/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows));cfg['native']['members']['rows.jsonl']=dict(bytes=(b/'rows.jsonl').stat().st_size,sha256=c.sha(b/'rows.jsonl'))
            with self.assertRaises(ValueError):p.packet(a,b,out,cfg)
            self.assertFalse((out/'input-packet.json').exists())

    def test_bounded_span_rejects_negative_and_overflow(self):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup);path=Path(td.name)/'bytes';path.write_bytes(b'1234')
        self.assertEqual(c.span(path,1,2),b'23')
        for offset,count in ((-1,2),(1,4),(0,0)):
            with self.assertRaises(ValueError):c.span(path,offset,count)

    def test_sdk_binds_failed_v3_source_and_every_page(self):
        source='a'*40;submitted='source='+source
        pin=dict(kernel_slug='authored',version_label='v3',status='ERROR',source=source,submitted_script_sha256=hashlib.sha256(submitted.encode()).hexdigest())
        class Request:
            def __str__(self):return json.dumps({'versionLabel':self.version_label})
        api=types.SimpleNamespace(ApiGetKernelRequest=Request,ApiGetKernelSessionStatusRequest=Request,ApiListKernelSessionOutputRequest=Request)
        class Response:
            def __str__(self):return json.dumps(dict(blob=dict(source=submitted)))
        class Client:
            def __init__(self):self.seen=[]
            def get_kernel(self,r):self.seen.append(r.version_label);return Response()
            def get_kernel_session_status(self,r):self.seen.append(r.version_label);return types.SimpleNamespace(status='ERROR')
            def list_kernel_session_output(self,r):
                self.seen.append(r.version_label);last=hasattr(r,'page_token')
                return types.SimpleNamespace(files=[types.SimpleNamespace(file_name='b'if last else'a',url='https://authored.invalid/retained')],next_page_token=None if last else'next')
        client=Client();files,proof=p.index(client,api,'authored-private-owner',pin)
        self.assertEqual(client.seen,['v3']*4);self.assertEqual(proof['status'],'ERROR');self.assertEqual(set(files),{'a','b'})
        bad=dict(pin,status='COMPLETE')
        with self.assertRaises(ValueError):p.index(Client(),api,'authored',bad)

    def test_probe_only_two_declared_cardinality_substitutions(self):
        original=(ROOT/'tests/moedern-text/probe.cpp').read_text();actual=(ROOT/'tests/moedern-three-doc/probe.cpp').read_text()
        expected='// Test-only copy of tests/moedern-text/probe.cpp at93d27614.\n// Only declared case cardinality10→3 and score cardinality3→0 differ.\n'+original.replace('require(cases == 10, "exact frozen ten text cases required");','require(cases == 3, "exact three diagnostic document cases required");').replace('require(pairs == 3 && scores.good(), "exact three raw-text scores required");','require(pairs == 0 && scores.good(), "no score pairs in this three-document diagnostic");')
        self.assertEqual(actual,expected)

    def test_original_before_native_and_old93_before_stage_comparison(self):
        controller=(ROOT/'tools/ci-heavy/moedern_three_doc_v1.py').read_text()
        self.assertLess(controller.index("command('original-three-doc'"),controller.index("for mode in ('plain','observed')"))
        comparison=(ROOT/'tools/moedern-three-doc/compare.py').read_text()
        self.assertLess(comparison.index("result['native93_final_anchor_exact']=True"),comparison.index('mapping=ref.stage_map()'))
        original=(ROOT/'tools/moedern-three-doc/original.py').read_text()
        self.assertLess(original.index("require(measured['all_metrics_agree']"),original.index("report['phase']='neutral-three-document-observers'"))
        self.assertIn('sdpa_count==2',original);self.assertIn('observer.capture_leaves',original)
        cmake=(ROOT/'tests/moedern-three-doc/CMakeLists.txt').read_text()
        self.assertIn('ROOT_BUILD}/ggml/src',cmake);self.assertNotIn('GGML_LLAMAFILE OFF',cmake)
        self.assertIn('-ffp-contract=off ${ARCH_FLAGS}',cmake)

    def test_closed_source_fixed_keys_and_unmapped_stages(self):
        cfg=c.declaration();self.assertEqual(len(cfg['native_stages']),55);self.assertEqual(len(cfg['unmapped_native']),4)
        self.assertEqual(cfg['keys'],c.KEYS)
        self.assertEqual(cfg['gates'],dict(max_absolute=.002,min_row_cosine=.99999))
        self.assertFalse(cfg['full_original_recapture']or cfg['publication_qualified'])
        self.assertEqual(len(cfg['historical_failures']),6)
        self.assertEqual(cfg['native']['service']['status'],'ERROR')


if __name__=='__main__':unittest.main()
