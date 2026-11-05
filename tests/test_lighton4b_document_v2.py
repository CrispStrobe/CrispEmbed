#!/usr/bin/env python3
"""Synthetic v2 packaging and realRef consumption; no original model forward."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
v2=load('document_v2','tools/lighton4b_document_v2.py')
fixtures=load('http_fixture','tests/test_lighton4b_sharded_transport.py')
gguf=load('tiny_archive_fixture','tests/test_lighton4b_document_parity.py')


def prepare(root,remote):
    source=root/'source';fixtures.fixture(source)
    names=[name for name,_ in v2.ordered_stages(1)if not name.startswith('generated.')]
    gguf.tiny_gguf(source/'official-delta.gguf',[(name,[index/100+1])for index,name in enumerate(names)])
    result=v2.transport.publish(source,remote,'synthetic-v2-session',root/'publication.json',part_bytes=4096)
    remote.get('synthetic-v2-session/manifest.json',result['revision'],root/'transport.json')
    original=v2.package(source,root/'transport.json','original',root/'original.json')
    native=v2.package(source,root/'transport.json','native',root/'native.json')
    return source,result,original,native


class Controls(unittest.TestCase):
    def test_exact_v2_coverage_and_unearned_real_readiness_rejected(self):
        with tempfile.TemporaryDirectory()as folder,fixtures.service()as(remote,state):
            root=Path(folder);source,r,a,b=prepare(root,remote)
            self.assertEqual(len(a['stages']),1037);ready=v2.readiness(a)
            self.assertTrue(ready['synthetic_packaging_valid']);self.assertFalse(ready['actual_original_output_ready'])
            self.assertFalse(ready['actual_full_vocabulary_geometry'])
            for change in('missing','duplicate','reorder','role','readiness','shape'):
                p=copy.deepcopy(a)
                if change=='missing':p['stages'].pop()
                elif change=='duplicate':p['stages'].append(p['stages'][0])
                elif change=='reorder':p['stages'][0],p['stages'][1]=p['stages'][1],p['stages'][0]
                elif change=='role':p['role']='original-capture'
                elif change=='readiness':p['actual_original_output_ready']=True
                else:p['stages'][-1]['shape']=[1,248320]
                with self.subTest(change=change),self.assertRaises(ValueError):v2.validate(p)
    def test_source_hash_stage_missing_and_wrong_model_claim_reject(self):
        with tempfile.TemporaryDirectory()as folder,fixtures.service()as(remote,state):
            root=Path(folder);source,r,a,b=prepare(root,remote)
            with self.assertRaisesRegex(ValueError,'not admitted'):v2.package('/missing-model-output',root/'transport.json','original',root/'no.json',role='original-capture')
            path=next((source/'generated').glob('*.f32'));path.write_bytes(b'bad')
            with self.assertRaises(ValueError):v2.package(source,root/'transport.json','original',root/'bad.json')
    def test_lazy_remote_stage_reads_verify_slice_and_part_hashes(self):
        with tempfile.TemporaryDirectory()as folder,fixtures.service()as(remote,state):
            root=Path(folder);source,r,a,b=prepare(root,remote)
            reader=v2.RemoteReader(remote,'synthetic-v2-session',r['revision'],r['manifest_sha256'],root/'cache')
            local=v2.LocalReader(source)
            for row in(a['stages'][0],a['stages'][500],a['stages'][-1]):
                self.assertEqual(reader.read(row['file'],row['offset'],row['bytes']),local.read(row['file'],row['offset'],row['bytes']))
            self.assertLessEqual(len(list((root/'cache').glob('part.bin'))),1)
            corrupt=next(iter(reader.parts));state.versions[r['revision']]['synthetic-v2-session/'+corrupt]=b'broken'
            reader.current=None
            with self.assertRaises(ValueError):reader.read('generated/generated.0.cache.0.conv_states.f32',0,4)
    @unittest.skipUnless(os.environ.get('LIGHTON4B_DOCUMENT_DIFF'),'hosted comparator build required')
    def test_actual_ref_remote_consumption_and_exact_decoded_pair(self):
        with tempfile.TemporaryDirectory()as folder,fixtures.service()as(remote,state):
            root=Path(folder);source,r,a,b=prepare(root,remote)
            readers=[v2.RemoteReader(remote,'synthetic-v2-session',r['revision'],r['manifest_sha256'],root/name)for name in('original-cache','native-cache')]
            report=v2.consume(a,b,*readers,os.environ['LIGHTON4B_DOCUMENT_DIFF'],root/'report')
            self.assertEqual(report['status'],'passed');self.assertEqual(report['compared_stages'],1037)
            self.assertFalse(report['actual_original_output_ready'])
            b=copy.deepcopy(b);b['decoded']='different independent text'
            report=v2.consume(a,b,v2.LocalReader(source),v2.LocalReader(source),os.environ['LIGHTON4B_DOCUMENT_DIFF'],root/'decoded-negative')
            self.assertEqual(report['status'],'failed');self.assertFalse(report['decoded_and_ids_exact'])
            b=copy.deepcopy(b);b['stages'][0]['sha256']='0'*64
            with self.assertRaises(ValueError):v2.consume(a,b,v2.LocalReader(source),v2.LocalReader(source),os.environ['LIGHTON4B_DOCUMENT_DIFF'],root/'hash-negative')
            self.assertEqual(json.loads((root/'hash-negative/consumption.json').read_text())['status'],'failed')
    def test_all_v1_capture_and_real_admissions_remain_closed(self):
        self.assertEqual(v2.ADMITTED_ORIGINAL_PRODUCERS,());self.assertEqual(v2.ADMITTED_NATIVE_PRODUCERS,())
        p=v2.v1.protocol();self.assertEqual(p['comparison']['max_bundle_bytes'],2*2**30)
        self.assertEqual(p['execution_admission']['original_producer_sources'],[])
        self.assertEqual(p['execution_admission']['native_producer_sources'],[])
        adapter=load('closed_adapter','tools/lighton4b_original_document.py');self.assertFalse(adapter.CAPTURE_TRANSPORT_ADMITTED)


if __name__=='__main__':unittest.main()
