#!/usr/bin/env python3
"""Independent HTTP service fixture; no external provider or model is used."""
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import tempfile
import threading
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('transport',ROOT/'tools/lighton4b_sharded_transport.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


@contextmanager
def service():
    state=SimpleNamespace(objects={},versions={},puts=0,gets=0,fail_at=None)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_PUT(self):
            state.puts+=1
            if state.puts==state.fail_at:self.send_error(500);return
            name=self.path[1:];data=self.rfile.read(int(self.headers['Content-Length']))
            state.objects[name]=data;revision=f'{state.puts:040x}';state.versions[revision]=dict(state.objects)
            self.send_response(200);self.end_headers();self.wfile.write(revision.encode())
        def do_GET(self):
            state.gets+=1
            if self.path=='/files':data=json.dumps(sorted(state.objects)).encode()
            else:
                revision,name=self.path[1:].split('/',1)
                if name not in state.versions.get(revision,{}):self.send_error(404);return
                data=state.versions[revision][name]
            self.send_response(200);self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    class Remote:
        base='http://127.0.0.1:'+str(server.server_port)
        def prepare(self,prefix):
            with urllib.request.urlopen(self.base+'/files',timeout=5)as response:names=json.load(response)
            t.require(not any(n.startswith(prefix+'/')for n in names),'remote prefix exists')
        def put(self,name,path):
            request=urllib.request.Request(self.base+'/'+name,data=Path(path).read_bytes(),method='PUT')
            with urllib.request.urlopen(request,timeout=5)as response:return response.read().decode()
        def get(self,name,revision,destination):
            with urllib.request.urlopen(self.base+'/'+revision+'/'+name,timeout=5)as source,Path(destination).open('wb')as target:
                shutil.copyfileobj(source,target,1024)
    try:yield Remote(),state
    finally:server.shutdown();server.server_close();thread.join(timeout=5)


def fixture(root):
    generated=root/'generated';generated.mkdir(parents=True);records={}
    for index,name in enumerate(sorted(t.fields(1))):
        data=struct.pack('<f',index/100);path=generated/(name+'.f32');path.write_bytes(data)
        records[name]=dict(file=path.name,shape=[1],bytes=4,sha256=hashlib.sha256(data).hexdigest())
    observer=dict(status='passed',synthetic_control=True,original_forward_observed=False,eos_completed=True,cap_hit=False,
        generated_ids=[248046],positions=1,observer_off_ids_and_all_logits_exact=True,raw_logits_equal_generate_scores=True,records=records)
    (generated/'observer.json').write_text(json.dumps(observer))
    return observer


class Controls(unittest.TestCase):
    def publish(self,root,remote):return t.publish(root/'source',remote,'synthetic-session-0001',root/'publication.json',part_bytes=1024)
    def test_independent_http_readback_whole_files_and_hashes(self):
        with tempfile.TemporaryDirectory()as folder,service()as(remote,state):
            root=Path(folder);fixture(root/'source');expected={str(p.relative_to(root/'source')):p.read_bytes()for p in(root/'source').rglob('*')if p.is_file()}
            receipt=self.publish(root,remote);self.assertEqual(receipt['status'],'passed');self.assertGreater(len(receipt['parts']),1)
            result=t.restore(remote,receipt['prefix'],receipt['revision'],receipt['manifest_sha256'],root/'independent')
            self.assertEqual(result['status'],'passed');self.assertEqual(result['files'],68)
            for name,data in expected.items():
                observed=(root/'independent/files'/name).read_bytes();self.assertEqual(observed,data)
                self.assertEqual(hashlib.sha256(observed).hexdigest(),hashlib.sha256(data).hexdigest())
            self.assertGreater(state.gets,2*len(receipt['parts']))
            with self.assertRaises(ValueError):t.publish(root/'source',remote,receipt['prefix'],root/'again.json',part_bytes=1024)
    def test_source_hash_and_complete_fields_reject_before_upload(self):
        for corruption in('bytes','missing','eos','original-label'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory()as folder,service()as(remote,state):
                root=Path(folder);observer=fixture(root/'source')
                if corruption=='bytes':next((root/'source/generated').glob('*.f32')).write_bytes(b'bad')
                elif corruption=='missing':observer['records'].pop(next(iter(observer['records'])))
                elif corruption=='eos':observer['generated_ids']=[1]
                else:observer['original_forward_observed']=True
                if corruption!='bytes':(root/'source/generated/observer.json').write_text(json.dumps(observer))
                with self.assertRaises(ValueError):self.publish(root,remote)
                self.assertEqual(state.puts,0);self.assertEqual(json.loads((root/'publication.json').read_text())['status'],'failed')
    def test_partial_upload_retains_negative_without_final_manifest(self):
        with tempfile.TemporaryDirectory()as folder,service()as(remote,state):
            root=Path(folder);fixture(root/'source');state.fail_at=2
            with self.assertRaises(urllib.error.HTTPError):self.publish(root,remote)
            self.assertFalse(any(n.endswith('manifest.json')for n in state.objects))
            report=json.loads((root/'publication.json').read_text());self.assertEqual(report['status'],'failed');self.assertEqual(len(report['parts']),1)
    def test_corrupt_truncated_and_missing_remote_parts_fail_independently(self):
        for mode in('corrupt','truncated','missing'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as folder,service()as(remote,state):
                root=Path(folder);fixture(root/'source');r=self.publish(root,remote);objects=state.versions[r['revision']]
                name=r['prefix']+'/'+r['parts'][0]['file'];raw=objects[name]
                if mode=='corrupt':objects[name]=bytes([raw[0]^1])+raw[1:]
                elif mode=='truncated':objects[name]=raw[:-1]
                else:del objects[name]
                with self.assertRaises((ValueError,urllib.error.HTTPError)):t.restore(remote,r['prefix'],r['revision'],r['manifest_sha256'],root/'failed')
                self.assertEqual(json.loads((root/'failed/readback.json').read_text())['status'],'failed')
    def test_noncanonical_overlap_path_and_wrong_manifest_hash_reject(self):
        for mode in('overlap','path','hash'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as folder,service()as(remote,state):
                root=Path(folder);fixture(root/'source');r=self.publish(root,remote);key=r['prefix']+'/manifest.json'
                m=json.loads(state.versions[r['revision']][key])
                if mode=='overlap':m['segments'][0]['file_offset']+=1
                elif mode=='path':m['files'][0]['file']='../escape'
                raw=json.dumps(m).encode();state.versions[r['revision']][key]=raw
                expected='0'*64 if mode=='hash'else hashlib.sha256(raw).hexdigest()
                with self.assertRaises(ValueError):t.restore(remote,r['prefix'],r['revision'],expected,root/'failed')
                self.assertFalse((root/'escape').exists())
    def test_original_admission_v1_and_resource_limits_remain_closed(self):
        self.assertFalse(t.ORIGINAL_TRANSPORT_ADMITTED)
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)
            with self.assertRaisesRegex(ValueError,'not admitted'):t.source_files(root,'original-capture')
        p=json.loads((ROOT/'docs/lighton4b-document-parity-v1.json').read_text())
        self.assertEqual(p['comparison']['max_bundle_bytes'],2*2**30)
        self.assertEqual(p['execution_admission']['original_producer_sources'],[])
        self.assertEqual(p['execution_admission']['native_producer_sources'],[])
        for size in(0,True,t.MAX_PART+1):
            with self.assertRaises(ValueError):t.layout([dict(file='safe',bytes=4)],size)


if __name__=='__main__':unittest.main()
