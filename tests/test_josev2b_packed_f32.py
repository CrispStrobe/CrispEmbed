"""Tiny synthetic GGUF controls; no model, framework, build or payload access."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('josev_pack',ROOT/'tools/pack_josev2b_f32.py')
p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)


def s(value):
    b=value.encode();return struct.pack('<Q',len(b))+b


def fixture(path,architecture,tensors,*,extra=(),offsets=None,tail=b'',padding_byte=0):
    """Independent tiny GGUF fixture writer; float payloads are authored controls."""
    items=[('general.architecture',architecture),('general.license','apache-2.0'),*extra]
    raw=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(items))
    for key,value in items:
        raw+=s(key)+struct.pack('<I',8)+s(value)
    current=0;blobs=[]
    for i,(name,shape,blob)in enumerate(tensors):
        off=(current+31)//32*32 if offsets is None else offsets[i]
        raw+=s(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*shape)+struct.pack('<IQ',0,off)
        blobs.append((off,blob));current=max(current,off+len(blob))
    raw+=bytes([padding_byte])*(-len(raw)%32)
    payload=bytearray(current)
    for off,blob in blobs:payload[off:off+len(blob)]=blob
    raw+=payload;raw+=b'\0'*(-len(raw)%32);path.write_bytes(raw+tail)
    return path


def expected(component):
    result={}
    with component.path.open('rb')as f:
        for name,t in component.tensors.items():
            f.seek(component.data_start+t.offset);b=f.read(t.nbytes)
            result[name]=dict(ne=list(t.ne),bytes=t.nbytes,f32_sha256=hashlib.sha256(b).hexdigest())
    return result


class Packed(unittest.TestCase):
    def test_declared_production_profile_exact442_reviewed_components(self):
        cfg=p.profile();self.assertFalse(cfg['composed_review']['reviewed'])
        self.assertEqual([len(x['tensors'])for x in cfg['components'].values()],[320,122,0])
        self.assertEqual(sum(x['bytes']for c in cfg['components'].values()for x in c['tensors'].values()),7649191184)
        for label,file in [('backbone','backbone-f32.json'),('head','head-f32-conversion.json')]:
            path=ROOT/'tests/results/josev2b'/file;receipt=json.loads(path.read_text());receipt=receipt['conversion']if label=='backbone'else receipt
            self.assertEqual(p.sha_file(path),cfg['components'][label]['schema_receipt_sha256'])
            self.assertEqual(receipt['gguf_sha256'],cfg['components'][label]['sha256'])
            declared={}
            for r in receipt['tensors']:
                shape=r['shape']if label=='backbone'else r['gguf_shape_numpy']
                declared[r['target']]=dict(ne=list(reversed(shape)),bytes=4*__import__('math').prod(shape),f32_sha256=r['f32_sha256'])
            self.assertEqual(declared,cfg['components'][label]['tensors'])
        self.assertEqual(cfg['components']['tokenizer']['bytes'],9954944)

    def test_pending_composed_proof_rejects_before_any_component_access(self):
        with mock.patch.object(p,'source_guard',return_value={}),mock.patch.dict('os.environ',{'GITHUB_ACTIONS':'true'}),mock.patch.object(p,'parse')as parse:
            with self.assertRaisesRegex(ValueError,'review pending'):
                p.pack('missing-backbone','missing-head','missing-tokenizer','missing-review','unused-output')
            parse.assert_not_called()

    def test_small_repack_keeps_every_float_bit_and_metadata_record(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);a=fixture(root/'a.gguf','back',[
                ('l.w',(2,2),struct.pack('<4I',0x80000000,0x3f800000,0x00000001,0x3eaaaaab)),
                ('l.norm',(2,),struct.pack('<2f',.5,1.))],extra=[('general.name','backbone'),('back.provenance','exact')])
            b=fixture(root/'b.gguf','head',[('head.w',(2,),struct.pack('<2f',1.,-2.))],extra=[('general.name','head'),('head.provenance','exact')])
            c=fixture(root/'c.gguf','token',[],extra=[('tokenizer.ggml.model','byte-preserved')])
            components={k:p.parse(v)for k,v in [('backbone',a),('head',b),('tokenizer',c)]}
            extra={k:p.kv(k,8,v)for k,v in [('general.architecture','josev2b_systemone'),('general.name','packed'),('josev2b.packed.format','test-only')]}
            schemas={k:expected(v)for k,v in components.items()}
            with mock.patch.object(p,'CHUNK',8):result=p._repack(components,root/'packed.gguf',extra,schemas,{k:p.sha_file(v.path)for k,v in components.items()})
            actual=p.parse(root/'packed.gguf');self.assertEqual(set(actual.tensors),{'l.w','l.norm','head.w'})
            self.assertEqual(actual.metadata['tokenizer.ggml.model'].raw,components['tokenizer'].metadata['tokenizer.ggml.model'].raw)
            self.assertEqual(actual.metadata['general.architecture'].value,'josev2b_systemone')
            self.assertEqual(actual.metadata['josev2b.packed.backbone.general_name'].value,'backbone')
            self.assertEqual(actual.metadata['josev2b.packed.head.general_name'].value,'head')
            self.assertEqual(len([k for k in actual.metadata if k=='general.license']),1)
            with actual.path.open('rb')as f:
                for name,row in result['tensors'].items():
                    t=actual.tensors[name];self.assertEqual(p.hash_range(f,actual.data_start+t.offset,t.nbytes),row['f32_sha256'])
            self.assertFalse((root/'packed.gguf.partial').exists())
            with self.assertRaisesRegex(ValueError,'Fresh'):p._repack(components,actual.path,extra,schemas)

    def test_parser_rejects_bad_header_duplicate_keys_names_and_large_rank(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);path=root/'bad.gguf'
            for tensors,extra in [([], [('general.license','apache-2.0')]),
                 ([('same',(1,),b'1234'),('same',(1,),b'5678')],[]),
                 ([('rank',(1,1,1,1,1),b'1234')],[]),([('zero',(0,),b'')],[])]:
                fixture(path,'test',tensors,extra=extra)
                with self.assertRaises(ValueError):p.parse(path)
            path.write_bytes(b'GGUF'+struct.pack('<IQQ',3,0,1)+struct.pack('<Q',2**63))
            with self.assertRaisesRegex(ValueError,'string length'):p.parse(path)
            path.write_bytes(b'GGUF'+struct.pack('<IQQ',3,443,0))
            with self.assertRaisesRegex(ValueError,'count'):p.parse(path)

    def test_parser_rejects_overlap_gap_padding_type_and_truncation(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);path=root/'bad.gguf';rows=[('a',(1,),b'1234'),('b',(1,),b'5678')]
            for offsets in ([0,0],[0,64],[1,32]):
                fixture(path,'test',rows,offsets=offsets)
                with self.assertRaises(ValueError):p.parse(path)
            fixture(path,'test',rows,tail=b'\0'*32)
            with self.assertRaises(ValueError):p.parse(path)
            fixture(path,'test',rows,padding_byte=1)
            with self.assertRaisesRegex(ValueError,'padding'):p.parse(path)
            fixture(path,'test',rows);blob=path.read_bytes();path.write_bytes(blob[:-31])
            with self.assertRaises(ValueError):p.parse(path)
            fixture(path,'test',[('huge',(1048576,1048576),b'1234')])
            with self.assertRaisesRegex(ValueError,'extent'):p.parse(path)
            fixture(path,'test',rows)
            # Corrupt actual F32 type field, independently locate first descriptor.
            blob=bytearray(path.read_bytes())
            at=blob.find(s('a'));self.assertGreaterEqual(at,0);struct.pack_into('<I',blob,at+len(s('a'))+4+8,1);path.write_bytes(blob)
            with self.assertRaisesRegex(ValueError,'F32'):p.parse(path)

    def test_metadata_collision_schema_and_tensor_corruption_reject(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);a=p.parse(fixture(root/'a','a',[('a',(1,),b'1234')],extra=[('private.shared','same')]))
            b=p.parse(fixture(root/'b','b',[('b',(1,),b'5678')],extra=[('private.shared','same')]))
            with self.assertRaisesRegex(ValueError,'undeclared duplicate'):p.merged_metadata({'a':a,'b':b},{})
            b=p.parse(fixture(root/'b','b',[('b',(1,),b'5678')]))
            components={'a':a,'b':b};schemas={k:expected(v)for k,v in components.items()}
            bad=deepcopy(schemas);bad['a']['a']['ne']=[2]
            with self.assertRaisesRegex(ValueError,'dimensions'):p._repack(components,root/'out',{},bad)
            bad=deepcopy(schemas);bad['a']['a']['f32_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'bits'):p._repack(components,root/'out2',{},bad)
            self.assertFalse((root/'out2').exists());self.assertTrue((root/'out2.partial').exists())
            with self.assertRaisesRegex(ValueError,'Input changed'):p._repack(components,root/'out3',{},schemas,{'a':'0'*64,'b':p.sha_file(b.path)})
            self.assertFalse((root/'out3').exists())

    def test_primitive_metadata_preserves_f64_and_rejects_nested_count_or_nan(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'header.gguf'
            def write(kind,body):
                raw=b'GGUF'+struct.pack('<IQQ',3,0,1)+s('scores')+struct.pack('<I',kind)+body
                path.write_bytes(raw+b'\0'*(-len(raw)%32))
            numbers=(.1,.19999999999999998)
            write(9,struct.pack('<IQ2d',12,2,*numbers))
            actual=p.parse(path).metadata['scores'];self.assertEqual(actual.value,(12,numbers))
            self.assertIn(struct.pack('<2d',*numbers),actual.raw)
            for kind,body in [(9,struct.pack('<IQ',9,1)),(9,struct.pack('<IQ',12,1000001)),
                              (12,struct.pack('<d',float('nan'))),(7,b'\2')]:
                write(kind,body)
                with self.assertRaises(ValueError):p.parse(path)

    def test_composed_receipt_requires_exact_positive227_source(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'review.json';cfg=p.profile();report=dict(completed=True,source='a'*40,numeric_and_complete_typed_parity=True,compared_stages=227,failures=[])
            path.write_text(json.dumps(report));cfg['composed_review'].update(reviewed=True,source='a'*40,receipt_sha256=p.sha_file(path))
            p.reviewed_composition(cfg,path)
            for key,value in [('completed',False),('source','b'*40),('numeric_and_complete_typed_parity',False),('compared_stages',226),('failures',['x'])]:
                changed=dict(report,**{key:value});path.write_text(json.dumps(changed));cfg['composed_review']['receipt_sha256']=p.sha_file(path)
                with self.assertRaisesRegex(ValueError,'outcome'):p.reviewed_composition(cfg,path)


if __name__=='__main__':unittest.main()
