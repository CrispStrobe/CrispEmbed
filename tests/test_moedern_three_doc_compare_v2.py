#!/usr/bin/env python3
"""Tiny authored adapters and input-authentication controls; no real arrays."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('retained_compare',ROOT/'tools/ci-heavy/moedern_three_doc_compare_v2.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
spec=importlib.util.spec_from_file_location('frozen_adapter',ROOT/'tools/moedern-native-v2/diff_adapter.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)


class Controls(unittest.TestCase):
    def test_real_hyphen_rejection_and_safe_lossless_ids(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);path=root/'owned.f32';data=struct.pack('<4f',1.,-0.,-1.,2.);path.write_bytes(data)
            item=dict(path=path,shape=[2,2],sha256=hashlib.sha256(data).hexdigest())
            with self.assertRaisesRegex(ValueError,'Unsafe/duplicate'):adapter.write_pair(root/'bad-a.gguf',root/'bad-b.gguf',[dict(name='doc-24726600.all',original=item,native=item)])
            pairs=[dict(name=w.safe_name(key,kind),original=item,native=item)for key in w.KEYS for kind in w.KINDS]
            result=adapter.write_pair(root/'a.gguf',root/'b.gguf',pairs)
            self.assertEqual(result['original']['sha256'],result['native']['sha256'])
            self.assertEqual(len(result['original']['tensors']),6)
            self.assertTrue(all(t['sha256']==hashlib.sha256(data).hexdigest()for t in result['original']['tensors'].values()))
            self.assertFalse((root/'bad-a.gguf').exists())
            with self.assertRaisesRegex(ValueError,'Unsafe/duplicate'):adapter.write_pair(root/'dup-a.gguf',root/'dup-b.gguf',pairs+[pairs[0]])
        for key,kind in [('d:foreign','all'),(w.KEYS[0],'unknown')]:
            with self.assertRaises(ValueError):w.safe_name(key,kind)

    def test_tamper_nonfinite_and_axes_not_relaxed(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp)/'owned.f32';p.write_bytes(struct.pack('<2f',1.,float('nan')))
            item=dict(path=p,shape=[1,2],sha256=w.sha(p))
            with self.assertRaisesRegex(ValueError,'Nonfinite'):adapter.source_bytes(item)
            p.write_bytes(struct.pack('<2f',1.,0.))
            with self.assertRaisesRegex(ValueError,'SHA256'):adapter.source_bytes(item)
            with self.assertRaisesRegex(ValueError,'shape/bytes'):adapter.source_bytes(dict(item,shape=[2,2],sha256=w.sha(p)))

    def test_exact_failed_service_rejects_before_archive_download(self):
        p=json.loads((ROOT/'docs/moedern-three-doc-compare-v2.json').read_text())
        pin=p['artifacts']['raw'];row=dict(expired=False,workflow_run=dict(id=40825166813,head_sha=w.SOURCE),name=pin['name'],size_in_bytes=pin['bytes'],digest='sha256:'+'0'*64)
        with mock.patch.object(w,'api',return_value=row),mock.patch.object(w.urllib.request,'build_opener')as opener:
            with self.assertRaisesRegex(ValueError,'artifact service identity'):w.download(pin,Path('unused.zip'))
            opener.assert_not_called()

    def test_archive_paths_exact_hashes_and_missing_rejected(self):
        import zipfile
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);z=root/'owned.zip';data=b'authored metadata only';pin=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            with zipfile.ZipFile(z,'w')as f:f.writestr('safe/value.json',data)
            self.assertEqual(w.extract(z,root/'good',{'safe/value.json':pin}),{'safe/value.json':pin})
            with self.assertRaisesRegex(ValueError,'missing'):w.extract(z,root/'missing',{'missing.json':pin})
            with self.assertRaisesRegex(ValueError,'SHA/extent'):w.extract(z,root/'wrong',{'safe/value.json':dict(pin,sha256='0'*64)})
            with zipfile.ZipFile(root/'unsafe.zip','w')as f:f.writestr('../outside',data)
            with self.assertRaisesRegex(ValueError,'unsafe'):w.extract(root/'unsafe.zip',root/'bad',{})

    def test_authored_input_shape_masks_spans_and_expansion_list(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);model=dict(loaded_tensor_sha256={'weights':{str(i):format(i,'064x')for i in range(32)}})
            packet=dict(completed=True,keys=w.KEYS,cases=[],all_raw_inputs_authenticated_before_framework=True,full_original_recapture=False,source_protocol_sha256='b'*64)
            current=dict(completed=False,source=w.SOURCE,arrays=[],error='ValueError: Unsafe/duplicate adapter tensor name',model=model,cases=[])
            old=dict(model=model,arrays={});native={'f32':{'capture':{'files':{}}}}
            for k in w.KINDS:
                old['arrays'][k+'.f32']=dict(bytes=100000,sha256='o'*64)
                native['f32']['capture']['files'][k+'.f32']=dict(bytes=100000,sha256='n'*64)
            for key in w.KEYS:
                cid='doc-'+key.split(':')[1];text='owned';case=dict(key=key,id=cid,role='document',text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest(),ids=[2,30001,3],supplied_mask=[1,1,1],effective_mask=[1,1,1],scoring_mask=[1,0,1],expansion_positions=[],anchors={'original':{},'native':{}})
                obs=dict(key=key,id=cid,anchor={})
                for kind,count in [('all',3),('selected',2)]:
                    data=struct.pack('<'+'f'*(count*128),*([1.]*(count*128)));digest=hashlib.sha256(data).hexdigest()
                    for side,whole in [('original','o'*64),('native','n'*64)]:
                        rel=f'anchors/{side}/{cid}-{kind}.f32';p=root/'three-doc-input'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
                        case['anchors'][side][kind]=dict(path=rel,shape=[count,128],bytes=len(data),sha256=digest,retained_file=kind+'.f32',retained_offset=0,retained_whole_sha256=whole)
                    rel=cid+'/'+kind+'.f32';p=root/'three-doc-out/raw/original'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
                    obs['anchor'][kind]=dict(shape=[count,128],frozen_sha256=digest,current_sha256=digest,current_file=rel)
                packet['cases'].append(case);current['cases'].append(obs)
            w.write(root/'three-doc-input/input-packet.json',packet)
            p=dict(initial_protocol_sha256='b'*64,members={'three-doc-input/input-packet.json':w.identity(root/'three-doc-input/input-packet.json')})
            w.write(root/'three-doc-input/prepare.json',dict(completed=True,source=w.SOURCE,input_packet_sha256=p['members']['three-doc-input/input-packet.json']['sha256']))
            w.write(root/'three-doc-out/raw/original/original.json',current)
            real=w.read
            def read(path):
                if path==ROOT/'tests/results/moedern-colbert/scifact-original-1p5m.json':return old
                if path==ROOT/'tests/results/moedern-colbert/scifact-native-1p5m-v3-native.json':return native
                return real(path)
            with mock.patch.object(w,'read',side_effect=read):
                self.assertEqual(len(w.validate_inputs(root,p)[2]),18)
                broken=copy.deepcopy(packet);broken['cases'][0]['expansion_positions']=[0];w.write(root/'three-doc-input/input-packet.json',broken)
                with self.assertRaisesRegex(ValueError,'expansion-position'):w.validate_inputs(root,p)
                broken=copy.deepcopy(packet);broken['cases'][0]['scoring_mask']=[1,0];w.write(root/'three-doc-input/input-packet.json',broken)
                with self.assertRaisesRegex(ValueError,'three-mask'):w.validate_inputs(root,p)
                w.write(root/'three-doc-input/input-packet.json',packet)
                (root/'three-doc-input'/packet['cases'][0]['anchors']['original']['all']['path']).write_bytes(b'tamper')
                with self.assertRaisesRegex(ValueError,'retained file differs'):w.validate_inputs(root,p)

    def test_initial_public_producers_are_verbatim_and_negative(self):
        base=ROOT/'tests/results/moedern-colbert'
        receipt=json.loads((base/'three-doc-initial-40825166813-review.json').read_text())
        for row in receipt['unchanged_producers'].values():self.assertEqual(w.identity(base/row['path']),{k:row[k]for k in ['bytes','sha256']})
        self.assertFalse(receipt['completed']);self.assertFalse(receipt['new_gold'])
        original=json.loads((base/'three-doc-initial-40825166813-original.json').read_text())
        self.assertEqual(original['arrays'],[])
        self.assertEqual(sum(not x['passed']for c in original['cases']for x in c['anchor'].values()),4)
        self.assertEqual(original['error'],'ValueError: Unsafe/duplicate adapter tensor name')
        self.assertTrue(all(not c['observer_neutral']for c in original['cases']))

    def test_closed_source_and_no_forward_or_framework_entrypoint(self):
        p=json.loads((ROOT/'docs/moedern-three-doc-compare-v2.json').read_text())
        for path,h in p['owned_sha256'].items():self.assertEqual(w.sha(ROOT/path),h,path)
        for path,obj in p['protected_git_objects'].items():self.assertEqual(subprocess.check_output(['git','rev-parse',w.SOURCE+':'+path],cwd=ROOT,text=True).strip(),obj,path)
        source=(ROOT/'tools/ci-heavy/moedern_three_doc_compare_v2.py').read_text()
        self.assertNotIn('import torch',source);self.assertNotIn('sentence_transformers',source);self.assertNotIn('ctypes',source)
        self.assertNotIn('snapshot_download',source);self.assertNotIn('encode_document(',source);self.assertNotIn('encode_tokens(',source)
        workflow=(ROOT/'.github/workflows/moedern-three-doc-compare-v2.yml').read_text()
        self.assertNotIn('cmake',workflow);self.assertNotIn('KAGGLE',workflow);self.assertNotIn('torch==',workflow)
        self.assertIn('if: always()',workflow);self.assertIn('numpy==2.2.6',workflow)
        self.assertEqual(p['forward_count'],0);self.assertFalse(p['model_qualified'])


if __name__=='__main__':unittest.main()
