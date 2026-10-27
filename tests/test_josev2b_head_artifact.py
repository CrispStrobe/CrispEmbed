"""Small header/access-source controls; no arrays/frameworks/network loading."""
import hashlib
import contextlib
import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('head_artifact', ROOT/'tools/ci-heavy/josev2b_head_artifact.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Controls(unittest.TestCase):
    def test_exact_reviewed_original_without_optional_imports(self):
        report, _, proof = m.reference(m.PUBLIC, m.RECEIPT, m.SOURCE)
        self.assertEqual(len(report['arrays']), 251)
        self.assertEqual(len(proof), 122)
        for name in ('numpy','torch','gguf','kaggle'):
            self.assertNotIn(name, sys.modules)

    def test_private_origin_requires_frozen_version(self):
        self.assertEqual(m.frozen_origin('owner/kernel/5'), 'owner/kernel/5')
        for value in ('', 'owner/kernel', 'owner/kernel/6', '/kernel/5', 'owner/kernel/latest'):
            with self.assertRaises(ValueError): m.frozen_origin(value)

    def test_every_paginated_request_sends_the_version_without_latest_fallback(self):
        calls=[]
        def request_response(request):
            calls.append(request)
            number=len(calls)
            return SimpleNamespace(files=[SimpleNamespace(file_name=str(number),url='https://example.test')],
                                   next_page_token='page2' if number==1 else None)
        client=SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(
            list_kernel_session_output=request_response)))
        api=SimpleNamespace(build_kaggle_client=lambda:contextlib.nullcontext(client))
        self.assertEqual(set(m.versioned_files(api,'owner/kernel/5',SimpleNamespace)),{'1','2'})
        for request in calls:
            self.assertEqual((request.user_name,request.kernel_slug,request.version_label),('owner','kernel','v5'))
        self.assertEqual(calls[1].page_token,'page2')

    def test_array_members_cannot_escape_or_target_other_payloads(self):
        self.assertEqual(str(m.safe_member('arrays/00001.npy')), 'arrays/00001.npy')
        for value in ('/arrays/0.npy', '../arrays/0.npy', 'arrays/../0.npy', 'head/0.npy', 'arrays/0.gguf', 'arrays/sub/0.npy'):
            with self.assertRaises(ValueError): m.safe_member(value)

    def test_npy_header_checks_dimensions_dtype_order_and_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tiny.npy'
            def data(shape=(2,), dtype='<f4', order=False):
                header = repr(dict(descr=dtype,fortran_order=order,shape=shape)).encode()+b'\n'
                path.write_bytes(b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+b'\0'*8)
                return dict(bytes=path.stat().st_size,dtype='float32',shape=[2])
            row=data(); self.assertEqual(m.array_header(path,row),8)
            for shape,dtype,order in [((1,2),'<f4',False),((2,),'<i4',False),((2,),'<f4',True)]:
                row=data(shape,dtype,order)
                with self.assertRaises(ValueError): m.array_header(path,row)
            row=data();path.write_bytes(path.read_bytes()+b'x')
            with self.assertRaises(ValueError): m.array_header(path,row)

    def test_help_cannot_reach_service_or_payload(self):
        process=subprocess.run([sys.executable, ROOT/'tools/ci-heavy/josev2b_head_artifact.py','--help'],capture_output=True,text=True,timeout=10)
        self.assertEqual(process.returncode,0)
        self.assertIn('--output',process.stdout)

    def test_per_field_occurrences_copy_raw_float_bits_without_batch_shape_rewrite(self):
        hidden=b'\0'*8192
        first=struct.pack('<f',1.)*2048; second=struct.pack('<f',-0.)*2048
        payloads={'h.npy':hidden,'a.npy':first,'b.npy':second}
        fake=SimpleNamespace(load=lambda path,**kwargs:SimpleNamespace(tobytes=lambda **kw:payloads[path.name]))
        fields=[dict(question_id=name,question_type=i,question_span=[0,1],option_spans=[[0,1]])
                for i,name in enumerate(('first','β'))]
        freeze=dict(records=[dict(id='case',encoded=dict(input_ids=[2],questions=fields))])
        events=[dict(case='case',stage='head/hidden_norm/input/0',file='arrays/h.npy',shape=[1,1,2048],dtype='float32',sha256='a'*64,occurrence=0)]
        for i,name in enumerate(('a.npy','b.npy')):
            events.append(dict(case='case',stage='head/option_lexical_projection/input/0',file='arrays/'+name,
                               shape=[1,2048],dtype='float32',sha256='b'*64,occurrence=i))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,{'numpy':fake}):
            root=Path(directory);out=root/'inputs'
            proof=m.export_probe_inputs(root,{},freeze,events,out)
            manifest=__import__('json').loads((out/'case/input.json').read_text())
            self.assertEqual(manifest['schema'],'crispembed-josev-head-input-v1')
            self.assertEqual([r['id'] for r in manifest['fields']],['first','β'])
            self.assertEqual((out/'case/lexical-0.f32').read_bytes(),first)
            self.assertEqual((out/'case/lexical-1.f32').read_bytes(),second)
            self.assertEqual(proof[0]['files']['hidden.f32']['original_shape'],[1,1,2048])
            events[-1]['occurrence']=0
            with self.assertRaisesRegex(ValueError,'occurrence/order'):
                m.export_probe_inputs(root,{},freeze,events,root/'wrong')


if __name__ == '__main__': unittest.main()
