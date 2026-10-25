"""Weight-free guards for the frozen first-FFN attribution protocol."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/ci-heavy'))
import ovisocr2_ffn as audit


class Contract(unittest.TestCase):
    def manifest(self):
        return {'coverage_complete':True,'probe':'decoder','phase':'q4-attention-f16','files':[
            {'stage':s,'file':'native-diff/q4-attention-f16/decoder/'+s+'.f32',
             'original_shape':[1,323,1024],'bytes':1323008,'sha256':h} for s,h in audit.RAW_STAGES.items()]}

    def test_exact_manifest(self):
        self.assertEqual(set(audit.require_rows(self.manifest())),set(audit.RAW_STAGES))

    def test_manifest_rejects_input_or_identity_drift(self):
        for field,value in [('sha256','0'*64),('original_shape',[1,322,1024]),('bytes',4),('file','../x')]:
            m=self.manifest();m['files'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):audit.require_rows(m)
        m=self.manifest();m['files'].append(copy.deepcopy(m['files'][0]))
        with self.assertRaises(ValueError):audit.require_rows(m)
        for field,value in [('coverage_complete',False),('probe','linear'),('phase','q4')]:
            m=self.manifest();m[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):audit.require_rows(m)

    def test_selected_member_rehashed_and_path_checked(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            for malicious in (False,True):
                buf=io.BytesIO()
                with zipfile.ZipFile(buf,'w') as z:
                    z.writestr('x',b'abc')
                    if malicious:z.writestr('../other',b'x')
                buf.seek(0)
                with zipfile.ZipFile(buf) as z:
                    if malicious:
                        with self.assertRaises(ValueError):audit.member(z,'x',Path(d)/'x',hashlib.sha256(b'abc').hexdigest(),3)
                    else:
                        audit.member(z,'x',Path(d)/'x',hashlib.sha256(b'abc').hexdigest(),3)
                        with self.assertRaises(ValueError):audit.member(z,'x',Path(d)/'x','0'*64,3)
                        with self.assertRaises(ValueError):audit.member(z,'x',Path(d)/'x','0'*64,4)

    def test_frozen_receipt_and_no_runtime_modification(self):
        receipt,source=audit.preflight()
        self.assertEqual(source['native_tree'],audit.NATIVE_TREE)
        self.assertFalse(receipt['producer']['quant_numerical']['passed'])


if __name__=='__main__':unittest.main()
