"""Small public-client receipt guards; no SDK, package install, build or models."""
import importlib.util,json,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('public_clients',ROOT/'tools/moedern-colbert/public_clients.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Controls(unittest.TestCase):
    def payload(self):return '\n'.join(json.dumps(x)for x in [dict(kind='case',id='one',all_vectors=[1.,0.],selected_vectors=[1.,0.]),dict(kind='score',id='pair',score=1.)])
    def test_no_local_client_or_sdk_execution(self):
        with patch.dict(m.os.environ,{},clear=True),patch.object(m.Path,'is_dir',return_value=False),patch.object(m.subprocess,'run')as process,patch.object(m.urllib.request,'urlopen')as transfer:
            with self.assertRaisesRegex(ValueError,'hosted'):m.execute('27M','unused','unused','unused','unused')
            with self.assertRaisesRegex(ValueError,'hosted'):m.dart_sdk('unused')
            process.assert_not_called();transfer.assert_not_called()
    def test_actual_output_inventory_and_nonfinite_reject(self):
        p=self.payload();self.assertEqual(len(m.parse_outputs(p,['one'],['pair'])),2)
        for bad in [p+'\n'+p,p.splitlines()[0],p.replace('"one"','"other"'),p.replace('1.0','NaN'),p.replace('1.0','1e999')]:
            with self.assertRaises(ValueError):m.parse_outputs(bad,['one'],['pair'])
    def test_signed_zero_and_cli_borrowed_scope(self):
        a=m.parse_outputs(self.payload(),['one'],['pair']);b=m.parse_outputs(self.payload().replace('0.0','-0'),['one'],['pair'])
        self.assertEqual(a[0]['all_vectors'],b[0]['all_vectors'])
        self.assertNotEqual(m.canonical_bits(a),m.canonical_bits(b))
        # CLI has no actual all-vector stream; its borrowed field is excluded.
        b=m.parse_outputs(self.payload(),['one'],['pair']);b[0]['all_vectors']=[99.,0.]
        self.assertEqual(m.canonical_bits(a,include_all=False),m.canonical_bits(b,include_all=False))
        b[0]['selected_vectors']=[-0.,1.]
        self.assertNotEqual(m.canonical_bits(a,include_all=False),m.canonical_bits(b,include_all=False))
    def test_all_models_and_exact_dart_identity(self):
        self.assertEqual(set(m.PROFILES),{'1.5M','27M','115M'})
        self.assertEqual(m.DART_SHA,'ea864bc64df30a6b8bdf30b2e32550f7717d9a890de8f40293aeabb924fe232b')
        self.assertTrue(m.DART_URL.endswith('/3.13.5/sdk/dartsdk-linux-x64-release.zip'))
if __name__=='__main__':unittest.main()
