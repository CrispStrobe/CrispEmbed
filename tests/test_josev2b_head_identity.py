#!/usr/bin/env python3
"""Weight-free original-head audit controls with tiny tensor doubles only."""
import importlib.util
import math
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('josev_head_audit',ROOT/'tools/dump_josev2b_reference.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Tensor:
    def __init__(self,values,dtype='torch.float32',shape=None):
        self.values=list(values);self.dtype=dtype;self.shape=(len(values),) if shape is None else shape
    def detach(self):return self
    def is_floating_point(self):return self.dtype in ('torch.float32','torch.bfloat16','torch.float16')
    def to(self,**kwargs):return Tensor(self.values,kwargs.get('dtype',self.dtype),self.shape)
    def contiguous(self):return self
    def numpy(self):return self
    def astype(self,*a,**kw):return self
    def tobytes(self):return struct.pack('<'+'f'*len(self.values),*self.values)
    def numel(self):return len(self.values)

class Open:
    def __init__(self,values):self.values=values
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def keys(self):return self.values.keys()
    def get_tensor(self,name):return self.values[name]

class HeadIdentityTests(unittest.TestCase):
    def audit(self,original,actual):
        with tempfile.TemporaryDirectory() as d:
            checkpoint=Path(d)/'head';checkpoint.write_bytes(b'published checkpoint placeholder')
            torch=types.ModuleType('torch');torch.float32='torch.float32'
            torch.isfinite=lambda tensor:types.SimpleNamespace(all=lambda:all(math.isfinite(v) for v in tensor.values))
            safetensors=types.ModuleType('safetensors');safetensors.safe_open=lambda *a,**kw:Open(original)
            self.weights={}
            with patch.dict(sys.modules,{'torch':torch,'safetensors':safetensors}):
                module.verify_loaded_head(types.SimpleNamespace(state_dict=lambda:actual),checkpoint,self.weights)
            return self.weights

    def test_bf16_tensor_and_f32_scalar_promotions_exact(self):
        original={'matrix':Tensor([.5,-1.25],'torch.bfloat16'),'scalar':Tensor([1.],shape=())}
        actual={'matrix':Tensor([.5,-1.25]),'scalar':Tensor([1.],shape=())}
        proof=self.audit(original,actual)
        self.assertTrue(proof['head_identity_exact'])
        self.assertEqual(proof['head_tensor_count'],2)
        self.assertEqual(proof['head_state_elements'],3)
        self.assertEqual(proof['head_tensors']['matrix']['checkpoint_dtype'],'bfloat16')
        self.assertEqual(proof['head_tensors']['scalar']['shape'],[])
        for row in proof['head_tensors'].values():
            self.assertEqual(row['checkpoint_promoted_f32_sha256'],row['loaded_f32_sha256'])

    def test_missing_extra_shape_dtype_and_value_transforms_reject(self):
        original={'weight':Tensor([1.,2.],'torch.bfloat16')}
        for actual in ({},{'weight':Tensor([1.,2.]),'extra':Tensor([0.])},
                       {'weight':Tensor([1.,2.],shape=(1,2))},
                       {'weight':Tensor([1.,2.],'torch.bfloat16')},
                       {'weight':Tensor([1.,2.5])}):
            with self.subTest(actual=actual),self.assertRaises(ValueError):self.audit(original,actual)
            self.assertFalse(self.weights['head_identity_exact'])

    def test_nonfinite_checkpoint_or_loaded_reject(self):
        for original,actual in [({'w':Tensor([float('nan')])},{'w':Tensor([1.])}),
                                ({'w':Tensor([1.])},{'w':Tensor([float('inf')])})]:
            with self.assertRaisesRegex(ValueError,'nonfinite'):self.audit(original,actual)

if __name__=='__main__':unittest.main()
