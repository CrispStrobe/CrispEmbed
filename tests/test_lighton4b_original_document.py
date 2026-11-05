#!/usr/bin/env python3
"""Hosted synthetic observer events only; never original model construction."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('adapter',ROOT/'tools/lighton4b_original_document.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)


class Tensor:
    def __init__(self,value):self.value=np.asarray(value);self.shape=self.value.shape
    def detach(self):return self
    def cpu(self):return self
    def float(self):return Tensor(self.value.astype(np.float32,copy=False))
    def contiguous(self):return Tensor(np.ascontiguousarray(self.value))
    def numpy(self):return self.value
    def is_floating_point(self):return np.issubdtype(self.value.dtype,np.floating)
    def element_size(self):return self.value.dtype.itemsize
    def __getitem__(self,key):return Tensor(self.value[key])


class Model:
    def register_forward_hook(self,callback):
        self.callback=callback;self.removed=False
        def remove():self.removed=True
        return SimpleNamespace(remove=remove)


class Controls(unittest.TestCase):
    def test_exact_four_injections_without_framework_import_or_old_file_changes(self):
        p=adapter.document.protocol();source=ROOT/'tools/dump_qwen35_ocr_reference.py';before=source.read_bytes()
        function,proof=adapter.adapted_capture(SimpleNamespace())
        self.assertTrue(callable(function));self.assertEqual(proof['injections'],dict(save=1,inputs=1,attach=1,verify=1))
        self.assertEqual(source.read_bytes(),before);self.assertNotIn('torch',adapter.__dict__)
    def test_transport_block_is_before_model_access(self):
        with tempfile.TemporaryDirectory()as folder,mock.patch.dict(os.environ,{'GITHUB_ACTIONS':'true'}):
            output=Path(folder)/'output'
            with self.assertRaisesRegex(ValueError,'transport'):adapter.capture('/missing-9GB-checkpoint',output)
            r=json.loads((output/'adapter.json').read_text());self.assertEqual(r['status'],'failed')
            self.assertFalse(r['document_bundle_admitted'])
    def test_duplicate_nonfinite_and_budgets_fail_before_partial_write(self):
        with tempfile.TemporaryDirectory()as folder:
            o=adapter.GeneratedObserver(Path(folder)/'stream',{},max_bytes=16,max_legacy_bytes=4)
            o.snapshot('generated.0.raw_logits',Tensor([1.,2.]))
            for name,value in [('generated.0.raw_logits',[1.]),('generated.0.nan',[np.nan]),('generated.0.large',[1.]*8)]:
                with self.assertRaises(ValueError):o.snapshot(name,Tensor(value))
            with self.assertRaisesRegex(ValueError,'legacy'):o.tensor('prefill.x',Tensor([1.,2.]))
            self.assertEqual(len(o.records),1);o.close()
    def test_full_width_generated_cache_event_and_neutrality(self):
        with tempfile.TemporaryDirectory()as folder:
            o=adapter.GeneratedObserver(Path(folder)/'stream',{})
            score=np.zeros((1,248320),dtype=np.float32);score[0,248046]=1
            conv=Tensor(np.zeros((1,8192,4),dtype=np.float32));state=Tensor(np.zeros((1,32,128,128),dtype=np.float32))
            kv=Tensor(np.zeros((1,4,1,256),dtype=np.float32))
            layers=[SimpleNamespace(keys=kv,values=kv)if i%4==3 else SimpleNamespace(conv_states=conv,recurrent_states=state)for i in range(32)]
            model=Model();o.attach(model,['generated.0'])
            o.tensor('generated.0.decoder.norm',Tensor(np.ones((1,1,2560),dtype=np.float32)))
            o.tensor('generated.0.decoder.position_ids',Tensor(np.zeros((4,1,1),dtype=np.int64)))
            before=state.value.tobytes()
            model.callback(model,(),SimpleNamespace(logits=Tensor(score[:,None,:]),past_key_values=SimpleNamespace(layers=layers)))
            self.assertEqual(len(o.records),67);self.assertEqual(state.value.tobytes(),before)
            with self.assertRaisesRegex(ValueError,'IDs'):o.finish([[248046]],[Tensor(score)],[[1]],[Tensor(score)],'form')
            changed=score.copy();changed[0,0]=.1
            with self.assertRaisesRegex(ValueError,'scores'):o.finish([[248046]],[Tensor(score)],[[248046]],[Tensor(changed)],'form')
            o.finish([[248046]],[Tensor(score)],[[248046]],[Tensor(score)],'form')
            self.assertEqual(o.report['status'],'passed');self.assertTrue(o.report['raw_logits_equal_generate_scores'])
            o.close();self.assertTrue(model.removed)
    def test_actual_geometry_rejected_before_model_construction(self):
        with tempfile.TemporaryDirectory()as folder:
            o=adapter.GeneratedObserver(Path(folder)/'stream',adapter.document.protocol())
            with self.assertRaisesRegex(ValueError,'template'):o.validate_inputs(None,{},None,'changed')
            o.close()


if __name__=='__main__':unittest.main()
