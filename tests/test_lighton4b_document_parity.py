#!/usr/bin/env python3
"""Model-free controls for the frozen document protocol and real diff executable."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('document',ROOT/'tools/lighton4b_document_parity.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def tiny_gguf(path,rows):
    """Small authored diagnostic vectors, not model tensors or inference."""
    def string(text):
        raw=text.encode();return struct.pack('<Q',len(raw))+raw
    header=struct.pack('<4sIQQ',b'GGUF',3,len(rows),0);payload=b''
    for name,values in rows:
        header+=string(name)+struct.pack('<I',1)+struct.pack('<QIQ',len(values),0,len(payload))
        data=struct.pack('<'+'f'*len(values),*values);payload+=data+b'\0'*((-len(data))%32)
    path.write_bytes(header+b'\0'*((-len(header))%32)+payload)


class Controls(unittest.TestCase):
    def test_exact_case_source_and_no_execution_admission(self):
        p=driver.protocol();self.assertEqual(p['case']['sha256'],'3d986ec0d131e4e429127627929e12fd4845313f43849c26bb3d1231c3b038dc')
        self.assertEqual(p['case']['max_new_tokens'],256)
        self.assertEqual(p['execution_admission']['original_producer_sources'],[])
        self.assertEqual(p['execution_admission']['native_producer_sources'],[])
        self.assertFalse(any(p['qualification'].values()))
    def test_all32_decoder24_vision_caches_and_generation_positions(self):
        rows=driver.stages(2);names=[n for n,_ in rows]
        self.assertEqual(len(names),len(set(names)))
        for i in range(24):self.assertIn('prefill.vision.layer.'+str(i),names)
        for phase in ('prefill','cached','full_prefix'):
            for i in range(32):self.assertIn(f'{phase}.decoder.layer.{i}',names)
            self.assertIn(phase+'.decoder.norm',names)
        self.assertIn('generated.scores.0',names);self.assertIn('generated.scores.1',names)
        self.assertLess(names.index('prefill.vision.layer.23'),names.index('prefill.decoder.layer.0'))
    def test_caps_reject_invalid_and_boolean_counts(self):
        for count in (0,257,-1,True,'1'):
            with self.subTest(count=count),self.assertRaises(ValueError):driver.stages(count)
    def test_no_framework_model_or_native_forward_import(self):
        self.assertFalse(any(n in driver.__dict__ for n in ('torch','transformers','numpy','gguf')))
    def test_comparison_retains_blocked_failure_without_forward(self):
        with tempfile.TemporaryDirectory()as folder:
            output=Path(folder)/'report'
            with mock.patch.dict(os.environ,{'GITHUB_ACTIONS':'true'}):
                self.assertEqual(driver.compare(folder,folder,'unused-binary',output),2)
            report=json.loads((output/'comparison.json').read_text());self.assertEqual(report['status'],'blocked')
            self.assertFalse(report['original_forward_executed_by_comparer'])
            with self.assertRaises(ValueError):driver.compare(folder,folder,'unused-binary',output)
    @unittest.skipUnless(os.environ.get('LIGHTON4B_DOCUMENT_DIFF'),'hosted compiled comparator required')
    def test_actual_crispembed_diff_positive_earliest_exact_nonfinite_and_override(self):
        binary=os.environ['LIGHTON4B_DOCUMENT_DIFF']
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);order=root/'order.tsv';order.write_text('input.ids exact\nvision.0 float\ndecoder.0 float\n')
            rows=[('input.ids',[248044.,248046.]),('vision.0',[1.,2.,3.,4.]),('decoder.0',[2.,4.,6.,8.])]
            original=root/'original.gguf';candidate=root/'candidate.gguf';tiny_gguf(original,rows)
            def run(changed):
                tiny_gguf(candidate,changed)
                return subprocess.run([binary,original,candidate,order],capture_output=True,text=True,
                    env=dict(os.environ,CRISPEMBED_DIFF_COS_THRESHOLD='0.1'))
            result=run(rows);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('stages=3 first=none',result.stdout)
            changed=[rows[0],('vision.0',[1.,2.,3.,4.01]),('decoder.0',[2.,4.,6.,8.02])]
            result=run(changed);self.assertEqual(result.returncode,1);self.assertIn('first=vision.0',result.stdout)
            self.assertIn('ref_norm=',result.stdout);self.assertIn('mean_abs=',result.stdout);self.assertIn('rms=',result.stdout)
            result=run([('input.ids',[248045.,248046.])]+rows[1:]);self.assertEqual(result.returncode,1)
            result=run([rows[0],('vision.0',[1.,2.,3.,float('nan')]),rows[2]])
            self.assertEqual(result.returncode,1);self.assertIn('finite=0',result.stdout)
            result=run(rows[:-1]);self.assertNotEqual(result.returncode,0)


if __name__=='__main__':unittest.main()
