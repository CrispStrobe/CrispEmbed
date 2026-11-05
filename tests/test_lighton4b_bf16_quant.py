#!/usr/bin/env python3
"""Model-free controls; synthetic scores never admit native execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_bf16_quant as q
class Controls(unittest.TestCase):
    def fixture(self,root):
        p=q.protocol();p=dict(p,positions=2)
        a={'generated_ids':[1,248046],'decoded':'synthetic','scores':[]}
        data=struct.pack('<f',1.0)*248320
        for i in range(2):
            name=f'scores-{i:03d}.f32';(root/name).write_bytes(data)
            a['scores'].append(dict(position=i,file=name,shape=[1,248320],bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        b=dict(schema='lighton4b-bf16-quant-native/v1',precision='q8_0',original_capture_sha256=p['original_capture_sha256'],protocol_sha256=q.sha(q.PROTOCOL),teacher_forced_ids=a['generated_ids'],free_generation_independent=True,generated_ids=a['generated_ids'],decoded='synthetic',stop='eos',cap_hit=False,scores=a['scores'],synthetic_control=True)
        return p,a,b
    def test_admission_and_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p,a,b=self.fixture(Path(d));q.candidate(b,a,p,True)
            with self.assertRaisesRegex(ValueError,'actual native'):q.candidate(b,a,p)
            for field,value in [('teacher_forced_ids',[1]),('free_generation_independent',False),('scores',b['scores'][:1]),('precision','f32'),('cap_hit',True)]:
                with self.subTest(field=field),self.assertRaises(ValueError):q.candidate(dict(b,**{field:value}),a,p,True)
            forged=dict(b,synthetic_control=False,native_forward_executed=True,producer_source_sha='0'*40,producer_run_url='https://github.com/example/run',weights_sha256='0'*64)
            with self.assertRaisesRegex(ValueError,'not admitted'):q.candidate(forged,a,p)
            self.assertEqual(q.protocol()['native_producers'],[])
    @unittest.skipUnless(os.environ.get('LIGHTON4B_DOCUMENT_DIFF'),'hosted actual Ref binary required')
    def test_actual_ref_equal_changed_and_tampered(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);orig=root/'original';native=root/'native';orig.mkdir();native.mkdir();p,a,b=self.fixture(orig)
            for row in a['scores']:(native/row['file']).write_bytes((orig/row['file']).read_bytes())
            (native/'native.json').write_text(json.dumps(b))
            with patch.object(q,'protocol',return_value=p),patch.object(q,'original',return_value=a):
                self.assertEqual(q.compare(orig,native,os.environ['LIGHTON4B_DOCUMENT_DIFF'],root/'equal',True),0)
                report=json.loads((root/'equal/comparison.json').read_text());self.assertFalse(report['native_execution_admitted']);self.assertEqual(report['compared_positions'],2)
                b['scores']=[dict(x) for x in b['scores']]
                first=native/b['scores'][0]['file'];saved=first.read_bytes();small=bytearray(saved);small[:4]=struct.pack('<f',1.001);first.write_bytes(small)
                b['scores'][0]['sha256']=hashlib.sha256(small).hexdigest();(native/'native.json').write_text(json.dumps(b))
                self.assertEqual(q.compare(orig,native,os.environ['LIGHTON4B_DOCUMENT_DIFF'],root/'small',True),0)
                report=json.loads((root/'small/comparison.json').read_text());self.assertEqual(report['earliest_observed_score_difference'],0);self.assertIsNone(report['earliest_score_budget_failure'])
                first.write_bytes(saved);b['scores'][0]['sha256']=hashlib.sha256(saved).hexdigest()
                data=bytearray((native/b['scores'][1]['file']).read_bytes());data[:4]=struct.pack('<f',2.0);(native/b['scores'][1]['file']).write_bytes(data)
                b['scores']=[dict(x) for x in b['scores']];b['scores'][1]['sha256']=hashlib.sha256(data).hexdigest();(native/'native.json').write_text(json.dumps(b))
                self.assertEqual(q.compare(orig,native,os.environ['LIGHTON4B_DOCUMENT_DIFF'],root/'changed',True),1)
                report=json.loads((root/'changed/comparison.json').read_text());self.assertEqual(report['earliest_observed_score_difference'],1);self.assertIn('ref_norm=',report['metrics'][1])
                (native/b['scores'][1]['file']).write_bytes(b'bad')
                self.assertEqual(q.compare(orig,native,os.environ['LIGHTON4B_DOCUMENT_DIFF'],root/'tampered',True),2)
if __name__=='__main__':unittest.main()
