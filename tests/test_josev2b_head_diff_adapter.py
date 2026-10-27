#!/usr/bin/env python3
"""Synthetic byte/collector controls only; no framework or C++ execution."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from josev2b_head_diff_adapter import collect,read_gguf,write_gguf,write_pair


class AdapterTests(unittest.TestCase):
    def data(self,directory,values=(0.,-0.,.1,-123.25,1e-30,1e30)):
        path=Path(directory)/'array.f32';data=struct.pack('<'+'f'*len(values),*values);path.write_bytes(data)
        return {'path':path,'shape':[2,3],'sha256':hashlib.sha256(data).hexdigest(),'label':'array.f32'}

    def test_lossless_bits_shape_axes_and_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            record=self.data(tmp);out=Path(tmp)/'reference.gguf'
            proof=write_gguf(out,[{'name':'stage',**record}])
            binary=out.read_bytes();self.assertEqual(struct.unpack_from('<IQQ',binary,4),(3,1,0))
            n=struct.unpack_from('<Q',binary,24)[0];at=32+n
            self.assertEqual(struct.unpack_from('<IQQ',binary,at),(2,3,2))
            self.assertEqual(read_gguf(out)['stage'],{'shape':[2,3],'bytes':24,'sha256':record['sha256']})
            self.assertEqual(proof['tensors']['stage']['sha256'],record['sha256'])
            self.assertIn(record['path'].read_bytes(),binary)

    def test_no_reshape_even_when_element_counts_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=self.data(tmp);native={**original,'shape':[1,2,3]}
            with self.assertRaisesRegex(ValueError,'no reshape'):
                write_pair(Path(tmp)/'a.gguf',Path(tmp)/'b.gguf',[{'name':'stage','original':original,'native':native}])

    def test_changed_missing_nonfinite_and_wrong_size_sources_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            item=self.data(tmp)
            for altered in [{**item,'sha256':'0'*64},{**item,'shape':[2,4]},
                            {**item,'path':Path(tmp)/'missing'}]:
                with self.assertRaises(ValueError):write_gguf(Path(tmp)/'bad.gguf',[{'name':'stage',**altered}])
            for bad in [float('nan'),float('inf'),-float('inf')]:
                item=self.data(tmp,(bad,0.,0.,0.,0.,0.))
                with self.assertRaisesRegex(ValueError,'Nonfinite'):
                    write_gguf(Path(tmp)/'bad.gguf',[{'name':'stage',**item}])

    def test_unsafe_duplicate_name_and_partial_payload_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            item=self.data(tmp)
            with self.assertRaisesRegex(ValueError,'tensor name'):
                write_gguf(Path(tmp)/'bad.gguf',[{'name':'../stage',**item}])
            with self.assertRaisesRegex(ValueError,'duplicate'):
                write_gguf(Path(tmp)/'bad.gguf',[{'name':'stage',**item},{'name':'stage',**item}])
            out=Path(tmp)/'good.gguf';write_gguf(out,[{'name':'stage',**item}]);out.write_bytes(out.read_bytes()[:-4])
            with self.assertRaisesRegex(ValueError,'truncated'):
                read_gguf(out)

    def record(self,passed=True):
        return {'shape':[1,2],'elements':2,'max_absolute':0. if passed else 1.,
                'mean_absolute':0. if passed else 1.,'mean_error':0.,'rms':0. if passed else 1.,
                'min_row_cosine':1. if passed else 0.,'original_l2':1.,'native_l2':1.,
                'original_min':0.,'original_max':1.,'original_abs_min':0.,'original_abs_max':1.,
                'native_min':0.,'native_max':1.,'native_abs_min':0.,'native_abs_max':1.,'passed':passed}

    def result(self,path,metrics):
        row={k:v for k,v in metrics.items() if k!='shape'}
        row.update(name='stage',found=True,ggml_shape=[2,1],row_axis=0)
        report={'schema':'crispembed-josev-head-ref-v1','max_absolute_budget':.002,'minimum_row_cosine':.99999,
                'status':'passed' if metrics['passed'] else 'numeric_rejection','stages':[row]}
        path.write_text(json.dumps(report));return report

    def test_numeric_rejection_returns_records_and_does_not_abort_collector(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json';measured=self.record(False);self.result(path,measured)
            checks={'stage':measured};result=collect(1,path,checks)
            self.assertEqual(result['exit_code'],1);self.assertFalse(checks['stage']['passed'])
            self.assertTrue(result['all_metrics_agree']);self.assertEqual(checks['stage']['cpp_ref']['rms'],1.)

    def test_metric_disagreement_cannot_turn_model_success_into_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json';measured=self.record();report=self.result(path,measured)
            report['stages'][0]['original_l2']=3.;path.write_text(json.dumps(report))
            checks={'stage':measured};collect(0,path,checks)
            self.assertTrue(checks['stage']['numpy_passed']);self.assertFalse(checks['stage']['passed'])
            self.assertFalse(checks['stage']['metric_agreement']['original_l2']['passed'])

    def test_exit_coverage_shape_budget_and_false_success_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json';original=self.result(path,self.record())
            for alter in [lambda r:r.update(status='numeric_rejection'),lambda r:r.update(stages=[]),
                          lambda r:r.update(max_absolute_budget=.003),lambda r:r['stages'][0].update(row_axis=1),
                          lambda r:r['stages'][0].update(elements=1),lambda r:r['stages'][0].update(max_absolute=.003)]:
                report=copy.deepcopy(original);alter(report);path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):collect(0,path,{'stage':self.record()})
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError,'structural/process'):
                collect(2,path,{'stage':self.record()})


if __name__=='__main__':unittest.main()
