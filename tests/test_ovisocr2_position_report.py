#!/usr/bin/env python3
"""Stdlib-only retained-JSON integrity/redaction regression; no model or arrays."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/ci-heavy'))
spec=importlib.util.spec_from_file_location('ovis_position_report_test',ROOT/'tools/ci-heavy/ovisocr2_position_replay.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class RetainedReport(unittest.TestCase):
    def records(self):
        expected={'source_model':'owner/model','source_revision':'a'*40,
                  'source_config':{'_name_or_path':'<scratch>/hf/hub/models--owner--model/snapshots/'+'a'*40,
                                   'hidden_size':1024},
                  'stages':{'prefill.decoder.norm':{'shape':[1,9,1024],'sha256':'b'*64}},
                  'generated_ids':[[1,2]],'decoded':['frozen text'],'generation_result':{'reached_eos':True}}
        raw=copy.deepcopy(expected)
        raw['source_config']['_name_or_path']=expected['source_config']['_name_or_path'].replace('<scratch>','/fixture/producer-scratch')
        payload=(json.dumps(raw,indent=2)+'\n').encode()
        producer={'original_capture':expected,'original_files':{'original.json':{'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}}}
        return payload,producer

    def test_one_public_prefix_difference_is_allowed_after_exact_raw_identity(self):
        payload,producer=self.records()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'original.json';path.write_bytes(payload)
            original,proof=driver.checked_retained_reference(path,producer)
            self.assertEqual(original,json.loads(payload))
            self.assertEqual(path.read_bytes(),payload)
            self.assertEqual(proof['normalized_field'],'source_config._name_or_path')
            self.assertTrue(proof['all_other_fields_exact'])
            self.assertTrue(proof['raw_file_unchanged'])

    def test_wrong_raw_bytes_or_hash_reject_before_json_normalization(self):
        payload,producer=self.records()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'original.json';path.write_bytes(b'not JSON')
            with self.assertRaisesRegex(ValueError,'before redaction normalization'):
                driver.checked_retained_reference(path,producer)
            path.write_bytes(payload)
            producer['original_files']['original.json']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'before redaction normalization'):
                driver.checked_retained_reference(path,producer)

    def test_every_scientific_or_other_config_difference_rejects(self):
        payload,producer=self.records()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'original.json';path.write_bytes(payload)
            for field in ('generated_ids','decoded','stages','generation_result','hidden_size'):
                candidate=copy.deepcopy(producer)
                if field=='hidden_size':candidate['original_capture']['source_config'][field]=2048
                else:candidate['original_capture'][field]={'changed':True}
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'beyond model-path redaction'):
                    driver.checked_retained_reference(path,candidate)

    def test_wrong_snapshot_or_prefix_is_not_an_allowed_redaction(self):
        payload,producer=self.records()
        for changed in ('relative/hf/hub/models--owner--model/snapshots/'+'a'*40,
                        '/fixture/../producer/hf/hub/models--owner--model/snapshots/'+'a'*40,
                        '/fixture/hf/hub/models--owner--model/snapshots/'+'c'*40):
            raw=json.loads(payload);raw['source_config']['_name_or_path']=changed
            altered=(json.dumps(raw,indent=2)+'\n').encode()
            candidate=copy.deepcopy(producer)
            candidate['original_files']['original.json']={'bytes':len(altered),'sha256':hashlib.sha256(altered).hexdigest()}
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'original.json';path.write_bytes(altered)
                with self.assertRaises(ValueError):driver.checked_retained_reference(path,candidate)

    def test_no_framework_dependencies(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('PIL',sys.modules)


if __name__=='__main__':unittest.main()
