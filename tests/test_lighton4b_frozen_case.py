#!/usr/bin/env python3
"""Fail-closed fixed-case contract controls; no model data or forwards."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
import copy
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('frozen',ROOT/'tools/ci-heavy/lighton4b_frozen_case.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class FixedCaseControls(unittest.TestCase):
 def test_exact_cases_and_pngs(self):
  for name in ('simple_table','arabic_printed_line','receipt_example','handwritten_letter'):
   protocol,case,frozen=module.load_contract(name)
   self.assertEqual(case['expected_grid'],frozen['expected_grid'])
   self.assertEqual(case['expected_input_tokens'],frozen['expected_input_tokens'])
   self.assertEqual(module.sha(ROOT/'tests/regression/lighton4b-mixed-v1'/frozen['png_file']),frozen['png_sha256'])
   self.assertEqual(protocol['candidate_recipe'],module.MIXED)
 def test_outside_case_rejected(self):
  for name in ('simple_form','../simple_table','new_case',''):
   with self.assertRaises(ValueError):module.load_contract(name)
 def test_manifest_digest_required(self):
  original=module.FROZEN_SHA
  try:
   module.FROZEN_SHA='0'*64
   with self.assertRaises(ValueError):module.load_contract('simple_table')
  finally:module.FROZEN_SHA=original
 def test_actual_fourcase_schema_without_invented_dtype(self):
  _,case,_=module.load_contract('simple_table')
  original={'schema':'lighton4b-fourcase-bf16-capture/v1','source_sha':module.ORIGINAL_SOURCE,
   'case_id':case['id'],'split':case['split'],'protocol_sha256':module.PROTOCOL_SHA,
   'input_manifest_sha256':module.FROZEN_SHA,'original_forward_executed':True,
   'source':{'model':'lightonai/LightOnOCR-3-4B','revision':'b71010f095dc3735de7ee6b9969bff8f8f50b39a'}}
  module.validate_original(original,case)
  for key,value in [('source_sha','0'*40),('case_id','receipt_example'),('protocol_sha256','0'*64),('requested_dtype','float32')]:
   bad=copy.deepcopy(original);bad[key]=value
   with self.assertRaises(ValueError):module.validate_original(bad,case)
 def test_prebuilt_source_and_binary_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);receipt={'schema':'lighton4b-frozen-case-build/v1','status':'passed','source_sha':'a'*40,'model_payload_bytes':0}
   for name in ('lighton4b-native-document','lighton4b-native-case-preflight','crispembed-diff-lighton4b-document'):
    (root/name).write_bytes(name.encode());receipt[name+'_sha256']=module.sha(root/name)
   path=root/'build.json';path.write_text(json.dumps(receipt));digest=module.sha(path)
   module.validate_build(root,path,digest,'a'*40)
   with self.assertRaises(ValueError):module.validate_build(root,path,digest,'b'*40)
   (root/'lighton4b-native-document').write_bytes(b'tampered')
   with self.assertRaises(ValueError):module.validate_build(root,path,digest,'a'*40)
if __name__=='__main__':unittest.main()
