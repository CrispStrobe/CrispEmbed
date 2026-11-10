import copy,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('full_form',ROOT/'tools/kaggle/lighton4b_physical_f16_full_form.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FullFormAdmission(unittest.TestCase):
    def setUp(self):self.cfg=json.loads((ROOT/'docs/lighton4b-physical-f16-full-form-v1.json').read_text())
    def test_frozen_full_scope(self):m.admit(self.cfg)
    def test_scope_and_receipt_mutations(self):
        for key,value in [('arms',['f16-full']),('cap',2),('baseline_cap',256),('teacher_positions',1),('threads',2),('context',4096),('required_integration_source_sha',None),('required_integration_receipt_sha256',None),('native_arm_timeout_seconds',0)]:
            with self.subTest(key=key):
                changed=copy.deepcopy(self.cfg);changed[key]=value
                with self.assertRaises(RuntimeError):m.admit(changed)
    def test_runtime_blob_drift(self):
        changed=copy.deepcopy(self.cfg);changed['runtime_sources']['src/qwen35_attention.cpp']='0'*64
        with self.assertRaises(RuntimeError):m.admit(changed)

if __name__=='__main__':unittest.main()
