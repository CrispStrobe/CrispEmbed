"""Model-free provenance/neutrality negative controls for paired capture consumer."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('paired_step33',ROOT/'tools/ci-heavy/lighton4b_mixed_step33_compare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Controls(unittest.TestCase):
    def fixture(self):
        # Structural fixture only; never fetched/admitted as a real service artifact.
        full={'mixed_final_bf16_parameter_sha256':{str(i):'a'for i in range(249)},'original_bf16_tensor_sha256':{str(i):'b'for i in range(723)},'arms':{'mixed':{'ids':list(range(178)),'scores':[{'position':i,'shape':[1,248320],'bytes':993280,'sha256':str(i)}for i in range(178)]}}}
        s={'source_sha':m.SOURCE,'activation_trace_completed':True,'comparison_completed':True,'trace_observer_all34_scores_bit_exact':True,'trace_observer_all34_ids_exact':True,'trace_stages':dict.fromkeys(m.stage_names(),{}),'mixed_final_bf16_parameter_sha256':full['mixed_final_bf16_parameter_sha256'],'original_bf16_tensor_sha256':full['original_bf16_tensor_sha256'],'total_model_forward_calls':35,'trace_input_ids':[32],'head_output_matches_generated_score':True,'arms':{'mixed':{'ids':list(range(34)),'scores':full['arms']['mixed']['scores'][:34],'forward_counts':{'root':34,'text':34,'vision':1},'generation_cap':34,'cap_hit':True,'eos_completed':False}}}
        return s,full
    def test_exact_structural_contract(self):
        s,f=self.fixture();m.validate_source(s,f)
    def test_wrong_source_scope_neutrality_and_counts_reject(self):
        for key,value in [('source_sha','0'*40),('activation_trace_completed',False),('trace_observer_all34_scores_bit_exact',False),('total_model_forward_calls',34),('trace_input_ids',[999]),('head_output_matches_generated_score',False),('error','late-audit-failure')]:
            s,f=self.fixture();s[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_source(s,f)
    def test_missing_stage_wrong_parameter_and_score_reject(self):
        for kind in ('stage','param','score','prefix','cap'):
            s,f=self.fixture();s=copy.deepcopy(s)
            if kind=='stage':s['trace_stages'].pop('greedy.33.layer0.qkv')
            elif kind=='param':s['mixed_final_bf16_parameter_sha256']['0']='wrong'
            elif kind=='score':s['arms']['mixed']['scores'][33]['sha256']='wrong'
            elif kind=='prefix':s['arms']['mixed']['ids'][32]=999
            else:s['arms']['mixed']['generation_cap']=256
            with self.subTest(kind=kind),self.assertRaises(ValueError):m.validate_source(s,f)
if __name__=='__main__':unittest.main()
