"""Observer scope/neutrality contracts without importing a framework/model."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('trace_contract',ROOT/'tools/lighton4b_bf16_mixed_step33.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Tools set a shared base.PROTOCOL at import; capture this tool's own path
# so later-collected sibling tools in one pytest session cannot redirect it.
OWN_PROTOCOL=m.base.PROTOCOL
class Controls(unittest.TestCase):
    def test_closed_stages_and_bound(self):
        p=json.loads(OWN_PROTOCOL.read_text());names=m.stage_names()
        self.assertEqual(len(names),50);self.assertEqual(p['trace_position'],33);self.assertEqual(p['max_new_tokens'],34)
        self.assertFalse(p['full_free_generation']);self.assertTrue(p['activation_trace']);self.assertEqual(p['maximum_score_bytes'],34*993280)
        self.assertTrue(all(n.startswith('greedy.33.')for n in names));self.assertTrue({'greedy.33.layer0.qkv','greedy.33.head.input','greedy.33.head.output','greedy.33.layer.31'}<=names)
        self.assertEqual(p['full_capture_sha256'],'ed4ef1b60c41d4d3b2a131d107c2563ee971c931873202f5f470683ebb9a1d58')
    def test_exact_prefix_neutrality(self):
        r={'ids':list(range(178)),'scores':[{'position':i,'sha256':str(i)}for i in range(178)]};a={'ids':r['ids'][:34],'scores':r['scores'][:34]};m.validate_neutrality(a,r)
        for kind in ('id','hash','coverage','position'):
            bad=copy.deepcopy(a)
            if kind=='id':bad['ids'][33]=999
            elif kind=='hash':bad['scores'][33]['sha256']='changed'
            elif kind=='position':bad['scores'][33]['position']=32
            else:bad['scores'].pop()
            with self.assertRaises(ValueError):m.validate_neutrality(bad,r)
    def test_partial_cap34_is_not_full256(self):
        m.full.validate_generation([27]*34,34,{'root':34,'text':34,'vision':1},34)
        with self.assertRaises(ValueError):m.full.validate_generation([27]*34,34,{'root':34,'text':34,'vision':1})
        with self.assertRaises(ValueError):m.full.validate_generation([27]*35,35,{'root':35,'text':35,'vision':1},34)
if __name__=='__main__':unittest.main()
