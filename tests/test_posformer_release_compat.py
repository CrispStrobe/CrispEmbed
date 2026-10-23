"""Model-free wrong-output controls for the actual released-binary oracle."""
from pathlib import Path
import copy
import json
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from probe_posformer_release import compare_release_records

class ReleasedControls(unittest.TestCase):
    def records(self):
        p=Path(__file__).parent/'results/posformer/released-compatibility.json'
        d=json.loads(p.read_text())
        return d['released'],d['candidate']

    def test_measured_actual_outputs_match_without_hiding_confidence_drift(self):
        result=compare_release_records(*self.records())
        self.assertEqual(result['cases'],50)
        self.assertGreater(result['max_confidence_abs'],0)

    def test_wrong_token_text_stop_and_image_rejected(self):
        for key,value in [('token_ids',[2]),('text','wrong'),('stop','length'),('image_sha256','0'*64)]:
            old,new=self.records();new=copy.deepcopy(new);new['cases'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):compare_release_records(old,new)

    def test_missing_duplicate_and_reordered_cases_rejected(self):
        for mode in ('missing','duplicate','reordered'):
            old,new=self.records()
            if mode=='missing':new['cases'].pop()
            elif mode=='duplicate':old['cases'][1]=copy.deepcopy(old['cases'][0])
            else:new['cases'].reverse()
            with self.subTest(mode=mode),self.assertRaises(ValueError):compare_release_records(old,new)

    def test_nonfinite_confidence_and_changed_model_rejected(self):
        for mode in ('nan','model','manifest','source'):
            old,new=self.records()
            if mode=='nan':new['cases'][0]['confidence'][0]=float('nan')
            elif mode=='model':new['model_sha256']='0'*64
            elif mode=='manifest':new['manifest_sha256']='0'*64
            else:old['library_source']='0'*40
            with self.subTest(mode=mode),self.assertRaises(ValueError):compare_release_records(old,new)

if __name__=='__main__':unittest.main()
