"""Synthetic public-schema controls, never measured model/reference outputs."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import josev2b_task_quality_v1 as q


def synthetic(data):
    responses={}
    for case in data['cases']:
        answers={}
        for key,question in case['request']['questions'].items():
            gold=case['gold'][key]['label'];kind=question['type']
            if kind=='noul':answer=dict(type=kind,noul=.8 if gold=='true'else .2)
            else:
                ids=list(question['criteria'])if kind=='choice'else[str(i)for i in range(3)]
                probabilities={i:.8 if i==gold else .1 for i in ids}
                answer=dict(type=kind,confidence=.8,probabilities=probabilities)
                if kind=='choice':answer['choice']=gold
                else:answer.update(score=round(sum(int(i)*p for i,p in probabilities.items()),4),legend=dict(zip(ids,question['criteria'])))
            answers[key]=answer
        responses[case['id']]=dict(model=case['request']['model'],answers=answers,usage=dict(input_tokens=300,output_tokens=0))
    return responses


class QualityControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=q.panel();cls.responses=synthetic(cls.data)

    def test_authored_scope_and_all_synthetic_gold_correct(self):
        self.assertEqual(len(self.data['cases']),20)
        result=q.evaluate(self.responses)
        self.assertEqual(result['overall'],dict(correct=60,total=60,accuracy=1.))
        self.assertEqual(result['request_accuracy'],1.)
        self.assertTrue(result['floors_passed'])
        self.assertTrue(q.compare_public(self.responses,self.responses,precision='q4')['independent_task_quality_passed'])
        self.assertFalse(q.compare_public(self.responses,self.responses,precision='q4')['publication_authorized'])

    def test_exact_decimal_one_percent_bound_and_strict_flags_separate(self):
        changed=copy.deepcopy(self.responses)
        case=self.data['cases'][0];answer=changed[case['id']]['answers']['truth'];answer['noul']-=.01
        result=q.compare_public(self.responses,changed,precision='q4')
        self.assertTrue(result['independent_task_quality_passed'])
        self.assertFalse(result['strict_f32_flags_all_passed'])
        self.assertFalse(q.compare_public(self.responses,changed,precision='f32')['independent_task_quality_passed'])
        answer['noul']=.7899
        self.assertFalse(q.compare_public(self.responses,changed,precision='q4')['independent_task_quality_passed'])

    def test_ordinal_is_true_index_expectation_not_normalized(self):
        row=next(c for c in self.data['cases']if c['gold']['progress']['label']=='2')
        old=self.responses[row['id']]['answers']['progress']
        self.assertEqual(old['score'],1.7)
        bad=copy.deepcopy(self.responses);bad[row['id']]['answers']['progress']['score']=.85
        with self.assertRaisesRegex(ValueError,'true0..N-1'):
            q.evaluate(bad)
        bad=copy.deepcopy(self.responses);answer=bad[row['id']]['answers']['progress']
        answer['probabilities']['2']-=.01;answer['probabilities']['0']+=.01
        answer['confidence']=.79;answer['score']=1.68
        result=q.compare_public(self.responses,bad,precision='q4')
        field=next(r for r in result['fields']if r['case']==row['id']and r['field']=='progress')
        self.assertEqual(field['maximum_public_probability_absolute'],.01)
        self.assertEqual(field['ordinal_index_absolute'],.02)
        self.assertFalse(field['task_ordinal_passed'])
        self.assertFalse(result['independent_task_quality_passed'])

    def test_low_original_quality_cannot_pass_by_teacher_agreement(self):
        poor=copy.deepcopy(self.responses)
        for case in self.data['cases'][:7]:
            answer=poor[case['id']]['answers']['truth'];answer['noul']=1-answer['noul']
        result=q.compare_public(poor,poor,precision='q4')
        self.assertTrue(all(r['discrete_agreement']for r in result['fields']))
        self.assertFalse(result['original']['floors_passed'])
        self.assertFalse(result['independent_task_quality_passed'])

    def test_missing_extra_or_nonfinite_answers_reject(self):
        bad=copy.deepcopy(self.responses);bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError):q.evaluate(bad)
        bad=copy.deepcopy(self.responses);first=next(iter(bad));bad[first]['answers']['truth']['noul']=float('nan')
        with self.assertRaises(ValueError):q.evaluate(bad)
        bad=copy.deepcopy(self.responses);bad[first]['answers']['extra']={}
        with self.assertRaises(ValueError):q.evaluate(bad)

    def test_native_model_access_is_pending_not_arbitrary_main(self):
        cfg=copy.deepcopy(q.declaration());cfg['native_prerequisites']['reviewed_original20']=False
        with patch.object(q,'declaration',return_value=cfg):
            with self.assertRaisesRegex(ValueError,'pins pending'):
                q.native_prerequisites()

    def test_mandatory_original_stage_plan_has_all74_per_request(self):
        for case in self.data['cases']:
            expected=q.expected_stages(case)
            self.assertEqual(sum(expected.values()),74)
            self.assertTrue(all(expected['backbone/layer/'+str(i)]==1 for i in range(24)))
            self.assertTrue(all(expected['logits/'+key]==1 and expected['probabilities/'+key]==1
                                for key in case['request']['questions']))

    def test_new_original_freezes_before_any_weight_download(self):
        recipe=(ROOT/'tools/kaggle/josev2b_task_quality_reference_v1.py').read_text()
        freeze=recipe.index('"prepare", *worker_args')
        weights=recipe.index('if filename.endswith(".safetensors"):',freeze)
        self.assertLess(freeze,weights)
        producer=(ROOT/'tools/dump_josev2b_task_quality_v1.py').read_text()
        self.assertIn('len(unclipped.input_ids) != len(record.input_ids)',producer)
        self.assertIn('actual != expected',producer)
        self.assertIn('source.load_release_model',producer)
        self.assertNotIn('case["id"] == "state-only-clip"',producer)

    def test_npy_owner_extent_and_finite_policy_without_numpy(self):
        import hashlib
        import struct
        import tempfile
        path=ROOT/'tools/ci-heavy/josev2b_task_quality_v1.py'
        spec=importlib.util.spec_from_file_location('quality_hosted',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def encoded(values,fortran=False):
            header=repr(dict(descr='<f4',fortran_order=fortran,shape=(1,len(values)))).encode()
            header+=b' '*((-(10+len(header)+1))%16)+b'\n'
            return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<'+'f'*len(values),*values)
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'array.npy'
            for values,stage,negative,fortran,accepted in [
                    ([1.,2.],'backbone/embedding',0,False,True),
                    ([1.,float('-inf')],'backbone/layer/3/input/attention_mask',1,False,True),
                    ([1.,float('-inf')],'backbone/embedding',1,False,False),
                    ([1.,float('nan')],'backbone/embedding',0,False,False),
                    ([1.,2.],'backbone/embedding',0,True,False)]:
                blob=encoded(values,fortran);target.write_bytes(blob)
                row=dict(bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),shape=[1,2],dtype='float32',
                         stage=stage,negative_infinity=negative)
                if accepted:module.checked_npy(target,row)
                else:
                    with self.assertRaises(ValueError):module.checked_npy(target,row)
            blob=encoded([1.,2.]);target.write_bytes(blob+b'!')
            row=dict(bytes=len(blob)+1,sha256=hashlib.sha256(blob+b'!').hexdigest(),shape=[1,2],dtype='float32',
                     stage='backbone/embedding',negative_infinity=0)
            with self.assertRaisesRegex(ValueError,'extent'):module.checked_npy(target,row)
            row['bytes']=len(blob);row['sha256']=hashlib.sha256(blob).hexdigest()
            target.write_bytes(encoded([2.,1.]))
            with self.assertRaisesRegex(ValueError,'SHA'):module.checked_npy(target,row)

    def test_frozen_protocol_closes_installer_scorer_and_new_stage_sources(self):
        cfg=q.declaration()
        for name in ['tools/ci-heavy/josev2b_gpu_install.py','tools/ci-heavy/josev2b_wheel_metadata.py',
                     'tools/josev2b_composed_panel.py','tools/josev2b_composed_diff.py',
                     'tests/josev-task-quality/probe.cpp','tests/josev-task-quality/diff.cpp']:
            self.assertEqual(q.sha256(ROOT/name),cfg['source_sha256'][name])
        worker=(ROOT/'tools/ci-heavy/josev2b_task_quality_v1.py').read_text()
        self.assertLess(worker.index('pins=native_prerequisites()'),worker.index('original_guard(args.reference,pins)'))
        self.assertIn('request_json=json.dumps',worker)
        self.assertIn("results['f32']['qualified_for_owned_panel']",worker)
        self.assertIn("result['q4_vs_f32_task_quality']",worker)


if __name__=='__main__':unittest.main()
