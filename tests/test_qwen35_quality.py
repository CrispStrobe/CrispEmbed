"""Model-free prerequisite, control-flow, stop and strict measurement regressions."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/ci-heavy')]
from qwen35_quality import reference_complete, require_reviewed_preparation, strict_prefill, summarize, validate_review


class Audit(unittest.TestCase):
    def evidence(self):
        panel = json.loads((ROOT / 'tests/regression/ocr-quality/panel.json').read_text())
        rows = [dict(id=c['id'], file=c['id']+'.png', source_sha256=c['source_sha256'], gold_sha256=c['gold_sha256'],
            size=c['resize'] or [448,448], nominal_patches=(c['resize'] or [448,448])[0]*(c['resize'] or [448,448])[1]//256,
            png_sha256='a'*64, rgb_sha256='b'*64) for c in panel['cases']]
        evidence = dict(policy='crispembed-ocr-quality-v1', panel_sha256='panel', scorer_sha256='scorer', split_evidence_sha256={'old':'sha'},
            workflow_conclusion='success', workflow='https://github.com/CrispStrobe/CrispEmbed/actions/runs/1',
            source_commit='1'*40, artifact_report_sha256='2'*64, cases=rows,
            visual_review=dict(reviewed_case_ids=[c['id'] for c in panel['cases']], notes='control only, not actual reviewed image evidence',
                png_sha256={c['id']:c['png_sha256'] for c in rows}))
        return panel, evidence, 'panel', 'scorer', {'old':'sha'}

    def test_actual_retained_preparation_matches_contract(self):
        self.assertEqual(len(require_reviewed_preparation(ROOT)['visual_review']['reviewed_case_ids']),3)

    def test_complete_review_controls(self):
        self.assertEqual(validate_review(*self.evidence())['workflow_conclusion'], 'success')

    def test_unreviewed_changed_and_unsuccessful_evidence_rejected(self):
        for mutate in (lambda r:r.update(workflow_conclusion='failure'), lambda r:r.update(scorer_sha256='changed'),
                lambda r:r.update(panel_sha256='changed'), lambda r:r['visual_review'].update(reviewed_case_ids=[]),
                lambda r:r['visual_review'].update(png_sha256={}), lambda r:r['visual_review'].update(notes=''),
                lambda r:r['cases'][0].update(file='../escape.png'), lambda r:r['cases'][1].update(gold_sha256='changed'),
                lambda r:r['cases'][0].update(nominal_patches=4096), lambda r:r['cases'][0].update(png_sha256='not-a-hash')):
            args=list(self.evidence());mutate(args[1])
            with self.assertRaises(ValueError):validate_review(*args)

    def bootstrap(self):
        spec=importlib.util.spec_from_file_location('quality_bootstrap',ROOT/'tools/ci-heavy/qwen35_ocr_bootstrap.py')
        result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
        return result

    def test_missing_review_rejected_before_imports_or_model_work(self):
        bootstrap=self.bootstrap(); audit=Mock()
        audit.require_reviewed_preparation.side_effect=ValueError('review absent')
        with patch.object(bootstrap,'module',return_value=audit), patch.object(sys,'argv',['bootstrap','--quality']), patch('builtins.__import__',side_effect=AssertionError('heavy import reached')):
            with self.assertRaisesRegex(ValueError,'review absent'):bootstrap.main()
        audit.preflight.assert_not_called()

    def test_render_drift_rejected_before_heavy_imports(self):
        bootstrap=self.bootstrap(); audit=Mock()
        audit.preflight.side_effect=ValueError('render changed')
        with patch.object(bootstrap,'module',return_value=audit), patch.object(sys,'argv',['bootstrap','--quality']), patch.dict('os.environ',{'HEAVY_OUT':'unused-control-output'}):
            with patch('builtins.__import__',side_effect=AssertionError('heavy import reached')):
                with self.assertRaisesRegex(ValueError,'render changed'):bootstrap.main()
        audit.require_reviewed_preparation.assert_called_once()

    def test_original_stop_and_cap_consistency(self):
        ref=dict(generated_ids=[1,2], eos_ids=[2], reached_eos=True, stop='eos', max_new_tokens=2)
        self.assertTrue(reference_complete(ref,2))
        for change in (dict(stop='max_tokens'), dict(reached_eos=False), dict(generated_ids=[True,2]),dict(max_new_tokens=1)):
            with self.assertRaises(ValueError):reference_complete(dict(ref,**change),2)
        ref.update(eos_ids=[99],reached_eos=False,stop='max_tokens')
        self.assertFalse(reference_complete(ref,2))
        with self.assertRaises(ValueError):reference_complete(dict(ref,generated_ids=[1]),2)

    def prefill(self):
        names=['stage.'+str(i) for i in range(77)]
        rows=[]
        for name in names:
            rows += [f'METRIC {name} n=1 cos_min=1 cos_global=1 max_abs=0 mean_abs=0 rms=0 norms=1/1',
                f'PASS {name}: cos_min=1',f'BUDGET {name}: strict_absolute_pass=1 calibrated=0 self_max_abs=0 max_ratio=0']
        return '\n'.join(rows)+'\nPREFILL_ONLY finite=1 metrics=77 selected=1 reference=1 strict_failures=0\n',names

    def test_prefill_failure_retained_with_magnitude(self):
        log,names=self.prefill()
        log=log.replace('PASS stage.2:', 'FAIL stage.2:').replace('strict_failures=0','strict_failures=1')
        result=strict_prefill(log,names)
        self.assertEqual(result['qualified_failed_stages'],['stage.2'])
        self.assertEqual(result['first_qualified_failure']['native_norm'],1)

    def test_prefill_missing_duplicate_nonfinite_and_contradictory_records(self):
        log,names=self.prefill()
        self.assertIsNone(strict_prefill(log,names)['first_qualified_failure'])
        for changed in (log.replace('stage.1','stage.0'),log.replace('norms=1/1','norms=nan/1',1),
                        log.replace('max_ratio=0','max_ratio=2',1),log.replace('PASS stage.2:', 'FAIL stage.2:'),
                        log.replace('BUDGET stage.0:','BUDGET unexpected:',1)):
            with self.assertRaises(ValueError):strict_prefill(changed,names)

    def test_summary_keeps_quality_and_numerical_failures_separate(self):
        case=dict(decode=dict(complete_document=True,ids_exact=True,text_exact=True,same_stop=True,first_teacher_argmax_difference=None),
            prefill=dict(qualified_failed_stages=['cache']),teacher_quality=dict(qualified_failures=2),
            reference_quality=dict(parse_ok=True),native_quality=dict(parse_ok=False))
        cases=[copy.deepcopy(case) for _ in range(3)]
        result=summarize(cases)
        self.assertEqual(result,dict(incomplete_cases=0,output_failures=0,qualified_prefill_failures=3,teacher_score_failures=6,quality_parse_failures=3))
        cases[0]['decode']['complete_document']=False;cases[1]['decode']['first_teacher_argmax_difference']=4
        self.assertEqual(summarize(cases)['incomplete_cases'],1)
        self.assertEqual(summarize(cases)['output_failures'],1)
        with self.assertRaises(ValueError):summarize(cases[:2])


if __name__ == '__main__':unittest.main()
