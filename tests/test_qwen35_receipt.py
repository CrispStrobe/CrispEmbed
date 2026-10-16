"""Model-free cap, probability, score verdict and frozen-prefix controls."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/ci-heavy'))
from qwen35_receipt import add_confidences, frozen_prefix, teacher_quality
spec = importlib.util.spec_from_file_location('receipt_reference', ROOT/'tools/dump_qwen35_ocr_page_reference.py')
reference_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference_module)

class Receipt(unittest.TestCase):
    def test_caps(self):
        reference_module.validate_token_cap(512)
        reference_module.validate_token_cap(1024, long_document=True)
        for cap in (0, -1, 513, 1024, True, 1.5):
            with self.assertRaises(ValueError): reference_module.validate_token_cap(cap)
        with self.assertRaises(ValueError): reference_module.validate_token_cap(1025, long_document=True)

    def confidences(self, line='FREE confidences: 0.5 0.75\n', **fields):
        measured = dict(native_ids=[11, 248046], native_mean_confidence=0.625, **fields)
        reference = dict(generated_ids=[11,248046], selected_token_confidences=[0.5,0.75])
        return add_confidences(line, measured, reference)

    def test_confidence_roundtrip(self):
        self.assertEqual(self.confidences()['confidence_comparison']['max_abs'], 0)

    def test_bad_confidences(self):
        for line in ('FREE confidences: .5\n', 'FREE confidences: nan .75\n', 'FREE confidences: .5 1.1\n',
                     'FREE confidences: .5 .74\n', 'FREE confidences: .5 .75\nFREE confidences: .5 .75\n'):
            with self.assertRaises(ValueError): self.confidences(line)

    def test_probabilities_after_id_divergence_not_aligned(self):
        measured = dict(native_ids=[11,12], native_mean_confidence=.625)
        ref = dict(generated_ids=[11,248046], selected_token_confidences=[.5,.7])
        self.assertEqual(add_confidences('FREE confidences: .5 .75\n',measured,ref)['confidence_comparison']['aligned_prefix_tokens'],1)

    def quality(self, verdict='PASS', strict=1, ratio=0.5, failures=0):
        return (f'{verdict} generated.scores.0: cos_min=1\n'
            f'BUDGET generated.scores.0: strict_absolute_pass={strict} calibrated=0 self_max_abs=0 max_ratio={ratio}\n'
            f'TEACHER_QUALITY stages=1 strict_failures={failures}\n')

    def test_quality(self):
        self.assertEqual(teacher_quality(self.quality(),1)['qualified_failures'],0)

    def test_real_quality_failure_retained(self):
        self.assertEqual(teacher_quality(self.quality('FAIL',0,2.5,1),1)['qualified_failures'],1)

    def test_bad_quality_verdicts(self):
        log=self.quality()
        for bad in (log+log,log.replace('calibrated=0','calibrated=1'),log.replace('strict_failures=0','strict_failures=1'),
                    log.replace('max_ratio=0.5','max_ratio=2'),log.replace('generated.scores.0','generated.scores.1'),
                    log.replace('TEACHER_QUALITY','INCOMPLETE')):
            with self.assertRaises(ValueError):teacher_quality(bad,1)

    def prefix(self):
        old=dict(transformers='pinned',tokenizers='pinned',torch='pinned',dtype='float32',attention='eager',
            image_sha256='hash',image_size=[1,1],grid=[[1,1,1]],patches=1,prompt='',rendered_prompt='image',eos_ids=[99],
            cpu_controls={'backend_off':[]},generated_ids=[11,12],selected_token_confidences=[.5,.75],decoded='old')
        original=copy.deepcopy(old);original['generated_ids']+=[99];original['selected_token_confidences']+=[1.]
        measured=dict(native_ids=[11,12,99],native_confidences=[.5,.75,1.],metrics=[dict(stage=str(i)) for i in range(80)],
            teacher_forced=[dict(step=i) for i in range(3)])
        before=dict(native_ids=[11,12],metrics=copy.deepcopy(measured['metrics'][:79]),teacher_forced=copy.deepcopy(measured['teacher_forced'][:2]),native_mean_confidence=.625)
        return original,measured,dict(reference=old),before,[.5],'old'

    def test_frozen_prefix(self):
        self.assertEqual(frozen_prefix(*self.prefix())['tokens'],2)

    def test_changed_prefix(self):
        for location in ('ids','scores','confidence','original'):
            args=list(self.prefix())
            if location=='ids':args[1]['native_ids'][0]=3
            if location=='scores':args[1]['metrics'][0]['stage']='changed'
            if location=='confidence':args[1]['native_confidences'][1]=.74
            if location=='original':args[0]['cpu_controls']={}
            with self.assertRaises(ValueError):frozen_prefix(*args)

if __name__=='__main__':unittest.main()
