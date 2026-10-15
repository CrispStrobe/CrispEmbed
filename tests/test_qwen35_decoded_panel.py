"""Model-free checks that decoded measurements cannot falsely claim parity."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
from qwen35_decoded_panel import validate


class DecodedPanel(unittest.TestCase):
    def setUp(self):
        frozen = json.loads((ROOT / 'tests/results/qwen35-ocr/page-parity.json').read_text())
        self.reference = copy.deepcopy(frozen['cases'][0]['reference'])
        self.reference['mean_confidence'] = 0.9
        self.measured = copy.deepcopy(frozen['arms'][0]['cases'][0])
        self.cap = frozen['cases'][0]['max_new_tokens']

    def check_invalid(self, change):
        change(self.measured)
        with self.assertRaises(ValueError):
            validate(self.measured, self.reference, self.cap)

    def test_original_output(self):
        self.assertTrue(validate(self.measured, self.reference, self.cap)['complete_document'])

    def test_false_id_equality(self):
        self.check_invalid(lambda m: m['native_ids'].__setitem__(0, (m['native_ids'][0] + 1) % 248320))

    def test_false_text_equality(self):
        self.check_invalid(lambda m: m.update(native_text='false', native_decoded_bytes=5))

    def test_byte_length(self):
        self.check_invalid(lambda m: m.update(native_decoded_bytes=1))

    def test_eos_stop(self):
        self.check_invalid(lambda m: m.update(stop=2))

    def test_nonfinite_confidence(self):
        self.check_invalid(lambda m: m.update(native_mean_confidence=float('nan')))

    def test_nonfinite_score(self):
        self.check_invalid(lambda m: m['metrics'][0].update(max_abs=float('inf')))

    def test_missing_teacher(self):
        self.check_invalid(lambda m: m['teacher_forced'].pop())

    def test_false_teacher_verdict(self):
        self.check_invalid(lambda m: m['teacher_forced'][0].update(same=False))

    def test_real_mismatch_retained(self):
        self.measured['native_ids'][0] = (self.measured['native_ids'][0] + 1) % 248320
        self.measured.update(ids_exact=False, native_text='different', native_decoded_bytes=9, text_exact=False)
        result = validate(self.measured, self.reference, self.cap)
        self.assertFalse(result['ids_exact'])
        self.assertEqual(result['first_id_difference'], 0)
        self.assertFalse(result['text_exact'])

    def test_cap_prefix_is_not_complete(self):
        count = len(self.measured['native_ids'])
        self.measured['native_ids'][-1] = 1
        self.reference.update(generated_ids=self.measured['native_ids'][:], reached_eos=False)
        self.measured.update(stop=2)
        self.measured['teacher_forced'][-1].update(reference=1, selected=1, same=True)
        result = validate(self.measured, self.reference, count)
        self.assertFalse(result['complete_document'])
        self.assertTrue(result['cap_control'])


if __name__ == '__main__':
    unittest.main()
