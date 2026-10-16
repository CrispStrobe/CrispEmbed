"""Model-free controls for continuation observers and strict diagnostic records."""
import copy
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/ci-heavy')]
from qwen35_book_isolation import STEPS, original_frozen, probe_names, probes, continuation
from qwen35_decode_reference import install, validate_steps


class Isolation(unittest.TestCase):
    def test_steps_reject_prefill_duplicate_out_of_order_and_bad_limits(self):
        self.assertEqual(validate_steps(STEPS, 1024), STEPS)
        for value in ((0,), (94, 94), (240, 94), (True,), (1024,), (1, 2, 3, 4, 5), '94', (float('nan'),)):
            with self.assertRaises(ValueError): validate_steps(value, 1024)

    def logs(self):
        return '\n'.join(line for name in probe_names() for line in (
            f'PROBE_METRIC {name} n=2 cos_min=1 cos_global=1 max_abs=0 mean_abs=0 rms=0 norms=1/1',
            f'PASS {name}: cos_min=1',
            f'BUDGET {name}: strict_absolute_pass=1 calibrated=0 self_max_abs=0 max_ratio=0'))

    def test_probe_order_and_count(self):
        self.assertEqual(len(probes(self.logs())), 150)
        self.assertEqual(len(set(probe_names())), 150)

    def test_probe_mutations_rejected(self):
        log = self.logs(); name = probe_names()[0]
        for text in (log.replace(name, 'missing', 1), log + '\n' + log,
                     log.replace('norms=1/1', 'norms=nan/1', 1),
                     log.replace('calibrated=0', 'calibrated=1', 1),
                     log.replace('max_ratio=0', 'max_ratio=2', 1),
                     log.replace('PASS ', 'FAIL ', 1), log.replace('self_max_abs=0', 'self_max_abs=0.01', 1)):
            with self.assertRaises(ValueError): probes(text)

    def test_failed_numerical_probe_retained(self):
        log = self.logs().replace('max_abs=0 ', 'max_abs=0.006 ', 1).replace('PASS ', 'FAIL ', 1)
        log = log.replace('strict_absolute_pass=1', 'strict_absolute_pass=0', 1).replace('max_ratio=0', 'max_ratio=3', 1)
        result = probes(log)
        self.assertFalse(result[0]['qualified_pass'])
        self.assertEqual(result[0]['max_abs'], .006)

    def original(self):
        reference = copy.deepcopy(next(c['reference'] for c in json.loads(
            (ROOT / 'tests/results/qwen35-ocr/quality.json').read_text())['cases'] if c['id'] == 'two-column-page'))
        reference['decode_probes'] = dict(steps=list(STEPS), forward_calls=683, fields_per_step=75,
                                         observer_off_ids_scores_exact=True)
        return reference

    def test_actual_frozen_fields_and_replay_requirement(self):
        reference = self.original()
        self.assertTrue(original_frozen(reference, copy.deepcopy(reference)))
        for field, value in [('generated_ids', []), ('grid', [[1, 1, 1]]), ('mean_confidence', 0), ('cpu_controls', {})]:
            changed = copy.deepcopy(reference); changed[field] = value
            with self.assertRaises(ValueError): original_frozen(changed, reference)
        changed = copy.deepcopy(reference); changed['decode_probes']['observer_off_ids_scores_exact'] = False
        with self.assertRaises(ValueError): original_frozen(changed, reference)

    def continuation_log(self):
        frozen = next(c for c in json.loads((ROOT / 'tests/results/qwen35-ocr/quality.json').read_text())['cases'] if c['id'] == 'two-column-page')
        lines = []
        for m in frozen['decode']['metrics']:
            lines.append('METRIC {stage} n={elements} cos_min={cos_min} cos_global={cos_global} max_abs={max_abs} mean_abs={mean_abs} rms={rms} norms={native_norm}/{reference_norm}'.format(**m))
        for t in frozen['decode']['teacher_forced']:
            lines.append('TEACHER step={step} selected={selected} reference={reference} same={same}'.format(**dict(t, same=int(t['same']))))
        for row in frozen['teacher_quality']['stages']:
            lines += [(('PASS' if row['qualified_pass'] else 'FAIL') + ' ' + row['stage'] + ': cos_min=1'),
                      'BUDGET {stage}: strict_absolute_pass={strict_absolute_pass} calibrated=0 self_max_abs=0 max_ratio={max_ratio}'.format(**dict(row, strict_absolute_pass=int(row['strict_absolute_pass'])))]
        lines += ['TEACHER_QUALITY stages=683 strict_failures=13', self.logs()]
        return '\n'.join(lines), frozen

    def test_baseline_continuation_preserves_actual_failures(self):
        log, frozen = self.continuation_log()
        result = continuation(log, frozen['reference'], frozen, False)
        self.assertEqual(result['teacher_quality']['qualified_failures'], 13)
        self.assertIsNone(result['first_teacher_argmax_difference'])
        for text in (log.replace('METRIC input.pixel_values ', 'METRIC other ', 1),
                     log.replace('TEACHER step=94 ', 'TEACHER step=93 ', 1),
                     log + '\nISOLATION reference_embeddings=1 test_only=1',
                     log.replace('TEACHER_QUALITY stages=683', 'TEACHER_QUALITY stages=682')):
            with self.assertRaises(ValueError): continuation(text, frozen['reference'], frozen, False)

    def test_isolation_requires_exact_input_and_no_free_output(self):
        log, frozen = self.continuation_log()
        # Same native metrics with a nonzero embedding error must never count as same-input.
        extra = '\nISOLATION reference_embeddings=1 test_only=1\nDECODE_ISOLATION finite=1 reference_embeddings=1 test_only=1 probes=150 scores=683 free_generation=0'
        with self.assertRaisesRegex(ValueError, 'not exact'): continuation(log + extra, frozen['reference'], frozen, True)
        with self.assertRaises(ValueError): continuation(log + extra + '\nFREE ids: 1', frozen['reference'], frozen, True)

    def test_observer_step_mapping_and_removal(self):
        class Module:
            def __init__(self): self.pre = []; self.post = []
            def register_forward_pre_hook(self, fn, **kwargs):
                self.pre.append(fn); return Mock(remove=lambda: self.pre.remove(fn))
            def register_forward_hook(self, fn):
                self.post.append(fn); return Mock(remove=lambda: self.post.remove(fn))
        language = Module(); language.layers = [Module() for _ in range(24)]; language.norm = Module()
        model = Module(); model.model = SimpleNamespace(language_model=language)
        saved = []
        remove, finish = install(model, (1, 3), lambda key, value: saved.append((key, value)))
        layers = [SimpleNamespace(**{k: None for k in ('keys', 'values', 'conv_states', 'recurrent_states')}) for _ in range(24)]
        # The fake modules exercise callback bookkeeping only; no tensor computation.
        class Logits:
            def __getitem__(self, key): return 'logits'
        output = SimpleNamespace(logits=Logits(), past_key_values=SimpleNamespace(layers=layers))
        for i in range(4):
            language.pre[0](None, (), {'inputs_embeds': i})
            for block in language.layers: block.post[0](None, (), i)
            language.norm.post[0](None, (), i)
            model.post[0](None, (), output)
        self.assertEqual(finish(4)['steps'], [1, 3])
        self.assertEqual([v for k, v in saved if k.endswith('inputs_embeds')], [1, 3])
        self.assertTrue(all(k.startswith(('generated.probe.1.', 'generated.probe.3.')) for k, _ in saved))
        with self.assertRaises(ValueError): finish(5)
        remove()
        self.assertFalse(language.pre or model.post or language.norm.post)
        self.assertTrue(all(not b.post for b in language.layers))


if __name__ == '__main__': unittest.main()
