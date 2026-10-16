"""Model-free validation of timing attribution and negative report controls."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/ci-heavy'))
from qwen35_profiling import parse_benchmark, check_bytes, paired_summary


def fixture(active=True):
    lines = ['BENCH ids: 11 248046', 'BENCH confidences: 0.8 0.9', 'BENCH utf8_hex:48',
             'BENCH stop=1 mean_confidence=0.85 bytes=1 load_ms=1 recognize_ms=10']
    events, matrices = [], []
    if active:
        for phase, event, calls, ns in [('other', 'request_file', 1, 9950000), ('other', 'request_raw', 1, 9900000),
            ('other', 'file_decode', 1, 1000), ('other', 'preprocess', 1, 1000), ('other', 'prompt', 1, 1000),
            ('vision', 'vision', 1, 2000000), ('other', 'embedding', 1, 1000), ('prefill', 'prefill', 1, 2000000),
            ('other', 'selection', 2, 2000), ('other', 'cached_prepare', 1, 1000),
            ('decode', 'cached_decode', 1, 2000000), ('other', 'text_decode', 1, 1000)]:
            events.append(f'PROFILE phase={phase} event={event} calls={calls} nanoseconds={ns}')
        for phase in ('vision', 'prefill', 'decode'):
            for event in ('project_total', 'project_setup', 'project_input', 'project_materialize', 'project_graph',
                          'project_compute', 'project_output', 'project_cleanup', 'source_buffer', 'source_read', 'source_convert', 'source_validate'):
                ns = 1000000 if event == 'project_total' else 100000
                events.append(f'PROFILE phase={phase} event={event} calls=1 nanoseconds={ns}')
            name, source, f32 = ('v.test', 16, 16) if phase == 'vision' else ('l.embed_tokens.weight', 508559360, 1017118720)
            matrices.append(f'PROFILE_MATRIX phase={phase} tensor={name} calls=1 source_bytes={source} f32_bytes={f32} materialize_ns=80000')
    return '\n'.join(lines + events + matrices + [f'PROFILE_COMPLETE active={int(active)} dropped=0 events={len(events)} matrices={len(matrices)}']) + '\n'


class Profiling(unittest.TestCase):
    def test_active_attribution_and_bytes(self):
        measured = parse_benchmark(fixture())
        check_bytes(measured, {'v.test': dict(shape=[2, 2], dtype='float32'),
            'l.embed_tokens.weight': dict(shape=[248320, 1024], dtype='float16')})
        self.assertEqual(measured['ids'], [11, 248046])

    def test_inactive_is_empty(self):
        self.assertFalse(parse_benchmark(fixture(False))['profile_active'])

    def reject(self, before, after):
        with self.assertRaises((ValueError, KeyError)):
            parse_benchmark(fixture().replace(before, after))

    def test_missing_completion(self):
        self.reject('PROFILE_COMPLETE', 'REMOVED_COMPLETE')

    def test_dropped_data(self):
        self.reject('dropped=0', 'dropped=1')

    def test_inactive_with_data(self):
        self.reject('active=1', 'active=0')

    def test_duplicate_phase(self):
        self.reject('event=project_graph', 'event=project_materialize')

    def test_wrong_counts(self):
        self.reject('event=selection calls=2', 'event=selection calls=1')

    def test_nan(self):
        self.reject('mean_confidence=0.85', 'mean_confidence=nan')

    def test_byte_length(self):
        self.reject('bytes=1 load_ms=', 'bytes=2 load_ms=')

    def test_missing_projection_event(self):
        self.reject('event=project_compute', 'event=unknown_compute')

    def test_impossible_projection_sum(self):
        self.reject('event=project_setup calls=1 nanoseconds=100000', 'event=project_setup calls=1 nanoseconds=9000000')

    def test_wrong_weight_bytes(self):
        with self.assertRaises(ValueError):
            check_bytes(parse_benchmark(fixture()), {'v.test': dict(shape=[2, 2], dtype='float16'),
                'l.embed_tokens.weight': dict(shape=[248320, 1024], dtype='float16')})


class PairedSummary(unittest.TestCase):
    def rows(self):
        return [dict(pair=pair, parallel_f16=enabled, profile_active=False, events=[], matrices=[],
            recognize_ms=20 if enabled else 30, resources={'max_rss_kib': 100})
            for pair, enabled in [(0, False), (0, True), (1, True), (1, False), (2, False), (2, True)]]

    def test_unprofiled_conversion_pairs(self):
        summary = paired_summary(self.rows(), parallel=True)
        self.assertEqual(summary['parallel_on']['median_ms'], 20)
        self.assertEqual(summary['parallel_off']['median_ms'], 30)
        self.assertEqual(summary['median_paired_on_over_off'], 2/3)

    def test_reject_missing_pair(self):
        with self.assertRaises(ValueError): paired_summary(self.rows()[:-1], parallel=True)

    def test_reject_profiled_conversion(self):
        rows = self.rows(); rows[0]['profile_active'] = True
        with self.assertRaises(ValueError): paired_summary(rows, parallel=True)

    def test_historical_profile_summaries(self):
        import json
        report = json.loads((Path(__file__).resolve().parent / 'results/qwen35-ocr/profiling.json').read_text())
        for item in report['benchmarks']:
            self.assertEqual(paired_summary(item['runs']), item['summary'])


if __name__ == '__main__':
    unittest.main()
