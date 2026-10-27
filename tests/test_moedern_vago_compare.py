#!/usr/bin/env python3
"""Stdlib unit tests for the pure MoEdern CrispEmbed-vs-author comparison helpers."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/kaggle/moedern-vago-compare'))
import compare_lib as lib  # noqa: E402


class Helpers(unittest.TestCase):
    def test_git_blob_sha1_and_sha256(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'x'; p.write_bytes(b'hello\n')
            self.assertEqual(lib.git_blob_sha1(p), 'ce013625030ba8dba906f756967f9e9ca394464a')  # `git hash-object`
            self.assertEqual(lib.sha256_file(p, chunk=2), hashlib.sha256(b'hello\n').hexdigest())

    def test_rows(self):
        self.assertEqual(lib.rows([1, 2, 3, 4], dim=2), [[1, 2], [3, 4]])
        with self.assertRaises(ValueError): lib.rows([1, 2, 3], dim=2)

    def test_vector_metrics(self):
        m = lib.vector_metrics([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.001], [0.0, 1.0]])
        self.assertTrue(m['row_count_match']); self.assertAlmostEqual(m['max_absolute'], 0.001)
        self.assertLess(m['min_row_cosine'], 1.0); self.assertGreater(m['min_row_cosine'], 0.9999)
        m = lib.vector_metrics([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])
        self.assertFalse(m['row_count_match']); self.assertIsNone(m['max_absolute'])
        self.assertEqual(lib.vector_metrics([[0.0, 0.0]], [[0.0, 0.0]])['min_row_cosine'], 1.0)
        with self.assertRaises(ValueError): lib.vector_metrics([[1.0]], [[1.0, 2.0]])

    def test_select_and_maxsim(self):
        self.assertEqual(lib.select([[1], [2], [3]], [1, 0, 1]), [[1], [3]])
        with self.assertRaises(ValueError): lib.select([[1]], [1, 0])
        q, d = [[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [1.0, 0.0]]
        self.assertAlmostEqual(lib.maxsim(q, d), 1.5)
        self.assertEqual(lib.maxsim([], d), 0.0)

    def test_token_diff(self):
        self.assertTrue(lib.token_diff([2, 30000, 5, 3], ['2', '30000', '5', '3'])['equal'])
        t = lib.token_diff([2, 9, 9, 5, 3], [2, 30000, 5, 3])
        self.assertFalse(t['equal']); self.assertEqual(t['first_difference'], 1); self.assertEqual(t['reference_window'][:2], [2, 30000])
        t = lib.token_diff([2, 3], [2, 3, 4])
        self.assertEqual(t['first_difference'], 2)

    def test_median_and_summary(self):
        self.assertEqual(lib.median([3, 1, 2]), 2); self.assertEqual(lib.median([4, 1, 2, 3]), 2.5)
        with self.assertRaises(ValueError): lib.median([])
        reps = [dict(documents_seconds=2.0, queries_seconds=1.0), dict(documents_seconds=4.0, queries_seconds=0.5),
                dict(documents_seconds=3.0, queries_seconds=2.0)]
        s = lib.summarize_timings(reps, 300, 100, 3000, 3200)
        self.assertEqual(s['documents_median_seconds'], 3.0); self.assertEqual(s['queries_median_seconds'], 1.0)
        self.assertEqual(s['documents_per_second'], 100.0); self.assertEqual(s['queries_per_second'], 100.0)
        self.assertEqual(s['total_median_seconds'], 4.5); self.assertAlmostEqual(s['tokens_per_second'], 6200 / 4.5)
        with self.assertRaises(ValueError): lib.summarize_timings([], 1, 1, 1, 1)

    def test_workload_deterministic(self):
        a = lib.make_workload(['frozen doc'], ['frozen query?'])
        b = lib.make_workload(['frozen doc'], ['frozen query?'])
        self.assertEqual(a, b)
        self.assertEqual((len(a['documents']), len(a['queries'])), (256, 256))
        self.assertEqual(a['documents'][0], 'frozen doc'); self.assertEqual(a['queries'][0], 'frozen query?')
        self.assertNotEqual(a['sha256'], lib.make_workload(['frozen doc'], ['frozen query?'], seed=1)['sha256'])
        words = [len(d.split()) for d in a['documents'][1:]]
        self.assertLessEqual(max(words), 360); self.assertLess(min(words), 10)  # varied, bounded (~<= 400 tokens)
        self.assertTrue(all(len(d.encode()) < 65536 for d in a['documents']))
        self.assertTrue(any('ß' in d or 'ä' in d for d in a['documents']))

    def test_speed_table(self):
        t = lib.speed_table([dict(runtime='author', build='native', model='F32', threads=2, batching='default',
                                  summary=dict(documents_per_second=10.0, queries_per_second=20.0, tokens_per_second=1000.0,
                                               documents_median_seconds=1.0, queries_median_seconds=0.5), peak_rss_mb=300.0),
                             dict(runtime='ours-python', build='native', model='F32', threads=4, batching='none', error='exit 1')])
        self.assertIn('| author | native | F32 | 2 | default | 10.0 | 20.0 | 1000 |', t)
        self.assertIn('error: exit 1', t)


if __name__ == '__main__':
    unittest.main()
