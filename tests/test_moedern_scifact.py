"""Stdlib tests for the MoEdern SciFact metric/data helpers on tiny authored fixtures."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import moedern_scifact as S  # noqa: E402

QRELS = b'query-id\tcorpus-id\tscore\nq1\td2\t1\nq2\td1\t1\nq2\td3\t1\n'


class Metrics(unittest.TestCase):
    def test_qrels(self):
        self.assertEqual(S.qrels(QRELS), {'q1': {'d2': 1}, 'q2': {'d1': 1, 'd3': 1}})
        for bad in (b'qid\tcid\tscore\n', b'query-id\tcorpus-id\tscore\nq\td\t2\n',
                    b'query-id\tcorpus-id\tscore\nq\td\t1\nq\td\t1\n'):
            with self.assertRaises(ValueError):
                S.qrels(bad)

    def test_single_relevant_rank2(self):
        r = S.rank_scores(['d1', 'd2', 'd3'], [0.9, 0.5, 0.1], ['d2'])
        self.assertAlmostEqual(r['ndcg_at_10'], 1 / math.log2(3))
        self.assertEqual((r['mrr_at_10'], r['recall_at_100'], r['top1']), (0.5, 1.0, 'd1'))

    def test_two_relevant_and_ties_by_id(self):
        # d3 and d1 tie; ascending ID puts d1 first.
        r = S.rank_scores(['d3', 'd2', 'd1'], [1.0, 0.0, 1.0], ['d1', 'd3'])
        self.assertEqual(r['relevant_ranks'], {'d1': 1, 'd3': 2})
        self.assertAlmostEqual(r['ndcg_at_10'], 1.0)
        self.assertEqual(r['mrr_at_10'], 1.0)

    def test_cutoffs(self):
        ids = ['d%03d' % i for i in range(150)]
        values = [-i for i in range(150)]
        r = S.rank_scores(ids, values, ['d011', 'd120'])
        self.assertEqual((r['ndcg_at_10'], r['mrr_at_10'], r['recall_at_100']), (0.0, 0.0, 0.5))  # rank 12 and 121
        r = S.rank_scores(ids, values, ['d009', 'd099', 'd100'])
        ideal = 1 + 1 / math.log2(3) + 0.5
        self.assertAlmostEqual(r['ndcg_at_10'], (1 / math.log2(11)) / ideal)
        self.assertAlmostEqual(r['mrr_at_10'], 0.1)
        self.assertAlmostEqual(r['recall_at_100'], 2 / 3)
        self.assertEqual(len(r['ranking']), 100)

    def test_mean_and_invalid(self):
        rows = [dict(ndcg_at_10=1., mrr_at_10=1., recall_at_100=1.), dict(ndcg_at_10=0., mrr_at_10=.5, recall_at_100=0.)]
        self.assertEqual(S.mean_metrics(rows), dict(ndcg_at_10=.5, mrr_at_10=.75, recall_at_100=.5))
        for args in ((['a', 'a'], [1, 2], ['a']), (['a', 'b'], [1, float('nan')], ['a']),
                     (['a', 'b'], [1, 2], ['c']), (['a', 'b'], [1, 2], [])):
            with self.assertRaises(ValueError):
                S.rank_scores(*args)
        with self.assertRaises(ValueError):
            S.mean_metrics([])

    def test_build_and_shard(self):
        corpus = [dict(_id='d2', title='T2', text=' x '), dict(_id='d1', title='', text='y'),
                  dict(_id='d3', title='T3', text='z')]
        queries = [dict(_id='q2', title='', text='Q2'), dict(_id='q1', title='', text='Q1'),
                   dict(_id='q9', title='', text='unused')]
        data, hashes = S.build(corpus, queries, S.qrels(QRELS), corpus_rows=3, test_queries=2, beir_queries=3)
        self.assertEqual(data['documents'], [dict(id='d1', text=' y'), dict(id='d2', text='T2  x '),
                                             dict(id='d3', text='T3 z')])
        self.assertEqual(data['queries'], [dict(id='q1', text='Q1', relevant=['d2']),
                                           dict(id='q2', text='Q2', relevant=['d1', 'd3'])])
        self.assertEqual(len(hashes['all_rows_sha256']), 64)
        with self.assertRaises(ValueError):
            S.build(corpus, queries, S.qrels(QRELS))
        self.assertEqual(S.shard([0, 1, 2, 3, 4], 1, 2), [1, 3])
        with self.assertRaises(ValueError):
            S.shard([1], 2, 2)

    def test_git_blob(self):
        self.assertEqual(S.git_blob(b''), 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')


if __name__ == '__main__':
    unittest.main()
