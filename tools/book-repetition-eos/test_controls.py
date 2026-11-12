import unittest
import audit


class Controls(unittest.TestCase):
    def test_periodicity_and_minimum_span(self):
        self.assertIsNone(audit.repetition([1, 2, 3] * 3)['first_periodic_run'])
        report = audit.repetition([8, 9] + [1, 2, 3] * 5)
        self.assertEqual(report['longest_periodic_run'], dict(start=2, end_exclusive=17, period=3, span=15, whole_cycles=5))
        self.assertNotIn('ids', str(report))
        self.assertIsNone(audit.repetition(list(range(200)))['first_periodic_run'])

    def test_shared_input_boundary(self):
        self.assertEqual(audit.common_prefix([1, 2, 3], [1, 2, 4]), 2)
        # Row2 has identical consumed tokens0/1 although next greedy IDs differ.
        self.assertEqual(audit.common_prefix([], [1]), 0)

    def test_score_numerics(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy unavailable; worker pins numpy separately')
        values = np.full(audit.VOCAB, -1000, dtype='float32')
        values[audit.EOS[0]] = 0
        values[audit.EOS[1]] = -1
        result = audit.score_stats(values)
        self.assertEqual([row['rank_strict'] for row in result['eos']], [1, 2])
        self.assertAlmostEqual(result['either_eos_probability'], 1)
        self.assertAlmostEqual(result['top_two_gap'], 1)
        shifted = audit.score_stats(values + 100)
        self.assertAlmostEqual(result['entropy'], shifted['entropy'])
        self.assertEqual(result['eos'], shifted['eos'])
        values[0] = float('nan')
        with self.assertRaises(ValueError):
            audit.score_stats(values)


if __name__ == '__main__':
    unittest.main()
