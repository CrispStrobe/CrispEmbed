import tempfile
from pathlib import Path
import unittest
import audit


class Controls(unittest.TestCase):
    def test_native_status_admission(self):
        source = dict(source_sha='e3164d09de4578370082e42c5954a77a6cd643e3',
                      status='measured_neutral_book_text_trace', all654_scores_observer_neutral=True,
                      historical_reference_sha256='9b88195d5a8058dea4fab87472009d3826bc1fa3b5d0575aa2e33698316d5722')
        for status in ['failed', 'measured_extended_planner_book']:
            with self.assertRaisesRegex(ValueError, 'native neutrality admission'):
                audit.analyze(None, source, None, dict(status=status))

    def test_head_mapping(self):
        self.assertEqual(audit.permutation('a', 32)[:4], [0, 16, 1, 17])
        for role, width in [('qkv', 8192), ('z', 4096), ('a', 32), ('b', 32)]:
            self.assertEqual(sorted(audit.permutation(role, width)), list(range(width)))
        self.assertEqual(audit.permutation('qkv', 8192)[:4096], list(range(4096)))

    def test_artifact_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / 'stage'
            file.write_bytes(b'good')
            manifest = {'stage': dict(bytes=4, sha256=audit.sha(file))}
            self.assertEqual(audit.owned(root, 'stage', manifest), file)
            file.write_bytes(b'evil')
            with self.assertRaises(ValueError):
                audit.owned(root, 'stage', manifest)
            with self.assertRaises(ValueError):
                audit.owned(root, '../stage', {'../stage': manifest['stage']})

    def test_shapes_and_magnitudes(self):
        import numpy as np
        a = np.array([[1, 2]], dtype='<f4')
        b = np.array([[1, 4]], dtype='<f4')
        row = audit.delta(a, b)
        self.assertAlmostEqual(row['error_rms'], 2**.5)
        self.assertEqual(row['error_max_abs'], 2)
        self.assertFalse(row['bit_exact'])
        with self.assertRaises(ValueError):
            audit.delta(a, b.reshape(2, 1))
        graph = dict(ne={'beta-0': [1, 32, 1, 1]}, arrays={'beta-0': np.arange(32, dtype='<f4')})
        self.assertEqual(audit.source_rows(graph, 'beta-0', 32, 1, 'b').shape, (1, 32))
        with self.assertRaises(ValueError):
            audit.source_rows(graph, 'beta-0', 32, 2, 'b')
        graph['ne']['beta-0'] = [2, 16, 1, 1]
        with self.assertRaises(ValueError):
            audit.source_rows(graph, 'beta-0', 32, 1, 'b')


if __name__ == '__main__':
    unittest.main()
