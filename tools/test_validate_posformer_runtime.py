"""Dependency-light rejection controls; model inference runs only in hosted CI."""
import copy
import unittest
from validate_posformer_runtime import APP, BASE, ORIGINAL_SOURCE, CORRECT_IDS, compare
from compare_handwriting_encoders_test import report
from audit_handwriting_vocabulary import FROZEN_MANIFEST_SHA256


def fixtures():
    left, right = report('default'), report('default')
    corrected = 'd' * 40
    for value, bridge, library in ((left, BASE, 'a' * 64), (right, corrected, 'b' * 64)):
        value.update(source=corrected, bridge_source=bridge, library_sha256=library,
                     bridge_patch_sha256=None, corpus_manifest_sha256=FROZEN_MANIFEST_SHA256,
                     exact_matches=7)
        for index, identifier in enumerate(sorted(CORRECT_IDS)):
            value['cases'][index]['id'] = identifier
            value['corpus']['cases'][index]['id'] = identifier
        for row in value['cases'][7:]:
            row.update(latex='z', status='different_transcription')
    provenance = {'bridge_source': corrected, 'baseline_source': BASE, 'app_source': APP,
                  'baseline_source_sha256': ORIGINAL_SOURCE, 'production_probes_absent': True,
                  'baseline_library_sha256': 'a' * 64, 'production_library_sha256': 'b' * 64}
    return left, right, provenance


class ProductionReportsTest(unittest.TestCase):
    def test_preserved_original_matches(self):
        result = compare(*fixtures())
        self.assertEqual(result['baseline_exact_matches'], 7)
        self.assertEqual(result['corrected_exact_matches'], 7)
        self.assertTrue(result['original_correct_ids_preserved'])

    def test_wrong_source_model_manifest_and_diagnostic_library_rejected(self):
        for key, value in [('bridge_source', BASE), ('model_sha256', 'f' * 64),
                           ('corpus_manifest_sha256', 'f' * 64), ('bridge_patch_sha256', 'f' * 64),
                           ('library_sha256', 'a' * 64), ('runtime_failures', 1)]:
            left, right, provenance = fixtures()
            right[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(left, right, provenance)

    def test_partial_or_changed_corpus_rejected(self):
        for mutation in ('drop', 'reference', 'image'):
            left, right, provenance = fixtures()
            if mutation == 'drop':
                right['cases'].pop()
            elif mutation == 'reference':
                right['cases'][0]['reference_latex'] = 'tampered'
            else:
                right['cases'][0]['image_sha256'] = 'f' * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare(left, right, provenance)

    def test_equal_count_with_different_correct_id_rejected(self):
        left, right, provenance = fixtures()
        right['cases'][0].update(latex='z', status='different_transcription')
        right['cases'][7].update(latex='x+1', status='exact_match')
        with self.assertRaisesRegex(ValueError, 'regressed'):
            compare(left, right, provenance)

    def test_production_probes_or_wrong_harness_rejected(self):
        for key, value in [('production_probes_absent', False), ('app_source', 'f' * 40),
                           ('baseline_source_sha256', 'f' * 64)]:
            left, right, provenance = fixtures()
            provenance[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(left, right, provenance)


if __name__ == '__main__':
    unittest.main()
