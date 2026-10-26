"""Synthetic, weight-free controls for inactive canonical SciFact v2 preparation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import ultrabert_scifact as v1
import ultrabert_scifact_canonical as canonical


class CanonicalControls(unittest.TestCase):
    def fixture(self):
        corpus = [dict(doc_id=i, title='  Title ' + str(i), abstract=['One.', 'Two.  ']) for i in range(5183)]
        beir = [dict(_id=str(i), title='  Title ' + str(i), text='One. Two.  ') for i in range(5183)]
        claims = [dict(id=i, claim=' Claim ' + str(i) + ' ', evidence={'1': []}, cited_doc_ids=[1]) for i in range(1109)]
        queries = [dict(_id=str(i), title='', text=' Claim ' + str(i) + ' ') for i in range(1109)]
        gold = {str(i): {'1': 1} for i in range(300)}
        old_corpus, old_queries = copy.deepcopy(beir), copy.deepcopy(queries)
        # Exactly the historical HF loader's missing-title default is permitted.
        for row in old_queries:
            row.pop('title')
        migration = v1.migration_equivalence(old_corpus, old_queries, beir, queries)
        return [corpus, claims, beir, queries, gold, migration]

    def test_new_protocol_preserves_all_budgets_licenses_and_v1_bytes(self):
        p, old = canonical.protocol(), v1.protocol()
        self.assertEqual(p['version'], 2)
        self.assertIn('Conditional source preparation only', p['activation'])
        for name in ('model', 'execution', 'acceptance', 'licenses', 'limits'):
            self.assertEqual(p[name], old[name], name)
        self.assertEqual(hashlib.sha256((ROOT / p['preserved_v1']['protocol']).read_bytes()).hexdigest(), canonical.V1_PROTOCOL_SHA)
        self.assertEqual(hashlib.sha256((ROOT / p['preserved_v1']['helper']).read_bytes()).hexdigest(), canonical.V1_HELPER_SHA)
        with tempfile.TemporaryDirectory() as d:
            altered = Path(d) / 'protocol.json'; altered.write_text('{}')
            with patch.object(canonical, 'PROTOCOL_PATH', altered), self.assertRaises(ValueError):
                canonical.protocol()

    def test_exact_canonical_rows_prefix_and_whitespace_are_preserved(self):
        values = self.fixture(); data, mapping = canonical.equivalence(*values)
        self.assertEqual(len(data['queries']), 300); self.assertEqual(len(data['documents']), 5183)
        self.assertEqual(len(mapping['model_input_utf8_sha256']), 5483)
        self.assertEqual(data['documents'][0]['text'], '  Title 0 One. Two.  ')
        self.assertEqual(data['queries'][0]['text'], ' Claim 0 ')
        self.assertEqual(mapping['model_input_utf8_sha256']['q:0'],
                         hashlib.sha256((canonical.protocol()['model']['query_prompt'] + ' Claim 0 ').encode()).hexdigest())
        self.assertTrue(mapping['author_corpus_columns_exact']); self.assertTrue(mapping['author_all1109_claim_text_exact'])
        self.assertFalse(mapping['strict_v1_author_equivalence_claimed'])
        self.assertFalse(mapping['source_columns_normalized']); self.assertFalse(mapping['rows_dropped'])
        self.assertFalse(mapping['model_outputs_produced']); self.assertFalse(mapping['activation_performed'])
        self.assertEqual(mapping['gold_sha256'], v1.digest(values[4]))
        actual = [r for i in range(8) for r in v1.shard_rows(data, i)]
        self.assertEqual(len(actual), 5483); self.assertEqual(len({r['id'] for r in actual}), 5483)

    def test_all_author_differences_journaled_without_changing_canonical_inputs(self):
        values = self.fixture(); expected, clean = canonical.equivalence(*values)
        for i in range(13):
            values[0][i]['title'] = 'different author title ' + str(i)
            values[0][i]['abstract'] = ['different', 'author abstract']
            values[1][i]['claim'] = 'different author claim ' + str(i)
        # Include a query outside test qrels: the full1109-query provenance is checked.
        values[1][1000]['claim'] = 'changed non-test author claim'
        values[1][0]['evidence'] = {'2': []}; values[1][1]['cited_doc_ids'] = [2]
        actual, journal = canonical.equivalence(*values)
        self.assertEqual(actual, expected)
        for key in ('all_rows_sha256', 'ordered_ids_sha256', 'canonical_columns_sha256', 'gold_sha256', 'model_input_utf8_sha256'):
            self.assertEqual(journal[key], clean[key], key)
        self.assertEqual(journal['author_corpus_mismatch_count'], 13)
        self.assertEqual(journal['author_query_mismatch_count'], 14)
        self.assertEqual(journal['author_gold_mismatch_count'], 2)
        self.assertEqual({r['id'] for r in journal['author_corpus_column_differences']}, {str(i) for i in range(13)})
        self.assertEqual({r['id'] for r in journal['author_query_text_differences']}, {str(i) for i in range(13)} | {'1000'})
        for row in journal['author_corpus_column_differences']:
            self.assertEqual(set(row['fields']), {'title', 'text'})
            self.assertNotEqual(row['fields']['text']['author_sha256'], row['fields']['text']['canonical_sha256'])
        self.assertFalse(journal['author_corpus_columns_exact']); self.assertFalse(journal['author_all1109_claim_text_exact'])
        self.assertFalse(journal['author_test_evidence_exact']); self.assertFalse(journal['author_test_citations_exact'])
        with self.assertRaises(v1.MappingMismatch):
            v1.equivalence(*values[:5])  # The same author-column mismatch remains rejected by strict v1.

    def test_required_exact_migration_has_no_fallback(self):
        values = self.fixture()
        with self.assertRaises(TypeError): canonical.equivalence(*values[:5])
        for migration in (None, {}, {'corpus': values[-1]['corpus']},
                          dict(values[-1], queries=dict(values[-1]['queries'], ids_and_columns_exact=False)),
                          dict(values[-1], corpus=dict(values[-1]['corpus'], rows=5182)),
                          dict(values[-1], corpus=dict(values[-1]['corpus'], ids_and_columns_exact=1)),
                          dict(values[-1], corpus=dict(values[-1]['corpus'], mismatched_ids=['0']))):
            with self.subTest(migration=migration), self.assertRaises(ValueError):
                canonical.equivalence(*values[:5], migration)

    def test_legacy_canonical_column_change_rejects_before_mapping(self):
        values = self.fixture(); old = copy.deepcopy(values[2]); old[0]['text'] += ' '
        with self.assertRaises(v1.MappingMismatch):
            v1.migration_equivalence(old, values[3], values[2], values[3])
        old = copy.deepcopy(values[3]); old[0]['text'] = 'changed'
        with self.assertRaises(v1.MappingMismatch):
            v1.migration_equivalence(values[2], old, values[2], values[3])

    def test_missing_extra_duplicate_and_malformed_data_are_rejected(self):
        for variant in ('author-missing', 'author-extra', 'author-duplicate', 'claims-missing', 'claims-duplicate',
                        'canonical-missing', 'canonical-duplicate', 'query-missing', 'query-duplicate',
                        'query-title', 'corpus-null', 'author-null', 'gold-missing', 'gold-doc', 'gold-score'):
            values = self.fixture()
            if variant == 'author-missing': values[0].pop()
            elif variant == 'author-extra': values[0].append(dict(doc_id=99999, title='extra', abstract=['extra']))
            elif variant == 'author-duplicate': values[0].append(copy.deepcopy(values[0][0]))
            elif variant == 'claims-missing': values[1].pop()
            elif variant == 'claims-duplicate': values[1].append(copy.deepcopy(values[1][0]))
            elif variant == 'canonical-missing': values[2].pop()
            elif variant == 'canonical-duplicate': values[2].append(copy.deepcopy(values[2][0]))
            elif variant == 'query-missing': values[3].pop()
            elif variant == 'query-duplicate': values[3].append(copy.deepcopy(values[3][0]))
            elif variant == 'query-title': values[3][0]['title'] = 'unexpected nonempty query title'
            elif variant == 'corpus-null': values[2][0]['text'] = None
            elif variant == 'author-null': values[0][0]['abstract'] = [None]
            elif variant == 'gold-missing': values[4].pop('0')
            elif variant == 'gold-doc': values[4]['0'] = {'99999': 1}
            else: values[4]['0'] = {'1': 2}
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                canonical.equivalence(*values)

    def test_duplicate_texts_and_non_test_author_claims_are_preserved(self):
        values = self.fixture(); values[2][1].update(title=values[2][0]['title'], text=values[2][0]['text'])
        values[1].append(dict(id=99999, claim='unused author test claim'))
        values[-1] = v1.migration_equivalence(copy.deepcopy(values[2]), values[3], values[2], values[3])
        data, mapping = canonical.equivalence(*values)
        docs = {row['id']: row for row in data['documents']}
        self.assertEqual(docs['d:0']['text'], docs['d:1']['text'])
        self.assertNotEqual(docs['d:0']['id'], docs['d:1']['id'])
        self.assertEqual(mapping['unused_author_query_ids'], ['99999'])
        self.assertEqual(len(data['documents']), 5183)


if __name__ == '__main__': unittest.main()
