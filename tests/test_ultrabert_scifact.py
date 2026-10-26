#!/usr/bin/env python3
"""Lightweight synthetic controls; no corpus, model, frameworks or network."""
import copy
import hashlib
import importlib.util
import json
import io
import os
import zipfile
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import ultrabert_scifact as scorer
spec = importlib.util.spec_from_file_location('scifact_driver', ROOT / 'tools/ci-heavy/ultrabert_scifact.py')
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)


class Contract(unittest.TestCase):
    def fixture(self):
        corpus = [dict(doc_id=i, title='Title', abstract=['One.', 'Two.']) for i in range(5183)]
        bc = [dict(_id=str(i), title='Title', text='One. Two.') for i in range(5183)]
        claims = [dict(id=i, claim='Claim ' + str(i), evidence={'1':[]}, cited_doc_ids=[1]) for i in range(1109)]
        bq = [dict(_id=str(i), title='', text='Claim ' + str(i)) for i in range(1109)]
        gold = {str(i): {'1':1} for i in range(300)}
        return corpus, claims, bc, bq, gold

    def test_frozen_protocol_and_explicit_budgets(self):
        p = scorer.protocol()
        self.assertEqual(p['dataset']['corpus'], 5183)
        self.assertEqual(p['dataset']['queries'], 300)
        self.assertEqual(p['acceptance']['vector_max_abs'], .002)
        self.assertEqual(p['acceptance']['vector_cosine_min'], .99999)
        self.assertEqual(p['acceptance']['top1_agreement_min'], .99)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'protocol.json'; path.write_text('{}')
            with patch.object(scorer, 'PROTOCOL_PATH', path), self.assertRaises(ValueError):
                scorer.protocol()

    def test_exact_mapping_and_all_shards(self):
        data, mapping = scorer.equivalence(*self.fixture())
        self.assertEqual(mapping['initial_mapping_status'], 'exact')
        self.assertEqual(data['documents'][0]['text'], 'Title One. Two.')
        rows = [r for i in range(8) for r in scorer.shard_rows(data, i)]
        self.assertEqual(len(rows), 5483)
        self.assertEqual(len({r['id'] for r in rows}), 5483)
        self.assertEqual(sorted(r['id'] for r in rows), sorted(r['id'] for r in data['queries'] + data['documents']))
        with self.assertRaises(ValueError): scorer.shard_rows(data, 8)

    def test_evidence_and_citations_never_silently_replace_gold(self):
        values = self.fixture(); values[1][0]['evidence'] = {'2': []}; values[1][1]['cited_doc_ids'] = [2]
        data, mapping = scorer.equivalence(*values)
        self.assertEqual(len(mapping['author_claim_mapping_differences']), 2)
        self.assertEqual(data['queries'][0]['relevant_document_ids'], ['d:1'])
        self.assertFalse(mapping['author_claim_mapping_differences'][0]['evidence_matches'])
        self.assertFalse(mapping['author_claim_mapping_differences'][1]['cited_matches'])

    def test_text_id_missing_duplicate_fail_before_outputs(self):
        for kind in ('text', 'missing', 'duplicate', 'query', 'gold'):
            values = self.fixture()
            if kind == 'text': values[2][0]['text'] = 'One.  Two.'
            elif kind == 'missing': values[0].pop()
            elif kind == 'duplicate': values[1].append(values[1][0])
            elif kind == 'query': values[3][0]['text'] = 'changed'
            else: values[4]['0'] = {'99999': 1}
            with self.subTest(kind=kind), self.assertRaises(ValueError): scorer.equivalence(*values)

    def test_author_canonical_splits_exclude_cross_validation_copies(self):
        a, b, c = [dict(id=i, claim=str(i)) for i in range(3)]
        split = {'claims_train.jsonl':[a], 'claims_dev.jsonl':[b], 'claims_test.jsonl':[c],
                 'claims_train_1.jsonl':[a, b], 'claims_dev_1.jsonl':[c]}
        self.assertEqual(scorer.canonical_claims(split), [a, b, c])
        split['claims_test.jsonl'].append(a)
        with self.assertRaises(ValueError): scorer.canonical_claims(split)
        del split['claims_test.jsonl']
        with self.assertRaises(ValueError): scorer.canonical_claims(split)

    def test_canonical_qrels_duplicate_score_rejected(self):
        header = 'query-id\tcorpus-id\tscore\n'
        self.assertEqual(scorer.qrels((header + '1\t2\t1\n').encode()), {'1':{'2':1}})
        for lines in ('1\t2\t1\n1\t2\t1\n', '1\t2\t0\n'):
            with self.assertRaises(ValueError): scorer.qrels((header + lines).encode())

    def test_corpus_diagnostics_preserve_strict_rejection_and_exact_codepoints(self):
        values = self.fixture()
        values[0][0]['abstract'] = [' One.', 'Two. ']
        values[0][1]['title'] = 'Title\u00a0'
        with self.assertRaises(scorer.MappingMismatch) as captured:
            scorer.equivalence(*values)
        details = captured.exception.details
        self.assertEqual(details['compared_corpus_rows'], 5183)
        self.assertEqual(details['mismatch_count'], 2)
        self.assertEqual(details['all_mismatched_ids'], ['0', '1'])
        first = details['first_eight_examples'][0]['fields']['text']
        self.assertEqual(first['first_differing_index'], 0)
        self.assertEqual((first['author_codepoint'], first['beir_codepoint']), (32, ord('O')))
        self.assertTrue(first['diagnostic_only_strip_would_match'])
        self.assertFalse(details['transformations_applied'])
        self.assertFalse(details['model_outputs_produced'])
        self.assertEqual(values[0][0]['abstract'], [' One.', 'Two. '])

    def test_corpus_diagnostics_bound_examples_and_keep_all_mismatch_ids(self):
        values = self.fixture()
        for row in values[0][:12]: row['title'] += ' ' * 1000
        with self.assertRaises(scorer.MappingMismatch) as captured: scorer.equivalence(*values)
        details = captured.exception.details
        self.assertEqual(len(details['all_mismatched_ids']), 12)
        self.assertEqual(len(details['first_eight_examples']), 8)
        for example in details['first_eight_examples']:
            self.assertLessEqual(len(example['fields']['title']['author_context']), 192)

    def test_legacy_parquet_comparison_preserves_columns_and_loader_title_default(self):
        values = self.fixture(); corpus, queries = values[2:4]
        legacy = copy.deepcopy(queries)
        for row in legacy: row.pop('title'); row['metadata'] = {'ignored by original loader':True}
        result = scorer.migration_equivalence(copy.deepcopy(corpus), legacy, corpus, queries)
        self.assertTrue(result['corpus']['ids_and_columns_exact'])
        self.assertTrue(result['queries']['ids_and_columns_exact'])
        for variant in ('text', 'missing', 'duplicate'):
            before = copy.deepcopy(corpus)
            if variant == 'text': before[0]['text'] += ' '
            elif variant == 'missing': before.pop()
            else: before.append(before[0])
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                scorer.migration_equivalence(before, legacy, corpus, queries)

    def test_rank_metrics_ties_misses_and_nonfinite(self):
        ids = ['d:' + str(i).zfill(3) for i in range(110)]
        result = scorer.rank_scores(ids, [1.] * 110, [ids[0], ids[10], ids[109]])
        self.assertEqual(result['top1'], ids[0]); self.assertEqual(result['top_margin'], 0.)
        self.assertEqual(result['mrr_at_10'], 1.)
        self.assertEqual(result['recall_at_100'], 2 / 3)
        self.assertEqual(result['ndcg_at_10'], 1 / (1 + 1 / math.log2(3) + .5))
        self.assertEqual(result['missed_at_100'], [ids[-1]])
        with self.assertRaises(ValueError): scorer.rank_scores(ids, [float('nan')] * 110, [ids[0]])
        with self.assertRaises(ValueError): scorer.rank_scores(ids, [0.] * 109, [ids[0]])

    def test_quality_preserves_failures_and_exact_boundary(self):
        before = [dict(ndcg_at_10=1., mrr_at_10=1., recall_at_100=1., top1='a') for _ in range(300)]
        after = copy.deepcopy(before)
        after[0]['top1'] = 'b'; after[1]['top1'] = 'b'; after[2]['top1'] = 'b'
        self.assertTrue(scorer.quality(before, after)['passed'])
        after[3]['top1'] = 'b'; self.assertFalse(scorer.quality(before, after)['passed'])
        after = copy.deepcopy(before); after[0]['recall_at_100'] = 0.
        self.assertFalse(scorer.quality(before, after)['passed'])
        with self.assertRaises(ValueError): scorer.quality(before, after[:-1])

    def test_reuse_rejects_unreviewed_or_incomplete_service_before_download(self):
        for meta in ({'path':'wrong'}, {'path':driver.WORKFLOW,'status':'completed','conclusion':'failure'}):
            with patch.object(driver, 'api', return_value=meta), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError): driver.reuse('123', 'prepared', Path(tmp))

    def test_checked_reuse_current_job_barrier_and_reviewed_bytes(self):
        closure = {'head':'a' * 40, 'files':{'owned':'b' * 64}}
        receipt = {'source':closure, 'files':{}}
        content = (json.dumps(receipt) + '\n').encode()
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as z: z.writestr('receipt.json', content)
        payload = stream.getvalue()
        artifact = {'id':1, 'name':'ultrabert-scifact-original-0', 'expired':False,
                    'size_in_bytes':len(payload), 'digest':'sha256:' + hashlib.sha256(payload).hexdigest(),
                    'expires_at':'2027-01-01T00:00:00Z'}
        def service(path):
            if path.endswith('/jobs?per_page=100'):
                return {'jobs':[{'name':'scifact / original-shard-0','status':'completed','conclusion':'success'}]}
            if path.endswith('/artifacts?per_page=100'): return {'artifacts':[artifact]}
            return {'path':driver.CALLER_WORKFLOW,'status':'in_progress','head_sha':'a' * 40}
        def command(*args, **kwargs): kwargs['stdout'].write(payload)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'GITHUB_RUN_ID':'123'}), \
                patch.object(driver, 'api', side_effect=service), patch.object(driver, 'source', return_value=closure), \
                patch.object(driver.subprocess, 'run', side_effect=command):
            actual, proof = driver.reuse('123', artifact['name'], Path(tmp) / 'positive',
                                         report_sha=hashlib.sha256(content).hexdigest())
            self.assertEqual(actual, receipt); self.assertEqual(proof['artifact_id'], 1)
            with self.assertRaises(ValueError):
                driver.reuse('123', artifact['name'], Path(tmp) / 'wrong-sha', report_sha='0' * 64)
            with patch.object(driver, 'api', return_value={'path':driver.CALLER_WORKFLOW,'jobs':[]}):
                with self.assertRaises(ValueError): driver.reuse('123', artifact['name'], Path(tmp) / 'missing-job')

    def test_historical_receipts_and_scoped_sources_are_pinned(self):
        self.assertGreaterEqual(len(driver.PINS), 14)
        for name, expected in driver.PINS.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)

    def test_member_inventory_fails_on_changed_or_extra_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp); (target / 'a').write_text('one'); recorded = driver.inventory(target)
            driver.verify_inventory(target, recorded)
            (target / 'a').write_text('two')
            with self.assertRaises(ValueError): driver.verify_inventory(target, recorded)
            (target / 'a').write_text('one'); (target / 'extra').write_text('x')
            with self.assertRaises(ValueError): driver.verify_inventory(target, recorded)


if __name__ == '__main__': unittest.main()
