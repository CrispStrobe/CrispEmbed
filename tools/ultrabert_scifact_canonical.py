"""Canonical-BEIR v2 source mapping; no model execution or normalization.

V1 remains a separate strict author-column equivalence target. This module only
prepares explicit canonical inputs and a complete source-difference journal.
"""
import hashlib
import json
from pathlib import Path

from ultrabert_scifact import digest, unique

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'tests/regression/ultrabert/scifact-protocol-v2.json'
PROTOCOL_SHA = '9efad942c987f9db417fc616b318ee632238ca99faad4c24388930dc536ef7ac'
V1_PROTOCOL_SHA = 'b2e9ab1b0cbb2d910e6b6eba971f2babab7a943fd195f0dfc1344eaee6d05ccb'
V1_HELPER_SHA = 'e944ae2458a0293a2390972ed1bb464c6ec36c54d8e664bad873f965cfeda249'


def protocol():
    payload = PROTOCOL_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PROTOCOL_SHA:
        raise ValueError('Frozen canonical-BEIR v2 protocol changed')
    old_payload = (ROOT / 'tests/regression/ultrabert/scifact-protocol-v1.json').read_bytes()
    if (hashlib.sha256(old_payload).hexdigest() != V1_PROTOCOL_SHA
            or hashlib.sha256((ROOT / 'tools/ultrabert_scifact.py').read_bytes()).hexdigest() != V1_HELPER_SHA):
        raise ValueError('Preserved v1 protocol/helper changed')
    result, old = json.loads(payload), json.loads(old_payload)
    for name in ('model', 'execution', 'acceptance', 'licenses', 'limits'):
        if result[name] != old[name]:
            raise ValueError('Inherited v1 model/budgets/license/scope differs: ' + name)
    for name in ('corpus', 'queries', 'beir_queries_total', 'beir_revision', 'qrels_revision',
                 'qrels_git_blob', 'qrels_bytes', 'gold', 'id_order'):
        if result['dataset'][name] != old['dataset'][name]:
            raise ValueError('Canonical v2 inventory/gold identity changed: ' + name)
    return result


def checked_migration(migration):
    """Require the complete actual old-loader/Parquet equality result, never fallback."""
    if not isinstance(migration, dict) or set(migration) != {'corpus', 'queries'}:
        raise ValueError('Actual complete legacy-to-Parquet migration evidence required')
    for name, count in (('corpus', 5183), ('queries', 1109)):
        row = migration[name]
        if (not isinstance(row, dict) or type(row.get('rows')) is not int or row['rows'] != count
                or row.get('ids_and_columns_exact') is not True or row.get('mismatched_ids') != []):
            raise ValueError('Legacy-to-Parquet IDs/columns did not pass: ' + name)
    return migration


def text_delta(author, canonical):
    """All differing columns get exact identities and a first-codepoint diagnostic."""
    if not isinstance(author, str) or not isinstance(canonical, str):
        raise ValueError('Source column is not text')
    index = next((i for i, (a, b) in enumerate(zip(author, canonical)) if a != b), min(len(author), len(canonical)))
    return dict(author_sha256=hashlib.sha256(author.encode()).hexdigest(),
        canonical_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        author_characters=len(author), canonical_characters=len(canonical),
        author_utf8_bytes=len(author.encode()), canonical_utf8_bytes=len(canonical.encode()),
        first_differing_index=index, author_codepoint=ord(author[index]) if index < len(author) else None,
        canonical_codepoint=ord(canonical[index]) if index < len(canonical) else None)


def equivalence(corpus, claims, beir_corpus, beir_queries, gold, migration):
    """Return canonical data and truthful provenance, not author-column equivalence.

    `migration` is mandatory and must be the complete positive report from the
    pinned legacy JSONL/Parquet comparison. The caller must separately verify and
    freeze those source payload hashes before invoking any model, as v2 declares.
    """
    spec = protocol()
    checked_migration(migration)
    author, ac = unique(corpus, 'doc_id'), unique(claims, 'id')
    bc, bq = unique(beir_corpus, '_id'), unique(beir_queries, '_id')
    if (len(bc) != 5183 or set(author) != set(bc) or len(bq) != 1109 or not set(bq) <= set(ac)
            or not isinstance(gold, dict) or len(gold) != 300):
        raise ValueError('Incomplete canonical/author source ID inventories')
    for name, row in bc.items():
        if not isinstance(row.get('title'), str) or not isinstance(row.get('text'), str):
            raise ValueError('Canonical corpus title/text is not literal text: ' + name)
        original = author[name]
        if (not isinstance(original.get('title'), str) or not isinstance(original.get('abstract'), list)
                or not all(isinstance(value, str) for value in original['abstract'])):
            raise ValueError('Author corpus columns do not follow the declared schema: ' + name)
    for name, row in bq.items():
        if row.get('title') != '' or not isinstance(row.get('text'), str) or not isinstance(ac[name].get('claim'), str):
            raise ValueError('Canonical query title/text or author claim schema differs: ' + name)
    for name, relevant in gold.items():
        if (name not in bq or not isinstance(relevant, dict) or not relevant or not set(relevant) <= set(bc)
                or any(type(score) is not int or score != 1 for score in relevant.values())):
            raise ValueError('Canonical test gold references invalid IDs/scores: ' + str(name))
    corpus_changes, query_changes, gold_changes = [], [], []
    documents = []
    for name in sorted(bc):
        original, row = author[name], bc[name]
        abstract = ' '.join(original['abstract'])
        fields = {}
        if original['title'] != row['title']:
            fields['title'] = text_delta(original['title'], row['title'])
        if abstract != row['text']:
            fields['text'] = text_delta(abstract, row['text'])
        if fields:
            corpus_changes.append(dict(id=name, fields=fields))
        # Literal stored canonical text, including leading/trailing whitespace.
        documents.append(dict(id='d:' + name, source_id=name, text=row['title'] + ' ' + row['text']))
    for name in sorted(bq):
        if ac[name]['claim'] != bq[name]['text']:
            query_changes.append(dict(id=name, fields={'text': text_delta(ac[name]['claim'], bq[name]['text'])}))
    queries = []
    for name in sorted(gold):
        original, relevant = ac[name], gold[name]
        evidence_value = original.get('evidence', {})
        cited_value = original.get('cited_doc_ids', [])
        if not isinstance(evidence_value, dict) or not isinstance(cited_value, list):
            raise ValueError('Author evidence/citation schema differs: ' + name)
        evidence = sorted(str(key) for key in evidence_value)
        cited = sorted(str(key) for key in cited_value)
        canonical = sorted(relevant)
        if evidence != canonical or cited != canonical:
            gold_changes.append(dict(query_id=name, canonical_beir=canonical, author_evidence=evidence,
                author_cited_doc_ids=cited, evidence_matches=evidence == canonical, cited_matches=cited == canonical))
        queries.append(dict(id='q:' + name, source_id=name, text=bq[name]['text'],
                            relevant_document_ids=['d:' + key for key in canonical]))
    data = dict(queries=queries, documents=documents)
    ordered = queries + documents
    if len(ordered) != 5483 or len({row['id'] for row in ordered}) != 5483:
        raise ValueError('Canonical all5483 row coverage differs')
    columns = dict(corpus=[dict(id=name, title=bc[name]['title'], text=bc[name]['text']) for name in sorted(bc)],
                   queries=[dict(id=name, title=bq[name]['title'], text=bq[name]['text']) for name in sorted(bq)])
    mapping = dict(schema='crispembed-scifact-canonical-source-mapping-v2', protocol_sha256=PROTOCOL_SHA,
        canonical_input_source='Pinned canonical BEIR stored columns, literal declared joining and query prefix',
        corpus=5183, test_queries=300, all_beir_queries=1109, prepared_rows=5483,
        corpus_ids_exact=True, author_claim_ids_cover_all1109_beir_queries=True,
        author_corpus_columns_exact=not corpus_changes, author_all1109_claim_text_exact=not query_changes,
        author_test_evidence_exact=all(row['evidence_matches'] for row in gold_changes),
        author_test_citations_exact=all(row['cited_matches'] for row in gold_changes),
        author_corpus_column_differences=corpus_changes, author_query_text_differences=query_changes,
        author_claim_mapping_differences=gold_changes,
        unused_author_query_ids=sorted(set(ac) - set(bq)), legacy_parquet_equivalence=migration,
        canonical_qrels_used=True, source_columns_normalized=False, rows_dropped=False,
        strict_v1_author_equivalence_claimed=False, v1_rejections_preserved=True,
        author_corpus_mismatch_count=len(corpus_changes), author_query_mismatch_count=len(query_changes),
        author_gold_mismatch_count=len(gold_changes), canonical_columns_sha256=digest(columns),
        all_rows_sha256=digest(data), ordered_ids_sha256=digest([row['id'] for row in ordered]),
        canonical_corpus_ids_sha256=digest(sorted(bc)), canonical_all_query_ids_sha256=digest(sorted(bq)),
        canonical_test_query_ids_sha256=digest(sorted(gold)), gold_sha256=digest(gold),
        model_input_utf8_sha256={row['id']: hashlib.sha256(((spec['model']['query_prompt'] if row['id'].startswith('q:') else '')
            + row['text']).encode()).hexdigest() for row in ordered},
        applicable_terms=spec['licenses'], inherited_acceptance=spec['acceptance'],
        model_outputs_produced=False, source_mapping_only=True,
        scope='Source mapping before model outputs; runner verifies pinned provenance and original-first execution')
    return data, mapping
