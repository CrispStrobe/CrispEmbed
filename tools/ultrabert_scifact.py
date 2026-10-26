"""Versioned SciFact protocol, mapping and ranking rules; standard library only."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'tests/regression/ultrabert/scifact-protocol-v1.json'
PROTOCOL_SHA = 'b2e9ab1b0cbb2d910e6b6eba971f2babab7a943fd195f0dfc1344eaee6d05ccb'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def protocol():
    payload = PROTOCOL_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PROTOCOL_SHA:
        raise ValueError('Frozen SciFact protocol changed')
    return json.loads(payload)


def unique(rows, key):
    result = {}
    for row in rows:
        name = str(row[key])
        if name in result:
            raise ValueError('Duplicate dataset ID: ' + name)
        result[name] = row
    return result


def canonical_claims(extracted):
    """Author README's canonical split; cross-validation files repeat IDs."""
    names = ('claims_train.jsonl', 'claims_dev.jsonl', 'claims_test.jsonl')
    if any(name not in extracted for name in names):
        raise ValueError('Author canonical train/dev/test split absent')
    rows = [row for name in names for row in extracted[name]]
    unique(rows, 'id')  # Still reject duplicates within or across canonical splits.
    return rows


class MappingMismatch(ValueError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def text_difference(author, beir):
    """Bounded diagnostics only: never apply a candidate transformation."""
    if not isinstance(author, str) or not isinstance(beir, str):
        raise ValueError('Corpus column is not text')
    index = next((i for i, (a, b) in enumerate(zip(author, beir)) if a != b), min(len(author), len(beir)))
    return dict(author_sha256=hashlib.sha256(author.encode()).hexdigest(),
                beir_sha256=hashlib.sha256(beir.encode()).hexdigest(),
                author_characters=len(author), beir_characters=len(beir), first_differing_index=index,
                author_context=author[max(0, index-64):index+128],
                beir_context=beir[max(0, index-64):index+128],
                author_codepoint=ord(author[index]) if index < len(author) else None,
                beir_codepoint=ord(beir[index]) if index < len(beir) else None,
                diagnostic_only_strip_would_match=author.strip() == beir,
                diagnostic_only_whitespace_collapse_would_match=' '.join(author.split()) == beir)


def corpus_diagnostics(author, beir):
    mismatches, examples, title_count, text_count = [], [], 0, 0
    for name in sorted(author):
        row, reference = author[name], beir[name]
        abstract = ' '.join(row['abstract'])
        title_bad, text_bad = row['title'] != reference['title'], abstract != reference['text']
        if not title_bad and not text_bad:
            continue
        mismatches.append(name); title_count += title_bad; text_count += text_bad
        if len(examples) < 8:
            fields = {}
            if title_bad: fields['title'] = text_difference(row['title'], reference['title'])
            if text_bad: fields['text'] = text_difference(abstract, reference['text'])
            examples.append(dict(id=name, fields=fields))
    return dict(compared_corpus_rows=len(author), mismatch_count=len(mismatches),
                title_mismatches=title_count, text_mismatches=text_count,
                all_mismatched_ids=mismatches, first_eight_examples=examples,
                transformations_applied=False, model_outputs_produced=False)


def migration_equivalence(legacy_corpus, legacy_queries, corpus, queries):
    """The published old HF loader only supplies missing title=''; no cleanup."""
    report = {}
    for name, before, after, count in [('corpus', legacy_corpus, corpus, 5183),
                                       ('queries', legacy_queries, queries, 1109)]:
        old, current = unique(before, '_id'), unique(after, '_id')
        if len(old) != count or set(old) != set(current):
            raise ValueError('Legacy BEIR/Parquet inventories differ: ' + name)
        changed = [key for key in sorted(old) if old[key].get('title', '') != current[key]['title']
                   or old[key]['text'] != current[key]['text']]
        report[name] = dict(rows=count, ids_and_columns_exact=not changed, mismatched_ids=changed)
        if changed:
            raise MappingMismatch('Legacy BEIR/Parquet columns differ: ' + name, report)
    return report


def qrels(payload):
    reader = csv.DictReader(io.StringIO(payload.decode()), delimiter='\t')
    if reader.fieldnames != ['query-id', 'corpus-id', 'score']:
        raise ValueError('Changed canonical qrels columns')
    result = {}
    for row in reader:
        q, d = row['query-id'], row['corpus-id']
        if row['score'] != '1' or d in result.setdefault(q, {}):
            raise ValueError('Invalid/duplicate qrel')
        result[q][d] = 1
    return result


def equivalence(corpus, claims, beir_corpus, beir_queries, gold):
    """Exact texts/IDs mandatory. Gold differences journaled before outputs.

    Evidence is not assumed equal to citation IDs or the retrieval gold. The
    pinned canonical BEIR mapping remains the declared evaluation target.
    """
    author = unique(corpus, 'doc_id')
    bc, bq = unique(beir_corpus, '_id'), unique(beir_queries, '_id')
    ac = unique(claims, 'id')
    if len(author) != 5183 or set(author) != set(bc) or len(bq) != 1109 or len(gold) != 300:
        raise ValueError('Changed complete SciFact inventories')
    diagnostic = corpus_diagnostics(author, bc)
    if diagnostic['mismatch_count']:
        raise MappingMismatch('Earliest corpus text mapping mismatch: ' + diagnostic['all_mismatched_ids'][0], diagnostic)
    documents = []
    for name in sorted(author):
        row = author[name]
        abstract = ' '.join(row['abstract'])
        documents.append(dict(id='d:' + name, source_id=name, text=row['title'] + ' ' + abstract))
    for name, row in sorted(bq.items()):
        if name not in ac or ac[name]['claim'] != row['text'] or row['title'] != '':
            raise ValueError('Earliest claim text mapping mismatch: ' + name)
    queries, differences = [], []
    for name, relevant in sorted(gold.items()):
        if name not in bq or not relevant or not set(relevant) <= set(author):
            raise ValueError('Gold references an absent query/corpus ID')
        row = ac[name]
        evidence = sorted(str(d) for d in row.get('evidence', {}))
        cited = sorted(str(d) for d in row.get('cited_doc_ids', []))
        canonical = sorted(relevant)
        if evidence != canonical or cited != canonical:
            differences.append(dict(query_id=name, canonical_beir=canonical, author_evidence=evidence,
                                    author_cited_doc_ids=cited, evidence_matches=evidence == canonical,
                                    cited_matches=cited == canonical))
        queries.append(dict(id='q:' + name, source_id=name, text=row['claim'],
                            relevant_document_ids=['d:' + d for d in canonical]))
    data = dict(queries=queries, documents=documents)
    mapping = dict(corpus_ids_and_text_exact=True, all1109_beir_query_ids_and_text_exact=True,
                   test_queries=300, corpus=5183, canonical_qrels_used=True,
                   author_claim_mapping_differences=differences,
                   initial_mapping_status='exact' if not differences else 'author gold derivation differs; canonical BEIR gold retained',
                   all_rows_sha256=digest(data), gold_sha256=digest(gold))
    return data, mapping


def shard_rows(data, shard):
    if type(shard) is not int or not 0 <= shard < 8:
        raise ValueError('Shard must be0..7')
    return [r for i, r in enumerate(data['queries'] + data['documents']) if i % 8 == shard]


def rank_scores(document_ids, values, relevant):
    if (len(values) != len(document_ids) or len(set(document_ids)) != len(document_ids)
            or not relevant or not set(relevant) <= set(document_ids)
            or not all(math.isfinite(float(v)) for v in values)):
        raise ValueError('Invalid complete ranking inputs')
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), document_ids[i]))
    ranks = {document_ids[i]: n + 1 for n, i in enumerate(order)}
    best = min(ranks[d] for d in relevant)
    dcg = math.fsum(1 / math.log2(ranks[d] + 1) for d in relevant if ranks[d] <= 10)
    ideal = math.fsum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    return dict(ndcg_at_10=dcg / ideal, mrr_at_10=1 / best if best <= 10 else 0.,
                recall_at_100=sum(ranks[d] <= 100 for d in relevant) / len(relevant),
                top1=document_ids[order[0]], top_margin=float(values[order[0]]) - float(values[order[1]]),
                relevant_ranks={d: ranks[d] for d in sorted(relevant)},
                missed_at_10=[d for d in sorted(relevant) if ranks[d] > 10],
                missed_at_100=[d for d in sorted(relevant) if ranks[d] > 100],
                ranking100=[dict(id=document_ids[i], score=float(values[i])) for i in order[:100]])


def quality(before, after):
    if not before or len(before) != len(after):
        raise ValueError('Incomplete quality panel')
    keys = ('ndcg_at_10', 'mrr_at_10', 'recall_at_100')
    baseline = {k: math.fsum(r[k] for r in before) / len(before) for k in keys}
    candidate = {k: math.fsum(r[k] for r in after) / len(after) for k in keys}
    agreement = sum(a['top1'] == b['top1'] for a, b in zip(before, after)) / len(before)
    passed = all(baseline[k] - candidate[k] <= .001 + 1e-15 for k in keys) and agreement >= .99
    return dict(original=baseline, native=candidate, top1_agreement=agreement, passed=passed,
                degradation={k: baseline[k] - candidate[k] for k in keys})
