"""Stdlib helpers for the MoEdern1.5M SciFact quality measurement.

Reuses the frozen UltraBERT SciFact v2 data rules (pinned BEIR snapshot and
qrels, document = title + one ASCII space + text, ascending string-ID order,
all 300 test queries and all 5183 documents, binary canonical qrels) and the
same metric definitions (NDCG@10, MRR@10, Recall@100; ties broken by ascending
document ID). Quality evidence only: there is no gate and no original
comparison here.
"""
import csv
import hashlib
import io
import json
import math

# Identities frozen by tests/regression/ultrabert/scifact-protocol-v2.json and
# the UltraBERT preparation tool (same revision, bytes and SHA256).
BEIR_REVISION = 'b3b5335604bf5ee3c4447671af975ea25143d4f5'
QRELS_REVISION = '2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029'
QRELS_GIT_BLOB = 'c62f58c75f10c0cee23483b7269bff2902ae4080'
QRELS_BYTES = 5389
PARQUET = {'corpus': (4469916, '243324b35f03d82bd6d98a5f575966876e86cad7ce16e5333a35b1b793dc4f45'),
           'queries': (64982, '1c37956c5dc8b810b60302323c24d1a9e79e26411ba8f5ad9d0888642e2a9034')}
AUTHOR_REVISION = '68b98a56d93e0f9da0d2aab4e6c3294699a0f72e'
NOTICES = [
    ('author-LICENSE.md', 'https://raw.githubusercontent.com/allenai/scifact/' + AUTHOR_REVISION + '/LICENSE.md',
     '56ee69bc639cbab8f0b2e4b67091ec5e9390aa99a1d65f0a35e4274686c9a806'),
    ('author-README.md', 'https://raw.githubusercontent.com/allenai/scifact/' + AUTHOR_REVISION + '/README.md',
     '3f0eb7f6d009b3785fede6592918b8b583e7193d26f14173fdf0c79aba1dcc79'),
    ('BEIR-scifact-README.md', 'https://huggingface.co/datasets/BeIR/scifact/raw/' + BEIR_REVISION + '/README.md',
     'b5f57be1839fdb75867073986a019a78afe0a149950f79d1165509efc5e5cf2f'),
    ('BEIR-qrels-README.md', 'https://huggingface.co/datasets/BeIR/scifact-qrels/raw/' + QRELS_REVISION + '/README.md',
     'fd34a2830e557f1e32d46d0b34a218d280c9800ee4071579610d4ff587b4fd45')]
CORPUS_ROWS, TEST_QUERIES, BEIR_QUERIES = 5183, 300, 1109
KEYS = ('ndcg_at_10', 'mrr_at_10', 'recall_at_100')


def parquet_url(kind):
    return ('https://huggingface.co/datasets/BeIR/scifact/resolve/' + BEIR_REVISION + '/'
            + kind + '/' + kind + '-00000-of-00001.parquet')


def qrels_url():
    return 'https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/' + QRELS_REVISION + '/test.tsv'


def git_blob(content):
    return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def qrels(payload):
    """Parse canonical BEIR binary qrels; reject changed columns, scores, duplicates."""
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8')), delimiter='\t')
    if reader.fieldnames != ['query-id', 'corpus-id', 'score']:
        raise ValueError('Changed canonical qrels columns')
    result = {}
    for row in reader:
        q, d = row['query-id'], row['corpus-id']
        if row['score'] != '1' or d in result.setdefault(q, {}):
            raise ValueError('Invalid/duplicate qrel')
        result[q][d] = 1
    return result


def unique(rows, key):
    out = {}
    for row in rows:
        name = str(row[key])
        if name in out:
            raise ValueError('Duplicate ID: ' + name)
        out[name] = row
    return out


def build(corpus, queries, gold, *, corpus_rows=CORPUS_ROWS, test_queries=TEST_QUERIES, beir_queries=BEIR_QUERIES):
    """Canonical ordered model inputs. No strip, normalization, dropping or selection."""
    bc, bq = unique(corpus, '_id'), unique(queries, '_id')
    if len(bc) != corpus_rows or len(bq) != beir_queries or len(gold) != test_queries:
        raise ValueError('Changed complete SciFact inventories')
    documents = [dict(id=name, text=bc[name]['title'] + ' ' + bc[name]['text']) for name in sorted(bc)]
    rows = []
    for name in sorted(gold):
        relevant = sorted(gold[name])
        if name not in bq or not relevant or not set(relevant) <= set(bc):
            raise ValueError('Gold references an absent query/corpus ID: ' + name)
        rows.append(dict(id=name, text=bq[name]['text'], relevant=relevant))
    data = dict(queries=rows, documents=documents)
    return data, dict(all_rows_sha256=digest(data), gold_sha256=digest(gold))


def rank_scores(document_ids, values, relevant, keep=100):
    """Same definitions as the frozen UltraBERT SciFact ranking helper."""
    if (len(values) != len(document_ids) or len(set(document_ids)) != len(document_ids) or len(values) < 2
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
                ranking=[dict(id=document_ids[i], score=float(values[i])) for i in order[:keep]])


def mean_metrics(rows):
    if not rows:
        raise ValueError('Empty quality panel')
    return {k: math.fsum(r[k] for r in rows) / len(rows) for k in KEYS}


def shard(items, index, count):
    if type(index) is not int or type(count) is not int or not 0 <= index < count:
        raise ValueError('Shard index out of range')
    return [r for i, r in enumerate(items) if i % count == index]
