"""Pure, stdlib-only helpers for the MoEdern CrispEmbed-vs-VAGO runtime comparison.

No model execution here: vector/MaxSim statistics, token-ID diffs, timing
summaries, the deterministic benchmark workload and the summary table. Unit
tested by tests/test_moedern_vago_compare.py.
"""
import hashlib
import math
import random

DIM = 128
QUERY_LENGTH = 32
SEED = 20261027


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(path):
    """Git object id of a file (HF records LFS-less files by blob sha1 only)."""
    with open(path, 'rb') as f:
        data = f.read()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def rows(flat, dim=DIM):
    if len(flat) % dim:
        raise ValueError('flat vector length is not a multiple of %d' % dim)
    return [list(flat[i:i + dim]) for i in range(0, len(flat), dim)]


def cosine(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return float(na == nb)
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def vector_metrics(a, b):
    """Row-wise comparison of two lists of equal-width rows.

    Row counts are reported; numeric metrics only when they match (no silent
    truncation or alignment)."""
    out = dict(rows_a=len(a), rows_b=len(b), row_count_match=len(a) == len(b))
    if not out['row_count_match'] or not a:
        out.update(max_absolute=None, min_row_cosine=None)
        return out
    if any(len(x) != len(y) for x, y in zip(a, b)):
        raise ValueError('row width differs')
    out['max_absolute'] = max(max(abs(x - y) for x, y in zip(ra, rb)) for ra, rb in zip(a, b))
    out['min_row_cosine'] = min(cosine(ra, rb) for ra, rb in zip(a, b))
    return out


def select(rows_, keep):
    if len(rows_) != len(keep):
        raise ValueError('mask length differs from row count')
    return [r for r, k in zip(rows_, keep) if k]


def maxsim(q, d):
    if not q or not d:
        return 0.0
    return sum(max(sum(x * y for x, y in zip(a, b)) for b in d) for a in q)


def token_diff(candidate, reference):
    """Exact token-ID comparison; on mismatch report the first differing position."""
    candidate, reference = [int(x) for x in candidate], [int(x) for x in reference]
    first = next((i for i, (x, y) in enumerate(zip(candidate, reference)) if x != y), None)
    if first is None and len(candidate) != len(reference):
        first = min(len(candidate), len(reference))
    out = dict(equal=candidate == reference, len_candidate=len(candidate), len_reference=len(reference), first_difference=first)
    if first is not None:
        out['candidate_window'] = candidate[max(0, first - 2):first + 4]
        out['reference_window'] = reference[max(0, first - 2):first + 4]
    return out


def median(xs):
    xs = sorted(xs)
    if not xs:
        raise ValueError('median of empty sequence')
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def summarize_timings(reps, n_docs, n_queries, doc_tokens, query_tokens):
    """reps: [{'documents_seconds': s, 'queries_seconds': s}, ...] (warm-up already excluded)."""
    if not reps:
        raise ValueError('no timed repetitions')
    d = median([r['documents_seconds'] for r in reps])
    q = median([r['queries_seconds'] for r in reps])
    t = median([r['documents_seconds'] + r['queries_seconds'] for r in reps])
    rate = lambda n, s: n / s if s > 0 else None
    return dict(repetitions=len(reps), documents_median_seconds=d, queries_median_seconds=q, total_median_seconds=t,
                documents_per_second=rate(n_docs, d), queries_per_second=rate(n_queries, q),
                document_tokens_per_second=rate(doc_tokens, d), query_tokens_per_second=rate(query_tokens, q),
                tokens_per_second=rate(doc_tokens + query_tokens, t))


_DE = ('Die Stadt liegt am Ufer eines großen Flusses und hat eine lange Geschichte '
       'Viele Menschen arbeiten in der Industrie oder im Handel während andere studieren '
       'Im Winter fällt oft Schnee in den Bergen und die Straßen werden gesperrt '
       'Das Krankenhaus wurde erweitert damit mehr Patienten behandelt werden können '
       'Die Bundesregierung beschloss neue Regeln für erneuerbare Energien und Netzausbau '
       'Unsere Großmutter backt jeden Sonntag einen Apfelkuchen mit Zimt und Sahne').split()
_EN = ('The committee reviewed the proposal and asked for more detailed cost estimates '
       'Researchers measured the temperature of the lake every morning for three years '
       'A small library opened near the station and quickly became popular with students '
       'Engineers replaced the old bridge because heavy trucks had damaged its supports '
       'Most visitors arrive by train although the airport handles several daily flights '
       'The recipe needs flour butter sugar eggs and a pinch of salt').split()
_QUESTION = {'de': ('Wie', 'Warum', 'Wann', 'Wo', 'Was'), 'en': ('How', 'Why', 'When', 'Where', 'What')}
# Word counts chosen so that the longest documents stay near ~400 tokens (< 512).
DOC_WORD_BUCKETS = (6, 16, 32, 64, 110, 170, 230, 280)


def _sentence(rng, words, n):
    picked = [rng.choice(words) for _ in range(n)]
    picked[0] = picked[0][:1].upper() + picked[0][1:]
    return ' '.join(picked) + rng.choice(('.', '.', '.', '?', '!', ';'))


def _text(rng, lang, n_words):
    words, out, left = (_DE if lang == 'de' else _EN), [], n_words
    while left > 0:
        n = min(left, rng.randint(4, 16))
        s = _sentence(rng, words, n)
        if rng.random() < 0.15:
            s = s[:-1] + ', %d%%' % rng.randint(1, 99) + s[-1]
        out.append(s)
        left -= n
    return ' '.join(out)


def make_workload(frozen_documents, frozen_queries, n_documents=256, n_queries=256, seed=SEED):
    """Deterministic German/English document/query lists; frozen texts come first."""
    rng = random.Random(seed)
    documents, queries = list(frozen_documents)[:n_documents], list(frozen_queries)[:n_queries]
    i = 0
    while len(documents) < n_documents:
        lang = 'de' if i % 2 == 0 else 'en'
        n = DOC_WORD_BUCKETS[i % len(DOC_WORD_BUCKETS)]
        documents.append(_text(rng, lang, max(1, n + rng.randint(-n // 4, n // 4))))
        i += 1
    i = 0
    while len(queries) < n_queries:
        lang = 'de' if i % 2 == 0 else 'en'
        words = _DE if lang == 'de' else _EN
        q = ' '.join([rng.choice(_QUESTION[lang])] + [rng.choice(words) for _ in range(rng.randint(2, 14))]) + '?'
        queries.append(q)
        i += 1
    return dict(seed=seed, documents=documents, queries=queries,
                sha256=hashlib.sha256('\0'.join(documents + ['\1'] + queries).encode('utf-8')).hexdigest())


def _fmt(v, digits=1):
    if v is None:
        return 'n/a'
    if isinstance(v, float):
        return '%.*f' % (digits, v)
    return str(v)


def speed_table(rows_):
    """rows_: dicts with runtime, build, model, threads, batching, summary, peak_rss_mb or error."""
    head = '| runtime | build | model | threads | batching | docs/s | queries/s | tokens/s | doc s (median) | query s (median) | peak RSS MB |\n'
    head += '|---|---|---|---|---|---|---|---|---|---|---|\n'
    body = ''
    for r in rows_:
        s = r.get('summary') or {}
        cells = [r.get('runtime'), r.get('build'), r.get('model'), r.get('threads'), r.get('batching')]
        if r.get('error'):
            cells += ['error: ' + str(r['error'])[:60]] + [''] * 5
        else:
            cells += [_fmt(s.get('documents_per_second')), _fmt(s.get('queries_per_second')), _fmt(s.get('tokens_per_second'), 0),
                      _fmt(s.get('documents_median_seconds'), 3), _fmt(s.get('queries_median_seconds'), 3), _fmt(r.get('peak_rss_mb'), 0)]
        body += '| ' + ' | '.join(_fmt(c) for c in cells) + ' |\n'
    return head + body
