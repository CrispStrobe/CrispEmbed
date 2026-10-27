#!/usr/bin/env python3
"""Drive CrispEmbed's public Python CrispMoEdern (one encode call per text: that is our API).

  ours_runner.py outputs REPO_PYTHON LIB MODEL THREADS CASES_JSON PAIRS_JSON OUT_JSON
  ours_runner.py bench   REPO_PYTHON LIB MODEL THREADS WORKLOAD_JSON REPS OUT_JSON

Requires the CRISPEMBED_MOEDERN_* gates in the environment (set by worker.py).
"""
import json
import resource
import sys
import time


def load(repo_python, lib, model, threads):
    sys.path.insert(0, repo_python)
    from crispembed import CrispMoEdern
    t = time.perf_counter()
    m = CrispMoEdern(model, n_threads=int(threads), lib_path=lib, auto_download=False, offline=True)
    return m, time.perf_counter() - t


def rss(): return dict(python_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def outputs(repo_python, lib, model, threads, cases_path, pairs_path, out):
    m, load_s = load(repo_python, lib, model, threads)
    res = dict(load_seconds=load_s, cases={}, scores={})
    texts = {}
    for c in json.load(open(cases_path)):
        r = m.encode(c['text'], c['role']); texts[c['id']] = c['text']
        res['cases'][c['id']] = {k: r[k] for k in ('role', 'ids', 'scoring_mask', 'selected_vectors', 'all_vectors', 'n_tokens', 'n_selected')}
    for p in json.load(open(pairs_path)):
        res['scores'][p['query'] + '|' + p['document']] = m.score(texts[p['query']], texts[p['document']])
    m.close(); res.update(rss())
    json.dump(res, open(out, 'w'))


def bench(repo_python, lib, model, threads, workload_path, reps, out):
    w = json.load(open(workload_path))
    docs, queries = w['documents'], w['queries']
    m, load_s = load(repo_python, lib, model, threads)
    t = time.perf_counter()
    for x in docs[:16]: m.encode(x, 'document')
    for x in queries[:16]: m.encode(x, 'query')
    warm = time.perf_counter() - t
    timed, doc_tokens, query_tokens, max_doc = [], 0, 0, 0
    for _ in range(int(reps)):
        t0 = time.perf_counter(); dt = [m.encode(x, 'document')['n_tokens'] for x in docs]
        t1 = time.perf_counter(); qt = [m.encode(x, 'query')['n_tokens'] for x in queries]
        t2 = time.perf_counter()
        timed.append(dict(documents_seconds=t1 - t0, queries_seconds=t2 - t1))
        doc_tokens, query_tokens, max_doc = sum(dt), sum(qt), max(dt)
    m.close()
    res = dict(load_seconds=load_s, warmup_seconds=warm, reps=timed, n_documents=len(docs), n_queries=len(queries),
               document_tokens=doc_tokens, query_tokens=query_tokens, max_document_tokens=max_doc, doc_batch=1, query_batch=1)
    res.update(rss())
    json.dump(res, open(out, 'w'))


if __name__ == '__main__':
    {'outputs': outputs, 'bench': bench}[sys.argv[1]](*sys.argv[2:])
