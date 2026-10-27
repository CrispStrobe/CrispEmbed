#!/usr/bin/env python3
"""Drive the model author's own runtime (moedern_gguf.py + ggml_infer_v10 --serve).

Runs in the clean venv (numpy + tokenizers). One process per measurement so the
peak RSS of this Python process and of the engine child are attributable.

  author_runner.py outputs VAGO_DIR ENGINE MODEL TOKENIZER THREADS CASES_JSON OUT_JSON
  author_runner.py bench   VAGO_DIR ENGINE MODEL TOKENIZER THREADS WORKLOAD_JSON DOC_BATCH QUERY_BATCH REPS OUT_JSON

DOC_BATCH/QUERY_BATCH 0 = the wrapper's own defaults (16 documents, 64 queries).
"""
import json
import resource
import sys
import time


def load(vago_dir, engine, model, tokenizer, threads):
    sys.path.insert(0, vago_dir)
    from moedern_gguf import MoEdernColBERTGGUF
    t = time.perf_counter()
    m = MoEdernColBERTGGUF(model, tokenizer_path=tokenizer, engine_path=engine, threads=int(threads))
    return m, time.perf_counter() - t


def rss():
    # Linux ru_maxrss is KiB. CHILDREN covers the engine once it has been waited for.
    return dict(python_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                engine_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


def outputs(vago_dir, engine, model, tokenizer, threads, cases_path, out):
    import numpy as np
    m, load_s = load(vago_dir, engine, model, tokenizer, threads)
    cases = json.load(open(cases_path))
    res = dict(load_seconds=load_s, skiplist_ids=[int(x) for x in m.skiplist_ids], cases={})
    for c in cases:
        if c['role'] == 'query':
            ids = m.tokenize_queries([c['text']])[0]
            vec = m.encode_queries([c['text']])[0]
            keep = [1] * len(ids)
        else:
            ids = m.tokenize_documents([c['text']])[0]
            vec = m.encode_documents([c['text']])[0]
            keep = [int(x) for x in ~np.isin(ids, m.skiplist_ids)]
        # Unfiltered engine rows for alignment diagnostics (same request as the wrapper sends).
        raw = m._run([ids])[0]
        res['cases'][c['id']] = dict(role=c['role'], ids=[int(x) for x in ids], keep=keep,
                                     vectors=vec.astype('float32').tolist(), engine_rows=raw.astype('float32').tolist())
    m.close()
    res.update(rss())
    json.dump(res, open(out, 'w'))


def bench(vago_dir, engine, model, tokenizer, threads, workload_path, doc_batch, query_batch, reps, out):
    w = json.load(open(workload_path))
    docs, queries = w['documents'], w['queries']
    m, load_s = load(vago_dir, engine, model, tokenizer, threads)
    dkw = {} if int(doc_batch) == 0 else dict(batch_size=int(doc_batch))
    qkw = {} if int(query_batch) == 0 else dict(batch_size=int(query_batch))
    doc_tokens = sum(len(x) for x in m.tokenize_documents(docs))
    query_tokens = sum(len(x) for x in m.tokenize_queries(queries))
    max_doc_tokens = max(len(x) for x in m.tokenize_documents(docs))
    t = time.perf_counter()
    m.encode_documents(docs[:16], **dkw); m.encode_queries(queries[:16], **qkw)
    warm = time.perf_counter() - t
    timed = []
    for _ in range(int(reps)):
        t0 = time.perf_counter(); d = m.encode_documents(docs, **dkw)
        t1 = time.perf_counter(); q = m.encode_queries(queries, **qkw)
        t2 = time.perf_counter()
        if len(d) != len(docs) or len(q) != len(queries): raise RuntimeError('incomplete output')
        timed.append(dict(documents_seconds=t1 - t0, queries_seconds=t2 - t1))
    m.close()
    res = dict(load_seconds=load_s, warmup_seconds=warm, reps=timed, n_documents=len(docs), n_queries=len(queries),
               document_tokens=doc_tokens, query_tokens=query_tokens, max_document_tokens=max_doc_tokens,
               doc_batch=dkw.get('batch_size', 'default(16)'), query_batch=qkw.get('batch_size', 'default(64)'))
    res.update(rss())
    json.dump(res, open(out, 'w'))


if __name__ == '__main__':
    {'outputs': outputs, 'bench': bench}[sys.argv[1]](*sys.argv[2:])
