#!/usr/bin/env python3
"""MoEdern1.5M SciFact retrieval quality on the published alias (heavy-cpu.yml).

Quality evidence only: no gate, no default change, no original comparison.

Steps: pinned BEIR SciFact snapshot/qrels + notices (same identities as the
frozen UltraBERT v2 protocol) -> root library build with LLAMAFILE ON -> alias
`moedern-1.5m` resolved into an empty fresh cache, SHA256 verified -> all 300
queries (role query) and 5183 documents (role document) through the public C
API via Python CrispMoEdern at 2 threads -> exact brute-force MaxSim over the
selected vectors -> NDCG@10, MRR@10, Recall@100 and timings.

Args: [--shard I --shards N] encode only a modulo slice of documents and write
its vectors (not needed at the expected cost; default is the whole panel).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import moedern_scifact as S  # noqa: E402

ALIAS = 'moedern-1.5m'
MODEL_FILE = 'moedern-1.5m-text-f32.gguf'
MODEL_SHA = '0e2b8773abd8c78a5825280f9a73cb808b88d56f0675a8e4bb7f0509f5b6cc51'
MODEL_BYTES = 7037952
HUB = 'cstr/SauerkrautLM-MoEdernColBERT-1.5M-CrispEmbed-GGUF@6f3b35143b3afdeeded7f77216b1ec072028fd5f'
GATES = ('CRISPEMBED_MOEDERN_PUBLIC_TEXT', 'CRISPEMBED_MOEDERN_PACKED', 'CRISPEMBED_MOEDERN_TEXT_COMPOSED',
         'CRISPEMBED_MOEDERN_TEXT_FRONTEND', 'CRISPEMBED_MOEDERN_COLBERT')
THREADS = 2
PREFLIGHT_ROWS = 40
BUDGET_SECONDS = 270 * 60
NATIVE_CHECK_PAIRS = 12


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n')


def fetch(url, path, *, size=None, checksum=None, cap=20 << 20):
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as out:
        count = 0
        while block := response.read(1 << 20):
            count += len(block)
            if count > cap:
                raise ValueError('Download exceeds bound: ' + url)
            out.write(block)
    if (size is not None and path.stat().st_size != size) or (checksum and sha256(path) != checksum):
        raise ValueError('Downloaded immutable bytes differ: ' + path.name)
    return dict(url=url, bytes=path.stat().st_size, sha256=sha256(path))


def dataset(out, scratch):
    import pyarrow.parquet as pq
    sources, raw = {}, scratch / 'data'
    for name, url, checksum in S.NOTICES:
        sources[name] = fetch(url, out / 'notices' / name, checksum=checksum)
    shutil.copy2(ROOT / 'docs/moedern-scifact.md', out / 'notices' / 'ATTRIBUTION.md')
    tables = {}
    for kind, (size, checksum) in S.PARQUET.items():
        target = raw / (kind + '.parquet')
        sources['BEIR/' + kind] = fetch(S.parquet_url(kind), target, size=size, checksum=checksum)
        tables[kind] = pq.read_table(target).to_pylist()
    target = raw / 'test.tsv'
    sources['BEIR/qrels'] = fetch(S.qrels_url(), target, size=S.QRELS_BYTES)
    content = target.read_bytes()
    if S.git_blob(content) != S.QRELS_GIT_BLOB:
        raise ValueError('Canonical BEIR qrels immutable Git blob differs')
    data, hashes = S.build(tables['corpus'], tables['queries'], S.qrels(content))
    return data, dict(sources=sources, **hashes)


def build_library(out, scratch):
    b = scratch / 'root'
    t0 = time.time()
    subprocess.run(['cmake', '-S', str(ROOT), '-B', str(b), '-DCMAKE_BUILD_TYPE=Release',
                    '-DCRISPEMBED_BUILD_SHARED=ON', '-DBUILD_SHARED_LIBS=OFF', '-DGGML_NATIVE=OFF',
                    '-DCRISPEMBED_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DGGML_METAL=OFF', '-DGGML_CUDA=OFF',
                    '-DGGML_LLAMAFILE=ON'], check=True, stdout=subprocess.DEVNULL)
    log = out / 'build.txt'
    with log.open('w') as f:
        subprocess.run(['cmake', '--build', str(b), '--target', 'crispembed-shared', '--parallel', '4'],
                       check=True, stdout=f, stderr=subprocess.STDOUT)
    if 'GGML_LLAMAFILE:BOOL=ON' not in (b / 'CMakeCache.txt').read_text():
        raise ValueError('LLAMAFILE is not ON')
    lib = b / 'libcrispembed.so'
    return lib, dict(seconds=time.time() - t0, llamafile=True, library_sha256=sha256(lib))


def encode(model, rows, role, np):
    vectors, counts, seconds, tokens = [], [], [], []
    for row in rows:
        t = time.perf_counter()
        r = model.encode(row['text'], role)
        seconds.append(time.perf_counter() - t)
        v = np.asarray(r['selected_vectors'], dtype=np.float32).reshape(r['n_selected'], 128)
        vectors.append(v)
        counts.append(r['n_selected'])
        tokens.append(r['n_tokens'])
    return vectors, dict(rows=len(rows), seconds=sum(seconds), max_row_seconds=max(seconds),
                         tokens=sum(tokens), max_tokens=max(tokens), at_512_tokens=sum(n == 512 for n in tokens),
                         selected=sum(counts)), seconds


def maxsim(queries, documents, np):
    """Exact brute-force MaxSim: sum over query vectors of the max dot over each document's vectors."""
    flat = np.concatenate(documents, axis=0)
    starts = np.cumsum([0] + [len(d) for d in documents[:-1]])
    out = np.empty((len(queries), len(documents)), dtype=np.float64)
    for i, q in enumerate(queries):
        sim = flat @ q.T  # tokens x query vectors (float32)
        out[i] = np.maximum.reduceat(sim, starts, axis=0).astype(np.float64).sum(axis=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = str(THREADS)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'pyarrow==21.0.0'], check=True)
    import numpy as np
    subprocess.run([sys.executable, str(ROOT / 'tests/test_moedern_scifact.py')], check=True)
    started = time.time()
    data, data_receipt = dataset(out, scratch)
    write(out / 'data.json', data_receipt)
    lib, build = build_library(out, scratch)

    cache = scratch / 'fresh-cache'
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True)
    os.environ['CRISPEMBED_CACHE_DIR'] = str(cache)
    for g in GATES:
        os.environ[g] = '1'
    sys.path.insert(0, str(ROOT / 'python'))
    from crispembed import CrispMoEdern, CrispEmbed
    t0 = time.time()
    model = CrispMoEdern(ALIAS, n_threads=THREADS, lib_path=str(lib), auto_download=True, offline=False)
    resolved = Path(CrispEmbed.resolve_model(ALIAS, auto_download=False, lib_path=str(lib))).resolve()
    if cache.resolve() not in resolved.parents or resolved.name != MODEL_FILE:
        raise ValueError('Alias did not resolve into the fresh cache: ' + resolved.name)
    if (resolved.stat().st_size, sha256(resolved)) != (MODEL_BYTES, MODEL_SHA):
        raise ValueError('Resolved model bytes/SHA256 differ')
    resolution = dict(alias=ALIAS, hub=HUB, sha256=MODEL_SHA, bytes=MODEL_BYTES, fresh_cache=True,
                      seconds=time.time() - t0, cache_members=sorted(p.name for p in cache.rglob('*') if p.is_file()))

    docs = S.shard(data['documents'], args.shard, args.shards)
    queries = data['queries'] if args.shard == 0 else []
    # Preflight: time the longest documents first, project and refuse an over-budget run.
    longest = sorted(docs, key=lambda r: (-len(r['text'].encode()), r['id']))[:PREFLIGHT_ROWS]
    _, pre, _ = encode(model, longest, 'document', np)
    projected = pre['seconds'] / len(longest) * (len(docs) + len(queries))
    write(out / 'preflight.json', dict(rows=len(longest), timing=pre, projected_encode_seconds=projected,
                                       budget_seconds=BUDGET_SECONDS))
    if projected > BUDGET_SECONDS:
        raise SystemExit(f'Projected {projected:.0f}s exceeds budget; rerun with --shards')

    qv, qtime, _ = encode(model, queries, 'query', np) if queries else ([], None, [])
    dv, dtime, _ = encode(model, docs, 'document', np)
    timing = dict(build=build, resolution_seconds=resolution['seconds'], queries=qtime, documents=dtime)
    if args.shards != 1:
        np.savez(out / f'shard-{args.shard}-of-{args.shards}.npz', ids=np.array([d['id'] for d in docs]),
                 counts=np.array([len(v) for v in dv]), vectors=np.concatenate(dv))
        write(out / f'shard-{args.shard}.json', dict(timing=timing, resolution=resolution))
        return 0

    t = time.time()
    scores = maxsim(qv, dv, np)
    timing['maxsim_seconds'] = time.time() - t
    ids = [d['id'] for d in docs]
    index = {d: i for i, d in enumerate(ids)}
    per_query = []
    for i, q in enumerate(queries):
        r = S.rank_scores(ids, scores[i].tolist(), q['relevant'])
        per_query.append(dict(query_id=q['id'], **r))
    metrics = S.mean_metrics(per_query)
    # Sanity: the native pair scorer on a few pairs against the brute-force score (recorded, not gated).
    check = []
    for q in queries[:NATIVE_CHECK_PAIRS // 2]:
        for d in (q['relevant'][0], per_query[queries.index(q)]['top1']):
            native = model.score(q['text'], data['documents'][index[d]]['text'])
            check.append(dict(query_id=q['id'], doc_id=d, native=native, brute=float(scores[queries.index(q), index[d]]),
                              abs_diff=abs(native - float(scores[queries.index(q), index[d]]))))
    model.close()
    timing['total_seconds'] = time.time() - started
    report = dict(schema='crispembed-moedern-scifact-quality-v1', status='quality evidence only; no gate',
                  model=resolution, threads=THREADS, gates=list(GATES), runner=dict(machine=platform.machine(),
                  cpus=os.cpu_count(), python=platform.python_version(), numpy=np.__version__),
                  source=os.environ.get('GITHUB_SHA'), data=data_receipt,
                  protocol='BEIR SciFact test (UltraBERT v2 data identities); document = title + " " + text; '
                           'raw query/document text through the public role API; frontend truncates at 512 tokens; '
                           'exact float32-dot/float64-sum MaxSim over selected vectors; ties by ascending document ID',
                  metrics=metrics, queries=len(per_query), documents=len(ids), timing=timing,
                  native_score_check=dict(pairs=check, max_abs_diff=max(c['abs_diff'] for c in check)),
                  original_comparison='none claimed: no frozen original-model SciFact ranking exists for this model')
    write(out / 'moedern-scifact.json', report)
    write(out / 'rankings.json', per_query)
    (out / 'summary.md').write_text(
        '### MoEdern1.5M SciFact (quality evidence, no gate)\n\n'
        f"NDCG@10 {metrics['ndcg_at_10']:.4f} | MRR@10 {metrics['mrr_at_10']:.4f} | "
        f"Recall@100 {metrics['recall_at_100']:.4f}\n\n"
        f"Queries {qtime['seconds']:.1f}s, documents {dtime['seconds']:.1f}s, MaxSim {timing['maxsim_seconds']:.1f}s, "
        f"total {timing['total_seconds']:.0f}s; native-vs-brute max |diff| {report['native_score_check']['max_abs_diff']:.2e}\n")
    print(json.dumps(dict(metrics=metrics, timing=timing), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
