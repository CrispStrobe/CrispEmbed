#!/usr/bin/env python3
"""Publish EXPERIMENTAL quantized Josev-2B SystemOne GGUFs next to the reviewed F32 file.

Driven by .github/workflows/josev-publish.yml (workflow_dispatch, `variants` input).
Protocol: docs/josev2b-q4k-publication-v1.json. Every gate before the Hub import is
stdlib-only and runs before any credential or weight access:

  plan VARIANTS        approval, recipes, cosine gate against the committed sweep receipt
                       (tests/results/josev2b/q4k-sweep-41615751516.json); prints the plan
  publish VARIANTS F32 QUANTIZER OUT
                       verify the SHA-pinned published F32, re-quantize each variant with the
                       exact measured sweep recipe (tools/ci-heavy/josev2b_q4k_sweep.py
                       quantize(): per-tensor plan + C quantizer + conversion verifier), require
                       the receipt's output SHA256, upload into the same repo keeping every
                       existing file, verify remote LFS identity and an empty-cache readback.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'docs/josev2b-q4k-publication-v1.json'
F32_PROTOCOL = 'docs/josev2b-publication-v1.json'
SCHEMA = 'crispembed-josev-q4k-publication-recipe-v1'
DESTINATION = 'cstr/Josev-2B-SystemOne-GGUF'
BARS = {'q8_0': 0.97, 'q4_k': 0.94}
EXISTING = {'.gitattributes', 'README.md', 'manifest.json', 'josev2b-systemone-f32.gguf'}
SECTION = '<!-- crispembed-experimental-quantized -->'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8 * 2**20):
            h.update(b)
    return h.hexdigest()


def load(root=ROOT):
    cfg = json.loads((root / PROTOCOL).read_text())
    require(cfg.get('schema') == SCHEMA, 'unexpected protocol schema')
    require(cfg.get('approved_by_owner') is True and cfg.get('dispatch_ready') is True,
            'quantized publication not approved by the owner; nothing accessed')
    require(cfg.get('destination') == DESTINATION, 'unreviewed destination')
    f32 = json.loads((root / F32_PROTOCOL).read_text())
    require(f32['destination'] == DESTINATION and f32['filename'] == cfg['source']['filename'], 'F32 protocol differs')
    require(re.fullmatch('[0-9a-f]{40}', cfg['source']['revision']) and re.fullmatch('[0-9a-f]{64}', cfg['source']['sha256']),
            'F32 source pin malformed')
    return cfg


def parse_variants(text, cfg):
    names = [v.strip().upper() for v in re.split(r'[,\s]+', text or '') if v.strip()]
    require(names and len(set(names)) == len(names), 'variants required, no duplicates')
    bad = [n for n in names if n not in cfg['recipes']]
    require(not bad, 'unknown variant(s): ' + ','.join(bad))
    return names


def gate(cfg, variant, root=ROOT):
    """Cosine gate from the committed sweep receipt (owner bars, docs/precision-acceptance.md)."""
    rec = cfg['recipes'][variant]
    require(rec.get('precision_gate') == 'cosine', f'{variant}: only the cosine gate is defined')
    path = root / rec['precision_receipt']
    r = json.loads(path.read_text())
    require(r.get('schema') == 'crispembed-josev-q4k-sweep-v1' and r.get('completed') is True, 'sweep receipt incomplete')
    require(r['build']['ggml_native'] == 'OFF', 'receipt not from a portable build')
    require(r['model']['sha256'] == cfg['source']['sha256'] and r['model']['revision'] == cfg['source']['revision'],
            'receipt measured a different F32 source')
    e = r['recipes'][rec['precision_dtype']]
    require(e.get('phase') == 'complete', f'{variant}: receipt recipe incomplete')
    bar = BARS[rec['precision_tier']]
    require(e['bar'] == bar and len(e['cases']) == 4, f'{variant}: bar/case coverage differs')
    worst = min(c['logits_cos'] for c in e['cases'])
    require(worst > bar and e['passes_logits_bar'] is True, f'{variant}: head-logit cosine {worst} not above {bar}')
    c = e['conversion']
    require(re.fullmatch('[0-9a-f]{64}', c['output_sha256']) and c['critical292_f32_byte_exact'], 'conversion record malformed')
    return {'variant': variant, 'recipe': rec['precision_dtype'], 'receipt': rec['precision_receipt'],
            'receipt_sha256': digest(path), 'receipt_commit': r['source'], 'bar': bar,
            'head_logit_cos_min': worst, 'final_norm_cos_min': min(x['final_norm_cos'] for x in e['cases']),
            'final_norm_bar_passed': e['passes_final_norm_bar'], 'typed_json_exact': e['summary']['typed_json_exact'],
            'argmax_exact': e['summary']['argmax_exact'], 'cases': len(e['cases']),
            'probabilities_max_abs': e['summary']['probabilities_max_abs'],
            'strict_stage_failures': e['strict_stage_failures'], 'counts': c['counts'],
            'expected_sha256': c['output_sha256'], 'expected_bytes': c['output_bytes']}


def section(cfg, files, gates, source_commit):
    rows = []
    for f in files:
        g = gates[f['variant']]
        rows.append(f"| `{f['filename']}` | {f['variant']} (`{g['recipe']}`; Q4_K/Q8_0/F32 tensors "
                    f"{g['counts']['Q4_K']}/{g['counts']['Q8_0']}/{g['counts']['F32']}) | {f['bytes']:,} | "
                    f"{g['head_logit_cos_min']:.5f} (bar > {g['bar']}) | {g['final_norm_cos_min']:.5f} | "
                    f"{g['argmax_exact']}/{g['cases']} | {g['typed_json_exact']}/{g['cases']} | "
                    f"{g['probabilities_max_abs']:.4f} | failed ({g['strict_stage_failures']} stages) | experimental |")
    sums = '\n'.join(f"- `{f['filename']}`: SHA256 `{f['sha256']}`" for f in files)
    notes = '\n'.join(f"- **{f['variant']}**: {cfg['recipes'][f['variant']]['note']}" for f in files)
    return f"""{SECTION}
## Experimental quantized variants (owner decision 7 November 2026)

**Experimental: not quality-qualified.** Use the F32 file above for exact typed responses. The
quantized files were produced from the F32 file of revision `{cfg['source']['revision']}` with
CrispEmbed's C quantizer (tensor types changed; no weights trained or merged; Apache-2.0 section 4(b)).

Quality vs the frozen original reference (four frozen typed requests, portable hosted CPU, receipt
`{gates[files[0]['variant']]['receipt']}`):

| File | Precision | Bytes | Head-logit cosine (min) | Final-norm cosine (min) | Option argmax exact | Complete typed JSON exact | Max probability error | Strict 0.002/0.99999 stage budgets | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(rows)}

- Option choices match the original on 4/4 cases, but the rounded four-decimal probabilities and
  scores move, so the complete typed JSON is not exact (Q8_0 1/4, Q4_K_MIX 0/4).
- Not measured: independent task-quality panel, public-client (C/Python/CLI/HTTP/Rust/Dart) runs of
  the quantized files, other platforms, GPU (the Josev runtime is CPU-only), latency/RSS.
- Runtime: the same seven default-off Josev gates as F32 plus `CRISPEMBED_JOSEV_QUANT=1` and
  `CRISPEMBED_QWEN35_QUANT=1`.

{notes}

{sums}

Built by CrispEmbed commit `{source_commit}` (workflow `josev-publish.yml`, protocol `{PROTOCOL}`).
"""


def readme(old, new_section):
    text = old.split(SECTION)[0].rstrip() + '\n\n'
    return text + new_section


def publish(cfg, variants, f32, quantizer, out, source_commit):
    require(re.fullmatch('[0-9a-f]{40}', source_commit or ''), 'exact source commit required')
    gates = {v: gate(cfg, v) for v in variants}
    f32 = Path(f32)
    require(f32.stat().st_size == 7659180448 and digest(f32) == cfg['source']['sha256'], 'F32 source bytes differ')
    sys.path[:0] = [str(ROOT / 'tools/ci-heavy'), str(ROOT / 'tools')]
    import josev2b_q4k_sweep as sweep
    import quantize_josev2b_protected as prot
    rows = prot.schema()
    files, conversions = [], {}
    work = Path(os.environ.get('HEAVY_SCRATCH', out)); work.mkdir(parents=True, exist_ok=True)
    for v in variants:
        rec, g = cfg['recipes'][v], gates[v]
        path = work / rec['filename']
        conversions[v] = sweep.quantize(f32, path, Path(quantizer), g['recipe'], rows)
        sha = digest(path)
        require(sha == g['expected_sha256'] and path.stat().st_size == g['expected_bytes'],
                f'{v}: re-quantized bytes differ from the measured receipt ({sha})')
        files.append({'variant': v, 'filename': rec['filename'], 'bytes': path.stat().st_size, 'sha256': sha})

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    require(os.environ.get('HF_TOKEN'), 'publication credential unavailable')
    api = HfApi(token=os.environ['HF_TOKEN'])
    f32cfg = json.loads((ROOT / F32_PROTOCOL).read_text())
    allowed = EXISTING | {'upstream/' + n for n in f32cfg['notices_required']} | \
        {r['filename'] for r in cfg['recipes'].values()}
    info = api.model_info(DESTINATION, files_metadata=True)
    current = {s.rfilename: s for s in info.siblings}
    require(set(current) <= allowed, 'destination holds files outside the allowlist')
    s = current[cfg['source']['filename']]
    require(s.lfs and s.lfs.sha256 == cfg['source']['sha256'], 'published F32 identity changed')
    cache0 = work / 'prev'
    prev_readme = Path(hf_hub_download(DESTINATION, 'README.md', revision=info.sha, cache_dir=cache0)).read_text()
    prev_manifest = json.loads(Path(hf_hub_download(DESTINATION, 'manifest.json', revision=info.sha, cache_dir=cache0)).read_text())
    card = readme(prev_readme, section(cfg, files, gates, source_commit)).encode()
    upstream = {('upstream/' + n): current['upstream/' + n].blob_id for n in f32cfg['notices_required']}
    manifest = dict(prev_manifest)
    manifest['experimental_quantized'] = {
        'protocol': PROTOCOL, 'protocol_sha256': digest(ROOT / PROTOCOL), 'source_commit': source_commit,
        'from': cfg['source'], 'files': files, 'precision': gates,
        'recipes': {v: cfg['recipes'][v] for v in variants}, 'previous_revision': info.sha}
    man = (json.dumps(manifest, indent=2) + '\n').encode()
    (out / 'README.md').write_bytes(card)
    (out / 'manifest.json').write_bytes(man)
    ops = [CommitOperationAdd(path_in_repo=f['filename'], path_or_fileobj=str(work / f['filename'])) for f in files]
    ops += [CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=card),
            CommitOperationAdd(path_in_repo='manifest.json', path_or_fileobj=man)]
    commit = api.create_commit(repo_id=DESTINATION, operations=ops,
                               commit_message='Add experimental quantized Josev-2B SystemOne GGUFs: ' + ', '.join(variants))
    after = {s.rfilename: s for s in api.model_info(DESTINATION, revision=commit.oid, files_metadata=True).siblings}
    require(set(after) <= allowed and set(current) <= set(after), 'remote file set changed unexpectedly')
    for f in files:
        s = after[f['filename']]
        require(s.size == f['bytes'] and s.lfs and s.lfs.sha256 == f['sha256'], 'remote LFS identity differs')
    s = after[cfg['source']['filename']]
    require(s.lfs.sha256 == cfg['source']['sha256'], 'F32 identity changed by the commit')
    require(all(after[n].blob_id == b for n, b in upstream.items()), 'upstream notices changed')
    expected = {f['filename']: f['sha256'] for f in files}
    expected.update({'README.md': hashlib.sha256(card).hexdigest(), 'manifest.json': hashlib.sha256(man).hexdigest()})
    cache = work / 'fresh-hf-cache'
    cache.mkdir(parents=True, exist_ok=False)
    small = {}
    for name in sorted(after):
        p = Path(hf_hub_download(DESTINATION, name, revision=commit.oid, cache_dir=cache, force_download=True, token=False))
        sha = digest(p)
        if name in expected:
            require(sha == expected[name], 'empty-cache readback differs: ' + name)
        elif name == cfg['source']['filename']:
            require(sha == cfg['source']['sha256'], 'F32 readback differs')
        small[name] = sha
        if name.endswith('.gguf'):
            p.unlink()
    return {'repo': DESTINATION, 'commit': commit.oid, 'previous_revision': info.sha, 'files': files,
            'all_files_sha256': small, 'precision': gates,
            'conversion_checks': {v: {k: c[k] for k in ('counts', 'critical292_f32_byte_exact', 'general_file_type',
                                                        'output_sha256') if k in c} for v, c in conversions.items()},
            'urls': {f['filename']: f'https://huggingface.co/{DESTINATION}/resolve/{commit.oid}/{f["filename"]}' for f in files}}


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ''
    cfg = load()
    if cmd == 'plan':
        for v in parse_variants(argv[2], cfg):
            g = gate(cfg, v)
            print(v, cfg['recipes'][v]['filename'], g['recipe'], g['expected_sha256'], g['head_logit_cos_min'])
        return
    require(cmd == 'publish' and len(argv) == 6, 'usage: plan VARIANTS | publish VARIANTS F32 QUANTIZER OUT')
    out = Path(argv[5]); out.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'crispembed-josev-quant-publication-report-v1', 'protocol': PROTOCOL, 'status': 'started'}
    save = lambda: (out / 'publication.json').write_text(json.dumps(report, indent=2) + '\n')
    try:
        variants = parse_variants(argv[2], cfg)
        report['variants'] = variants; save()
        report['remote'] = publish(cfg, variants, argv[3], argv[4], out, os.environ.get('SOURCE_COMMIT'))
        report['status'] = 'published and re-verified'; save()
    except BaseException as e:
        report.update(status='failed or rejected', error=f'{type(e).__name__}: {e}'); save()
        raise


if __name__ == '__main__':
    main(sys.argv)
