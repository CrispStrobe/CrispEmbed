#!/usr/bin/env python3
"""Owner-approved EXPERIMENTAL publication of MoEdern 27M/115M Q4_K_MIX (GitHub Actions).

Driven by .github/workflows/moedern-q4k-publish.yml; protocol docs/moedern-q4k-publication-v1.json.

  plan VARIANT                 stdlib-only gates: owner approval, recipe, published F32 pin,
                               lane receipt arms (x86 + second ISA) above the Q4_K bar
  card VARIANT OUT             render the card section and quality table offline (no network)
  publish VARIANT OUT          (hosted only) fetch the SHA-pinned published F32, build the
                               portable tests/moedern-q4k quantizer, quantize with the recipe,
                               require the exact SHA/bytes recorded in the receipt, upload next
                               to the F32 (F32 card section kept, Q4 section appended, upstream/
                               files untouched), verify remote LFS identity and an empty-cache
                               readback of every file, write publication.json

The quality table only quotes committed receipts; anything not measured says so.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'docs/moedern-q4k-publication-v1.json'
ALLOWED = {'27M': 'cstr/SauerkrautLM-MoEdernColBERT-27M-CrispEmbed-GGUF',
           '115M': 'cstr/SauerkrautLM-MoEdernColBERT-115M-CrispEmbed-GGUF'}
MARK = '<!-- crispembed-q4k-mix-section -->'
BAR = 0.94


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(variant, root=ROOT):
    cfg = json.loads((root / PROTOCOL).read_text())
    require(cfg.get('approved_by_owner') is True, 'publication not approved by the owner; nothing accessed')
    require(variant in ALLOWED and variant in cfg['recipes'], 'unsupported variant: ' + variant)
    rec = dict(cfg['recipes'][variant]['Q4_K_MIX'])
    require(rec['repo'] == ALLOWED[variant], 'unreviewed destination')
    require(rec['precision_gate'] == 'cosine' and rec['precision_tier'] == 'q4_k', 'cosine Q4_K gate required')
    exp = rec['expected']
    require(re.fullmatch('[0-9a-f]{64}', exp['sha256']) and exp['bytes'] > 10**6, 'expected output pin malformed')
    pub = json.loads((root / rec['from']).read_text())
    require(pub['status'] == 'published and re-verified' and pub['repo'] == rec['repo']
            and re.fullmatch('[0-9a-f]{40}', pub['commit']) and pub['commit'] in pub['url'], 'F32 publication pin malformed')
    rec['f32'] = pub
    rec['gates'] = cfg['runtime_gates']
    return cfg, rec


def evidence(variant, rec, root=ROOT):
    """Receipt rows for the card; refuses unless the exact built arm passes the bar on both ISAs."""
    r = json.loads((root / rec['precision_receipt']).read_text())
    require(r['variant'] == variant and r['bars']['Q4_K'] == BAR, 'lane receipt variant/bar differs')
    runs = {x['run']: x for x in r['runs']}
    exp, second = rec['expected'], rec['second_isa']
    main = runs[exp['run']]['arms'][exp['arm']]
    require(main['conversion']['output_sha256'] == exp['sha256'] and main['conversion']['output_bytes'] == exp['bytes'],
            'expected SHA/bytes are not the receipt arm')
    arm2 = runs[second['run']]['arms'][second['arm']]
    for a in (main, arm2):
        require(a['ids_exact'] is True and len(a['cases']) == 10 and a['final_cos_min'] > BAR,
                'receipt arm does not clear the Q4_K bar with exact IDs')
    f32 = next(x['arms']['f32'] for x in r['runs'] if 'f32' in x['arms'])
    q8 = next(x['arms']['q8'] for x in r['runs'] if 'q8' in x['arms'])
    return dict(receipt=rec['precision_receipt'], receipt_sha256=digest(root / rec['precision_receipt']),
                main=main, main_cpu=runs[exp['run']]['build']['cpu'], main_run=exp['run'],
                second=arm2, second_cpu=runs[second['run']]['build']['cpu'], second_run=second['run'],
                f32=f32, q8=q8)


def section(variant, rec, ev, sha=None, nbytes=None):
    m, s, f32, q8 = ev['main'], ev['second'], ev['f32'], ev['q8']
    sha, nbytes = sha or rec['expected']['sha256'], nbytes or rec['expected']['bytes']
    scores = '; '.join(f"{x['id']} {x['native']:.3f} vs original {x['original']:.3f}" for x in m['scores'])
    gates = ' '.join(g + '=1' for g in rec['gates'])
    mib = lambda b: f'{b / 2**20:.2f} MiB'  # noqa: E731
    return f"""{MARK}
## EXPERIMENTAL: `{rec['filename']}` (Q4_K_MIX, policy `{rec['policy']}`)

Status: **experimental** (owner decision, 7 November 2026). It clears the owner's Q4_K token-vector bar
(> {BAR}) on the frozen ten-case panel, but retrieval quality has **not** been measured. Do not mix its vectors
into an index built with the F32 package; re-encode the whole collection with one file.

- File `{rec['filename']}`, {nbytes:,} bytes, SHA256 `{sha}`.
- Recipe: {rec['note']}{(' ' + rec['build_note']) if rec.get('build_note') else ''}
- Produced from the F32 file above (`{rec['f32']['sha256']}`) by CrispEmbed's
  `tools/moedern-colbert/quantize_q4k_mixed.py` with the portable `tests/moedern-q4k` C quantizer; the build
  must reproduce the exact bytes measured in run {ev['main_run']} or the workflow refuses to publish.
- Statement of changes (Apache-2.0 section 4(b)): the upstream weights were repacked and **quantized**
  (row-padded Q4_K / Q8_0, critical tensors F32); nothing was trained, merged or fine-tuned.
  `upstream/LICENSE` and `upstream/README.md` remain the unmodified upstream files.

### Quality vs the original model (receipt `{ev['receipt']}`)

Reference: frozen original-model token vectors, ten raw-text cases, hosted CPU portable builds.

| Package | Size | Status | Token IDs/masks | Min token-vector cosine x86 / ARM | Min per-case matrix cosine | MaxSim max abs error | SciFact nDCG@10 | Public clients |
|---|---:|---|---|---|---|---|---|---|
| F32 (published, above) | {mib(f32['bytes'])} | reference package | exact | {f32['final_cos_min']:.6f} | {f32['flattened_cos_min']:.6f} | {f32['maxsim_max_abs']:.2g} | not measured for this package | C/Python/CLI |
| Q8_0 (not published) | {mib(q8['bytes'])} | fails 0.97 Q8 bar | exact | {q8['final_cos_min']:.4f} | {q8['flattened_cos_min']:.4f} | {q8['maxsim_max_abs']:.3f} | not measured | not measured |
| **Q4_K_MIX (this file)** | **{mib(nbytes)}** | **experimental** | exact | **{m['final_cos_min']:.4f}** / {s['final_cos_min']:.4f} | {m['flattened_cos_min']:.4f} | {m['maxsim_max_abs']:.3f} | **not measured** | **not measured** |

- Worst token: `{m['worst_case']}` (x86 {ev['main_cpu']}, run {ev['main_run']}; ARM {ev['second_cpu']}, run {ev['second_run']}).
- MaxSim on the three frozen pairs: {scores}.
- Strict 0.002 absolute / 0.99999 cosine budgets: **not met** (expected for 4-bit; recorded in the receipt).
- Known limitation: MoEdern's hash-MoE router makes near-tie top-2 expert choices; a small perturbation can
  flip one expert for one token, which dominates the minimum (see `docs/moedern-q4k-mixed.md`, "routing flips").
- Not measured: SciFact/other retrieval (nDCG/MRR/Recall, top-1 agreement), public C#/Rust/Dart/HTTP clients,
  GPU (the MoEdern engine is CPU-only), long inputs beyond the panel.

### Usage

Load it through CrispEmbed's opt-in MoEdern raw-text API (C/Python/CLI, as for the F32 file) with every gate set:
`{gates}`.
All gates are default-off; with them unset nothing changes.
"""


def card(variant, rec, ev, base, sha, nbytes):
    base = base.split(MARK)[0].rstrip() + '\n'
    return base + '\n' + section(variant, rec, ev, sha, nbytes)


def publish(variant, out):
    require(os.environ.get('GITHUB_ACTIONS') == 'true', 'hosted execution required')
    cfg, rec = load(variant)
    ev = evidence(variant, rec)
    sys.path[:0] = [str(ROOT / 'tools/ci-heavy'), str(ROOT / 'tools/moedern-colbert')]
    import moedern_q4k_lanes as lanes
    import quantize_q4k_mixed as mixed
    scratch = Path(os.environ['HEAVY_SCRATCH'])
    report = dict(schema='crispembed-moedern-q4k-publication-report-v1', variant=variant, status='started',
                  protocol=PROTOCOL, protocol_sha256=digest(ROOT / PROTOCOL), source=os.environ.get('SOURCE_COMMIT'))
    save = lambda: (out / 'publication.json').write_text(json.dumps(report, indent=2) + '\n')  # noqa: E731
    try:
        require(re.fullmatch('[0-9a-f]{40}', report['source'] or ''), 'exact source commit required')
        report['build'] = lanes.build(out, scratch / 'build')
        f32 = scratch / 'published-f32.gguf'
        lanes.fetch(rec['f32']['url'], f32, rec['f32']['sha256'])
        model = scratch / rec['filename']
        conv = mixed.quantize(f32, model, variant, scratch / 'build/moedern-q4k-quantize', rec['build_recipe'])
        (out / 'quantize-report.json').write_text(json.dumps(conv, indent=2) + '\n')
        sha, nbytes = digest(model), model.stat().st_size
        report['built'] = dict(sha256=sha, bytes=nbytes, counts=conv['counts'], policy=conv['policy'])
        save()
        require(sha == rec['expected']['sha256'] and nbytes == rec['expected']['bytes'],
                f'built {sha}/{nbytes} differs from the measured receipt arm; refusing to publish')
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
        repo, base_rev = rec['repo'], rec['f32']['commit']
        api = HfApi(token=os.environ['HF_TOKEN'])
        head = api.model_info(repo, files_metadata=True)
        f32name = rec['f32']['url'].rsplit('/', 1)[1]
        allowed = {'.gitattributes', 'README.md', 'manifest.json', 'upstream/LICENSE', 'upstream/README.md',
                   f32name, rec['filename']}
        require({s.rfilename for s in head.siblings} <= allowed, 'destination holds unexpected files')
        base = Path(hf_hub_download(repo, 'README.md', revision=head.sha, cache_dir=scratch / 'cur')).read_text()
        card_bytes = card(variant, rec, ev, base, sha, nbytes).encode()
        upstream = {n: next(s for s in head.siblings if s.rfilename == n) for n in ('upstream/LICENSE', 'upstream/README.md')}
        up_sha = {n: digest(hf_hub_download(repo, n, revision=base_rev, cache_dir=scratch / 'base')) for n in upstream}
        files = [dict(filename=f32name, variant='F32', bytes=rec['f32']['bytes'], sha256=rec['f32']['sha256'],
                      status='reference package (unchanged)'),
                 dict(filename=rec['filename'], variant='Q4_K_MIX', bytes=nbytes, sha256=sha, status='experimental',
                      policy=rec['policy'], recipe=conv['recipe_spec'], counts=conv['counts'])]
        manifest = dict(schema='crispembed-moedern-q4k-manifest-v1', repo=repo, source_commit=report['source'],
                        protocol=PROTOCOL, protocol_sha256=report['protocol_sha256'], files=files,
                        upstream_files_sha256=up_sha, readme_sha256=hashlib.sha256(card_bytes).hexdigest(),
                        precision=dict(receipt=ev['receipt'], receipt_sha256=ev['receipt_sha256'], bar=BAR,
                                       final_cos_min_x86=ev['main']['final_cos_min'],
                                       final_cos_min_arm=ev['second']['final_cos_min'],
                                       maxsim_max_abs=ev['main']['maxsim_max_abs'], ids_exact=True),
                        not_measured=['SciFact retrieval', 'public clients', 'GPU'], status='experimental')
        man_bytes = (json.dumps(manifest, indent=2) + '\n').encode()
        (out / 'README.md').write_bytes(card_bytes)
        (out / 'manifest.json').write_bytes(man_bytes)
        commit = api.create_commit(repo_id=repo, commit_message=f'Add EXPERIMENTAL MoEdern {variant} Q4_K_MIX',
                                   operations=[CommitOperationAdd(rec['filename'], str(model)),
                                               CommitOperationAdd('README.md', card_bytes),
                                               CommitOperationAdd('manifest.json', man_bytes)])
        info = api.model_info(repo, revision=commit.oid, files_metadata=True)
        remote = {s.rfilename: s for s in info.siblings}
        for f in files:
            s = remote[f['filename']]
            require(s.size == f['bytes'] and s.lfs and s.lfs.sha256 == f['sha256'], 'remote LFS identity differs')
        expected = {f['filename']: f['sha256'] for f in files}
        expected.update(up_sha, **{'README.md': manifest['readme_sha256'],
                                   'manifest.json': hashlib.sha256(man_bytes).hexdigest()})
        fresh = scratch / 'fresh-hf-cache'
        fresh.mkdir(parents=True, exist_ok=False)
        for name, want in expected.items():
            p = Path(hf_hub_download(repo, name, revision=commit.oid, cache_dir=fresh, force_download=True, token=False))
            require(digest(p) == want, 'empty-cache readback differs: ' + name)
        report.update(status='published and re-verified', repo=repo, commit=commit.oid, files=files,
                      small_files_sha256={n: expected[n] for n in expected if not n.endswith('.gguf')},
                      url=f'https://huggingface.co/{repo}/resolve/{commit.oid}/{rec["filename"]}',
                      run=int(os.environ.get('GITHUB_RUN_ID', 0)))
        save()
    except BaseException as e:
        report.update(status='failed or rejected', error=f'{type(e).__name__}: {e}')
        save()
        raise


def main(argv):
    require(len(argv) >= 3, 'usage: plan VARIANT | card VARIANT OUT | publish VARIANT OUT')
    cmd, variant = argv[1], argv[2]
    cfg, rec = load(variant)
    ev = evidence(variant, rec)
    if cmd == 'plan':
        print(json.dumps(dict(variant=variant, repo=rec['repo'], recipe=rec['build_recipe'], expected=rec['expected'],
                              f32=rec['f32']['url'], x86=ev['main']['final_cos_min'],
                              arm=ev['second']['final_cos_min']), indent=1))
        return
    out = Path(argv[3])
    out.mkdir(parents=True, exist_ok=True)
    if cmd == 'card':
        (out / 'section.md').write_text(section(variant, rec, ev))
        return
    require(cmd == 'publish', 'unknown command')
    publish(variant, out)


if __name__ == '__main__':
    main(sys.argv)
