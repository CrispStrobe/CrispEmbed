#!/usr/bin/env python3
"""Publish the reviewed packed Josev F32 GGUF, then re-verify it from an empty cache.

Rejects before any credential, API or payload access unless the owner approval and
destination are committed in docs/josev2b-publication-v1.json. Uploads only the
packed GGUF, the retained upstream notices under upstream/, a card and a manifest.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/ci-heavy'))
PROTOCOL = 'docs/josev2b-publication-v1.json'


def require(ok, why):
    if not ok: raise ValueError(why)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(4*2**20): h.update(b)
    return h.hexdigest()


def preflight():
    cfg = json.loads((ROOT/PROTOCOL).read_text())
    require(cfg['approved_by_owner'] is True and cfg['destination'], 'publication not approved by the owner; nothing accessed')
    receipt = json.loads((ROOT/cfg['consumer_receipt']).read_text())
    require(receipt['independently_reviewed'] and receipt['run'] == cfg['consumer_run'] and all(receipt['consumers'].values())
            and receipt['root_profile']['compared_stages'] == 227 and not receipt['root_profile']['failures'], 'consumer receipt not a full pass')
    return cfg, receipt


def card(cfg, receipt, model_sha, model_bytes, source):
    up = cfg['upstream']
    return f"""---
license: apache-2.0
base_model: {up['repo']}
tags: [gguf, crispembed]
---
# Josev-2B SystemOne packed F32 GGUF (CrispEmbed)

Byte-preserving F32 packing of [{up['repo']}](https://huggingface.co/{up['repo']}/tree/{up['revision']})
(backbone, joint head and tokenizer in one file) for CrispEmbed's opt-in SystemOne API. No tensor
arithmetic or quantization was performed. Upstream license and notices are kept unchanged under `upstream/`.

- File: `{cfg['filename']}`, {model_bytes} bytes, SHA256 `{model_sha}`
- Evidence source commit: `{source}`; packing run {receipt['packed_producer']['run']}; consumer run {receipt['run']}
- Verified: 227 original stages and four complete typed responses in C/Python/CLI/HTTP/Rust/Dart on Linux x64 CPU.
- Not verified: task quality, other platforms, GPU, quantization. All seven model gates stay default-off.

Modified from the upstream checkpoint by repackaging only (Apache-2.0 section 4).
"""


def main():
    out = Path(os.environ['HEAVY_OUT']); report = dict(schema='crispembed-josev-publication-report-v1', status='started')
    save = lambda: (out/'publication.json').write_text(json.dumps(report, indent=2)+'\n')
    try:
        cfg, receipt = preflight(); report['phase'] = 'preflight-passed'; save()
        require(os.environ.get('HF_TOKEN'), 'dedicated publication credential unavailable')
        import systemone_consumers as w
        scfg = w.load(ROOT/w.PROTOCOL); pin = scfg['packed_producer']; scratch = Path(os.environ['HEAVY_SCRATCH'])/'josev-publish'
        scratch.mkdir(parents=True)
        notices = ['raw/notices/'+n for n in cfg['notices_required']]
        asset, service = w.fetch(dict(pin['asset_artifact'], run=pin['run'], source=pin['source']), scratch,
                                 include={pin['model_member'], *notices})
        model = asset/pin['model_member']
        require(model.stat().st_size == pin['model_bytes'] and digest(model) == pin['model_sha256'], 'packed GGUF bytes differ')
        report.update(phase='exact-artifact-verified', artifact=service); save()
        source = os.environ['GITHUB_SHA']
        card_bytes = card(cfg, receipt, pin['model_sha256'], pin['model_bytes'], source).encode()
        manifest = dict(filename=cfg['filename'], sha256=pin['model_sha256'], bytes=pin['model_bytes'], source=source,
                        upstream=cfg['upstream'], notices={n: digest(asset/'raw/notices'/n) for n in cfg['notices_required']},
                        consumer_receipt_sha256=digest(ROOT/cfg['consumer_receipt']))
        manifest_bytes = (json.dumps(manifest, indent=2)+'\n').encode()
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
        api = HfApi(token=os.environ['HF_TOKEN'])
        allowed = {cfg['filename'], 'README.md', 'manifest.json', '.gitattributes', *('upstream/'+n for n in cfg['notices_required'])}
        if api.repo_exists(cfg['destination']):
            current = {s.rfilename for s in api.model_info(cfg['destination']).siblings}
            require(current <= allowed, 'destination holds files outside the allowlist')
        else:
            api.create_repo(cfg['destination'], private=False, exist_ok=False)
        ops = [CommitOperationAdd(path_in_repo=cfg['filename'], path_or_fileobj=str(model)),
               CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=card_bytes),
               CommitOperationAdd(path_in_repo='manifest.json', path_or_fileobj=manifest_bytes)]
        ops += [CommitOperationAdd(path_in_repo='upstream/'+n, path_or_fileobj=str(asset/'raw/notices'/n)) for n in cfg['notices_required']]
        commit = api.create_commit(repo_id=cfg['destination'], operations=ops, commit_message='Publish reviewed Josev-2B SystemOne packed F32 GGUF')
        report.update(phase='immutable-remote-verification', published_commit=commit.oid); save()
        info = api.model_info(cfg['destination'], revision=commit.oid, files_metadata=True)
        sib = next(s for s in info.siblings if s.rfilename == cfg['filename'])
        require(sib.size == pin['model_bytes'] and sib.lfs and sib.lfs.sha256 == pin['model_sha256'], 'remote LFS identity differs')
        cache = scratch/'fresh-cache'; cache.mkdir()
        fetched = Path(hf_hub_download(cfg['destination'], cfg['filename'], revision=commit.oid, cache_dir=cache, force_download=True))
        require(digest(fetched) == pin['model_sha256'], 'empty-cache download differs')
        report.update(status='published and re-verified', remote=dict(repo=cfg['destination'], commit=commit.oid,
                      url=f"https://huggingface.co/{cfg['destination']}/resolve/{commit.oid}/{cfg['filename']}",
                      sha256=pin['model_sha256'], bytes=pin['model_bytes'])); save()
    except BaseException as e:
        report.update(status='failed or rejected', error=f'{type(e).__name__}: {e}'); save(); raise


if __name__ == '__main__': main()
