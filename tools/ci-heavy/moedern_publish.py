#!/usr/bin/env python3
"""Owner-approved publication of the proven MoEdern1.5M packed F32 GGUF (GitHub Actions).

Re-packs from the same immutable reviewed inputs as public-consumer run 40470702013,
requires the exact proven packed SHA256, keeps upstream Apache-2.0 LICENSE/README,
and re-verifies an empty-cache download. HF_TOKEN must be a write-scoped secret.
"""
import hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
REPO = 'cstr/SauerkrautLM-MoEdernColBERT-1.5M-CrispEmbed-GGUF'
UPSTREAM = ('VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M', '9d360b8b88ae11bea5784bf9012ff509fdd5c2da')
NOTICES = {'LICENSE': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
           'README.md': '5b7d2ba6ee057c9d0c8b81c0c76a5fe9ec6c2d09820119b7c82498406b522c2e'}
COMMIT = os.environ['GITHUB_SHA']
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def token(_): return os.environ['HF_TOKEN']
receipt = json.load(open('tests/results/moedern-colbert/public-consumers-1p5m.json'))
assert all(receipt['consumers'].values()) and receipt['typed_exact'] and receipt['scores_passed'], 'proof receipt not a full pass'
sys.path[:0] = ['tools', 'tools/ci-heavy']
import moedern_public_consumers as w, pack_moedern_text_f32 as packer
scratch = Path(os.environ['RUNNER_TEMP'])/'pub'; inputs = scratch/'inputs'; inputs.mkdir(parents=True)
w.prepare(scratch, inputs)
packed = scratch/'moedern-1.5m-text-f32.gguf'
packer.pack(inputs/'model.gguf', inputs/'tokenizer.gguf', inputs/'composed.json', packed)
assert digest(packed) == receipt['pack']['sha256'] and packed.stat().st_size == receipt['pack']['bytes'], 'repack differs from proven bytes'
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
notices = {}
for name, want in NOTICES.items():
    p = hf_hub_download(UPSTREAM[0], name, revision=UPSTREAM[1], cache_dir=str(scratch/'up'))
    assert digest(p) == want, 'upstream notice differs: ' + name
    notices[name] = p
card = f"""---
license: apache-2.0
base_model: {UPSTREAM[0]}
tags: [gguf, crispembed, colbert]
---
# SauerkrautLM-MoEdernColBERT-1.5M packed F32 GGUF (CrispEmbed)

Byte-preserving F32 packing (model + tokenizer metadata, no tensor arithmetic) of
[{UPSTREAM[0]}](https://huggingface.co/{UPSTREAM[0]}/tree/{UPSTREAM[1]}) for CrispEmbed's opt-in MoEdern
raw-text API. Upstream LICENSE and README are kept unchanged under `upstream/`.

- File `moedern-1.5m-text-f32.gguf`, {receipt['pack']['bytes']} bytes, SHA256 `{receipt['pack']['sha256']}`
- Verified (run {receipt['run']}): exact token IDs/masks/query expansion, all 20 frozen vectors within
  {receipt['worst_max_absolute']:.2e}, 3 MaxSim scores, in C/Python/Rust/Dart/CLI/HTTP on Linux x64 CPU.
- Not verified: retrieval quality, other platforms, GPU, quantization. Gates stay default-off.

Modified from the upstream checkpoint by repackaging only (Apache-2.0 section 4).
""".encode()
api = HfApi(token=token('HF_TOKEN'))
if not api.repo_exists(REPO): api.create_repo(REPO, private=False)
ops = [CommitOperationAdd('moedern-1.5m-text-f32.gguf', str(packed)), CommitOperationAdd('README.md', card)]
ops += [CommitOperationAdd('upstream/' + n, p) for n, p in notices.items()]
commit = api.create_commit(repo_id=REPO, operations=ops, commit_message='Publish proven MoEdern1.5M packed F32 GGUF')
fetched = hf_hub_download(REPO, 'moedern-1.5m-text-f32.gguf', revision=commit.oid, cache_dir=str(scratch/'fresh'), force_download=True)
assert digest(fetched) == receipt['pack']['sha256'], 'empty-cache download differs'
report = dict(status='published and re-verified', repo=REPO, commit=commit.oid, sha256=receipt['pack']['sha256'],
              bytes=receipt['pack']['bytes'], url=f'https://huggingface.co/{REPO}/resolve/{commit.oid}/moedern-1.5m-text-f32.gguf', source=COMMIT)
out = Path(os.environ['RUNNER_TEMP'])/'out'; out.mkdir(exist_ok=True); json.dump(report, open(out/'publication.json', 'w'), indent=2); print(json.dumps(report, indent=1))
