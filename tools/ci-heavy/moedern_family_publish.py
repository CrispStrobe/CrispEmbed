#!/usr/bin/env python3
"""Owner-approved publication of a proven MoEdern 27M/115M packed F32 GGUF (GitHub Actions).

Args: VARIANT MOUNT OUT. MOUNT holds the frozen original capture (review/original.json +
model-F32.gguf) and the pinned tokenizer component downloaded from the Kaggle outputs.
Re-packs, requires the proven packed SHA256 from the committed receipt, uploads with the
upstream Apache-2.0 LICENSE/README (sha-pinned), re-verifies an empty-cache download.
HF_TOKEN must be a write-scoped secret.
"""
import hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'tools/kaggle/moedern-family-public')]
import pack_moedern_text_f32 as packer, worker
VARIANT, MOUNT, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
SLUG = VARIANT.lower()
COMMIT = os.environ['GITHUB_SHA']
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt = json.load(open(f'tests/results/moedern-colbert/public-consumers-{SLUG}.json'))
assert receipt['variant'] == VARIANT and all(receipt['consumers'].values()) and receipt['typed_exact'], 'proof receipt not a full pass'
root, tokenizer = worker.locate(MOUNT, VARIANT)
OUT.mkdir(parents=True, exist_ok=True)
packed = Path(os.environ['RUNNER_TEMP'])/f'moedern-{SLUG}-text-f32.gguf'
packer.pack_family(VARIANT, root/'model-F32.gguf', tokenizer, root/'review/original.json', packed)
assert digest(packed) == receipt['pack']['sha256'] and packed.stat().st_size == receipt['pack']['bytes'], 'repack differs from proven bytes'
contract = json.load(open('docs/moedern-colbert-source-contract.json'))['variants'][VARIANT]['original']
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
notices = {}
for name in ('LICENSE', 'README.md'):
    p = hf_hub_download(contract['repository'], name, revision=contract['revision'], cache_dir=os.environ['RUNNER_TEMP']+'/up')
    assert digest(p) == contract['files'][name]['sha256'], 'upstream notice differs: ' + name
    notices[name] = p
REPO = f'cstr/SauerkrautLM-MoEdernColBERT-{VARIANT}-CrispEmbed-GGUF'
fname = packed.name
card = f"""---
license: apache-2.0
base_model: {contract['repository']}
tags: [gguf, crispembed, colbert]
---
# SauerkrautLM-MoEdernColBERT-{VARIANT} packed F32 GGUF (CrispEmbed)

Byte-preserving F32 packing (model + tokenizer metadata, no tensor arithmetic) of
[{contract['repository']}](https://huggingface.co/{contract['repository']}/tree/{contract['revision']}) for CrispEmbed's
opt-in MoEdern raw-text API (gates CRISPEMBED_MOEDERN_* including CRISPEMBED_MOEDERN_FAMILY).
Upstream LICENSE and README are kept unchanged under `upstream/`.

- File `{fname}`, {receipt['pack']['bytes']} bytes, SHA256 `{receipt['pack']['sha256']}`
- Verified on Linux x64 CPU (C/Python/CLI): exact token IDs/masks/query expansion, ten frozen cases' vectors within
  {receipt['worst_max_absolute']:.2e} of the frozen original, three MaxSim scores within 0.002.
- Not verified: retrieval quality, other platforms, GPU, quantization. Gates stay default-off.

Modified from the upstream checkpoint by repackaging only (Apache-2.0 section 4).
""".encode()
api = HfApi(token=os.environ['HF_TOKEN'])
if not api.repo_exists(REPO): api.create_repo(REPO, private=False)
ops = [CommitOperationAdd(fname, str(packed)), CommitOperationAdd('README.md', card)]
ops += [CommitOperationAdd('upstream/' + n, p) for n, p in notices.items()]
commit = api.create_commit(repo_id=REPO, operations=ops, commit_message=f'Publish proven MoEdern {VARIANT} packed F32 GGUF')
fetched = hf_hub_download(REPO, fname, revision=commit.oid, cache_dir=os.environ['RUNNER_TEMP']+'/fresh', force_download=True)
assert digest(fetched) == receipt['pack']['sha256'], 'empty-cache download differs'
report = dict(status='published and re-verified', variant=VARIANT, repo=REPO, commit=commit.oid, sha256=receipt['pack']['sha256'],
              bytes=receipt['pack']['bytes'], url=f'https://huggingface.co/{REPO}/resolve/{commit.oid}/{fname}', source=COMMIT)
json.dump(report, open(OUT/'publication.json', 'w'), indent=2)
print(json.dumps(report, indent=1))
