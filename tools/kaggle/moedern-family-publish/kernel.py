"""Kaggle kernel: owner-approved publication of a proven MoEdern 27M/115M packed F32 GGUF.

Re-packs from the mounted frozen original + pinned tokenizer component exactly as the
public-consumer kernel did, requires the proven packed SHA256 from the committed receipt,
uploads with the upstream Apache-2.0 LICENSE/README (sha-pinned), re-verifies an
empty-cache download. The write credential is read from the attached private dataset.
"""
import glob, hashlib, json, os, subprocess, sys
from pathlib import Path
COMMIT, VARIANT, SLUG = '@COMMIT@', '@VARIANT@', '@VARIANT_SLUG@'
def sh(*a): print('+', *a[:3], flush=True); subprocess.run(list(a), check=True)
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT)
sh(sys.executable, '-m', 'pip', 'install', '-q', 'huggingface-hub==1.33.0')
sys.path[:0] = ['tools', 'tools/kaggle/moedern-family-public']
import pack_moedern_text_f32 as packer, worker
receipt = json.load(open(f'tests/results/moedern-colbert/public-consumers-{SLUG}.json'))
assert receipt['variant'] == VARIANT and all(receipt['consumers'].values()) and receipt['typed_exact'], 'proof receipt not a full pass'
root, tokenizer = worker.locate(Path('/kaggle/input'), VARIANT)
packed = Path(f'/kaggle/temp/moedern-{SLUG}-text-f32.gguf')
packer.pack_family(VARIANT, root/'model-F32.gguf', tokenizer, root/'review/original.json', packed)
assert digest(packed) == receipt['pack']['sha256'] and packed.stat().st_size == receipt['pack']['bytes'], 'repack differs from proven bytes'
contract = json.load(open('docs/moedern-colbert-source-contract.json'))['variants'][VARIANT]['original']
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
notices = {}
for name in ('LICENSE', 'README.md'):
    p = hf_hub_download(contract['repository'], name, revision=contract['revision'], cache_dir='/kaggle/temp/up')
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
api = HfApi(token=open(glob.glob('/kaggle/input/**/hf_token.txt', recursive=True)[0]).read().strip())
if not api.repo_exists(REPO): api.create_repo(REPO, private=False)
ops = [CommitOperationAdd(fname, str(packed)), CommitOperationAdd('README.md', card)]
ops += [CommitOperationAdd('upstream/' + n, p) for n, p in notices.items()]
commit = api.create_commit(repo_id=REPO, operations=ops, commit_message=f'Publish proven MoEdern {VARIANT} packed F32 GGUF')
fetched = hf_hub_download(REPO, fname, revision=commit.oid, cache_dir='/kaggle/temp/fresh', force_download=True)
assert digest(fetched) == receipt['pack']['sha256'], 'empty-cache download differs'
report = dict(status='published and re-verified', variant=VARIANT, repo=REPO, commit=commit.oid, sha256=receipt['pack']['sha256'],
              bytes=receipt['pack']['bytes'], url=f'https://huggingface.co/{REPO}/resolve/{commit.oid}/{fname}', source=COMMIT)
os.makedirs('/kaggle/working', exist_ok=True); json.dump(report, open('/kaggle/working/publication.json', 'w'), indent=2)
print(json.dumps(report, indent=1))
