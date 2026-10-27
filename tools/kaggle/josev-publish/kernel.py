"""Kaggle kernel: owner-approved publication of the reviewed packed Josev F32 GGUF.

Same gates as tools/ci-heavy/josev_publish.py (committed approval, reviewed consumer
receipt, exact pinned artifact digest and model hash, upstream notices kept), with
credentials read from the attached private token dataset (never printed). After the
upload it re-verifies LFS identity and an empty-cache download.
"""
import glob, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path
COMMIT = '@COMMIT@'
def sh(*a): print('+', *a[:3], flush=True); subprocess.run(list(a), check=True)
def token(name):
    for p in glob.glob(f'/kaggle/input/**/{name}', recursive=True): return open(p).read().strip()
    raise SystemExit('missing credential file: ' + name)
def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(8 << 20): h.update(b)
    return h.hexdigest()
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT)
sh(sys.executable, '-m', 'pip', 'install', '-q', 'huggingface-hub==1.33.0', 'requests')
sys.path.insert(0, 'tools/ci-heavy')
import josev_publish as jp, requests
cfg, receipt = jp.preflight()
pin = json.load(open('docs/systemone-public-consumers-v1.json'))['packed_producer']
art = pin['asset_artifact']
gh = {'Authorization': 'Bearer ' + token('gh_token.txt'), 'Accept': 'application/vnd.github+json'}
meta = requests.get(f"https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/artifacts/{art['id']}", headers=gh, timeout=60).json()
assert meta['name'] == art['name'] and meta['size_in_bytes'] == art['bytes'] and meta['digest'] == 'sha256:' + art['sha256'] and not meta['expired'], 'artifact identity differs'
zpath = Path('/kaggle/temp/asset.zip')
with requests.get(meta['archive_download_url'], headers=gh, stream=True, timeout=600) as r, open(zpath, 'wb') as f:
    r.raise_for_status()
    for chunk in r.iter_content(8 << 20): f.write(chunk)
assert zpath.stat().st_size == art['bytes'] and digest(zpath) == art['sha256'], 'downloaded ZIP differs'
out = Path('/kaggle/temp/asset'); members = [pin['model_member']] + ['raw/notices/' + n for n in cfg['notices_required']]
with zipfile.ZipFile(zpath) as z:
    for m in members: z.extract(m, out)
zpath.unlink()
model = out / pin['model_member']
assert model.stat().st_size == pin['model_bytes'] and digest(model) == pin['model_sha256'], 'packed GGUF differs'
card = jp.card(cfg, receipt, pin['model_sha256'], pin['model_bytes'], COMMIT).encode()
manifest = (json.dumps(dict(filename=cfg['filename'], sha256=pin['model_sha256'], bytes=pin['model_bytes'], source=COMMIT,
            upstream=cfg['upstream'], notices={n: digest(out / 'raw/notices' / n) for n in cfg['notices_required']},
            consumer_receipt_sha256=digest(cfg['consumer_receipt'])), indent=2) + '\n').encode()
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
api = HfApi(token=token('hf_token.txt'))
allowed = {cfg['filename'], 'README.md', 'manifest.json', '.gitattributes', *('upstream/' + n for n in cfg['notices_required'])}
if api.repo_exists(cfg['destination']):
    assert {s.rfilename for s in api.model_info(cfg['destination']).siblings} <= allowed, 'destination holds other files'
else:
    api.create_repo(cfg['destination'], private=False, exist_ok=False)
ops = [CommitOperationAdd(cfg['filename'], str(model)), CommitOperationAdd('README.md', card), CommitOperationAdd('manifest.json', manifest)]
ops += [CommitOperationAdd('upstream/' + n, str(out / 'raw/notices' / n)) for n in cfg['notices_required']]
commit = api.create_commit(repo_id=cfg['destination'], operations=ops, commit_message='Publish reviewed Josev-2B SystemOne packed F32 GGUF')
info = api.model_info(cfg['destination'], revision=commit.oid, files_metadata=True)
sib = next(s for s in info.siblings if s.rfilename == cfg['filename'])
assert sib.size == pin['model_bytes'] and sib.lfs and sib.lfs.sha256 == pin['model_sha256'], 'remote LFS identity differs'
model.unlink()
fetched = hf_hub_download(cfg['destination'], cfg['filename'], revision=commit.oid, cache_dir='/kaggle/temp/fresh-cache', force_download=True)
assert digest(fetched) == pin['model_sha256'], 'empty-cache download differs'
report = dict(status='published and re-verified', repo=cfg['destination'], commit=commit.oid, sha256=pin['model_sha256'],
              bytes=pin['model_bytes'], url=f"https://huggingface.co/{cfg['destination']}/resolve/{commit.oid}/{cfg['filename']}", source=COMMIT)
os.makedirs('/kaggle/working', exist_ok=True); json.dump(report, open('/kaggle/working/publication.json', 'w'), indent=2)
print(json.dumps(report, indent=1))
