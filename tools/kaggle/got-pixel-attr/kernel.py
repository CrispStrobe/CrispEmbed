"""Kaggle CPU kernel: attribute the GOT-OCR2 native-vs-original divergence on the four early-diverging gold pages
(docs/got-ocr2-q4-research.md section 10) at @COMMIT@. Per page:
  A. native F16-r2 on the pinned JPEG/PNG bytes: pixels.f32 + gen_ids (CRISPEMBED_GOT_OCR_DUMP_DIR)
  B. native F16-r2 on a lossless PNG of the PIL-decoded RGB (separates the image decoder, stb_image vs
     Pillow/libjpeg, from everything after it)
  C. original chat() pixels (tools/ci-heavy/got_ocr2_orig_probe.py GOTImageEvalProcessor) for both inputs
  D. native teacher-forced on the ORIGINAL ids (CRISPEMBED_GOT_OCR_FORCE_IDS / _LOGITS_OUT): first step whose
     raw argmax differs from the original next token, with the original token's margin
Original ids: tests/results/got-ocr2-q4/gold-orig-kaggle-20261108/original.json. Output /kaggle/working/out/."""
import hashlib, io, json, os, subprocess, sys, urllib.request
COMMIT = '@COMMIT@'
PAGES = ('fr-old-tachard-siam-102', 'de-table-meyers-b1-168', 'receipt-cord-test-20', 'de-table-rgbl-1867-403')
TMP = '/kaggle/temp'; OUT = '/kaggle/working/out'; os.makedirs(OUT, exist_ok=True)
def sh(*a, **k): print('+', *a, flush=True); return subprocess.run(list(map(str, a)), **k)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', TMP + '/ce', check=True)
os.chdir(TMP + '/ce'); sh('git', 'checkout', '-q', COMMIT, check=True)
sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml', check=True)
sh('cmake', '-S', '.', '-B', TMP + '/b', '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', check=True)
sh('cmake', '--build', TMP + '/b', '--target', 'crispembed-cli', '--parallel', str(os.cpu_count()), check=True)
CLI = TMP + '/b/crispembed'
UA = {'User-Agent': 'CrispEmbed-OCR-eval/1.0 (https://github.com/CrispStrobe/CrispEmbed)'}
def fetch(u): return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=300).read()
F16 = TMP + '/f16.gguf'
open(F16, 'wb').write(fetch('https://huggingface.co/cstr/got-ocr2-crispembed-GGUF/resolve/'
                            'c0f1aa7b5b52c15002e911d5d630e6d0dcb3ba8f/got-ocr2-f16-r2.gguf'))
sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17', check=True)
sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', TMP + '/v', check=True); PY = TMP + '/v/bin/python'
sh(PY, '-m', 'pip', 'install', '-q', 'torch==2.4.1', 'torchvision==0.19.1', '--index-url', 'https://download.pytorch.org/whl/cpu', check=True)
sh(PY, '-m', 'pip', 'install', '-q', 'transformers==4.37.2', 'tokenizers<0.16', 'tiktoken', 'verovio', 'accelerate', 'safetensors',
   'numpy<2', 'pillow==11.3.0', 'pyarrow==17.0.0', check=True)
gold = {p['id']: p for p in json.load(open('docs/ocr-gold-corpus-v1.json'))['pages']}
orig = {r['page']: r for r in json.load(open('tests/results/got-ocr2-q4/gold-orig-kaggle-20261108/original.json'))['pages']}
env = dict(os.environ, CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE='1', OMP_NUM_THREADS=str(os.cpu_count()))
cases = []
for pid in PAGES:
    im = gold[pid]['image']; data = None
    if 'hf_parquet' in im:
        pq = TMP + '/cord.parquet'
        if not os.path.exists(pq): open(pq, 'wb').write(fetch(im['hf_parquet']['url']))
        data = subprocess.run([PY, '-c', 'import sys,pyarrow.parquet as pq;sys.stdout.buffer.write(pq.read_table(sys.argv[1],'
                               'columns=["image"]).slice(int(sys.argv[2]),1).to_pylist()[0]["image"]["bytes"])', pq,
                               str(im['hf_parquet']['row'])], capture_output=True, check=True).stdout
    else:
        for _ in range(4):
            for u in [im['url']] + im.get('fallback_urls', []):
                try: d = fetch(u)
                except Exception: continue
                if hashlib.sha256(d).hexdigest() == im['sha256']: data = d; break
            if data: break
    assert data and hashlib.sha256(data).hexdigest() == im['sha256'], f'pin mismatch {pid}'
    src = f'{TMP}/{pid}.src'; open(src, 'wb').write(data)
    png = f'{TMP}/{pid}.pil.png'
    sh(PY, '-c', 'import sys; from PIL import Image; Image.open(sys.argv[1]).convert("RGB").save(sys.argv[2])', src, png, check=True)
    cases.append((pid, src, png))
    oid = orig[pid]['ids']; mt = max(len(oid) + 8, 64)
    for tag, path in (('src', src), ('pilpng', png)):
        d = f'{OUT}/{pid}.{tag}'; os.makedirs(d, exist_ok=True)
        r = sh(CLI, '-m', F16, '--ocr', path, '--ocr-max-tokens', mt, '-t', os.cpu_count(),
               env=dict(env, CRISPEMBED_GOT_OCR_DUMP_DIR=d), capture_output=True, text=True)
        open(d + '/stderr.txt', 'w').write(r.stderr[-4000:])
    fi = f'{TMP}/{pid}.orig_ids.txt'; open(fi, 'w').write(' '.join(map(str, oid)))
    sh(CLI, '-m', F16, '--ocr', src, '--ocr-max-tokens', len(oid), '-t', os.cpu_count(),
       env=dict(env, CRISPEMBED_GOT_OCR_FORCE_IDS=fi, CRISPEMBED_GOT_OCR_LOGITS_OUT=f'{TMP}/{pid}.forced.f32'), capture_output=True)
args = [f'{pid}={src}' for pid, src, _ in cases] + [f'{pid}__pilpng={png}' for pid, _, png in cases]
sh(PY, '-I', 'tools/ci-heavy/got_ocr2_orig_probe.py', OUT, '1', '20', *args)
an = r'''
import json, os, sys
import numpy as np
OUT, TMP = sys.argv[1], sys.argv[2]; pages = sys.argv[3].split(',')
orig = {r['page']: r for r in json.load(open('tests/results/got-ocr2-q4/gold-orig-kaggle-20261108/original.json'))['pages']}
V = 151860; rep = {}
def ids(p):
    f = p + '/gen_ids.txt'
    return [int(x) for x in open(f).read().split()] if os.path.exists(f) else None
def first_div(a, b):
    if a is None: return 'missing'
    return None if a == b else next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
for pid in pages:
    o = orig[pid]['ids']; r = {}
    for tag, okey in (('src', pid), ('pilpng', pid + '__pilpng')):
        nat = np.fromfile(f'{OUT}/{pid}.{tag}/pixels.f32', np.float32) if os.path.exists(f'{OUT}/{pid}.{tag}/pixels.f32') else None
        op = np.load(f'{OUT}/orig.{okey}.pixels.npy').reshape(-1)
        d = None if nat is None or nat.size != op.size else np.abs(nat - op)
        r[tag] = dict(first_divergence_vs_original=first_div(ids(f'{OUT}/{pid}.{tag}'), o),
                      pixel_max_abs=None if d is None else float(d.max()), pixel_mean_abs=None if d is None else float(d.mean()),
                      pixel_frac_diff=None if d is None else float((d > 1e-6).mean()))
    oa = np.load(f'{OUT}/orig.{pid}.pixels.npy'); ob = np.load(f'{OUT}/orig.{pid}__pilpng.pixels.npy')
    r['orig_src_vs_orig_pilpng_max_abs'] = float(np.abs(oa - ob).max())
    f = f'{TMP}/{pid}.forced.f32'
    if os.path.exists(f):
        L = np.fromfile(f, np.float32); L = L.reshape(-1, V)[:len(o)]
        am = L.argmax(1); bad = np.nonzero(am != np.array(o[:len(am)]))[0]
        s = int(bad[0]) if bad.size else None
        r['forced'] = dict(steps=int(L.shape[0]), first_raw_argmax_mismatch=s, mismatches=int(bad.size),
                           at_step=None if s is None else dict(orig_token=o[s], native_argmax=int(am[s]),
                               margin_native_minus_orig=float(L[s, am[s]] - L[s, o[s]])))
    rep[pid] = r
json.dump(rep, open(OUT + '/attribution.json', 'w'), indent=1); print(json.dumps(rep, indent=1))
'''
sh(PY, '-I', '-c', an, OUT, TMP, ','.join(PAGES))
for f in os.listdir(OUT):
    if f.endswith('.npy'): os.remove(os.path.join(OUT, f))
for pid in PAGES:
    for tag in ('src', 'pilpng'):
        p = f'{OUT}/{pid}.{tag}/pixels.f32'
        if os.path.exists(p): os.remove(p)
