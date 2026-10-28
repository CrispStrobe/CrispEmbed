"""Kaggle CPU kernel: OCR quality on REAL scans (docs/ocr-real-scans-v1.json).

Clones CrispEmbed at @COMMIT@, fetches every page image by URL (or CORD-v2 parquet row) and verifies its
sha256, then runs one MODE (@MODE@; the same file is pushed once per mode so the four arms run in parallel
sessions):
  native-uocr2   published alias `unlimited-ocr2` (F16), fast decoder, gundam tiles (the original default)
  native-minicpm published alias `minicpm-ocr` (F16), prompt "Text Recognition:"
  ref-uocr2      original PyTorch Unlimited-OCR-2 (tools/unlimited-ocr2/reference.py contract), F32 CPU
  ref-minicpm    original Transformers MiniCPM-OCR-1.0 (tools/minicpm-ocr/reference.py contract), F32 CPU
Pages run in manifest priority order until the time budget. Per page: text, token IDs, CER/WER/word recall
(tools/kaggle/ocr-real-scans/score.py), wall time, peak RSS. Output: /kaggle/working/ocr-real-scans/.
Native-vs-original agreement is computed by `score.py summarize` over the shard receipts.
"""
import hashlib, json, os, re, subprocess, sys, time, urllib.request
T_START = time.time()
COMMIT = '@COMMIT@'
MODE = '@MODE@'
BUDGET = 10.9 * 3600            # stop starting pages after this
OUT = '/kaggle/working/ocr-real-scans'; os.makedirs(OUT, exist_ok=True)
TMP = '/kaggle/temp'
def el(): return time.time() - T_START
def sh(*a, **k): print('+', *a, flush=True); subprocess.run(list(a), check=True, **k)

sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', TMP + '/ce')
os.chdir(TMP + '/ce'); sh('git', 'checkout', '-q', COMMIT)
sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')
sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', TMP + '/venv'); PY = TMP + '/venv/bin/python'
sh(PY, '-m', 'pip', 'install', '-q', 'pillow==11.3.0', 'numpy==2.2.6', 'pyarrow==17.0.0', 'rapidfuzz==3.9.7')
sh(sys.executable, '-m', 'pip', 'install', '-q', 'rapidfuzz==3.9.7')
sys.path.insert(0, 'tools/kaggle/ocr-real-scans')
NT = str(os.cpu_count())
manifest = json.load(open('docs/ocr-real-scans-v1.json'))
report = dict(schema='crispembed-ocr-real-scans-run-v1', mode=MODE, commit=COMMIT, cpu_count=os.cpu_count(), images=[], runs=[],
              pages_meta={p['id']: dict(priority=p['priority'], tags=p['tags'], max_tokens=p['max_tokens']) for p in manifest['pages']})
try:
    report['cpu_model'] = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    report['mem_total_gb'] = round(int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemTotal'))) / 2**20, 1)
except Exception as e:
    print('cpuinfo', e)
def save(): report['elapsed_seconds'] = round(el(), 1); json.dump(report, open(OUT + '/result.json', 'w'), indent=1, ensure_ascii=False)


# ---- images: fetch + verify ----
UA = {'User-Agent': 'CrispEmbed-OCR-eval/1.0 (https://github.com/CrispStrobe/CrispEmbed)'}
def fetch(url):
    for i in range(5):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()
        except Exception as e: err = e; time.sleep(10 * (i + 1))
    raise err
os.makedirs(TMP + '/img', exist_ok=True)
parquets = {}
pages = []
for p in manifest['pages']:
    im = p['image']; rec = dict(id=p['id'], expected_sha256=im['sha256'])
    try:
        if 'hf_parquet' in im:
            hp = im['hf_parquet']
            if hp['url'] not in parquets:
                path = TMP + '/cord.parquet'; open(path, 'wb').write(fetch(hp['url'])); parquets[hp['url']] = path
                rec['parquet_sha256_ok'] = hashlib.sha256(open(path, 'rb').read()).hexdigest() == hp['sha256']
            out = subprocess.run([PY, '-c', 'import sys,pyarrow.parquet as pq;sys.stdout.buffer.write('
                                  'pq.read_table(sys.argv[1],columns=["image"]).slice(int(sys.argv[2]),1).to_pylist()[0]["image"]["bytes"])',
                                  parquets[hp['url']], str(hp['row'])], capture_output=True, check=True).stdout
            data = out
        else:
            data = fetch(im['url'])
        rec['sha256'] = hashlib.sha256(data).hexdigest(); rec['sha256_ok'] = rec['sha256'] == im['sha256']
        path = f"{TMP}/img/{p['id']}.jpg"; open(path, 'wb').write(data)
        pages.append(dict(id=p['id'], path=path, max_tokens=p['max_tokens'], gt=p['ground_truth'], priority=p['priority']))
    except Exception as e:
        rec['error'] = repr(e)[:500]
    report['images'].append(rec); print('image', rec, flush=True)
save()
from score import score  # noqa: E402  (rapidfuzz may be absent in the Kaggle base python: pure-python fallback)


def run(args, env):
    so, se = open(TMP + '/so', 'w+'), open(TMP + '/se', 'w+')
    t = time.monotonic(); p = subprocess.Popen(args, env=env, stdout=so, stderr=se)
    _, status, ru = os.wait4(p.pid, 0); wall = time.monotonic() - t
    so.seek(0); se.seek(0)
    return os.waitstatus_to_exitcode(status), so.read(), se.read(), round(wall, 2), round(ru.ru_maxrss / 1024, 1)


def native(alias, arm, env_extra, extra_args):
    t0 = time.time()
    sh('cmake', '-S', '.', '-B', TMP + '/b', '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF')
    sh('cmake', '--build', TMP + '/b', '--target', 'crispembed-cli', '--parallel', NT)
    report['build_seconds'] = round(time.time() - t0, 1)
    cli = TMP + '/b/crispembed'
    env = dict(os.environ, CRISPEMBED_CACHE_DIR=TMP + '/cache', OMP_NUM_THREADS=NT, **env_extra)
    env.pop('CRISPEMBED_ALLOW_UNPINNED_MODEL', None)
    small = min(pages, key=lambda p: os.path.getsize(p['path']))
    code, out, err, wall, rss = run([cli, '-m', alias, '--ocr', small['path'], '--ocr-max-tokens', '4', '-t', NT] + extra_args, env)
    report['alias'] = dict(alias=alias, exit=code, wall_seconds=wall, verified='Verifying SHA-256' in err, stderr_tail=err[-1500:])
    save(); print('alias', report['alias'], flush=True)
    for p in pages:
        row = dict(page=p['id'], arm=arm, max_tokens=p['max_tokens'])
        if el() > BUDGET:
            row['skipped'] = 'time budget'; report['runs'].append(row); save(); continue
        e = dict(env)
        if 'CRISPEMBED_MINICPM_OCR' in e:
            e['CRISPEMBED_MINICPM_OCR_DUMP'] = f"{TMP}/dump/{p['id']}"; os.makedirs(e['CRISPEMBED_MINICPM_OCR_DUMP'], exist_ok=True)
        code, out, err, wall, rss = run([cli, '-m', alias, '--ocr', p['path'], '--ocr-max-tokens', str(p['max_tokens']), '-t', NT] + extra_args, e)
        m = re.search(r'gen_ids \(\d+\):([^\n]*)', err)
        ids = [int(x) for x in m.group(1).split()] if m else None
        row.update(exit=code, wall_seconds=wall, peak_rss_mb=rss, n_ids=len(ids) if ids else None, ids=ids,
                   cap_hit=bool(ids) and len(ids) >= p['max_tokens'], text=out.strip(), stderr_tail=err[-1500:] if code else err[-300:])
        if code == 0: row.update(score(out, p['gt']))
        report['runs'].append(row); save()
        print(f"{arm} {p['id']} exit={code} {wall}s rss={rss} ids={row['n_ids']} cer={row.get('cer')} rec={row.get('word_recall')}", flush=True)


def reference(kind, arm, pins):
    sh(PY, '-m', 'pip', 'install', '-q', 'torch==2.13.0', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu')
    sh(PY, '-m', 'pip', 'install', '-q', *pins)
    lst = TMP + '/ref-pages.json'
    json.dump([dict(id=p['id'], path=p['path'], max_tokens=p['max_tokens']) for p in pages], open(lst, 'w'))
    ref_out = TMP + '/ref/reference.json'; os.makedirs(TMP + '/ref', exist_ok=True)
    deadline = T_START + BUDGET
    t0 = time.time()
    r = subprocess.run([PY, 'tools/kaggle/ocr-real-scans/ref_worker.py', kind, lst, ref_out, str(deadline)],
                       env=dict(os.environ, OMP_NUM_THREADS=NT, HF_HUB_DISABLE_TELEMETRY='1'),
                       timeout=max(600, 11.6 * 3600 - el()), capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-5000:], flush=True)
    ref = json.load(open(ref_out)) if os.path.exists(ref_out) else {}
    report['reference_meta'] = {k: v for k, v in ref.items() if k != 'cases'}
    report['reference_meta'].update(exit=r.returncode, stderr_tail=r.stderr[-3000:], wall_total=round(time.time() - t0, 1))
    gt = {p['id']: p['gt'] for p in pages}
    for c in ref.get('cases', []):
        row = dict(page=c['id'], arm=arm, wall_seconds=c.get('seconds'), peak_rss_mb=c.get('peak_rss_mb_so_far'), ids=c.get('ids'),
                   n_ids=len(c['ids']) if c.get('ids') else None, text=c.get('text'), cap_hit=c.get('cap_hit'),
                   max_tokens=c.get('max_tokens'), skipped=c.get('skipped'), error=c.get('error'))
        if c.get('text') is not None: row.update(score(c['text'], gt[c['id']]))
        report['runs'].append({k: v for k, v in row.items() if v is not None})
    save()


try:
    if MODE == 'native-uocr2':
        native('unlimited-ocr2', 'uocr2-f16-fast-tiled', dict(CRISPEMBED_UNLIMITED_OCR2='1', CRISPEMBED_UNLIMITED_OCR2_FAST='1',
                                                             CRISPEMBED_UNLIMITED_OCR2_TILES='1', UOCR_DBG='1'), [])
    elif MODE == 'native-minicpm':
        native('minicpm-ocr', 'minicpm-f16', dict(CRISPEMBED_MINICPM_OCR='1', CRISPEMBED_QWEN35_OCR='1',
                                                  CRISPEMBED_QWEN35_WEIGHT_CACHE='1'), ['--ocr-prompt', 'Text Recognition:'])
    elif MODE == 'ref-uocr2':
        reference('uocr2', 'uocr2-original-f32', ['transformers==4.52.1', 'huggingface-hub==0.36.0', 'tokenizers==0.21.4',
                                                  'addict==2.4.0', 'einops==0.8.2', 'easydict==1.13'])
    elif MODE == 'ref-minicpm':
        reference('minicpm', 'minicpm-original-f32', ['transformers==5.21.0'])
    else:
        raise SystemExit('unknown MODE ' + MODE)
except Exception as e:
    report['fatal'] = repr(e)[:3000]; save(); raise
finally:
    save()
    import score as S
    S.summarize(OUT + '/summary.md', [OUT + '/result.json'])
