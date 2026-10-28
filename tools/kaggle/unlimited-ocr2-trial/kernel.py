"""Kaggle CPU kernel: user-style trial of the PUBLISHED Unlimited-OCR-2 GGUFs.

Builds crispembed-cli at @COMMIT@, resolves the `unlimited-ocr2-q8` / `unlimited-ocr2`
aliases exactly as a user would (download + pinned SHA-256 check into an empty cache,
then a second resolution that must be a cache hit), renders license-clean synthetic
pages (tools/kaggle/unlimited-ocr2-trial/trial_worker.py) and OCRs them across
Q8_0/F16 x untiled/tiled x fast/exact decoder. Per run: text, CER vs ground truth,
wall time, peak RSS, token IDs. Then (time permitting) the original PyTorch model on
a subset for token-ID agreement. Output: /kaggle/working/trial/{trial.json,summary.md}.
No public-domain scans are fetched (no pinned, license-verified source); a degraded
"scan-sim" page stands in.
"""
import hashlib, json, os, re, subprocess, sys, time
T_START = time.time()
COMMIT = '@COMMIT@'
BUDGET_NATIVE = 7.0 * 3600      # stop starting native runs after this
BUDGET_TOTAL = 11.2 * 3600      # reference must finish before this
OUT = '/kaggle/working/trial'; os.makedirs(OUT, exist_ok=True)
def el(): return time.time() - T_START
def sh(*a, **k): print('+', *a, flush=True); subprocess.run(list(a), check=True, **k)

sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT); sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')
sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv'); PY = '/kaggle/temp/venv/bin/python'
sh(PY, '-m', 'pip', 'install', '-q', 'pillow==11.3.0', 'numpy==2.2.6')
t0 = time.time()
sh('cmake', '-S', '.', '-B', '/kaggle/temp/b', '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF')
sh('cmake', '--build', '/kaggle/temp/b', '--target', 'crispembed-cli', '--parallel', str(os.cpu_count()))
build_s = time.time() - t0
CLI = '/kaggle/temp/b/crispembed'
WORKER = 'tools/kaggle/unlimited-ocr2-trial/trial_worker.py'
sh(PY, WORKER, 'render', OUT + '/pages')
pages = json.load(open(OUT + '/pages/pages.json'))
NT = str(os.cpu_count())
report = dict(schema='crispembed-unlimited-ocr2-trial-v1', commit=COMMIT, hub_revision='e3af0c0c4bfa241c161db64ca36f3c63d3ca8378',
              cpu_count=os.cpu_count(), build_seconds=round(build_s, 1), pages=pages, alias=[], runs=[], reference=None)
try:
    report['cpu_model'] = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    report['mem_total_gb'] = round(int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemTotal'))) / 2**20, 1)
except Exception as e:
    print('cpuinfo', e)
save = lambda: json.dump(report, open(OUT + '/trial.json', 'w'), indent=2, ensure_ascii=False)


def run(args, env):
    """Run the CLI, return (exit, stdout, stderr, wall, peak_rss_mb) with per-child rusage via wait4."""
    so, se = open('/kaggle/temp/so', 'w+'), open('/kaggle/temp/se', 'w+')
    t = time.monotonic()
    p = subprocess.Popen(args, env=env, stdout=so, stderr=se)
    _, status, ru = os.wait4(p.pid, 0)
    wall = time.monotonic() - t
    so.seek(0); se.seek(0); out, err = so.read(), se.read()
    code = os.waitstatus_to_exitcode(status)
    return code, out, err, round(wall, 2), round(ru.ru_maxrss / 1024, 1)


def base_env(**kw):
    e = dict(os.environ, CRISPEMBED_CACHE_DIR='/kaggle/temp/cache', CRISPEMBED_UNLIMITED_OCR2='1', UOCR_DBG='1',
             OMP_NUM_THREADS=NT)
    e.pop('CRISPEMBED_ALLOW_UNPINNED_MODEL', None); e.pop('HF_HUB_OFFLINE', None)
    e.update(kw); return e


# ---- alias resolution: fresh download + pinned hash, then cache hit ----
EXPECTED = {}
for line in open('examples/cli/model_hashes.h'):
    m = re.search(r'Unlimited-OCR-2-GGUF/resolve/[0-9a-f]+/(unlimited-ocr-2-[^"]+\.gguf)",\s*"([0-9a-f]{64})"', line)
    if m: EXPECTED[m.group(1)] = m.group(2)
if not EXPECTED:  # pins may sit on the following line
    txt = open('examples/cli/model_hashes.h').read()
    for m in re.finditer(r'(unlimited-ocr-2-[\w.]+\.gguf)"\s*,\s*"([0-9a-f]{64})"', txt): EXPECTED[m.group(1)] = m.group(2)
report['pinned_sha256'] = EXPECTED
MODELS = {}
probe = pages[0]['path']
for alias, fname in (('unlimited-ocr2-q8', 'unlimited-ocr-2-q8_0.gguf'), ('unlimited-ocr2', 'unlimited-ocr-2-f16.gguf')):
    rec = dict(alias=alias, file=fname)
    for attempt in ('fresh', 'cached'):
        code, out, err, wall, rss = run([CLI, '-m', alias, '--ocr', probe, '--ocr-max-tokens', '8', '-t', NT], base_env())
        rec[attempt] = dict(exit=code, wall_seconds=wall, peak_rss_mb=rss, verified='Verifying SHA-256' in err,
                            downloaded='Download' in err or 'Verifying' in err, stderr_head=err[:1500], stderr_tail=err[-1500:])
        print(alias, attempt, code, wall, flush=True)
    path = '/kaggle/temp/cache/' + fname
    rec['cached_file_exists'] = os.path.exists(path)
    if rec['cached_file_exists']:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while b := f.read(16 << 20): h.update(b)
        rec['bytes'] = os.path.getsize(path); rec['sha256'] = h.hexdigest()
        rec['sha256_matches_pin'] = rec['sha256'] == EXPECTED.get(fname)
        MODELS[alias] = path
    report['alias'].append(rec); save()


def norm(s):
    s = re.sub(r'<BLOCK>\w*\[[^\]]*\]<FORK>', '\n', s)
    s = re.sub(r'<[^>]{1,40}>', ' ', s)                   # layout / html tags
    s = s.replace('|', ' ').replace('&amp;', '&').replace('&quot;', '"')
    return ' '.join(s.split())


def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, gt):
    p, g = norm(pred), norm(gt)
    return round(lev(p, g) / max(1, len(g)), 4)


# ---- run matrix, ordered by priority (late entries are dropped first by the time budget) ----
by_id = {p['id']: p for p in pages}
plan = []
for pid in by_id:                                   # 1: Q8 fast, both tilings
    plan += [('unlimited-ocr2-q8', pid, False, True), ('unlimited-ocr2-q8', pid, True, True)]
for pid in by_id:                                   # 2: F16 fast, both tilings
    plan += [('unlimited-ocr2', pid, False, True), ('unlimited-ocr2', pid, True, True)]
for pid in ('receipt', 'paragraphs', 'german'):     # 3: exact decoder controls
    plan += [('unlimited-ocr2-q8', pid, True, False), ('unlimited-ocr2', pid, True, False)]
for alias, pid, tiled, fast in plan:
    row = dict(alias=alias, page=pid, tiled=tiled, fast=fast)
    if alias not in MODELS:
        row['skipped'] = 'alias did not resolve'; report['runs'].append(row); save(); continue
    if el() > BUDGET_NATIVE:
        row['skipped'] = 'time budget'; report['runs'].append(row); save(); continue
    mt = '2048' if pid == 'a4-dense' else '1024'
    env = base_env(CRISPEMBED_UNLIMITED_OCR2_FAST='1' if fast else '0')
    if tiled: env['CRISPEMBED_UNLIMITED_OCR2_TILES'] = '1'
    code, out, err, wall, rss = run([CLI, '-m', MODELS[alias], '--ocr', by_id[pid]['path'], '--ocr-max-tokens', mt, '-t', NT], env)
    m = re.search(r'gen_ids \(\d+\):([^\n]*)', err)
    ids = [int(x) for x in m.group(1).split()] if m else None
    row.update(exit=code, wall_seconds=wall, peak_rss_mb=rss, max_tokens=int(mt), n_ids=len(ids) if ids else None,
               cap_hit=bool(ids) and len(ids) >= int(mt), ids=ids, text=out.strip(),
               cer=cer(out, by_id[pid]['ground_truth']) if code == 0 else None,
               stderr_tail=err[-2500:] if code != 0 else err[-600:])
    report['runs'].append(row); save()
    print(f"{alias:18s} {pid:11s} tiled={tiled} fast={fast} exit={code} {wall:8.1f}s rss={rss}MB "
          f"ids={row['n_ids']} cer={row['cer']}", flush=True)

# ---- original PyTorch reference (time permitting) ----
REF_IDS = ['receipt', 'paragraphs', 'german', 'table']
remaining = BUDGET_TOTAL - el()
if remaining < 2.5 * 3600:
    report['reference'] = dict(skipped=f'not enough time left ({remaining/3600:.1f} h) after native runs')
else:
    try:
        sh(PY, '-m', 'pip', 'install', '-q', 'torch==2.13.0', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu')
        sh(PY, '-m', 'pip', 'install', '-q', 'transformers==4.52.1', 'huggingface-hub==0.36.0', 'tokenizers==0.21.4',
           'addict==2.4.0', 'einops==0.8.2', 'easydict==1.13')
        ref_out = '/kaggle/temp/ref/reference.json'; os.makedirs('/kaggle/temp/ref', exist_ok=True)
        deadline = T_START + BUDGET_TOTAL - 1800
        r = subprocess.run([PY, WORKER, 'reference', OUT + '/pages/pages.json', ref_out, str(deadline)] + REF_IDS,
                           env=dict(os.environ, OMP_NUM_THREADS=NT, HF_HUB_DISABLE_TELEMETRY='1'),
                           timeout=max(600, BUDGET_TOTAL - el() - 900), capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        ref = json.load(open(ref_out)) if os.path.exists(ref_out) else dict(error='no output', stderr=r.stderr[-3000:])
        ref['exit'] = r.returncode
    except Exception as e:
        ref = dict(error=repr(e)[:3000])
        if os.path.exists('/kaggle/temp/ref/reference.json'):
            ref.update(partial=json.load(open('/kaggle/temp/ref/reference.json')))
    for c in ref.get('cases', []):
        if c.get('text') is not None:
            c['cer'] = cer(c['text'], by_id[c['id']]['ground_truth'])
        agree = []
        for row in report['runs']:
            if row['page'] == c['id'] and row.get('ids') and c.get('ids'):
                a, b = row['ids'], c['ids']
                fd = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
                agree.append(dict(alias=row['alias'], tiled=row['tiled'], fast=row['fast'], ids_exact=a == b,
                                  first_difference=fd, prefix_agreement=round((fd if fd is not None else len(a)) / max(1, len(b)), 4),
                                  native_n=len(a), reference_n=len(b)))
        c['native_agreement'] = agree
    report['reference'] = ref
report['elapsed_seconds'] = round(el(), 1); save()

# ---- summary.md ----
L = ['# Unlimited-OCR-2 published-model trial', '', f'Commit `{COMMIT}`, Hub revision `{report["hub_revision"]}`, '
     f'{report.get("cpu_model", "?")} x{os.cpu_count()}, {report.get("mem_total_gb", "?")} GB RAM. Build {build_s:.0f}s.', '',
     '## Alias resolution', '', '| alias | fresh exit | fresh s | SHA verified | cached exit | cached s | re-download on cache hit | sha256 == pin |',
     '|---|---|---|---|---|---|---|---|']
for a in report['alias']:
    f, c = a['fresh'], a['cached']
    L.append(f"| {a['alias']} | {f['exit']} | {f['wall_seconds']} | {f['verified']} | {c['exit']} | {c['wall_seconds']} | {c['downloaded']} | {a.get('sha256_matches_pin')} |")
L += ['', '## OCR runs', '', '| page | model | tiled | decoder | exit | wall s | peak RSS MB | ids | cap | CER |', '|---|---|---|---|---|---|---|---|---|---|']
for r in report['runs']:
    if 'skipped' in r:
        L.append(f"| {r['page']} | {r['alias']} | {r['tiled']} | {'fast' if r['fast'] else 'exact'} | skipped: {r['skipped']} ||||||"); continue
    L.append(f"| {r['page']} | {r['alias']} | {r['tiled']} | {'fast' if r['fast'] else 'exact'} | {r['exit']} | {r['wall_seconds']} | "
             f"{r['peak_rss_mb']} | {r['n_ids']} | {r['cap_hit']} | {r['cer']} |")
ref = report['reference'] or {}
L += ['', '## Original PyTorch reference', '']
if 'skipped' in ref or 'error' in ref:
    L.append(f"Not available: {ref.get('skipped') or ref.get('error')}")
for c in ref.get('cases', []):
    L.append(f"- **{c['id']}**: {c.get('seconds')} s, CER {c.get('cer')}, {c.get('skipped') or c.get('error') or ''}")
    for g in c.get('native_agreement', []):
        L.append(f"  - {g['alias']} tiled={g['tiled']} fast={g['fast']}: ids_exact={g['ids_exact']} first_diff={g['first_difference']} "
                 f"prefix={g['prefix_agreement']} ({g['native_n']} vs {g['reference_n']})")
L += ['', f"Elapsed {report['elapsed_seconds']/3600:.2f} h."]
open(OUT + '/summary.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
