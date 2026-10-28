"""Kaggle CPU kernel: verify the Unlimited-OCR-2 trial fixes at @COMMIT@.

Uses the PUBLISHED GGUFs (aliases unlimited-ocr2 / unlimited-ocr2-q8, pinned SHA-256) and the
trial's synthetic pages (tools/kaggle/unlimited-ocr2-trial/trial_worker.py render, checked
against the trial's page hashes). Ground truth for agreement is the ORIGINAL model's ids and
decoded text frozen in tests/results/unlimited-ocr2/trial.json (reference.cases). Checks:
  1. text: default native stdout == original decoded text byte-for-byte (unit test
     test-unlimited-ocr2-text + end-to-end), plus the `clean` and `legacy` renderings;
  2. tiling: the DEFAULT env (no TILES var) reproduces the original ids (fast decoder on all
     4 pages x F16/Q8_0; exact = engine-default decoder on one page);
  3. speed: before (TILES=1, VISION_CACHE=0, UOCR2_FAST_CHUNK=256 = old code paths) vs after
     (defaults), same box, ids must be equal; per-stage [time] lines; ablations; vision
     features byte-equal with the cache on/off; a GGML_NATIVE=ON build for reference.
Output: /kaggle/working/verify/{verify.json,summary.md}.
"""
import hashlib, json, os, re, subprocess, sys, time
T_START = time.time()
COMMIT = '@COMMIT@'
BUDGET = 10.5 * 3600
OUT = '/kaggle/working/verify'; os.makedirs(OUT, exist_ok=True)
def el(): return time.time() - T_START
def sh(*a, **k): print('+', *a, flush=True); subprocess.run(list(a), check=True, **k)

sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT); sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')
sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv'); PY = '/kaggle/temp/venv/bin/python'
sh(PY, '-m', 'pip', 'install', '-q', 'pillow==11.3.0', 'numpy==2.2.6')
NT = str(os.cpu_count())
report = dict(schema='crispembed-unlimited-ocr2-verify-v1', commit=COMMIT, cpu_count=os.cpu_count(), tests={}, runs=[],
              vision_bytes=[], checks={})
try:
    report['cpu_model'] = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    report['cpu_flags_avx512f'] = 'avx512f' in open('/proc/cpuinfo').read()
except Exception as e:
    print('cpuinfo', e)
save = lambda: json.dump(report, open(OUT + '/verify.json', 'w'), indent=2, ensure_ascii=False)

def build(bdir, native):
    t0 = time.time()
    sh('cmake', '-S', '.', '-B', bdir, '-DCMAKE_BUILD_TYPE=Release', f'-DGGML_NATIVE={"ON" if native else "OFF"}', '-DGGML_CUDA=OFF')
    sh('cmake', '--build', bdir, '--target', 'crispembed-cli', 'test-unlimited-ocr2-text', 'test-unlimited-ocr2-tiles',
       '--parallel', NT)
    return round(time.time() - t0, 1)

report['build_seconds'] = build('/kaggle/temp/b', False); save()
CLI = '/kaggle/temp/b/crispembed'
for t in ('test-unlimited-ocr2-text', 'test-unlimited-ocr2-tiles'):
    r = subprocess.run(['/kaggle/temp/b/' + t], capture_output=True, text=True)
    report['tests'][t] = dict(exit=r.returncode, out=(r.stdout + r.stderr)[-3000:]); print(t, r.returncode, r.stdout, r.stderr, flush=True)
save()

WORKER = 'tools/kaggle/unlimited-ocr2-trial/trial_worker.py'
sh(PY, WORKER, 'render', OUT + '/pages')
pages = {p['id']: p for p in json.load(open(OUT + '/pages/pages.json'))}
trial = json.load(open('tests/results/unlimited-ocr2/trial.json'))
trial_sha = {p['id']: p['sha256'] for p in trial['pages']}
report['page_sha_matches_trial'] = {k: pages[k]['sha256'] == trial_sha.get(k) for k in pages}
REF = {}
for c in trial['reference']['cases']:
    ids = c['ids'] if isinstance(c['ids'], list) else json.loads(c['ids'])
    REF[c['id']] = dict(ids=ids, text=c['text'])
save()


def run(args, env):
    so, se = open('/kaggle/temp/so', 'wb+'), open('/kaggle/temp/se', 'w+')
    t = time.monotonic()
    p = subprocess.Popen(args, env=env, stdout=so, stderr=se)
    _, status, ru = os.wait4(p.pid, 0)
    wall = time.monotonic() - t
    so.seek(0); se.seek(0); out, err = so.read(), se.read()
    return os.waitstatus_to_exitcode(status), out, err, round(wall, 2), round(ru.ru_maxrss / 1024, 1)


def env_of(**kw):
    e = dict(os.environ, CRISPEMBED_CACHE_DIR='/kaggle/temp/cache', CRISPEMBED_UNLIMITED_OCR2='1', UOCR_DBG='1',
             OMP_NUM_THREADS=NT)
    for k in ('CRISPEMBED_ALLOW_UNPINNED_MODEL', 'HF_HUB_OFFLINE', 'CRISPEMBED_UNLIMITED_OCR2_TILES',
              'CRISPEMBED_UNLIMITED_OCR2_OUTPUT', 'CRISPEMBED_UNLIMITED_OCR2_VISION_CACHE', 'UOCR2_FAST_CHUNK',
              'CRISPEMBED_UNLIMITED_OCR2_FAST'):
        e.pop(k, None)
    e.update({k: v for k, v in kw.items() if v is not None}); return e


# ---- resolve aliases (download + pinned hash) ----
MODELS = {}
for alias in ('unlimited-ocr2', 'unlimited-ocr2-q8'):
    code, out, err, wall, rss = run([CLI, '-m', alias, '--ocr', pages['receipt']['path'], '--ocr-max-tokens', '2', '-t', NT],
                                    env_of(CRISPEMBED_UNLIMITED_OCR2_FAST='1'))
    m = re.search(r'(/kaggle/temp/cache/\S+\.gguf)', err)
    fn = 'unlimited-ocr-2-f16.gguf' if alias == 'unlimited-ocr2' else 'unlimited-ocr-2-q8_0.gguf'
    path = '/kaggle/temp/cache/' + fn
    report.setdefault('alias', []).append(dict(alias=alias, exit=code, wall=wall, file_exists=os.path.exists(path),
                                               verified='Verifying SHA-256' in err, err_tail=err[-800:] if code else ''))
    if os.path.exists(path): MODELS[alias] = path
    print(alias, code, wall, flush=True)
save()

OLD = dict(CRISPEMBED_UNLIMITED_OCR2_TILES='1', CRISPEMBED_UNLIMITED_OCR2_VISION_CACHE='0', UOCR2_FAST_CHUNK='256')
TIME_RE = re.compile(r'^\s*\[(?:time|uocr2-fast)\][^\n]*', re.M)


def ocr(tag, alias, page, cli=None, max_tokens='1024', **kw):
    row = dict(tag=tag, alias=alias, page=page, env={k: v for k, v in kw.items() if v is not None})
    if alias not in MODELS or el() > BUDGET:
        row['skipped'] = 'model missing' if alias not in MODELS else 'time budget'
        report['runs'].append(row); save(); return row
    code, out, err, wall, rss = run([cli or CLI, '-m', MODELS[alias], '--ocr', pages[page]['path'], '--ocr-max-tokens', max_tokens,
                                     '-t', NT], env_of(**kw))
    m = re.search(r'gen_ids \(\d+\):([^\n]*)', err)
    ids = [int(x) for x in m.group(1).split()] if m else None
    text = out.decode('utf-8', 'replace')
    if text.endswith('\n'): text = text[:-1]          # the CLI appends one newline
    ref = REF.get(page)
    row.update(exit=code, wall_seconds=wall, peak_rss_mb=rss, n_ids=len(ids) if ids else None, ids=ids, text=text,
               timing=[l.strip() for l in TIME_RE.findall(err)][-60:],
               stderr_tail=err[-2500:] if code else err[-400:])
    if ref and ids is not None:
        a, b = ids, ref['ids']
        row['ids_equal_original'] = a == b
        row['first_difference'] = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
        row['text_bytes_equal_original'] = text == ref['text']
    report['runs'].append(row); save()
    print(f"{tag:22s} {alias:18s} {page:11s} exit={code} {wall:8.1f}s ids={row['n_ids']} "
          f"ids_eq={row.get('ids_equal_original')} text_eq={row.get('text_bytes_equal_original')}", flush=True)
    return row


P4 = ['table', 'receipt', 'paragraphs', 'german']
# 1+2: defaults (after), fast decoder, both precisions
for pid in P4:
    ocr('after', 'unlimited-ocr2', pid, CRISPEMBED_UNLIMITED_OCR2_FAST='1')
for pid in P4:
    ocr('after', 'unlimited-ocr2-q8', pid, CRISPEMBED_UNLIMITED_OCR2_FAST='1')
# 3: before (old code paths, tiled), F16
for pid in P4:
    ocr('before', 'unlimited-ocr2', pid, CRISPEMBED_UNLIMITED_OCR2_FAST='1', **OLD)
# renderings
ocr('clean', 'unlimited-ocr2', 'table', CRISPEMBED_UNLIMITED_OCR2_FAST='1', CRISPEMBED_UNLIMITED_OCR2_OUTPUT='clean')
ocr('legacy-text', 'unlimited-ocr2', 'table', CRISPEMBED_UNLIMITED_OCR2_FAST='1', CRISPEMBED_UNLIMITED_OCR2_OUTPUT='legacy')
# forced untiled (speed opt-out)
ocr('untiled', 'unlimited-ocr2', 'table', CRISPEMBED_UNLIMITED_OCR2_FAST='1', CRISPEMBED_UNLIMITED_OCR2_TILES='0')
# ablations (F16 table): each change alone
ocr('ablate-cache-only', 'unlimited-ocr2', 'table', CRISPEMBED_UNLIMITED_OCR2_FAST='1', UOCR2_FAST_CHUNK='256')
ocr('ablate-chunk-only', 'unlimited-ocr2', 'table', CRISPEMBED_UNLIMITED_OCR2_FAST='1', CRISPEMBED_UNLIMITED_OCR2_VISION_CACHE='0')

# vision features byte-equal with the cache on/off (shared by both decoders)
for alias in ('unlimited-ocr2', 'unlimited-ocr2-q8'):
    hs = {}
    for vc in ('1', '0'):
        dump = f'/kaggle/temp/vis_{alias}_{vc}.f32'
        r = ocr(f'vision-dump-cache{vc}', alias, 'table', max_tokens='1', CRISPEMBED_UNLIMITED_OCR2_FAST='1',
                CRISPEMBED_UNLIMITED_OCR2_VISION_CACHE=vc, UOCR2_DUMP_VISION=dump)
        hs[vc] = hashlib.sha256(open(dump, 'rb').read()).hexdigest() if os.path.exists(dump) else None
    report['vision_bytes'].append(dict(alias=alias, sha_cache_on=hs['1'], sha_cache_off=hs['0'],
                                       equal=hs['1'] is not None and hs['1'] == hs['0']))
    save()

# engine-default (exact) decoder with default tiling, one page
ocr('after-exact-decoder', 'unlimited-ocr2', 'paragraphs')

# GGML_NATIVE=ON build (what a local source build gets by default)
if el() < BUDGET - 3600:
    try:
        report['build_native_seconds'] = build('/kaggle/temp/bn', True)
        for pid in ('table', 'receipt'):
            ocr('after-native-build', 'unlimited-ocr2', pid, cli='/kaggle/temp/bn/crispembed', CRISPEMBED_UNLIMITED_OCR2_FAST='1')
        ocr('after-native-build', 'unlimited-ocr2-q8', 'table', cli='/kaggle/temp/bn/crispembed', CRISPEMBED_UNLIMITED_OCR2_FAST='1')
    except Exception as e:
        report['build_native_error'] = repr(e)[:2000]
report['elapsed_seconds'] = round(el(), 1); save()

# ---- checks + summary ----
R = [r for r in report['runs'] if 'skipped' not in r]
def rows(tag, alias=None): return [r for r in R if r['tag'] == tag and (alias is None or r['alias'] == alias)]
ck = report['checks']
ck['unit_tests_pass'] = all(v['exit'] == 0 for v in report['tests'].values())
ck['f16_default_ids_equal_original_all4'] = len(rows('after', 'unlimited-ocr2')) == 4 and all(r.get('ids_equal_original') for r in rows('after', 'unlimited-ocr2'))
ck['f16_default_text_bytes_equal_original_all4'] = len(rows('after', 'unlimited-ocr2')) == 4 and all(r.get('text_bytes_equal_original') for r in rows('after', 'unlimited-ocr2'))
ck['q8_default_ids_equal_original'] = {r['page']: r.get('ids_equal_original') for r in rows('after', 'unlimited-ocr2-q8')}
ck['q8_text_bytes_equal_when_ids_equal'] = all(r.get('text_bytes_equal_original') for r in rows('after', 'unlimited-ocr2-q8') if r.get('ids_equal_original'))
before = {r['page']: r for r in rows('before')}
ck['before_after_ids_equal'] = {r['page']: (before.get(r['page'], {}).get('ids') == r['ids']) for r in rows('after', 'unlimited-ocr2')}
ck['ablation_ids_equal_after'] = {r['tag']: r['ids'] == next((a['ids'] for a in rows('after', 'unlimited-ocr2') if a['page'] == 'table'), None)
                                  for r in R if r['tag'].startswith('ablate')}
ck['vision_bytes_equal_cache_on_off'] = {v['alias']: v['equal'] for v in report['vision_bytes']}
ck['exact_decoder_default_ids_equal_original'] = [r.get('ids_equal_original') for r in rows('after-exact-decoder')]
ck['native_build_ids_equal_original'] = {f"{r['alias']}/{r['page']}": r.get('ids_equal_original') for r in rows('after-native-build')}
save()

L = ['# Unlimited-OCR-2 trial-fix verification', '', f'Commit `{COMMIT}`, {report.get("cpu_model", "?")} x{os.cpu_count()} '
     f'(avx512f={report.get("cpu_flags_avx512f")}), build {report["build_seconds"]}s. Pages re-rendered; sha256 == trial: '
     f'{report["page_sha_matches_trial"]}.', '', '## Checks', '']
for k, v in ck.items(): L.append(f'- **{k}**: {v}')
L += ['', '## Runs', '', '| tag | model | page | exit | wall s | RSS MB | ids | ids==orig | first diff | text==orig |', '|---|---|---|---|---|---|---|---|---|---|']
for r in report['runs']:
    if 'skipped' in r: L.append(f"| {r['tag']} | {r['alias']} | {r['page']} | skipped: {r['skipped']} |||||||"); continue
    L.append(f"| {r['tag']} | {r['alias']} | {r['page']} | {r['exit']} | {r['wall_seconds']} | {r['peak_rss_mb']} | {r['n_ids']} | "
             f"{r.get('ids_equal_original')} | {r.get('first_difference')} | {r.get('text_bytes_equal_original')} |")
L += ['', '## Speed (F16, fast decoder, tiled): before -> after', '', '| page | before s | after s | speedup |', '|---|---|---|---|']
for r in rows('after', 'unlimited-ocr2'):
    b = before.get(r['page'])
    if b and b.get('wall_seconds'):
        L.append(f"| {r['page']} | {b['wall_seconds']} | {r['wall_seconds']} | {b['wall_seconds'] / r['wall_seconds']:.2f}x |")
L += ['', '## Stage timing (table, F16)', '']
for tag in ('before', 'after', 'ablate-cache-only', 'ablate-chunk-only', 'untiled', 'after-native-build'):
    for r in rows(tag, 'unlimited-ocr2'):
        if r['page'] == 'table':
            L += [f'### {tag}', '```'] + [t for t in r['timing'] if 'tile ' not in t][:40] + ['```']
for tag in ('clean', 'legacy-text'):
    for r in rows(tag):
        L += [f'## `{tag}` rendering (table)', '```', r['text'], '```']
open(OUT + '/summary.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L), flush=True)
