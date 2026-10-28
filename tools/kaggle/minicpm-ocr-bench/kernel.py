"""Kaggle CPU kernel: per-stage benchmark of native MiniCPM-OCR-1.0 (CRISPEMBED_MINICPM_OCR)
against the frozen Transformers reference (crispembed-minicpm-ocr-reference output, mounted
via kernel_sources), plus Transformers and llama.cpp speed baselines on the same host.

Pinned checkpoint -> F32 / F16 GGUF (converter) -> Q8_0 (crispembed-quantize from F32) ->
CLI build -> for each arm (precision x threads x opt-in gate) run the 5 reference cases with
CRISPEMBED_MINICPM_OCR_TIMING=1 and record per-stage wall time (preprocess, vision encoder and
projector per slice, embedding, prefill, every decode step), generated IDs vs reference (token-ID
exactness) and CER. Then: Transformers 5.21.0 F32 generate() at 4/2 threads (TTFT + per-token via
a logits-processor clock) and llama.cpp (MIT, pinned commit, built here, not vendored) with
OpenBMB's official GGUF (openbmb/MiniCPM-OCR-1.0-gguf, pinned) as external baselines.
Arms run in priority order under a wall-clock budget; skipped arms are recorded.
Small JSON/Markdown output only (/kaggle/working/bench/). Replace @COMMIT@ before pushing.
"""
import glob, hashlib, json, os, re, subprocess, sys, time
COMMIT = '@COMMIT@'
MAX_NEW = 1024
T0 = time.monotonic()
BUDGET_S = float(os.environ.get('MINICPM_BENCH_BUDGET_S', str(11.2 * 3600)))
LLAMA_CPP = ('https://github.com/ggml-org/llama.cpp.git', 'a71c00f0bb2d60a4a52837d80baa534b651ebd91')  # tag b11951
OUT = '/kaggle/working/bench'
os.makedirs(OUT, exist_ok=True)


def sh(*a, **k):
    print('+', *a, flush=True)
    subprocess.run(list(a), check=True, **k)


def elapsed():
    return time.monotonic() - T0


sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT); sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')
sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv'); PY = '/kaggle/temp/venv/bin/python'
sh(PY, '-m', 'pip', 'install', '-q', 'torch==2.13.0', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu')
sh(PY, '-m', 'pip', 'install', '-q', 'transformers==5.21.0', 'pillow==11.3.0', 'numpy==2.2.6', 'gguf==0.17.1',
   'safetensors==0.6.2')

ref_json = glob.glob('/kaggle/input/**/minicpm-ocr/reference.json', recursive=True)[0]
ref = json.load(open(ref_json)); base = os.path.dirname(ref_json)
assert ref.get('completed'), 'reference capture incomplete'
ckpt = os.path.join(os.path.dirname(base), 'checkpoint')
src = ckpt if os.path.exists(os.path.join(ckpt, 'model.safetensors')) else '/kaggle/temp/minicpm-src'
env = dict(os.environ, OMP_NUM_THREADS='1', TMPDIR='/kaggle/temp')
sh('cmake', '-S', '.', '-B', '/kaggle/temp/b', '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF')
sh('cmake', '--build', '/kaggle/temp/b', '--target', 'crispembed-cli', 'crispembed-quantize', '--parallel', '4')
QUANT = next(p for p in glob.glob('/kaggle/temp/b/**/crispembed-quantize', recursive=True) if os.access(p, os.X_OK))
for dt in ('f32', 'f16'):
    extra = [] if src == ckpt or dt == 'f16' else ['--download']
    sh(PY, 'models/convert-minicpm-ocr-to-gguf.py', src, f'/kaggle/temp/minicpm-{dt}.gguf', '--dtype', dt, *extra, env=env)
q8 = subprocess.run([QUANT, '/kaggle/temp/minicpm-f32.gguf', '/kaggle/temp/minicpm-q8_0.gguf', 'q8_0'],
                    capture_output=True, text=True)
print(q8.stdout[-3000:], q8.stderr[-3000:], flush=True)
GGUF = {dt: f'/kaggle/temp/minicpm-{dt}.gguf' for dt in ('f32', 'f16', 'q8_0')}
SIZES = {dt: os.path.getsize(p) for dt, p in GGUF.items() if os.path.exists(p)}
SETUP_S = round(elapsed(), 1)


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def cer(hyp, gold):
    return round(levenshtein(hyp, gold) / max(1, len(gold)), 6)


def first_difference(a, b):
    if a is None or b is None:
        return None
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


TIME_RE = re.compile(r'^\s*\[time\] minicpm_ocr\.(\w+) (.*)$')


def parse_timing(stderr):
    t = dict(vision=[])
    for line in stderr.splitlines():
        m = TIME_RE.match(line)
        if not m:
            continue
        name, rest = m.groups()
        if name == 'vision':
            kv = dict(re.findall(r'(\w+)=([\d.]+)(?:ms)?', rest))
            size = re.search(r' (\d+)x(\d+) ', ' ' + rest + ' ')
            t['vision'].append(dict(part=int(kv['part']), size=[int(size.group(1)), int(size.group(2))] if size else None,
                                    patches=int(kv['patches']), tokens=int(kv['tokens']),
                                    **{k + '_ms': float(kv[k]) for k in ('total', 'setup', 'encoder', 'projector', 'readback')}))
        elif name == 'decode':
            kv = dict(re.findall(r'(\w+)=([\d.]+)ms', rest))
            steps = re.search(r'steps=(\d+)', rest)
            lst = re.search(r'list=([\d.,]*)', rest)
            t['decode'] = dict(steps=int(steps.group(1)), **{k + '_ms': float(v) for k, v in kv.items()},
                               step_ms=[float(x) for x in lst.group(1).split(',') if x] if lst else [])
        elif name == 'ids':
            lst = re.search(r'list=([\d,]*)', rest)
            t['ids'] = [int(x) for x in lst.group(1).split(',') if x] if lst else []
        else:
            ms = re.match(r'([\d.]+)ms', rest)
            t[name + '_ms'] = float(ms.group(1)) if ms else None
            extra = dict(re.findall(r'(\w+)=(\S+)', rest))
            if extra:
                t[name + '_info'] = extra
    if t['vision']:
        t['vision_encoder_ms'] = round(sum(v['encoder_ms'] for v in t['vision']), 3)
        t['vision_projector_ms'] = round(sum(v['projector_ms'] for v in t['vision']), 3)
        t['vision_total_ms'] = round(sum(v['total_ms'] for v in t['vision']), 3)
    d = t.get('decode')
    if d and d['step_ms']:
        s = d['step_ms']
        d['first_step_ms'] = s[0]
        d['steady_mean_ms'] = round(sum(s[1:]) / max(1, len(s) - 1), 3)
        d['step_ms_sample'] = s[:8] + (['...'] + s[-4:] if len(s) > 12 else s[8:])
        del d['step_ms']
    return t


# Arms in priority order: (name, dtype, threads, extra env, timing)
BASE_ENV = dict(CRISPEMBED_MINICPM_OCR='1', CRISPEMBED_QWEN35_OCR='1', CRISPEMBED_QWEN35_WEIGHT_CACHE='1')
ARMS = [
    ('f16-t4', 'f16', 4, {}, True),
    ('q8_0-t4', 'q8_0', 4, dict(CRISPEMBED_QWEN35_QUANT='1'), True),
    ('transformers-f32-t4', None, 4, None, None),
    ('llama.cpp-f16-t4', None, 4, None, None),
    ('f16-t2', 'f16', 2, {}, True),
    ('transformers-f32-t2', None, 2, None, None),
    ('llama.cpp-f16-t2', None, 2, None, None),
    ('f16-t4-vision-flash', 'f16', 4, dict(CRISPEMBED_MINICPM_OCR_VISION_FLASH='1'), True),
    ('f16-t4-parallel-f16', 'f16', 4, dict(CRISPEMBED_QWEN35_PARALLEL_F16='1'), True),
    ('f16-t4-cache-1g', 'f16', 4, dict(CRISPEMBED_QWEN35_WEIGHT_CACHE_BYTES='1073741824'), True),
    ('f16-t4-no-cache', 'f16', 4, dict(CRISPEMBED_QWEN35_WEIGHT_CACHE='0'), True),
    ('f32-t4', 'f32', 4, {}, True),
    ('q8_0-t2', 'q8_0', 2, dict(CRISPEMBED_QWEN35_QUANT='1'), True),
    ('f16-t4-no-timing', 'f16', 4, {}, False),
]
CASES = ref['cases']
report = dict(schema='crispembed-minicpm-ocr-bench-v1', commit=COMMIT, reference_revision=ref['revision'],
              reference_versions=ref.get('versions'), max_new_tokens=MAX_NEW, cpu_count=os.cpu_count(),
              cpu_model=next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), None),
              build=dict(cmake=['-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF']),
              gguf_bytes=SIZES, quantize=dict(exit=q8.returncode, stderr_tail=q8.stderr[-1500:]),
              setup_seconds=SETUP_S, budget_seconds=BUDGET_S, arms=[], skipped=[], baselines={})


def save(final=False):
    report['elapsed_seconds'] = round(elapsed(), 1)
    if final:
        report['completed'] = True
    json.dump(report, open(f'{OUT}/bench.json', 'w'), indent=1, ensure_ascii=False)


def run_native(name, dtype, threads, extra, timing):
    arm = dict(name=name, dtype=dtype, threads=threads, env=extra, timing=timing, cases=[])
    report['arms'].append(arm)
    for case in CASES:
        run_env = dict(os.environ, **BASE_ENV, **extra, OMP_NUM_THREADS=str(threads))
        if timing:
            run_env['CRISPEMBED_MINICPM_OCR_TIMING'] = '1'
        t0 = time.monotonic()
        r = subprocess.run(['/kaggle/temp/b/crispembed', '-m', GGUF[dtype], '--ocr', os.path.join(base, case['id'] + '.png'),
                            '--ocr-prompt', case['task'], '--ocr-max-tokens', str(MAX_NEW), '-t', str(threads)],
                           env=run_env, capture_output=True, text=True)
        wall = round(time.monotonic() - t0, 3)
        text = r.stdout.rstrip('\n')
        t = parse_timing(r.stderr) if timing else {}
        ids = t.pop('ids', None)
        row = dict(id=case['id'], exit=r.returncode, wall_seconds=wall, n_ref_tokens=len(case['ids']),
                   ids_exact=(ids == case['ids']) if ids is not None else None,
                   first_id_difference=first_difference(ids, case['ids']), n_tokens=None if ids is None else len(ids),
                   text_exact=text == case['text'], cer=cer(text, case['text']), timing=t)
        if r.returncode or (ids is not None and ids != case['ids']):
            row['text'] = text[:2000]
            row['stderr_tail'] = r.stderr[-2500:]
        arm['cases'].append(row)
        print(name, case['id'], dict(wall=wall, exact=row['ids_exact'], text=row['text_exact'], cer=row['cer'],
                                     prefill=t.get('prefill_ms'), vision=t.get('vision_total_ms'),
                                     decode=(t.get('decode') or {}).get('mean_ms')), flush=True)
        save()
    arm['all_ids_exact'] = all(c['ids_exact'] is True for c in arm['cases']) if timing else None
    arm['all_text_exact'] = all(c['text_exact'] for c in arm['cases'])
    arm['total_wall_seconds'] = round(sum(c['wall_seconds'] for c in arm['cases']), 3)
    save()


TF_SCRIPT = r'''
import json, os, sys, time
import torch, transformers
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessor, LogitsProcessorList
ckpt, base, threads, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ref = json.load(open(os.path.join(base, 'reference.json')))
torch.set_num_threads(threads)
t0 = time.monotonic()
proc = AutoProcessor.from_pretrained(ckpt)
model = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=torch.float32).eval()
load_s = time.monotonic() - t0
class Clock(LogitsProcessor):
    def __init__(self): self.marks = []
    def __call__(self, input_ids, scores):
        self.marks.append(time.monotonic()); return scores
rows = []
for case in ref['cases']:
    image = Image.open(os.path.join(base, case['id'] + '.png')).convert('RGB')
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': case['task']}]}]
    t0 = time.monotonic()
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[prompt], images=[[image]], return_tensors='pt')
    t1 = time.monotonic()
    clock = Clock()
    with torch.inference_mode():
        gen = model.generate(**inputs, do_sample=False, max_new_tokens=1024, logits_processor=LogitsProcessorList([clock]))
    t2 = time.monotonic()
    ids = gen[0, inputs['input_ids'].shape[1]:].tolist()
    steps = [b - a for a, b in zip(clock.marks, clock.marks[1:])]
    rows.append(dict(id=case['id'], preprocess_ms=round((t1 - t0) * 1e3, 3),
                     vision_plus_prefill_ms=round((clock.marks[0] - t1) * 1e3, 3) if clock.marks else None,
                     decode_total_ms=round((t2 - clock.marks[0]) * 1e3, 3) if clock.marks else None,
                     decode_steady_mean_ms=round(sum(steps) / len(steps) * 1e3, 3) if steps else None,
                     generate_seconds=round(t2 - t1, 3), wall_seconds=round(t2 - t0, 3), n_tokens=len(ids),
                     ids_exact=ids == case['ids'], text=proc.tokenizer.decode(ids, skip_special_tokens=True)))
    print(rows[-1]['id'], rows[-1]['wall_seconds'], rows[-1]['ids_exact'], flush=True)
json.dump(dict(threads=threads, load_seconds=round(load_s, 3), dtype='float32',
               versions=dict(torch=torch.__version__, transformers=transformers.__version__), cases=rows),
          open(out, 'w'), indent=1, ensure_ascii=False)
'''


def run_transformers(name, threads):
    path = '/kaggle/temp/tf_bench.py'
    open(path, 'w').write(TF_SCRIPT)
    out = f'/kaggle/temp/{name}.json'
    t0 = time.monotonic()
    r = subprocess.run([PY, path, src, base, str(threads), out], capture_output=True, text=True,
                       env=dict(os.environ, OMP_NUM_THREADS=str(threads), HF_HUB_OFFLINE='1'))
    res = json.load(open(out)) if os.path.exists(out) else dict(error=r.stderr[-3000:])
    res['exit'] = r.returncode
    res['process_seconds'] = round(time.monotonic() - t0, 3)
    for c in res.get('cases', []):
        gold = next(x for x in CASES if x['id'] == c['id'])
        c['cer'] = cer(c['text'], gold['text'])
        c['text_exact'] = c['text'] == gold['text']
        if c['text_exact']:
            del c['text']
    report['baselines'][name] = res
    save()


LLAMA = {}


def llama_setup():
    if LLAMA:
        return LLAMA.get('ok')
    try:
        from_src = json.load(open('docs/minicpm-ocr-source.json'))['upstream_gguf']
        sh('git', 'clone', '-q', '--filter=blob:none', LLAMA_CPP[0], '/kaggle/temp/llama.cpp')
        sh('git', '-C', '/kaggle/temp/llama.cpp', 'checkout', '-q', LLAMA_CPP[1])
        sh('cmake', '-S', '/kaggle/temp/llama.cpp', '-B', '/kaggle/temp/lb', '-DCMAKE_BUILD_TYPE=Release',
           '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=OFF')
        sh('cmake', '--build', '/kaggle/temp/lb', '--target', 'llama-mtmd-cli', '--parallel', '4')
        from huggingface_hub import hf_hub_download  # kernel image hub client, download only
        files = {}
        for n in ('MiniCPM-OCR-1_0-F16.gguf', 'mmproj-model-f16-4x.gguf'):
            p = hf_hub_download(from_src['repository'], n, revision=from_src['revision'], local_dir='/kaggle/temp/llgguf')
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for b in iter(lambda: f.read(1 << 23), b''):
                    h.update(b)
            assert h.hexdigest() == from_src['files'][n]['sha256'], n
            files[n] = p
        LLAMA.update(ok=True, bin=glob.glob('/kaggle/temp/lb/**/llama-mtmd-cli', recursive=True)[0], files=files,
                     revision=from_src['revision'])
    except Exception as e:  # record and keep benchmarking the native runtime
        LLAMA.update(ok=False, error=f'{type(e).__name__}: {e}')
    report['llama_cpp'] = dict(repo=LLAMA_CPP[0], commit=LLAMA_CPP[1], gguf_revision=LLAMA.get('revision'),
                               ok=LLAMA['ok'], error=LLAMA.get('error'))
    save()
    return LLAMA['ok']


PERF = {
    'load_ms': re.compile(r'load time =\s*([\d.]+) ms'),
    'prompt_eval_ms': re.compile(r'prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens'),
    'eval_ms': re.compile(r'\s eval time =\s*([\d.]+) ms /\s*(\d+) runs'),
    'image_encode_ms': re.compile(r'image slice encoded in (\d+) ms'),
    'image_decode_ms': re.compile(r'image decoded \(batch \d+/\d+\) in (\d+) ms'),
}


def run_llama(name, threads):
    if not llama_setup():
        report['baselines'][name] = dict(skipped='llama.cpp setup failed')
        save()
        return
    rows = []
    for case in CASES:
        t0 = time.monotonic()
        r = subprocess.run([LLAMA['bin'], '-m', LLAMA['files']['MiniCPM-OCR-1_0-F16.gguf'], '--mmproj',
                            LLAMA['files']['mmproj-model-f16-4x.gguf'], '--image', os.path.join(base, case['id'] + '.png'),
                            '-p', case['task'], '--jinja', '-c', '8192', '--temp', '0', '-n', str(MAX_NEW), '-t', str(threads),
                            '-ngl', '0', '--no-warmup'], capture_output=True, text=True)
        wall = round(time.monotonic() - t0, 3)
        text = r.stdout.strip()
        row = dict(id=case['id'], exit=r.returncode, wall_seconds=wall, cer=cer(text, case['text']),
                   text_exact=text == case['text'])
        for k, rx in PERF.items():
            found = rx.findall(r.stderr)
            if k in ('image_encode_ms', 'image_decode_ms'):
                row[k] = [float(x) for x in found]
            elif found:
                v = found[-1]
                row[k] = float(v[0] if isinstance(v, tuple) else v)
                if isinstance(v, tuple):
                    row[k.replace('_ms', '_tokens')] = int(v[1])
        if not row['text_exact']:
            row['text'] = text[:2000]
        if r.returncode:
            row['stderr_tail'] = r.stderr[-2500:]
        rows.append(row)
        print(name, case['id'], wall, row['cer'], flush=True)
    report['baselines'][name] = dict(threads=threads, model='MiniCPM-OCR-1_0-F16.gguf + mmproj-model-f16-4x.gguf',
                                     cases=rows)
    save()


durations = {}
for name, dtype, threads, extra, timing in ARMS:
    # Estimate from the closest finished arm with the same thread count; skip if it would overrun.
    est = next((v for k, v in durations.items() if k[1] == threads), 0) if dtype else 0
    if elapsed() + 1.1 * est > BUDGET_S:
        report['skipped'].append(dict(name=name, reason=f'budget: elapsed {elapsed():.0f}s + estimate {est:.0f}s'))
        save()
        continue
    t0 = time.monotonic()
    if dtype is None and name.startswith('transformers'):
        run_transformers(name, threads)
    elif dtype is None:
        run_llama(name, threads)
    else:
        if dtype not in SIZES:
            report['skipped'].append(dict(name=name, reason=f'{dtype} GGUF missing'))
            continue
        run_native(name, dtype, threads, extra, timing)
        durations[(name, threads)] = time.monotonic() - t0


def md():
    L = [f"# MiniCPM-OCR-1.0 bench @ {COMMIT[:10]}", '', f"CPU: {report['cpu_model']} x{report['cpu_count']}", '',
         '| arm | exact | text | CER max | wall s (sum) | vision enc ms | proj ms | prefill ms | decode ms/tok |', '|---|---|---|---|---|---|---|---|---|']
    for a in report['arms']:
        cs = a['cases']
        f = lambda k: round(sum((c['timing'].get(k) or 0) for c in cs))
        dm = [c['timing'].get('decode', {}).get('steady_mean_ms') for c in cs if c['timing'].get('decode')]
        L.append(f"| {a['name']} | {a.get('all_ids_exact')} | {a.get('all_text_exact')} | {max(c['cer'] for c in cs)} | "
                 f"{round(sum(c['wall_seconds'] for c in cs))} | {f('vision_encoder_ms')} | {f('vision_projector_ms')} | "
                 f"{f('prefill_ms')} | {round(sum(dm) / len(dm), 1) if dm else ''} |")
    L += ['', '## Per case (first arm)', '', '| case | wall s | preprocess | vision enc | proj | prefill (tok) | decode steps x mean ms |', '|---|---|---|---|---|---|---|']
    if report['arms']:
        for c in report['arms'][0]['cases']:
            t = c['timing']; d = t.get('decode', {})
            L.append(f"| {c['id']} | {c['wall_seconds']} | {t.get('preprocess_ms')} | {t.get('vision_encoder_ms')} | "
                     f"{t.get('vision_projector_ms')} | {t.get('prefill_ms')} ({(t.get('prefill_info') or {}).get('tokens')}) | "
                     f"{d.get('steps')} x {d.get('steady_mean_ms')} |")
    L += ['', '## Baselines', '']
    for k, v in report['baselines'].items():
        cs = v.get('cases', [])
        L.append(f"- {k}: " + ', '.join(f"{c['id']} {c['wall_seconds']}s CER {c['cer']}" for c in cs) + (f" ({v.get('skipped') or v.get('error', '')[:200]})" if not cs else ''))
    L += ['', f"Skipped: {report['skipped']}"]
    open(f'{OUT}/bench.md', 'w').write('\n'.join(L) + '\n')


save(final=True)
md()
print(open(f'{OUT}/bench.md').read())
