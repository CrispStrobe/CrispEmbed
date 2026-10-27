"""Kaggle CPU kernel: native Unlimited-OCR-2 (CRISPEMBED_UNLIMITED_OCR2) vs the frozen
original reference (mounted read-only from the reference kernel via kernel_sources).

Pinned checkpoint -> lossless F32 GGUF -> CLI build -> greedy decode on the untiled
page and the tiled 1024x768 pages (CRISPEMBED_UNLIMITED_OCR2_TILES=1) -> exact token-ID and text comparison.
Each case runs the exact reference decoder and the opt-in fast ggml decoder
(CRISPEMBED_UNLIMITED_OCR2_FAST=1); wall time and IDs are recorded for both. Small JSON report only.
"""
import glob, json, os, re, subprocess, sys, time
COMMIT = '@COMMIT@'
def sh(*a, **k): print('+', *a, flush=True); subprocess.run(list(a), check=True, **k)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT); sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')
sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv'); PY = '/kaggle/temp/venv/bin/python'
sh(PY, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'gguf==0.17.1', 'tokenizers==0.21.4', 'huggingface-hub==0.36.0', 'safetensors==0.6.2')
contract = json.load(open('docs/unlimited-ocr2-source.json'))
src = '/kaggle/temp/uocr2-src'
sh(PY, '-c', f"""
import json,hashlib
from huggingface_hub import hf_hub_download
c=json.load(open('docs/unlimited-ocr2-source.json'))
for n,p in c['files'].items():
    f=hf_hub_download(c['repository'],n,revision=c['revision'],local_dir='{src}')
    if 'sha256' in p:
        h=hashlib.sha256(open(f,'rb').read()).hexdigest(); assert h==p['sha256'],n
""")
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TMPDIR='/kaggle/temp')
sh(PY, 'models/convert-unlimited-ocr-to-gguf.py', '--model-dir', src, '--output', '/kaggle/temp/uocr2-f32.gguf', env=env)
sh('cmake', '-S', '.', '-B', '/kaggle/temp/b', '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF')
sh('cmake', '--build', '/kaggle/temp/b', '--target', 'crispembed-cli', '--parallel', '4')
ref_json = glob.glob('/kaggle/input/**/uocr2/reference.json', recursive=True)[0]
ref = json.load(open(ref_json)); base = os.path.dirname(ref_json)
# v2 schema: every case runs the exact reference decoder (default) and the opt-in fast ggml
# decoder (CRISPEMBED_UNLIMITED_OCR2_FAST=1). Top-level per-case fields stay the reference
# arm (v1-compatible); 'arms' records IDs and wall time for both. Acceptance for the fast
# arm = identical generated token IDs (small float drift only while IDs stay identical).
report = dict(schema='crispembed-unlimited-ocr2-native-v2', commit=COMMIT, reference_revision=ref['revision'], cases=[])
subprocess.run(['bash', '-c', 'command -v gdb || (apt-get -qq update && apt-get -qq install -y gdb) >/dev/null 2>&1'])


def first_difference(a, b):
    if a is None or b is None:
        return None
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def run_arm(img, tiled, fast):
    run_env = dict(os.environ, CRISPEMBED_UNLIMITED_OCR2='1', UOCR_DBG='1', OMP_NUM_THREADS=str(os.cpu_count()))
    if tiled:
        run_env['CRISPEMBED_UNLIMITED_OCR2_TILES'] = '1'
    run_env['CRISPEMBED_UNLIMITED_OCR2_FAST'] = '1' if fast else '0'
    run_env['ASAN_OPTIONS'] = 'detect_leaks=0:abort_on_error=0:halt_on_error=1'
    t0 = time.monotonic()
    r = subprocess.run(['/kaggle/temp/b/crispembed', '-m', '/kaggle/temp/uocr2-f32.gguf', '--ocr', img,
                        '--ocr-max-tokens', '384', '-t', str(os.cpu_count())], env=run_env, capture_output=True, text=True)
    wall = time.monotonic() - t0
    m = re.search(r'gen_ids \(\d+\):([^\n]*)', r.stderr)
    ids = [int(x) for x in m.group(1).split()] if m else None
    return dict(fast=fast, exit=r.returncode, wall_seconds=round(wall, 3), ids=ids, text=r.stdout.strip(),
                stderr_tail=r.stderr[-6000:])


for case in ref['cases']:
    if case['id'] not in ('small-untiled', 'para-en', 'table'):
        continue
    tiled = case['id'] != 'small-untiled'  # 1024x768 pages: native gundam tiling (gated)
    img = os.path.join(base, case['id'] + '.png')
    ref_ids = next((case[k] for k in ('ids', 'token_ids', 'output_ids') if k in case), None)
    exact, fast = run_arm(img, tiled, False), run_arm(img, tiled, True)
    for a in (exact, fast):
        a['first_id_difference'] = first_difference(a['ids'], ref_ids)
        a['ids_exact'] = a['ids'] is not None and a['ids'] == ref_ids
    report['cases'].append(dict(id=case['id'], tiled=tiled, exit=exact['exit'], native_text=exact['text'], reference_text=case.get('text'),
                                native_ids=exact['ids'], reference_ids=ref_ids, first_id_difference=exact['first_id_difference'],
                                ids_exact=exact['ids_exact'], stderr_tail=exact['stderr_tail'],
                                arms=dict(exact=exact, fast=fast),
                                fast_ids_equal_exact=fast['ids'] is not None and fast['ids'] == exact['ids'],
                                fast_first_difference_vs_exact=first_difference(fast['ids'], exact['ids']),
                                speedup=round(exact['wall_seconds'] / fast['wall_seconds'], 3) if fast['wall_seconds'] else None))
os.makedirs('/kaggle/working/native', exist_ok=True)
json.dump(report, open('/kaggle/working/native/native.json', 'w'), indent=2, ensure_ascii=False)
print(json.dumps([dict({k: c[k] for k in ('id', 'exit', 'ids_exact', 'first_id_difference', 'fast_ids_equal_exact', 'speedup')},
                       exact_s=c['arms']['exact']['wall_seconds'], fast_s=c['arms']['fast']['wall_seconds'],
                       fast_exit=c['arms']['fast']['exit'], fast_ids_exact=c['arms']['fast']['ids_exact'])
                  for c in report['cases']], indent=1))
