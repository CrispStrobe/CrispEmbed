"""Kaggle CPU kernel: localize the earliest native Unlimited-OCR-2 vision divergence.

Mounts the frozen reference kernel output (kernel_sources) read-only, builds the CLI at
an exact commit, runs each case with UOCR2_DUMP_VISION / UOCR_DUMP_PIXELS and a capped
decode, and runs tools/kaggle/unlimited-ocr2-vision-diff/diff.py (pixels + per-tile
projector features vs the reference). Tiled cases run two arms: reference-exact prep
(default inside the tiles gate) and CRISPEMBED_UNLIMITED_OCR2_TILES_LEGACY_PREP=1.
Small JSON report in /kaggle/working/vision.
"""
import glob, json, os, re, subprocess, sys
COMMIT = '@COMMIT@'
MAX_NEW = '4'
def sh(*a, **k): print('+', *a, flush=True); subprocess.run(list(a), check=True, **k)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT); sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')
sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv'); PY = '/kaggle/temp/venv/bin/python'
sh(PY, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'gguf==0.17.1', 'tokenizers==0.21.4', 'huggingface-hub==0.36.0',
   'safetensors==0.6.2', 'pillow==11.3.0')  # Pillow pinned like the reference kernel
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
DIFF = os.path.abspath('tools/kaggle/unlimited-ocr2-vision-diff/diff.py')
report = dict(schema='crispembed-unlimited-ocr2-vision-diff-v1', commit=COMMIT, reference_revision=ref['revision'], cases=[])
os.makedirs('/kaggle/working/vision', exist_ok=True)
for case in ref['cases']:
    if case['id'] not in ('small-untiled', 'para-en', 'table'):
        continue
    tiled = case['id'] != 'small-untiled'
    img = os.path.join(base, case['id'] + '.png')
    projs = sorted(p for p in glob.glob(os.path.join(base, 'arrays', case['id'], '*.npy')) if p.endswith('projector.npy'))
    arms = ([('ref-prep', {}), ('legacy-prep', {'CRISPEMBED_UNLIMITED_OCR2_TILES_LEGACY_PREP': '1'})] if tiled
            else [('untiled', {})])
    for arm, extra in arms:
        tag = f"{case['id']}-{arm}"
        vis, gpx = f'/kaggle/temp/{tag}.vis.f32', f'/kaggle/temp/{tag}.global.f32'
        run_env = dict(os.environ, CRISPEMBED_UNLIMITED_OCR2='1', OMP_NUM_THREADS=str(os.cpu_count()),
                       UOCR2_DUMP_VISION=vis, UOCR_DUMP_PIXELS=gpx, UOCR_MAX_NEW=MAX_NEW, **extra)
        if tiled:
            run_env['CRISPEMBED_UNLIMITED_OCR2_TILES'] = '1'
        r = subprocess.run(['/kaggle/temp/b/crispembed', '-m', '/kaggle/temp/uocr2-f32.gguf', '--ocr', img,
                            '--ocr-max-tokens', MAX_NEW, '-t', str(os.cpu_count())], env=run_env, capture_output=True, text=True)
        row = dict(id=case['id'], arm=arm, exit=r.returncode, reference_projector=[os.path.relpath(p, base) for p in projs],
                   dump_line=(re.findall(r'unlimited_ocr: vision dump[^\n]*', r.stderr) or [None])[-1])
        if r.returncode == 0 and projs and os.path.exists(vis):
            out = f'/kaggle/working/vision/{tag}.json'
            d = subprocess.run([PY, '-I', DIFF, img, vis, gpx, projs[0], out], capture_output=True, text=True)
            row['diff_exit'] = d.returncode
            if d.returncode == 0:
                row['diff'] = json.load(open(out))
            else:
                row['diff_stderr'] = d.stderr[-3000:]
        else:
            row['stderr_tail'] = r.stderr[-4000:]
        report['cases'].append(row)
        json.dump(report, open('/kaggle/working/vision/vision-diff.json', 'w'), indent=2, ensure_ascii=False)
summary = []
for c in report['cases']:
    d = c.get('diff', {})
    summary.append(dict(id=c['id'], arm=c['arm'], exit=c['exit'], max_abs=d.get('max_abs'), min_cosine=d.get('min_cosine'),
                        first_block_over_1e3=d.get('first_block_over_1e-3'),
                        global_px_diff=(d.get('global_pixels') or {}).get('n_diff'),
                        tile_px_first_diff=(d.get('tile_pixels') or {}).get('first_differing_tile')))
print(json.dumps(summary, indent=1))
