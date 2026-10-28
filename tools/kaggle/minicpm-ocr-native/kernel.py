"""Kaggle CPU kernel: native MiniCPM-OCR-1.0 (CRISPEMBED_MINICPM_OCR) vs the frozen
original reference (crispembed-minicpm-ocr-reference output, mounted via kernel_sources).

Pinned checkpoint -> F32 and F16 GGUF -> CLI build -> for each of the 5 reference cases
and each dtype: native greedy decode with CRISPEMBED_MINICPM_OCR_DUMP, then stage-by-stage
comparison at the earliest divergence: input ids, pixel values, vision embeddings /
layers 0,6,26 / post_layernorm, merger, decoder layers, final norm, first logits,
generated IDs and text. Small JSON report only (/kaggle/working/native/native.json).
Replace @COMMIT@ with the pushed commit before pushing (tools/kaggle/kpush.py).
"""
import glob, json, os, subprocess, sys, time
COMMIT = '@COMMIT@'
MAX_NEW = int(os.environ.get('MINICPM_MAX_NEW_TOKENS', '1024'))


def sh(*a, **k):
    print('+', *a, flush=True)
    subprocess.run(list(a), check=True, **k)


sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT); sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')
sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv'); PY = '/kaggle/temp/venv/bin/python'
sh(PY, '-m', 'pip', 'install', '-q', 'torch==2.13.0', '--index-url', 'https://download.pytorch.org/whl/cpu')
sh(PY, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'gguf==0.17.1', 'huggingface-hub==0.36.0', 'safetensors==0.6.2')

ref_json = glob.glob('/kaggle/input/**/minicpm-ocr/reference.json', recursive=True)[0]
ref = json.load(open(ref_json)); base = os.path.dirname(ref_json)
assert ref.get('completed'), 'reference capture incomplete'
# The reference kernel leaves its verified checkpoint next to its output; the converter
# re-verifies every pinned size/sha256, so reuse it and only download if absent.
ckpt = os.path.join(os.path.dirname(base), 'checkpoint')
src = ckpt if os.path.exists(os.path.join(ckpt, 'model.safetensors')) else '/kaggle/temp/minicpm-src'
env = dict(os.environ, OMP_NUM_THREADS='1', TMPDIR='/kaggle/temp')
conv = ['models/convert-minicpm-ocr-to-gguf.py', src]
for dt in ('f32', 'f16'):
    extra = [] if src == ckpt or dt == 'f16' else ['--download']
    sh(PY, *conv, f'/kaggle/temp/minicpm-{dt}.gguf', '--dtype', dt, *extra, env=env)
sh('cmake', '-S', '.', '-B', '/kaggle/temp/b', '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF')
sh('cmake', '--build', '/kaggle/temp/b', '--target', 'crispembed-cli', '--parallel', '4')

import numpy as np  # noqa: E402  (kernel image numpy; arrays only)


def first_difference(a, b):
    if a is None or b is None:
        return None
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def metrics(r, n):
    r = np.asarray(r, np.float64).ravel(); n = np.asarray(n, np.float64).ravel()
    if r.size != n.size:
        return dict(match=False, ref_size=int(r.size), native_size=int(n.size))
    d = np.abs(r - n); den = np.linalg.norm(r) * np.linalg.norm(n)
    return dict(match=True, max_abs=float(d.max()) if d.size else 0.0, mean_abs=float(d.mean()) if d.size else 0.0,
                ref_max_abs=float(np.abs(r).max()) if r.size else 0.0, cosine=float(r @ n / den) if den else None,
                argmax_abs=int(d.argmax()) if d.size else None)


def align_rows(refa, native, lengths):
    """Reference may be [sumL,C], [1,sumL,C] or padded [N,Lmax,C]; native is [sumL,C]."""
    if refa.size == native.size:
        return refa.reshape(native.shape), 'flat'
    if refa.ndim == 3 and refa.shape[0] == len(lengths) and refa.shape[1] >= max(lengths):
        return np.concatenate([refa[i, :l] for i, l in enumerate(lengths)]), 'padded'
    return None, f'unaligned ref{list(refa.shape)} native{list(native.shape)}'


def patchify(chw, p=14):
    """MiniCPM-V reshape_by_patch: [3,H,W] -> [3, p, L*p] (patches row-major)."""
    c, h, w = chw.shape
    x = chw.reshape(c, h // p, p, w // p, p).transpose(0, 2, 1, 3, 4)  # c, ky, py, px, kx
    return x.reshape(c, p, -1)


def compare_pixels(refa, parts):
    natives = [np.load(p) for p in parts]
    cands = {
        'chw_list': natives,
        'by_patch': [patchify(x) for x in natives],
    }
    out = {}
    for name, arrs in cands.items():
        flat = np.concatenate([a.ravel() for a in arrs])
        if refa.size == flat.size:
            out[name] = metrics(refa, flat)
        elif refa.ndim == 4 and refa.shape[0] == len(arrs):  # padded [N,3,p,Lmax*p] or [N,3,H,W]
            try:
                out[name] = metrics(np.concatenate([refa[i][tuple(slice(0, s) for s in a.shape)].ravel()
                                                    for i, a in enumerate(arrs)]), flat)
            except Exception as e:  # shapes differ per layout; record and continue
                out[name] = dict(match=False, error=str(e))
    return out or dict(match=False, ref_shape=list(refa.shape), native_shapes=[list(a.shape) for a in natives])


def run_case(case, dtype):
    dump = f'/kaggle/temp/dump/{dtype}/{case["id"]}'; os.makedirs(dump, exist_ok=True)
    run_env = dict(os.environ, CRISPEMBED_MINICPM_OCR='1', CRISPEMBED_QWEN35_OCR='1', CRISPEMBED_QWEN35_WEIGHT_CACHE='1',
                   CRISPEMBED_MINICPM_OCR_DUMP=dump, OMP_NUM_THREADS=str(os.cpu_count()))
    t0 = time.monotonic()
    r = subprocess.run(['/kaggle/temp/b/crispembed', '-m', f'/kaggle/temp/minicpm-{dtype}.gguf',
                        '--ocr', os.path.join(base, case['id'] + '.png'), '--ocr-prompt', case['task'],
                        '--ocr-max-tokens', str(MAX_NEW), '-t', str(os.cpu_count())],
                       env=run_env, capture_output=True, text=True)
    row = dict(dtype=dtype, exit=r.returncode, wall_seconds=round(time.monotonic() - t0, 3), text=r.stdout.rstrip('\n'),
               stderr_tail=r.stderr[-4000:], stages={})
    load = lambda n: np.load(os.path.join(dump, n + '.npy')) if os.path.exists(os.path.join(dump, n + '.npy')) else None
    meta = json.load(open(os.path.join(dump, 'native_meta.json'))) if os.path.exists(os.path.join(dump, 'native_meta.json')) else None
    row['native_meta'] = meta
    arrays = case.get('arrays', {})
    refload = lambda n: np.load(os.path.join(base, arrays[n]['path'])) if n in arrays else None
    ids_in = load('input_ids')
    row['input_ids_exact'] = ids_in is not None and ids_in.tolist() == case['input_ids']
    row['input_ids_first_difference'] = first_difference(None if ids_in is None else ids_in.tolist(), case['input_ids'])
    row['input_ids_len'] = [None if ids_in is None else int(ids_in.size), len(case['input_ids'])]
    if meta:
        pix = refload('inputs.pixel_values')
        if pix is not None:
            row['stages']['pixel_values'] = compare_pixels(pix, [os.path.join(dump, f'native_pixels_{i}.npy') for i in range(len(meta['parts']))])
        pl = [p['w'] // 14 * (p['h'] // 14) for p in meta['parts']]
        tl = [p['tokens'] for p in meta['parts']]
        for ref_name, nat_name, lengths in [
                ('model.vision_tower.embeddings', 'model_vision_tower_embeddings', pl),
                ('model.vision_tower.encoder.layers.0', 'model_vision_tower_encoder_layers_0', pl),
                ('model.vision_tower.encoder.layers.6', 'model_vision_tower_encoder_layers_6', pl),
                ('model.vision_tower.encoder.layers.26', 'model_vision_tower_encoder_layers_26', pl),
                ('model.vision_tower.post_layernorm', 'model_vision_tower_post_layernorm', pl),
                ('model.merger', 'model_merger', tl)]:
            refa, nat = refload(ref_name), load(nat_name)
            if refa is None or nat is None:
                row['stages'][ref_name] = dict(missing=[refa is None, nat is None]); continue
            aligned, how = align_rows(refa, nat, lengths)
            row['stages'][ref_name] = dict(layout=how, **(metrics(aligned, nat) if aligned is not None else {}))
    if row['input_ids_exact']:
        for i in range(24):
            refa, nat = refload(f'model.language_model.layers.{i}'), load(f'model_language_model_layers_{i}')
            if refa is not None and nat is not None:
                row['stages'][f'decoder.layer.{i}'] = metrics(refa, nat)
        for ref_name, nat_name in [('model.language_model.norm', 'model_language_model_norm'), ('lm_head', 'lm_head')]:
            refa, nat = refload(ref_name), load(nat_name)
            if refa is not None and nat is not None:
                row['stages'][ref_name] = metrics(refa, nat)
                if ref_name == 'lm_head':
                    row['first_logit_argmax'] = [int(np.argmax(refa)), int(np.argmax(nat))]
    gen = load('ids')
    row['ids'] = None if gen is None else gen.tolist()
    row['ids_exact'] = row['ids'] == case['ids']
    row['first_id_difference'] = first_difference(row['ids'], case['ids'])
    row['text_exact'] = row['text'] == case['text']
    row['eos'] = bool(row['ids']) and row['ids'][-1] in (248044, 248046)
    stage_order = ['pixel_values', 'model.vision_tower.embeddings', 'model.vision_tower.encoder.layers.0',
                   'model.vision_tower.encoder.layers.6', 'model.vision_tower.encoder.layers.26',
                   'model.vision_tower.post_layernorm', 'model.merger'] + [f'decoder.layer.{i}' for i in range(24)]
    worst = [(k, v.get('max_abs')) for k, v in row['stages'].items() if isinstance(v, dict) and v.get('max_abs') is not None]
    row['worst_stage'] = max(worst, key=lambda kv: kv[1]) if worst else None
    row['stage_order'] = [k for k in stage_order if k in row['stages']]
    return row


report = dict(schema='crispembed-minicpm-ocr-native-v1', commit=COMMIT, reference_revision=ref['revision'],
              reference_versions=ref.get('versions'), max_new_tokens=MAX_NEW, cases=[])
for case in ref['cases']:
    arms = {dt: run_case(case, dt) for dt in ('f32', 'f16')}
    report['cases'].append(dict(id=case['id'], task=case['task'], size=case['size'], reference_ids=case['ids'],
                                reference_text=case['text'], reference_eos=case.get('eos'),
                                reference_cap_hit=case.get('cap_hit'), arms=arms))
    os.makedirs('/kaggle/working/native', exist_ok=True)
    json.dump(report, open('/kaggle/working/native/native.json', 'w'), indent=1, ensure_ascii=False)
report['completed'] = True
json.dump(report, open('/kaggle/working/native/native.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps([dict(id=c['id'], **{dt: dict(exit=a['exit'], input_ids=a['input_ids_exact'], ids=a['ids_exact'],
                                             first_diff=a['first_id_difference'], text=a['text_exact'],
                                             worst=a['worst_stage'], s=a['wall_seconds'])
                                      for dt, a in c['arms'].items()}) for c in report['cases']], indent=1))
