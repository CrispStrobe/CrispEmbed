#!/usr/bin/env python3
"""Private hosted two-crop Surya 2.5 HF F32 vs native F16 diagnostic.

Kernel prerequisites (not installed or built here): pinned CPU Python stack from
surya-ocr-25-quant/kernel.py, built test-surya25-run, this staged source tree at
the pinned base plus the direct-F16 changes, and SURYA_UTIL_PATH pointing to the
pinned util.py. Reports live in /kaggle/working; large model artifacts and
caches remain in /kaggle/temp. Model download is anonymous.

Outputs and logs are private; model licence may impose output sharing terms.
Two candidate line crops are not full pages; there is no reviewed crop gold,
CER/WER, release, speed qualification, or publication in this experiment.
"""
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKING = Path('/kaggle/working')
BASE = '9498e67e3558a9a0b0ec20f93c40698b2434204e'
MODEL = 'datalab-to/surya-ocr-2.5'
REVISION = '00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3'
CHECKPOINT_SHA = '6940baca2d16a20b15b7302cac7253246c78c95469920fcdb064963f48b5cb17'
CHECKPOINT_BYTES = 1378772208
F16_SHA = '5f968d8321e8656e17a5b648725911ba83feaa948addd5eaa3768551e46c5358'
UTIL_SHA = 'd853246db9767d01c8648058b779047ad0d93b3c74c0f7783dcf9faeef33b11b'
UTIL_COMMIT = '9d5aadfebb07939755c24e8a80d0e61b3be80895'
PROMPT = 'OCR this block image to HTML.'
CAP, THREADS, ROUNDS, EOS = 256, 4, 2, 2
PACKAGE_VERSIONS = {'numpy': '2.2.6', 'gguf': '0.19.0', 'safetensors': '0.8.0',
                    'huggingface-hub': '1.33.0', 'pillow': '12.1.0', 'tokenizers': '0.23.3',
                    'transformers': '5.16.1', 'torch': '2.14.0', 'torchvision': '0.29.0'}
SOURCE_PINS = {
    'models/qwen35_surya25_profile.py': '08cef7838c5d8af49cf8d50bc7e1e3104e5c15eceead818fec28e2c49065d972',
    'models/convert-qwen35-ocr-to-gguf.py': 'b85629c77e37296d6355514fe7e8badc4b16f205395157ce03793710fe67d3f3',
    'tests/surya25_run.cpp': '135b8ce4475e5b5985cddf7711e6c4cd91594b7c57b482e95533d6e137015c10',
    'src/qwen35_linear.cpp': '38e342810caff1aca42dfdae1bff67ed4fddc60049fcfbd68d7f01f076275af3',
}
CASES = {
    'german': {'path': 'tests/regression/images/cc0/german_official_print.jpg',
               'source_sha256': 'cc3392810b29cf5a2d0b1831cff95d07a2cbf9e00a3e91e785779a96c0a12a2e',
               'box': [560, 1280, 1060, 1342],
               'rgb_sha256': '50d327a0e38ca4e10dff2f6c65c2e19ef9f2eb4533f6d8a429a2389ca614ecd3',
               'png_sha256': 'a0d46ab22e294c160bcbea2a4ca4d5932fdb49222c8c93a9d4effd2ca2b926dc'},
    'receipt': {'path': 'tests/regression/images/cc0/receipt_historical.png',
                'source_sha256': 'ee74bdb8c9c4d5f9e8d66b128041c03bcaa3acf8194448a8e0157f9478ad3d97',
                'box': [130, 250, 620, 320],
                'rgb_sha256': '4867e90a9efbccbce998ce848ef9d0c3952c2d6b37a41e7bca304abe66a21e36',
                'png_sha256': '3449c9d417ce53045273d6f5a3a83c85d538d62e444d298502a2183c58b6e204'},
}
REPORT = {'version': 'surya25-real-scan-v1', 'phase': 'startup', 'status': 'running',
          'model': MODEL, 'revision': REVISION, 'base': BASE, 'cases': {},
          'limits': [
              'Two scanned line crops only; no full-page, reviewed crop gold, CER/WER or release claim.',
              'HF F32 versus F16 is a precision and implementation comparison, not numerical equivalence.',
              'HF generation timing excludes package scaling, processor, loading, and score postprocessing; native recognition includes image decode/preprocessing but excludes initialization. Process wall includes initialization. Never divide these as equivalent scopes.',
              'Separate runs, no dedicated warmup, no controlled filesystem cache or CPU affinity; timing is diagnostic only.',
              'Both rounds share a loaded HF model, while native uses a fresh process each time; no HF/native timing ratio is computed.',
              'Direct flag is requested; the CLI has no per-operation dispatch evidence. Binary digest does not attest linkage to staged source.',
              'Actual HF processor grid is checked against 1024 unmerged patches; native scaled pixels and native processor grid are not exposed by this CLI.',
              'Output, pixels, logs, weights and reports are private and may be subject to model output licence terms.',
          ]}
OUT = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_png(image):
    """RGB PNG with deterministic stored DEFLATE blocks and no metadata."""
    if image.mode != 'RGB':
        raise ValueError('canonical PNG requires RGB')
    w, h = image.size
    raw = image.tobytes()
    rows = b''.join(b'\0' + raw[y*w*3:(y+1)*w*3] for y in range(h))
    blocks = [rows[i:i+65535] for i in range(0, len(rows), 65535)]
    stream = b'\x78\x01' + b''.join(
        bytes([int(i == len(blocks)-1)]) + struct.pack('<HH', len(b), len(b) ^ 65535) + b
        for i, b in enumerate(blocks)) + struct.pack('>I', zlib.adler32(rows))
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', stream) + chunk(b'IEND', b''))


def save(phase):
    REPORT['phase'] = phase
    if OUT is not None:
        path = OUT / 'report.json'
        tmp = OUT / 'report.json.tmp'
        tmp.write_text(json.dumps(REPORT, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot import ' + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    proc = subprocess.run(['git', '-C', str(ROOT), *args], text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError('git provenance failed: ' + proc.stderr[-1000:])
    return proc.stdout.strip()


def compare(ref, arm):
    a, b = ref['token_ids'], arm['token_ids']
    if not isinstance(ref['text'], str) or not isinstance(arm['text'], str):
        raise ValueError('missing decoded text')
    if ref['stop'] not in ('EOS', 'cap', 'context_limit') or arm['stop'] not in ('EOS', 'cap', 'context_limit'):
        raise ValueError('unknown stop')
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                 None if len(a) == len(b) else min(len(a), len(b)))
    n = first if first is not None else len(a)
    diffs = [arm['confidences'][i] - ref['confidences'][i] for i in range(n)]
    return {'ids_exact': a == b, 'text_exact': ref['text'] == arm['text'],
            'stop_exact': ref['stop'] == arm['stop'], 'reference_stop': ref['stop'], 'arm_stop': arm['stop'],
            'both_reached_eos': ref['stop'] == 'EOS' and arm['stop'] == 'EOS',
            'first_divergent_step': first, 'reference_token': a[first] if first is not None and first < len(a) else None,
            'arm_token': b[first] if first is not None and first < len(b) else None,
            'reference_count': len(a), 'arm_count': len(b),
            'confidence_identical_prefix': {'count': n, 'arm_minus_reference': diffs,
                                            'max_abs': max(map(abs, diffs), default=None)}}


def validate_result(data):
    ids, conf = data['token_ids'], data['confidences']
    if (not isinstance(ids, list) or not 0 < len(ids) <= CAP or not isinstance(conf, list)
            or len(conf) != len(ids) or any(type(i) is not int or not 0 <= i < 88526 for i in ids)
            or any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in conf)
            or data['stop'] not in ('EOS', 'cap', 'context_limit')
            or (data['stop'] == 'EOS') != (ids[-1] == EOS)):
        raise ValueError('invalid IDs, confidence, or stop')


def native(binary, f16, image, arm):
    env = {k: os.environ[k] for k in ('PATH', 'LD_LIBRARY_PATH', 'LANG', 'LC_ALL') if k in os.environ}
    env.update({'HOME': str(OUT), 'TMPDIR': str(OUT / 'tmp'), 'OMP_NUM_THREADS': str(THREADS),
                'OPENBLAS_NUM_THREADS': str(THREADS), 'MKL_NUM_THREADS': str(THREADS),
                'CUDA_VISIBLE_DEVICES': '', 'CRISPEMBED_QWEN35_OCR': '1',
                'CRISPEMBED_SURYA25_FRONTEND': '1', 'CRISPEMBED_SURYA25_INFERENCE': '1',
                'CRISPEMBED_SURYA25_SCALE_TO_FIT': '1', 'CRISPEMBED_QWEN35_QUANT': '0',
                'CRISPEMBED_QWEN35_DIRECT_F16': '1' if arm == 'direct' else '0',
                'CRISPEMBED_QWEN35_PROFILE': '0'})
    command = [str(binary), str(f16), str(image), str(CAP), str(THREADS), PROMPT]
    start = time.perf_counter()
    proc = subprocess.run(command, env=env, text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    lines = [x for x in proc.stdout.splitlines() if x.startswith('SURYA25_RESULT ')]
    if proc.returncode or len(lines) != 1:
        raise RuntimeError(f'native {arm} exit={proc.returncode}, stdout={proc.stdout[-4000:]}, stderr={proc.stderr[-4000:]}')
    data = json.loads(lines[0].split(' ', 1)[1])
    data['text'] = bytes.fromhex(data.pop('raw_text_utf8_hex')).decode('utf-8', 'strict')
    data['stop'] = {1: 'EOS', 2: 'cap', 3: 'context_limit'}.get(data['stop_kind'], 'unknown')
    data.update(process_wall_seconds=elapsed, requested_direct_gate=env['CRISPEMBED_QWEN35_DIRECT_F16'],
                stderr_tail=proc.stderr[-4000:])
    validate_result(data)
    for key in ('wall_seconds', 'init_seconds', 'process_wall_seconds'):
        if not math.isfinite(data[key]) or data[key] < 0:
            raise ValueError('invalid native timing')
    return data


def main():
    global OUT
    if not WORKING.is_dir():
        raise RuntimeError('requires /kaggle/working')
    OUT = Path(tempfile.mkdtemp(prefix='surya25-real-scan-', dir=WORKING))
    (OUT / 'tmp').mkdir()
    # Set before third-party imports: no inherited credentials, caches or debug gates.
    for key in list(os.environ):
        if (key.startswith(('HF_', 'HUGGINGFACE_', 'HUGGING_FACE_', 'CRISPEMBED_', 'TRANSFORMERS_', 'TORCH_', 'CUDA_'))
                or key.lower().endswith('_proxy')):
            os.environ.pop(key, None)
    model_dir = Path('/kaggle/temp/surya25-real-scan-model')
    model_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update({'HOME': str(OUT), 'TMPDIR': str(OUT / 'tmp'), 'HF_HOME': str(model_dir / 'hf-home'),
                       'XDG_CACHE_HOME': str(model_dir / 'cache'),
                       'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'TRANSFORMERS_CACHE': str(model_dir / 'hf-home'), 'CUDA_VISIBLE_DEVICES': '',
                       'OMP_NUM_THREADS': str(THREADS), 'MKL_NUM_THREADS': str(THREADS),
                       'OPENBLAS_NUM_THREADS': str(THREADS), 'TOKENIZERS_PARALLELISM': 'false',
                       'PYTHONDONTWRITEBYTECODE': '1'})
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(OUT / 'tmp')
    REPORT.update(output_dir=str(OUT), host={'platform': platform.platform(), 'cpus': os.cpu_count(),
                  'threads': THREADS, 'free_bytes': shutil.disk_usage(OUT).free},
                  timing={'rounds_per_case_arm': ROUNDS, 'warmup': 'none dedicated',
                          'native_order': [['off', 'direct'], ['direct', 'off']],
                          'hf': 'generate() only; CPU F32, excludes preprocessing/load/softmax',
                          'native': 'fresh process per call; wall_seconds is recognize_file including image decode/preprocess; process_wall_seconds includes process/init/recognition'})
    save('admission')
    if REPORT['host']['free_bytes'] < 8 * 1024**3:
        raise RuntimeError('less than 8 GiB free under /kaggle/working')
    head = git('rev-parse', 'HEAD')
    REPORT['source'] = {'head': head, 'status_porcelain': git('status', '--porcelain'),
                        'submodules': git('submodule', 'status'), 'runner_sha256': digest(__file__), 'pins': {}}
    if head != BASE:
        raise ValueError('staged repository base differs from pinned commit')
    for name, expected in SOURCE_PINS.items():
        actual = digest(ROOT / name)
        REPORT['source']['pins'][name] = {'actual': actual, 'expected': expected}
        save('source-pins')
        if actual != expected:
            raise ValueError('staged source pin mismatch: ' + name)
    source_paths = git('ls-files', '--', 'CMakeLists.txt', 'src', 'include', 'models', 'tests/surya25_run.cpp').splitlines()
    REPORT['source']['tracked_source_sha256'] = {
        name: digest(ROOT / name) for name in source_paths if (ROOT / name).is_file()}
    REPORT['source']['tracked_source_missing'] = [name for name in source_paths if not (ROOT / name).is_file()]
    save('source-closure')
    if REPORT['source']['tracked_source_missing']:
        raise ValueError('staged source tree is incomplete')
    util_path = Path(os.environ.get('SURYA_UTIL_PATH', ''))
    # SURYA_UTIL_PATH is intentionally not an HF_/CRISPEMBED_ variable.
    if not util_path.is_file():
        raise RuntimeError('SURYA_UTIL_PATH must name the staged pinned util.py')
    REPORT['package_util'] = {'path': str(util_path), 'commit': UTIL_COMMIT, 'sha256': digest(util_path),
                              'expected_sha256': UTIL_SHA, 'package_version': '0.23.0'}
    save('package-source')
    if REPORT['package_util']['sha256'] != UTIL_SHA:
        raise ValueError('pinned package util.py digest mismatch')
    util = load_module('surya25_pinned_util', util_path)
    if not callable(getattr(util, 'scale_to_fit', None)):
        raise ValueError('pinned util.py does not expose scale_to_fit')
    binary = Path(os.environ.get('SURYA25_NATIVE', '/kaggle/working/build/test-surya25-run'))
    if not binary.is_file():
        raise RuntimeError('prebuilt test-surya25-run missing: ' + str(binary))
    REPORT['binary'] = {'path': str(binary), 'sha256': digest(binary)}
    REPORT['versions'] = {'python': sys.version.split()[0]}
    for package, expected in PACKAGE_VERSIONS.items():
        actual = importlib.metadata.version(package)
        REPORT['versions'][package] = actual
        save('dependencies')
        if actual.split('+', 1)[0] != expected:
            raise ValueError(f'dependency {package}: expected {expected}, got {actual}')
    save('dependencies')
    from PIL import Image, features
    REPORT['pillow_codecs'] = {'jpeg': features.version('jpg'), 'zlib': features.version('zlib')}
    provenance_path = ROOT / 'tests/regression/images/cc0/MANIFEST.json'
    provenance_rows = json.loads(provenance_path.read_text())
    REPORT['fixture_provenance'] = {'manifest_sha256': digest(provenance_path), 'entries': {
        row['name']: row for row in provenance_rows
        if row.get('name') in ('german_official_print', 'receipt_historical')}}
    if set(REPORT['fixture_provenance']['entries']) != {'german_official_print', 'receipt_historical'}:
        raise ValueError('fixture source provenance entries missing')
    REPORT['fixture_provenance']['receipt_note'] = 'The frozen PNG is the repository derivative; manifest provenance names the source TIFF.'
    paths = {}
    for name, spec in CASES.items():
        source = ROOT / spec['path']
        actual_source = digest(source)
        case = {'source': dict(spec), 'actual_source_sha256': actual_source, 'arms': {'hf_f32': [], 'off': [], 'direct': []}}
        REPORT['cases'][name] = case
        save('crop-' + name)
        if actual_source != spec['source_sha256']:
            raise ValueError('fixture source digest mismatch: ' + name)
        with Image.open(source) as src:
            case['source_size'] = list(src.size)
            box = tuple(spec['box'])
            if not (0 <= box[0] < box[2] <= src.width and 0 <= box[1] < box[3] <= src.height):
                raise ValueError('crop outside source bounds')
            crop = src.convert('RGB').crop(box)
        rgb_sha = hashlib.sha256(crop.tobytes()).hexdigest()
        png = canonical_png(crop)
        png_sha = hashlib.sha256(png).hexdigest()
        case['crop'] = {'size': list(crop.size), 'rgb_sha256': rgb_sha, 'png_sha256': png_sha,
                        'png_encoding': 'RGB8 PNG, filter 0, stored DEFLATE, no metadata'}
        save('crop-' + name)
        if rgb_sha != spec['rgb_sha256'] or png_sha != spec['png_sha256']:
            raise ValueError('frozen crop bytes mismatch: ' + name)
        path = OUT / (name + '-raw.png')
        path.write_bytes(png)
        paths[name] = path
        start = time.perf_counter()
        scaled = util.scale_to_fit(crop).convert('RGB')
        elapsed = time.perf_counter() - start
        scaled_png = canonical_png(scaled)
        (OUT / (name + '-scaled.png')).write_bytes(scaled_png)
        case['package_scaled'] = {'size': list(scaled.size), 'rgb_sha256': hashlib.sha256(scaled.tobytes()).hexdigest(),
                                  'png_sha256': hashlib.sha256(scaled_png).hexdigest(), 'seconds': elapsed,
                                  'function': 'verified pinned util.py scale_to_fit'}
        save('crop-' + name)
    sys.path.insert(0, str(ROOT / 'models'))
    import qwen35_surya25_profile as profile
    if (profile.MODEL, profile.REVISION, profile.CHECKPOINT_SHA256, profile.CHECKPOINT_BYTES, profile.PROMPTS['block']) != (MODEL, REVISION, CHECKPOINT_SHA, CHECKPOINT_BYTES, PROMPT):
        raise ValueError('source profile differs from runner pins')
    from huggingface_hub import snapshot_download
    snapshot = model_dir / 'checkpoint'
    save('download-metadata')
    snapshot_download(MODEL, revision=REVISION, token=False, local_dir=str(snapshot), allow_patterns=list(profile.METADATA_FILES))
    profile.verify_metadata(snapshot)
    REPORT['metadata_sha256'] = {p: digest(snapshot / p) for p in profile.METADATA_FILES}
    save('download-checkpoint')
    snapshot_download(MODEL, revision=REVISION, token=False, local_dir=str(snapshot),
                      allow_patterns=list(profile.METADATA_FILES) + ['model.safetensors'])
    profile.verify_metadata(snapshot)
    weight = snapshot / 'model.safetensors'
    REPORT['checkpoint'] = {'sha256': digest(weight), 'bytes': weight.stat().st_size, 'anonymous_token_false': True}
    save('download-checkpoint')
    if (REPORT['checkpoint']['sha256'], REPORT['checkpoint']['bytes']) != (CHECKPOINT_SHA, CHECKPOINT_BYTES):
        raise ValueError('checkpoint digest or length mismatch')
    save('convert')
    converter = load_module('surya25_converter', ROOT / 'models/convert-qwen35-ocr-to-gguf.py')
    f16 = model_dir / 'surya25-f16.gguf'
    manifest = converter.convert(snapshot, f16, MODEL, REVISION, 'f16', (), profile='surya25')
    REPORT['conversion'] = {'sha256': digest(f16), 'expected_sha256': F16_SHA, 'bytes': f16.stat().st_size,
                            'manifest_bytes': manifest['bytes'], 'checkpoint_tensor_count': manifest['checkpoint_tensor_count'],
                            'export_tensor_count': manifest['export_tensor_count']}
    save('convert')
    if REPORT['conversion']['sha256'] != F16_SHA:
        raise ValueError('converted F16 digest mismatch')
    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    import inspect
    torch.manual_seed(0)
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    REPORT['hf_implementation'] = {'model_file_sha256': digest(inspect.getfile(Qwen3_5ForConditionalGeneration)),
                                   'model_file': inspect.getfile(Qwen3_5ForConditionalGeneration),
                                   'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                                   'device': 'cpu', 'dtype': 'float32', 'prompt': PROMPT, 'cap': CAP,
                                   'generation': {'do_sample': False, 'max_new_tokens': CAP, 'eos_token_id': EOS,
                                                  'pad_token_id': 0, 'use_cache': True}}
    save('load-hf')
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True, token=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(snapshot, local_files_only=True, token=False, dtype=torch.float32)
    model.eval()
    REPORT['hf_implementation']['model_generation_config'] = model.generation_config.to_dict()
    if any(p.device.type != 'cpu' or (p.is_floating_point() and p.dtype != torch.float32) for p in model.parameters()):
        raise ValueError('HF model not entirely CPU float32')
    REPORT['hf_implementation']['load_seconds'] = time.perf_counter() - start
    for name in CASES:
        case = REPORT['cases'][name]
        with Image.open(OUT / (name + '-scaled.png')) as img:
            scaled = img.convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if rendered != profile.render_prompt(PROMPT):
            raise ValueError('chat template differs from source profile')
        start = time.perf_counter()
        inputs = processor(text=[rendered], images=[scaled], return_tensors='pt',
                           min_pixels=profile.IMAGE_MIN_PIXELS, max_pixels=profile.IMAGE_MAX_PIXELS)
        processor_seconds = time.perf_counter() - start
        grid = inputs['image_grid_thw'][0].tolist()
        if len(grid) != 3 or any(type(v) is not int or v <= 0 for v in grid):
            raise ValueError('invalid processor grid')
        patches = math.prod(grid)
        pixels = inputs['pixel_values'].detach().cpu().numpy()
        pixel_path = OUT / (name + '-processor-pixel-values.npy')
        np.save(pixel_path, pixels, allow_pickle=False)
        case['processor'] = {'rendered_prompt': rendered, 'input_ids': inputs['input_ids'][0].tolist(),
                             'grid_thw': grid, 'unmerged_patches': patches, 'native_max_patches': 1024,
                             'pixel_values_shape': list(pixels.shape), 'pixel_values_dtype': str(pixels.dtype),
                             'pixel_values_bytes_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                             'pixel_values_npy_sha256': digest(pixel_path), 'seconds': processor_seconds}
        save('processor-' + name)
        if patches > 1024 or pixels.shape[0] != patches:
            raise ValueError(f'{name} actual processor patch grid exceeds native bound or shape differs')
        for index in range(ROUNDS):
            save(f'hf-{name}-{index}')
            start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=CAP, do_sample=False, eos_token_id=EOS,
                                        pad_token_id=0, use_cache=True, output_scores=True, return_dict_in_generate=True)
            seconds = time.perf_counter() - start
            ids = output.sequences[0, inputs['input_ids'].shape[1]:].tolist()
            conf = [float(torch.softmax(score[0].float(), dim=-1)[token]) for score, token in zip(output.scores, ids)]
            result = {'token_ids': ids, 'text': processor.tokenizer.decode(ids, skip_special_tokens=True),
                      'confidences': conf, 'stop': 'EOS' if ids and ids[-1] == EOS else 'cap',
                      'generate_seconds': seconds}
            validate_result(result)
            if result['stop'] == 'cap' and len(ids) != CAP:
                raise ValueError('HF stopped early without EOS; unknown stop')
            if index:
                result['vs_first_hf'] = compare(case['arms']['hf_f32'][0], result)
            case['arms']['hf_f32'].append(result)
            save(f'hf-{name}-{index}')
    del model, processor, inputs, output
    gc.collect()
    save('native')
    for name in CASES:
        case = REPORT['cases'][name]
        ref = case['arms']['hf_f32'][0]
        for index in range(ROUNDS):
            for arm in (('off', 'direct') if index % 2 == 0 else ('direct', 'off')):
                save(f'native-{name}-{index}-{arm}')
                result = native(binary, f16, paths[name], arm)
                result['vs_hf_f32'] = compare(ref, result)
                case['arms'][arm].append(result)
                save(f'native-{name}-{index}-{arm}')
            case['arms']['direct'][index]['vs_f16_off_same_round'] = compare(case['arms']['off'][index], case['arms']['direct'][index])
            save(f'compare-{name}-{index}')
    REPORT['status'] = 'completed measurements; differences are diagnostic, not a quality or performance qualification'
    save('complete')
    return 0


if __name__ == '__main__':
    REPORT['started_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    try:
        code = main()
    except BaseException as error:
        REPORT.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        code = 2
    REPORT['exit_code'] = code
    REPORT['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    try:
        save(REPORT['phase'])
    except BaseException:
        print('REPORT_WRITE_FAILED', traceback.format_exc(), file=sys.stderr)
        code = 2
    print('SURYA25_REAL_SCAN_RESULT', json.dumps({'status': REPORT['status'], 'phase': REPORT['phase'],
                                                  'report': str(OUT / 'report.json') if OUT else None,
                                                  'exit_code': code}), flush=True)
    raise SystemExit(code)
