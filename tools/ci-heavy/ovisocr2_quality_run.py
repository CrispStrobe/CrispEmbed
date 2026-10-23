"""Hosted original/native OvisOCR2 OCR quality and protected quantization lane.

All model/framework imports occur after a reviewed positive native baseline
receipt and frozen input preflight. Large weights live only in HEAVY_SCRATCH.
"""

import argparse
import ast
import gc
import hashlib
import importlib.util
import inspect
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'tests/results/qwen35-ocr'
BASELINE = EVIDENCE / 'ovisocr2-model-audit.json'
Q8_REVIEW = EVIDENCE / 'ovisocr2-q8-quality.json'
EXPORT = EVIDENCE / 'ovisocr2-export.json'
F16_SHA = 'd3a733a5d5ba5d63f5275500fad4cff585b82e2fb90d1def2fe767d04edf57db'
F16_BYTES = 1717222848
MODEL = 'ATH-MaaS/OvisOCR2'
REVISION = 'cbdede82e048659b83310476ff807da825f8fe78'
PHASES = ('original', 'f16', 'q8', 'q4')
ENV_GATES = {'CRISPEMBED_QWEN35_OCR': '1', 'CRISPEMBED_OVISOCR2_FRONTEND': '1',
             'CRISPEMBED_OVISOCR2_INFERENCE': '1'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def require_baseline(phase):
    if phase not in PHASES:
        raise ValueError('Unknown OvisOCR2 quality phase')
    if not BASELINE.is_file():
        raise ValueError('Reviewed positive OvisOCR2 native baseline receipt absent')
    receipt = json.loads(BASELINE.read_text())
    review = receipt.get('independent_review', {})
    if (receipt.get('status') != 'completed diagnostic audit'
            or receipt.get('source_commit') is None
            or not re.fullmatch(r'[0-9a-f]{40}', receipt['source_commit'])
            or receipt.get('failures') != [] or review.get('f16_positive') is not True
            or receipt.get('exports', {}).get('f16', {}).get('model_sha256') != F16_SHA):
        raise ValueError('OvisOCR2 original/native F16 baseline is not positively reviewed')
    if phase == 'q4':
        if not Q8_REVIEW.is_file():
            raise ValueError('Reviewed Q8 OvisOCR2 result absent; Q4 remains blocked')
        q8 = json.loads(Q8_REVIEW.read_text())
        if (q8.get('panel_canonical_sha256') !=
                '0bec97a7a04683c7c0aec95063aa01cf9a73bcf78db61f433d7d66d89588f212'
                or q8.get('independent_review', {}).get('q8_positive_for_q4') is not True):
            raise ValueError('OvisOCR2 Q8 result not independently qualified for Q4')
    export = json.loads(EXPORT.read_text())
    if (export['run'] != 39992662147 or
            export['producer']['exports']['f16']['model_sha256'] != F16_SHA or
            export['producer']['exports']['f16']['bytes'] != F16_BYTES):
        raise ValueError('Pinned OvisOCR2 F16 export declaration differs')
    return {'baseline_source_commit': receipt['source_commit'], 'baseline_sha256': sha(BASELINE),
            'export_sha256': sha(EXPORT), 'review': review}


def original_readme_functions(readme):
    blocks = re.findall(r'```python\s*\n(.*?)```', readme, re.S)
    prompts, cleaners, filters = [], [], []
    for block in blocks:
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prompt'
                                                      for t in node.targets):
                prompts.append(node.value)
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == 'OvisOCR2Parser'):
            cleaners.extend(node for node in cls.body if isinstance(node, ast.FunctionDef)
                            and node.name == '_clean_truncated_repeats')
            for method in (node for node in cls.body if isinstance(node, ast.FunctionDef)
                           and node.name == 'parse'):
                for condition in (node for node in ast.walk(method) if isinstance(node, ast.If)
                                  and isinstance(node.test, ast.Name) and node.test.id == 'filter_imgtags'):
                    filters.extend(node.value for node in condition.body
                                   if isinstance(node, ast.Assign)
                                   and any(isinstance(target, ast.Name) and target.id == 'text'
                                           for target in node.targets))
    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return literal(node.left) + literal(node.right)
        raise ValueError('Original README prompt is not a literal expression')
    if len(prompts) != 1 or len(cleaners) != 1 or len(filters) != 1:
        raise ValueError('Pinned original README prompt/cleaner/filter ambiguous')
    namespace = {}
    code = ast.fix_missing_locations(ast.Module(body=cleaners, type_ignores=[]))
    exec(compile(code, 'pinned-ovisocr2-README-cleaner', 'exec'), namespace)
    filter_code = ast.fix_missing_locations(ast.Expression(body=filters[0]))
    filter_fn = lambda text: eval(compile(filter_code, 'pinned-ovisocr2-README-filter', 'eval'),
                                  {'text': text})
    return literal(prompts[0]), lambda value: namespace['_clean_truncated_repeats'](None, value), filter_fn


def process_readme_output(raw, cleaner, filter_fn):
    stripped = raw.strip()
    return cleaner(filter_fn(stripped))


def command(label, argv, out, report, timeout=1800, env=None):
    path = out / (label + '.log')
    start = time.monotonic()
    try:
        with path.open('w') as log:
            result = subprocess.run([str(arg) for arg in argv], stdout=log,
                                    stderr=subprocess.STDOUT, env=env, timeout=timeout)
        exit_code, timed_out = result.returncode, False
    except subprocess.TimeoutExpired:
        exit_code, timed_out = None, True
    record = {'exit_code': exit_code, 'timed_out': timed_out, 'log': path.name,
              'log_sha256': sha(path), 'wall_seconds': time.monotonic() - start}
    report['commands'][label] = record
    (out / 'ovisocr2-quality.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if timed_out or exit_code:
        raise ValueError('Hosted OvisOCR2 command failed: ' + label)
    return path


def original_cases(source, cases, profile, out, report, save):
    import torch
    import transformers
    import tokenizers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, GenerationConfig

    if (torch.__version__.split('+')[0] != '2.11.0' or transformers.__version__ != '5.16.1'
            or tokenizers.__version__ != '0.23.3'):
        raise ValueError('Original OvisOCR2 quality framework versions differ')
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    sources = {'generation': GenerationConfig, 'model': Qwen3_5ForConditionalGeneration,
               'processor': type(processor), 'image_processor': type(processor.image_processor),
               'tokenizer': type(processor.tokenizer)}
    actual_sources = {name: sha(inspect.getfile(cls)) for name, cls in sources.items()}
    if actual_sources != profile['qualification']['loaded_source_sha256']:
        raise ValueError('Original OvisOCR2 quality framework source bytes differ')
    if (processor.tokenizer.eos_token_id != profile['primary_eos']
            or type(processor).__module__ + '.' + type(processor).__name__ != profile['processor']['class_name']
            or type(processor.image_processor).__module__ + '.' + type(processor.image_processor).__name__
            != profile['processor']['image_class']
            or getattr(processor.image_processor, 'backend', None) != profile['processor']['image_backend']
            or processor.image_processor.to_dict() != profile['processor']['image_config']):
        raise ValueError('Original OvisOCR2 quality processor/EOS differs')
    readme = (source / 'README.md').read_text()
    prompt, cleaner, filter_fn = original_readme_functions(readme)
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    rendered = processor.tokenizer.apply_chat_template(messages, tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=False)
    if rendered != profile['rendered_prompt']:
        raise ValueError('Original OvisOCR2 README prompt rendering differs')
    report['original_source'] = {'readme_sha256': sha(source / 'README.md'),
                                 'framework_source_sha256': actual_sources,
                                 'checkpoint_sha256': profile['checkpoint_sha256']}
    save()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        source, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation='eager').eval()
    if model.generation_config.to_diff_dict() != profile['generation']:
        raise ValueError('Original OvisOCR2 generation configuration differs')
    outputs = {}
    for case in cases:
        image = Image.open(case['path']).convert('RGB')
        if hashlib.sha256(image.tobytes()).hexdigest() != case['rgb_sha256']:
            raise ValueError('Frozen OvisOCR2 quality RGB differs: ' + case['id'])
        inputs = processor(text=[rendered], images=[image], return_tensors='pt',
                           return_mm_token_type_ids=True,
                           images_kwargs={'min_pixels': profile['image_min_pixels'],
                                          'max_pixels': profile['image_max_pixels']})
        if ('mm_token_type_ids' not in inputs
                or inputs['mm_token_type_ids'].shape != inputs['input_ids'].shape):
            raise ValueError('Original multimodal token type IDs missing: ' + case['id'])
        start = time.monotonic()
        with torch.inference_mode():
            sequence = model.generate(**inputs, max_new_tokens=case['cap'], do_sample=False,
                                      use_cache=True, eos_token_id=profile['stop_ids'],
                                      return_dict_in_generate=True).sequences[0]
        ids = sequence[inputs['input_ids'].shape[1]:].tolist()
        if not ids or len(ids) > case['cap'] or any(type(token) is not int or not 0 <= token < 248320
                                                  for token in ids):
            raise ValueError('Original generated IDs/cap differ: ' + case['id'])
        raw = processor.tokenizer.decode(ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        stop_id = ids[-1] if ids and ids[-1] in profile['stop_ids'] else None
        record = {'input_rgb_sha256': case['rgb_sha256'], 'input_png_sha256': case['png_sha256'],
                  'input_ids_sha256': hashlib.sha256(inputs['input_ids'].numpy().tobytes()).hexdigest(),
                  'token_ids': ids, 'raw_text': raw, 'processed_text': process_readme_output(raw, cleaner, filter_fn),
                  'finish_reason': 'stop' if stop_id is not None else 'length', 'stop_id': stop_id,
                  'cap': case['cap'], 'wall_seconds': time.monotonic() - start,
                  'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'confidence_available': False}
        outputs[case['id']] = record
        report['original_cases'] = outputs
        save()
    del model
    gc.collect()
    return outputs, cleaner, filter_fn


def gguf_tensors(path):
    import gguf
    reader = gguf.GGUFReader(str(path))
    return {tensor.name: {'shape': tensor.shape.tolist(), 'type': tensor.tensor_type.name,
                          'sha256': hashlib.sha256(tensor.data.tobytes()).hexdigest()}
            for tensor in reader.tensors}


def native_cases(binary, model, cases, cleaner, filter_fn, out, report, save, arm):
    outputs = {}
    env = dict(os.environ, **ENV_GATES)
    env['CRISPEMBED_QWEN35_QUANT'] = '1' if arm in ('q8', 'q4') else '0'
    for case in cases:
        label = arm + '-' + case['id']
        log = command(label, [binary, model, case['path'], case['cap'], 2],
                      out, report, timeout=1800, env=env)
        lines = [line.removeprefix('OVIS_QUALITY_RESULT ') for line in log.read_text().splitlines()
                 if line.startswith('OVIS_QUALITY_RESULT ')]
        if len(lines) != 1:
            raise ValueError('Missing/repeated test-only Ovis quality result: ' + label)
        data = json.loads(lines[0])
        if (data['cap'] != case['cap'] or not isinstance(data['token_ids'], list)
                or data['stop_kind'] not in (1, 2, 3)
                or (data['stop_kind'] == 1) != (data['stop_id'] in (248044, 248046))
                or len(data['token_ids']) != len(data['confidences'])):
            raise ValueError('Invalid test-only Ovis quality result: ' + label)
        raw = bytes.fromhex(data['raw_text_utf8_hex']).decode('utf-8')
        outputs[case['id']] = {
            'input_rgb_sha256': case['rgb_sha256'], 'input_png_sha256': case['png_sha256'],
            'raw_text': raw, 'processed_text': process_readme_output(raw, cleaner, filter_fn),
            'token_ids': data['token_ids'], 'stop_id': data['stop_id'],
            'finish_reason': 'stop' if data['stop_kind'] == 1 else 'length',
            'stop_kind': data['stop_kind'], 'cap': data['cap'],
            'confidences': data['confidences'], 'mean_confidence': data['mean_confidence'],
            'wall_seconds': data['wall_seconds'],
            'child_peak_rss_kib_upper_bound': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
        report.setdefault('native_cases', {})[arm] = outputs
        save()
    return outputs


def cabi_smoke(library, model, case, quantized=False):
    """Separate public API smoke; this API does not expose actual IDs/stops."""
    import ctypes
    from PIL import Image
    os.environ.update(ENV_GATES)
    os.environ['CRISPEMBED_QWEN35_QUANT'] = '1' if quantized else '0'
    image = Image.open(case['path']).convert('RGB')
    raw_pixels = image.tobytes()
    if hashlib.sha256(raw_pixels).hexdigest() != case['rgb_sha256']:
        raise ValueError('C ABI smoke input differs')
    lib = ctypes.CDLL(str(library))
    lib.crispembed_ocr_model_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_ocr_model_init.restype = ctypes.c_void_p
    lib.crispembed_ocr_model_set_max_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.crispembed_ocr_model_recognize.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                  ctypes.POINTER(ctypes.c_int)]
    lib.crispembed_ocr_model_recognize.restype = ctypes.c_char_p
    lib.crispembed_ocr_model_confidences.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.crispembed_ocr_model_confidences.restype = ctypes.POINTER(ctypes.c_float)
    lib.crispembed_ocr_model_mean_confidence.argtypes = [ctypes.c_void_p]
    lib.crispembed_ocr_model_mean_confidence.restype = ctypes.c_float
    lib.crispembed_ocr_model_free.argtypes = [ctypes.c_void_p]
    ctx = lib.crispembed_ocr_model_init(str(model).encode(), 2)
    if not ctx:
        raise ValueError('Public C ABI OCR model init failed')
    try:
        lib.crispembed_ocr_model_set_max_tokens(ctx, case['cap'])
        pixel_buf = ctypes.create_string_buffer(raw_pixels)
        output_len = ctypes.c_int(0)
        start = time.monotonic()
        result = lib.crispembed_ocr_model_recognize(ctx, pixel_buf, image.width,
                                                    image.height, 3, ctypes.byref(output_len))
        if result is None or output_len.value < 0:
            raise ValueError('Public C ABI OCR recognition failed')
        if len(result) != output_len.value:
            raise ValueError('Public C ABI OCR byte length differs')
        text = result.decode('utf-8') if result else ''
        count = ctypes.c_int(0)
        values = lib.crispembed_ocr_model_confidences(ctx, ctypes.byref(count))
        confidences = [values[index] for index in range(count.value)] if values else []
        return {'case': case['id'], 'raw_text': text, 'raw_utf8_hex': text.encode().hex(),
                'confidences': confidences, 'mean_confidence': float(lib.crispembed_ocr_model_mean_confidence(ctx)),
                'wall_seconds': time.monotonic() - start,
                'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'actual_token_ids': None, 'actual_stop_id': None,
                'limits': 'Public C ABI exposes text/confidence only; IDs and stop are from separate test-only runner'}
    finally:
        lib.crispembed_ocr_model_free(ctx)


def prepare_documents(out):
    """Freeze the previously reviewed realistic derivatives with their original renderer."""
    if importlib.metadata.version('Pillow') != '11.3.0' or importlib.metadata.version('CairoSVG') != '2.8.2':
        raise ValueError('Reviewed document renderer versions differ')
    preparation = load_module('ovis_quality_preparation', ROOT / 'tools/ci-heavy/ocr_quality_prepare.py')
    quality = load_module('ovis_quality_preparation_gate', ROOT / 'tools/ci-heavy/qwen35_quality.py')
    quality.require_reviewed_preparation(ROOT)
    result = preparation.prepare(ROOT, out / 'documents')
    if result['frozen_derivatives_exact'] is not True:
        raise ValueError('Reviewed document derivatives changed')
    return result


def frozen_cases(out, contract):
    from PIL import Image
    panel = contract.load_panel()
    rendered = contract.render_panel(out / 'synthetic', panel)
    prior = json.loads((ROOT / 'tests/results/ocr-quality-prepared.json').read_text())
    prepared = json.loads((out / 'documents/prepared.json').read_text())
    preparation = load_module('ovis_quality_frozen_docs', ROOT / 'tools/ci-heavy/ocr_quality_prepare.py')
    preparation.verify_prepared(prior, prepared)
    doc_panel = preparation.validate_panel(ROOT)
    cases = []
    for spec, frozen in zip(panel['cases'], rendered['cases']):
        cases.append({'id': spec['id'], 'split': spec['split'], 'gold': spec['gold'],
                      'path': out / 'synthetic' / frozen['file'], 'cap': panel['max_new_tokens'],
                      'rgb_sha256': frozen['rgb_sha256'], 'png_sha256': frozen['png_sha256'],
                      'license': 'CC0-1.0', 'kind': 'synthetic-smoke'})
    for spec, frozen in zip(doc_panel['cases'], prepared['cases']):
        if spec['id'] != frozen['id']:
            raise ValueError('Reviewed document order changed')
        path = out / 'documents' / frozen['file']
        image = Image.open(path).convert('RGB')
        if sha(path) != frozen['png_sha256'] or hashlib.sha256(image.tobytes()).hexdigest() != frozen['rgb_sha256']:
            raise ValueError('Reviewed document pixels changed: ' + spec['id'])
        gold = json.loads((ROOT / spec['gold']).read_text())['text']
        cases.append({'id': spec['id'], 'split': 'realistic-evaluation', 'gold': gold,
                      'path': path, 'cap': spec['max_new_tokens'],
                      'rgb_sha256': frozen['rgb_sha256'], 'png_sha256': frozen['png_sha256'],
                      'license': spec['license'], 'kind': spec['kind']})
    return rendered, cases, prepared


def score_outputs(contract, rendered, cases, outputs):
    synthetic = contract.score_panel(rendered, {case['id']: outputs[case['id']]
                                                for case in cases if case['kind'] == 'synthetic-smoke'})
    documents = []
    for case in cases:
        if case['split'] == 'realistic-evaluation':
            record = outputs[case['id']]
            if (record['input_rgb_sha256'] != case['rgb_sha256']
                    or record['input_png_sha256'] != case['png_sha256']):
                raise ValueError('Realistic document input changed')
            documents.append({'id': case['id'], 'license': case['license'],
                              'rgb_sha256': case['rgb_sha256'],
                              **contract.score_case(case['gold'], record, case['cap'])})
    if len(documents) != 3:
        raise ValueError('Realistic OCR document coverage differs')
    return {'synthetic': synthetic, 'realistic_documents': documents,
            'realistic_scope': 'Three separately licensed and visually reviewed documents; no fitting or calibration'}


def download_original(report, save):
    from huggingface_hub import hf_hub_download, snapshot_download
    profile_module = load_module('ovis_quality_profile', ROOT / 'models/qwen35_ovis_profile.py')
    source = Path(snapshot_download(MODEL, revision=REVISION,
                                    allow_patterns=list(profile_module.metadata_files())))
    profile = profile_module.load_profile(source, MODEL, REVISION)
    profile_module.validate_config(json.loads((source / 'config.json').read_text()), profile)
    report['metadata_source_sha256'] = {name: sha(source / name)
                                        for name in profile_module.metadata_files()}
    report['profile_qualification'] = profile['qualification']
    save()
    checkpoint = Path(hf_hub_download(MODEL, 'model.safetensors', revision=REVISION))
    if checkpoint.parent.resolve() != source.resolve():
        raise ValueError('Pinned metadata and checkpoint snapshots differ')
    profile_module.verify_checkpoint(checkpoint, profile)
    report['checkpoint'] = {'sha256': profile['checkpoint_sha256'],
                            'bytes': profile['checkpoint_bytes']}
    save()
    return source, profile


def run_main(phase, preflight=False, documents=False):
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'ovis-quality'
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'ovisocr2-quality.json'
    report = {'status': 'started', 'phase': 'preflight', 'requested_phase': phase,
              'source_commit': os.environ.get('GITHUB_SHA'), 'commands': {},
              'source_sha256': {name: sha(ROOT / name) for name in (
                  'tools/ci-heavy/ovisocr2_quality_run.py',
                  'tools/ci-heavy/ovisocr2_quality_contract.py',
                  'models/qwen35_ovis_profile.py',
                  'models/convert-qwen35-ocr-to-gguf.py')},
              'limits': ['Experimental OvisOCR2 profile only; native alias remains blocked',
                         'No quantization artifact is qualified without independent review',
                         'Synthetic text is a smoke panel; three documents are a bounded evaluation',
                         'No importance matrix or heldout calibration is used']}

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    save()
    try:
        report['prerequisite'] = require_baseline(phase)
        contract = load_module('ovis_quality_contract', ROOT / 'tools/ci-heavy/ovisocr2_quality_contract.py')
        contract.load_panel()
        contract.source_schema()
        quality = load_module('ovis_quality_preflight', ROOT / 'tools/ci-heavy/qwen35_quality.py')
        reviewed = quality.require_reviewed_preparation(ROOT)
        report['reviewed_documents_sha256'] = sha(ROOT / 'tests/results/ocr-quality-prepared.json')
        report['reviewed_document_ids'] = reviewed['visual_review']['reviewed_case_ids']
        report['phase'] = 'preflight-complete'
        save()
        if preflight:
            report['status'] = 'preflight passed; no model/framework loaded'
            save()
            return report
        if documents:
            report['phase'] = 'freeze-reviewed-documents'
            save()
            prepared = prepare_documents(out)
            report['prepared_documents_sha256'] = sha(out / 'documents/prepared.json')
            report['document_renderer'] = {key: prepared[key] for key in ('pillow', 'cairosvg', 'cairo', 'font')}
            report['status'] = 'documents frozen before model outputs'
            save()
            return report
        if not (out / 'documents/prepared.json').is_file():
            raise ValueError('Frozen reviewed documents absent before model download')
        report['phase'] = 'freeze-synthetic-and-verify-documents'
        save()
        rendered, cases, prepared = frozen_cases(out, contract)
        report['inputs'] = [{'id': c['id'], 'split': c['split'], 'kind': c['kind'],
                             'license': c['license'], 'gold': c['gold'], 'cap': c['cap'],
                             'rgb_sha256': c['rgb_sha256'], 'png_sha256': c['png_sha256']}
                            for c in cases]
        report['rendered_panel_sha256'] = sha(out / 'synthetic/rendered-panel.json')
        report['prepared_documents_sha256'] = sha(out / 'documents/prepared.json')
        report['phase'] = 'pinned-original-checkpoint'
        save()
        source, profile = download_original(report, save)
        report['versions'] = {name: importlib.metadata.version(name) for name in
                              ('torch', 'transformers', 'tokenizers', 'numpy', 'gguf', 'Pillow',
                               'huggingface-hub', 'safetensors', 'CairoSVG')}
        report['phase'] = 'original-hf-f32-on-frozen-pixels'
        save()
        original, cleaner, filter_fn = original_cases(source, cases, profile, out, report, save)
        report['scores'] = {'original': score_outputs(contract, rendered, cases, original)}
        save()
        if phase == 'original':
            report['status'] = 'completed original HF quality baseline; review pending'
            save()
            return report

        scratch.mkdir(parents=True, exist_ok=True)
        build = scratch / 'build'
        report['phase'] = 'portable-native-build'
        save()
        command('cmake-configure', ['cmake', '-S', ROOT, '-B', build,
                                   '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF',
                                   '-DCRISPEMBED_NATIVE=OFF', '-DGGML_BLAS=OFF',
                                   '-DGGML_METAL=OFF', '-DCRISPEMBED_BUILD_SHARED=ON'],
                out, report, timeout=600)
        command('cpu-baseline-check', [sys.executable, ROOT / 'scripts/check-cpu-baseline.py', build],
                out, report, timeout=120)
        command('native-build', ['cmake', '--build', build, '--config', 'Release', '--target',
                                 'test-ovisocr2-quality-run', 'crispembed-shared',
                                 'crispembed-quantize', '-j2'], out, report, timeout=2400)
        runner = build / 'test-ovisocr2-quality-run'
        library = build / 'libcrispembed.so'
        quantizer = build / 'crispembed-quantize'
        report['native_binary_sha256'] = {p.name: sha(p) for p in (runner, library, quantizer)}
        report['native_source_sha256'] = {
            str(p.relative_to(ROOT)): sha(p) for p in
            [ROOT / 'tests/ovisocr2_quality_run.cpp', ROOT / 'src/crispembed.cpp',
             ROOT / 'tools/quantize.cpp', *sorted((ROOT / 'src').glob('qwen35*'))]
            if p.is_file() and p.suffix in ('.cpp', '.h')}
        f16 = scratch / 'ovisocr2-f16.gguf'
        report['phase'] = 'qualified-f16-reconstruction'
        save()
        command('f16-convert', [sys.executable, ROOT / 'models/convert-qwen35-ocr-to-gguf.py',
                                '--model', MODEL, '--revision', REVISION, '--profile', 'ovisocr2',
                                '--dtype', 'f16', '--precision-recipe', 'original-f16', '--output', f16],
                out, report, timeout=2400)
        if f16.stat().st_size != F16_BYTES or sha(f16) != F16_SHA:
            raise ValueError('Actual F16 GGUF differs from original qualified export')
        report['models'] = {'f16': {'sha256': F16_SHA, 'bytes': F16_BYTES}}
        report['phase'] = 'f16-native-quality'
        save()
        f16_results = native_cases(runner, f16, cases, cleaner, filter_fn, out, report, save, 'f16')
        report['scores']['f16'] = score_outputs(contract, rendered, cases, f16_results)
        report['f16_paired_synthetic'] = contract.paired_quality_screen(
            report['scores']['original']['synthetic'], report['scores']['f16']['synthetic'])
        report['cabi_smoke'] = {'f16': cabi_smoke(library, f16, cases[0])}
        save()
        if phase == 'f16':
            report['status'] = 'completed F16 quality observations; independent review pending'
            save()
            return report

        recipe = 'q8-protected' if phase == 'q8' else 'q4-attention-q8'
        candidate = scratch / ('ovisocr2-' + phase + '.gguf')
        report['phase'] = phase + '-quantize'
        save()
        command(phase + '-quantize', contract.quant_command(quantizer, f16, candidate, recipe),
                out, report, timeout=2400)
        report['models'][phase] = {'sha256': sha(candidate), 'bytes': candidate.stat().st_size,
                                   'recipe': recipe}
        report['tensor_audit'] = contract.validate_quant_tensors(
            gguf_tensors(f16), gguf_tensors(candidate), recipe)
        report['phase'] = phase + '-native-quality'
        save()
        outputs = native_cases(runner, candidate, cases, cleaner, filter_fn, out, report, save, phase)
        report['scores'][phase] = score_outputs(contract, rendered, cases, outputs)
        report['paired_synthetic'] = contract.paired_quality_screen(
            report['scores']['f16']['synthetic'], report['scores'][phase]['synthetic'])
        report['cabi_smoke'][phase] = cabi_smoke(library, candidate, cases[0], quantized=True)
        report['status'] = 'completed quantized quality observations; independent review pending'
        save()
        return report
    except BaseException as exc:
        report['status'] = 'failed hosted OvisOCR2 quality run'
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc).replace(str(ROOT), '<repo>').replace(str(out), '<out>') \
            .replace(str(scratch), '<scratch>')[:400]
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=PHASES, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--prepare-documents', action='store_true')
    args = parser.parse_args()
    if args.preflight and args.prepare_documents:
        parser.error('Choose one preparation mode')
    run_main(args.phase, args.preflight, args.prepare_documents)
