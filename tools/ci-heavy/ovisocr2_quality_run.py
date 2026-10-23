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
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'tests/results/qwen35-ocr'
BASELINE = EVIDENCE / 'ovisocr2-model-audit.json'
Q8_REVIEW = EVIDENCE / 'ovisocr2-q8-quality.json'
ORIGINAL_REVIEW = EVIDENCE / 'ovisocr2-original-quality.json'
EXPORT = EVIDENCE / 'ovisocr2-export.json'
F16_SHA = 'd3a733a5d5ba5d63f5275500fad4cff585b82e2fb90d1def2fe767d04edf57db'
F16_BYTES = 1717222848
MODEL = 'ATH-MaaS/OvisOCR2'
REVISION = 'cbdede82e048659b83310476ff807da825f8fe78'
PHASES = ('original', 'f16', 'q8', 'q4', 'q4-calibrated')
ENV_GATES = {'CRISPEMBED_QWEN35_OCR': '1', 'CRISPEMBED_OVISOCR2_FRONTEND': '1',
             'CRISPEMBED_OVISOCR2_INFERENCE': '1'}
DIFF_FLOAT_ZERO = ('CRISPEMBED_QWEN35_PRECISE_RMS',
                   'CRISPEMBED_QWEN35_VISION_PRECISE_NORM',
                   'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN',
                   'CRISPEMBED_QWEN35_VISION_CASCADE_NORM',
                   'CRISPEMBED_QWEN35_IMATRIX', 'CRISPEMBED_QWEN35_PARALLEL_F16',
                   'CRISPEMBED_QWEN35_PROFILE', 'CRISPEMBED_QWEN35_TOKENIZER_MARKS')


def native_environment(arm):
    if arm not in ('f16', 'q8', 'q4', 'q4-calibrated'):
        raise ValueError('Unknown native OvisOCR2 quality arm')
    env = dict(os.environ, **ENV_GATES)
    env['CRISPEMBED_QWEN35_QUANT'] = '0' if arm == 'f16' else '1'
    return env


def numerical_environment(arm):
    if arm not in ('q8', 'q4', 'q4-calibrated'):
        raise ValueError('Numerical OvisOCR2 diff is for a quantized candidate only')
    env = native_environment(arm)
    env.update({name: '0' for name in DIFF_FLOAT_ZERO})
    env['CRISPEMBED_QWEN35_AUDIT_ALL_LAYERS'] = '1'
    return env


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_selected_confidences(ids, confidences, cap):
    if (not isinstance(ids, list) or not isinstance(confidences, list)
            or type(cap) is not int or not 0 < len(ids) <= cap
            or len(ids) != len(confidences)
            or any(type(token) is not int or not 0 <= token < 248320 for token in ids)
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or value < 0 or value > 1 for value in confidences)):
        raise ValueError('Actual selected-token confidence/ID record invalid')
    return sum(confidences) / len(confidences)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def checked_model_receipt(receipt):
    """Bind reviewed wrapper to its exact producer, run, and archived artifact."""
    if not isinstance(receipt, dict):
        raise ValueError('Reviewed OvisOCR2 model receipt wrapper absent')
    producer = receipt.get('producer')
    artifact = receipt.get('artifact')
    review = receipt.get('independent_review')
    if not all(isinstance(value, dict) for value in (producer, artifact, review)):
        raise ValueError('Reviewed OvisOCR2 model producer/artifact/review absent')
    canonical = hashlib.sha256(json.dumps(producer, sort_keys=True, separators=(',', ':'),
                                          ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    if (type(receipt.get('run')) is not int or receipt['run'] < 1
            or type(artifact.get('id')) is not int or artifact['id'] < 1
            or not re.fullmatch('[0-9a-f]{64}', artifact.get('zip_sha256', ''))
            or review.get('artifact_zip_sha256') != artifact['zip_sha256']
            or review.get('producer_canonical_sha256') != canonical
            or review.get('f16_positive') is not True
            or producer.get('status') != 'completed diagnostic audit'
            or not re.fullmatch('[0-9a-f]{40}', producer.get('source_commit', ''))
            or producer.get('failures') != []
            or producer.get('exports', {}).get('f16', {}).get('model_sha256') != F16_SHA):
        raise ValueError('OvisOCR2 original/native F16 baseline is not positively reviewed')
    raw_sha = receipt.get('producer_report_sha256')
    if raw_sha is not None and not re.fullmatch('[0-9a-f]{64}', raw_sha):
        raise ValueError('OvisOCR2 archived raw producer report SHA is invalid')
    return {'run': receipt['run'], 'source_commit': producer['source_commit'],
            'artifact_id': artifact['id'], 'artifact_zip_sha256': artifact['zip_sha256'],
            'producer_canonical_sha256': canonical, 'producer_report_sha256': raw_sha}


def verify_model_receipt_service(pins):
    """Read-only hosted service proof before dependencies, model files or builds."""
    def api(path):
        return json.loads(subprocess.check_output(
            ['gh', 'api', 'repos/CrispStrobe/CrispEmbed/' + path], text=True, timeout=60))
    run = api('actions/runs/' + str(pins['run']))
    artifact = api('actions/artifacts/' + str(pins['artifact_id']))
    if (run.get('id') != pins['run'] or run.get('status') != 'completed'
            or run.get('conclusion') != 'success'
            or run.get('head_sha') != pins['source_commit']
            or run.get('path') != '.github/workflows/ovisocr2-metadata.yml'
            or artifact.get('id') != pins['artifact_id']
            or artifact.get('expired') is not False
            or artifact.get('workflow_run', {}).get('id') != pins['run']
            or artifact.get('digest') != 'sha256:' + pins['artifact_zip_sha256']):
        raise ValueError('Reviewed OvisOCR2 model run/source/artifact service proof differs')
    return {'run': pins['run'], 'source_commit': pins['source_commit'],
            'artifact_id': pins['artifact_id'], 'artifact_zip_sha256': pins['artifact_zip_sha256']}


def checked_q8_diagnostic_review(q8):
    """A reviewed failed Q8 numerical result may still authorize a Q4 diagnostic."""
    if not isinstance(q8, dict):
        raise ValueError('Reviewed Q8 diagnostic receipt absent')
    q8_producer, q8_review = q8.get('producer'), q8.get('independent_review')
    if not isinstance(q8_producer, dict) or not isinstance(q8_review, dict):
        raise ValueError('Reviewed Q8 diagnostic producer/review absent')
    q8_canonical = hashlib.sha256(json.dumps(q8_producer, sort_keys=True,
                                   separators=(',', ':'), ensure_ascii=True,
                                   allow_nan=False).encode()).hexdigest()
    if (q8_producer.get('requested_phase') != 'q8'
            or q8_producer.get('scores', {}).get('q8') is None
            or q8_producer.get('quant_numerical') is None
            or not re.fullmatch('[0-9a-f]{64}',
                                q8_producer.get('models', {}).get('q8', {}).get('sha256', ''))
            or q8_review.get('producer_canonical_sha256') != q8_canonical
            or q8_review.get('q8_diagnostic_complete') is not True
            or q8_review.get('allow_q4_diagnostic') is not True):
        raise ValueError('OvisOCR2 Q8 result not reviewed for Q4 diagnostic comparison')
    return {'producer_canonical_sha256': q8_canonical,
            'q8_numerical_passed': q8_producer['quant_numerical'].get('passed') is True,
            'q8_positive_for_publication': q8_review.get('q8_positive_for_publication') is True}


def require_baseline(phase, original_run=None):
    if phase not in PHASES:
        raise ValueError('Unknown OvisOCR2 quality phase')
    if not BASELINE.is_file():
        raise ValueError('Reviewed positive OvisOCR2 native baseline receipt absent')
    receipt = json.loads(BASELINE.read_text())
    pins = checked_model_receipt(receipt)
    if phase in ('q4', 'q4-calibrated'):
        if not Q8_REVIEW.is_file():
            raise ValueError('Reviewed Q8 OvisOCR2 result absent; Q4 remains blocked')
        q8 = json.loads(Q8_REVIEW.read_text())
        q8_gate = checked_q8_diagnostic_review(q8)
    export = json.loads(EXPORT.read_text())
    if (export['run'] != 39992662147 or
            export['producer']['exports']['f16']['model_sha256'] != F16_SHA or
            export['producer']['exports']['f16']['bytes'] != F16_BYTES):
        raise ValueError('Pinned OvisOCR2 F16 export declaration differs')
    if phase != 'original':
        if not isinstance(original_run, int) or original_run < 1 or not ORIGINAL_REVIEW.is_file():
            raise ValueError('Reviewed original quality run and receipt required before candidate work')
        original_receipt = json.loads(ORIGINAL_REVIEW.read_text())
        review = original_receipt.get('independent_review', {})
        if (original_receipt.get('run') != original_run
                or not re.fullmatch('[0-9a-f]{40}', original_receipt.get('source_commit', ''))
                or not re.fullmatch('[0-9a-f]{64}', original_receipt.get('artifact_zip_sha256', ''))
                or not re.fullmatch('[0-9a-f]{64}', original_receipt.get('report_sha256', ''))
                or type(original_receipt.get('artifact_id')) is not int
                or review.get('original_11_complete') is not True):
            raise ValueError('Original quality artifact lacks a positive independent review')
    return {'baseline_source_commit': pins['source_commit'], 'baseline_sha256': sha(BASELINE),
            'baseline_pins': pins, 'export_sha256': sha(EXPORT),
            'review': receipt['independent_review'],
            'q8_review_sha256': sha(Q8_REVIEW) if phase in ('q4', 'q4-calibrated') else None,
            'q8_diagnostic_gate': q8_gate if phase in ('q4', 'q4-calibrated') else None,
            'original_review_sha256': sha(ORIGINAL_REVIEW) if phase != 'original' else None}


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


def reviewed_original_artifact(run_id, out):
    """Fetch only the small reviewed original report archive, with service proof."""
    receipt = json.loads(ORIGINAL_REVIEW.read_text())
    def api(path):
        return json.loads(subprocess.check_output(
            ['gh', 'api', 'repos/CrispStrobe/CrispEmbed/' + path], text=True, timeout=60))
    run = api('actions/runs/' + str(run_id))
    if (run.get('id') != run_id or run.get('head_sha') != receipt['source_commit']
            or run.get('status') != 'completed' or run.get('conclusion') != 'success'
            or run.get('path') != '.github/workflows/ovisocr2-quality.yml'):
        raise ValueError('Reviewed original quality service run/source/workflow differs')
    artifact = api('actions/artifacts/' + str(receipt['artifact_id']))
    if (artifact.get('name') != 'ovisocr2-quality-reports-' + str(run_id)
            or artifact.get('expired') is not False
            or artifact.get('workflow_run', {}).get('id') != run_id
            or artifact.get('digest') != 'sha256:' + receipt['artifact_zip_sha256']
            or not 0 < artifact.get('size_in_bytes', 0) <= 50_000_000):
        raise ValueError('Reviewed original quality artifact identity/digest differs')
    archive_path = out / 'reviewed-original-quality.zip'
    with archive_path.open('wb') as stream:
        subprocess.run(['gh', 'api', 'repos/CrispStrobe/CrispEmbed/actions/artifacts/'
                        + str(artifact['id']) + '/zip'], stdout=stream, check=True, timeout=300)
    if sha(archive_path) != receipt['artifact_zip_sha256']:
        raise ValueError('Reviewed original quality ZIP payload differs')
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or names.count('ovisocr2-quality.json') != 1
                or any(Path(name).is_absolute() or '..' in Path(name).parts for name in names)):
            raise ValueError('Reviewed original quality archive members differ')
        payload = archive.read('ovisocr2-quality.json')
    if hashlib.sha256(payload).hexdigest() != receipt['report_sha256']:
        raise ValueError('Reviewed original quality report bytes differ')
    report = json.loads(payload)
    (out / 'reviewed-original-quality.json').write_bytes(payload)
    return report, {'run': run_id, 'source_commit': run['head_sha'],
                    'artifact_id': artifact['id'], 'artifact_zip_sha256': sha(archive_path),
                    'report_sha256': hashlib.sha256(payload).hexdigest(),
                    'receipt_sha256': sha(ORIGINAL_REVIEW)}


def original_diff_reference(pins, producer, scratch, license_sha):
    """Fetch the separate source-run activation artifact; never fetch model weights."""
    expected = producer.get('reference_artifacts', {})
    readable = producer.get('readable', {})
    for name in ('gguf', 'json'):
        if not re.fullmatch('[0-9a-f]{64}', expected.get(name, {}).get('sha256', '')):
            raise ValueError('Reviewed original numerical reference SHA missing: ' + name)
    if not re.fullmatch('[0-9a-f]{64}', readable.get('png_sha256', '')):
        raise ValueError('Reviewed original readable PNG SHA missing')
    listing = json.loads(subprocess.check_output([
        'gh', 'api', 'repos/CrispStrobe/CrispEmbed/actions/runs/' + str(pins['run'])
        + '/artifacts?per_page=100'], text=True, timeout=60))
    candidates = [item for item in listing.get('artifacts', [])
                  if item.get('name') == 'ovisocr2-model-diff-reference-' + str(pins['run'])
                  and item.get('expired') is False]
    if (listing.get('total_count', 0) > 100 or len(candidates) != 1
            or candidates[0].get('workflow_run', {}).get('id') != pins['run']
            or not 0 < candidates[0].get('size_in_bytes', 0) <= 750_000_000
            or not re.fullmatch('sha256:[0-9a-f]{64}', candidates[0].get('digest', ''))):
        raise ValueError('Unique small original numerical reference artifact missing')
    artifact = candidates[0]
    folder = scratch / 'ovis-diff-reference'
    folder.mkdir(parents=True, exist_ok=True)
    archive_path = folder / 'reference.zip'
    with archive_path.open('wb') as stream:
        subprocess.run(['gh', 'api', 'repos/CrispStrobe/CrispEmbed/actions/artifacts/'
                        + str(artifact['id']) + '/zip'], stdout=stream, check=True, timeout=300)
    digest = sha(archive_path)
    if artifact['digest'] != 'sha256:' + digest:
        raise ValueError('Original numerical reference artifact ZIP digest differs')
    required = {'official-delta.gguf': expected['gguf']['sha256'],
                'ovisocr2-original.json': expected['json']['sha256'],
                'owned-readable.png': readable['png_sha256'], 'LICENSE': license_sha}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or not set(required) <= set(names)
                or any(Path(name).is_absolute() or '..' in Path(name).parts
                       or Path(name).name != name for name in names)):
            raise ValueError('Original numerical reference archive member set/path differs')
        for name, key in (('official-delta.gguf', 'gguf'), ('ovisocr2-original.json', 'json')):
            if (type(expected[key].get('bytes')) is not int
                    or archive.getinfo(name).file_size != expected[key]['bytes']):
                raise ValueError('Original numerical reference ZIP declaration differs: ' + name)
        for name in required:
            with archive.open(name) as source, (folder / name).open('wb') as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            if sha(folder / name) != required[name]:
                raise ValueError('Original numerical reference payload SHA differs: ' + name)
    for name, key in (('official-delta.gguf', 'gguf'), ('ovisocr2-original.json', 'json')):
        if (type(expected[key].get('bytes')) is not int
                or (folder / name).stat().st_size != expected[key]['bytes']):
            raise ValueError('Original numerical reference byte count differs: ' + name)
    return folder / 'official-delta.gguf', {
        'run': pins['run'], 'source_commit': pins['source_commit'],
        'artifact_id': artifact['id'], 'artifact_zip_sha256': digest,
        'payload_sha256': required,
        'scope': 'Original activations, JSON, readable PNG, and license only; no checkpoint or GGUF weights'}


def parse_numerical_diagnostic(log_text, exit_code):
    """Keep actual diff magnitudes and earliest failure; a partial trace cannot pass."""
    raw_log = log_text if isinstance(log_text, bytes) else log_text.encode('utf-8')
    log_text = raw_log.decode('utf-8', errors='replace')
    metric_pattern = re.compile(
        r'^(PASS|FAIL) ([^:\n]+): cos_min=([^ ]+) cos_global=([^ ]+) '
        r'max_abs=([^ ]+) mean_abs=([^ ]+) norms=([^ /]+)/([^\n ]+)$', re.M)
    metrics = []
    metric_lines = []
    for match in metric_pattern.finditer(log_text):
        metric_lines.append(match.group(0))
        try:
            values = [float(value) for value in match.groups()[2:]]
        except ValueError:
            values = [float('nan')] * 6
        finite = all(math.isfinite(value) for value in values)
        metrics.append({'verdict': match.group(1), 'stage': match.group(2),
                        **dict(zip(('cos_min', 'cos_global', 'max_abs', 'mean_abs',
                                    'native_norm', 'reference_norm'),
                                   [value if math.isfinite(value) else None for value in values])),
                        'finite': finite})
    failure_line = next((line for line in log_text.splitlines()
                         if line.startswith(('FAIL ', 'Error:', 'ERROR ', 'Traceback'))), None)
    failure = next((row for row, line in zip(metrics, metric_lines) if line == failure_line), None)
    if failure is None and failure_line is not None:
        failure = {'line': failure_line[:400]}
    if failure is None:
        failure = next((row for row in metrics if not row['finite']), None)
    ids_match = re.search(r'^Native generated IDs:([^\n]*)$', log_text, re.M)
    try:
        ids = [int(value) for value in ids_match.group(1).split()] if ids_match else None
    except ValueError:
        ids = None
    decoded = re.search(rb'^(PASS|FAIL) native decoded text exact bytes=(\d+): ', raw_log, re.M)
    declared_bytes = int(decoded.group(2)) if decoded else None
    decoded_payload = raw_log[decoded.end():decoded.end() + declared_bytes] if decoded else None
    try:
        decoded_text = decoded_payload.decode('utf-8') if decoded_payload is not None else None
    except UnicodeDecodeError:
        decoded_text = None
    layer_count = sum(row['stage'].startswith(('prefill.decoder.layer.', 'cached.decoder.layer.',
                                               'full_prefix.decoder.layer.')) for row in metrics)
    coverage = (layer_count == 72 and any(row['stage'].startswith('generated.scores.')
                                           for row in metrics))
    positive = (exit_code == 0 and failure is None and coverage
                and ids is not None and len(ids) > 0
                and decoded is not None and decoded.group(1) == b'PASS'
                and decoded_text is not None and len(decoded_payload) == declared_bytes
                and 'PASS composed 24-layer native decoder and greedy IDs' in log_text)
    return {'passed': bool(positive), 'exit_code': exit_code,
            'comparison_coverage': coverage, 'metric_count': len(metrics),
            'decoder_layer_count': layer_count, 'metric_lines': metric_lines,
            'budget_lines': [line for line in log_text.splitlines() if line.startswith('BUDGET ')],
            'trace_lines': [line for line in log_text.splitlines() if line.startswith('TRACE ')],
            'metrics': metrics, 'first_failure': failure,
            'generated_ids': ids, 'stop_id': ids[-1] if ids and ids[-1] in (248044, 248046) else None,
            'eos_observed': bool(ids and ids[-1] in (248044, 248046)),
            'decoded_text_exact': decoded.group(1) == b'PASS' if decoded else None,
            'decoded_text': decoded_text,
            'decoded_text_utf8_hex': decoded_payload.hex() if decoded_text is not None else None,
            'declared_decoded_bytes': declared_bytes,
            'limits': 'Original F32/F16 absolute and cosine thresholds are diagnostic for quantized weights; failure does not identify a float-runtime bug'}


def numerical_diagnostic(binary, candidate, reference, phase, out, report, save):
    label = phase + '-official-numerical-diff'
    path = out / (label + '.log')
    started = time.monotonic()
    exit_code, error = None, None
    with path.open('w') as stream:
        try:
            completed = subprocess.run([str(binary), str(candidate), str(reference), '--native-frontend'],
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       env=numerical_environment(phase), timeout=1800)
            exit_code = completed.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = type(exc).__name__
            stream.write('\nDiagnostic process error: ' + error + '\n')
    result = parse_numerical_diagnostic(path.read_bytes(), exit_code)
    result.update({'log': path.name, 'log_sha256': sha(path), 'wall_seconds': time.monotonic() - started,
                   'source_reference_sha256': sha(reference),
                   'process_error': error,
                   'source_thresholds': {'cos_min': 0.99999, 'absolute_base': 0.002,
                                         'calibration': 'unchanged named original backend variation'}})
    report['quant_numerical'] = result
    save()
    return result


def validate_reused_original(previous, current, cases, rendered, contract, profile, readme_sha):
    """Require exact frozen inputs and source proof before reusing selected tokens."""
    receipt = json.loads(ORIGINAL_REVIEW.read_text())
    if (previous.get('status') != 'completed original HF quality baseline; review pending'
            or previous.get('requested_phase') != 'original'
            or previous.get('source_commit') != receipt['source_commit']
            or previous.get('inputs') != current['inputs']
            or previous.get('rendered_panel_sha256') != current['rendered_panel_sha256']
            or previous.get('reviewed_documents_sha256') != current['reviewed_documents_sha256']
            or previous.get('original_source', {}).get('readme_sha256') != readme_sha
            or previous['original_source'].get('checkpoint_sha256') != profile['checkpoint_sha256']
            or previous['original_source'].get('framework_source_sha256')
            != profile['qualification']['loaded_source_sha256']):
        raise ValueError('Reviewed original quality source/input/prompt proof differs')
    outputs = previous.get('original_cases', {})
    if set(outputs) != {case['id'] for case in cases} or len(outputs) != 11:
        raise ValueError('Reviewed original quality case coverage differs')
    for case in cases:
        row = outputs[case['id']]
        ids = row.get('token_ids', [])
        confidence = row.get('confidences', [])
        try:
            mean = validate_selected_confidences(ids, confidence, case['cap'])
        except ValueError as exc:
            raise ValueError('Reviewed original quality confidence record differs: ' + case['id']) from exc
        if (row.get('input_rgb_sha256') != case['rgb_sha256']
                or row.get('input_png_sha256') != case['png_sha256']
                or row.get('cap') != case['cap']
                or row.get('stop_id') != (ids[-1] if ids[-1] in (248044, 248046) else None)
                or row.get('finish_reason') != ('stop' if row['stop_id'] is not None else 'length')):
            raise ValueError('Reviewed original quality ID/confidence/stop record differs: ' + case['id'])
        recorded_mean = row.get('mean_confidence')
        if (type(recorded_mean) not in (int, float) or not math.isfinite(recorded_mean)
                or not math.isclose(recorded_mean, mean, rel_tol=0, abs_tol=1e-12)):
            raise ValueError('Reviewed original quality confidence mean differs: ' + case['id'])
    scores = score_outputs(contract, rendered, cases, outputs)
    if scores != previous.get('scores', {}).get('original'):
        raise ValueError('Reviewed original quality score recomputation differs')
    return outputs, scores


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
            generated = model.generate(**inputs, max_new_tokens=case['cap'], do_sample=False,
                                       use_cache=True, eos_token_id=profile['stop_ids'],
                                       return_dict_in_generate=True, output_scores=True)
            sequence = generated.sequences[0]
        ids = sequence[inputs['input_ids'].shape[1]:].tolist()
        if not ids or len(ids) > case['cap'] or any(type(token) is not int or not 0 <= token < 248320
                                                  for token in ids):
            raise ValueError('Original generated IDs/cap differ: ' + case['id'])
        if len(generated.scores) != len(ids):
            raise ValueError('Original selected-token score count differs: ' + case['id'])
        confidences = []
        for token, scores in zip(ids, generated.scores):
            if list(scores.shape) != [1, 248320]:
                raise ValueError('Original selected-token vocabulary differs: ' + case['id'])
            confidence = float(torch.softmax(scores[0].float(), dim=-1)[token])
            confidences.append(confidence)
        mean_confidence = validate_selected_confidences(ids, confidences, case['cap'])
        raw = processor.tokenizer.decode(ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        stop_id = ids[-1] if ids and ids[-1] in profile['stop_ids'] else None
        record = {'input_rgb_sha256': case['rgb_sha256'], 'input_png_sha256': case['png_sha256'],
                  'input_ids_sha256': hashlib.sha256(inputs['input_ids'].numpy().tobytes()).hexdigest(),
                  'token_ids': ids, 'raw_text': raw, 'processed_text': process_readme_output(raw, cleaner, filter_fn),
                  'confidences': confidences,
                  'mean_confidence': mean_confidence,
                  'finish_reason': 'stop' if stop_id is not None else 'length', 'stop_id': stop_id,
                  'cap': case['cap'], 'wall_seconds': time.monotonic() - start,
                  'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'confidence_source': 'original generate output_scores softmax at actual selected IDs'}
        outputs[case['id']] = record
        report['original_cases'] = outputs
        save()
        del generated, sequence, inputs
        gc.collect()
    del model
    gc.collect()
    return outputs, cleaner, filter_fn


def gguf_tensors(path):
    import gguf
    reader = gguf.GGUFReader(str(path))
    return {tensor.name: {'shape': tensor.shape.tolist(), 'type': tensor.tensor_type.name,
                          'sha256': hashlib.sha256(tensor.data.tobytes()).hexdigest()}
            for tensor in reader.tensors}


def development_imatrix(build, f16, cases, scratch, out, report):
    """Collect 72 FFN activation vectors on the four frozen development images only."""
    import gguf
    import numpy as np
    development = [case for case in cases if case['split'] == 'development']
    if len(development) != 4 or any(case['kind'] != 'synthetic-smoke' for case in development):
        raise ValueError('Ovis importance calibration development split differs')
    stats = scratch / 'development-imatrix-stats.tsv'
    env = dict(os.environ, **ENV_GATES)
    env['CRISPEMBED_QWEN35_QUANT'] = '0'
    log = command('development-imatrix-collector',
                  [build / 'test-qwen35-imatrix', f16, stats,
                   *[case['path'] for case in development]], out, report, timeout=1800, env=env)
    if 'PASS native FFN importance: 72 matrices' not in log.read_text():
        raise ValueError('Native development-only FFN collection did not pass')
    expected = {f'l.layers.{index}.mlp.{kind}_proj.weight': 3584 if kind == 'down' else 1024
                for index in range(24) for kind in ('gate', 'up', 'down')}
    path = scratch / 'development-imatrix.gguf'
    writer = gguf.GGUFWriter(str(path), 'crispembed-imatrix')
    writer.add_uint32('imatrix.version', 1)
    rows = []
    for line in stats.read_text().splitlines():
        name, count_text, width_text, *values = line.split()
        count, width = int(count_text), int(width_text)
        if name not in expected or width != expected.pop(name) or count < 1 or len(values) != width:
            raise ValueError('Development FFN importance schema differs')
        weights = np.asarray(values, dtype=np.float64).astype(np.float32)
        if not np.isfinite(weights).all() or np.min(weights) < 0 or not np.max(weights) > 0:
            raise ValueError('Development FFN importance values invalid')
        writer.add_uint64('count.' + name, count)
        writer.add_tensor(name, weights)
        rows.append({'name': name, 'rows': count, 'columns': width,
                     'sum_sha256': hashlib.sha256(weights.tobytes()).hexdigest()})
    if expected or len(rows) != 72 or len({row['rows'] for row in rows}) != 1:
        raise ValueError('Development FFN importance coverage differs')
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    report['calibration'] = {
        'scope': 'Four frozen synthetic development images only; no heldout images or documents',
        'input_ids': [case['id'] for case in development],
        'input_rgb_sha256': {case['id']: case['rgb_sha256'] for case in development},
        'collector_source_sha256': sha(ROOT / 'tests/test_qwen35_imatrix.cpp'),
        'stats_sha256': sha(stats), 'imatrix_sha256': sha(path),
        'imatrix_bytes': path.stat().st_size, 'vectors': rows}
    return path


def native_cases(binary, model, cases, cleaner, filter_fn, out, report, save, arm):
    outputs = {}
    env = native_environment(arm)
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
    ocr_quality = load_module('ovis_quality_document_scorer', ROOT / 'tools/ocr_quality.py')
    synthetic = contract.score_panel(rendered, {case['id']: outputs[case['id']]
                                                for case in cases if case['kind'] == 'synthetic-smoke'})
    documents = []
    for case in cases:
        if case['split'] == 'realistic-evaluation':
            record = outputs[case['id']]
            if (record['input_rgb_sha256'] != case['rgb_sha256']
                    or record['input_png_sha256'] != case['png_sha256']):
                raise ValueError('Realistic document input changed')
            gold_path = next(row['gold'] for row in json.loads(
                (ROOT / 'tests/regression/ocr-quality/panel.json').read_text())['cases']
                if row['id'] == case['id'])
            gold = json.loads((ROOT / gold_path).read_text())
            scored = contract.score_case(case['gold'], record, case['cap'])
            structure = ocr_quality.score(
                gold, record['processed_text'],
                complete=scored['completed'])
            documents.append({'id': case['id'], 'license': case['license'],
                              'rgb_sha256': case['rgb_sha256'],
                              **scored,
                              'structure': structure})
    if len(documents) != 3:
        raise ValueError('Realistic OCR document coverage differs')
    return {'synthetic': synthetic, 'realistic_documents': documents,
            'realistic_scope': 'Three separately licensed and visually reviewed documents; no fitting or calibration'}


def download_original(report, save, checkpoint=True):
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
    if not checkpoint:
        return source, profile
    checkpoint = Path(hf_hub_download(MODEL, 'model.safetensors', revision=REVISION))
    if checkpoint.parent.resolve() != source.resolve():
        raise ValueError('Pinned metadata and checkpoint snapshots differ')
    profile_module.verify_checkpoint(checkpoint, profile)
    report['checkpoint'] = {'sha256': profile['checkpoint_sha256'],
                            'bytes': profile['checkpoint_bytes']}
    save()
    return source, profile


def run_main(phase, preflight=False, documents=False, original_run=None):
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
                         'Only q4-calibrated uses development-only importance; no heldout calibration']}

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    save()
    try:
        report['prerequisite'] = require_baseline(phase, original_run)
        report['baseline_service'] = verify_model_receipt_service(report['prerequisite']['baseline_pins'])
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
        source, profile = download_original(report, save, checkpoint=phase == 'original')
        report['versions'] = {name: importlib.metadata.version(name) for name in
                              ('torch', 'transformers', 'tokenizers', 'numpy', 'gguf', 'Pillow',
                               'huggingface-hub', 'safetensors', 'CairoSVG')}
        if phase == 'original':
            report['phase'] = 'original-hf-f32-on-frozen-pixels'
            save()
            original, cleaner, filter_fn = original_cases(source, cases, profile, out, report, save)
            report['scores'] = {'original': score_outputs(contract, rendered, cases, original)}
        else:
            report['phase'] = 'reviewed-original-artifact-reuse'
            save()
            previous, artifact_proof = reviewed_original_artifact(original_run, out)
            report['original_artifact_reuse'] = artifact_proof
            readme = (source / 'README.md').read_text()
            prompt, cleaner, filter_fn = original_readme_functions(readme)
            if previous['original_source'].get('readme_sha256') != sha(source / 'README.md') \
                    or prompt not in profile['rendered_prompt']:
                raise ValueError('Pinned README quality prompt differs before reuse')
            original, scores = validate_reused_original(
                previous, report, cases, rendered, contract, profile, sha(source / 'README.md'))
            report['original_cases'] = original
            report['original_source'] = previous['original_source']
            report['scores'] = {'original': scores}
        save()
        if phase == 'original':
            report['status'] = 'completed original HF quality baseline; review pending'
            save()
            return report

        diff_reference = None
        if phase in ('q8', 'q4', 'q4-calibrated'):
            report['phase'] = 'reviewed-original-numerical-reference'
            save()
            baseline_producer = json.loads(BASELINE.read_text())['producer']
            diff_reference, diff_proof = original_diff_reference(
                report['prerequisite']['baseline_pins'], baseline_producer, scratch,
                report['metadata_source_sha256']['LICENSE'])
            report['numerical_reference'] = diff_proof
            save()

        scratch.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / 'LICENSE', scratch / 'LICENSE')
        shutil.copyfile(source / 'README.md', scratch / 'UPSTREAM_README.md')
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
        targets = ['test-ovisocr2-quality-run', 'crispembed-shared', 'crispembed-quantize']
        if phase in ('q8', 'q4', 'q4-calibrated'):
            targets.append('test-qwen35-decoder')
        if phase == 'q4-calibrated':
            targets.append('test-qwen35-imatrix')
        command('native-build', ['cmake', '--build', build, '--config', 'Release', '--target',
                                 *targets, '-j2'], out, report, timeout=2400)
        runner = build / 'test-ovisocr2-quality-run'
        library = build / 'libcrispembed.so'
        quantizer = build / 'crispembed-quantize'
        report['native_binary_sha256'] = {p.name: sha(p) for p in
                                          [runner, library, quantizer,
                                           *([build / 'test-qwen35-decoder'] if diff_reference else []),
                                           *([build / 'test-qwen35-imatrix'] if phase == 'q4-calibrated' else [])]}
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
        report['f16_paired_documents'] = contract.paired_document_screen(
            report['scores']['original']['realistic_documents'],
            report['scores']['f16']['realistic_documents'])
        report['cabi_smoke'] = {'f16': cabi_smoke(library, f16, cases[0])}
        save()
        if not report['f16_paired_synthetic']['screen_passed'] or not report['f16_paired_documents']['screen_passed']:
            raise ValueError('F16 paired OCR quality screen failed; observations retained')
        if phase == 'f16':
            report['status'] = 'completed F16 quality observations; independent review pending'
            save()
            return report

        recipe = 'q8-protected' if phase == 'q8' else 'q4-attention-q8'
        candidate = scratch / ('ovisocr2-' + phase + '.gguf')
        imatrix = None
        if phase == 'q4-calibrated':
            report['phase'] = 'development-only-native-importance'
            save()
            imatrix = development_imatrix(build, f16, cases, scratch, out, report)
        report['phase'] = phase + '-quantize'
        save()
        argv = contract.quant_command(quantizer, f16, candidate, recipe)
        if imatrix is not None:
            argv.extend(['--imatrix', str(imatrix)])
        quant_log = command(phase + '-quantize', argv, out, report, timeout=2400)
        if imatrix is not None and ', 72 with imatrix' not in quant_log.read_text():
            raise ValueError('Quantizer skipped one or more development FFN importance vectors')
        report['models'][phase] = {'sha256': sha(candidate), 'bytes': candidate.stat().st_size,
                                   'recipe': recipe}
        report['tensor_audit'] = contract.validate_quant_tensors(
            gguf_tensors(f16), gguf_tensors(candidate), recipe)
        report['phase'] = phase + '-official-numerical-diagnostic'
        save()
        numerical_diagnostic(build / 'test-qwen35-decoder', candidate, diff_reference,
                             phase, out, report, save)
        report['phase'] = phase + '-native-quality'
        save()
        outputs = native_cases(runner, candidate, cases, cleaner, filter_fn, out, report, save, phase)
        report['scores'][phase] = score_outputs(contract, rendered, cases, outputs)
        report['paired_synthetic'] = contract.paired_quality_screen(
            report['scores']['f16']['synthetic'], report['scores'][phase]['synthetic'])
        report['paired_documents'] = {
            reference: contract.paired_document_screen(report['scores'][reference]['realistic_documents'],
                                                       report['scores'][phase]['realistic_documents'])
            for reference in ('original', 'f16')}
        report['cabi_smoke'][phase] = cabi_smoke(library, candidate, cases[0], quantized=True)
        save()
        if (not report['paired_synthetic']['screen_passed']
                or any(not result['screen_passed'] for result in report['paired_documents'].values())):
            raise ValueError('Quantized paired OCR quality screen failed; observations retained')
        if report['quant_numerical']['passed'] is not True:
            raise ValueError('Quantized numerical diagnostic failed; quality observations retained')
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
    parser.add_argument('--original-run', type=int)
    args = parser.parse_args()
    if args.preflight and args.prepare_documents:
        parser.error('Choose one preparation mode')
    run_main(args.phase, args.preflight, args.prepare_documents, args.original_run)
