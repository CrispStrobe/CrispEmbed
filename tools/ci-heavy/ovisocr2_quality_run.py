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
import stat
import subprocess
import sys
import time
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'tests/results/qwen35-ocr'
BASELINE = EVIDENCE / 'ovisocr2-model-audit.json'
Q8_REVIEW = EVIDENCE / 'ovisocr2-q8-quality.json'
ORIGINAL_REVIEW = EVIDENCE / 'ovisocr2-original-quality.json'
F16_REVIEW = EVIDENCE / 'ovisocr2-f16-quality.json'
EXPORT = EVIDENCE / 'ovisocr2-export.json'
PREPARED_PRESERVATION = EVIDENCE / 'ovisocr2-quality-prepared-preservation.json'
PREPARED_PRESERVATION_SHA = '6cf0c8750202b0c11480ac7a22970ed2e9d1dd40121c7ba185cc46b916aef46f'
PREPARED_ARCHIVE_SHA = 'a890c3ad8151706948d3d89529e5f5716405e5f790a3749436f36d78f20ca9cc'
PREPARED_ARCHIVE_BYTES = 724951
F16_SHA = 'd3a733a5d5ba5d63f5275500fad4cff585b82e2fb90d1def2fe767d04edf57db'
F16_BYTES = 1717222848
MODEL = 'ATH-MaaS/OvisOCR2'
REVISION = 'cbdede82e048659b83310476ff807da825f8fe78'
PHASES = ('original', 'f16', 'q8', 'q4', 'q4-calibrated', 'q4-attention-f16')
QUANT_PHASES = ('q8', 'q4', 'q4-calibrated', 'q4-attention-f16')
Q4_PHASES = ('q4', 'q4-calibrated', 'q4-attention-f16')
QUALITY_WORKFLOW_PATHS = ('.github/workflows/ovisocr2-quality.yml',
                          '.github/workflows/ovisocr2-metadata.yml')
F16_HOST_CONTROLS = ('full', 'dev-hello')
# The tree closes over all native runtime sources. The remaining paths are the
# exact build recipe, test runner, quantizer, registry source used by the shared
# library, and ggml gitlink. A changed source uses the explicit full control.
F16_SOURCE_CLOSURE_PATHS = ('src', 'CMakeLists.txt', 'tests/ovisocr2_quality_run.cpp',
    'tools/quantize.cpp', 'examples/cli/model_mgr.cpp', 'ggml',
    'tools/ci-heavy/ovisocr2_vision_env.py', 'tools/ci-heavy/ovisocr2_quality_contract.py',
    'tools/ocr_quality.py', 'models/convert-qwen35-ocr-to-gguf.py')
F16_AST_FUNCTIONS = ('native_environment', 'numerical_environment', 'vision_contract')
F16_AST_GLOBALS = ('ENV_GATES', 'DIFF_FLOAT_ZERO')
F16_AST_CONTRACTS = (*F16_AST_FUNCTIONS, *F16_AST_GLOBALS, 'cmake-configure')
ENV_GATES = {'CRISPEMBED_QWEN35_OCR': '1', 'CRISPEMBED_OVISOCR2_FRONTEND': '1',
             'CRISPEMBED_OVISOCR2_INFERENCE': '1'}
DIFF_FLOAT_ZERO = ('CRISPEMBED_QWEN35_PRECISE_RMS',
                   'CRISPEMBED_QWEN35_IMATRIX', 'CRISPEMBED_QWEN35_PARALLEL_F16',
                   'CRISPEMBED_QWEN35_PROFILE', 'CRISPEMBED_QWEN35_TOKENIZER_MARKS')


def vision_contract():
    return load_module('ovis_vision_env_quality', ROOT / 'tools/ci-heavy/ovisocr2_vision_env.py')


def native_environment(arm, selection):
    if arm not in ('f16', 'q8', 'q4', 'q4-calibrated'):
        raise ValueError('Unknown native OvisOCR2 quality arm')
    env = dict(os.environ, **ENV_GATES)
    env['CRISPEMBED_QWEN35_QUANT'] = '0' if arm == 'f16' else '1'
    return vision_contract().apply_selection(env, selection)


def numerical_environment(arm, selection):
    if arm not in ('q8', 'q4', 'q4-calibrated'):
        raise ValueError('Numerical OvisOCR2 diff is for a quantized candidate only')
    env = native_environment(arm, selection)
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


def f16_runtime_ast_contracts(source):
    """Fingerprint the two native env builders and literal CMake configure argv."""
    tree = ast.parse(source)
    result = {}
    for name in F16_AST_FUNCTIONS:
        nodes = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(nodes) != 1:
            raise ValueError('First F16 native environment source is ambiguous: ' + name)
        result[name] = hashlib.sha256(ast.dump(nodes[0], include_attributes=False).encode()).hexdigest()
    for name in F16_AST_GLOBALS:
        nodes = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == name
                         for target in node.targets)]
        if len(nodes) != 1:
            raise ValueError('First F16 native gate constant is ambiguous: ' + name)
        result[name] = hashlib.sha256(ast.dump(nodes[0], include_attributes=False).encode()).hexdigest()
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'command'
             and len(node.args) >= 2 and isinstance(node.args[0], ast.Constant)
             and node.args[0].value == 'cmake-configure' and isinstance(node.args[1], ast.List)]
    if len(calls) != 1:
        raise ValueError('First F16 CMake configure argv is ambiguous')
    result['cmake-configure'] = hashlib.sha256(
        ast.dump(calls[0].args[1], include_attributes=False).encode()).hexdigest()
    return result


def validate_f16_source_closure(record, first_head, candidate_head, first_native, candidate_native):
    """Pure proof used by both the hosted driver and publication reviewer."""
    if not isinstance(record, dict):
        raise ValueError('First F16 source closure absent')
    def head(value):
        return isinstance(value, str) and re.fullmatch('[0-9a-f]{40}', value) is not None
    paths = record.get('paths')
    asts = record.get('ast_contracts')
    if (not head(first_head) or not head(candidate_head)
            or record.get('first_f16_source_commit') != first_head
            or record.get('candidate_source_commit') != candidate_head
            or not isinstance(paths, dict) or set(paths) != set(F16_SOURCE_CLOSURE_PATHS)
            or not isinstance(asts, dict) or set(asts) != set(F16_AST_CONTRACTS)
            or not isinstance(first_native, dict) or not first_native
            or not isinstance(candidate_native, dict)
            or first_native != candidate_native):
        raise ValueError('First F16 source closure path/source/map differs')
    for name, pair in paths.items():
        if (not isinstance(pair, dict) or set(pair) != {'first_oid', 'candidate_oid'}
                or not head(pair['first_oid']) or pair['candidate_oid'] != pair['first_oid']):
            raise ValueError('First F16 source tree differs: ' + name)
    for name, pair in asts.items():
        if (not isinstance(pair, dict) or set(pair) != {'first_sha256', 'candidate_sha256'}
                or not isinstance(pair['first_sha256'], str)
                or re.fullmatch('[0-9a-f]{64}', pair['first_sha256']) is None
                or pair['candidate_sha256'] != pair['first_sha256']):
            raise ValueError('First F16 native environment/build AST differs: ' + name)
    if record.get('first_native_source_sha256') != first_native:
        raise ValueError('First F16 recorded native source map differs')
    return True


def f16_source_closure(first_producer, first_head, candidate_head):
    """Compare exact first/current Git blobs and build/environment AST before weights."""
    if (not isinstance(first_producer, dict)
            or first_producer.get('source_commit') != first_head
            or not isinstance(first_producer.get('native_source_sha256'), dict)
            or not isinstance(first_producer.get('source_sha256'), dict)):
        raise ValueError('First F16 source proof absent')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, timeout=120).strip()
    if git('rev-parse', 'HEAD') != candidate_head:
        raise ValueError('Candidate checkout differs from declared source commit')
    try:
        git('cat-file', '-e', first_head + '^{commit}')
    except subprocess.CalledProcessError:
        subprocess.run(['git', '-C', str(ROOT), 'fetch', '--depth=1', 'origin', first_head],
                       check=True, timeout=180, stdout=subprocess.DEVNULL)
        git('cat-file', '-e', first_head + '^{commit}')
    if git('status', '--porcelain', '--', *F16_SOURCE_CLOSURE_PATHS):
        raise ValueError('Candidate native source closure has uncommitted changes')
    paths = {name: {'first_oid': git('rev-parse', first_head + ':' + name),
                    'candidate_oid': git('rev-parse', candidate_head + ':' + name)}
             for name in F16_SOURCE_CLOSURE_PATHS}
    first_bytes = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        first_head + ':tools/ci-heavy/ovisocr2_quality_run.py'], timeout=120)
    first_source = first_bytes.decode('utf-8')
    first_sha = hashlib.sha256(first_bytes).hexdigest()
    if first_sha != first_producer['source_sha256'].get('tools/ci-heavy/ovisocr2_quality_run.py'):
        raise ValueError('First F16 runner source differs from reviewed report')
    old_ast = f16_runtime_ast_contracts(first_source)
    new_ast = f16_runtime_ast_contracts((ROOT / 'tools/ci-heavy/ovisocr2_quality_run.py').read_text())
    asts = {name: {'first_sha256': old_ast[name], 'candidate_sha256': new_ast[name]}
            for name in F16_AST_CONTRACTS}
    first_native = first_producer['native_source_sha256']
    if any(Path(name).is_absolute() or '..' in Path(name).parts
           or not (ROOT / name).is_file() for name in first_native):
        raise ValueError('First F16 native source map contains an unsafe or missing path')
    current_native = {name: sha(ROOT / name) for name in first_native}
    record = {'first_f16_source_commit': first_head, 'candidate_source_commit': candidate_head,
              'paths': paths, 'ast_contracts': asts,
              'first_native_source_sha256': first_native,
              'first_runner_source_sha256': first_sha}
    validate_f16_source_closure(record, first_head, candidate_head, first_native, current_native)
    return record


def f16_runtime_closure(first_producer, candidate_report):
    """Require same pinned packages, thread settings, and machine ISA family."""
    first_cpu = first_producer.get('cpu_runtime', {})
    candidate_cpu = candidate_report.get('cpu_runtime', {})
    versions = first_producer.get('versions')
    threads = first_cpu.get('thread_env')
    if (not isinstance(versions, dict) or not versions
            or versions != candidate_report.get('versions')
            or not isinstance(threads, dict) or not threads
            or threads != candidate_cpu.get('thread_env')
            or not first_cpu.get('machine')
            or first_cpu['machine'] != candidate_cpu.get('machine')):
        raise ValueError('First F16 package, thread or CPU architecture differs')
    return {'first_versions': versions, 'candidate_versions': candidate_report['versions'],
            'first_thread_env': threads, 'candidate_thread_env': candidate_cpu['thread_env'],
            'first_machine': first_cpu['machine'], 'candidate_machine': candidate_cpu['machine'],
            'first_cpu_model': first_cpu.get('cpu_model'),
            'candidate_cpu_model': candidate_cpu.get('cpu_model'),
            'scope': 'Exact packages/threads/architecture; CPU model recorded, not assumed equivalent'}


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
    selected_vision = vision_contract().reviewed_selection(producer)
    raw_sha = receipt.get('producer_report_sha256')
    if raw_sha is not None and not re.fullmatch('[0-9a-f]{64}', raw_sha):
        raise ValueError('OvisOCR2 archived raw producer report SHA is invalid')
    return {'run': receipt['run'], 'source_commit': producer['source_commit'],
            'artifact_id': artifact['id'], 'artifact_zip_sha256': artifact['zip_sha256'],
            'producer_canonical_sha256': canonical, 'producer_report_sha256': raw_sha,
            'selected_vision_preset': selected_vision}


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


def checked_f16_quality_receipt(receipt, model_sha, original_sha, selection):
    """Require the first independently reviewed eleven-case F16 run for quant arms."""
    if not isinstance(receipt, dict):
        raise ValueError('First reviewed F16 quality receipt absent')
    producer, review = receipt.get('producer'), receipt.get('independent_review')
    if not isinstance(producer, dict) or not isinstance(review, dict):
        raise ValueError('First F16 producer/review absent')
    canonical = hashlib.sha256(json.dumps(producer, sort_keys=True, separators=(',', ':'),
                                          ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    cases = producer.get('native_cases', {}).get('f16', {})
    inputs = producer.get('inputs', [])
    if (review.get('quality_positive') is not True or review.get('f16_11_complete') is not True
            or review.get('producer_canonical_sha256') != canonical
            or type(receipt.get('run')) is not int or receipt['run'] < 1
            or not re.fullmatch('[0-9a-f]{40}', receipt.get('source_commit', ''))
            or producer.get('source_commit') != receipt['source_commit']
            or type(receipt.get('artifact_id')) is not int or receipt['artifact_id'] < 1
            or receipt.get('artifact_name') != f"ovisocr2-quality-reports-{receipt['run']}"
            or not re.fullmatch('[0-9a-f]{64}', receipt.get('artifact_zip_sha256', ''))
            or review.get('artifact_zip_sha256') != receipt['artifact_zip_sha256']
            or not re.fullmatch('[0-9a-f]{64}', receipt.get('report_sha256', ''))
            or producer.get('status') != 'completed F16 quality observations; independent review pending'
            or producer.get('requested_phase') != 'f16'
            or producer.get('prerequisite', {}).get('baseline_sha256') != model_sha
            or producer.get('prerequisite', {}).get('original_review_sha256') != original_sha
            or producer.get('selected_vision_preset') != selection
            or producer.get('models', {}).get('f16', {}).get('sha256') != F16_SHA
            or producer.get('f16_paired_synthetic', {}).get('screen_passed') is not True
            or producer.get('f16_paired_documents', {}).get('screen_passed') is not True
            or not isinstance(inputs, list) or len(inputs) != 11
            or len({row.get('id') for row in inputs if isinstance(row, dict)}) != 11
            or not isinstance(cases, dict) or set(cases) != {row['id'] for row in inputs}
            or not isinstance(producer.get('scores', {}).get('f16'), dict)):
        raise ValueError('First F16 quality source, cases or independent proof differs')
    return {'run': receipt['run'], 'source_commit': receipt['source_commit'],
            'artifact_id': receipt['artifact_id'], 'artifact_zip_sha256': receipt['artifact_zip_sha256'],
            'report_sha256': receipt['report_sha256'], 'producer_canonical_sha256': canonical}


def require_baseline(phase, original_run=None):
    if phase not in PHASES:
        raise ValueError('Unknown OvisOCR2 quality phase')
    if not BASELINE.is_file():
        raise ValueError('Reviewed positive OvisOCR2 native baseline receipt absent')
    receipt = json.loads(BASELINE.read_text())
    pins = checked_model_receipt(receipt)
    if phase in Q4_PHASES:
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
    f16_pins = None
    if phase in QUANT_PHASES:
        if not F16_REVIEW.is_file():
            raise ValueError('First reviewed F16 quality receipt required before quantization')
        f16_pins = checked_f16_quality_receipt(json.loads(F16_REVIEW.read_text()), sha(BASELINE),
                                                sha(ORIGINAL_REVIEW), pins['selected_vision_preset'])
    return {'baseline_source_commit': pins['source_commit'], 'baseline_sha256': sha(BASELINE),
            'baseline_pins': pins, 'selected_vision_preset': pins['selected_vision_preset'],
            'export_sha256': sha(EXPORT),
            'review': receipt['independent_review'],
            'q8_review_sha256': sha(Q8_REVIEW) if phase in Q4_PHASES else None,
            'q8_diagnostic_gate': q8_gate if phase in Q4_PHASES else None,
            'original_review_sha256': sha(ORIGINAL_REVIEW) if phase != 'original' else None,
            'first_f16_review_sha256': sha(F16_REVIEW) if f16_pins else None,
            'first_f16_pins': f16_pins}


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


def reviewed_quality_artifact(run_id, out, receipt_path, retained_name):
    """Fetch only a small reviewed quality report, with exact service proof."""
    receipt = json.loads(Path(receipt_path).read_text())
    def api(path):
        return json.loads(subprocess.check_output(
            ['gh', 'api', 'repos/CrispStrobe/CrispEmbed/' + path], text=True, timeout=60))
    run = api('actions/runs/' + str(run_id))
    if (run.get('id') != run_id or run.get('head_sha') != receipt['source_commit']
            or run.get('status') != 'completed' or run.get('conclusion') != 'success'
            or run.get('path') not in QUALITY_WORKFLOW_PATHS):
        raise ValueError('Reviewed quality service run/source/workflow differs')
    artifact = api('actions/artifacts/' + str(receipt['artifact_id']))
    if (artifact.get('name') != 'ovisocr2-quality-reports-' + str(run_id)
            or artifact.get('expired') is not False
            or artifact.get('workflow_run', {}).get('id') != run_id
            or artifact.get('digest') != 'sha256:' + receipt['artifact_zip_sha256']
            or not 0 < artifact.get('size_in_bytes', 0) <= 50_000_000):
        raise ValueError('Reviewed quality artifact identity/digest differs')
    archive_path = out / (retained_name + '.zip')
    with archive_path.open('wb') as stream:
        subprocess.run(['gh', 'api', 'repos/CrispStrobe/CrispEmbed/actions/artifacts/'
                        + str(artifact['id']) + '/zip'], stdout=stream, check=True, timeout=300)
    if sha(archive_path) != receipt['artifact_zip_sha256']:
        raise ValueError('Reviewed quality ZIP payload differs')
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or names.count('ovisocr2-quality.json') != 1
                or any(Path(name).is_absolute() or '..' in Path(name).parts for name in names)):
            raise ValueError('Reviewed quality archive members differ')
        payload = archive.read('ovisocr2-quality.json')
    if hashlib.sha256(payload).hexdigest() != receipt['report_sha256']:
        raise ValueError('Reviewed quality report bytes differ')
    report = json.loads(payload)
    (out / (retained_name + '.json')).write_bytes(payload)
    return report, {'run': run_id, 'source_commit': run['head_sha'],
                    'artifact_id': artifact['id'], 'artifact_zip_sha256': sha(archive_path),
                    'report_sha256': hashlib.sha256(payload).hexdigest(),
                    'receipt_sha256': sha(receipt_path)}


def reviewed_original_artifact(run_id, out):
    return reviewed_quality_artifact(run_id, out, ORIGINAL_REVIEW, 'reviewed-original-quality')


def numerical_reference_artifact(listing, pins, expected):
    """Bound compressed activation bytes by the reviewed uncompressed payloads."""
    sizes = [expected.get(name, {}).get('bytes') for name in ('gguf', 'json')]
    if any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError('Reviewed original numerical reference byte counts missing')
    # The complete frozen activation GGUF compresses to 1.3 GB. The previous
    # 750 MB bound rejected that real artifact. Allow reviewed payload bytes
    # plus bounded PNG/LICENSE/ZIP overhead; still verify every extracted SHA.
    maximum_zip_bytes = sum(sizes) + 5_000_000
    candidates = [item for item in listing.get('artifacts', [])
                  if item.get('name') == 'ovisocr2-model-diff-reference-' + str(pins['run'])
                  and item.get('expired') is False]
    if (listing.get('total_count', 0) > 100 or len(candidates) != 1
            or type(candidates[0].get('id')) is not int or candidates[0]['id'] < 1
            or candidates[0].get('workflow_run', {}).get('id') != pins['run']
            or not 0 < candidates[0].get('size_in_bytes', 0) <= maximum_zip_bytes
            or not re.fullmatch('sha256:[0-9a-f]{64}', candidates[0].get('digest', ''))):
        raise ValueError('Unique bounded original numerical reference artifact missing')
    return candidates[0]


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
    artifact = numerical_reference_artifact(listing, pins, expected)
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
        if (len(names) != len(set(names)) or set(required) != set(names)
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


def parse_first_linear_diagnostic(log_text, exit_code):
    """Require the observed first linear operation only, without decoder coverage."""
    result = parse_numerical_diagnostic(log_text, exit_code)
    expected = {f'{phase}.linear.0.{stage}' for phase in ('prefill', 'cached')
                for stage in ('input', 'qkv', 'z', 'a', 'b', 'norm', 'output')}
    expected |= {f'{phase}.cache.0.{stage}' for phase in ('prefill', 'cached')
                 for stage in ('conv_states', 'recurrent_states')}
    actual = {row['stage'] for row in result['metrics']}
    result['local_expected_stages'] = sorted(expected)
    result['local_missing_stages'] = sorted(expected - actual)
    result['comparison_coverage'] = actual == expected
    result['passed'] = bool(exit_code == 0 and result['first_failure'] is None
                            and result['comparison_coverage']
                            and all(row['verdict'] == 'PASS' and row['finite']
                                    for row in result['metrics']))
    result['scope'] = 'First linear attention block only; no decoded OCR or full decoder claim'
    return result


def frozen_numerical_shapes(reference, baseline_producer):
    """Bind extracted original stage shapes to the reviewed 943-stage capture SHA."""
    original = Path(reference).with_name('ovisocr2-original.json')
    expected = baseline_producer['original_capture']['sha256']
    if sha(original) != expected:
        raise ValueError('Frozen original numerical stage JSON SHA differs')
    data = json.loads(original.read_text())
    stages = data.get('stages')
    if (not isinstance(stages, dict) or len(stages) != 943
            or data.get('missing_stages') != []
            or any(not isinstance(shape, list) or not shape
                   or any(type(dim) is not int or dim <= 0 for dim in shape)
                   for shape in stages.values())):
        raise ValueError('Frozen original numerical stage shapes incomplete')
    return stages, expected


def first_linear_weight_records(tensor_records):
    """Bind the first operation's nine loaded GGUF weights to the tensor audit."""
    suffixes = ('in_proj_qkv.weight', 'in_proj_z.weight', 'in_proj_a.weight',
                'in_proj_b.weight', 'out_proj.weight', 'conv1d.weight',
                'A_log', 'dt_bias', 'norm.weight')
    names = ['l.layers.0.linear_attn.' + suffix for suffix in suffixes]
    result = {}
    for precision in ('before', 'after'):
        records = tensor_records.get(precision, {})
        if any(name not in records for name in names):
            raise ValueError('First linear operation GGUF tensor record absent: ' + precision)
        result[precision] = {name: records[name] for name in names}
    return {'weights': result,
            'scope': 'Actual F16 and candidate GGUF tensor hashes/types/shapes; activations compared separately'}


def quant_raw_diff_manifest(folder, stages, metric_lines, trace_lines, phase, probe, capture_sha):
    """Inventory every actual Ref.compare dump against original F32 stage shapes."""
    folder = Path(folder)
    observed_lines = [match.group(1) for line in metric_lines
                      if (match := re.match(r'^(?:PASS|FAIL) ([A-Za-z0-9_.]+): cos_min=', line))]
    observed_lines += [match.group(1) for line in trace_lines
                       if (match := re.match(r'^TRACE ([A-Za-z0-9_.]+) ', line))]
    observed = set(observed_lines)
    duplicates = sorted(name for name in observed if observed_lines.count(name) != 1)
    required = set(observed)
    if probe == 'decoder':
        required |= {f'{part}.decoder.layer.{index}' for part in
                     ('prefill', 'cached', 'full_prefix') for index in range(24)}
        required |= {name for name in stages if re.fullmatch(r'generated\.scores\.\d+', name)}
        required.add('generated.scores.0')
    elif probe in ('first-linear', 'first-linear-control'):
        required |= set(parse_first_linear_diagnostic('', 1)['local_expected_stages'])
    else:
        raise ValueError('Unknown quant raw diff probe')
    records, invalid = [], []
    for path in sorted(folder.glob('*.f32')):
        stage = path.name[:-4]
        shape = stages.get(stage)
        shape_valid = (isinstance(shape, list) and bool(shape)
                       and all(type(dim) is int and dim > 0 for dim in shape))
        if not shape_valid:
            invalid.append(stage + ': frozen original shape absent')
        expected_bytes = math.prod(shape) * 4 if shape_valid else None
        actual_bytes = path.stat().st_size
        if shape_valid and actual_bytes != expected_bytes:
            invalid.append(stage + ': raw F32 byte count differs')
        records.append({'stage': stage, 'file': f'native-diff/{phase}/{probe}/{path.name}',
                        'original_shape': shape if shape_valid else None, 'bytes': actual_bytes,
                        'expected_bytes': expected_bytes, 'sha256': sha(path)})
    actual = {row['stage'] for row in records}
    missing = sorted(required - actual)
    unexpected = sorted(actual - observed)
    return {'phase': phase, 'probe': probe, 'original_capture_sha256': capture_sha,
            'observed_metric_stages': sorted(observed), 'required_stages': sorted(required),
            'missing_stages': missing, 'unreported_dump_stages': unexpected,
            'duplicate_observed_stages': duplicates,
            'invalid_files': invalid, 'file_count': len(records), 'files': records,
            'coverage_complete': bool(observed) and not missing and not invalid
                                 and not unexpected and not duplicates,
            'scope': 'Actual native Ref.compare/TRACE raw F32 vectors; shapes from frozen original JSON'}


def quant_probe(binary, candidate, reference, phase, out, scratch, report, save,
                selection, stages, capture_sha, probe):
    label = phase + {'decoder': '-official-numerical-diff',
                     'first-linear': '-first-linear-diff',
                     'first-linear-control': '-f16-first-linear-control'}[probe]
    dump = Path(scratch) / 'native-diff' / phase / probe
    dump.mkdir(parents=True, exist_ok=True)
    env = numerical_environment('q4' if phase == 'q4-attention-f16' else phase, selection)
    if probe == 'first-linear-control':
        env['CRISPEMBED_QWEN35_QUANT'] = '0'
    env['CRISPEMBED_DIFF_DUMP_DIR'] = str(dump)
    path = out / (label + '.log')
    started = time.monotonic()
    exit_code, error = None, None
    with path.open('w') as stream:
        try:
            args = [str(binary), str(candidate), str(reference)]
            args += ['--native-frontend'] if probe == 'decoder' else ['--first-layer']
            completed = subprocess.run(args,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       env=env, timeout=1800)
            exit_code = completed.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = type(exc).__name__
            stream.write('\nDiagnostic process error: ' + error + '\n')
    parser = parse_numerical_diagnostic if probe == 'decoder' else parse_first_linear_diagnostic
    result = parser(path.read_bytes(), exit_code)
    result.update({'log': path.name, 'log_sha256': sha(path), 'wall_seconds': time.monotonic() - started,
                   'source_reference_sha256': sha(reference),
                   'model_sha256': sha(candidate),
                   'quant_gate': env['CRISPEMBED_QWEN35_QUANT'],
                   'process_error': error,
                   'source_thresholds': {'cos_min': 0.99999, 'absolute_base': 0.002,
                                         'calibration': 'unchanged named original backend variation'},
                   'selected_vision_preset': selection})
    try:
        manifest = quant_raw_diff_manifest(dump, stages, result['metric_lines'],
                                           result['trace_lines'], phase, probe, capture_sha)
        manifest_path = out / (label + '-raw-manifest.json')
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        result['raw_diff_manifest'] = manifest_path.name
        result['raw_diff_manifest_sha256'] = sha(manifest_path)
        result['raw_diff_coverage_complete'] = manifest['coverage_complete']
        if not manifest['coverage_complete']:
            result['passed'] = False
    except (OSError, ValueError, TypeError) as exc:
        result['raw_diff_error'] = type(exc).__name__
        result['raw_diff_coverage_complete'] = False
        result['passed'] = False
    key = {'decoder': 'quant_numerical', 'first-linear': 'quant_first_linear',
           'first-linear-control': 'quant_first_linear_control'}[probe]
    report[key] = result
    save()
    return result


def quant_numerical_diagnostics(build, candidate, reference, phase, out, report, save,
                               selection, scratch, *, baseline_f16=None):
    """Run candidate probes and optional same-run F16 local control."""
    baseline = json.loads(BASELINE.read_text())['producer']
    stages, capture_sha = frozen_numerical_shapes(reference, baseline)
    results = {}
    invoked = []
    if baseline_f16 is not None:
        results['first-linear-control'] = quant_probe(
            Path(build) / 'test-qwen35-linear', baseline_f16, reference, phase,
            out, scratch, report, save, selection, stages, capture_sha,
            'first-linear-control')
        invoked.append('first-linear-control')
    else:
        results['first-linear-control'] = None
        report['quant_first_linear_control'] = {
            'run': False, 'reason': 'Candidate-only replay; no F16 file supplied'}
        save()
    for probe, target in (('first-linear', 'test-qwen35-linear'),
                          ('decoder', 'test-qwen35-decoder')):
        results[probe] = quant_probe(Path(build) / target, candidate, reference, phase,
                                     out, scratch, report, save, selection,
                                     stages, capture_sha, probe)
        invoked.append(probe)
    results['raw_complete'] = all(results[probe]['raw_diff_coverage_complete']
                                  for probe in invoked)
    results['invoked_probes'] = invoked
    report['quant_raw_complete'] = results['raw_complete']
    save()
    return results


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


def validate_reused_f16(previous, current, cases, rendered, contract):
    """Recompute the first reviewed F16 scores on exactly these eleven inputs."""
    receipt = json.loads(F16_REVIEW.read_text())
    if (previous != receipt.get('producer')
            or previous.get('inputs') != current.get('inputs')
            or previous.get('original_source') != current.get('original_source')
            or previous.get('scores', {}).get('original') != current.get('scores', {}).get('original')
            or previous.get('selected_vision_preset') != current.get('selected_vision_preset')):
        raise ValueError('First F16 source, frozen HF scores or eleven inputs differ')
    outputs = previous.get('native_cases', {}).get('f16', {})
    if not isinstance(outputs, dict) or set(outputs) != {case['id'] for case in cases}:
        raise ValueError('First F16 native case coverage differs')
    for case in cases:
        row = outputs[case['id']]
        ids, confidence = row.get('token_ids'), row.get('confidences')
        try:
            mean = validate_selected_confidences(ids, confidence, case['cap'])
        except ValueError as exc:
            raise ValueError('First F16 ID/confidence record differs: ' + case['id']) from exc
        stop = ids[-1] if ids[-1] in (248044, 248046) else None
        if (row.get('input_rgb_sha256') != case['rgb_sha256']
                or row.get('input_png_sha256') != case['png_sha256']
                or row.get('cap') != case['cap'] or row.get('stop_id') != stop
                or row.get('finish_reason') != ('stop' if stop is not None else 'length')
                or type(row.get('mean_confidence')) not in (int, float)
                or not math.isclose(row['mean_confidence'], mean, rel_tol=0, abs_tol=1e-6)):
            raise ValueError('First F16 source case or stop differs: ' + case['id'])
    scores = score_outputs(contract, rendered, cases, outputs)
    if scores != previous.get('scores', {}).get('f16'):
        raise ValueError('First F16 quality score recomputation differs')
    return outputs, scores


def validate_full_f16_host_control(actual, frozen, cases):
    """A fallback after source drift still needs all eleven exact IDs and text."""
    if (not isinstance(actual, dict) or not isinstance(frozen, dict)
            or len(cases) != 11 or set(actual) != set(frozen)
            or set(actual) != {case['id'] for case in cases}):
        raise ValueError('Full F16 host control lacks eleven frozen cases')
    for case in cases:
        name = case['id']
        row, old = actual[name], frozen[name]
        ids = row.get('token_ids')
        if (not isinstance(ids, list) or not ids
                or row.get('input_rgb_sha256') != case['rgb_sha256']
                or row.get('input_png_sha256') != case['png_sha256']
                or row.get('cap') != case['cap']
                or any(row.get(key) != old.get(key) for key in
                       ('token_ids', 'raw_text', 'processed_text', 'stop_id', 'finish_reason'))):
            raise ValueError('Full F16 host control IDs/text differ: ' + name)
        mean = validate_selected_confidences(ids, row.get('confidences'), case['cap'])
        if (type(row.get('mean_confidence')) not in (int, float)
                or not math.isfinite(row['mean_confidence'])
                or not math.isclose(row['mean_confidence'], mean, rel_tol=0, abs_tol=1e-6)):
            raise ValueError('Full F16 host control confidence differs: ' + name)
    return {'mode': 'full', 'native_case_count': 11,
            'frozen_ids_text_stop_exact': True,
            'scope': 'Same-run eleven-case control after source closure was unavailable'}


def validate_short_f16_host_control(actual, frozen, case):
    """One same-run smoke; it does not assert eleven-case host equivalence."""
    if case.get('id') != 'dev-hello' or not isinstance(actual, dict) or not isinstance(frozen, dict):
        raise ValueError('Short F16 host control requires frozen dev-hello')
    try:
        ids = actual['token_ids']
        confidences = actual['confidences']
        mean = validate_selected_confidences(ids, confidences, case['cap'])
        actual_mean = actual['mean_confidence']
        frozen_mean = frozen['mean_confidence']
    except (KeyError, TypeError) as exc:
        raise ValueError('Short F16 host control lacks actual selected-token scores') from exc
    if (actual.get('input_rgb_sha256') != case.get('rgb_sha256')
            or actual.get('input_png_sha256') != case.get('png_sha256')
            or actual.get('cap') != case.get('cap')
            or ids != frozen.get('token_ids') or ids[-1] != 248046
            or actual.get('raw_text') != frozen.get('raw_text')
            or actual.get('processed_text') != frozen.get('processed_text')
            or actual.get('stop_id') != frozen.get('stop_id') or actual.get('stop_id') != 248046
            or actual.get('finish_reason') != 'stop'
            or type(actual_mean) not in (int, float) or not math.isfinite(actual_mean)
            or not math.isclose(actual_mean, mean, rel_tol=0, abs_tol=1e-6)
            or type(frozen_mean) not in (int, float) or not math.isfinite(frozen_mean)
            or not isinstance(actual.get('raw_text'), str) or not actual['raw_text']
            or not isinstance(actual.get('processed_text'), str)
            or type(actual.get('wall_seconds')) not in (int, float)
            or not math.isfinite(actual['wall_seconds']) or actual['wall_seconds'] < 0
            or type(actual.get('child_peak_rss_kib_upper_bound')) is not int
            or actual['child_peak_rss_kib_upper_bound'] <= 0):
        raise ValueError('Short F16 host control differs in frozen pixels, IDs, text, EOS or mean')
    return {'mode': 'dev-hello', 'native_case_count': 1,
            'case_id': 'dev-hello', 'input_rgb_sha256': case['rgb_sha256'],
            'input_png_sha256': case['png_sha256'], 'token_ids': ids,
            'raw_text': actual['raw_text'], 'processed_text': actual['processed_text'],
            'stop_id': 248046, 'cap': case['cap'],
            'actual_mean_confidence': actual_mean,
            'frozen_first_f16_mean_confidence': frozen_mean,
            'mean_confidence_abs_delta': abs(actual_mean - frozen_mean),
            'wall_seconds': actual.get('wall_seconds'),
            'child_peak_rss_kib_upper_bound': actual.get('child_peak_rss_kib_upper_bound'),
            'scope': 'One same-run F16 dev-hello diagnostic; all eleven F16 scores come from the reviewed first run'}


def command(label, argv, out, report, timeout=1800, env=None):
    print('OVIS_QUALITY_COMMAND start ' + label, flush=True)
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
    print(f'OVIS_QUALITY_COMMAND done {label} exit={exit_code} '
          f'timeout={timed_out} seconds={record["wall_seconds"]:.3f}', flush=True)
    report['commands'][label] = record
    (out / 'ovisocr2-quality.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if timed_out or exit_code:
        raise ValueError('Hosted OvisOCR2 command failed: ' + label)
    return path


def reference_contracts():
    """Reuse the exact processor/generation guards validated by frozen capture."""
    return load_module('ovis_quality_reference_contracts',
                       ROOT / 'tools/dump_ovisocr2_reference.py')


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
    contracts = reference_contracts()
    report['original_processor_contract'] = contracts.processor_contract(processor, profile)
    save()
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
    report['original_generation_contract'] = contracts.generation_contract(
        model, profile, json.loads((source / 'config.json').read_text()), GenerationConfig)
    save()
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


def preserve_calibration_proof(stats, imatrix, out, report):
    """Keep exact small development-only vector proof in the report artifact."""
    folder = Path(out) / 'calibration'
    folder.mkdir(parents=True, exist_ok=True)
    expected = {'development-imatrix-stats.tsv': report['calibration']['stats_sha256'],
                'development-imatrix.gguf': report['calibration']['imatrix_sha256']}
    copied = {}
    for name, source in (('development-imatrix-stats.tsv', stats),
                         ('development-imatrix.gguf', imatrix)):
        source = Path(source)
        if not source.is_file() or source.stat().st_size <= 0 or sha(source) != expected[name]:
            raise ValueError('Development-only calibration proof source differs: ' + name)
        target = folder / name
        shutil.copyfile(source, target)
        if sha(target) != expected[name] or target.stat().st_size != source.stat().st_size:
            raise ValueError('Retained calibration proof differs: ' + name)
        copied[name] = {'path': 'calibration/' + name, 'sha256': expected[name],
                        'bytes': target.stat().st_size}
    report['calibration']['report_files'] = copied
    return copied


def development_imatrix(build, f16, cases, scratch, out, report, selection):
    """Collect 72 FFN activation vectors on the four frozen development images only."""
    import gguf
    import numpy as np
    development = [case for case in cases if case['split'] == 'development']
    if len(development) != 4 or any(case['kind'] != 'synthetic-smoke' for case in development):
        raise ValueError('Ovis importance calibration development split differs')
    stats = scratch / 'development-imatrix-stats.tsv'
    env = native_environment('f16', selection)
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
    preserve_calibration_proof(stats, path, out, report)
    return path


def native_case_timeout(case_id, document_timeout=1800):
    # Hosted audit resource budget only. Token caps and quality gates are fixed.
    if type(document_timeout) is not int or document_timeout not in (1800, 3600):
        raise ValueError('Native document audit timeout must be1800 or3600 seconds')
    return document_timeout if case_id in ('historical-receipt', 'two-column-page', 'inventory-table') else 1800


def native_cases(binary, model, cases, cleaner, filter_fn, out, report, save, arm, selection,
                 document_timeout=1800):
    outputs = {}
    env = native_environment('q4' if arm == 'q4-attention-f16' else arm, selection)
    for case in cases:
        label = arm + '-' + case['id']
        log = command(label, [binary, model, case['path'], case['cap'], 2],
                      out, report, timeout=native_case_timeout(case['id'], document_timeout), env=env)
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
        print(f'OVIS_QUALITY_CASE {label} tokens={len(data["token_ids"])} '
              f'stop={data["stop_id"]} seconds={data["wall_seconds"]:.3f}', flush=True)
        report.setdefault('native_cases', {})[arm] = outputs
        save()
    return outputs


def cabi_smoke(library, model, case, selection, quantized=False):
    """Separate public API smoke; this API does not expose actual IDs/stops."""
    import ctypes
    from PIL import Image
    os.environ.update(ENV_GATES)
    os.environ.update(vision_contract().apply_selection({}, selection))
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


def prepared_preservation():
    """Pin the public, anonymously reverified original preparation archive."""
    if not PREPARED_PRESERVATION.is_file() or sha(PREPARED_PRESERVATION) != PREPARED_PRESERVATION_SHA:
        raise ValueError('Original quality preparation preservation receipt missing/altered')
    receipt = json.loads(PREPARED_PRESERVATION.read_text())
    source = receipt.get('source', {})
    if (receipt.get('repo') != 'cstr/crispembed-regression-fixtures'
            or receipt.get('revision') != '057e12668bf8109ec83c65929c9b8d6cb99ca566'
            or receipt.get('file') != 'ocr-quality-v1/prepared-38879605915/prepared.zip'
            or receipt.get('url') != 'https://huggingface.co/datasets/cstr/crispembed-regression-fixtures/resolve/057e12668bf8109ec83c65929c9b8d6cb99ca566/ocr-quality-v1/prepared-38879605915/prepared.zip'
            or receipt.get('sha256') != PREPARED_ARCHIVE_SHA
            or receipt.get('bytes') != PREPARED_ARCHIVE_BYTES
            or receipt.get('anonymous_immutable_readback_verified') is not True
            or source.get('original_run') != 38879605915
            or source.get('original_artifact_id') != 12037153992
            or source.get('artifact_name') != 'ocr-quality-contract'
            or source.get('original_zip_sha256') != PREPARED_ARCHIVE_SHA
            or source.get('original_zip_bytes') != PREPARED_ARCHIVE_BYTES):
        raise ValueError('Original quality preparation mirror/source pin differs')
    return receipt


def verify_prepared_archive(archive, receipt, prior):
    """Read exactly four pinned members without extracting any archive path."""
    if Path(archive).stat().st_size != receipt['bytes'] or sha(archive) != receipt['sha256']:
        raise ValueError('Original prepared ZIP size/SHA differs')
    cases = receipt['source']['case_inputs']
    if len(cases) != 3 or len({case['id'] for case in cases}) != 3:
        raise ValueError('Original prepared ZIP case list differs')
    expected = {'prepared.json': prior['artifact_report_sha256']}
    expected.update({case['file']: case['png_sha256'] for case in cases})
    if len(expected) != 4:
        raise ValueError('Original prepared ZIP member names overlap')
    with zipfile.ZipFile(archive) as zipped:
        infos = zipped.infolist()
        if (len(infos) != 4 or {item.filename for item in infos} != set(expected)
                or any(item.filename != Path(item.filename).name or item.is_dir()
                       or item.flag_bits & 1 or stat.S_ISLNK(item.external_attr >> 16)
                       or item.file_size > PREPARED_ARCHIVE_BYTES for item in infos)):
            raise ValueError('Original prepared ZIP members/path contract differs')
        members = {}
        for item in infos:
            content = zipped.read(item)
            if hashlib.sha256(content).hexdigest() != expected[item.filename]:
                raise ValueError('Original prepared ZIP member SHA differs: ' + item.filename)
            members[item.filename] = content
    prepared = json.loads(members['prepared.json'])
    if prepared.get('cases') != cases:
        raise ValueError('Original prepared ZIP case metadata differs from preservation')
    return prepared, members


def prepare_documents(out, archive=None):
    """Reuse byte-identical reviewed derivatives; never rerender evaluation PNGs."""
    receipt = prepared_preservation()
    preparation = load_module('ovis_quality_preparation', ROOT / 'tools/ci-heavy/ocr_quality_prepare.py')
    quality = load_module('ovis_quality_preparation_gate', ROOT / 'tools/ci-heavy/qwen35_quality.py')
    quality.require_reviewed_preparation(ROOT)
    panel = preparation.validate_panel(ROOT)
    prior = json.loads((ROOT / 'tests/results/ocr-quality-prepared.json').read_text())
    if (sha(ROOT / preparation.PANEL) != prior['panel_sha256']
            or sha(ROOT / 'tools/ocr_quality.py') != prior['scorer_sha256']
            or [case['id'] for case in panel['cases']] != [case['id'] for case in receipt['source']['case_inputs']]
            or any(sha(ROOT / name) != value
                   for name, value in prior['split_evidence_sha256'].items())):
        raise ValueError('Reviewed document source/scorer/split contract differs')
    if archive is None:
        archive = Path(os.environ['HEAVY_SCRATCH']) / 'ovis-quality' / 'prepared.zip'
        archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(receipt['url'], timeout=90) as response, archive.open('xb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                if output.tell() > PREPARED_ARCHIVE_BYTES:
                    raise ValueError('Original prepared ZIP exceeds pinned byte count')
    prepared, members = verify_prepared_archive(archive, receipt, prior)
    preparation.verify_prepared(prior, prepared)
    if (prepared.get('source_commit') != prior['source_commit']
            or prepared.get('pillow') != '11.3.0' or prepared.get('cairosvg') != '2.8.2'
            or prepared.get('cairo') != '1.18.0' or prepared.get('font') != prior['font']):
        raise ValueError('Original reviewed renderer provenance differs')
    for spec, frozen in zip(panel['cases'], prepared['cases']):
        if (spec['id'] != frozen['id'] or spec['source_sha256'] != frozen['source_sha256']
                or spec['gold_sha256'] != frozen['gold_sha256']
                or spec['license'] != next(row['license'] for row in receipt['source']['licenses']
                                           if row['id'] == spec['id'])):
            raise ValueError('Original prepared source/license binding differs')
    destination = out / 'documents'
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in members.items():
        (destination / name).write_bytes(content)
    return prepared, {'mirror_revision': receipt['revision'], 'mirror_file': receipt['file'],
                      'zip_sha256': receipt['sha256'], 'zip_bytes': receipt['bytes'],
                      'preservation_receipt_sha256': PREPARED_PRESERVATION_SHA,
                      'original_run': receipt['source']['original_run'],
                      'original_artifact_id': receipt['source']['original_artifact_id'],
                      'prepared_json_sha256': prior['artifact_report_sha256'],
                      'member_sha256': {name: hashlib.sha256(content).hexdigest()
                                        for name, content in members.items()}}


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


# Only this presentation attribute is ignored for the Ovis document structure
# view. The unchanged strict-v1 scorer and its raw metrics remain in the report.
TABLE_BORDER_OPEN = re.compile(
    r'<table[ \t\r\n]+border[ \t\r\n]*=[ \t\r\n]*(?:[0-9]{1,10}|"[0-9]{1,10}"|\'[0-9]{1,10}\')[ \t\r\n]*>')


def score_ovis_document(ocr_quality, gold, text, *, complete, policy):
    """Score one exact Ovis table presentation variant through frozen v1 grammar."""
    if policy != 'ovisocr2-table-border-presentation-v1':
        raise ValueError('Unknown OvisOCR2 document presentation policy')
    strict = ocr_quality.score(gold, text, complete=complete)
    if strict['policy'] != 'crispembed-ocr-quality-v1':
        raise ValueError('Frozen strict OCR quality policy changed')
    normalized, count = TABLE_BORDER_OPEN.subn('<table>', text)
    primary = ocr_quality.score(gold, normalized, complete=complete)
    # The attribute affects only the structure view. Raw text error and size
    # always describe the actual unmodified model output.
    for key in ('raw_text', 'raw', 'characters', 'lines'):
        primary[key] = strict[key]
    primary.update(policy=policy, strict_v1=strict, normalized_text=normalized,
                   normalized_sha256=hashlib.sha256(normalized.encode('utf-8')).hexdigest(),
                   border_attribute_count=count)
    return primary


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
            structure = score_ovis_document(
                ocr_quality, gold, record['processed_text'],
                complete=scored['completed'], policy=contract.DOCUMENT_POLICY)
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


def run_main(phase, preflight=False, documents=False, original_run=None,
             f16_host_control='full', native_document_timeout=1800):
    native_case_timeout('historical-receipt', native_document_timeout)
    if native_document_timeout != 1800 and phase != 'f16':
        raise ValueError('Extended document resource budget is declared only for full F16 audit')
    if f16_host_control not in F16_HOST_CONTROLS:
        raise ValueError('Unknown F16 host control mode')
    effective_control = f16_host_control if phase in QUANT_PHASES else 'full'
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'ovis-quality'
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'ovisocr2-quality.json'
    report = {'status': 'started', 'phase': 'preflight', 'requested_phase': phase,
              'requested_f16_host_control': f16_host_control,
              'effective_f16_host_control': effective_control,
              'native_audit_timeout_seconds': {'synthetic': 1800, 'documents': native_document_timeout},
              'source_commit': os.environ.get('GITHUB_SHA'), 'commands': {},
              'source_sha256': {name: sha(ROOT / name) for name in (
                  'tools/ci-heavy/ovisocr2_quality_run.py',
                  'tools/dump_ovisocr2_reference.py',
                  'tools/ci-heavy/ovisocr2_model_audit.py',
                  'tools/ci-heavy/ovisocr2_vision_env.py',
                  'tools/ci-heavy/ovisocr2_quality_contract.py',
                  'tools/ci-heavy/ovisocr2_selective_precision.py',
                  'tools/ocr_quality.py',
                  'tests/results/qwen35-ocr/ovisocr2-quality-prepared-preservation.json',
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
        report['cpu_runtime'] = load_module(
            'ovis_quality_cpu_runtime', ROOT / 'tools/ci-heavy/ovisocr2_model_audit.py'
        ).cpu_runtime_attribution()
        selection = report['prerequisite']['selected_vision_preset']
        report['selected_vision_preset'] = selection
        report['baseline_service'] = verify_model_receipt_service(report['prerequisite']['baseline_pins'])
        contract = load_module('ovis_quality_contract', ROOT / 'tools/ci-heavy/ovisocr2_quality_contract.py')
        contract.load_panel()
        contract.source_schema()
        quality = load_module('ovis_quality_preflight', ROOT / 'tools/ci-heavy/qwen35_quality.py')
        reviewed = quality.require_reviewed_preparation(ROOT)
        preserved = prepared_preservation()
        report['reviewed_documents_sha256'] = sha(ROOT / 'tests/results/ocr-quality-prepared.json')
        report['reviewed_document_ids'] = reviewed['visual_review']['reviewed_case_ids']
        report['prepared_archive_pin'] = {'revision': preserved['revision'],
            'file': preserved['file'], 'zip_sha256': preserved['sha256'],
            'zip_bytes': preserved['bytes'],
            'receipt_sha256': PREPARED_PRESERVATION_SHA}
        report['phase'] = 'preflight-complete'
        save()
        if preflight:
            report['status'] = 'preflight passed; no model/framework loaded'
            save()
            return report
        if documents:
            report['phase'] = 'freeze-reviewed-documents'
            save()
            prepared, proof = prepare_documents(out)
            report['prepared_documents_sha256'] = sha(out / 'documents/prepared.json')
            report['prepared_archive_verified'] = proof
            (out / 'documents/archive-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
            report['document_renderer'] = {key: prepared[key] for key in ('pillow', 'cairosvg', 'cairo', 'font')}
            report['status'] = 'documents frozen before model outputs'
            save()
            return report
        if not (out / 'documents/prepared.json').is_file():
            raise ValueError('Frozen reviewed documents absent before model download')
        proof_path = out / 'documents/archive-proof.json'
        if not proof_path.is_file():
            raise ValueError('Verified prepared archive proof absent before model download')
        proof = json.loads(proof_path.read_text())
        if (proof.get('zip_sha256') != PREPARED_ARCHIVE_SHA
                or proof.get('zip_bytes') != PREPARED_ARCHIVE_BYTES
                or proof.get('preservation_receipt_sha256') != PREPARED_PRESERVATION_SHA
                or proof.get('prepared_json_sha256') != sha(out / 'documents/prepared.json')):
            raise ValueError('Prepared archive proof differs before model download')
        report['prepared_archive_verified'] = proof
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
                               'huggingface-hub', 'safetensors')}
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

        first_f16, frozen_rows, frozen_scores = None, None, None
        if phase in QUANT_PHASES:
            report['phase'] = 'reviewed-first-f16-artifact-and-source-before-weights'
            save()
            first_f16, first_proof = reviewed_quality_artifact(
                report['prerequisite']['first_f16_pins']['run'], out, F16_REVIEW,
                'reviewed-first-f16-quality')
            frozen_rows, frozen_scores = validate_reused_f16(
                first_f16, report, cases, rendered, contract)
            if first_proof['receipt_sha256'] != report['prerequisite']['first_f16_review_sha256']:
                raise ValueError('First F16 receipt changed after preflight')
            report['first_f16_baseline'] = {**first_proof,
                'producer_canonical_sha256': report['prerequisite']['first_f16_pins']['producer_canonical_sha256'],
                'case_count': len(frozen_rows), 'model_sha256': F16_SHA,
                'selected_vision_preset': selection}
            if effective_control == 'dev-hello':
                if phase == 'q4-attention-f16':
                    try:
                        report['f16_source_closure'] = f16_source_closure(
                            first_f16, report['prerequisite']['first_f16_pins']['source_commit'],
                            report['source_commit'])
                        report['f16_runtime_closure'] = f16_runtime_closure(first_f16, report)
                    except (ValueError, subprocess.CalledProcessError) as exc:
                        effective_control = 'full'
                        report['effective_f16_host_control'] = 'full'
                        report['short_control_ineligible'] = str(exc)[:250]
                        save()
                else:
                    report['f16_source_closure'] = f16_source_closure(
                        first_f16, report['prerequisite']['first_f16_pins']['source_commit'],
                        report['source_commit'])
                    report['f16_runtime_closure'] = f16_runtime_closure(first_f16, report)
            report['scores']['f16'] = frozen_scores
            report['f16_paired_synthetic'] = contract.paired_quality_screen(
                report['scores']['original']['synthetic'], frozen_scores['synthetic'])
            report['f16_paired_documents'] = contract.paired_document_screen(
                report['scores']['original']['realistic_documents'],
                frozen_scores['realistic_documents'])
            if (report['f16_paired_synthetic'] != first_f16.get('f16_paired_synthetic')
                    or report['f16_paired_documents'] != first_f16.get('f16_paired_documents')
                    or report['f16_paired_synthetic'].get('screen_passed') is not True
                    or report['f16_paired_documents'].get('screen_passed') is not True):
                raise ValueError('Frozen first F16 paired screens changed before candidate weights')
            report['f16_paired_source'] = 'reviewed first F16 eleven-case receipt, recomputed on unchanged originals'
            save()

        diff_reference = None
        if phase in QUANT_PHASES:
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
        if phase in QUANT_PHASES:
            targets.extend(('test-qwen35-linear', 'test-qwen35-decoder'))
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
                                           *([build / 'test-qwen35-linear'] if diff_reference else []),
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
        f16_cases = cases if effective_control == 'full' else [cases[0]]
        f16_results = native_cases(runner, f16, f16_cases, cleaner, filter_fn, out, report, save,
                                   'f16', selection, document_timeout=native_document_timeout)
        if effective_control == 'full':
            if phase == 'q4-attention-f16':
                report['full_f16_frozen_control'] = validate_full_f16_host_control(
                    f16_results, frozen_rows, cases)
            host_scores = score_outputs(contract, rendered, cases, f16_results)
            host_synthetic = contract.paired_quality_screen(
                report['scores']['original']['synthetic'], host_scores['synthetic'])
            host_documents = contract.paired_document_screen(
                report['scores']['original']['realistic_documents'],
                host_scores['realistic_documents'])
            if host_synthetic.get('screen_passed') is not True or host_documents.get('screen_passed') is not True:
                raise ValueError('Same-run full F16 quality screens failed')
            if phase == 'f16':
                report['scores']['f16'] = host_scores
                report['f16_paired_synthetic'] = host_synthetic
                report['f16_paired_documents'] = host_documents
            else:
                report['scores']['f16_host_control'] = host_scores
                report['f16_host_control'] = {'mode': 'full', 'native_case_count': len(f16_results),
                    'paired_synthetic': host_synthetic, 'paired_documents': host_documents,
                    'scope': 'Same-run F16 eleven-case diagnostic; frozen first F16 is the quant comparison baseline'}
        else:
            report['f16_host_control'] = validate_short_f16_host_control(
                f16_results['dev-hello'], frozen_rows['dev-hello'], cases[0])
        report['cabi_smoke'] = {'f16': cabi_smoke(library, f16, cases[0], selection)}
        save()
        if phase == 'f16':
            report['status'] = 'completed F16 quality observations; independent review pending'
            save()
            return report

        recipe = ('q4-attention-f16' if phase == 'q4-attention-f16' else
                  'q8-protected' if phase == 'q8' else 'q4-attention-q8')
        candidate = scratch / ('ovisocr2-' + phase + '.gguf')
        imatrix = None
        if phase == 'q4-calibrated':
            report['phase'] = 'development-only-native-importance'
            save()
            imatrix = development_imatrix(build, f16, cases, scratch, out, report, selection)
        report['phase'] = phase + '-quantize'
        save()
        selective = None
        before_tensors = None
        if phase == 'q4-attention-f16':
            selective = load_module('ovis_selective_precision',
                                    ROOT / 'tools/ci-heavy/ovisocr2_selective_precision.py')
            before_tensors = gguf_tensors(f16)
            argv = selective.quant_command(quantizer, f16, candidate, before_tensors, contract)
        else:
            argv = contract.quant_command(quantizer, f16, candidate, recipe)
        if imatrix is not None:
            argv.extend(['--imatrix', str(imatrix)])
        quant_log = command(phase + '-quantize', argv, out, report, timeout=2400)
        if imatrix is not None and ', 72 with imatrix' not in quant_log.read_text():
            raise ValueError('Quantizer skipped one or more development FFN importance vectors')
        report['models'][phase] = {'sha256': sha(candidate), 'bytes': candidate.stat().st_size,
                                   'recipe': recipe}
        before_tensors = before_tensors if before_tensors is not None else gguf_tensors(f16)
        after_tensors = gguf_tensors(candidate)
        report['tensor_audit'] = (selective.validate_quant_tensors(before_tensors, after_tensors,
                                                                 contract) if selective else
                                  contract.validate_quant_tensors(before_tensors, after_tensors,
                                                                  recipe))
        report['tensor_records'] = {'before': before_tensors, 'after': after_tensors}
        report['first_linear_weights'] = first_linear_weight_records(report['tensor_records'])
        save()
        report['phase'] = phase + '-official-numerical-diagnostic'
        save()
        quant_numerical_diagnostics(build, candidate, diff_reference, phase, out,
                                    report, save, selection, scratch, baseline_f16=f16)
        report['phase'] = phase + '-native-quality'
        save()
        outputs = native_cases(runner, candidate, cases, cleaner, filter_fn, out, report, save,
                               phase, selection)
        report['scores'][phase] = score_outputs(contract, rendered, cases, outputs)
        report['paired_synthetic'] = contract.paired_quality_screen(
            report['scores']['f16']['synthetic'], report['scores'][phase]['synthetic'])
        report['paired_synthetic_original'] = contract.paired_quality_screen(
            report['scores']['original']['synthetic'], report['scores'][phase]['synthetic'])
        report['paired_documents'] = {
            reference: contract.paired_document_screen(report['scores'][reference]['realistic_documents'],
                                                       report['scores'][phase]['realistic_documents'])
            for reference in ('original', 'f16')}
        report['cabi_smoke'][phase] = cabi_smoke(library, candidate, cases[0], selection,
                                                  quantized=True)
        save()
        if (not report['paired_synthetic']['screen_passed']
                or not report['paired_synthetic_original']['screen_passed']
                or any(not result['screen_passed'] for result in report['paired_documents'].values())):
            raise ValueError('Quantized paired OCR quality screen failed; observations retained')
        if report['quant_raw_complete'] is not True:
            raise ValueError('Quantized native raw diagnostic capture incomplete; quality observations retained')
        if (report['quant_first_linear_control']['passed'] is not True
                or report['quant_first_linear']['passed'] is not True):
            raise ValueError('First linear F16 control or candidate diagnostic failed; quality observations retained')
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
    parser.add_argument('--f16-host-control', choices=F16_HOST_CONTROLS, default='full')
    parser.add_argument('--native-document-timeout', type=int, choices=(1800, 3600), default=1800,
                        help='Hosted F16 document resource budget only; quality/token gates unchanged')
    args = parser.parse_args()
    if args.preflight and args.prepare_documents:
        parser.error('Choose one preparation mode')
    run_main(args.phase, args.preflight, args.prepare_documents, args.original_run,
             args.f16_host_control, args.native_document_timeout)
