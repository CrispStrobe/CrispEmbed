"""Verify quantization evidence without weights; retained failures are not waived."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
from qwen35_quality_quant import PROFILES, frozen_baseline, frozen_original, summary, tensor_audit
from qwen35_quality import strict_prefill
from qwen35_decoded_panel import validate
from qwen35_page_parity import parse
from qwen35_receipt import add_confidences, teacher_quality
from ocr_quality import score

DIAGNOSTICS = ('tools/ci-heavy/qwen35_quality_quant.py', 'tools/ci-heavy/qwen35_quality.py',
    'tools/dump_qwen35_ocr_page_reference.py', 'tests/test_qwen35_quant.cpp',
    'tests/qwen35_parity.h', 'tools/quantize.cpp')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen(report, repo, expected_source):
    require(bool(re.fullmatch(r'[0-9a-f]{40}', expected_source))
        and report['source_commit'] == expected_source, 'Wrong source commit')
    require(report['result'].startswith('completed') and report['production_changes'] == 'none'
        and report['publication'] == 'none', 'Incomplete or out-of-scope diagnostic')
    prior_path = repo / 'tests/results/qwen35-ocr/quality.json'
    prior = json.loads(prior_path.read_text())
    require(report['prior_report_sha256'] == sha(prior_path), 'Frozen report hash changed')
    for field in ('model', 'revision', 'policy', 'weights', 'native_environment', 'runtime_sources'):
        require(report[field] == prior[field], 'Frozen ' + field + ' changed')
    require(report['prior_workflow'] == prior['workflow'], 'Original workflow changed')
    require(set(report['diagnostic_sources']) == set(DIAGNOSTICS), 'Missing/extra diagnostic sources')
    for group in ('runtime_sources', 'diagnostic_sources'):
        for path, digest in report[group].items():
            require(sha(repo / path) == digest, 'Source bytes changed: ' + path)
    old = next(c for c in prior['cases'] if c['id'] == report['case']['id'])
    require(report['case'] == {k: old[k] for k in ('id', 'specification', 'prepared', 'gold')},
        'Frozen input/gold/specification changed')
    require(report['frozen_original_exact'] and report['frozen_f16_exact'], 'Missing replay guards')
    frozen_original(report['reference'], old['reference'])
    require(report['reference_quality'] == old['reference_quality'], 'Original quality changed')
    require(report['summary'] == summary(report['arms']), 'Summary contradicts measured arms')
    frozen_baseline(report['arms'][0], old)
    name, kind = PROFILES[report['profile']]
    recipe_path = 'tests/results/qwen35-ocr/' + (
        'ffn-calibration-parity.json' if report['profile'] == 'q4' else 'page-parity.json')
    require(report['recipe_source'] == recipe_path and report['recipe_report_sha256'] == sha(repo / recipe_path),
        'Preselected recipe source/hash changed')
    recipe = next(a for a in json.loads((repo / recipe_path).read_text())['arms'] if a['name'] == name)
    candidate = report['arms'][1]
    require(candidate['name'] == name and candidate['kind'] == kind
        and candidate['overrides'] == recipe['overrides'], 'Reported precision recipe changed')
    require(candidate['bytes'] == recipe['bytes'] and candidate['gguf_sha256'] == recipe['gguf_sha256'],
        'Frozen quantized file identity changed')
    before = {t['name']: dict(type={'float32': 'F32', 'float16': 'F16'}[t['dtype']],
        shape=t['shape'][::-1], sha256=t['sha256'],
        bytes=math.prod(t['shape']) * {'float32': 4, 'float16': 2}[t['dtype']])
        for t in report['weights']['tensor_audit']}
    tensor_audit(before, candidate['tensor_audit'], recipe['overrides'], kind)
    require(candidate['protected_tensor_bytes_exact'] and candidate['frozen_quantized_weights_exact'],
        'Missing quantized weight guards')
    return old, before


def verify_quantizer(log, before, after):
    count = len(before)
    rows = re.findall(r'^\[\s*(\d+)/(\d+)\] (\S+)\s+-\s+(\S+), (.*)$', log, re.M)
    require([int(r[0]) for r in rows] == list(range(1, count + 1))
        and all(int(r[1]) == count for r in rows)
        and [r[2] for r in rows] == list(after), 'Missing/reordered quantizer records')
    quantized = 0
    for _, _, name, source_type, operation in rows:
        require(source_type.upper() == before[name]['type'], 'Wrong quantizer source type: ' + name)
        match = re.search(r'quantizing to (\w+)\.\.\.', operation)
        if match:
            quantized += 1
            require(after[name]['type'] == match[1].upper(), 'Wrong quantizer target type: ' + name)
        else:
            require(after[name] == before[name], 'Copied payload changed: ' + name)
    totals = re.findall(r'^(\d+) quantized, (\d+) kept$', log, re.M)
    require(totals == [(str(quantized), str(count - quantized))], 'Wrong/repeated quantizer totals')
    return dict(quantized=quantized, kept=count - quantized)


def audit(report_path, logs, expected_source, repo=ROOT):
    require(report_path.stat().st_size <= 8 * 1024 * 1024, 'Oversized report')
    report = json.loads(report_path.read_text())
    old, before = verify_frozen(report, repo, expected_source)
    log_hashes = {}
    def read_log(name):
        path = logs / name
        require(path.stat().st_size <= 2 * 1024 * 1024, 'Oversized log: ' + name)
        log_hashes[name] = sha(path)
        return path.read_text()
    names = [m['stage'] for m in old['prefill']['metrics']]
    for arm in report['arms']:
        name = arm['name']
        prefill = read_log(name + '-prefill.log')
        decode = read_log(name + '-decode.log')
        require(strict_prefill(prefill, names) == arm['prefill'], 'Prefill log/report disagreement: ' + name)
        measured = add_confidences(decode, validate(parse(decode, report['reference']), report['reference'],
            old['specification']['max_new_tokens']), report['reference'])
        require(measured == arm['decode'], 'Decoded log/report disagreement: ' + name)
        require(teacher_quality(decode, len(report['reference']['generated_ids'])) == arm['teacher_quality'],
            'Teacher log/report disagreement: ' + name)
        quality = score(old['gold'], measured['native_text'],
            complete=measured['native_ids'][-1] in report['reference']['eos_ids'])
        require(quality == arm['native_quality'], 'Gold quality disagreement: ' + name)
    counts = verify_quantizer(read_log('quantizer.log'), before, report['arms'][1]['tensor_audit'])
    merger = lambda arm: next(m for m in arm['prefill']['metrics'] if m['stage'] == 'prefill.vision.merger')
    require(merger(report['arms'][0]) == merger(report['arms'][1]), 'Protected vision merger metrics changed')
    return dict(evidence_verified=True, source_commit=expected_source, case=report['case']['id'],
        profile=report['profile'], report_sha256=sha(report_path), auditor_sha256=sha(Path(__file__)),
        log_sha256=log_hashes, quantizer=counts, retained_failures=report['summary'],
        scope='Record/log/source/weight-manifest verification only; not new inference, model parity or promotion.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    try:
        result = audit(args.report, args.logs, args.source_commit)
    except (ValueError, KeyError, StopIteration, AssertionError, OSError) as error:
        parser.exit(1, 'Evidence audit failed: ' + (str(error) or type(error).__name__) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
