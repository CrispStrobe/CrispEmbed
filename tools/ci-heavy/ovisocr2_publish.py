"""Hosted conditional OvisOCR2 GGUF publication with empty-cache remote verification.

The preflight is stdlib-only and must finish before auth, packages, or weights.
This module never uploads original probes, NPZ files, or test-only weights.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'tests/results/qwen35-ocr'
MODEL = 'ATH-MaaS/OvisOCR2'
REVISION = 'cbdede82e048659b83310476ff807da825f8fe78'
DESTINATION = 'cstr/OvisOCR2-GGUF'
F16_SHA = 'd3a733a5d5ba5d63f5275500fad4cff585b82e2fb90d1def2fe767d04edf57db'
LICENSE_SHA = 'bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a'
UPSTREAM_README_SHA = 'f2510d6d02471c01a2808102da79cb47106963f18486f2ac05dd5c0b53fc1c7a'
RECIPES = {'f16': 'original-f16', 'q8': 'q8-protected', 'q4': 'q4-attention-q8',
           'q4-calibrated': 'q4-attention-q8'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(name):
    path = EVIDENCE / name
    if not path.is_file():
        raise ValueError('Required independently reviewed receipt absent: ' + name)
    return json.loads(path.read_text()), digest(path)


def producer(receipt):
    value = receipt.get('producer', receipt)
    if not isinstance(value, dict):
        raise ValueError('Receipt producer missing')
    return value


def positive(receipt, field):
    review = receipt.get('independent_review')
    if not isinstance(review, dict) or review.get(field) is not True:
        raise ValueError('Independent review is not positive: ' + field)
    return review


def sha_field(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def source_head(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{40}', value) is not None


def valid_cache_error(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= .002


def require_receipts(precision, destination=DESTINATION):
    """Pure gate. No Hugging Face import, token lookup, or weight access."""
    if precision not in RECIPES or destination != DESTINATION:
        raise ValueError('Unknown precision or unreviewed destination')
    gpu, gpu_sha = load('ovisocr2-vllm-reference.json')
    positive(gpu, 'genuine_gpu_forward_completed')
    gp = producer(gpu)
    worker = gp.get('worker', {})
    if (gp.get('status') != 'completed bounded original vLLM T4 reference'
            or worker.get('status') != 'completed bounded original vLLM reference'
            or worker.get('gpu_control', {}).get('passed') is not True
            or worker.get('runtime_stop_probe', {}).get('all_stop_ids') != [248044, 248046]):
        raise ValueError('Original GPU inference receipt differs')
    model, model_sha = load('ovisocr2-model-audit.json')
    positive(model, 'f16_positive')
    mp = producer(model)
    model_artifact = model.get('artifact', {})
    reference_artifact = model.get('reference_artifact', {})
    model_review = model['independent_review']
    canonical_producer_sha = hashlib.sha256(json.dumps(
        mp, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()
    cache_error = mp.get('original_capture', {}).get('cached_vs_full_prefix_max_abs')
    model_run = model.get('run')
    f16 = mp.get('exports', {}).get('f16', {})
    if (type(model_run) is not int or model_run < 1 or not source_head(mp.get('source_commit'))
            or mp.get('status') != 'completed diagnostic audit' or mp.get('failures') != []
            or type(model_artifact.get('id')) is not int
            or not sha_field(model_artifact.get('zip_sha256'))
            or model_artifact.get('name') not in (None, f'ovisocr2-model-reports-{model_run}')
            or model_review.get('artifact_zip_sha256') != model_artifact['zip_sha256']
            or model_review.get('producer_canonical_sha256') != canonical_producer_sha
            or type(reference_artifact.get('id')) is not int
            or not sha_field(reference_artifact.get('zip_sha256'))
            or reference_artifact.get('name') != f'ovisocr2-model-reference-{model_run}'
            or model_review.get('reference_artifact_zip_sha256') != reference_artifact['zip_sha256']
            or f16.get('model_sha256') != F16_SHA
            or mp.get('original_capture', {}).get('cached_vs_full_prefix_same_argmax') is not True
            or not valid_cache_error(cache_error)):
        raise ValueError('Original/native F16 source or cache receipt differs')
    public, public_sha = load('ovisocr2-public-surfaces.json')
    positive(public, 'public_positive')
    pp = producer(public)
    if (pp.get('status') != 'completed public audit' or pp.get('failures') != []
            or not source_head(pp.get('source_commit'))
            or pp.get('model_prerequisite', {}).get('receipt_sha256') != model_sha
            or pp.get('artifacts', {}).get('model', {}).get('sha256') != F16_SHA
            or pp.get('public_confidence_comparison', {}).get('passed') is not True
            or pp.get('abi_layout', {}).get('passed') is not True):
        raise ValueError('Model-backed public surface review/source differs')
    decision, decision_sha = load('ovisocr2-quant-attempts.json')
    decision_review = positive(decision, 'selection_positive')
    attempts = decision.get('attempts', {})
    if (decision_review.get('all_attempts_completed') is not True
            or decision.get('selected_precision') != precision
            or set(attempts) != {'q8', 'q4', 'q4-calibrated'}):
        raise ValueError('All quantization attempts and selection review are required')
    for name, attempt in attempts.items():
        status = attempt.get('status')
        if (status not in ('qualified', 'rejected')
                or not sha_field(attempt.get('report_sha256'))
                or type(attempt.get('run')) is not int or attempt['run'] < 1
                or (status == 'rejected' and not attempt.get('reason'))):
            raise ValueError('Quantization attempt outcome/evidence is incomplete: ' + name)
    quality, quality_sha = load(f'ovisocr2-{precision}-quality.json')
    positive(quality, 'quality_positive')
    qp = producer(quality)
    candidate_artifact = quality.get('candidate_artifact', {})
    original_quality, original_quality_sha = load('ovisocr2-original-quality.json')
    if (original_quality.get('independent_review', {}).get('original_11_complete') is not True
            or not sha_field(original_quality.get('report_sha256'))):
        raise ValueError('Reviewed original 11-case quality baseline differs')
    expected_status = ('completed F16 quality observations; independent review pending'
                       if precision == 'f16' else
                       'completed quantized quality observations; independent review pending')
    candidate = qp.get('models', {}).get(precision, {})
    if (qp.get('status') != expected_status or qp.get('requested_phase') != precision
            or not source_head(qp.get('source_commit'))
            or qp.get('prerequisite', {}).get('baseline_sha256') != model_sha
            or not sha_field(candidate.get('sha256')) or not isinstance(candidate.get('bytes'), int)
            or candidate['bytes'] < 200_000_000 or (precision == 'f16' and candidate['sha256'] != F16_SHA)
            or (precision != 'f16' and candidate.get('recipe') != RECIPES[precision])
            or qp.get('prerequisite', {}).get('original_review_sha256') != original_quality_sha
            or 'original' not in qp.get('scores', {}) or 'f16' not in qp.get('scores', {})
            or precision not in qp['scores'] or qp.get('cabi_smoke', {}).get(precision) is None):
        raise ValueError('Reviewed quality phase/model/recipe differs')
    if precision != 'f16' and (type(candidate_artifact.get('id')) is not int
                               or not sha_field(candidate_artifact.get('zip_sha256'))
                               or quality.get('independent_review', {}).get('candidate_artifact_zip_sha256')
                               != candidate_artifact['zip_sha256']):
        raise ValueError('Reviewed retained quantized candidate artifact absent')
    if (qp.get('f16_paired_synthetic', {}).get('screen_passed') is not True
            or qp.get('f16_paired_documents', {}).get('screen_passed') is not True):
        raise ValueError('F16 quality screens did not pass')
    calibration_sha = None
    cp = {}
    if precision == 'q4-calibrated':
        calibration, calibration_sha = load('ovisocr2-q4-calibrated-calibration.json')
        positive(calibration, 'calibration_positive')
        cp = producer(calibration)
        if (cp.get('precision') != precision or cp.get('candidate_sha256') != candidate['sha256']
                or cp.get('recipe') != RECIPES[precision]
                or cp.get('quality_receipt_sha256') != quality_sha
                or cp.get('selection_frozen_before_heldout') is not True
                or cp.get('heldout_not_used_for_recipe') is not True):
            raise ValueError('Independent development-only calibration review differs')
    if precision != 'f16':
        audit = qp.get('tensor_audit', {})
        numerical = qp.get('quant_numerical', {})
        cases = qp.get('native_cases', {}).get(precision, {})
        if (quality.get('independent_review', {}).get('numerical_positive') is not True
                or audit.get('recipe') != RECIPES[precision]
                or audit.get('tensor_count') != 473
                or audit.get('protected_byte_exact') is not True
                or not isinstance(audit.get('changed_matrix_count'), int)
                or audit['changed_matrix_count'] < 1
                or numerical.get('passed') is not True
                or numerical.get('exit_code') != 0
                or numerical.get('comparison_coverage') is not True
                or numerical.get('first_failure') is not None
                or numerical.get('source_reference_sha256')
                   != mp.get('reference_artifacts', {}).get('gguf', {}).get('sha256')
                or not sha_field(numerical.get('log_sha256'))
                or not isinstance(numerical.get('metric_lines'), list)
                or not numerical['metric_lines']
                or len(cases) != 11):
            raise ValueError('Quantized protected tensors or native numeric evidence differ')
        for case in cases.values():
            ids, scores, cap = (case.get('token_ids'), case.get('confidences'), case.get('cap'))
            if (not isinstance(ids, list) or not isinstance(scores, list)
                    or not isinstance(cap, int) or not 0 < len(ids) <= cap
                    or len(scores) != len(ids)
                    or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                           or value < 0 or value > 1 for value in scores)):
                raise ValueError('Selected-token numeric confidence evidence incomplete')
        if (qp.get('paired_synthetic', {}).get('screen_passed') is not True
                or any(row.get('screen_passed') is not True
                       for row in qp.get('paired_documents', {}).values())
                or set(qp.get('paired_documents', {})) != {'original', 'f16'}):
            raise ValueError('Quantized heldout/doc quality screens did not pass')
    if precision == 'q4-calibrated':
        calibration_record = qp.get('calibration', {})
        panel = json.loads((EVIDENCE/'ovisocr2-quality-panel.json').read_text())
        development_ids = [row['id'] for row in panel['cases'] if row['split'] == 'development']
        heldout_ids = {row['id'] for row in panel['cases'] if row['split'] == 'heldout'}
        if (len(calibration_record.get('vectors', [])) != 72
                or not sha_field(calibration_record.get('imatrix_sha256'))
                or not sha_field(calibration_record.get('stats_sha256'))
                or calibration_record.get('input_ids') != development_ids
                or set(development_ids) & heldout_ids
                or cp.get('imatrix_sha256') != calibration_record['imatrix_sha256']):
            raise ValueError('Calibrated Q4 lacks exact development-only importance evidence')
    if precision in ('q4', 'q4-calibrated'):
        q8, q8_sha = load('ovisocr2-q8-quality.json')
        if (q8.get('independent_review', {}).get('q8_diagnostic_complete') is not True
                or q8.get('independent_review', {}).get('allow_q4_diagnostic') is not True
                or attempts['q8'].get('report_sha256') != q8_sha):
            raise ValueError('Q4 lacks exact reviewed Q8 diagnostic permission')
    if precision != 'f16' and (attempts[precision].get('status') != 'qualified'
                               or attempts[precision].get('candidate_sha256') != candidate['sha256']
                               or attempts[precision].get('report_sha256') != quality_sha):
        raise ValueError('Selected quant recipe was failed, unqualified, or points to another report')
    return {'precision': precision, 'recipe': RECIPES[precision],
            'candidate_sha256': candidate['sha256'], 'candidate_bytes': candidate['bytes'],
            'model_run_id': mp.get('service_run', {}).get('id') or
                            model.get('run') or model.get('model_run_id'),
            'model_report_artifact_id': model_artifact['id'],
            'model_report_artifact_zip_sha256': model_artifact['zip_sha256'],
            'model_reference_artifact_id': reference_artifact['id'],
            'model_reference_artifact_zip_sha256': reference_artifact['zip_sha256'],
            'model_producer_canonical_sha256': canonical_producer_sha,
            'model_source_commit': mp['source_commit'], 'public_source_commit': pp['source_commit'],
            'quality_source_commit': qp['source_commit'],
            'quality_run_id': quality.get('run') or qp.get('run'),
            'candidate_artifact_id': candidate_artifact.get('id'),
            'candidate_artifact_zip_sha256': candidate_artifact.get('zip_sha256'),
            'receipt_sha256': {'gpu': gpu_sha, 'model': model_sha, 'public': public_sha,
                               'quality': quality_sha, 'original_quality': original_quality_sha,
                               'quant_attempts': decision_sha,
                               **({'calibration': calibration_sha} if calibration_sha else {})},
            'destination': destination}


def service_run(run_id, head, workflow, call=subprocess.check_output):
    if not str(run_id).isdigit() or not int(run_id):
        raise ValueError('Exact hosted artifact run ID required')
    value = json.loads(call(['gh', 'api', f'repos/CrispStrobe/CrispEmbed/actions/runs/{run_id}']))
    if (value.get('status') != 'completed' or value.get('head_sha') != head or
            value.get('path') != workflow or value.get('conclusion') != 'success'):
        raise ValueError('Retained artifact service run differs')
    return {key: value[key] for key in ('id', 'status', 'conclusion', 'head_sha', 'path')}


def service_artifact(artifact_id, run_id, expected_name, expected_zip_sha,
                     call=subprocess.check_output):
    if type(artifact_id) is not int or artifact_id < 1 or not sha_field(expected_zip_sha):
        raise ValueError('Reviewed artifact identity absent')
    value = json.loads(call(['gh', 'api',
        f'repos/CrispStrobe/CrispEmbed/actions/artifacts/{artifact_id}']))
    if (value.get('id') != artifact_id or value.get('name') != expected_name
            or value.get('expired') is not False
            or value.get('workflow_run', {}).get('id') != int(run_id)
            or value.get('digest') != 'sha256:' + expected_zip_sha):
        raise ValueError('Retained artifact service digest/name/run differs')
    return {'id': artifact_id, 'name': expected_name, 'zip_sha256': expected_zip_sha}


def find_artifact(folder, name, expected_sha, expected_bytes):
    matches = list(Path(folder).rglob(name))
    if (len(matches) != 1 or matches[0].stat().st_size != expected_bytes
            or digest(matches[0]) != expected_sha):
        raise ValueError('Selected retained GGUF missing or exact SHA/bytes differ')
    return matches[0]


def card(pins, filename):
    template = (ROOT/'models/cards/ovisocr2-card.template.md').read_text()
    values = {'source_model': MODEL, 'source_revision': REVISION,
              'audited_source_commit': pins['public_source_commit'],
              'filename': filename, 'model_sha256': pins['candidate_sha256'],
              'model_bytes': str(pins['candidate_bytes']), 'precision': pins['precision'],
              'recipe': pins['recipe'],
              'quant_gate': ('CRISPEMBED_QWEN35_QUANT=1 ' if pins['precision'] != 'f16' else ''),
              'calibration_note': ('This candidate used 72 development-only FFN importance vectors; '
                                   'the exact imatrix and heldout separation have an independent review.'
                                   if pins['precision'] == 'q4-calibrated' else
                                   'This candidate uses no importance-matrix calibration.')}
    for key, value in values.items():
        template = template.replace('${'+key+'}', str(value))
    if '${' in template:
        raise ValueError('Unfilled model card field')
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', required=True, choices=tuple(RECIPES))
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--model-run-id', required=True)
    parser.add_argument('--quality-run-id', required=True)
    parser.add_argument('--artifact-dir', type=Path)
    parser.add_argument('--license-dir', type=Path)
    parser.add_argument('--destination', default=DESTINATION)
    args = parser.parse_args()
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH'])/'ovis-publish'
    out.mkdir(parents=True, exist_ok=True)
    report_path = out/'ovisocr2-publication.json'
    report = {'status': 'started', 'phase': 'preflight', 'source_commit': os.environ.get('GITHUB_SHA'),
              'source_sha256': {name: digest(ROOT/name) for name in (
                  'tools/ci-heavy/ovisocr2_publish.py', 'models/cards/ovisocr2-card.template.md',
                  '.github/workflows/ovisocr2-publish.yml')},
              'destination': args.destination}
    def save(): report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    try:
        pins = require_receipts(args.precision, args.destination)
        report['qualification'] = pins
        report['service_runs'] = {
            'model': service_run(args.model_run_id, pins['model_source_commit'],
                                 '.github/workflows/ovisocr2-metadata.yml'),
            'quality': service_run(args.quality_run_id, pins['quality_source_commit'],
                                   '.github/workflows/ovisocr2-quality.yml')}
        report['service_artifacts'] = {
            'model_reports': service_artifact(
                pins['model_report_artifact_id'], args.model_run_id,
                f'ovisocr2-model-reports-{args.model_run_id}',
                pins['model_report_artifact_zip_sha256']),
            'model_reference': service_artifact(
                pins['model_reference_artifact_id'], args.model_run_id,
                f'ovisocr2-model-reference-{args.model_run_id}',
                pins['model_reference_artifact_zip_sha256'])}
        if args.precision != 'f16':
            report['service_artifacts']['candidate'] = service_artifact(
                pins['candidate_artifact_id'], args.quality_run_id,
                f'ovisocr2-quality-candidate-{args.quality_run_id}',
                pins['candidate_artifact_zip_sha256'])
        if pins['model_run_id'] is not None and int(args.model_run_id) != int(pins['model_run_id']):
            raise ValueError('Model artifact run ID differs from reviewed receipt')
        if pins['quality_run_id'] is not None and int(args.quality_run_id) != int(pins['quality_run_id']):
            raise ValueError('Quality artifact run ID differs from reviewed receipt')
        report['phase'] = 'preflight-complete'
        save()
        if args.preflight:
            report['status'] = 'preflight passed; no auth/packages/weights accessed'
            save()
            return
        if not args.artifact_dir or not args.license_dir:
            raise ValueError('Retained candidate and pinned license artifact directories required')
        scratch.mkdir(parents=True, exist_ok=True)
        filename = f'ovisocr2-{args.precision}.gguf'
        model = find_artifact(args.artifact_dir, filename, pins['candidate_sha256'], pins['candidate_bytes'])
        license_candidates = list(args.license_dir.rglob('LICENSE'))
        if len(license_candidates) != 1 or digest(license_candidates[0]) != LICENSE_SHA:
            raise ValueError('Pinned original Apache-2.0 LICENSE missing or altered')
        readme_candidates = list(args.license_dir.rglob('UPSTREAM_README.md'))
        if len(readme_candidates) != 1 or digest(readme_candidates[0]) != UPSTREAM_README_SHA:
            raise ValueError('Pinned upstream card provenance missing or altered')
        manifest = {'source_model': MODEL, 'source_revision': REVISION,
                    'audited_source_commit': pins['public_source_commit'],
                    'model_audit_source_commit': pins['model_source_commit'],
                    'model_producer_canonical_sha256': pins['model_producer_canonical_sha256'],
                    'quality_source_commit': pins['quality_source_commit'],
                    'publisher_source_commit': os.environ.get('GITHUB_SHA'),
                    'publication_workflow_run': os.environ.get('GITHUB_RUN_ID'),
                    'candidate': {'filename': filename, 'sha256': pins['candidate_sha256'],
                                  'bytes': pins['candidate_bytes'], 'precision': args.precision,
                                  'recipe': pins['recipe']},
                    'license_sha256': LICENSE_SHA,
                    'upstream_readme_sha256': UPSTREAM_README_SHA,
                    'receipts': pins['receipt_sha256'],
                    'limits': ['CrispEmbed experimental format-2 reader; no stock llama.cpp claim',
                               'No runtime model alias or release tag is activated by this upload',
                               'Bounded OCR panels do not establish broad OCR quality']}
        report['phase'] = 'exact-artifact-verified'
        save()
        card_bytes = card(pins, filename).encode()
        manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
        (out/'README.md').write_bytes(card_bytes)
        (out/'manifest.json').write_bytes(manifest_bytes)
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
        api = HfApi(token=os.environ['HF_TOKEN'])
        required_remote = {filename, 'LICENSE', 'UPSTREAM_README.md', 'README.md', 'manifest.json'}
        # Hub may create .gitattributes to route the large GGUF through LFS.
        allowed_remote = required_remote | {'.gitattributes'}
        if api.repo_exists(args.destination):
            current = api.model_info(args.destination, files_metadata=True)
            if {item.rfilename for item in current.siblings} - allowed_remote:
                raise ValueError('Destination contains files outside the reviewed publication allowlist')
        else:
            api.create_repo(args.destination, private=False, exist_ok=False)
        operations = [CommitOperationAdd(path_in_repo=filename, path_or_fileobj=str(model)),
                      CommitOperationAdd(path_in_repo='LICENSE', path_or_fileobj=str(license_candidates[0])),
                      CommitOperationAdd(path_in_repo='UPSTREAM_README.md',
                                         path_or_fileobj=str(readme_candidates[0])),
                      CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=card_bytes),
                      CommitOperationAdd(path_in_repo='manifest.json', path_or_fileobj=manifest_bytes)]
        commit = api.create_commit(repo_id=args.destination, operations=operations,
                                   commit_message='Publish independently qualified OvisOCR2 '+args.precision+' GGUF')
        report['phase'] = 'immutable-remote-verification'
        report['published_commit'] = commit.oid
        save()
        info = api.model_info(args.destination, revision=commit.oid, files_metadata=True)
        sibling = next((x for x in info.siblings if x.rfilename == filename), None)
        if (not sibling or sibling.size != pins['candidate_bytes'] or
                not sibling.lfs or sibling.lfs.sha256 != pins['candidate_sha256']):
            raise ValueError('Immutable remote LFS SHA/size differs')
        remote_names = {item.rfilename for item in info.siblings}
        if not required_remote <= remote_names or remote_names - allowed_remote:
            raise ValueError('Published model repository file allowlist differs')
        empty_cache = scratch/'fresh-hf-cache'
        if empty_cache.exists():
            raise ValueError('Remote verification cache is not empty')
        empty_cache.mkdir()
        fetched = Path(hf_hub_download(repo_id=args.destination, filename=filename,
                                       revision=commit.oid, cache_dir=empty_cache,
                                       local_files_only=False, force_download=True, token=os.environ['HF_TOKEN']))
        if fetched.stat().st_size != pins['candidate_bytes'] or digest(fetched) != pins['candidate_sha256']:
            raise ValueError('Empty-cache immutable remote download differs')
        expected_small = {'LICENSE': LICENSE_SHA, 'UPSTREAM_README.md': UPSTREAM_README_SHA,
                          'README.md': hashlib.sha256(card_bytes).hexdigest(),
                          'manifest.json': hashlib.sha256(manifest_bytes).hexdigest()}
        verified_small = {}
        for name, expected_sha in expected_small.items():
            path = Path(hf_hub_download(repo_id=args.destination, filename=name,
                                        revision=commit.oid, cache_dir=empty_cache,
                                        local_files_only=False, force_download=True,
                                        token=os.environ['HF_TOKEN']))
            if digest(path) != expected_sha:
                raise ValueError('Immutable remote small file differs: ' + name)
            verified_small[name] = expected_sha
        report['status'] = 'published and immutable remote file reverified'
        report['remote'] = {'repo': args.destination, 'commit': commit.oid,
                            'filename': filename, 'sha256': digest(fetched),
                            'bytes': fetched.stat().st_size, 'small_files_sha256': verified_small}
        save()
    except BaseException as exc:
        report['status'] = 'failed publication or preflight'
        report['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise


if __name__ == '__main__':
    main()
