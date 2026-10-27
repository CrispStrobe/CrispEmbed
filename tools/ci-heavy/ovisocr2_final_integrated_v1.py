"""Prepared final-source Ovis validation; no historical guard overrides."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(Path(__file__).parent)]
from ovisocr2_final_integrated_v1 import checked_protocol, checked_binaries, source_guard, compare_quality, sha
from successor_assets import api, artifact
import ovisocr2_large_inputs as old
import ovisocr2_quality_run as quality

SPEC = ROOT / 'docs/ovisocr2-final-integrated-v1.json'


def save(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    # Only the new report's task paths are publicized; retained raw references
    # and their byte identities are never rewritten.
    for variable, marker in (('HEAVY_SCRATCH', '<scratch>'), ('HEAVY_OUT', '<out>'), ('AUDIT_BIN_DIR', '<bin>')):
        prefix = os.environ.get(variable)
        if prefix and Path(prefix).is_absolute(): text = text.replace(prefix, marker)
    path.write_text(text.replace(str(ROOT), '<workspace>'), encoding='utf-8')


def save_rejection(out, phase, gate, error):
    out.mkdir(parents=True, exist_ok=True)
    suffix = '-gate' + str(gate) if gate in (0, 1) else ''
    path = out / ('final-rejection-' + phase + suffix + '.json')
    save(path, {'phase': phase, 'gate': gate, 'source_commit': os.environ.get('GITHUB_SHA'),
                'error': str(error), 'qualification': False})
    return path


def preflight(out):
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise ValueError('Hosted validation only')
    protocol = checked_protocol(json.loads(SPEC.read_text()))  # Pending reviews reject before any download.
    closure = source_guard(ROOT, protocol)
    if closure['source_head'] != os.environ['GITHUB_SHA']:
        raise ValueError('Service/checkout source differs')
    auditors = ('tools/ovisocr2_final_integrated_v1.py', 'tools/ci-heavy/ovisocr2_final_integrated_v1.py',
                'docs/ovisocr2-final-integrated-v1.json', '.github/workflows/ovisocr2-final-integrated-v1.yml')
    for path in auditors:
        if subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Dirty final validation protocol/helper: ' + path)
    out.mkdir(parents=True, exist_ok=True)
    proofs = {}
    for name, dependency in protocol['dependencies'].items():
        run = api('actions/runs/' + str(dependency['run']))
        if (run['head_sha'] != dependency['source'] or run['path'] != dependency['workflow']
                or run['event'] != 'workflow_dispatch' or run['status'] != 'completed'
                or run['conclusion'] != dependency['conclusion']):
            raise ValueError('Reviewed completion service differs: ' + name)
        records = []
        for index, pin in enumerate(dependency['artifacts']):
            path = out / (name + '-' + str(index) + '.zip')
            if path.exists():
                path.unlink()  # Only this task's previously downloaded small review transport.
            identity = artifact(dependency['run'], pin['name'], path, 8 * 1024**2)
            if (identity['id'], identity['size_in_bytes'], identity['digest']) != (pin['id'], pin['bytes'], 'sha256:' + pin['sha256']):
                raise ValueError('Reviewed completion artifact differs: ' + name)
            with zipfile.ZipFile(path) as package:
                if len(package.namelist()) != len(set(package.namelist())):
                    raise ValueError('Duplicate dependency ZIP members')
                for member, digest in pin['reports'].items():
                    raw = package.read(member)
                    import hashlib
                    if hashlib.sha256(raw).hexdigest() != digest:
                        raise ValueError('Reviewed completion report differs: ' + member)
                    json.loads(raw)
            records.append(identity)
        proofs[name] = {'reviewed_outcome': dependency['reviewed_outcome'],
                        'experimental_panel_completed': True,
                        'experimental_candidate_qualified': dependency['reviewed_outcome'] == 'candidate-supported',
                        'candidate_enabled_by_default': False, 'release_qualified': False,
                        'service': {k: run[k] for k in ('id', 'head_sha', 'path', 'conclusion')},
                        'artifacts': records, 'pins': dependency['artifacts']}
    result = {'schema': 'crispembed-ovisocr2-final-integrated-source-v1', 'closure': closure,
              'protocol_sha256': sha(SPEC), 'reviewed_dependencies': proofs,
              'auditor_source_sha256': {p: sha(ROOT / p) for p in auditors},
              'qualification': False, 'model_results': 'not established by source/dependency proof'}
    save(out / 'final-source-contract.json', result)
    return result


def frozen(out, scratch, build, gate):
    proof = preflight(out)
    scratch.mkdir(parents=True, exist_ok=True)
    manifest = checked_binaries(json.loads((build / 'binaries.json').read_text()), os.environ['GITHUB_SHA'], sha(SPEC))
    for name, digest in manifest['binary_sha256'].items():
        if sha(build / name) != digest:
            raise ValueError('Final native binary bytes differ')
        (build / name).chmod(0o700)
    r = {'schema': 'crispembed-ovisocr2-final-integrated-frozen-v1', 'gate': gate,
         'source_commit': os.environ['GITHUB_SHA'], 'source_proof': proof, 'build': manifest,
         'commands': {}, 'arms': {}, 'owning_cases': {}, 'failures': [],
         'status': 'started', 'qualification': False,
         'limits': ['Frozen943 source fields/14 programs only; no larger-grid qualification.',
                    'Original references, calibrated budgets and raw strict flags unchanged.']}
    def write(): save(out / 'frozen-result.json', r)
    def command(name, args, env=None):
        ok = old.timed_command(name, args, out, r, env, timeout=7200)
        write()
        return ok
    write()
    try:
        if not command('bounds-controls', [build / 'test-ovisocr2-image-bounds']):
            raise ValueError('Actual bounds/recovery control failed')
        if shutil.disk_usage(scratch).free < 24 * 1024**3:
            raise ValueError('At least24GiB hosted scratch required before model transfer')
        source = old.source_and_exports(out, scratch, r)
        reference = old.replay.source_reference(old.ab, scratch, out, r, source)
        ref = scratch / 'official-delta.gguf'; image = scratch / 'owned-readable.png'
        r['reference_sha256'] = sha(ref)
        baseline = json.loads((ROOT / 'tests/results/qwen35-ocr/ovisocr2-larger-inputs.json').read_text())['cases']['frozen-control']
        producer_path = ROOT / baseline['producer_report_git_path']
        if sha(producer_path) != baseline['producer_report_sha256']:
            raise ValueError('Frozen control producer changed')
        before = json.loads(producer_path.read_text())
        env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')}
        known = set(old.audit.NATIVE_ENV) | {'CRISPEMBED_QWEN35_VISION_PRECISE_NORM',
            'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN', 'CRISPEMBED_QWEN35_VISION_CASCADE_NORM',
            'CRISPEMBED_OVISOCR2_LARGE_IMAGES'}
        env.update({k:v for k,v in old.native_environment('frozen-control').items() if k in known})
        env.update({k: '0' for k in quality.DIFF_FLOAT_ZERO})
        env.update(CRISPEMBED_QWEN35_WEIGHT_CACHE='0', CRISPEMBED_WASM_Q8_0_ROUND='0',
                   CRISPEMBED_QWEN35_VISION_SOURCE_INTERPOLATION=str(gate))
        r['native_gates'] = {k:v for k,v in env.items() if k.startswith('CRISPEMBED_')}
        for precision, (filename, _, _) in old.ab.MODELS.items():
            for label, args in old.audit.native_arms(precision, build, scratch / filename, image, ref).items():
                folder = scratch / 'native-diff' / label; folder.mkdir(parents=True, exist_ok=True)
                passed = command(label, args, dict(env, CRISPEMBED_DIFF_DUMP_DIR=str(folder)))
                log = (out / (label + '.log')).read_text()
                try: raw = old.raw_manifest(folder, reference, log, label)
                except (ValueError, KeyError) as error: raw = {'coverage_complete': False, 'error': str(error)}
                if label.endswith('-context') and 'OVIS_CONTEXT' not in log: passed = False
                record = {'passed': passed and raw['coverage_complete'], 'raw': raw}
                if gate == 0 and not label.endswith('-context') and raw['coverage_complete']:
                    def hashes(value): return {(x['stage'], x.get('variant', 'primary')): x['sha256'] for x in value['files']}
                    exact = hashes(raw) == hashes(before['arms'][label]['raw'])
                    record['retained_ordinary_vectors_bit_exact'] = exact
                    record['passed'] = record['passed'] and exact
                r['arms'][label] = record
                if not record['passed']: r['failures'].append(label)
                write()
            label = precision + '-owning-ocr'
            passed = command(label, [build / 'test-ovisocr2-quality-run', scratch / filename, image, '32', '2'], env)
            lines = [x.removeprefix('OVIS_QUALITY_RESULT ') for x in (out / (label + '.log')).read_text().splitlines() if x.startswith('OVIS_QUALITY_RESULT ')]
            if len(lines) != 1:
                r['failures'].append(label); write(); continue
            actual = json.loads(lines[0]); text = bytes.fromhex(actual['raw_text_utf8_hex']).decode()
            exact = (actual['token_ids'] == reference['generated_ids'][0] and text == reference['decoded'][0]
                     and actual['stop_kind'] == 1 and actual['stop_id'] == reference['generated_stop_id'])
            r['owning_cases'][precision] = {**actual, 'raw_text': text, 'original_ids_text_eos_exact': exact}
            if not passed or not exact: r['failures'].append(label)
            write()
        if len(r['arms']) != 14 or len(r['owning_cases']) != 2:
            raise ValueError('Incomplete frozen14/owning coverage')
        r['scope_passed'] = not r['failures']; r['status'] = 'completed' if r['scope_passed'] else 'rejected'
        write()
        if not r['scope_passed']: raise ValueError('Final frozen numerical/decoded controls failed')
    except Exception as error:
        r['error'] = str(error); write(); raise


def quality_check(out, gate):
    proof = preflight(out)
    current = json.loads((out / 'ovisocr2-quality.json').read_text())
    if (current.get('source_commit') != os.environ['GITHUB_SHA']
            or current.get('status') != 'completed F16 quality observations; independent review pending'
            or current.get('models', {}).get('f16', {}).get('sha256') != quality.F16_SHA
            or 'f16' not in current.get('cabi_smoke', {})):
        raise ValueError('Completed current-source F16 producer/model/C ABI evidence absent')
    first = json.loads((ROOT / 'tests/results/qwen35-ocr/ovisocr2-f16-quality.json').read_text())['producer']
    result = compare_quality(current, first, gate)
    cabi = current['cabi_smoke']['f16']; native = current['native_cases']['f16']['dev-hello']
    import math
    values = cabi.get('confidences', [])
    if (cabi.get('raw_text') != native['raw_text'] or len(values) != len(native['token_ids'])
            or any(type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
            or type(cabi.get('mean_confidence')) not in (int,float)
            or not math.isfinite(cabi['mean_confidence'])
            or abs(cabi['mean_confidence'] - sum(values) / len(values)) > 1e-6):
        raise ValueError('Actual C ABI text/selected-confidence smoke differs')
    result['cabi_text_and_confidence_contract_passed'] = True
    if result['token_count'] != 1841: raise ValueError('Frozen first-F16 total token count differs')
    result.update(schema='crispembed-ovisocr2-final-integrated-quality-v1', source_proof=proof,
                  source_commit=os.environ['GITHUB_SHA'], producer_report_sha256=sha(out / 'ovisocr2-quality.json'))
    save(out / 'quality-comparison.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=('preflight', 'frozen', 'quality-check'), required=True)
    p.add_argument('--gate', type=int, choices=(0, 1)); args = p.parse_args()
    if args.phase != 'preflight' and args.gate is None:
        p.error('--gate is required for frozen/quality-check')
    out = Path(os.environ['HEAVY_OUT'])
    try:
        if args.phase == 'preflight': preflight(out)
        elif args.phase == 'quality-check': quality_check(out, args.gate)
        else: frozen(out, Path(os.environ['HEAVY_SCRATCH']), Path(os.environ['AUDIT_BIN_DIR']), args.gate)
    except Exception as error:
        save_rejection(out, args.phase, args.gate, error)
        raise
