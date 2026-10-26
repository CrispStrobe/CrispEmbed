"""Hosted frozen-input RoPE attribution; no checkpoint or full-model execution."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from ultrabert_contract import MODEL, REVISION, profile, sha256, validate_sources
from ultrabert_native import read_frozen_manifest

ORIGINAL_RUN = 38997562520
ORIGINAL_HEAD = '0573f1847db68914052a98341a22fceeb1358518'
COPY_RUN = 39913800444
COPY_HEAD = '07864ab26a8e45d0c606dc25d352f100156d64b4'
ARTIFACT = (12483912080, 'ultrabert-original-38997562520-preserved-39913800444', 511514340,
            '9fe5627ad89c4dede1e67bf69ce3689c773fadc0ff9d272bcd9ae95320b9d3c1')
GGML = '890278a8342c620197c90e702e1188bcab94f510'
PINS = {
    'tests/results/ultrabert/reference-audit.json': 'a3c16863b1a78ebaebcc2d50e24555b696e691d3309ee17b75075c0051032a36',
    'tests/results/ultrabert/reference-preservation.json': 'eeffa6df624d68a6f5f325866a59bfa3c3875d614e08f8c56a5e8ec8001ea1f7',
    'tests/results/ultrabert/source-audit.json': '582c99998d5f20bd1f05b22493325e1f74e8e9f9d8b3dbba0aab83bfa09ab50b',
    'src/ultrabert_embed.cpp': '1bc4b04421234373b2568b12cb0de89055231596888795f34e7ef199d70bbd18',
    'src/ultrabert_profile.h': '1ad434a05dc2261dda652f0f1fca5905ba6ee80435d2f4b2184ee0b252004aa7',
    'tests/test_ultrabert_diff.cpp': '7ef62cb263c8980bfa78d8df0d716bed0d316da8f7899813069bd4527b72064b',
    'src/crispembed_diff.h': '6bef0b4e32b52af7fe6cd6702bfc937845dde85762477b08a82ff2a68388df94',
    'tools/ultrabert_contract.py': '16b7f9c93017f4be97b367fa5aec33877a8ff053dfdc12771ec71109a311c9d8',
    'tools/ci-heavy/ultrabert_native.py': '4882f2fb4104286f3034f9e8b2afa34a7183c7d5f157fde00032de4eab393c9c',
}
FILES = ('tests/test_ultrabert_rope.cpp', 'tools/dump_ultrabert_rope_reference.py',
         'tools/ci-heavy/ultrabert_rope.py', 'tests/test_ultrabert_rope.py',
         '.github/workflows/ultrabert-rope.yml', 'CMakeLists.txt', 'src/crispembed_diff.h',
         'tools/ultrabert_contract.py', 'tools/ci-heavy/ultrabert_native.py')
CASES = {'empty', 'document', 'query', 'unicode', 'pair', 'window-boundary', 'padded-batch'}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, timeout=120).strip()


def preflight():
    actual = {path: sha256(ROOT / path) for path in PINS}
    if actual != PINS:
        raise ValueError('Frozen receipts or scoped native UltraBERT source changed')
    if (git('rev-parse', 'HEAD:ggml') != GGML or git('-C', str(ROOT / 'ggml'), 'rev-parse', 'HEAD') != GGML
            or git('-C', str(ROOT / 'ggml'), 'status', '--porcelain')
            or git('diff', '--name-only', 'HEAD', '--', 'src', 'ggml')
            or git('ls-files', '--others', '--exclude-standard', '--', 'src')):
        raise ValueError('Actual native source/ggml checkout differs')
    return dict(head=git('rev-parse', 'HEAD'), native_pins=actual, ggml=GGML,
                files={path: sha256(ROOT / path) for path in FILES})


def expected_metrics():
    result = set()
    for i in range(24):
        for scope in ('frozen', 'current'):
            result |= {f'normal_{scope}/layer_{i}.{kind}_{scope}' for kind in ('basis', 'q', 'k')}
        result |= {f'source_frozen/layer_{i}.{kind}_frozen' for kind in ('q', 'k')}
    return result


def parse_log(text):
    metrics, controls, totals = {}, {}, None
    for line in text.splitlines():
        fields = line.split('\t')
        if fields[0] == 'ROPE':
            if len(fields) != 11:
                raise ValueError('Malformed RoPE comparison')
            key = fields[1] + '/' + fields[2]
            numbers = [float(value) for value in fields[3:9]]
            if (key in metrics or key not in expected_metrics() or not all(math.isfinite(v) for v in numbers)
                    or any(v < 0 for v in numbers[1:]) or fields[9] not in ('0', '1') or fields[10] not in ('0', '1')):
                raise ValueError('Invalid/duplicate/nonfinite RoPE comparison')
            passed = numbers[0] >= .99999 and numbers[1] <= .002
            if bool(int(fields[9])) != passed or (fields[10] == '1' and numbers[1] != 0):
                raise ValueError('RoPE comparison verdict differs')
            metrics[key] = dict(zip(('cos_min', 'max_abs', 'mean_abs', 'rms', 'native_norm', 'reference_norm'), numbers))
            metrics[key].update(passed=passed, bit_exact=fields[10] == '1')
        elif fields[0] == 'ROPEEXACT':
            if len(fields) != 3 or fields[1] in controls or fields[2] not in ('0', '1'):
                raise ValueError('Invalid/duplicate bit control')
            controls[fields[1]] = fields[2] == '1'
        elif fields[0] == 'ROPEEND':
            if len(fields) != 4 or totals is not None:
                raise ValueError('Invalid/duplicate RoPE totals')
            totals = [int(v) for v in fields[1:]]
    required_controls = {f'{kind}.layer_{i}' for i in range(24) for kind in ('observer', 'inputs')}
    failures = sum(not r['passed'] for r in metrics.values()) + sum(not v for v in controls.values())
    if set(metrics) != expected_metrics() or set(controls) != required_controls or totals != [192, 48, failures]:
        raise ValueError('Incomplete RoPE coverage/totals')
    return dict(metrics=metrics, controls=controls, failures=failures)


def checked_archive(archive, inventory):
    names = archive.namelist()
    if (len(names) != len(set(names)) or set(names) != set(inventory)
            or sum(row.file_size for row in archive.infolist()) > 2 * 1024 ** 3
            or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names)):
        raise ValueError('Unsafe or changed preserved archive member set')
    for name in names:
        if archive.getinfo(name).file_size != inventory[name]['bytes']:
            raise ValueError('Preserved member bytes differ')
    return names


def service():
    def api(path):
        return json.loads(subprocess.check_output(['gh', 'api', 'repos/CrispStrobe/CrispEmbed/' + path], timeout=60))
    checks = []
    for run, head, workflow in [(ORIGINAL_RUN, ORIGINAL_HEAD, '.github/workflows/heavy-cpu.yml'),
                                (COPY_RUN, COPY_HEAD, '.github/workflows/preserve-ultrabert-reference.yml')]:
        meta = api('actions/runs/' + str(run))
        if (meta.get('id') != run or meta.get('head_sha') != head or meta.get('path') != workflow
                or meta.get('status') != 'completed' or meta.get('conclusion') != 'success'):
            raise ValueError('Original/preservation service lineage differs')
        checks.append({key: meta[key] for key in ('id', 'head_sha', 'path', 'status', 'conclusion')})
    artifact = api('actions/artifacts/' + str(ARTIFACT[0]))
    if (artifact.get('id') != ARTIFACT[0] or artifact.get('name') != ARTIFACT[1]
            or artifact.get('size_in_bytes') != ARTIFACT[2] or artifact.get('digest') != 'sha256:' + ARTIFACT[3]
            or artifact.get('expired') is not False or artifact.get('workflow_run', {}).get('id') != COPY_RUN
            or artifact.get('workflow_run', {}).get('head_sha') != COPY_HEAD):
        raise ValueError('Preserved artifact service identity differs')
    return dict(runs=checks, artifact={key: artifact[key] for key in
                ('id', 'name', 'size_in_bytes', 'digest', 'expires_at', 'expired')})


def run():
    closure = preflight()
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'ultrabert-rope'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    report = dict(schema='crispembed-ultrabert-rope-attribution-v1', status='started', source=closure,
        original_run=ORIGINAL_RUN, original_source=ORIGINAL_HEAD, preservation_run=COPY_RUN,
        acceptance=dict(max_abs=.002, cos_min=.99999, input_and_observer_controls='bit exact'),
        hypothesis='ggml recurrent F32 phase update versus original independent F32 inverse-frequency powers',
        limits=['Only frozen-input local RoPE operators; no checkpoint, conversion or full-model forward.',
                'Current original tables and their drift are separate from immutable frozen tables.',
                'Historical 183 F32/1294 F16 stage failures, strict replay failures and quality limits remain.',
                'No production gate/code change, reference replacement, release or model publication.'], cases=[])

    def save():
        text = json.dumps(report, indent=2, allow_nan=False)
        for path, label in [(str(ROOT), '<repo>'), (str(scratch), '<scratch>'), (str(out), '<out>'),
                            (os.environ.get('RUNNER_TEMP', ''), '<runner-temp>')]:
            if path:
                text = text.replace(path, label)
        (out / 'ultrabert-rope.json').write_text(text + '\n')

    save()
    try:
        if shutil.disk_usage(scratch).free < 12 * 1024 ** 3:
            raise ValueError('At least12GiB hosted scratch required before archive transfer')
        report['service'] = service()
        archive = scratch / 'preserved.zip'
        with archive.open('wb') as stream:
            subprocess.run(['gh', 'api', 'repos/CrispStrobe/CrispEmbed/actions/artifacts/' + str(ARTIFACT[0]) + '/zip'],
                           stdout=stream, stderr=subprocess.PIPE, check=True, timeout=1200)
        if archive.stat().st_size != ARTIFACT[2] or sha256(archive) != ARTIFACT[3]:
            raise ValueError('Preserved ZIP size/SHA256 differs')
        frozen = scratch / 'frozen'
        inventory = json.loads((ROOT / 'tests/results/ultrabert/reference-preservation.json').read_text())['files']
        with zipfile.ZipFile(archive) as z:
            checked_archive(z, inventory)
            z.extractall(frozen)
        for name, row in inventory.items():
            if sha256(frozen / name) != row['sha256']:
                raise ValueError('Preserved member SHA256 differs: ' + name)
        archive.unlink()  # Known task-owned, verified ZIP; preserved service artifact remains untouched.
        baseline = json.loads((ROOT / 'tests/results/ultrabert/reference-audit.json').read_text())
        original = read_frozen_manifest(baseline, frozen)
        if len(original['reference']['cases']) != 7 or {r['case']['id'] for r in original['reference']['cases']} != CASES:
            raise ValueError('Frozen seven-case inventory differs')
        report['preserved_members'] = inventory
        report['historical_receipts_sha256'] = {name: sha256(ROOT / 'tests/results/ultrabert' / name) for name in
            ('initial-native-parity.json', 'local-operator-audit.json', 'reproducibility.json',
             'initial-linear-capture-failure.json', 'full-forward-capture-failure.json', 'm1-validation.json')}
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(MODEL, revision=REVISION,
                         allow_patterns=list(profile()['inspected_file_sha256'])))
        validate_sources(directory)
        if (directory / 'model.safetensors').exists():
            raise ValueError('Operator-only snapshot must not contain checkpoint payload')
        from dump_ultrabert_rope_reference import build, metrics
        import gguf
        import numpy as np
        operators = scratch / 'operators'
        operators.mkdir()
        binary = Path(os.environ['AUDIT_BIN_DIR']) / 'test-ultrabert-rope'
        report['binary_sha256'] = sha256(binary)
        env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')}
        env.update(CRISPEMBED_ULTRABERT='1', CRISPEMBED_FORCE_CPU='1')
        report['native_gates'] = {'CRISPEMBED_ULTRABERT': '1', 'CRISPEMBED_FORCE_CPU': '1'}
        failures = []
        for record in original['reference']['cases']:
            name = record['case']['id']
            target = operators / (name + '.gguf')
            entry = dict(case=name, status='building original operator targets')
            report['cases'].append(entry)
            save()
            entry['python'] = build(directory, frozen / 'reference' / record['archive'], record, target)
            shutil.copy2(target.with_suffix('.json'), out / (name + '-original-operators.json'))
            raw_root = scratch / 'native' / name
            log = out / (name + '-native.log')
            with log.open('w') as stream:
                result = subprocess.run([str(binary), str(target), str(raw_root)], env=env,
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
            entry.update(native_exit_code=result.returncode, native_log=log.name, native_log_sha256=sha256(log))
            parsed = parse_log(log.read_text())
            entry['native'] = parsed
            expected = {t.name: t.data for t in gguf.GGUFReader(str(target)).tensors}
            found = {file.relative_to(raw_root).as_posix().removesuffix('.f32'): file for file in raw_root.rglob('*.f32')}
            if set(found) != expected_metrics():
                raise ValueError('Actual raw operator file coverage differs')
            measured = {}
            for key, file in sorted(found.items()):
                reference = expected[key.split('/')[1]]
                if file.stat().st_size != reference.size * 4:
                    raise ValueError('Raw operator shape/bytes differ')
                value = np.fromfile(file, dtype='<f4').reshape(reference.shape)
                row = metrics(value, reference)
                observed = parsed['metrics'][key]
                if (row['passed'] != observed['passed'] or row['bit_exact'] != observed['bit_exact']
                        or not math.isclose(row['max_abs'], observed['max_abs'], rel_tol=1e-6, abs_tol=1e-12)):
                    raise ValueError('Native log versus independent raw recomputation differs')
                measured[key] = dict(row, bytes=file.stat().st_size, sha256=sha256(file), shape=list(value.shape))
            entry['raw_metrics'] = measured
            entry['current_original_table_bit_differences'] = sum(not row[k]['bit_exact']
                for row in entry['python']['layer_observations'].values()
                for k in ('current_cos_vs_frozen', 'current_sin_vs_frozen'))
            if result.returncode not in (0, 1) or (result.returncode == 0) != (parsed['failures'] == 0):
                raise ValueError('Actual native exit differs from complete comparisons')
            entry['status'] = 'complete all operator comparisons'
            if parsed['failures']:
                failures.append(name)
            save()
        report.update(status='completed local RoPE attribution' if not failures else 'rejected local RoPE numerical gates',
                      failed_cases=failures, case_count=7, frozen_tables_observed=336,
                      native_comparisons=1344, bit_controls=336, no_full_model_or_checkpoint=True)
        save()
        return int(bool(failures))
    except Exception as error:
        report.update(status='failed RoPE attribution structure/source', error=str(error))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), indent=2))
    else:
        raise SystemExit(run())
