"""Hosted actual C quantizer and packed Q8 admission fixtures; no model forward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/ci-heavy')]
from ci_process import run
from gguf_file_type_canonicalize import parse, require, sha, span
from gguf_file_type_controls import CONTROL_TYPES, METHODS
PROFILE = ROOT / 'docs/portfolio-q8-profile-controls-v1.json'


def source(reviewed):
    cfg = json.loads(PROFILE.read_text())
    git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT)
    require(git('rev-parse', 'HEAD').decode().strip() == reviewed, 'Exact reviewed source required')
    require(cfg['schema'] == 'crispembed-portfolio-q8-profile-controls-v1' and
            cfg['model_forwards'] == 0 and cfg['publication_qualified'] is False, 'Model-free fixture scope required')
    for group in ('protected_sha256', 'owned_sha256'):
        for name, pin in cfg[group].items():
            p = ROOT / name
            require(p.is_file() and not p.is_symlink() and sha(p) == pin and
                    git('show', 'HEAD:' + name) == p.read_bytes(), 'Fixture source differs: ' + name)
            if group == 'protected_sha256':
                require(git('show', cfg['base'] + ':' + name) == p.read_bytes(), 'Runtime/base source differs: ' + name)
    allowed = set(cfg['owned_sha256']) | {str(PROFILE.relative_to(ROOT))}
    require(set(git('diff', '--name-only', cfg['base'], 'HEAD').decode().splitlines()) <= allowed,
            'Undeclared Q8 fixture delta')
    require(git('rev-parse', 'HEAD:ggml').decode().strip() == cfg['ggml'], 'Frozen backend differs')
    require(not git('status', '--porcelain', '--untracked-files=no'), 'Dirty tracked source')
    return cfg


def controls(args):
    source(args.reviewed_source)
    require(os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_SHA') == args.reviewed_source,
            'Hosted source-bound compiler fixtures required')
    require(shutil.disk_usage(args.out.parent).free >= 5 * 2**30, '5GiB hosted scratch required')
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='crispembed-portfolio-q8-profile-controls-result-v1', source=args.reviewed_source,
                  completed=False, model_forwards=0, models_accessed=False, commands={}, methods=[],
                  publication_qualified=False, source_profile_sha256=sha(PROFILE))
    def save():
        (args.out / 'controls.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    def command(name, argv, allowed=(0,)):
        log = args.out / (name + '.log')
        p = run(argv, log, cwd=ROOT, timeout=1200)
        result['commands'][name] = dict(p, log_sha256=sha(log))
        save()
        require(not p['timed_out'] and p['exit'] in allowed, 'Actual fixture rejected: ' + name)
        return p
    save()
    try:
        command('configure', ['cmake', '-S', ROOT / 'tests/portfolio-q8-profiles', '-B', args.build,
                              '-DCMAKE_BUILD_TYPE=Release'])
        command('build', ['cmake', '--build', args.build, '--parallel', '2'])
        for family in ('josev', 'ultrabert'):
            command(family + '-profile', [args.build / ('q8-' + family + '-profile')])
        fixture = args.build / 'q8-quantizer-fixture'
        quantizer = args.build / 'q8-quantize'
        original = args.out / 'authored-f32.gguf'
        command('authored-input', [fixture, 'write', original])
        for method, tensor, legacy, standard in METHODS:
            left, right = args.out / (method + '-legacy.gguf'), args.out / (method + '-standard.gguf')
            command(method + '-legacy', [quantizer, original, left, method, '--legacy-file-type-header'])
            command(method + '-standard', [quantizer, original, right, method])
            command(method + '-legacy-readback', [fixture, 'check', left, str(tensor), str(legacy)])
            command(method + '-standard-readback', [fixture, 'check', right, str(tensor), str(standard)])
            a, b = parse(left, CONTROL_TYPES), parse(right, CONTROL_TYPES)
            offset = a['metadata']['general.file_type']['offset']
            require(a['tensors'] == b['tensors'] and a['bytes'] == b['bytes'], 'Standard/legacy layout changed')
            with left.open('rb') as x, right.open('rb') as y:
                require(span(x, 0, offset) == span(y, 0, offset) and
                        span(x, offset + 4, a['bytes'] - offset - 4) ==
                        span(y, offset + 4, b['bytes'] - offset - 4), 'Non-header payload changed')
            result['methods'].append(dict(method=method, actual_tensor_type=tensor,
                                          legacy_header=legacy, standard_header=standard,
                                          legacy_sha256=sha(left), standard_sha256=sha(right),
                                          all_non_header_bytes_exact=True))
            save()
        baseline = args.out / 'q8_0-standard.gguf'
        stamped = args.out / 'q8-stamped.gguf'
        command('stamp', [quantizer, original, stamped, 'q8_0', '--metadata-string',
                           'general.architecture=authored_fixture_q8_mixed_profile_v2'])
        a, b = parse(baseline, CONTROL_TYPES), parse(stamped, CONTROL_TYPES)
        require(a['metadata']['general.architecture']['value'] == 'authored_fixture' and
                b['metadata']['general.architecture']['value'] == 'authored_fixture_q8_mixed_profile_v2' and
                b['metadata']['general.file_type']['value'] == 7, 'Actual profile stamp/header differs')
        payloads = []
        with baseline.open('rb') as x, stamped.open('rb') as y:
            for name, row in a['tensors'].items():
                other = b['tensors'][name]
                require(all(row[k] == other[k] for k in ('stored_shape', 'type', 'bytes')), 'Stamp changed tensor schema')
                left = span(x, row['data_offset'], row['bytes'])
                require(left == span(y, other['data_offset'], other['bytes']), 'Stamp changed tensor payload')
                payloads.append(dict(name=name, type=row['type'], bytes=row['bytes'], payload_sha256=left))
        for label, overrides in [('numeric', ['general.file_type=bad']), ('missing', ['missing.profile=bad']),
                                 ('duplicate', ['general.architecture=a', 'general.architecture=b']),
                                 ('empty', ['general.architecture='])]:
            output = args.out / (label + '-must-not-exist.gguf')
            argv = [quantizer, original, output, 'q8_0']
            for item in overrides:
                argv.extend(['--metadata-string', item])
            p = command('reject-' + label, argv, allowed=(1,))
            require(p['exit'] == 1 and not output.exists(), 'Bad override created output')
        result.update(completed=True, actual_cli_methods=13, actual_profile_controls_passed=True,
                      metadata_stamp_payloads=payloads, metadata_stamp_whole_sha256=sha(stamped),
                      invalid_override_outputs_absent=True)
        save()
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        save()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reviewed-source', required=True)
    p.add_argument('--source-only', action='store_true')
    p.add_argument('--out', type=Path)
    p.add_argument('--build', type=Path)
    args = p.parse_args()
    if args.source_only:
        cfg = source(args.reviewed_source)
        print(json.dumps(dict(protected=len(cfg['protected_sha256']), owned=len(cfg['owned_sha256']))))
    else:
        require(args.out is not None and args.build is not None, 'Explicit scratch/output required')
        controls(args)
