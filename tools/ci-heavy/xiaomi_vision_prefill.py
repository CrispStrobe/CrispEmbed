#!/usr/bin/env python3
"""GH-hosted CPU-only Xiaomi qianziwen vision parity audit.

Dispatch heavy-cpu.yml with this script and args:
  --model-url https://APPROVED_HOST/IMMUTABLE_MODEL
  --reference-url https://APPROVED_HOST/IMMUTABLE_REFERENCE
Both must be non-secret, operator-approved HTTPS URLs; neither is currently
known to be publicly available. No token, query, fragment, or credentials in
URLs. The private Kaggle kernel outputs are NOT available through this workflow:
it does not supply Kaggle credentials. Verify access/redistribution policy first.
The payloads are pinned by SHA-256 independent of the transport. A missing
source is a setup blocker, not a parity result. Large files stay in scratch;
only compact evidence/logs go to HEAVY_OUT. No external publication occurs.

Requires a clean committed checkout, the pinned base's dependency bytes, the exact five
reviewed source overlays and ggml gitlink below. Never run this on Kaggle CPU.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BASE = '15adb672f71f8470a23db1284dafa102515adae9'
GGML = 'a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
MODEL_SHA = '1759b8da2a6fa2651de9cacdb5e90fe0551d7ab8b1fe783198f7339389451235'
REF_SHA = '2a58fc3fc2a4ae2d603ff600371cf71fa1f54e40ebd260d7680bfa4c1f868334'
PINNED_SOURCES = {
    'tools/kaggle/xiaomi-ocr-0-vision-prefill/CMakeLists.txt': 'b17c50331b9a47a9be6eca2bc374aa901a178b285a1ce1bf8ef55bb02bdccf63',
    'tests/test_xiaomi_vision_prefill.cpp': 'c6cdba6816f085fc8148c9b3ddbe813519592eacdc3fa1542147fb3c76b7e73a',
    'src/qwen35_vision.cpp': 'b3de0c502e769b1f6b51e1e66c7d50750d38896b7bd7f63bf9d205f703c7fe62',
    'src/qwen35_vision.h': '157c0b7090a8510ce48dc08585d034fd2a842494d528cc835e2e883abd6413f1',
    'src/qwen35_image_bounds.h': '9f8168efb1fc07690108a058d2c6411216bf1b0ff7eefe30f40aed5ba7f18f31',
}
NAMES = (['patch_embed', 'position', 'embedded', 'rotary_cos', 'rotary_sin'] +
         [f'layer.{i}' for i in range(12)] +
         ['merger.norm', 'merger.up', 'merger.activation', 'merger'])
EXPECTED = {'prefill.vision.' + name: ([64, 1984] if name in ('rotary_cos', 'rotary_sin') else
             [3072, 496] if name in ('merger.up', 'merger.activation') else
             [1024, 496] if name == 'merger' else [768, 1984]) for name in NAMES}
METRICS = ('max_abs', 'mean_abs', 'rms', 'cos_min', 'cos_mean', 'cos_global', 'native_norm', 'reference_norm')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_url(url):
    parsed = urllib.parse.urlsplit(url or '')
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or
            parsed.query or parsed.fragment or any(c.isspace() for c in url or '')):
        raise ValueError('no usable approved HTTPS sources: existing workflow has no Kaggle credential; '
                         'provide two approved HTTPS URLs without credentials, query or fragment')


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Storage CDNs may use signed queries on redirects; do not record them.
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('unsafe download redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_url(url):
    return urllib.request.build_opener(HTTPSRedirect()).open(url, timeout=120)


def download(url, path, expected, max_bytes):
    check_url(url)
    hasher, count = hashlib.sha256(), 0
    try:
        with open_url(url) as response, path.open('xb') as handle:
            for block in iter(lambda: response.read(4 << 20), b''):
                count += len(block)
                if count > max_bytes:
                    raise ValueError('download exceeds pinned size bound')
                hasher.update(block)
                handle.write(block)
        actual = hasher.hexdigest()
        if actual != expected:
            raise ValueError('download SHA-256 does not match frozen artifact')
        return {'sha256': actual, 'bytes': count}
    except Exception:
        path.unlink(missing_ok=True)
        raise


def retrieve_inputs(model_url, ref_url, scratch, report):
    check_url(model_url)
    check_url(ref_url)
    report['inputs'] = {}
    model, ref = scratch / 'model-f32.gguf', scratch / 'qianziwen-prefill.gguf'
    for name, url, path, sha, limit in (
            ('model', model_url, model, MODEL_SHA, 3421929920),
            ('reference', ref_url, ref, REF_SHA, 2 * 1024**3)):
        # No URL is written to logs: URLs might contain private host paths.
        report['inputs'][name] = {'url_sha256': hashlib.sha256(url.encode()).hexdigest(),
                                  'expected_sha256': sha, 'status': 'retrieving'}
        report['inputs'][name].update(download(url, path, sha, limit) or {})
        report['inputs'][name]['status'] = 'verified'
    return model, ref


def require_hosted():
    if (os.environ.get('GITHUB_ACTIONS') != 'true' or os.environ.get('RUNNER_ENVIRONMENT') != 'github-hosted'
            or os.environ.get('RUNNER_OS') != 'Linux' or not os.environ.get('GITHUB_SHA')):
        raise ValueError('requires the Linux GitHub-hosted heavy-cpu workflow')


def command(label, args, cwd, report, out, env):
    try:
        p = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except OSError:
        report['commands'].append({'name': label, 'exit': None})
        raise ValueError(f'{label} could not start') from None
    report['commands'].append({'name': label, 'exit': p.returncode})
    (out / (label + '.log')).write_text(p.stdout)
    return p


def git(args, cwd, env):
    p = subprocess.run(['git'] + args, cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode:
        raise ValueError('source checkout verification failed: git ' + args[0])
    return p.stdout.strip()


# Transitive local includes of the standalone target, excluding the pinned overlays.
BASE_DEPENDENCIES = (
    'src/core/clean_exit.h', 'src/core/env_gate.h', 'src/core/gguf_loader.cpp',
    'src/core/gguf_loader.h', 'src/crispembed_diff.h', 'src/qwen35_delta.h',
    'src/qwen35_linear.cpp', 'src/qwen35_linear.h', 'src/qwen35_math.h',
    'src/qwen35_norm.cpp', 'src/qwen35_norm.h', 'src/qwen35_profile.cpp',
    'src/qwen35_profile.h', 'src/qwen35_weight_cache.h',
)


def verify_base_dependencies(env):
    git(['diff', '--quiet', BASE, 'HEAD', '--', *BASE_DEPENDENCIES], ROOT, env)


def source_closure(report, env):
    head = git(['rev-parse', 'HEAD'], ROOT, env)
    if head != os.environ['GITHUB_SHA']:
        raise ValueError('checkout differs from GitHub SHA')
    # actions/checkout defaults to a shallow clone; fetch the immutable base
    # object, rather than assuming merge ancestry is locally available.
    try:
        git(['cat-file', '-e', BASE + '^{commit}'], ROOT, env)
    except ValueError:
        git(['fetch', '--no-tags', '--depth=1', 'origin', BASE], ROOT, env)
        git(['cat-file', '-e', BASE + '^{commit}'], ROOT, env)
    verify_base_dependencies(env)
    if git(['rev-parse', 'HEAD:ggml'], ROOT, env) != GGML or git(['rev-parse', 'HEAD'], ROOT / 'ggml', env) != GGML:
        raise ValueError('ggml gitlink or checkout differs from the pinned revision')
    git(['diff', '--quiet', 'HEAD', '--'], ROOT, env)
    git(['diff', '--quiet', 'HEAD', '--'], ROOT / 'ggml', env)
    required = list(PINNED_SOURCES) + [str(Path(__file__).relative_to(ROOT))]
    for name in required:
        git(['ls-files', '--error-unmatch', name], ROOT, env)
    tracked = git(['ls-files', 'src'], ROOT, env).splitlines()
    tracked += required
    closure = {name: digest(ROOT / name) for name in sorted(set(tracked)) if (ROOT / name).is_file()}
    report['source_closure'] = {'checkout': head, 'pinned_base': BASE, 'ggml_commit': GGML,
                                'files_sha256': closure}
    for name, expected in PINNED_SOURCES.items():
        if closure.get(name) != expected:
            raise ValueError('reviewed source differs: ' + name)


def evaluate(log, exit_code, report):
    report['native_exit'] = exit_code
    rows = []
    report['stages'] = rows  # retain complete preceding stages on a malformed row
    def reject_constant(_):
        raise ValueError('non-finite JSON stage metric')
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError('non-finite JSON stage metric')
        return parsed
    for line in log.splitlines():
        if line.startswith('STAGE '):
            rows.append(json.loads(line[6:], parse_constant=reject_constant, parse_float=finite_float))
    if exit_code not in (0, 1):
        raise ValueError('native setup/control exit, not a numerical mismatch')
    if [r.get('name') for r in rows] != list(EXPECTED):
        raise ValueError('stage set/order incomplete')
    backend = next((line for line in log.splitlines() if line.startswith('AUDIT_BACKEND CPU ')), None)
    control = any(line.startswith('NEGATIVE_CONTROL rejected=1 ') for line in log.splitlines())
    if not backend or not control:
        raise ValueError('CPU backend or negative control evidence missing')
    report['backend_evidence'], report['negative_control_rejected'] = backend, control
    failed = []
    for row in rows:
        shape = EXPECTED[row['name']]
        if row.get('shape') != shape or row.get('n') != shape[0] * shape[1]:
            raise ValueError('stage shape or element count differs')
        if any(not isinstance(row.get(k), (int, float)) or isinstance(row[k], bool) or
               not math.isfinite(row[k]) for k in METRICS):
            raise ValueError('non-finite or absent stage metric')
        strict = row['max_abs'] <= 0.002 and row['cos_min'] >= 0.99999
        if type(row.get('pass')) is not bool or strict != row['pass']:
            raise ValueError('stage verdict differs from literal strict gate')
        if not strict:
            failed.append(row['name'])
    report.update(stages_total=len(rows), stages_passed=len(rows) - len(failed),
                  strict_failures=failed, first_failing_stage=failed[0] if failed else None)
    if exit_code != int(bool(failed)):
        raise ValueError('native exit disagrees with strict verdict')
    return exit_code


def prepare_cmake(text):
    # Cache implementation is part of qwen35_linear.cpp, not a separate source.
    if ('src/qwen35_weight_cache.cpp' in text or
            'converted_weight_cache::get(' not in (ROOT / 'src/qwen35_linear.cpp').read_text()):
        raise ValueError('unexpected standalone CMake/cache implementation')
    return text


def run(args, out, scratch, report):
    require_hosted()
    # Do not pass CI secrets, environment-dependent diff gates or precision switches to subprocesses.
    env = {k: os.environ[k] for k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL') if k in os.environ}
    check_url(args.model_url)
    check_url(args.reference_url)
    source_closure(report, env)
    if shutil.disk_usage(scratch).free < 8 * 1024**3:
        raise ValueError('scratch has less than 8 GiB free')
    model, reference = retrieve_inputs(args.model_url, args.reference_url, scratch, report)
    source = scratch / 'audit-source'
    source.mkdir()
    original_cmake = ROOT / 'tools/kaggle/xiaomi-ocr-0-vision-prefill/CMakeLists.txt'
    prepared = prepare_cmake(original_cmake.read_text())
    (source / 'CMakeLists.txt').write_text(prepared)
    report['scratch_cmake'] = {'sha256': digest(source / 'CMakeLists.txt'),
        'transformation': 'none'}
    for name in ('src', 'tests', 'ggml'):
        (source / name).symlink_to(ROOT / name, target_is_directory=True)
    build = scratch / 'build-vision-prefill'
    commands = [('configure', ['cmake', '-S', str(source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                               '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', '-DGGML_VULKAN=OFF',
                               '-DGGML_BLAS=OFF', '-DGGML_OPENMP=OFF']),
                ('build', ['cmake', '--build', str(build), '--target', 'test-xiaomi-vision-prefill', '--parallel', '4'])]
    for label, argv in commands:
        if command(label, argv, ROOT, report, out, env).returncode:
            raise ValueError(label + ' failed; see log')
    binary = build / 'test-xiaomi-vision-prefill'
    report['binary_sha256'] = digest(binary)
    result = command('vision-prefill', [str(binary), str(model), str(reference)], ROOT, report, out, env)
    return evaluate(result.stdout, result.returncode, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-url')
    parser.add_argument('--reference-url')
    args = parser.parse_args()
    if not os.environ.get('HEAVY_OUT') or not os.environ.get('HEAVY_SCRATCH'):
        print('requires HEAVY_OUT and HEAVY_SCRATCH from heavy-cpu.yml', file=sys.stderr)
        return 2
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    report = {'status': 'setup_failure', 'commands': [], 'scope': 'CPU vision prefill only; no frontend, decoder, decoded text, quantized or end-to-end parity',
              'reference_precision': 'BF16 execution widened to F32',
              'strict_gate': {'max_abs': 0.002, 'cos_min': 0.99999, 'relaxed_for_dtype': False},
              'expected_input_sha256': {'model': MODEL_SHA, 'reference': REF_SHA}}
    try:
        code = run(args, out, scratch, report)
        report['status'] = 'strict_mismatch' if code == 1 else 'strict_pass'
    except Exception as exc:
        code = 2
        # Do not leak URLs, signed redirects, subprocess stderr, or credentials.
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, UnicodeError) else 'setup or retrieval failed; verify approved artifact access and dependencies'
        if any(s in report['error'].lower() for s in ('https:', 'http:', 'token', 'authorization')):
            report['error'] = 'setup or retrieval failed; verify approved artifact access and dependencies'
    report['exit'] = code
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out / 'summary.md').write_text('# Xiaomi CPU vision prefill\n\nStatus: ' + report['status'] +
                                    '\n\nSee report.json and logs. No decoded or end-to-end parity claim.\n')
    print('Xiaomi vision prefill:', report['status'], 'exit', code, flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
