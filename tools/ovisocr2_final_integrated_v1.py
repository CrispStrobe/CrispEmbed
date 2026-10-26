"""Dependency-free final-source contract; historical audit guards stay immutable."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

BASE = 'f3f1b9e7d5385d900b68bdf4bd14677425a15eee'
GGML = 'a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
BINARIES = ('test-qwen35-decoder', 'test-qwen35-vision', 'test-qwen35-delta',
            'test-ovisocr2-context', 'test-ovisocr2-quality-run', 'test-ovisocr2-image-bounds')
DECLARED = {
    'src/qwen35_vision.cpp': 'bc479ff5438138f55815769c59776bd761e2d6189182c137ed2d8fa82b01caa3',
    'src/crispembed.cpp': '99bb8a2308021ce997e1570927322d749155c89155dd08af11fdcac6b99e7be1',
    'src/ultrabert_profile.h': 'fc5af20f91a6b2cb6fa9c07efa875a9d42d3e7e076038b833a908019a0affc31',
}


PROTECTED_PATHS = ('src',
 'ggml',
 'cmake',
 'CMakeLists.txt',
 'VERSION',
 'Cargo.lock',
 'examples',
 'python',
 'flutter',
 'crispembed',
 'crispembed-sys',
 'wasm',
 'tools/quantize.cpp',
 'build-wasm.sh',
 'build-embed-wasm.sh',
 'scripts/check-cpu-baseline.py',
 'scripts/check-bundled-deps.py',
 'tools/successor_asset_manifest.py',
 '.gitattributes',
 '.github/workflows/release.yml',
 '.github/workflows/release-wasm.yml',
 'LICENSE',
 'tests/regression',
 'tests/qwen35_parity.h',
 'tests/test_qwen35_decoder.cpp',
 'tests/test_qwen35_delta.cpp',
 'tests/test_qwen35_vision.cpp',
 'tests/test_ovisocr2_context.cpp',
 'tests/ovisocr2_quality_run.cpp',
 'tools/ci-heavy/ovisocr2_position_replay.py',
 'tools/ci-heavy/ovisocr2_large_inputs.py',
 'tools/ci-heavy/ovisocr2_2048_attribution.py',
 'tools/ci-heavy/ovisocr2_native_replay.py',
 'tools/ci-heavy/ovisocr2_model_audit.py',
 'tools/ci-heavy/ovisocr2_vision_audit.py',
 'tools/ci-heavy/ovisocr2_vision_env.py',
 'tools/ci-heavy/ovisocr2_quality_run.py',
 'tools/ci-heavy/ovisocr2_quality_contract.py',
 'tools/ocr_quality.py',
 'models/convert-qwen35-ocr-to-gguf.py',
 'models/qwen35_ovis_profile.py',
 'tests/results/qwen35-ocr/ovisocr2-model-audit.json',
 'tests/results/qwen35-ocr/ovisocr2-larger-inputs.json',
 'tests/results/qwen35-ocr/ovisocr2-large-input-panel.json',
 'tests/results/qwen35-ocr/ovisocr2-f16-quality.json',
 'tests/results/qwen35-ocr/ovisocr2-original-quality.json',
 'tests/results/qwen35-ocr/ovisocr2-export.json',
 'tests/results/qwen35-ocr/ovisocr2-quality-panel.json',
 'tests/results/qwen35-ocr/ovisocr2-quality-prepared-preservation.json',
 'README.md')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_binaries(manifest, source, protocol_sha):
    if (manifest.get('source_commit') != source or manifest.get('protocol_sha256') != protocol_sha
            or manifest.get('portable_cpu_baseline_checked') is not True
            or not isinstance(manifest.get('binary_sha256'), dict)
            or set(manifest['binary_sha256']) != set(BINARIES)
            or any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h)
                   for h in manifest['binary_sha256'].values())):
        raise ValueError('Exact six final binaries/source/protocol/SHA manifest required')
    return manifest


def checked_protocol(value, require_reviews=True):
    if (value.get('schema') != 'crispembed-ovisocr2-final-integrated-v1'
            or value.get('qualification') is not False or value.get('runtime_base') != BASE
            or value.get('ggml') != GGML or value.get('declared_source_sha256') != DECLARED
            or value.get('gates') != [0, 1]
            or value.get('declared_defaults') != {'source_interpolation': 'off', 'wasm_q8_rounding': 'off'}
            or set(value.get('dependencies', {})) != {'ovis_composed', 'wasm_rounding_ab'}):
        raise ValueError('Final integrated protocol/source/default scope differs')
    if not require_reviews:
        return value
    for name, item in value['dependencies'].items():
        if (item.get('reviewed') is not True or item.get('panel_completed') is not True
                or item.get('reviewed_outcome') not in ('candidate-supported', 'candidate-rejected-default-off')
                or item.get('final_default') != 'off'):
            raise ValueError('Reviewed completion and explicit defaultOFF decision pending: ' + name)
        if (type(item.get('run')) is not int or item['run'] <= 0
                or not re.fullmatch('[0-9a-f]{40}', item.get('source') or '')
                or item.get('workflow') != '.github/workflows/ovisocr2-metadata.yml'
                or item.get('conclusion') not in ('success', 'failure')
                or not item.get('artifacts')):
            raise ValueError('Reviewed dependency service identity absent: ' + name)
        ids = set()
        for artifact in item['artifacts']:
            if (type(artifact.get('id')) is not int or artifact['id'] <= 0 or artifact['id'] in ids
                    or type(artifact.get('bytes')) is not int or not 0 < artifact['bytes'] <= 8 * 1024**2
                    or not re.fullmatch('[0-9a-f]{64}', artifact.get('sha256') or '')
                    or not artifact.get('name') or not artifact.get('reports')):
                raise ValueError('Reviewed dependency artifact/report pins absent: ' + name)
            ids.add(artifact['id'])
            for member, digest in artifact['reports'].items():
                if (Path(member).is_absolute() or '..' in Path(member).parts
                        or not re.fullmatch('[0-9a-f]{64}', digest or '')):
                    raise ValueError('Unsafe/unpinned dependency report member')
    return value


def source_guard(root, protocol, require_submodule=True):
    root = Path(root)
    checked_protocol(protocol, require_reviews=False)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    objects = protocol['protected_git_objects']
    # Authenticate the declaration against the immutable base, not a caller's list.
    if set(objects) != set(PROTECTED_PATHS):
        raise ValueError('Incomplete integrated runtime/build/scorer closure')
    for path, expected in objects.items():
        if git('rev-parse', BASE + ':' + path) != expected or git('rev-parse', 'HEAD:' + path) != expected:
            raise ValueError('Final integrated source object differs: ' + path)
    if git('status', '--porcelain', '--', *objects):
        raise ValueError('Dirty integrated runtime/build/comparison source')
    for path, digest in DECLARED.items():
        if sha(root / path) != digest:
            raise ValueError('Declared source bytes differ: ' + path)
    if git('rev-parse', 'HEAD:ggml') != GGML:
        raise ValueError('Final dependency gitlink differs')
    if require_submodule:
        if not (root / 'ggml/.git').exists():
            raise ValueError('Initialized final dependency checkout required')
        if (git('-C', str(root / 'ggml'), 'rev-parse', 'HEAD') != GGML
                or git('-C', str(root / 'ggml'), 'status', '--porcelain')):
            raise ValueError('Final dependency checkout differs')
    return {'runtime_base': BASE, 'source_head': git('rev-parse', 'HEAD'),
            'protected_git_objects': objects, 'declared_source_sha256': DECLARED,
            'ggml': GGML, 'default_decisions': protocol['declared_defaults']}


def compare_quality(current, frozen, gate):
    """Keep first-F16 outputs/scores; measure gate1 confidence drift explicitly."""
    actual = current.get('native_cases', {}).get('f16', {})
    old = frozen['native_cases']['f16']
    if gate not in (0, 1) or len(actual) != 11 or set(actual) != set(old):
        raise ValueError('Full eleven-case first-F16 coverage required')
    keys = ('input_rgb_sha256', 'input_png_sha256', 'raw_text', 'processed_text',
            'token_ids', 'stop_id', 'finish_reason', 'stop_kind', 'cap')
    rows = {}
    for name, before in old.items():
        row = actual[name]
        if any(row.get(k) != before.get(k) for k in keys):
            raise ValueError('First-F16 input/IDs/text/EOS/cap differs: ' + name)
        values = row.get('confidences')
        if (not isinstance(values, list) or not values or len(values) != len(row['token_ids'])
                or len(values) != len(before['confidences'])
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
                or type(row.get('mean_confidence')) not in (int, float)
                or not math.isfinite(row['mean_confidence'])
                or abs(row['mean_confidence'] - sum(values) / len(values)) > 1e-6):
            raise ValueError('Selected confidence/mean record invalid: ' + name)
        exact = values == before['confidences'] and row['mean_confidence'] == before['mean_confidence']
        if gate == 0 and not exact:
            raise ValueError('Gate0 first-F16 selected confidence/mean differs: ' + name)
        rows[name] = {'token_count': len(values), 'selected_confidences_and_mean_exact': exact,
                      'max_selected_confidence_abs_drift': max(abs(a-b) for a,b in zip(values, before['confidences'])),
                      'mean_confidence_abs_drift': abs(row['mean_confidence'] - before['mean_confidence'])}
    for key in ('f16_paired_synthetic', 'f16_paired_documents'):
        if current.get(key) != frozen[key] or current[key].get('screen_passed') is not True:
            raise ValueError('Frozen paired quality screen differs: ' + key)
    if current.get('scores', {}).get('f16') != frozen['scores']['f16']:
        raise ValueError('Frozen first-F16 scores differ')
    return {'scope_passed': True, 'release_qualified': False, 'gate': gate, 'cases': rows,
            'token_count': sum(r['token_count'] for r in rows.values()),
            'confidence_policy': 'Gate0 exact; gate1 finite with mean validation and measured drift, no new confidence tolerance.'}
