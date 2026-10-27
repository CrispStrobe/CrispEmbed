"""Dependency-free final-source contract; historical audit guards stay immutable."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

BASE = '4f830552fa83794e4045aaaa20e52e3d785785bd'
GGML = 'a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
DECLARED = {
             "src/qwen35_vision.cpp": "bc479ff5438138f55815769c59776bd761e2d6189182c137ed2d8fa82b01caa3",
             "src/crispembed.cpp": "99bb8a2308021ce997e1570927322d749155c89155dd08af11fdcac6b99e7be1",
             "src/ultrabert_profile.h": "fc5af20f91a6b2cb6fa9c07efa875a9d42d3e7e076038b833a908019a0affc31"
            }
BINARIES = ('test-qwen35-decoder', 'test-qwen35-vision', 'test-qwen35-delta',
            'test-ovisocr2-context', 'test-ovisocr2-quality-run', 'test-ovisocr2-image-bounds')
FIXTURE_BASE = 'f3f1b9e7d5385d900b68bdf4bd14677425a15eee'
FIXTURE_ADDITION = 'tests/regression/josev2b/text-v1.json'
FIXTURE_SHA256 = '162f811fdc8a057e886915fde4f9a2bb69d8a4713c13ac98f9420772c5ec9c04'
AUDITORS = ('tools/ovisocr2_final_integrated_v2.py','tools/ci-heavy/ovisocr2_final_integrated_v2.py',
            '.github/workflows/ovisocr2-final-integrated-v2.yml','tests/test_ovisocr2_final_integrated_v2.py')
DEFAULTS = {'source_interpolation':'off','wasm_q8_rounding':'off'}
OVIS_SOURCE = '8bb9f074d1ca6fa2cb2be6a892fc2fd116202912'
DEPENDENCIES = {'ovis_composed':(40376882009,OVIS_SOURCE,'success','candidate-supported'),
                'wasm_rounding_ab':(40391544739,'580e3ae3be3671ddce5e945b0725d0583ec34110','failure','candidate-rejected-default-off')}


PROTECTED_PATHS = ('.gitmodules', 'tools/ci-heavy/successor_assets.py',
 'tools/ci-heavy/ovisocr2_published_consumer.py', 'src',
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
 'README.md',
 'tools/ci-heavy/ocr_quality_prepare.py',
 'tools/ci-heavy/qwen35_decoded_panel.py',
 'tools/ci-heavy/qwen35_page_parity.py',
 'tools/ci-heavy/qwen35_quality.py',
 'tools/ci-heavy/qwen35_quant_calibration.py',
 'tools/ci-heavy/qwen35_receipt.py',
 'tools/ci-heavy/qwen35_vision_operators.py',
 'tools/dump_ovisocr2_reference.py',
 'tools/ci-heavy/ovisocr2_selective_precision.py',
 'tests/results/ocr-quality-prepared.json',
 'tests/test_ovisocr2_image_bounds.cpp')


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


def checked_protocol(value, require_reviews=True, require_closure=True):
    if (value.get('schema') != 'crispembed-ovisocr2-final-integrated-v2'
            or value.get('qualification') is not False or value.get('runtime_base') != BASE
            or value.get('ggml') != GGML or value.get('declared_source_sha256') != DECLARED
            or value.get('gates') != [0,1] or value.get('declared_defaults') != DEFAULTS
            or set(value.get('dependencies',{})) != set(DEPENDENCIES)):
        raise ValueError('Final-v2 protocol/source/default scope differs')
    if require_closure and (value.get('runtime_closure_reviewed') is not True
            or not re.fullmatch('[0-9a-f]{40}',BASE) or not re.fullmatch('[0-9a-f]{40}',GGML)
            or any(not re.fullmatch('[0-9a-f]{64}',h) for h in DECLARED.values())
            or any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in value.get('auditor_source_sha256',{}).values())):
        raise ValueError('Reviewed exact final runtime/base/closure pending')
    inheritance=value.get('fixture_inheritance',{})
    if (inheritance.get('base') != FIXTURE_BASE or set(inheritance.get('additions',{})) != {FIXTURE_ADDITION}
            or inheritance['additions'][FIXTURE_ADDITION].get('sha256') != FIXTURE_SHA256
            or not re.fullmatch('[0-9a-f]{40}',inheritance['additions'][FIXTURE_ADDITION].get('git_object',''))):
        raise ValueError('Only the exact additional Josev fixture is declared')
    if set(value.get('auditor_source_sha256',{})) != set(AUDITORS):
        raise ValueError('Complete versioned auditor source declaration required')
    if not require_reviews:return value
    for name,item in value['dependencies'].items():
        run,source,conclusion,outcome=DEPENDENCIES[name]
        if (item.get('reviewed') is not True or item.get('panel_completed') is not True
                or item.get('reviewed_outcome') != outcome or item.get('final_default') != 'off'):
            raise ValueError('Reviewed completion and explicit defaultOFF pending: '+name)
        if (item.get('run'),item.get('source'),item.get('conclusion'),item.get('workflow')) != (run,source,conclusion,'.github/workflows/ovisocr2-metadata.yml'):
            raise ValueError('Exact reviewed dependency service identity differs: '+name)
        pins=item.get('artifacts',[])
        if len(pins)!=(8 if name=='ovis_composed' else 1):raise ValueError('Complete reviewed small artifact coverage required')
        if name=='ovis_composed':
            if {(p.get('case'),p.get('gate')) for p in pins}!={(c,g) for c in ('frozen-control','control-square','large-landscape','large-boundary') for g in (0,1)}:
                raise ValueError('Exact eight composed grid/gate reports required')
            if (item.get('global_default_qualified') is not False or item.get('input_2048_qualified') is not False
                    or item.get('strict_numerical_qualified') is not False or item.get('checker')!='unchanged-calibrated-with-raw-strict-flags'):
                raise ValueError('Composed review must preserve calibrated/strict/default/2048 limits')
            binaries=item.get('reviewed_binary_sha256',{})
            if set(binaries)!=set(BINARIES) or any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in binaries.values()):raise ValueError('Reviewed binary source identities required')
        ids=set()
        for pin in pins:
            if (type(pin.get('id')) is not int or pin['id']<=0 or pin['id'] in ids
                    or type(pin.get('bytes')) is not int or not 0<pin['bytes']<=8*1024**2
                    or not re.fullmatch('[0-9a-f]{64}',pin.get('sha256','')) or not pin.get('name') or not pin.get('reports')):
                raise ValueError('Reviewed artifact ZIP/report identities absent')
            ids.add(pin['id'])
            for member,digest in {**pin.get('members',{}),**pin['reports']}.items():
                if Path(member).is_absolute() or '..' in Path(member).parts or '\\' in member or not re.fullmatch('[0-9a-f]{64}',digest):
                    raise ValueError('Unsafe/unpinned dependency member')
            if name=='ovis_composed' and (not pin.get('members') or 'ovisocr2-large-inputs.json' not in pin['reports']
                    or any(pin['members'].get(k)!=v for k,v in pin['reports'].items())):
                raise ValueError('Every small ZIP member and report must be pinned')
    return value


def fixture_guard(root, final_base, declaration):
    root=Path(root)
    def leaves(revision):
        rows=subprocess.check_output(['git','ls-tree','-r',revision,'--','tests/regression'],cwd=root,text=True).splitlines()
        return {r.split('\t',1)[1]:r.split('\t',1)[0] for r in rows}
    inherited=leaves(FIXTURE_BASE)
    expected=dict(inherited)
    for name,row in declaration['additions'].items():expected[name]='100644 blob '+row['git_object']
    for revision in (final_base,'HEAD'):
        actual=leaves(revision)
        if actual!=expected:raise ValueError('Historical fixtures changed/deleted or undeclared additions: '+revision)
    if sha(root/FIXTURE_ADDITION)!=FIXTURE_SHA256:raise ValueError('Additional Josev fixture bytes differ')
    return {'historical_base':FIXTURE_BASE,'historical_files':len(inherited),'new_fixture':declaration['additions']}


def source_guard(root, protocol, require_submodule=True):
    root=Path(root);checked_protocol(protocol,require_reviews=False)
    def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
    objects=protocol['protected_git_objects']
    if set(objects)!=set(PROTECTED_PATHS):raise ValueError('Incomplete final-v2 runtime/build/scorer closure')
    for path,expected in objects.items():
        if not re.fullmatch('[0-9a-f]{40}',expected) or git('rev-parse',BASE+':'+path)!=expected or git('rev-parse','HEAD:'+path)!=expected:
            raise ValueError('Final-v2 explicit source object differs: '+path)
    if git('status','--porcelain','--',*objects):raise ValueError('Dirty final-v2 source')
    for path,digest in DECLARED.items():
        if sha(root/path)!=digest:raise ValueError('Declared final-v2 source bytes differ: '+path)
    inherited=fixture_guard(root,BASE,protocol['fixture_inheritance'])
    if git('rev-parse','HEAD:ggml')!=GGML:raise ValueError('Exact final-v2 ggml gitlink differs')
    if require_submodule:
        if not (root/'ggml/.git').exists():raise ValueError('Initialized final dependency required')
        if git('-C',str(root/'ggml'),'rev-parse','HEAD')!=GGML or git('-C',str(root/'ggml'),'status','--porcelain'):
            raise ValueError('Exact final-v2 dependency checkout differs')
    return {'runtime_base':BASE,'source_head':git('rev-parse','HEAD'),'protected_git_objects':objects,
            'declared_source_sha256':DECLARED,'ggml':GGML,'default_decisions':protocol['declared_defaults'],
            'fixture_inheritance':inherited}


def checked_composed_report(report,pin,dependency):
    gate=pin['gate']
    if (report.get('schema')!='crispembed-ovisocr2-position-composed-replay-v1'
            or report.get('case')!=pin['case'] or report.get('gate')!=gate
            or report.get('closure',{}).get('source_head')!=OVIS_SOURCE
            or report.get('native_gates',{}).get('CRISPEMBED_QWEN35_VISION_SOURCE_INTERPOLATION')!=str(gate)
            or report.get('build',{}).get('source_commit')!=OVIS_SOURCE
            or report.get('build',{}).get('binary_sha256')!=dependency['reviewed_binary_sha256']
            or len(report.get('arms',{}))!=14 or set(report.get('owning_cases',{}))!={'f32','f16'}):
        raise ValueError('Actual composed report source/binary/grid/gate/coverage differs')
    if gate==0:
        if report.get('baseline_control_passed') is not True or report.get('candidate_numerically_qualified') is not False:
            raise ValueError('Actual gate0 baseline negative/control disposition differs')
    elif (report.get('candidate_numerically_qualified') is not True or report.get('failures')
            or any(r.get('passed') is not True for r in report['arms'].values())
            or any(r.get('original_ids_text_eos_exact') is not True for r in report['owning_cases'].values())):
        raise ValueError('Actual gate1 calibrated/decoded support differs')
    return {'calibrated_candidate_supported':gate==1,'global_default_qualified':False,
            'input_2048_qualified':False,'strict_numerical_qualified':False,'raw_strict_flags_preserved':True}


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
