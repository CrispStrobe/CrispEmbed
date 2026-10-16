"""One preselected quantization recipe/document, frozen original and F16 controls."""
import fnmatch
import gc
import hashlib
import json
import os
import subprocess

from qwen35_decoded_panel import validate
from qwen35_page_parity import parse
from qwen35_quality import (require_reviewed_preparation, reference_complete, strict_prefill,
                            score, sha, validate_panel, verify_prepared)
from qwen35_quant_calibration import file_sha
from qwen35_receipt import add_confidences, teacher_quality


PROFILES = {'q8': ('q8-vision-f32', 'q8_0'),
            'q8-q6': ('attention-q8-ffn-q6', 'q4_k'),
            'q4': ('q4-vision-f32', 'q4_k')}


def frozen_original(current, previous):
    # Timings are observations; every other original field must reproduce.
    stable = lambda r: {k: v for k, v in r.items() if k != 'generation_seconds'}
    if stable(current) != stable(previous):
        raise ValueError('Frozen original output, confidence, inputs or CPU controls changed')
    return True


def frozen_baseline(current, previous):
    for name in ('prefill', 'teacher_quality', 'native_quality'):
        if current[name] != previous[name]:
            raise ValueError('Frozen F16 ' + name + ' changed')
    stable = lambda r: {k: v for k, v in r.items() if k != 'native_recognize_ms'}
    if stable(current['decode']) != stable(previous['decode']):
        raise ValueError('Frozen F16 IDs/text/EOS/confidence or diagnostic metrics changed')
    return True


def tensor_audit(before, after, overrides, kind):
    """Require exact protected bytes, full schema and effective rule precedence."""
    if set(before) != set(after) or not before:
        raise ValueError('Changed tensor names/count')
    for name, old in before.items():
        new = after[name]
        if new['shape'] != old['shape']:
            raise ValueError('Changed tensor shape: ' + name)
        protected = name.startswith('v.') or len(old['shape']) != 2
        if protected:
            if new != old:
                raise ValueError('Changed protected vision/vector/conv bytes: ' + name)
        else:
            expected = 'Q8_0' if 'embed' in name else kind.upper()
            for rule in overrides:
                pattern, precision = rule.rsplit('=', 1)
                if fnmatch.fnmatchcase(name, pattern):
                    expected = precision.upper()
            if new['type'] != expected:
                raise ValueError('Wrong effective tensor precision: ' + name)
    return True


def summary(arms):
    if (len(arms) != 2 or arms[0]['name'] != 'f16-baseline'
            or arms[1]['name'] not in {name for name, _ in PROFILES.values()}):
        raise ValueError('Incomplete paired quantization audit')
    result = []
    for arm in arms:
        d = arm['decode']
        result.append(dict(name=arm['name'], incomplete=not d['complete_document'],
            output_failure=not (d['ids_exact'] and d['text_exact'] and d['same_stop']),
            teacher_argmax_failure=d['first_teacher_argmax_difference'] is not None,
            prefill_failures=len(arm['prefill']['qualified_failed_stages']),
            score_failures=arm['teacher_quality']['qualified_failures'],
            quality_parse_failure=not arm['native_quality']['parse_ok']))
    return result


def run(repo, build, baseline, local, out, scratch, module, anchor_image, anchor_npz, profile, case_id):
    import gguf
    from PIL import Image
    prior_path = repo / 'tests/results/qwen35-ocr/quality.json'
    prior = json.loads(prior_path.read_text())
    old = next(c for c in prior['cases'] if c['id'] == case_id)
    evidence = require_reviewed_preparation(repo)
    panel = validate_panel(repo)
    spec = next(c for c in panel['cases'] if c['id'] == case_id)
    prepared = next(c for c in evidence['cases'] if c['id'] == case_id)
    if spec != old['specification'] or prepared != old['prepared']:
        raise ValueError('Frozen evaluation specification changed')
    if not verify_prepared(evidence, json.loads((out / 'quality-inputs/prepared.json').read_text())):
        raise ValueError('Frozen preparation changed')
    name, kind = PROFILES[profile]
    recipe_path = repo / ('tests/results/qwen35-ocr/ffn-calibration-parity.json' if profile == 'q4'
                          else 'tests/results/qwen35-ocr/page-parity.json')
    recipes = json.loads(recipe_path.read_text())
    recipe = next(a for a in recipes['arms'] if a['name'] == name)
    root = out / ('quality-quant-' + profile + '-' + case_id); root.mkdir(exist_ok=True)
    target = out / (root.name + '.json')
    report = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        model=prior['model'], revision=prior['revision'], policy=prior['policy'], profile=profile,
        prior_workflow=prior['workflow'], prior_report_sha256=sha(prior_path),
        recipe_source=str(recipe_path.relative_to(repo)), recipe_report_sha256=sha(recipe_path),
        scope='One frozen document and preselected existing recipe; paired F16 replay and quantized owning-API free generation, separate original-prefix teacher diagnostics. Not a corpus, calibration or promotion.',
        production_changes='none', publication='none', result='not completed', arms=[])
    try:
        report['runtime_sources'] = {p: sha(repo / p) for p in prior['runtime_sources']}
        if report['runtime_sources'] != prior['runtime_sources']:
            raise ValueError('Frozen production runtime changed')
        report['weights'] = dict(bytes=baseline.stat().st_size, sha256=file_sha(baseline),
            tensor_audit=json.loads((out / 'export-manifest.json').read_text())['records'])
        if report['weights'] != prior['weights']:
            raise ValueError('Frozen F16 weight bytes/schema changed')
        report['diagnostic_sources'] = {p: sha(repo / p) for p in
            ('tools/ci-heavy/qwen35_quality_quant.py', 'tools/ci-heavy/qwen35_quality.py',
             'tools/dump_qwen35_ocr_page_reference.py', 'tests/test_qwen35_quant.cpp',
             'tests/qwen35_parity.h', 'tools/quantize.cpp')}
        capture = module(repo / 'tools/dump_qwen35_ocr_page_reference.py', 'quality_quant_reference')
        report['compact_anchor'] = capture.dump_reference(local, anchor_image, root / 'anchor',
            max_new_tokens=12, anchor=anchor_npz)
        gc.collect()
        image = out / 'quality-inputs' / prepared['file']
        rgb = Image.open(image).convert('RGB')
        if (sha(image) != prepared['png_sha256'] or list(rgb.size) != prepared['size']
                or hashlib.sha256(rgb.tobytes()).hexdigest() != prepared['rgb_sha256']):
            raise ValueError('Frozen input pixels changed')
        gold = json.loads((repo / spec['gold']).read_text())
        report['case'] = dict(id=case_id, specification=spec, prepared=prepared, gold=gold)
        ref = capture.dump_reference(local, image, root, max_new_tokens=spec['max_new_tokens'],
            prompt=spec['prompt'], long_document=True, prefill_controls=True, vision_probes=True)
        report['reference'] = ref
        report['frozen_original_exact'] = frozen_original(ref, old['reference'])
        report['reference_quality'] = score(gold, ref['decoded'], complete=reference_complete(ref, spec['max_new_tokens']))
        if report['reference_quality'] != old['reference_quality']:
            raise ValueError('Frozen original quality changed')
        report['all_original_outputs_frozen_before_native'] = True
        gc.collect()
        explicit = dict(prior['native_environment'])
        report['native_environment'] = explicit
        env = dict(os.environ, **explicit)
        names = [m['stage'] for m in old['prefill']['metrics']]
        reference = root / 'official-delta.gguf'
        candidate = scratch / (name + '.gguf')
        def schema(path):
            reader = gguf.GGUFReader(str(path))
            return {t.name: dict(shape=t.shape.tolist(), type=t.tensor_type.name,
                bytes=t.data.nbytes, sha256=hashlib.sha256(t.data.tobytes()).hexdigest()) for t in reader.tensors}
        before = schema(baseline)
        for arm_name, weights in [('f16-baseline', baseline), (name, candidate)]:
            arm = dict(name=arm_name); report['arms'].append(arm)
            if weights == candidate:
                arm.update(kind=kind, overrides=recipe['overrides'])
                command = [str(build / 'crispembed-quantize'), str(baseline), str(candidate), kind]
                for rule in recipe['overrides']: command += ['--tensor-type', rule]
                p = subprocess.run(command, capture_output=True, text=True)
                (root / 'quantizer.log').write_text(p.stdout + p.stderr); p.check_returncode()
                arm.update(bytes=candidate.stat().st_size, gguf_sha256=file_sha(candidate), tensor_audit=schema(candidate))
                arm['protected_tensor_bytes_exact'] = tensor_audit(before, arm['tensor_audit'], recipe['overrides'], kind)
                if arm['bytes'] != recipe['bytes'] or arm['gguf_sha256'] != recipe['gguf_sha256']:
                    raise ValueError('Frozen quantized recipe bytes/hash did not reproduce')
                arm['frozen_quantized_weights_exact'] = True
            command = [str(build / 'test-qwen35-quant'), str(weights), str(image), str(reference),
                       spec['prompt'], str(spec['max_new_tokens'])]
            p = subprocess.run(command + ['--prefill-only'], capture_output=True, text=True, env=env)
            log = p.stdout + p.stderr; (root / (arm_name + '-prefill.log')).write_text(log); p.check_returncode()
            arm['prefill'] = strict_prefill(log, names)
            target.write_text(json.dumps(report, indent=2) + '\n')
            p = subprocess.run(command + ['--full-quality'], capture_output=True, text=True, env=env)
            log = p.stdout + p.stderr; (root / (arm_name + '-decode.log')).write_text(log); p.check_returncode()
            arm['decode'] = add_confidences(log, validate(parse(log, ref), ref, spec['max_new_tokens']), ref)
            arm['teacher_quality'] = teacher_quality(log, len(ref['generated_ids']))
            if arm['decode']['metrics'][:77] != arm['prefill']['metrics']:
                raise ValueError('Diagnostic prefill differs from full native prefill')
            arm['native_quality'] = score(gold, arm['decode']['native_text'],
                complete=arm['decode']['native_ids'][-1] in ref['eos_ids'])
            if arm_name == 'f16-baseline':
                report['frozen_f16_exact'] = frozen_baseline(arm, old)
            target.write_text(json.dumps(report, indent=2) + '\n')
            print(profile, case_id, arm_name, 'IDs/text/EOS', arm['decode']['ids_exact'],
                  arm['decode']['text_exact'], arm['decode']['complete_document'],
                  'prefill/score failures', len(arm['prefill']['qualified_failed_stages']),
                  arm['teacher_quality']['qualified_failures'], flush=True)
        report['summary'] = summary(report['arms'])
        report['result'] = 'completed paired document quantization diagnostic; retained numerical/output/quality failures prevent promotion'
        if any(any(v for k, v in s.items() if k != 'name') for s in report['summary']):
            raise AssertionError('Retained strict/output/quality failures; inspect ' + target.name)
    finally:
        target.write_text(json.dumps(report, indent=2) + '\n')
