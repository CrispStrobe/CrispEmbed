"""Frozen held-out F16 OCR audit; shared original dumper, native API and diff gates."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_quality import POLICY, score
from ocr_quality_prepare import PANEL, prepare, sha, validate_panel, verify_prepared
from qwen35_decoded_panel import validate
from qwen35_page_parity import parse
from qwen35_quant_calibration import file_sha
from qwen35_receipt import add_confidences, teacher_quality
from qwen35_vision_operators import prefill_records

EVIDENCE = 'tests/results/ocr-quality-prepared.json'


def validate_review(panel, evidence, panel_sha, scorer_sha, split_hashes):
    """A render success is insufficient: require retained hashes and explicit image review."""
    ids = [c['id'] for c in panel['cases']]
    rows = evidence['cases']
    review = evidence['visual_review']
    valid_sha = lambda value: isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None
    if (evidence['policy'] != POLICY or evidence['panel_sha256'] != panel_sha
            or evidence['scorer_sha256'] != scorer_sha or evidence['split_evidence_sha256'] != split_hashes
            or evidence['workflow_conclusion'] != 'success'
            or not re.fullmatch(r'https://github.com/CrispStrobe/CrispEmbed/actions/runs/\d+', evidence['workflow'])
            or not re.fullmatch(r'[0-9a-f]{40}', evidence['source_commit'])
            or not valid_sha(evidence['artifact_report_sha256'])
            or [c['id'] for c in rows] != ids or review['reviewed_case_ids'] != ids
            or not isinstance(review['notes'], str) or not review['notes'].strip()
            or review['png_sha256'] != {c['id']: c['png_sha256'] for c in rows}):
        raise ValueError('Incomplete, changed or unreviewed frozen OCR preparation evidence')
    for spec, row in zip(panel['cases'], rows):
        if (row['source_sha256'] != spec['source_sha256'] or row['gold_sha256'] != spec['gold_sha256']
                or row['file'] != spec['id'] + '.png' or not valid_sha(row['png_sha256'])
                or not valid_sha(row['rgb_sha256']) or len(row['size']) != 2
                or any(type(v) is not int or v < 32 or v % 32 for v in row['size'])
                or row['nominal_patches'] != row['size'][0] * row['size'][1] // 256
                or row['nominal_patches'] > 1024
                or (spec['resize'] is not None and row['size'] != spec['resize'])):
            raise ValueError('Changed reviewed OCR derivative schema or hashes')
    return evidence


def require_reviewed_preparation(repo):
    path = repo / EVIDENCE
    if not path.is_file():
        raise ValueError('Missing retained, reviewed OCR preparation report; finish hosted preparation and three-image review before model work')
    panel = validate_panel(repo)
    try:
        return validate_review(panel, json.loads(path.read_text()), sha(repo / PANEL), sha(repo / 'tools/ocr_quality.py'),
            {name: sha(repo / name) for name in [*panel['development_reports'], panel['calibration_report']]})
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Malformed retained OCR preparation/review evidence') from exc


def preflight(repo, out):
    """Called before heavyweight imports/downloads. Rerendering is hosted-only."""
    evidence = require_reviewed_preparation(repo)
    current = prepare(repo, out)
    if not verify_prepared(evidence, current) or current['frozen_derivatives_exact'] is not True:
        raise ValueError('Frozen OCR preparation did not reproduce')
    return current


def reference_complete(reference, cap):
    ids = reference['generated_ids']
    if (not ids or len(ids) > cap or any(type(v) is not int or not 0 <= v < 248320 for v in ids)
            or reference['max_new_tokens'] != cap):
        raise ValueError('Invalid original generated IDs/cap')
    eos = ids[-1] in reference['eos_ids']
    if (reference['reached_eos'] is not eos or reference['stop'] != ('eos' if eos else 'max_tokens')
            or (not eos and len(ids) != cap)):
        raise ValueError('Contradictory original EOS/cap metadata')
    return eos


def strict_prefill(log, expected_names):
    measured = prefill_records(log)
    metrics, budgets = measured['metrics'], measured['budgets']
    verdicts = re.findall(r'^(PASS|FAIL) (\S+): cos_min=', log, re.M)
    if (len(set(expected_names)) != 77 or [m['stage'] for m in metrics] != expected_names
            or [b['stage'] for b in budgets] != expected_names or [s for _, s in verdicts] != expected_names
            or [s for v, s in verdicts if v == 'FAIL'] != measured['qualified_failed_stages']
            or any(m['elements'] < 1 or any(not math.isfinite(v) for v in m.values() if isinstance(v, float)) for m in metrics)
            or any(not math.isfinite(b[k]) or b[k] < 0 for b in budgets for k in ('self_max_abs', 'max_ratio'))
            or any(v == 'PASS' and b['max_ratio'] > 1 for (v, _), b in zip(verdicts, budgets))):
        raise ValueError('Missing, repeated, nonfinite or contradictory strict prefill records')
    measured['first_qualified_failure'] = next((m for m in metrics if m['stage'] in measured['qualified_failed_stages']), None)
    measured['first_raw_absolute_exceedance'] = next((m for m in metrics if m['max_abs'] > .002), None)
    return measured


def summarize(cases):
    if len(cases) != 3 or any('decode' not in c or 'native_quality' not in c for c in cases):
        raise ValueError('Incomplete held-out audit; do not report completion')
    return dict(incomplete_cases=sum(not c['decode']['complete_document'] for c in cases),
        output_failures=sum(not c['decode']['ids_exact'] or not c['decode']['text_exact'] or not c['decode']['same_stop']
            or c['decode']['first_teacher_argmax_difference'] is not None for c in cases),
        qualified_prefill_failures=sum(len(c['prefill']['qualified_failed_stages']) for c in cases),
        teacher_score_failures=sum(c['teacher_quality']['qualified_failures'] for c in cases),
        quality_parse_failures=sum(not c[k]['parse_ok'] for c in cases for k in ('reference_quality', 'native_quality')))


def run(repo, build, baseline, local, out, module, anchor_image, anchor_npz):
    from PIL import Image
    evidence = require_reviewed_preparation(repo)
    panel = validate_panel(repo)
    inputs = out / 'quality-inputs'
    verify_prepared(evidence, json.loads((inputs / 'prepared.json').read_text()))
    prior = json.loads((repo / 'tests/results/qwen35-ocr/f16-materialization.json').read_text())
    historical = json.loads((repo / 'tests/results/qwen35-ocr/decoded-panel.json').read_text())
    expected_names = [m['stage'] for m in next(c['prefill'] for c in historical['arms'][0]['cases'] if 'prefill' in c)['metrics']]
    capture = module(repo / 'tools/dump_qwen35_ocr_page_reference.py', 'qwen35_quality_reference')
    root = out / 'quality'; root.mkdir(exist_ok=True)
    target = out / 'quality.json'
    report = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        model=historical['model'], revision=historical['revision'], policy=POLICY,
        scope='Three frozen explicit derivatives, original F32 eager versus ordinary native F16-text/F32-vision. No corpus benchmark, quantization or default promotion.',
        preparation=evidence, preparation_reproduced=True, production_changes='none', publication='none', cases=[],
        archived_numerical_debt=dict(records=924, failures=56, rerun=False), result='not completed')
    explicit_env = dict(CRISPEMBED_QWEN35_PARALLEL_F16='1', CRISPEMBED_QWEN35_PROFILE='0',
        CRISPEMBED_QWEN35_PRECISE_RMS='0', CRISPEMBED_QWEN35_IMATRIX='0', CRISPEMBED_QWEN35_QUANT='0',
        CRISPEMBED_QWEN35_VISION_PROBES='0', CRISPEMBED_QWEN35_VISION_PRECISE_NORM='0',
        CRISPEMBED_QWEN35_VISION_PRECISE_ATTN='0', CRISPEMBED_QWEN35_VISION_CASCADE_NORM='0',
        CRISPEMBED_QWEN35_TOKENIZER_MARKS='0')
    env = dict(os.environ, **explicit_env)
    report['native_environment'] = explicit_env
    try:
        report['runtime_sources'] = {name: sha(repo / name) for name in prior['sources']}
        if report['runtime_sources'] != prior['sources']:
            raise ValueError('Frozen production runtime changed')
        report['weights'] = dict(bytes=baseline.stat().st_size, sha256=file_sha(baseline),
            tensor_audit=json.loads((out / 'export-manifest.json').read_text())['records'])
        if report['weights'] != prior['weights']:
            raise ValueError('Frozen weight bytes/schema changed')
        report['diagnostic_sources'] = {name: sha(repo / name) for name in
            ('tools/ocr_quality.py', 'tools/ci-heavy/qwen35_quality.py', 'tools/dump_qwen35_ocr_page_reference.py',
             'tests/test_qwen35_quant.cpp', 'tests/qwen35_parity.h')}
        report['compact_anchor'] = capture.dump_reference(local, anchor_image, root / 'anchor', max_new_tokens=12, anchor=anchor_npz)
        gc.collect()
        # First freeze ALL original outputs. No evaluation-native command in this loop.
        for spec, prepared in zip(panel['cases'], evidence['cases']):
            path = inputs / prepared['file']
            image = Image.open(path).convert('RGB')
            if (sha(path) != prepared['png_sha256'] or list(image.size) != prepared['size']
                    or hashlib.sha256(image.tobytes()).hexdigest() != prepared['rgb_sha256']):
                raise ValueError('Prepared image changed after preflight')
            folder = root / spec['id']; folder.mkdir(exist_ok=True)
            case = dict(id=spec['id'], specification=spec, prepared=prepared,
                gold=json.loads((repo / spec['gold']).read_text()))
            report['cases'].append(case)
            case['reference'] = capture.dump_reference(local, path, folder, max_new_tokens=spec['max_new_tokens'],
                prompt=spec['prompt'], long_document=True, prefill_controls=True, vision_probes=True)
            ref = case['reference']
            case['reference_quality'] = score(case['gold'], ref['decoded'], complete=reference_complete(ref, spec['max_new_tokens']))
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            if (ref['image_sha256'] != prepared['png_sha256'] or ref['image_size'] != prepared['size']
                    or ref['grid'] != [[1, image.height // 16, image.width // 16]]
                    or ref['patches'] != prepared['nominal_patches'] or ref['patches'] > 1024 or ref['prompt'] != spec['prompt']):
                raise ValueError('Original processor changed frozen input/grid/prompt')
            gc.collect()
        report['all_original_outputs_frozen_before_native'] = True
        for case in report['cases']:
            spec, ref = case['specification'], case['reference']
            folder = root / case['id']
            command = [str(build / 'test-qwen35-quant'), str(baseline), str(inputs / case['prepared']['file']),
                str(folder / 'official-delta.gguf'), spec['prompt'], str(spec['max_new_tokens'])]
            p = subprocess.run(command + ['--prefill-only'], capture_output=True, text=True, env=env)
            log = p.stdout + p.stderr; (folder / 'baseline-prefill.log').write_text(log); p.check_returncode()
            case['prefill'] = strict_prefill(log, expected_names)
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            p = subprocess.run(command + ['--full-quality'], capture_output=True, text=True, env=env)
            log = p.stdout + p.stderr; (folder / 'baseline-decode.log').write_text(log); p.check_returncode()
            case['decode'] = add_confidences(log, validate(parse(log, ref), ref, spec['max_new_tokens']), ref)
            case['teacher_quality'] = teacher_quality(log, len(ref['generated_ids']))
            if case['decode']['metrics'][:77] != case['prefill']['metrics']:
                raise ValueError('Diagnostic prefill differs from full native prefill')
            native_eos = case['decode']['native_ids'][-1] in ref['eos_ids']
            case['native_quality'] = score(case['gold'], case['decode']['native_text'], complete=native_eos)
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(case['id'], 'IDs/text/EOS', case['decode']['ids_exact'], case['decode']['text_exact'],
                case['decode']['complete_document'], 'raw/content', case['native_quality']['raw'], case['native_quality']['content'], flush=True)
        report.update(summarize(report['cases']))
        report['result'] = 'completed held-out F16 audit; retained numerical, incomplete and output failures are not waived by raw/secondary OCR scores'
        if any(report[k] for k in ('incomplete_cases', 'output_failures', 'qualified_prefill_failures', 'teacher_score_failures', 'quality_parse_failures')):
            raise AssertionError('Held-out audit retains incomplete/output/numerical/parse failures')
    finally:
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
