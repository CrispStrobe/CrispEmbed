"""Frozen book: original-embedding continuation isolation, no production changes."""
import gc
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

from qwen35_decoded_panel import validate
from qwen35_page_parity import parse
from qwen35_quality import require_reviewed_preparation, strict_prefill
from qwen35_receipt import add_confidences, teacher_quality
from qwen35_vision_operators import BUDGET, METRIC, metric
from ocr_quality import score
from ocr_quality_prepare import sha

STEPS = (94, 240)


def probe_names(steps=STEPS):
    names = ['decoder.inputs_embeds', *['decoder.layer.' + str(i) for i in range(24)], 'decoder.norm', 'logits']
    for i in range(24):
        names.extend('cache.' + str(i) + '.' + field for field in
                     (('keys', 'values') if i % 4 == 3 else ('conv_states', 'recurrent_states')))
    return [f'generated.probe.{step}.{name}' for step in steps for name in names]


def probes(log, steps=STEPS):
    rows = []
    lines = log.splitlines()
    for i, line in enumerate(lines):
        match = re.fullmatch(r'PROBE_METRIC (\S+) ' + METRIC, line)
        if not match:
            continue
        name = match[1]
        if i + 2 >= len(lines):
            raise ValueError('Missing probe verdict/budget')
        verdict = re.match(r'^(PASS|FAIL) ' + re.escape(name) + r': cos_min=', lines[i + 1])
        budget = re.fullmatch(BUDGET, lines[i + 2])
        if not verdict or not budget or budget[1] != name or budget[2] not in ('0', '1') or budget[3] not in ('0', '1'):
            raise ValueError('Missing or contradictory probe verdict/budget')
        row = dict(stage=name, **metric(match.groups()[1:]), qualified_pass=verdict[1] == 'PASS',
                   strict_absolute_pass=bool(int(budget[2])), calibrated=bool(int(budget[3])),
                   self_max_abs=float(budget[4]), max_ratio=float(budget[5]))
        if (row['elements'] < 1 or any(not math.isfinite(v) for v in row.values() if isinstance(v, float))
                or any(row[k] < 0 for k in ('max_abs', 'mean_abs', 'rms', 'native_norm', 'reference_norm'))
                or row['calibrated'] or row['self_max_abs'] != 0 or row['max_ratio'] < 0
                or row['strict_absolute_pass'] != (row['max_abs'] <= .002)
                or row['qualified_pass'] != (row['cos_min'] >= .99999 and row['max_abs'] <= .002 and row['max_ratio'] <= 1)):
            raise ValueError('Invalid or relaxed probe budget')
        rows.append(row)
    expected = probe_names(steps)
    verdict_names = [name for _, name in re.findall(r'^(PASS|FAIL) (generated\.probe\.\S+): cos_min=', log, re.M)]
    budget_names = [m[0] for m in re.findall(BUDGET, log, re.M) if m[0].startswith('generated.probe.')]
    if [row['stage'] for row in rows] != expected or verdict_names != expected or budget_names != expected:
        raise ValueError('Missing/repeated/out-of-order original decode probes')
    return rows


def original_frozen(reference, frozen):
    # Durations and new observer metadata differ; original inference fields must not.
    fields = ('transformers', 'tokenizers', 'torch', 'dtype', 'attention', 'image_sha256', 'image_size',
              'grid', 'patches', 'prompt', 'rendered_prompt', 'max_new_tokens', 'generated_ids', 'decoded',
              'eos_ids', 'reached_eos', 'stop', 'selected_token_confidences', 'mean_confidence', 'cpu_controls')
    if any(reference[k] != frozen[k] for k in fields):
        raise ValueError('Changed frozen original generation, confidence, grid or CPU controls')
    report = reference['decode_probes']
    if (report['steps'] != list(STEPS) or report['forward_calls'] != len(reference['generated_ids'])
            or report['fields_per_step'] != 75 or report['observer_off_ids_scores_exact'] is not True):
        raise ValueError('Missing exact original observer-off replay')
    return True


def continuation(log, reference, frozen, reference_embeddings):
    records = [dict(stage=m[0], **metric(m[1:])) for m in re.findall(r'^METRIC (\S+) ' + METRIC + '$', log, re.M)]
    expected = [m['stage'] for m in frozen['decode']['metrics']]
    if ([m['stage'] for m in records] != expected
            or any(m['elements'] < 1 or any(not math.isfinite(v) for v in m.values() if isinstance(v, float)) for m in records)):
        raise ValueError('Missing/repeated/nonfinite continuation metrics')
    teacher = [dict(step=int(m[0]), selected=int(m[1]), reference=int(m[2]), same=bool(int(m[3])))
               for m in re.findall(r'^TEACHER step=(\d+) selected=(\d+) reference=(\d+) same=(\d+)$', log, re.M)]
    ids = reference['generated_ids']
    if (len(teacher) != len(ids) or any(t['step'] != i or t['reference'] != ids[i]
            or t['same'] != (t['selected'] == t['reference']) for i, t in enumerate(teacher))):
        raise ValueError('Incomplete or contradictory teacher prefix')
    if log.count('ISOLATION reference_embeddings=1 test_only=1') != int(reference_embeddings):
        raise ValueError('Incorrect diagnostic embedding source')
    if reference_embeddings:
        summary = re.findall(r'^DECODE_ISOLATION finite=1 reference_embeddings=1 test_only=1 probes=150 scores=(\d+) free_generation=0$', log, re.M)
        if summary != [str(len(ids))] or re.search(r'^FREE ', log, re.M):
            raise ValueError('Isolation must be explicitly test-only, without substituted free generation')
        if next(m for m in records if m['stage'] == 'prefill.decoder.inputs_embeds')['max_abs'] != 0:
            raise ValueError('Original embedding input is not exact')
    else:
        if records != frozen['decode']['metrics'] or teacher != frozen['decode']['teacher_forced']:
            raise ValueError('Composed baseline metrics or teacher selections changed')
    return dict(metrics=records, teacher_forced=teacher, teacher_quality=teacher_quality(log, len(ids)),
                probes=probes(log), reference_embeddings_test_only=reference_embeddings,
                first_teacher_argmax_difference=next((t['step'] for t in teacher if not t['same']), None))


def run(repo, build, baseline, local, out, module, anchor_image, anchor_npz):
    retained = json.loads((repo / 'tests/results/qwen35-ocr/quality.json').read_text())
    frozen = next(c for c in retained['cases'] if c['id'] == 'two-column-page')
    evidence = require_reviewed_preparation(repo)
    root = out / 'book-isolation'; root.mkdir(exist_ok=True)
    target = out / 'book-isolation.json'
    report = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                  model=retained['model'], revision=retained['revision'], prior_workflow=retained['workflow'],
                  prior_report_sha256=sha(repo / 'tests/results/qwen35-ocr/quality.json'),
                  scope='One frozen book; ordinary native baseline versus test-only original-embedding continuation. No composed parity claim for isolation.',
                  production_changes='none', publication='none', result='not completed', arms=[])
    try:
        report['runtime_sources'] = {name: sha(repo / name) for name in retained['runtime_sources']}
        if report['runtime_sources'] != retained['runtime_sources']:
            raise ValueError('Frozen runtime arithmetic changed')
        report['weights'] = dict(bytes=baseline.stat().st_size, sha256=sha(baseline),
                                tensor_audit=json.loads((out / 'export-manifest.json').read_text())['records'])
        if report['weights'] != retained['weights']:
            raise ValueError('Frozen weight schema/bytes changed')
        report['diagnostic_sources'] = {name: sha(repo / name) for name in
            ('tools/ci-heavy/qwen35_book_isolation.py', 'tools/dump_qwen35_ocr_page_reference.py',
             'tools/qwen35_decode_reference.py', 'tests/test_qwen35_quant.cpp', 'tests/qwen35_parity.h')}
        capture = module(repo / 'tools/dump_qwen35_ocr_page_reference.py', 'book_reference')
        report['anchor'] = capture.dump_reference(local, anchor_image, root / 'anchor', max_new_tokens=12, anchor=anchor_npz)
        gc.collect()
        image = out / 'quality-inputs' / frozen['prepared']['file']
        if sha(image) != frozen['prepared']['png_sha256']:
            raise ValueError('Frozen book pixels changed')
        report['preparation'] = evidence
        report['specification'] = frozen['specification']
        report['reference'] = capture.dump_reference(local, image, root, max_new_tokens=1024, long_document=True,
            prefill_controls=True, vision_probes=True, decode_probe_steps=STEPS)
        original_frozen(report['reference'], frozen['reference'])
        report['original_frozen_fields_exact'] = True
        gc.collect()
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        command = [str(build / 'test-qwen35-quant'), str(baseline), str(image), str(root / 'official-delta.gguf'), '', '1024']
        env = dict(os.environ, **retained['native_environment'])
        report['native_environment'] = retained['native_environment']
        for name, flag in [('composed', '--full-quality'), ('original-embeddings-prefill', '--prefill-reference-embeddings'),
                           ('original-embeddings', '--scores-reference-embeddings')]:
            p = subprocess.run(command + [flag], capture_output=True, text=True, env=env)
            log = p.stdout + p.stderr; (root / (name + '.log')).write_text(log); p.check_returncode()
            if name.endswith('-prefill'):
                report['isolated_prefill'] = strict_prefill(log, [m['stage'] for m in frozen['prefill']['metrics']])
            else:
                arm = dict(name=name, **continuation(log, report['reference'], frozen, name != 'composed'))
                report['arms'].append(arm)
                if name == 'composed':
                    decoded = add_confidences(log, validate(parse(log, report['reference']), report['reference'], 1024), report['reference'])
                    if decoded != frozen['decode']:
                        # Wall-clock recognition time is intentionally not an equality condition.
                        before = dict(frozen['decode']); before.pop('native_recognize_ms')
                        after = dict(decoded); after.pop('native_recognize_ms')
                        if before != after: raise ValueError('Frozen ordinary native decoded output/confidence changed')
                    arm['decode'] = decoded
                    arm['quality'] = score(frozen['gold'], decoded['native_text'], complete=decoded['complete_document'])
                    if arm['quality'] != frozen['native_quality']: raise ValueError('Frozen quality scores changed')
                elif arm['metrics'][:77] != report['isolated_prefill']['metrics']:
                    raise ValueError('Isolated prefill differs during continuation')
                print(name, 'strict score failures', arm['teacher_quality']['qualified_failures'],
                      'probe failures', sum(not m['qualified_pass'] for m in arm['probes']), flush=True)
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        report['result'] = 'completed frozen book localization; isolation is diagnostic, not production/native quality promotion'
        report['qualified_score_failures'] = sum(a['teacher_quality']['qualified_failures'] for a in report['arms'])
        report['qualified_probe_failures'] = sum(not m['qualified_pass'] for a in report['arms'] for m in a['probes'])
        if report['qualified_score_failures'] or report['qualified_probe_failures'] or report['isolated_prefill']['qualified_failed_stages']:
            raise AssertionError('Retained strict numerical failures; inspect book-isolation.json')
    finally:
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
