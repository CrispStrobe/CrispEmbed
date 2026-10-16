"""One frozen book: existing vision operators and faithful attention diagnostics."""
import gc
import json
import os
from pathlib import Path
import subprocess

from ocr_quality_prepare import sha
from qwen35_attention_trace import records as attention_records
from qwen35_quality import require_reviewed_preparation, strict_prefill
from qwen35_quant_calibration import file_sha
from qwen35_vision_operators import vision_records


def frozen_reference(reference, old):
    fields = ('transformers', 'tokenizers', 'torch', 'dtype', 'attention', 'image_sha256', 'image_size',
              'grid', 'patches', 'prompt', 'rendered_prompt', 'eos_ids')
    if (any(reference[k] != old[k] for k in fields) or reference['max_new_tokens'] != 1
            or reference['generated_ids'] != old['generated_ids'][:1]
            or reference['selected_token_confidences'] != old['selected_token_confidences'][:1]):
        raise ValueError('Frozen book original inputs or first generated score changed')
    if set(reference['cpu_controls']) != set(old['cpu_controls']):
        raise ValueError('Missing original CPU control arms')
    for name, rows in reference['cpu_controls'].items():
        if len(rows) != 229 or [m for m in rows if '.trace.' not in m['stage']] != old['cpu_controls'][name]:
            raise ValueError('Changed frozen original CPU controls')
    trace = reference['attention_trace']
    if (trace['original_calls'] != {'prefill': 2, 'backend_off': 2, 'patch_chunked': 2}
            or trace['probability_and_output_exact_controls'] != 12
            or trace['projection_input_exact_controls'] != 6 or trace['selected_blocks'] != [5, 11]
            or len(trace['query_positions']) != 16 or trace['full_key_axis'] is not True):
        raise ValueError('Incomplete original attention observations')
    return True


def compile_isolation(commands):
    production = [c['command'] for c in commands if c['file'].endswith('/qwen35_vision.cpp')]
    operators = [c['command'] for c in commands if c['file'].endswith('/test_qwen35_vision_ops.cpp')]
    attention = [c['command'] for c in commands if c['file'].endswith('/test_qwen35_attention_trace.cpp')]
    macro = 'CRISPEMBED_QWEN35_VISION_OPERATOR_AUDIT'
    if (not production or not operators or not attention or any(macro in c for c in production + attention)
            or any(macro + '=1' not in c for c in operators)):
        raise ValueError('Reference operator copy must stay in its separate test translation unit')
    return True


def run(repo, build, baseline, local, out, module, anchor_image, anchor_npz):
    previous = json.loads((repo / 'tests/results/qwen35-ocr/quality.json').read_text())
    frozen = next(c for c in previous['cases'] if c['id'] == 'two-column-page')
    evidence = require_reviewed_preparation(repo)
    root = out / 'book-vision'; root.mkdir(exist_ok=True)
    target = out / 'book-vision.json'
    report = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                  model=previous['model'], revision=previous['revision'], prior_workflow=previous['workflow'],
                  prior_report_sha256=sha(repo / 'tests/results/qwen35-ocr/quality.json'),
                  scope='Frozen book, cap1 prefill/vision diagnostics. Earlier683-token EOS owning-API quality remains retained, not rerun or inferred from component scores.',
                  production_changes='none', publication='none', result='not completed', arms=[])
    try:
        report['runtime_sources'] = {name: sha(repo / name) for name in previous['runtime_sources']}
        if report['runtime_sources'] != previous['runtime_sources']: raise ValueError('Frozen runtime changed')
        report['weights'] = dict(bytes=baseline.stat().st_size, sha256=file_sha(baseline),
                                tensor_audit=json.loads((out / 'export-manifest.json').read_text())['records'])
        if report['weights'] != previous['weights']: raise ValueError('Frozen weights changed')
        report['compile_isolation'] = compile_isolation(json.loads((build / 'compile_commands.json').read_text()))
        report['diagnostic_sources'] = {name: sha(repo / name) for name in
            ('tools/ci-heavy/qwen35_book_vision.py', 'tools/dump_qwen35_ocr_page_reference.py',
             'tools/qwen35_attention_reference.py', 'tests/test_qwen35_vision_ops.cpp',
             'tests/test_qwen35_attention_trace.cpp', 'tests/qwen35_parity.h')}
        capture = module(repo / 'tools/dump_qwen35_ocr_page_reference.py', 'book_vision_reference')
        report['compact_anchor'] = capture.dump_reference(local, anchor_image, root / 'anchor', max_new_tokens=12, anchor=anchor_npz)
        gc.collect()
        image = out / 'quality-inputs' / frozen['prepared']['file']
        if sha(image) != frozen['prepared']['png_sha256']: raise ValueError('Reviewed book image changed')
        report['preparation'] = evidence
        report['reference'] = capture.dump_reference(local, image, root, max_new_tokens=1,
            prefill_controls=True, vision_probes=True, attention_trace=True)
        report['frozen_original_controls_exact'] = frozen_reference(report['reference'], frozen['reference'])
        if report['reference']['attention_trace']['installed_source_sha256'] != json.loads(
                (repo / 'tests/results/qwen35-ocr/book-isolation.json').read_text())['reference']['decode_probes']['original_modeling_sha256']:
            raise ValueError('Pinned original modeling implementation changed')
        gc.collect()
        reference = root / 'official-delta.gguf'
        names = [m['stage'] for m in frozen['prefill']['metrics']]
        recipes = [('baseline', 0, 0, 0), ('precise-both', 1, 1, 0),
                   ('cascade', 0, 0, 1), ('cascade-attention', 0, 1, 1)]
        for name, norm, attention, cascade in recipes:
            explicit = dict(previous['native_environment'], CRISPEMBED_QWEN35_VISION_PRECISE_NORM=str(norm),
                            CRISPEMBED_QWEN35_VISION_PRECISE_ATTN=str(attention),
                            CRISPEMBED_QWEN35_VISION_CASCADE_NORM=str(cascade))
            env = dict(os.environ, **explicit)
            arm = dict(name=name, native_environment=explicit)
            report['arms'].append(arm)
            for kind, executable, parser in [('operators', 'test-qwen35-vision-ops', vision_records),
                                             ('attention', 'test-qwen35-attention-trace', attention_records)]:
                command = [str(build / executable), str(baseline), str(reference)]
                if kind == 'operators' and name == 'baseline': command.append('--gate-controls')
                p = subprocess.run(command, capture_output=True, text=True, env=env)
                log = p.stdout + p.stderr; (root / (name + '-' + kind + '.log')).write_text(log)
                parsed = parser(log); arm[kind] = parsed
                rows = parsed['metrics'] if kind == 'attention' else parsed
                expected_exit = int(any(not m['qualified_pass'] for m in rows)
                                    or (kind == 'attention' and not parsed['controls_passed']))
                if p.returncode != expected_exit: raise ValueError('Numerical verdict and executable exit disagree')
                if kind == 'attention' and not parsed['controls_passed']:
                    raise ValueError('Native equality control failure invalidates attention interpretation')
            p = subprocess.run([str(build / 'test-qwen35-quant'), str(baseline), str(image), str(reference), '', '1', '--prefill-only'],
                               capture_output=True, text=True, env=env)
            log = p.stdout + p.stderr; (root / (name + '-prefill.log')).write_text(log); p.check_returncode()
            arm['prefill'] = strict_prefill(log, names)
            merger = next(m for m in arm['prefill']['metrics'] if m['stage'] == 'prefill.vision.merger')
            for rows in (arm['operators'], arm['attention']['metrics']):
                entry = next(m for m in rows if m['mode'] == 'composed' and m['stage'] == merger['stage'])
                if any(entry[k] != value for k, value in merger.items()):
                    raise ValueError('Vision trace differs from ordinary composed prefill')
            if name == 'baseline' and arm['prefill'] != frozen['prefill']:
                raise ValueError('Frozen baseline prefill did not reproduce exactly')
            target.write_text(json.dumps(report, indent=2) + '\n')
            print(name, 'prefill failures', len(arm['prefill']['qualified_failed_stages']),
                  'operator failures', sum(not m['qualified_pass'] for m in arm['operators']),
                  'attention failures', sum(not m['qualified_pass'] for m in arm['attention']['metrics']), flush=True)
        report['native_equality_controls'] = 24
        report['qualified_failure_count'] = sum(len(a['prefill']['qualified_failed_stages'])
            + sum(not m['qualified_pass'] for m in a['operators'])
            + sum(not m['qualified_pass'] for m in a['attention']['metrics']) for a in report['arms'])
        report['result'] = 'completed unchanged-runtime book vision localization; component results do not establish decoded quality or warrant default promotion'
        if report['qualified_failure_count']: raise AssertionError('Retained strict vision/prefill failures; inspect book-vision.json')
    finally:
        target.write_text(json.dumps(report, indent=2) + '\n')
