"""Four arithmetic settings on frozen documents; unchanged strict stage gates."""
import gc
import json
import math
import os
import subprocess

from qwen35_page_parity import parse
from qwen35_quant_calibration import errors, file_sha
from qwen35_vision_operators import prefill_records


RECIPES = [('baseline', False, False, False), ('precise-both', True, True, False),
           ('cascade', False, False, True), ('cascade-attention', False, True, True)]


def validate(measured, reference, cap):
    """Reject incomplete/contradictory measurements, retain real quality failures."""
    ids, text = measured['native_ids'], measured['native_text']
    if (not ids or len(ids) > cap or any(type(x) is not int or not 0 <= x < 248320 for x in ids)
            or measured['native_decoded_bytes'] != len(text.encode('utf-8'))
            or measured['ids_exact'] != (ids == reference['generated_ids'])
            or measured['text_exact'] != (text == reference['decoded'])
            or not math.isfinite(measured['native_mean_confidence'])
            or not 0 <= measured['native_mean_confidence'] <= 1
            or not math.isfinite(measured['native_recognize_ms']) or measured['native_recognize_ms'] <= 0):
        raise ValueError('Invalid decoded output, equality claim, confidence or timing')
    eos = ids[-1] in reference['eos_ids']
    if measured['stop'] != (1 if eos else 2) or (not eos and len(ids) != cap):
        raise ValueError('EOS/cap stop disagrees with actual generated IDs')
    if (len(reference['generated_ids']) > cap
            or reference['reached_eos'] != (reference['generated_ids'][-1] in reference['eos_ids'])
            or (not reference['reached_eos'] and len(reference['generated_ids']) != cap)):
        raise ValueError('Invalid original EOS/cap control')
    metrics = measured['metrics']
    if (len({m['stage'] for m in metrics}) != len(metrics)
            or any(not math.isfinite(v) for m in metrics for v in m.values() if isinstance(v, float))):
        raise ValueError('Nonfinite or repeated stage measurements')
    teacher = measured['teacher_forced']
    if (len(teacher) != len(reference['generated_ids'])
            or any(t['step'] != i or t['reference'] != reference['generated_ids'][i]
                   or t['same'] != (t['selected'] == t['reference']) for i, t in enumerate(teacher))):
        raise ValueError('Missing or contradictory teacher diagnostics')
    measured['same_stop'] = measured['stop'] == (1 if reference['reached_eos'] else 2)
    measured['complete_document'] = eos and reference['reached_eos']
    measured['cap_control'] = not reference['reached_eos']
    measured['native_characters'] = len(text)
    measured['reference_characters'] = len(reference['decoded'])
    measured['first_id_difference'] = next((i for i in range(max(len(ids), len(reference['generated_ids'])))
        if i >= len(ids) or i >= len(reference['generated_ids']) or ids[i] != reference['generated_ids'][i]), None)
    measured['first_teacher_argmax_difference'] = next((t['step'] for t in teacher if not t['same']), None)
    measured['confidence_difference'] = measured['native_mean_confidence'] - reference['mean_confidence']
    return measured


def run(repo, build, baseline, local, out, module, anchor_image, anchor_npz, tail=False):
    from PIL import Image
    capture = module(repo / 'tools/dump_qwen35_ocr_page_reference.py', 'qwen35_decoded_reference')
    results = repo / 'tests/results/qwen35-ocr'
    frozen = json.loads((results / 'page-parity.json').read_text())
    prior = json.loads((results / 'attention-trace.json').read_text())
    initial = json.loads((results / 'decoded-panel-initial.json').read_text()) if tail else None
    report_name = 'decoded-panel-tail' if tail else 'decoded-panel'
    root = out / report_name
    root.mkdir(exist_ok=True)
    report = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        model=frozen['model'], revision=frozen['revision'],
        scope='Four fixed permissively licensed documents, four arithmetic settings; receipt retains original384-token cap. No corpus benchmark or performance claim.',
        reference='Original pinned Transformers5.5.4 F32 eager generate, independently captured before native runs',
        native='Actual owning recognize_file API generates freely; separate teacher-forced and strict prefill diagnostics use crispembed-diff',
        production_changes='none', publication='none', cases=[], arms=[])
    target = out / (report_name + '.json')
    if tail:
        report['scope'] = 'Completion of two unfinished cascade-attention cases only: receipt384-token prefix and German header; fresh original references required to reproduce the incomplete run.'
        report['completion_of'] = {k: initial[k] for k in ('source_commit', 'workflow', 'artifact_report_sha256')}
        if (initial['audit_summary']['completed_decode_cases'] != 14
                or [len(a['cases']) for a in initial['arms']] != [4, 4, 4, 2]):
            raise ValueError('Unexpected incomplete parent audit')
    try:
        report['production_source_sha256'] = file_sha(repo / 'src/qwen35_vision.cpp')
        report['norm_source_sha256'] = file_sha(repo / 'src/qwen35_norm.cpp')
        if (report['production_source_sha256'] != prior['production_source_sha256']
                or report['norm_source_sha256'] != prior['norm_source_sha256']):
            raise ValueError('Production arithmetic changed')
        report['weights'] = dict(bytes=baseline.stat().st_size, sha256=file_sha(baseline),
            tensor_audit=json.loads((out / 'export-manifest.json').read_text())['records'])
        if (report['weights'] != prior['weights']):
            raise ValueError('Frozen weight bytes/schema changed')
        report['compact_anchor'] = capture.dump_reference(local, anchor_image, root / 'anchor',
            max_new_tokens=12, anchor=anchor_npz)
        gc.collect()
        for old in frozen['cases']:
            if tail and old['name'] not in ('receipt', 'german-header'):
                continue
            source = repo / old['source']['local_file']
            if old['source']['license'] not in ('CC0-1.0', 'Public-Domain') or file_sha(source) != old['source']['sha256']:
                raise ValueError('Changed source license/checksum')
            folder = root / old['name']
            folder.mkdir(exist_ok=True)
            image = Image.open(source).convert('RGB')
            if old['crop']:
                image = image.crop(old['crop'])
            if old['explicit_resize']:
                image = image.resize(old['explicit_resize'], resample=Image.Resampling.BICUBIC)
            path = folder / 'input.png'
            image.save(path)
            if file_sha(path) != old['input_sha256']:
                raise ValueError('Changed frozen evaluation derivative')
            item = {k: old[k] for k in ('name', 'source', 'original_size', 'crop', 'explicit_resize', 'resampler',
                'input_sha256', 'max_new_tokens', 'gold', 'annotation_policy')}
            report['cases'].append(item)
            large = old['name'] != 'arabic-line'
            item['reference'] = capture.dump_reference(local, path, folder, max_new_tokens=old['max_new_tokens'],
                prefill_controls=large, vision_probes=large)
            original = item['reference']
            fields = ('transformers', 'tokenizers', 'torch', 'dtype', 'attention', 'image_sha256', 'image_size', 'grid',
                'patches', 'prompt', 'rendered_prompt', 'max_new_tokens', 'generated_ids', 'decoded', 'eos_ids', 'reached_eos', 'stop')
            item['prior_original_output_exact'] = all(original[k] == old['reference'][k] for k in fields)
            if large:
                earlier = next(c for c in prior['cases'] if c['name'] == old['name'])
                controls = {name: [m for m in rows if '.trace.' not in m['stage']]
                    for name, rows in earlier['reference']['cpu_controls'].items()}
                item['prior_original_cpu_controls_exact'] = original['cpu_controls'] == controls
                if any(len(rows) != 217 for rows in original['cpu_controls'].values()):
                    raise ValueError('Incomplete original CPU controls')
            if tail:
                previous = next(c['reference'] for c in initial['cases'] if c['name'] == old['name'])
                item['initial_reference_exact'] = all(original[k] == previous[k]
                    for k in (*fields, 'cpu_controls', 'selected_token_confidences', 'mean_confidence'))
            if item['gold']:
                item['reference_vs_gold'] = errors(item['gold']['text'], original['decoded'])
            gc.collect()
        for name, norm, attention, cascade in (RECIPES[-1:] if tail else RECIPES):
            env = dict(os.environ, CRISPEMBED_QWEN35_VISION_PRECISE_NORM=str(int(norm)),
                CRISPEMBED_QWEN35_VISION_PRECISE_ATTN=str(int(attention)),
                CRISPEMBED_QWEN35_VISION_CASCADE_NORM=str(int(cascade)),
                CRISPEMBED_QWEN35_PRECISE_RMS='0', CRISPEMBED_QWEN35_VISION_PROBES='0',
                CRISPEMBED_QWEN35_IMATRIX='0')
            arm = dict(name=name, precise_norm=norm, precise_attention=attention, cascade_norm=cascade, cases=[])
            report['arms'].append(arm)
            earlier_arm = next(a for a in prior['arms'] if a['name'] == name)
            for case in report['cases']:
                folder = root / case['name']
                command = [str(build / 'test-qwen35-quant'), str(baseline), str(folder / 'input.png'),
                    str(folder / 'official-delta.gguf'), '', str(case['max_new_tokens'])]
                process = subprocess.run(command, capture_output=True, text=True, env=env)
                log = process.stdout + process.stderr
                (folder / (name + '-decode.log')).write_text(log)
                process.check_returncode()
                entry = dict(name=case['name'], decode=validate(parse(log, case['reference']), case['reference'], case['max_new_tokens']))
                arm['cases'].append(entry)
                if case['gold']:
                    entry['decode']['native_vs_gold'] = errors(case['gold']['text'], entry['decode']['native_text'])
                if name == 'baseline':
                    old = next(c for c in frozen['arms'][0]['cases'] if c['name'] == case['name'])
                    fields = ('native_ids', 'native_text', 'ids_exact', 'text_exact', 'stop', 'native_mean_confidence',
                        'native_decoded_bytes', 'teacher_forced', 'metrics', 'score_summary')
                    entry['prior_baseline_decode_exact'] = all(entry['decode'][k] == old[k] for k in fields)
                if case['name'] != 'arabic-line':
                    process = subprocess.run(command + ['--prefill-only'], capture_output=True, text=True, env=env)
                    prefill_log = process.stdout + process.stderr
                    (folder / (name + '-prefill.log')).write_text(prefill_log)
                    process.check_returncode()
                    entry['prefill'] = prefill_records(prefill_log)
                    earlier = next(c for c in earlier_arm['cases'] if c['name'] == case['name'])
                    entry['prior_prefill_exact'] = entry['prefill'] == earlier['prefill']
                    entry['decode_prefill_exact'] = entry['decode']['metrics'][:77] == entry['prefill']['metrics']
                target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
                d = entry['decode']
                print(name, case['name'], 'IDs/text/stop', d['ids_exact'], d['text_exact'], d['same_stop'],
                    'EOS complete', d['complete_document'], 'confidence', d['native_mean_confidence'], flush=True)
        report['reproduction_failures'] = sum(not c['prior_original_output_exact'] or not c.get('prior_original_cpu_controls_exact', True)
            or not c.get('initial_reference_exact', True)
            for c in report['cases']) + sum(not c.get('prior_baseline_decode_exact', True) or not c.get('prior_prefill_exact', True)
                or not c.get('decode_prefill_exact', True) for a in report['arms'] for c in a['cases'])
        report['qualified_prefill_failure_count'] = sum(len(c['prefill']['qualified_failed_stages'])
            for a in report['arms'] for c in a['cases'] if 'prefill' in c)
        report['decoded_failure_count'] = sum(not (c['decode']['ids_exact'] and c['decode']['text_exact'] and c['decode']['same_stop'])
            for a in report['arms'] for c in a['cases'])
        report['result'] = 'completed fixed-panel decoded A/B; retained strict failures are not waived by exact output'
        if report['reproduction_failures'] or report['qualified_prefill_failure_count'] or report['decoded_failure_count']:
            raise AssertionError(f"Decoded panel retains {report['qualified_prefill_failure_count']} strict prefill failures, "
                f"{report['decoded_failure_count']} decoded differences and {report['reproduction_failures']} reproduction failures")
    finally:
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
