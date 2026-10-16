"""Hosted, paired profiling attribution; no optimization/default promotion."""
import gc
import json
import math
import os
import re
import statistics
import subprocess

from qwen35_decoded_panel import RECIPES, validate
from qwen35_page_parity import parse
from qwen35_quant_calibration import errors, file_sha
from qwen35_vision_operators import prefill_records


OUTPUT_FIELDS = ('ids', 'text', 'confidences', 'stop', 'mean_confidence', 'bytes')


def parse_benchmark(log):
    def line(pattern):
        matches = re.findall(pattern, log, re.M)
        if len(matches) != 1:
            raise ValueError('Missing/repeated benchmark field: ' + pattern)
        return matches[0]
    ids = list(map(int, line(r'^BENCH ids:(.*)$').split()))
    probabilities = list(map(float, line(r'^BENCH confidences:(.*)$').split()))
    text = bytes.fromhex(line(r'^BENCH utf8_hex:(.*)$')).decode('utf-8')
    stop, confidence, size, load, elapsed = line(r'^BENCH stop=(\d+) mean_confidence=(\S+) bytes=(\d+) load_ms=(\S+) recognize_ms=(\S+)$')
    active, dropped, event_count, matrix_count = map(int, line(r'^PROFILE_COMPLETE active=(\d+) dropped=(\d+) events=(\d+) matrices=(\d+)$'))
    events = [dict(phase=m[0], event=m[1], calls=int(m[2]), nanoseconds=int(m[3])) for m in re.findall(
        r'^PROFILE phase=(\S+) event=(\S+) calls=(\d+) nanoseconds=(\d+)$', log, re.M)]
    matrices = [dict(phase=m[0], tensor=m[1], calls=int(m[2]), source_bytes=int(m[3]), f32_bytes=int(m[4]), materialize_ns=int(m[5]))
        for m in re.findall(r'^PROFILE_MATRIX phase=(\S+) tensor=(\S+) calls=(\d+) source_bytes=(\d+) f32_bytes=(\d+) materialize_ns=(\d+)$', log, re.M)]
    if (active not in (0, 1) or dropped or len(events) != event_count or len(matrices) != matrix_count
            or len({(m['phase'], m['event']) for m in events}) != event_count
            or len({(m['phase'], m['tensor']) for m in matrices}) != matrix_count
            or any(m['calls'] < 1 for m in events + matrices)
            or (not active and (events or matrices))):
        raise ValueError('Incomplete/dropped/duplicate/disabled profiling data')
    result = dict(ids=ids, confidences=probabilities, text=text, stop=int(stop), mean_confidence=float(confidence),
        bytes=int(size), load_ms=float(load), recognize_ms=float(elapsed), profile_active=bool(active), events=events, matrices=matrices)
    if (not ids or any(not 0 <= n < 248320 for n in ids) or len(ids) != len(probabilities)
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities + [result['mean_confidence']])
            or result['bytes'] != len(text.encode('utf-8')) or result['stop'] not in (1, 2)
            or any(not math.isfinite(result[k]) or result[k] <= 0 for k in ('load_ms', 'recognize_ms'))):
        raise ValueError('Invalid generated output/confidence/timing')
    if active:
        if not matrices or not events:
            raise ValueError('Active profiler has no measurements')
        lookup = {(m['phase'], m['event']): m for m in events}
        for phase, event, count in [('other', 'request_file', 1), ('other', 'request_raw', 1),
            ('other', 'file_decode', 1), ('other', 'preprocess', 1), ('other', 'prompt', 1),
            ('vision', 'vision', 1), ('other', 'embedding', 1), ('prefill', 'prefill', 1),
            ('other', 'selection', len(ids)), ('other', 'cached_prepare', len(ids)-1),
            ('decode', 'cached_decode', len(ids)-1), ('other', 'text_decode', 1)]:
            if lookup.get((phase, event), {}).get('calls', 0) != count:
                raise ValueError('Incomplete recognition phase counters')
        request = lookup['other', 'request_file']['nanoseconds'] / 1e6
        if not result['recognize_ms'] * 0.99 <= request <= result['recognize_ms']:
            raise ValueError('Request counter disagrees with external API timing')
        for phase in ('vision', 'prefill', 'decode'):
            rows = [m for m in matrices if m['phase'] == phase]
            if not rows:
                raise ValueError('Missing materialization phase')
            calls = sum(m['calls'] for m in rows)
            total = lookup[phase, 'project_total']['calls']
            for event in ('project_setup', 'project_input', 'project_compute', 'project_output', 'project_cleanup'):
                if lookup.get((phase, event), {}).get('calls') != total:
                    raise ValueError('Incomplete projection event counts')
            for event in ('project_materialize', 'project_graph'):
                if lookup.get((phase, event), {}).get('calls') != calls:
                    raise ValueError('Incomplete matrix event counts')
            pieces = sum(lookup[phase, event]['nanoseconds'] for event in ('project_setup', 'project_input',
                'project_materialize', 'project_graph', 'project_compute', 'project_output', 'project_cleanup'))
            if pieces > lookup[phase, 'project_total']['nanoseconds']:
                raise ValueError('Exclusive projection pieces exceed their parent')
            for event in ('source_buffer', 'source_read', 'source_convert', 'source_validate'):
                if lookup.get((phase, event), {}).get('calls', 0) < calls:
                    raise ValueError('Missing source conversion events')
    return result


def check_bytes(measured, schema):
    for m in measured['matrices']:
        tensor = schema[m['tensor']]
        elements = math.prod(tensor['shape'])
        expected_source = elements * {'float16': 2, 'float32': 4}[tensor['dtype']] * m['calls']
        if m['source_bytes'] != expected_source or m['f32_bytes'] != elements * 4 * m['calls']:
            raise ValueError('Materialization byte count disagrees with frozen weights')
    if measured['profile_active']:
        named = {(m['phase'], m['tensor']): m for m in measured['matrices']}
        if (named['prefill', 'l.embed_tokens.weight']['calls'] != 1
                or named['decode', 'l.embed_tokens.weight']['calls'] != len(measured['ids']) - 1):
            raise ValueError('Vocabulary head materialization count disagrees with generation')


def paired_summary(rows, parallel=False):
    key = 'parallel_f16' if parallel else 'profile_active'
    expected = [(0, False), (0, True), (1, True), (1, False), (2, False), (2, True)]
    if ([(r['pair'], r[key]) for r in rows] != expected
            or any(type(r[key]) is not bool for r in rows)
            or (parallel and any(r['profile_active'] or r['events'] or r['matrices'] for r in rows))):
        raise ValueError('Incomplete/nonalternating pairs or active profiler in conversion benchmark')
    summary = {}
    for enabled in (False, True):
        arm = [r for r in rows if r[key] == enabled]
        label = ('parallel_' if parallel else 'profile_') + ('on' if enabled else 'off')
        summary[label] = dict(median_ms=statistics.median(r['recognize_ms'] for r in arm),
            min_ms=min(r['recognize_ms'] for r in arm), max_ms=max(r['recognize_ms'] for r in arm),
            peak_rss_kib=max(r['resources']['max_rss_kib'] for r in arm))
    summary['median_paired_on_over_off'] = statistics.median(
        next(r['recognize_ms'] for r in rows if r['pair'] == pair and r[key]) /
        next(r['recognize_ms'] for r in rows if r['pair'] == pair and not r[key]) for pair in range(3))
    return summary


def run(repo, build, baseline, local, out, module, anchor_image, anchor_npz, parallel=False):
    from PIL import Image
    from transformers import AutoProcessor
    capture = module(repo / 'tools/dump_qwen35_ocr_page_reference.py', 'qwen35_profile_reference')
    frozen = json.loads((repo / 'tests/results/qwen35-ocr/decoded-panel.json').read_text())
    label = 'f16-materialization' if parallel else 'profiling'
    root = out / label; root.mkdir(exist_ok=True)
    target = out / (label + '.json')
    report = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        model=frozen['model'], revision=frozen['revision'],
        scope='Default-off diagnostic counters only, full baseline decoded controls, four-arm strict prefill reproduction, three alternating pairs per measured input; no optimization/default change',
        timing_scope='Fresh process/context each run. API recognize_file includes image decode/preprocess/vision/prefill/free decode, excludes model load and output printing. GNU time peak RSS includes load and recognition.',
        counter_scope='Nested counters: request_file includes request_raw; phase totals include projection totals; source conversion children also include small norms/biases outside projections. Do not sum nested levels.',
        publication='none', cases=[], validation=[], benchmarks=[])
    if parallel:
        report['scope'] = 'Default-off bounded parallel F16 conversion: candidate reproduces frozen four-arm prefill and baseline full decoded records; three alternating unprofiled off/on pairs per input'
        report['conversion_gate'] = 'CRISPEMBED_QWEN35_PARALLEL_F16'
        report['counter_scope'] = 'Profiling is disabled for both performance arms; all timing records must have empty events and matrices.'
        report['validation_gate'] = '1 for candidate strict prefill/full baseline decoding; performance pairs alternate0/1'
    env = dict(os.environ, CRISPEMBED_QWEN35_PARALLEL_F16=str(int(parallel)), CRISPEMBED_QWEN35_PROFILE='0', CRISPEMBED_QWEN35_VISION_PRECISE_NORM='0',
        CRISPEMBED_QWEN35_VISION_PRECISE_ATTN='0', CRISPEMBED_QWEN35_VISION_CASCADE_NORM='0',
        CRISPEMBED_QWEN35_PRECISE_RMS='0', CRISPEMBED_QWEN35_VISION_PROBES='0', CRISPEMBED_QWEN35_IMATRIX='0')
    try:
        report['sources'] = {name: file_sha(repo / name) for name in ('src/qwen35_math.h', 'src/qwen35_linear.cpp', 'src/qwen35_ocr.cpp',
            'src/qwen35_profile.cpp', 'src/qwen35_profile.h', 'src/qwen35_vision.cpp', 'src/qwen35_norm.cpp', 'tests/qwen35_parity.h')}
        for path, key in [('src/qwen35_vision.cpp', 'production_source_sha256'), ('src/qwen35_norm.cpp', 'norm_source_sha256')]:
            if report['sources'][path] != frozen[key]: raise ValueError('Vision/norm arithmetic changed')
        manifest = json.loads((out / 'export-manifest.json').read_text())
        report['weights'] = dict(bytes=baseline.stat().st_size, sha256=file_sha(baseline), tensor_audit=manifest['records'])
        if report['weights'] != frozen['weights']: raise ValueError('Changed frozen weights')
        schema = {t['name']: t for t in report['weights']['tensor_audit']}
        if parallel:
            report['materialized_arrays'] = json.loads((out / 'materialization-arrays.json').read_text())
            if not report['materialized_arrays']['python_bits_exact']:
                raise ValueError('Missing exact materialized-array controls')
        report['compact_anchor'] = capture.dump_reference(local, anchor_image, root / 'anchor', max_new_tokens=12, anchor=anchor_npz)
        gc.collect()
        for old in frozen['cases']:
            folder = root / old['name']; folder.mkdir(exist_ok=True)
            source = repo / old['source']['local_file']
            if old['source']['license'] not in ('CC0-1.0', 'Public-Domain') or file_sha(source) != old['source']['sha256']:
                raise ValueError('Source license/checksum changed')
            image = Image.open(source).convert('RGB')
            if old['crop']: image = image.crop(old['crop'])
            if old['explicit_resize']: image = image.resize(old['explicit_resize'], resample=Image.Resampling.BICUBIC)
            image.save(folder / 'input.png')
            if file_sha(folder / 'input.png') != old['input_sha256']: raise ValueError('Frozen derivative changed')
            case = {k: old[k] for k in ('name', 'source', 'crop', 'explicit_resize', 'input_sha256', 'max_new_tokens', 'gold', 'annotation_policy')}
            report['cases'].append(case)
            case['reference'] = capture.dump_reference(local, folder / 'input.png', folder,
                max_new_tokens=old['max_new_tokens'], prefill_controls=old['name'] != 'arabic-line', vision_probes=old['name'] != 'arabic-line')
            fields = ('transformers', 'tokenizers', 'torch', 'dtype', 'attention', 'image_sha256', 'image_size', 'grid', 'patches',
                'prompt', 'rendered_prompt', 'max_new_tokens', 'generated_ids', 'decoded', 'eos_ids', 'reached_eos', 'stop',
                'selected_token_confidences', 'mean_confidence')
            case['original_reference_exact'] = all(case['reference'][k] == old['reference'][k] for k in fields)
            case['original_cpu_controls_exact'] = case['reference'].get('cpu_controls') == old['reference'].get('cpu_controls')
            gc.collect()
        for name, norm, attention, cascade in RECIPES:
            arm = dict(name=name, cases=[]); report['validation'].append(arm)
            arm_env = dict(env, CRISPEMBED_QWEN35_VISION_PRECISE_NORM=str(int(norm)),
                CRISPEMBED_QWEN35_VISION_PRECISE_ATTN=str(int(attention)), CRISPEMBED_QWEN35_VISION_CASCADE_NORM=str(int(cascade)))
            for case in report['cases']:
                folder = root / case['name']
                command = [str(build / 'test-qwen35-quant'), str(baseline), str(folder / 'input.png'),
                    str(folder / 'official-delta.gguf'), '', str(case['max_new_tokens'])]
                entry = dict(name=case['name']); arm['cases'].append(entry)
                old = next(c for c in next(a for a in frozen['arms'] if a['name'] == name)['cases'] if c['name'] == case['name'])
                if case['name'] != 'arabic-line':
                    p = subprocess.run(command + ['--prefill-only'], capture_output=True, text=True, env=arm_env)
                    log = p.stdout + p.stderr; (folder / (name + '-prefill.log')).write_text(log); p.check_returncode()
                    entry['prefill'] = prefill_records(log)
                    entry['prior_prefill_exact'] = entry['prefill'] == old['prefill']
                if name == 'baseline':
                    p = subprocess.run(command, capture_output=True, text=True, env=arm_env)
                    log = p.stdout + p.stderr; (folder / 'baseline-decode.log').write_text(log); p.check_returncode()
                    entry['decode'] = validate(parse(log, case['reference']), case['reference'], case['max_new_tokens'])
                    fields = ('native_ids', 'native_text', 'ids_exact', 'text_exact', 'stop', 'native_mean_confidence',
                        'native_decoded_bytes', 'teacher_forced', 'metrics', 'score_summary')
                    entry['prior_decode_exact'] = all(entry['decode'][k] == old['decode'][k] for k in fields)
                target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
                print('validation', name, case['name'], 'prefill exact', entry.get('prior_prefill_exact'), 'decode exact', entry.get('prior_decode_exact'), flush=True)
        processor = AutoProcessor.from_pretrained(local, local_files_only=True, trust_remote_code=False)
        for name, cap in [('form', 192), ('german-header', 128), ('receipt', 32)]:
            case = next(c for c in report['cases'] if c['name'] == name)
            reference = case['reference']; expected_ids = reference['generated_ids'][:cap]
            expected_text = processor.batch_decode([expected_ids], skip_special_tokens=True)[0]
            expected_stop = 1 if expected_ids[-1] in reference['eos_ids'] else 2
            item = dict(name=name, token_cap=cap, prefix_control=cap < len(reference['generated_ids']), expected_ids=expected_ids,
                expected_text=expected_text, expected_stop=expected_stop, runs=[])
            report['benchmarks'].append(item)
            for pair in range(3):
                for enabled in ((0, 1) if pair % 2 == 0 else (1, 0)):
                    label = f'pair-{pair}-parallel-{enabled}' if parallel else f'pair-{pair}-profile-{enabled}'
                    folder = root / name
                    resource_path = folder / (label + '-resources.json')
                    command = ['/usr/bin/time', '-f', '{"max_rss_kib":%M,"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S}', '-o', str(resource_path),
                        str(build / 'test-qwen35-profile'), str(baseline), str(folder / 'input.png'), str(cap)]
                    p = subprocess.run(command, capture_output=True, text=True, env=dict(env, CRISPEMBED_QWEN35_PARALLEL_F16=str(enabled), CRISPEMBED_QWEN35_PROFILE='0') if parallel else dict(env, CRISPEMBED_QWEN35_PROFILE=str(enabled)))
                    log = p.stdout + p.stderr; (folder / (label + '.log')).write_text(log); p.check_returncode()
                    measured = parse_benchmark(log); check_bytes(measured, schema)
                    measured.update(pair=pair, resources=json.loads(resource_path.read_text()))
                    resource = measured['resources']
                    if (type(resource['max_rss_kib']) is not int or resource['max_rss_kib'] <= 0
                            or any(not math.isfinite(resource[k]) or resource[k] < 0
                                for k in ('wall_seconds', 'user_seconds', 'system_seconds'))):
                        raise ValueError('Invalid isolated process resource counters')
                    measured['output_matches_python'] = measured['ids'] == expected_ids and measured['text'] == expected_text and measured['stop'] == expected_stop
                    measured['paired'] = errors(expected_text, measured['text'])
                    measured['output_matches_first_run_exact'] = not item['runs'] or all(measured[k] == item['runs'][0][k] for k in OUTPUT_FIELDS)
                    if measured['profile_active'] != (False if parallel else bool(enabled)): raise ValueError('Profile gate mismatch')
                    if parallel: measured['parallel_f16'] = bool(enabled)
                    item['runs'].append(measured)
                    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
                    print(name, label, 'ms', measured['recognize_ms'], 'RSS KiB', measured['resources']['max_rss_kib'],
                        'exact', measured['output_matches_python'], measured['output_matches_first_run_exact'], flush=True)
            item['summary'] = paired_summary(item['runs'], parallel=parallel)
        report['reproduction_failures'] = sum(not c['original_reference_exact'] or not c['original_cpu_controls_exact'] for c in report['cases']) + sum(
            not c.get('prior_prefill_exact', True) or not c.get('prior_decode_exact', True) for a in report['validation'] for c in a['cases'])
        report['output_failures'] = sum(not r['output_matches_python'] or not r['output_matches_first_run_exact'] for b in report['benchmarks'] for r in b['runs'])
        report['qualified_prefill_failure_count'] = sum(len(c['prefill']['qualified_failed_stages']) for a in report['validation'] for c in a['cases'] if 'prefill' in c)
        report['result'] = ('completed gated F16 materialization paired timing/RSS and frozen numerical/output controls; no default promotion' if parallel else 'completed diagnostic profiling with paired overhead/RSS and frozen numerical/output controls; no optimization or promotion')
        if report['reproduction_failures'] or report['output_failures'] or report['qualified_prefill_failure_count']:
            raise AssertionError(f"Profiling retains {report['qualified_prefill_failure_count']} prefill failures, {report['reproduction_failures']} reproduction failures, {report['output_failures']} output differences")
    finally:
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
