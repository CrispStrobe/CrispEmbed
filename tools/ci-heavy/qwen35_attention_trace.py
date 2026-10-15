"""Selected original/native vision attention boundaries with invalidating controls."""
import json
import os
import re
import subprocess
from collections import Counter
from qwen35_vision_operators import stage_records, prefill_records
from qwen35_cascade_blocks import decode_controls


def records(log):
    rows = stage_records(log, 'ATTN_METRIC')
    status = re.search(r'^ATTN_AUDIT finite=1 metrics=93 qualified_failures=(\d+) controls=(\d+) exact_controls=6$', log, re.M)
    expected = {'composed': 23, 'original-qkv-original-geometry': 14, 'original-qkv-native-geometry': 14,
                'original-norm-original-geometry': 16, 'original-norm-native-geometry': 16,
                'original-qk-v': 8, 'original-probabilities-v': 2}
    controls = [dict(name=m[0], passed=bool(int(m[1])), max_abs=float(m[2]),
                     native_elements=int(m[3]), expected_elements=int(m[4]))
                for m in re.findall(r'^ATTN_CONTROL (\S+) pass=(\d+) max_abs=(\S+) n=(\d+)/(\d+)$', log, re.M)]
    if (not status or len(rows) != 93 or Counter(m['mode'] for m in rows) != expected
            or len({(m['mode'], m['stage']) for m in rows}) != 93
            or sum(not m['qualified_pass'] for m in rows) != int(status[1])):
        raise ValueError('Incomplete finite attention diagnostics')
    expected_controls = {f'{kind}.{i}' for i in (5, 11) for kind in ('native-qkv', 'native-pooled', 'native-probabilities-pool')}
    if len(controls) != 6 or {c['name'] for c in controls} != expected_controls:
        raise ValueError('Incomplete native QKV/attention controls')
    for c in controls:
        if c['native_elements'] != c['expected_elements'] or c['native_elements'] < 1 or c['passed'] != (c['max_abs'] == 0):
            raise ValueError('Invalid native equality control')
    if bool(int(status[2])) != all(c['passed'] for c in controls):
        raise ValueError('Native attention control verdict mismatch')
    return dict(metrics=rows, native_controls=controls, controls_passed=bool(int(status[2])))


def check_prior(prior_arm, entry):
    prior = next(c for c in prior_arm['cases'] if c['name'] == entry['name'])
    entry['prior_prefill_exact'] = entry['prefill'] == prior['prefill']
    old = {m['stage']: m for m in prior['vision'] if m['mode'] == 'composed'}
    shared = [m for m in entry['attention']['metrics'] if m['mode'] == 'composed' and m['stage'] in old]
    entry['prior_shared_vision_metrics_exact'] = len(shared) == 9 and all(m == old[m['stage']] for m in shared)
    entry['first_argmax_exact'] = entry['prefill']['selected'] == entry['prefill']['reference']
    entry['vision_merger_matches_composed_prefill'] = (next(m['max_abs'] for m in entry['attention']['metrics']
        if m['mode'] == 'composed' and m['stage'] == 'prefill.vision.merger') == next(m['max_abs']
        for m in entry['prefill']['metrics'] if m['stage'] == 'prefill.vision.merger'))
    return all(entry[k] for k in ('prior_prefill_exact', 'prior_shared_vision_metrics_exact',
                                  'first_argmax_exact', 'vision_merger_matches_composed_prefill'))


def run_cases(repo, build, baseline, root, out, report):
    target = out / 'attention-trace.json'
    prior = json.loads((repo / 'tests/results/qwen35-ocr/cascade-blocks.json').read_text())
    commands = json.loads((build / 'compile_commands.json').read_text())
    compilation = [c['command'] for c in commands if c['file'].endswith('/test_qwen35_attention_trace.cpp')]
    if not compilation or any('CRISPEMBED_QWEN35_VISION_OPERATOR_AUDIT' in c for c in compilation):
        raise ValueError('Attention executable must link ordinary production kernels')
    report['scope'] = ('Original attention observations on blocks5/11; full Q/K/V and pooled tensors, '
        '16 deterministic query positions with all heads/keys for scores and probabilities; '
        'conditional same-input projection/geometry/reduction experiments; no runtime promotion')
    for case in report['cases']:
        original = next(c for c in prior['cases'] if c['name'] == case['name'])
        case['prior_original_cpu_control_metrics_exact'] = all(
            [m for m in rows if '.trace.' not in m['stage']] == original['reference']['cpu_controls'][name]
            for name, rows in case['reference']['cpu_controls'].items())
        metadata = case['reference']['attention_trace']
        if (metadata['original_calls'] != {'prefill': 2, 'backend_off': 2, 'patch_chunked': 2}
                or metadata['probability_and_output_exact_controls'] != 12
                or metadata['projection_input_exact_controls'] != 6):
            raise ValueError('Incomplete original attention observations')
    recipes = [('baseline', False, False, False), ('precise-both', True, True, False),
               ('cascade', False, False, True), ('cascade-attention', False, True, True)]
    for name, norm, attention, cascade in recipes:
        arm = dict(name=name, precise_norm=norm, precise_attention=attention, cascade_norm=cascade, cases=[])
        report['arms'].append(arm)
        prior_arm = next(a for a in prior['arms'] if a['name'] == name)
        env = dict(os.environ, CRISPEMBED_QWEN35_VISION_PRECISE_NORM=str(int(norm)),
            CRISPEMBED_QWEN35_VISION_PRECISE_ATTN=str(int(attention)), CRISPEMBED_QWEN35_VISION_CASCADE_NORM=str(int(cascade)),
            CRISPEMBED_QWEN35_PRECISE_RMS='0', CRISPEMBED_QWEN35_VISION_PROBES='0')
        for case in report['cases']:
            folder = root / case['name']; ref = folder / 'official-delta.gguf'
            process = subprocess.run([str(build / 'test-qwen35-attention-trace'), str(baseline), str(ref)],
                capture_output=True, text=True, env=env)
            log = process.stdout + process.stderr
            (folder / (name + '-attention.log')).write_text(log)
            entry = dict(name=case['name'], attention=records(log), attention_exit_code=process.returncode)
            arm['cases'].append(entry)
            expected_exit = int(not entry['attention']['controls_passed'] or any(not m['qualified_pass'] for m in entry['attention']['metrics']))
            if process.returncode != expected_exit: raise ValueError('Attention exit disagrees with controls/verdicts')
            process = subprocess.run([str(build / 'test-qwen35-quant'), str(baseline), str(folder / 'input.png'), str(ref), '', '1', '--prefill-only'],
                capture_output=True, text=True, env=env)
            log = process.stdout + process.stderr
            (folder / (name + '-prefill.log')).write_text(log); process.check_returncode()
            entry['prefill'] = prefill_records(log)
            entry['reproduction_controls_passed'] = check_prior(prior_arm, entry)
            target.write_text(json.dumps(report, indent=2) + '\n')
            print(name, case['name'], 'native controls', entry['attention']['controls_passed'],
                'prefill failures', entry['prefill']['qualified_failed_stages'], flush=True)
    decode_controls(build, baseline, root, out, report, recipes)
    report['qualified_failure_count'] = sum(sum(not m['qualified_pass'] for m in c['attention']['metrics'])
        + len(c['prefill']['qualified_failed_stages']) for a in report['arms'] for c in a['cases'])
    report['failed_native_controls'] = sum(not m['passed'] for a in report['arms'] for c in a['cases'] for m in c['attention']['native_controls'])
    report['reproduction_failures'] = sum(not c['reproduction_controls_passed'] for a in report['arms'] for c in a['cases'])
    report['original_cpu_reproduction_failures'] = sum(not c['prior_original_cpu_control_metrics_exact'] for c in report['cases'])
    report['result'] = 'completed unchanged-gate selected attention diagnostic; failed native controls invalidate traced arithmetic interpretation'
    target.write_text(json.dumps(report, indent=2) + '\n')
    if (report['qualified_failure_count'] or report['failed_native_controls'] or report['reproduction_failures']
            or report['original_cpu_reproduction_failures'] or any(not c['passed'] for c in report['small_decode_controls'])):
        raise AssertionError(f"Attention audit retains {report['qualified_failure_count']} qualified failures, {report['failed_native_controls']} failed native controls, and {report['reproduction_failures']} reproduction failures")
