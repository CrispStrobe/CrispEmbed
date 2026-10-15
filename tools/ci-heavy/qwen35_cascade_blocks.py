"""Controls for native block replay across the retained Qwen vision alternatives."""
import json
import os
import re
import subprocess


def check_prior(repo, name, entry):
    results = repo / 'tests/results/qwen35-ocr'
    if name in ('baseline', 'precise-both'):
        prior = json.loads((results / 'vision-blocks.json').read_text())
        arm = next(a for a in prior['arms'] if a['name'] == name)
        case = next(c for c in arm['cases'] if c['name'] == entry['name'])
        entry['prior_block_metrics_exact'] = entry['vision'] == case['vision']
        entry['prior_block_controls_exact'] = entry['block_controls'] == case['block_controls']
        prefill = case['prefill']
    else:
        prior = json.loads((results / 'cascade-norm.json').read_text())
        case = next(c for c in prior['arms'][0]['cases'] if c['name'] == entry['name'])
        prefill = next(a['prefill'] for a in case['runtime_prefill'] if a['name'] == name)
    entry['prior_prefill_exact'] = entry['prefill'] == prefill
    entry['first_argmax_exact'] = entry['prefill']['selected'] == entry['prefill']['reference']
    controls = ('prior_prefill_exact', 'first_argmax_exact', 'vision_merger_matches_composed_prefill')
    if name in ('baseline', 'precise-both'):
        controls += ('prior_block_metrics_exact', 'prior_block_controls_exact')
    if not all(entry[k] for k in controls):
        raise ValueError('Native replay failed frozen reproduction controls: ' + name + '/' + entry['name'])


def decode_controls(build, baseline, root, out, report, recipes):
    report['small_decode_controls'] = []
    for name, norm, attention, cascade in recipes:
        env = dict(os.environ, CRISPEMBED_QWEN35_VISION_PRECISE_NORM=str(int(norm)),
                   CRISPEMBED_QWEN35_VISION_PRECISE_ATTN=str(int(attention)),
                   CRISPEMBED_QWEN35_VISION_CASCADE_NORM=str(int(cascade)),
                   CRISPEMBED_QWEN35_PRECISE_RMS='0', CRISPEMBED_QWEN35_VISION_PROBES='0')
        result = subprocess.run([str(build / 'test-qwen35-quant'), str(baseline),
            str(out / 'reference-input.png'), str(root / 'compact-anchor/official-delta.gguf'), '', '12'],
            capture_output=True, text=True, env=env)
        log = result.stdout + result.stderr
        (root / (name + '-small-decode.log')).write_text(log)
        matched = re.search(r'^FREE ids_exact=(\d+) text_exact=(\d+) stop=(\d+) mean_confidence=(\S+) bytes=(\d+)$', log, re.M)
        ids = re.search(r'^FREE ids:(.*)$', log, re.M)
        text = re.search(r'^FREE utf8_hex:(.*)$', log, re.M)
        if not matched or not ids or not text:
            raise ValueError('Incomplete actual small decode: ' + name)
        generated = list(map(int, ids[1].split()))
        decoded = bytes.fromhex(text[1]).decode('utf-8')
        passed = (result.returncode == 0 and tuple(matched.group(i) for i in (1, 2, 3)) == ('1', '1', '1')
                  and generated == report['compact_anchor']['generated_ids']
                  and decoded == report['compact_anchor']['decoded']
                  and len(decoded.encode('utf-8')) == int(matched[5]))
        report['small_decode_controls'].append(dict(name=name, passed=passed, exit_code=result.returncode,
            generated_ids=generated, decoded=decoded, stop=int(matched[3]),
            mean_confidence=float(matched[4]), bytes=int(matched[5])))
