"""Frozen OvisOCR2 OCR panel and candidate quantization contract.

This module imports only the standard library. Hosted drivers render images,
run the pinned original and native models, and supply actual tensor records.
No result from this helper alone qualifies an alias or a public model.
"""

import fnmatch
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
import unicodedata


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / 'tests/results/qwen35-ocr/ovisocr2-quality-panel.json'
EVIDENCE_PATH = ROOT / 'tests/results/qwen35-ocr/ovisocr2-metadata.json'
PANEL_SHA = '0bec97a7a04683c7c0aec95063aa01cf9a73bcf78db61f433d7d66d89588f212'
PRODUCER_SHA = '233dc37fdaa2ff6b4a8b3ca3832b578305b5f3b6ed4d0d88d54c73591626b372'
STOPS = {248044, 248046}

# Every recipe starts with original F16 vision/embedding and recurrent gates.
# One bounded Q8 candidate precedes any more aggressive Q4 exploration.
RECIPES = {
    'q8-protected': ('q8_0', (
        'v.*=f16', 'l.embed_tokens.weight=f16',
        'l.layers.*.linear_attn.in_proj_a.weight=f16',
        'l.layers.*.linear_attn.in_proj_b.weight=f16',
    )),
    'q4-attention-q8': ('q4_k', (
        'v.*=f16', 'l.embed_tokens.weight=f16',
        'l.layers.*.self_attn.*.weight=q8_0',
        'l.layers.*.linear_attn.in_proj_qkv.weight=q8_0',
        'l.layers.*.linear_attn.in_proj_z.weight=q8_0',
        'l.layers.*.linear_attn.out_proj.weight=q8_0',
        'l.layers.*.linear_attn.in_proj_a.weight=f16',
        'l.layers.*.linear_attn.in_proj_b.weight=f16',
    )),
}


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def load_panel(path=PANEL_PATH):
    panel = json.loads(Path(path).read_text())
    if canonical_sha(panel) != PANEL_SHA or panel['license'] != 'CC0-1.0':
        raise ValueError('Frozen OvisOCR2 quality panel differs')
    rows = panel['cases']
    if (len(rows) != 8 or len({row['id'] for row in rows}) != 8
            or [row['split'] for row in rows] != ['development'] * 4 + ['heldout'] * 4
            or len({row['gold'] for row in rows}) != 8
            or panel['max_new_tokens'] != 256):
        raise ValueError('OvisOCR2 development/heldout panel contract differs')
    return panel


def render_panel(destination, panel=None):
    """Render once on hosted CPU before any model result; save exact RGB/PNG hashes."""
    panel = load_panel() if panel is None else panel
    if canonical_sha(panel) != PANEL_SHA:
        raise ValueError('Unreviewed quality panel')
    if importlib.metadata.version('Pillow') != panel['renderer']['version']:
        raise ValueError('Pillow renderer version differs')
    from PIL import Image, ImageDraw

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for row in panel['cases']:
        image = Image.new('RGB', tuple(panel['renderer']['size']), panel['renderer']['background'])
        ImageDraw.Draw(image).text((row['x'], row['y']), row['gold'],
                                   fill=panel['renderer']['fill'], font_size=row['font_size'])
        file = root / (row['id'] + '.png')
        image.save(file)
        records.append({'id': row['id'], 'split': row['split'], 'gold': row['gold'],
                        'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                        'png_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                        'file': file.name})
    receipt = {'panel_canonical_sha256': PANEL_SHA, 'renderer_version': panel['renderer']['version'],
               'license': panel['license'], 'cases': records,
               'status': 'rendered before any candidate outputs'}
    (root / 'rendered-panel.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return receipt


def source_schema(path=EVIDENCE_PATH):
    """Return audited GGUF shapes; the producer evidence retains original HF shapes."""
    evidence = json.loads(Path(path).read_text())
    producer = evidence['producer']
    if (canonical_sha(producer) != PRODUCER_SHA or evidence['run'] != 39983101540
            or producer['schema']['checkpoint_tensors'] != 473
            or producer['schema']['tied_head_absent'] is not True):
        raise ValueError('Original OvisOCR2 schema proof differs')
    mapping = {}
    for name, shape in producer['schema']['official_state_dict'].items():
        if name == 'lm_head.weight':
            continue
        if name.startswith('model.visual.'):
            mapped = 'v.' + name.removeprefix('model.visual.')
        elif name.startswith('model.language_model.'):
            mapped = 'l.' + name.removeprefix('model.language_model.')
        else:
            raise ValueError('Unknown original OvisOCR2 tensor: ' + name)
        if mapped == 'v.patch_embed.proj.weight':
            # The pinned exporter flattens only this Conv3d [out,C,T,H,W]
            # into a GGUF matrix. The original five-axis shape remains in
            # the reviewed producer evidence and is checked here exactly.
            if shape != [768, 3, 2, 16, 16]:
                raise ValueError('Original OvisOCR2 patch source shape differs')
            shape = [768, 1536]
        elif len(shape) > 4:
            raise ValueError('Unexpected unflattened OvisOCR2 tensor: ' + name)
        mapping[mapped] = shape
    if len(mapping) != 473:
        raise ValueError('OvisOCR2 mapped schema differs')
    return mapping


def quant_command(binary, baseline, candidate, recipe, schema=None):
    if recipe not in RECIPES:
        raise ValueError('Unknown OvisOCR2 quantization recipe')
    schema = source_schema() if schema is None else schema
    kind, overrides = RECIPES[recipe]
    for rule in overrides:
        pattern, _ = rule.rsplit('=', 1)
        if not any(fnmatch.fnmatchcase(name, pattern) for name in schema):
            raise ValueError('Unmatched OvisOCR2 quantization rule: ' + pattern)
    command = [str(binary), str(baseline), str(candidate), kind]
    for rule in overrides:
        command += ['--tensor-type', rule]
    return command


def _protected(name, shape):
    return (len(shape) != 2 or name.startswith('v.') or name == 'l.embed_tokens.weight'
            or '.norm.' in name or 'layernorm' in name
            or name.endswith(('.linear_attn.in_proj_a.weight', '.linear_attn.in_proj_b.weight')))


def validate_quant_tensors(before, after, recipe, schema=None):
    """Check actual GGUF name/shape/type/raw-byte hashes after quantization."""
    if recipe not in RECIPES:
        raise ValueError('Unknown OvisOCR2 quantization recipe')
    schema = source_schema() if schema is None else schema
    if set(before) != set(schema) or set(after) != set(schema):
        raise ValueError('OvisOCR2 quantized tensor coverage differs')
    kind, overrides = RECIPES[recipe]
    changed = 0
    for name, old in before.items():
        new = after[name]
        expected_shape = list(reversed(schema[name]))
        if old['shape'] != expected_shape or new['shape'] != expected_shape:
            raise ValueError('OvisOCR2 quantized tensor shape differs: ' + name)
        if _protected(name, schema[name]):
            if new != old:
                raise ValueError('Protected OvisOCR2 tensor bytes/type differ: ' + name)
            continue
        expected = kind.upper()
        for rule in overrides:
            pattern, precision = rule.rsplit('=', 1)
            if fnmatch.fnmatchcase(name, pattern):
                expected = precision.upper()
        if new['type'].upper() != expected:
            raise ValueError('OvisOCR2 quantized tensor precision differs: ' + name)
        if new['sha256'] != old['sha256']:
            changed += 1
    if changed == 0:
        raise ValueError('No OvisOCR2 matrix payload was quantized')
    return {'recipe': recipe, 'tensor_count': len(schema), 'changed_matrix_count': changed,
            'protected_byte_exact': True}


def _distance(left, right):
    previous = list(range(len(right) + 1))
    for index, a in enumerate(left, 1):
        row = [index]
        for j, b in enumerate(right, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = row
    return previous[-1]


def score_case(gold, output, cap=256):
    """Keep raw output and separately score visible text and stop/format gates."""
    raw = output['raw_text']
    processed = output['processed_text']
    ids = output['token_ids']
    if (not isinstance(raw, str) or not isinstance(processed, str) or not isinstance(ids, list)
            or type(cap) is not int or cap < 1 or not 0 < len(ids) <= cap
            or any(type(token) is not int or not 0 <= token < 248320 for token in ids)):
        raise ValueError('Incomplete OCR output record')
    clean = lambda text: unicodedata.normalize('NFC', text.replace('\r\n', '\n')).strip()
    reference, actual = clean(gold), clean(processed)
    stop = output.get('stop_id', output.get('stop_reason'))
    finish = output.get('finish_reason')
    native_stop = output.get('stop_kind')
    reached_eos = ids[-1] in STOPS
    native_disagrees = native_stop is not None and (type(native_stop) is not int
                                                   or native_stop not in (1, 2, 3)
                                                   or (native_stop == 1) != reached_eos)
    if (stop != (ids[-1] if reached_eos else None)
            or ('stop_id' in output and 'stop_reason' in output
                and output['stop_id'] != output['stop_reason'])
            or any(token in STOPS for token in ids[:-1])
            or finish != ('stop' if reached_eos else 'length')
            or native_disagrees):
        raise ValueError('Contradictory OvisOCR2 generated IDs/EOS/finish reason')
    if not reached_eos and ((native_stop is None and len(ids) != cap)
                            or (native_stop == 2 and len(ids) != cap)):
        raise ValueError('Unexplained OvisOCR2 length stop before cap')
    issues = []
    if not actual:
        issues.append('empty')
    if '<img src="images/bbox_' in actual:
        issues.append('unexpected-image-tag')
    if any(ord(char) < 32 and char not in '\n\t' for char in actual):
        issues.append('control-character')
    if actual.count('```') % 2:
        issues.append('unclosed-code-fence')
    cap_hit = not reached_eos and len(ids) == cap
    completed = reached_eos and not issues
    return {'raw_text': raw, 'processed_text': processed, 'gold': gold,
            'raw_exact': clean(raw) == reference, 'processed_exact': actual == reference,
            'cer': _distance(reference, actual) / max(1, len(reference)),
            'wer': _distance(reference.split(), actual.split()) / max(1, len(reference.split())),
            'generated_tokens': len(ids), 'cap_hit': cap_hit, 'completed': completed,
            'stop_id': stop, 'finish_reason': finish, 'stop_kind': native_stop,
            'format_issues': issues}


def score_panel(rendered, outputs, panel=None):
    panel = load_panel() if panel is None else panel
    if (canonical_sha(panel) != PANEL_SHA or rendered['panel_canonical_sha256'] != PANEL_SHA
            or [row['id'] for row in rendered['cases']] != [row['id'] for row in panel['cases']]
            or set(outputs) != {row['id'] for row in panel['cases']}):
        raise ValueError('OvisOCR2 rendered/evaluated case set differs')
    scores = []
    for spec, image in zip(panel['cases'], rendered['cases']):
        output = outputs[spec['id']]
        if (output.get('input_rgb_sha256') != image['rgb_sha256']
                or output.get('input_png_sha256') != image['png_sha256']):
            raise ValueError('OvisOCR2 quality input bytes differ: ' + spec['id'])
        scores.append({'id': spec['id'], 'split': spec['split'],
                       'input_rgb_sha256': image['rgb_sha256'],
                       **score_case(spec['gold'], output, panel['max_new_tokens'])})
    summary = {}
    for split in ('development', 'heldout'):
        group = [row for row in scores if row['split'] == split]
        summary[split] = {'cases': len(group), 'complete': sum(row['completed'] for row in group),
                          'exact': sum(row['processed_exact'] for row in group),
                          'cap_hits': sum(row['cap_hit'] for row in group),
                          'format_failures': sum(bool(row['format_issues']) for row in group),
                          'mean_cer': sum(row['cer'] for row in group) / len(group),
                          'mean_wer': sum(row['wer'] for row in group) / len(group)}
    return {'panel_canonical_sha256': PANEL_SHA, 'license': panel['license'],
            'cases': scores, 'summary': summary,
            'scope': 'Self-authored short synthetic text only; not representative document OCR quality'}


def paired_quality_screen(reference, candidate):
    """Strict heldout screen; a pass still requires separate numerical/release review."""
    panel = load_panel()
    expected = [(row['id'], row['split'], row['gold']) for row in panel['cases']]
    if (reference['panel_canonical_sha256'] != PANEL_SHA
            or candidate['panel_canonical_sha256'] != PANEL_SHA
            or [(row['id'], row['split'], row['gold']) for row in reference['cases']] != expected
            or [(row['id'], row['split'], row['gold']) for row in candidate['cases']] != expected):
        raise ValueError('OvisOCR2 paired quality panels differ')
    failures = []
    for old, new in zip(reference['cases'], candidate['cases']):
        required = ('raw_text', 'processed_text', 'input_rgb_sha256', 'generated_tokens',
                    'cap_hit', 'completed', 'format_issues', 'cer', 'wer', 'finish_reason',
                    'raw_exact', 'processed_exact', 'stop_id')
        if (any(key not in row for row in (old, new) for key in required)
                or any(not isinstance(row['raw_text'], str) or not isinstance(row['processed_text'], str)
                       or not isinstance(row['format_issues'], list)
                       or any(not isinstance(issue, str) for issue in row['format_issues'])
                       or type(row['generated_tokens']) is not int or row['generated_tokens'] < 1
                       or type(row['cap_hit']) is not bool or type(row['completed']) is not bool
                       or type(row['raw_exact']) is not bool or type(row['processed_exact']) is not bool
                       or row['finish_reason'] not in ('stop', 'length')
                       or row['stop_id'] not in STOPS | {None}
                       or not isinstance(row['input_rgb_sha256'], str)
                       or not re.fullmatch('[0-9a-f]{64}', row['input_rgb_sha256'])
                       or any(not isinstance(row[metric], (int, float)) or not math.isfinite(row[metric])
                              or row[metric] < 0 for metric in ('cer', 'wer'))
                       for row in (old, new))):
            raise ValueError('Incomplete OvisOCR2 paired quality case')
        if old['input_rgb_sha256'] != new['input_rgb_sha256']:
            raise ValueError('OvisOCR2 paired input pixels differ')
        if old['split'] != 'heldout':
            continue
        if not old['completed'] or old['format_issues']:
            failures.append(old['id'] + ': original reference incomplete or malformed')
        if not new['completed'] or new['format_issues']:
            failures.append(new['id'] + ': candidate incomplete or malformed')
        if new['cer'] > old['cer'] or new['wer'] > old['wer']:
            failures.append(new['id'] + ': heldout OCR error increased')
    return {'screen_passed': not failures, 'failures': failures,
            'requires_separate_numerical_and_release_review': True,
            'scope': 'Paired heldout nonregression on four short owned synthetic pages only'}


def paired_document_screen(reference, candidate):
    """Fail closed on the separately reviewed three-document OCR and table panel."""
    expected = ('historical-receipt', 'two-column-page', 'inventory-table')
    if (tuple(row.get('id') for row in reference) != expected
            or tuple(row.get('id') for row in candidate) != expected):
        raise ValueError('Reviewed realistic OCR document set/order differs')
    failures = []
    for old, new in zip(reference, candidate):
        if (old.get('rgb_sha256') != new.get('rgb_sha256')
                or not re.fullmatch('[0-9a-f]{64}', old.get('rgb_sha256', ''))
                or old.get('license') != new.get('license')):
            raise ValueError('Reviewed realistic OCR pixel/license identity differs')
        for side, row in (('original', old), ('candidate', new)):
            structure = row.get('structure', {})
            if (row.get('cap_hit') or not row.get('completed')
                    or row.get('format_issues') or structure.get('complete') is not True
                    or structure.get('parse_ok') is not True):
                failures.append(row['id'] + ': ' + side + ' incomplete or malformed')
        left, right = old.get('structure', {}), new.get('structure', {})
        for view in ('raw', 'content'):
            for metric in ('cer', 'wer'):
                a = (left.get(view) or {}).get(metric)
                b = (right.get(view) or {}).get(metric)
                if (not isinstance(a, (int, float)) or not isinstance(b, (int, float))
                        or not math.isfinite(a) or not math.isfinite(b) or a < 0 or b < 0):
                    failures.append(old['id'] + ': missing finite ' + view + '/' + metric)
                elif b > a:
                    failures.append(old['id'] + ': ' + view + '/' + metric + ' increased')
        if old['id'] == 'inventory-table':
            old_table, new_table = left.get('table') or {}, right.get('table') or {}
            if (old_table.get('expected_cells') != 15 or new_table.get('expected_cells') != 15
                    or not isinstance(new_table.get('cell_errors'), int)
                    or not isinstance(old_table.get('cell_errors'), int)
                    or new_table['cell_errors'] > old_table['cell_errors']
                    or new_table['cell_errors'] != 0):
                failures.append('inventory-table: 15-cell grid failed or regressed')
    return {'screen_passed': not failures, 'failures': failures,
            'scope': 'Three reviewed licensed documents; completion, raw/content CER/WER, and 15 table cells'}
