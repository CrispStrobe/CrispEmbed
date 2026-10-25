"""Pure, explicit OvisOCR2 attention-F16 / FFN-Q4_K recipe and tensor audit.

The frozen quality contract owns the 473-tensor source schema and protected
tensor policy. This module consumes that contract without modifying its old
recipes, constants, or functions. No model/framework imports occur here.
"""

import re


PHASE = 'q4-attention-f16'
RECIPE = 'q4-attention-f16'
TARGETS = ('down_proj', 'gate_proj', 'up_proj')
_FFN = re.compile(r'l\.layers\.(\d+)\.mlp\.(down_proj|gate_proj|up_proj)\.weight\Z')
_HEX64 = re.compile(r'[0-9a-f]{64}\Z')


def target_names(schema, contract):
    """Select exactly three eligible language FFN matrices per decoder layer."""
    if not isinstance(schema, dict) or len(schema) != 473:
        raise ValueError('Selective OvisOCR2 source schema is not 473 tensors')
    names = set()
    for name, shape in schema.items():
        match = _FFN.fullmatch(name)
        if match is None:
            continue
        if (not isinstance(shape, list) or len(shape) != 2
                or shape[1] % 256 != 0 or contract._protected(name, shape)):
            raise ValueError('Selective OvisOCR2 FFN target is ineligible: ' + name)
        names.add(name)
    expected = {f'l.layers.{layer}.mlp.{projection}.weight'
                for layer in range(24) for projection in TARGETS}
    if names != expected:
        raise ValueError('Selective OvisOCR2 FFN target set differs')
    return names


def _checked_baseline(before, schema, targets):
    if not isinstance(before, dict) or set(before) != set(schema):
        raise ValueError('Selective OvisOCR2 F16 tensor coverage differs')
    for name, shape in schema.items():
        row = before[name]
        if (not isinstance(row, dict) or set(row) != {'shape', 'type', 'sha256'}
                or row['shape'] != list(reversed(shape))
                or row['type'] not in ('F16', 'F32')
                or not isinstance(row['sha256'], str)
                or _HEX64.fullmatch(row['sha256']) is None):
            raise ValueError('Selective OvisOCR2 F16 record differs: ' + name)
        if name in targets and row['type'] != 'F16':
            raise ValueError('Selective OvisOCR2 FFN baseline is not F16: ' + name)


def quant_command(binary, baseline, candidate, before, contract, schema=None):
    """Use Q4_K by default and exact-name overrides for all 401 other tensors.

    The frozen quantizer supports f32/f16 overrides but not q4_k overrides.
    Therefore targets use the base q4_k type, while each other tensor receives
    an explicit old-type rule. The post-conversion audit proves byte identity.
    """
    schema = contract.source_schema() if schema is None else schema
    targets = target_names(schema, contract)
    _checked_baseline(before, schema, targets)
    command = [str(binary), str(baseline), str(candidate), 'q4_k']
    for name in sorted(set(schema) - targets):
        command += ['--tensor-type', name + '=' + before[name]['type'].lower()]
    if len(command) != 4 + 2 * 401:
        raise ValueError('Selective OvisOCR2 non-FFN override count differs')
    return command


def validate_quant_tensors(before, after, contract, schema=None):
    """Require only 72 Q4_K FFNs to change; all 401 others retain raw bytes."""
    schema = contract.source_schema() if schema is None else schema
    targets = target_names(schema, contract)
    _checked_baseline(before, schema, targets)
    if not isinstance(after, dict) or set(after) != set(schema):
        raise ValueError('Selective OvisOCR2 candidate tensor coverage differs')
    attention = {name for name, shape in schema.items()
                 if name.startswith('l.layers.') and len(shape) == 2
                 and ('.self_attn.' in name or '.linear_attn.' in name)
                 and not contract._protected(name, shape)}
    if len(attention) != 78 or attention & targets:
        raise ValueError('Selective OvisOCR2 attention inventory differs')
    for name, old in before.items():
        new = after[name]
        if not isinstance(new, dict) or set(new) != set(old):
            raise ValueError('Selective OvisOCR2 tensor record differs: ' + name)
        if name in targets:
            if (new['shape'] != old['shape'] or new['type'] != 'Q4_K'
                    or not isinstance(new['sha256'], str)
                    or _HEX64.fullmatch(new['sha256']) is None
                    or new['sha256'] == old['sha256']):
                raise ValueError('Selective OvisOCR2 FFN Q4_K conversion differs: ' + name)
        elif new != old:
            raise ValueError('Selective OvisOCR2 non-FFN tensor bytes/type differ: ' + name)
    return {'recipe': RECIPE, 'tensor_count': 473, 'q4_k_ffn_count': 72,
            'unchanged_tensor_count': 401, 'unchanged_attention_matrix_count': 78,
            'non_ffn_byte_exact': True}
