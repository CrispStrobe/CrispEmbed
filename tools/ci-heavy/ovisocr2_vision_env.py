"""Pure reviewed OvisOCR2 vision gate transfer for hosted follow-on lanes."""

FLAGS = ('CRISPEMBED_QWEN35_VISION_PRECISE_NORM',
         'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN',
         'CRISPEMBED_QWEN35_VISION_CASCADE_NORM')
PRESETS = {
    'baseline': (),
    'precise_norm': (FLAGS[0],),
    'precise_attn': (FLAGS[1],),
    'both': (FLAGS[0], FLAGS[1]),
    'cascade': (FLAGS[2],),
    'cascade_precise_attn': (FLAGS[2], FLAGS[1]),
}


def reviewed_selection(producer):
    """Reject an absent, ambiguous, or ambiently altered baseline vision mode."""
    if not isinstance(producer, dict):
        raise ValueError('Reviewed OvisOCR2 baseline producer absent')
    selected = producer.get('selected_vision_preset')
    recorded = producer.get('native_environment')
    if not isinstance(selected, dict) or not isinstance(recorded, dict):
        raise ValueError('Reviewed OvisOCR2 baseline vision environment absent')
    name = selected.get('name')
    if (name not in PRESETS or selected.get('enabled_gates') != list(PRESETS[name])
            or selected.get('all_other_native_gates_unchanged') is not True
            or selected.get('no_production_default_promoted') is not True):
        raise ValueError('Reviewed OvisOCR2 selected vision preset differs')
    expected = {flag: '1' if flag in PRESETS[name] else '0' for flag in FLAGS}
    if any(recorded.get(flag) not in ('0', '1') or recorded[flag] != expected[flag]
           for flag in FLAGS):
        raise ValueError('Reviewed OvisOCR2 baseline vision flag values differ')
    return {'name': name, 'enabled_gates': list(PRESETS[name]), 'values': expected}


def apply_selection(env, selection):
    """Set only the three audited precision flags; preserve all other controls."""
    if (not isinstance(selection, dict) or selection.get('name') not in PRESETS
            or selection.get('enabled_gates') != list(PRESETS[selection['name']])
            or selection.get('values') != {flag: '1' if flag in PRESETS[selection['name']]
                                           else '0' for flag in FLAGS}):
        raise ValueError('Validated OvisOCR2 vision selection required')
    result = dict(env)
    result.update(selection['values'])
    return result
