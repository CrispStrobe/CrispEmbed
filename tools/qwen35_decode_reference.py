"""Observe selected original generate forwards; test-only, no arithmetic replacement."""


def validate_steps(steps, cap):
    if (not isinstance(steps, (list, tuple)) or len(steps) > 4
            or any(type(v) is not int or not 0 < v < cap for v in steps)
            or list(steps) != sorted(set(steps))):
        raise ValueError('Decode probes require up to four ordered unique continuation steps within cap')
    return tuple(steps)


def install(model, steps, save):
    """Step n is the forward producing generated.scores.n, before selecting ID n."""
    language = model.model.language_model
    active = [-1]
    seen = []
    handles = []

    def input_hook(module, args, kwargs):
        active[0] += 1
        if active[0] in steps:
            seen.append(active[0])
            save(f'generated.probe.{active[0]}.decoder.inputs_embeds', kwargs['inputs_embeds'])

    def output_hook(label):
        def capture(module, args, value):
            if active[0] in steps:
                save(f'generated.probe.{active[0]}.{label}', value[0] if isinstance(value, tuple) else value)
        return capture

    def cache_hook(module, args, output):
        if active[0] in steps:
            prefix = f'generated.probe.{active[0]}.'
            save(prefix + 'logits', output.logits[:, -1])
            for index, layer in enumerate(output.past_key_values.layers):
                for name in ('conv_states', 'recurrent_states', 'keys', 'values'):
                    value = getattr(layer, name, None)
                    if value is not None:
                        save(prefix + f'cache.{index}.{name}', value)

    handles.append(language.register_forward_pre_hook(input_hook, with_kwargs=True))
    for index, block in enumerate(language.layers):
        handles.append(block.register_forward_hook(output_hook(f'decoder.layer.{index}')))
    handles.append(language.norm.register_forward_hook(output_hook('decoder.norm')))
    handles.append(model.register_forward_hook(cache_hook))

    def remove():
        for handle in handles:
            handle.remove()

    def finish(count):
        if seen != list(steps) or active[0] + 1 != count:
            raise ValueError('Original generate probe steps missing or misaligned')
        return dict(steps=list(steps), forward_calls=count, fields_per_step=75,
                    scope='Unchanged original generate forwards; observers removed for exact ID/score replay')

    return remove, finish
