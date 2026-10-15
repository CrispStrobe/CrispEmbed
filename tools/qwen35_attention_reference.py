"""Test-only observation of original eager vision attention; returns original outputs."""
import hashlib
from pathlib import Path


def install(visual, phase, arrays, save, patches):
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as blueprint
    original = blueprint.eager_attention_forward
    selected = {id(visual.blocks[i].attn): i for i in (5, 11)}
    count = min(16, patches)
    queries = torch.tensor([i * (patches - 1) // (count - 1) if count > 1 else 0
                            for i in range(count)], dtype=torch.long)
    save('input.vision_attention_queries', queries)
    controls = {}

    def observe(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        output, probabilities = original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        if id(module) not in selected:
            return output, probabilities
        if attention_mask is not None or dropout != 0 or module.training or module.num_key_value_groups != 1:
            raise ValueError('Attention observation requires unmasked F32 eager evaluation')
        if query.dtype != torch.float32 or query.shape != key.shape or query.shape != value.shape or query.shape[0] != 1 or query.shape[2] != patches:
            raise ValueError('Unexpected original attention layout')
        scores = torch.matmul(query, key.transpose(2, 3)) * scaling
        reconstructed = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
        pooled = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
        if not torch.equal(probabilities, reconstructed) or not torch.equal(output, pooled):
            raise ValueError('Attention reconstruction differs from original function output')
        prefix = f'prefill.vision.probe.{selected[id(module)]}.trace.'
        for name, tensor in (('query', query), ('key', key), ('value', value)):
            save(prefix + name, tensor[0].transpose(0, 1).reshape(patches, -1))
        save(prefix + 'scores', scores[0].index_select(1, queries))
        save(prefix + 'probabilities', probabilities[0].index_select(1, queries))
        save(prefix + 'pooled_sample', output.reshape(patches, -1).index_select(0, queries))
        controls[phase[0]] = controls.get(phase[0], 0) + 1
        return output, probabilities

    blueprint.eager_attention_forward = observe

    def check_pool():
        for i in (5, 11):
            prefix = f'{phase[0]}.vision.probe.{i}.'
            observed = arrays[prefix + 'trace.pooled_sample']
            expected = arrays[prefix + 'pooled'][queries.numpy()]
            if not (observed.shape == expected.shape and (observed == expected).all()):
                raise ValueError('Observed eager output differs from original projection input')

    def restore():
        blueprint.eager_attention_forward = original

    def finish(prefill_controls):
        restore()
        expected_phases = {'prefill', 'backend_off', 'patch_chunked'} if prefill_controls else {'prefill'}
        if set(controls) != expected_phases or any(v != 2 for v in controls.values()):
            raise ValueError('Incomplete original selected attention calls')
        return dict(selected_blocks=[5, 11], query_positions=queries.tolist(), full_key_axis=True,
                    original_calls=controls, probability_and_output_exact_controls=2 * sum(controls.values()),
                    projection_input_exact_controls=sum(controls.values()),
                    source_revision='75d3bdcd4b3ba70cee21287218d9764f33da41f0',
                    installed_source_sha256=hashlib.sha256(Path(blueprint.__file__).read_bytes()).hexdigest(),
                    scope='Full original Q/K/V and pooled inputs; scores/probabilities sampled on query axis only. Original eager results returned unchanged.')
    return check_pool, finish, restore
