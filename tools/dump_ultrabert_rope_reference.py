#!/usr/bin/env python3
"""Additional same-input original RoPE observations; never a full-model recapture."""
import hashlib
import inspect
import json
from pathlib import Path
import sys

from ultrabert_contract import MODEL, REVISION, profile, sha256, validate_sources


def metrics(actual, expected):
    import numpy as np
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid operator comparison shape/values')
    ar, br = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    an, bn = np.linalg.norm(ar, axis=1), np.linalg.norm(br, axis=1)
    denominator = an * bn
    cosine = np.divide((ar * br).sum(axis=1), denominator, out=np.zeros_like(an), where=denominator != 0)
    cosine[(an == 0) & (bn == 0)] = 1
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    global_cos = float((a * b).sum() / (norm_a * norm_b)) if norm_a * norm_b else float(norm_a == norm_b)
    maximum = float(abs(a - b).max())
    return dict(max_abs=maximum, mean_abs=float(abs(a - b).mean()),
                rms=float(np.sqrt(np.mean((a - b) ** 2))), native_norm=norm_a, reference_norm=norm_b,
                native_max_abs=float(abs(a).max()), reference_max_abs=float(abs(b).max()),
                cos_min=float(cosine.min()), cos_global=global_cos,
                bit_exact=np.asarray(actual).tobytes() == np.asarray(expected).tobytes(),
                passed=maximum <= .002 and float(cosine.min()) >= .99999)


def environment(directory):
    import importlib.metadata
    import torch
    import transformers
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    expected = {'transformers': '5.16.1', 'numpy': '2.2.6', 'gguf': '0.19.0',
                'huggingface-hub': '1.33.0'}
    actual = {key: importlib.metadata.version(key) for key in expected}
    if actual != expected or torch.__version__.split('+')[0] != '2.14.0':
        raise ValueError('Pinned original operator environment differs')
    sources = validate_sources(directory)
    # Loading this class imports the inspected module; it does not instantiate a
    # model or read/download a checkpoint. Only RotaryEmbedding is constructed.
    cls = get_class_from_dynamic_module(profile()['config']['auto_map']['AutoModel'],
                                       str(directory), local_files_only=True)
    module = sys.modules[cls.__module__]
    if sha256(inspect.getfile(module)) != sources['modeling_ultrabert.py']:
        raise ValueError('Actual loaded original source differs')
    torch.set_num_threads(2)
    return module, dict(packages=actual, torch=torch.__version__, loaded_source_sha256=sources,
                        torch_config=torch.__config__.show(),
                        cpu_capability=torch.backends.cpu.get_cpu_capability(), threads=2,
                        source_scope='Inspected custom source/config only; no checkpoint or full model instantiated')


def build(directory, original, record, output):
    import gguf
    import numpy as np
    import torch
    module, env = environment(directory)
    if original.stat().st_size != record['archive_bytes'] or sha256(original) != record['archive_sha256']:
        raise ValueError('Immutable original archive differs')
    reader = gguf.GGUFReader(str(original))
    tensors = {t.name: t for t in reader.tensors}
    if len(tensors) != len(reader.tensors) or set(tensors) != set(record['stages']):
        raise ValueError('Original stage set differs')
    selected = {}

    def frozen(name):
        t = tensors[name]
        entry = record['stages'][name]
        if (t.tensor_type != gguf.GGMLQuantizationType.F32
                or list(reversed(t.shape.tolist())) != entry['shape']
                or hashlib.sha256(t.data.tobytes()).hexdigest() != entry['payload_sha256']):
            raise ValueError('Frozen selected payload/shape differs: ' + name)
        value = np.array(t.data, dtype=np.float32, copy=True).reshape(entry['shape'])
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite original payload')
        selected[name] = dict(shape=entry['shape'], sha256=entry['payload_sha256'])
        return torch.from_numpy(value)

    ids = frozen('input_ids')
    if ids.ndim != 2 or not 1 <= ids.shape[0] <= 2 or not 1 <= ids.shape[1] <= 512:
        raise ValueError('Unqualified original batch/context')
    batch, tokens = ids.shape
    prepared, observations = {}, {}
    theta = profile()['config']['rope_theta']
    if theta != [2560000 if i % 3 == 2 else 10000 for i in range(24)]:
        raise ValueError('Original theta schedule differs')
    for i in range(24):
        prefix = f'layer_{i}.'
        q, k = frozen(prefix + 'q'), frozen(prefix + 'k')
        cos, sin = frozen(prefix + 'rope_cos'), frozen(prefix + 'rope_sin')
        positions = frozen(prefix + 'position_ids')
        if (q.shape != (batch, tokens, 768) or k.shape != q.shape
                or cos.shape != (1, tokens, 128) or sin.shape != cos.shape
                or positions.shape != (1, tokens) or not torch.equal(positions, positions.floor())
                or torch.any(positions < 0) or torch.any(positions >= 512)):
            raise ValueError('Unexpected original RoPE shapes/positions')
        # Preserve every layer's two frozen table anchors, even when values repeat.
        rotary = module.RotaryEmbedding(base=theta[i], dim=128, device='cpu')
        with torch.inference_mode():
            current_cos, current_sin = rotary(q, positions.long())
            again_cos, again_sin = rotary(q, positions.long())
        if (not torch.equal(current_cos.view(torch.int32), again_cos.view(torch.int32))
                or not torch.equal(current_sin.view(torch.int32), again_sin.view(torch.int32))):
            raise ValueError('Repeated original RoPE table observation changes bits')
        basis = torch.zeros((batch, 1, tokens, 128), dtype=torch.float32)
        basis[..., :64] = 1
        qb = q.reshape(batch, tokens, 6, 128).transpose(1, 2).contiguous()
        kb = k.reshape(batch, tokens, 6, 128).transpose(1, 2).contiguous()
        inputs_before = [x.clone() for x in (qb, kb, cos, sin)]
        for scope, co, si in [('frozen', cos, sin), ('current', current_cos, current_sin)]:
            with torch.inference_mode():
                oq, ok = module.apply_rotary_pos_emb(qb, kb, co, si)
                rq, rk = module.apply_rotary_pos_emb(qb, kb, co, si)
                unit, _ = module.apply_rotary_pos_emb(basis, basis, co, si)
            if (not torch.equal(oq.view(torch.int32), rq.view(torch.int32))
                    or not torch.equal(ok.view(torch.int32), rk.view(torch.int32))):
                raise ValueError('Repeated actual original rotation changes bits')
            prepared[prefix + 'q_' + scope] = oq.transpose(1, 2).contiguous().reshape(batch, tokens, 768).numpy()
            prepared[prefix + 'k_' + scope] = ok.transpose(1, 2).contiguous().reshape(batch, tokens, 768).numpy()
            prepared[prefix + 'basis_' + scope] = unit[:, 0].contiguous().numpy()
        if any(not torch.equal(a.view(torch.int32), b.view(torch.int32))
               for a, b in zip(inputs_before, (qb, kb, cos, sin))):
            raise ValueError('Original function mutated frozen inputs')
        prepared.update({prefix + 'q_input': q.numpy(), prefix + 'k_input': k.numpy(),
                         prefix + 'cos_frozen': cos.expand(batch, -1, -1).contiguous().numpy(),
                         prefix + 'sin_frozen': sin.expand(batch, -1, -1).contiguous().numpy(),
                         prefix + 'positions': positions.expand(batch, -1).contiguous().numpy()})
        observations[prefix[:-1]] = dict(theta=theta[i],
            current_cos_vs_frozen=metrics(current_cos.numpy(), cos.numpy()),
            current_sin_vs_frozen=metrics(current_sin.numpy(), sin.numpy()),
            q_current_vs_frozen=metrics(prepared[prefix + 'q_current'], prepared[prefix + 'q_frozen']),
            k_current_vs_frozen=metrics(prepared[prefix + 'k_current'], prepared[prefix + 'k_frozen']),
            repeated_actual_function_bit_exact=True, frozen_inputs_unchanged=True)
    if len(selected) != 121 or len(prepared) != 264:
        raise ValueError('Incomplete 48-table/48-QK/24-position original input coverage')
    writer = gguf.GGUFWriter(str(output), 'ultrabert_rope_operator_reference')
    writer.add_string('ref.scope', 'same-input-local-rope-v1')
    writer.add_string('ref.model', MODEL)
    writer.add_string('ref.revision', REVISION)
    writer.add_string('ref.case', record['case']['id'])
    for name, value in prepared.items():
        writer.add_tensor(name, np.ascontiguousarray(value, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    report = dict(case=record['case']['id'], environment=env, batch=batch, tokens=tokens,
                  original_archive_sha256=record['archive_sha256'], selected_original_payloads=selected,
                  frozen_table_count=48, layer_observations=observations, reference_sha256=sha256(output),
                  arrays={name: dict(shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
                          for name, value in prepared.items()},
                  scope='Additional same-input RoPE operator targets only; old full-model archives remain immutable')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
