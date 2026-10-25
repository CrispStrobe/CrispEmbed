#!/usr/bin/env python3
"""Test-only same-weight first-FFN targets; immutable full-model arrays stay intact."""
import hashlib
import inspect
import json
from pathlib import Path

SOURCE_SHA = '788d4bad50a8d39be2fe79125f0f40134773cd23b1791606fb6b3ab0bc6d2263'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def metrics(actual, reference):
    import numpy as np
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid comparison shape/values')
    ar, br = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    an, bn = np.linalg.norm(ar, axis=1), np.linalg.norm(br, axis=1)
    dots = (ar*br).sum(axis=1)
    cosine = np.divide(dots, an*bn, out=np.zeros_like(dots), where=an*bn != 0)
    cosine[(an == 0) & (bn == 0)] = 1
    maximum = float(abs(a-b).max())
    return {'max_abs': maximum, 'mean_abs': float(abs(a-b).mean()),
            'rms_error': float(np.sqrt(np.mean((a-b)**2))),
            'actual_norm': float(np.linalg.norm(a)), 'reference_norm': float(np.linalg.norm(b)),
            'actual_max_abs': float(abs(a).max()), 'reference_max_abs': float(abs(b).max()),
            'cos_min': float(cosine.min()),
            'cos_global': float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum())),
            'bit_exact': bool(np.array_equal(np.asarray(actual).view('u4'), np.asarray(reference).view('u4'))),
            'passed': maximum <= .002 and float(cosine.min()) >= .99999}


def observe(mlp, x):
    import torch
    captured, handles = {}, []
    def save(name, value):
        if name in captured or not torch.isfinite(value).all():
            raise ValueError('Duplicate/nonfinite original MLP observation')
        captured[name] = value.detach().clone().contiguous()
    for name, module in [('gate', mlp.gate_proj), ('up', mlp.up_proj), ('silu', mlp.act_fn)]:
        handles.append(module.register_forward_hook(lambda module, args, value, name=name: save(name, value)))
    handles.append(mlp.down_proj.register_forward_pre_hook(lambda module, args: save('product', args[0])))
    try:
        with torch.inference_mode():
            result = mlp(x)
    finally:
        for h in handles: h.remove()
    with torch.inference_mode():
        ordinary = mlp(x)
    if not torch.equal(result.view(torch.int32), ordinary.view(torch.int32)):
        raise ValueError('Original module observer changed output bits')
    save('down', result)
    return {'input': x, **captured}


def require_environment():
    import importlib.metadata
    import torch
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as blueprint
    expected = {'transformers':'5.16.1','tokenizers':'0.23.3','numpy':'2.2.6',
                'gguf':'0.19.0','safetensors':'0.8.0','huggingface-hub':'1.33.0','pillow':'12.1.0'}
    actual = {name:importlib.metadata.version(name) for name in expected}
    if actual != expected or torch.__version__.split('+')[0] != '2.11.0':
        raise ValueError('Original framework environment differs')
    source = Path(inspect.getfile(blueprint))
    if sha(source) != SOURCE_SHA: raise ValueError('Actual original Python module differs')
    torch.set_num_threads(2)
    return {'packages':actual,'torch':torch.__version__,'blueprint_sha256':sha(source)}


def build(original_file, candidate_file, model_dir, native_input, native_layer, output, baseline_records):
    import gguf
    from gguf.quants import dequantize
    import numpy as np
    import torch
    import transformers
    from safetensors import safe_open
    from transformers.models.qwen3_5 import modeling_qwen3_5 as blueprint
    from transformers import Qwen3_5TextConfig
    require_environment()
    source = Path(inspect.getfile(blueprint))
    frozen = {t.name: t for t in gguf.GGUFReader(str(original_file)).tensors}
    def original(name):
        t = frozen[name]
        if t.tensor_type != gguf.GGMLQuantizationType.F32: raise ValueError('Frozen target is not F32')
        return torch.from_numpy(np.array(t.data, dtype=np.float32, copy=True))
    embeddings = original('prefill.decoder.inputs_embeds')
    attention = original('prefill.linear.output')
    anchor = original('prefill.decoder.layer.0')
    residual = embeddings + attention
    config = Qwen3_5TextConfig(hidden_size=1024, intermediate_size=3584, hidden_act='silu', rms_norm_eps=1e-6)
    norm = blueprint.Qwen3_5RMSNorm(1024, eps=1e-6)
    full = blueprint.Qwen3_5MLP(config, 3584).eval()
    prefix = 'model.language_model.layers.0.'
    with safe_open(str(Path(model_dir)/'model.safetensors'), framework='pt', device='cpu') as f:
        norm.load_state_dict({'weight': f.get_tensor(prefix+'post_attention_layernorm.weight').float()})
        full.load_state_dict({key: f.get_tensor(prefix+'mlp.'+key).float() for key in full.state_dict()})
    norm.eval()
    with torch.inference_mode(): x = norm(residual)
    orig = observe(full, x)
    baseline_state = {}
    baseline_payloads = {}
    for name, weight in full.state_dict().items():
        half = weight.half().contiguous()
        digest = hashlib.sha256(half.numpy().tobytes()).hexdigest()
        expected = baseline_records['l.layers.0.mlp.'+name]
        if expected['type'] != 'F16' or digest != expected['sha256']:
            raise ValueError('Original source conversion does not reproduce frozen F16 matrix: '+name)
        baseline_state[name] = half.float()
        baseline_payloads[name] = digest
    full.load_state_dict(baseline_state)
    baseline = observe(full, x)
    original_layer = (residual + orig['down']).numpy()
    anchor_check = metrics(original_layer, anchor.numpy())
    candidate = {t.name:t for t in gguf.GGUFReader(str(candidate_file)).tensors}
    weights, records = {}, {}
    for name in ('gate','up','down'):
        t = candidate['l.layers.0.mlp.'+name+'_proj.weight']
        if t.tensor_type != gguf.GGMLQuantizationType.Q4_K: raise ValueError('Candidate FFN type differs')
        a = np.ascontiguousarray(dequantize(t.data, t.tensor_type), dtype=np.float32)
        expected = (1024,3584) if name == 'down' else (3584,1024)
        if a.shape != expected: raise ValueError('Independent dequant shape differs: '+str(a.shape))
        weights[name] = a
        records[name] = {'shape':list(a.shape),'packed_sha256':hashlib.sha256(t.data.tobytes()).hexdigest(),
                         'dequantized_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
                         'blocks':int(a.size/256), 'type':'Q4_K'}
    full.load_state_dict({name+'_proj.weight': torch.from_numpy(a) for name,a in weights.items()})
    quant = observe(full, x)
    quant_layer = (residual + quant['down']).numpy()
    prepared = {'quant.'+name:value.numpy() for name,value in quant.items()}
    prepared.update({'weight.'+name:a for name,a in weights.items()})
    native = np.fromfile(native_input,dtype='<f4').reshape(embeddings.shape)
    old_layer = np.fromfile(native_layer,dtype='<f4').reshape(anchor.shape)
    positions=original('prefill.decoder.position_ids').numpy()
    if positions.shape[0] == 4: positions=positions[-3:]
    if positions.size != 3 * embeddings.shape[1]: raise ValueError('Position source shape differs')
    prepared.update({'native.embeddings':native,'native.layer0':old_layer,
                     'original.positions':positions,'original.embeddings':embeddings.numpy(),
                     'original.layer0':anchor.numpy(),'quant.layer0':quant_layer})
    writer = gguf.GGUFWriter(str(output), 'ovisocr2_ffn_test_reference')
    for name,a in prepared.items(): writer.add_tensor(name,np.ascontiguousarray(a,dtype=np.float32))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    stages={name:metrics(quant[name].numpy(),orig[name].numpy()) for name in quant}
    report={'scope':'additional same-original-input first MLP only; not a replacement full-model reference',
            'blueprint_sha256':sha(source),'dequantizer_sha256':sha(inspect.getfile(dequantize)),
            'observer_neutrality_bit_exact':True,'original_layer0_anchor':anchor_check,
            'intended_precision_loss':stages,'quant_layer_vs_original':metrics(quant_layer,anchor.numpy()),
            'retained_native_layer_vs_original':metrics(old_layer,anchor.numpy()),
            'weights':records,'frozen_f16_payloads':baseline_payloads,
            'f16_vs_original':{name:metrics(baseline[name].numpy(),orig[name].numpy()) for name in orig},
            'q4_vs_f16':{name:metrics(quant[name].numpy(),baseline[name].numpy()) for name in quant},
            'reference_sha256':sha(output),
            'frozen_original_sha256':sha(original_file),
            'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__},
            'arrays':{name:{'shape':list(a.shape),'sha256':hashlib.sha256(a.tobytes()).hexdigest()} for name,a in prepared.items()}}
    Path(output).with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Preserve the generated diagnostic even when the original anchor drifts.
    if not anchor_check['passed']: raise ValueError('Original isolated MLP exceeds unchanged frozen anchor budget')
    return report
