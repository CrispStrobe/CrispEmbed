#!/usr/bin/env python3
"""Hosted-only original Python capture; no CrispEmbed native model execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

from contract import CONTRACT, PANEL, ROOT, digest, read_contract, validate_checkpoint, verify_file


def download(directory, declaration):
    """Pinned resolve paths only; verify every byte before executing remote code."""
    directory = Path(directory)
    for name, identity in declaration['files'].items():
        path = directory/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = f"https://huggingface.co/{declaration['repository']}/resolve/{declaration['revision']}/{name}"
            with urllib.request.urlopen(url, timeout=180) as response, path.open('wb') as out:
                while block := response.read(1024*1024):
                    out.write(block)
        verify_file(path, identity)


def capture_original(variant, output):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    import sentence_transformers
    from sentence_transformers import MultiVectorEncoder

    contract = read_contract()
    declaration = contract['variants'][variant]['original']
    runtime = contract['reference_runtime']
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'sentence-transformers', 'numpy']}
    for package, expected in [('torch', runtime['torch']), ('transformers', runtime['transformers']),
                              ('sentence-transformers', runtime['sentence_transformers']['version']), ('numpy', runtime['numpy'])]:
        if versions[package].split('+')[0] != expected:
            raise ValueError(f'package version differs: {package}')
    st_root = Path(inspect.getfile(sentence_transformers)).parent.parent
    for name, identity in runtime['sentence_transformers']['files'].items():
        verify_file(st_root/name, identity)
    checkpoint = output/'checkpoint'
    download(checkpoint, declaration)
    identities, inventory = validate_checkpoint(checkpoint, variant)
    tokenizer_json = json.loads((checkpoint/'tokenizer.json').read_text())
    def summarize(value):
        if isinstance(value, str) and len(value) > 1024:
            return {'bytes_utf8': len(value.encode()), 'sha256_utf8': hashlib.sha256(value.encode()).hexdigest()}
        if isinstance(value, dict):
            return {k: summarize(v) for k, v in value.items() if k not in ('vocab', 'merges')}
        if isinstance(value, list):
            return [summarize(v) for v in value]
        return value
    tokenizer_descriptor = {key: summarize(tokenizer_json.get(key)) for key in
                            ['model', 'normalizer', 'pre_tokenizer', 'post_processor', 'decoder', 'added_tokens', 'padding', 'truncation']}
    early_review = output/'review'
    early_review.mkdir(exist_ok=True)
    (early_review/'preparation.json').write_text(json.dumps({
        'variant': variant, 'revision': declaration['revision'], 'checkpoint_files': identities,
        'inventory': inventory, 'tokenizer_descriptor': tokenizer_descriptor,
        'source_contract_sha256': digest(CONTRACT), 'original_inference_completed': False
    }, indent=2)+'\n')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    model = MultiVectorEncoder(str(checkpoint), device='cpu', trust_remote_code=True,
                               local_files_only=True, model_kwargs={'dtype': torch.float32})
    model.eval()
    backbone = model[0].model
    verify_file(inspect.getfile(type(backbone)), declaration['files']['modeling_modernbert_hash_moe_v10.py'])
    if any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('original CPU F32 required')
    from safetensors import safe_open
    loaded_inventory = {}
    for file, module in [('model.safetensors', backbone), ('1_Dense/model.safetensors', model[1])]:
        state = module.state_dict()
        with safe_open(str(checkpoint/file), framework='pt', device='cpu') as weights:
            if set(state) != set(weights.keys()):
                raise ValueError('loaded original parameter set differs')
            loaded_inventory[file] = {}
            for name, value in state.items():
                if not torch.equal(value.detach(), weights.get_tensor(name)):
                    raise ValueError(f'loaded original parameter differs: {name}')
                loaded_inventory[file][name] = hashlib.sha256(value.detach().contiguous().numpy().astype('<f4', copy=False).tobytes()).hexdigest()
    # Check the loader's actual legacy PyLate migration rather than silently
    # substituting a hand-written preprocessing pipeline.
    module_types = [type(m).__name__ for m in model]
    if module_types != ['Transformer', 'Dense', 'MultiVectorMask', 'Normalize']:
        raise ValueError(f'unexpected legacy pipeline: {module_types}')
    if model[0].query_expansion != {'strategy': 'fixed', 'attend': False, 'length': 32}:
        # The runtime may normalize a missing optional key to token=None.
        expansion = {k: v for k, v in model[0].query_expansion.items() if v is not None}
        if expansion != {'strategy': 'fixed', 'attend': False, 'length': 32}:
            raise ValueError(f'query expansion differs: {expansion}')
    panel = json.loads(PANEL.read_text())
    report = {'schema': 'crispembed.moedern-colbert.original.v1', 'completed': False,
              'variant': variant, 'revision': declaration['revision'],
              'source_contract_sha256': digest(CONTRACT), 'panel_sha256': digest(PANEL),
              'versions': versions, 'pipeline': module_types, 'tokenizer_descriptor': tokenizer_descriptor, 'checkpoint_files': identities,
              'inventory': inventory, 'loaded_tensor_sha256': loaded_inventory, 'cases': [], 'scores': [], 'stage_arrays': [],
              'scope': 'Original CPU F32 owned controls; author GGUF/native/CoreML unexecuted; representative quality unqualified'}
    report['driver_sources'] = {str(p.relative_to(ROOT)): digest(p) for p in
                                [Path(__file__), Path(__file__).with_name('contract.py'), Path(__file__).with_name('convert_f32.py')]}
    report['github_sha'] = os.environ.get('GITHUB_SHA')
    report['python'] = sys.version
    selected_by_id = {}
    review = output/'review'
    review.mkdir(exist_ok=True)
    def checkpoint_report():
        (review/'original.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    checkpoint_report()

    def save(case_id, stage, value):
        if isinstance(value, torch.Tensor):
            # Snapshot before later mutation of a feature dictionary or tensor.
            value = value.detach().clone().contiguous().cpu()
            if value.is_floating_point():
                if value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError(f'nonfinite/non-F32 original stage: {stage}')
                array = value.numpy().astype('<f4', copy=False)
                extension = 'f32'
            elif value.dtype == torch.bool:
                array, extension = value.numpy().astype('u1'), 'u8'
            else:
                array, extension = value.numpy().astype('<i8'), 'i64'
            relative = f"arrays/{case_id}/{hashlib.sha256(stage.encode()).hexdigest()[:24]}.{extension}"
            path = output/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise ValueError(f'duplicate stage capture: {stage}')
            path.write_bytes(array.tobytes())
            report['stage_arrays'].append({'case': case_id, 'stage': stage, 'path': relative,
                                          'shape': list(value.shape), 'dtype': extension,
                                          'bytes': path.stat().st_size, 'sha256': digest(path)})
        elif isinstance(value, dict):
            for key, item in value.items():
                save(case_id, f'{stage}.{key}', item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                save(case_id, f'{stage}.{index}', item)

    def equal_features(a, b):
        if set(a) != set(b):
            return False
        for key in a:
            if isinstance(a[key], torch.Tensor):
                if not isinstance(b[key], torch.Tensor) or not torch.equal(a[key], b[key]):
                    return False
            elif a[key] != b[key]:
                return False
        return True

    for case in panel['cases']:
        cid, role, text = case['id'], case['role'], case['text']
        encode = model.encode_query if role == 'query' else model.encode_document
        t0 = time.monotonic()
        ordinary = encode(text, output_value=None, show_progress_bar=False)
        # Freeze ordinary feature dictionaries before the next call mutates anything.
        ordinary = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in ordinary.items()}
        public = encode(text, show_progress_bar=False)
        selected = ordinary['token_embeddings'][ordinary['attention_mask'].bool()]
        if not torch.equal(public, selected):
            raise ValueError(f'public role output differs from original scoring-mask slice: {cid}')
        handles, counts, actual_masks = [], {}, {}
        def backbone_input(_m, _args, kwargs):
            for key in ['input_ids', 'attention_mask']:
                actual_masks['supplied_'+key] = kwargs[key].detach().clone().cpu().tolist()
        def first_layer_input(_m, _args, kwargs):
            actual_masks['effective_attention_mask'] = kwargs['attention_mask'].detach().clone().cpu().tolist()
        handles.append(backbone.register_forward_pre_hook(backbone_input, with_kwargs=True))
        handles.append(backbone.layers[0].register_forward_pre_hook(first_layer_input, with_kwargs=True))

        def event(name, value):
            index = counts.get(name, 0)
            counts[name] = index+1
            save(cid, f'{name}#{index}', value)

        if cid in panel['full_capture']:
            for name, module in backbone.named_modules():
                def before(_m, args, kwargs, name=name):
                    event('backbone.'+name+'.input', {'args': args, 'kwargs': kwargs})
                def after(_m, _args, value, name=name):
                    event('backbone.'+name+'.output', value)
                    if name.endswith('.ffn.router'):
                        # Explicitly derived from actual original router logits.
                        # These are not claimed as observed internal softmax/topk arrays.
                        probabilities = value.detach().clone().softmax(dim=-1)
                        weights, indices = probabilities.topk(backbone.config.top_k, dim=-1)
                        event('derived.'+name+'.probabilities', probabilities)
                        event('derived.'+name+'.topk_indices', indices)
                        event('derived.'+name+'.renormalized_weights', weights/(weights.sum(dim=-1, keepdim=True)+1e-9))
                        ordered = probabilities.sort(dim=-1, descending=True).values
                        event('derived.'+name+'.selection_margin', ordered[..., backbone.config.top_k-1]-ordered[..., backbone.config.top_k])
                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(module.register_forward_hook(after))
            for index, module in enumerate(model):
                if index:
                    handles.append(module.register_forward_hook(
                        lambda _m, _args, value, index=index: event(f'pipeline.{index}.output', value)))
        original_sdpa = functional.scaled_dot_product_attention
        sdpa_count = 0

        def observed_sdpa(*args, **kwargs):
            nonlocal sdpa_count
            name = f'actual.sdpa.{sdpa_count}'
            sdpa_count += 1
            save(cid, name+'.input', {'args': args, 'kwargs': kwargs})
            result = original_sdpa(*args, **kwargs)
            save(cid, name+'.output', result)
            return result

        try:
            if cid in panel['full_capture']:
                functional.scaled_dot_product_attention = observed_sdpa
            observed = encode(text, output_value=None, show_progress_bar=False)
        finally:
            functional.scaled_dot_product_attention = original_sdpa
            for handle in handles:
                handle.remove()
        if not equal_features(ordinary, observed):
            raise ValueError(f'ordinary/observer output differs: {cid}')
        ids = ordinary['input_ids']
        if ids.ndim != 1 or not 1 <= ids.numel() <= (32 if role == 'query' else 512):
            raise ValueError('bounded token shape differs')
        if ids[0].item() != 2 or ids[1].item() != (30000 if role == 'query' else 30001):
            raise ValueError('prefix-marker tokenization differs')
        if role == 'query' and (ids.numel() != 32 or not ordinary['attention_mask'].bool().all()):
            raise ValueError('query expansion scoring mask differs')
        supplied = actual_masks['supplied_attention_mask'][0]
        effective = actual_masks['effective_attention_mask'][0]
        if actual_masks['supplied_input_ids'][0] != ids.tolist():
            raise ValueError('observed supplied IDs differ')
        if effective != ([1]*len(supplied) if role == 'query' else supplied):
            raise ValueError('original effective backbone mask differs')
        save(cid, 'ordinary.features', ordinary)
        save(cid, 'ordinary.selected', selected)
        selected_by_id[cid] = selected
        report['cases'].append({**case, 'token_ids': ids.tolist(),
                                'scoring_mask': ordinary['attention_mask'].tolist(),
                                'supplied_attention_mask': supplied, 'effective_attention_mask': effective,
                                'shape': list(selected.shape), 'ordinary_observer_bit_exact': True,
                                'public_slice_bit_exact': True, 'full_capture': cid in panel['full_capture'],
                                'seconds_including_observer': time.monotonic()-t0})
        checkpoint_report()
    for query, document in panel['score_pairs']:
        q, d = selected_by_id[query], selected_by_id[document]
        if q.shape[0] == 0 or d.shape[0] == 0:
            raise ValueError('MaxSim requires at least one scoring token')
        original = model.similarity([q], [d]).item()
        independent = (q @ d.T).max(dim=1).values.sum().item()
        if abs(original-independent) > contract['gates']['maxsim_max_absolute']:
            raise ValueError('original MaxSim differs from explicit query-row reduction')
        report['scores'].append({'query': query, 'document': document,
                                 'original_maxsim': original, 'explicit_maxsim': independent})
    import resource
    report['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['completed'] = True
    checkpoint_report()
    return review/'original.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['1.5M', '27M', '115M'], default='1.5M')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--convert-after-original', action='store_true')
    args = p.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        p.error('model access/inference is hosted-only')
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        receipt = capture_original(args.variant, args.output)
        if args.convert_after_original:
            from convert_f32 import convert
            convert(args.output/'checkpoint', args.variant, args.output/'model-F32.gguf', receipt, digest(receipt))
    except Exception as error:
        review = args.output/'review'
        review.mkdir(exist_ok=True)
        (review/'rejection.json').write_text(json.dumps({'completed': False, 'error': str(error)}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
