#!/usr/bin/env python3
"""Runs only in the Kaggle isolated pinned environment after public CI gates."""

import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys


OUT = Path(os.environ['OVIS_OUT'])
SCRATCH = Path(os.environ['OVIS_SCRATCH'])
MODEL = 'ATH-MaaS/OvisOCR2'
REVISION = 'cbdede82e048659b83310476ff807da825f8fe78'
CASES = (('square', 448, 448), ('odd', 449, 417), ('small-control', 96, 64))
REPORT = {'phase': 'gpu-control', 'status': 'started', 'cases': [],
          'scope': 'Pinned original processor anchors, then one bounded original vLLM greedy image pass; T4 auto-FP16 only'}


def save():
    (OUT / 'ovisocr2-vllm-worker.json').write_text(json.dumps(REPORT, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def literal_text(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal_text(node.left) + literal_text(node.right)
    raise ValueError('Original README prompt is not a literal expression')


def original_prompt(readme):
    found = []
    for block in re.findall(r'```python\s*\n(.*?)```', readme, re.S):
        for node in ast.walk(ast.parse(block)):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'prompt' for t in node.targets)):
                found.append(literal_text(node.value))
    if len(found) != 1:
        raise ValueError('Original README example prompt ambiguous')
    return found[0]


def clean_truncated_repeats(text, min_text_len=8000, max_period=200,
                            min_period=1, min_repeat_chars=100, min_repeat_times=5):
    n = len(text)
    if n < min_text_len:
        return text
    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue
        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1
        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len
        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len:]
    return text


def main():
    save()
    import numpy as np
    import torch
    import transformers
    import tokenizers
    from PIL import Image
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoProcessor

    versions = {name: importlib.metadata.version(name) for name in (
        'torch', 'torchvision', 'transformers', 'tokenizers', 'numpy', 'Pillow', 'huggingface-hub', 'vllm')}
    REPORT['versions'] = versions
    REPORT['torch_cuda'] = torch.version.cuda
    if (versions['torch'].split('+')[0] != '2.11.0' or versions['transformers'] != '5.16.1'
            or versions['tokenizers'] != '0.23.3'
            or versions['vllm'] not in ('0.22.1', '0.22.1+cu129')
            or not str(torch.version.cuda).startswith('12.9') or not torch.cuda.is_available()):
        raise ValueError('Pinned package or CUDA runtime differs')
    if torch.cuda.get_device_capability() != (7, 5) or 'T4' not in torch.cuda.get_device_name():
        raise ValueError('Torch selected another GPU; no fallback')
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda', dtype=torch.float16)
    right = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device='cuda', dtype=torch.float16)
    actual = (left @ right).cpu().tolist()
    if actual != [[2.0, 4.0], [6.0, 8.0]]:
        raise ValueError('Pinned T4 FP16 CUDA matmul control failed')
    REPORT['gpu_control'] = {'device': torch.cuda.get_device_name(), 'capability': [7, 5],
                             'dtype': 'float16', 'result': actual, 'passed': True}
    REPORT['phase'] = 'original-metadata'; save()

    evidence_url = ('https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/'
                    + os.environ['OVIS_PROOF_COMMIT']
                    + '/tests/results/qwen35-ocr/ovisocr2-metadata.json')
    import urllib.request
    with urllib.request.urlopen(evidence_url, timeout=30) as response:
        evidence = json.load(response)
    producer = evidence['producer']
    source = Path(snapshot_download(MODEL, revision=REVISION,
        allow_patterns=sorted(producer['checkpoint_files_sha256'])))
    for name, expected in producer['checkpoint_files_sha256'].items():
        if sha(source / name) != expected:
            raise ValueError('Pinned metadata file differs: ' + name)
    REPORT['metadata_files_sha256'] = producer['checkpoint_files_sha256']
    processor = AutoProcessor.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    prompt = original_prompt((source / 'README.md').read_text())
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    rendered = processor.tokenizer.apply_chat_template(messages, tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=False)
    rendered_sha = hashlib.sha256(rendered.encode()).hexdigest()
    if rendered_sha != producer['rendered_prompt_sha256'] or rendered != evidence['rendered_prompt']:
        raise ValueError('Original rendered prompt differs')
    REPORT['rendered_prompt_sha256'] = rendered_sha
    rows = {row['case']: row for row in producer['inputs']}
    if set(rows) != {label for label, _, _ in CASES}:
        raise ValueError('Original processor case set differs')
    first_image = None
    for label, width, height in CASES:
        REPORT['phase'] = 'original-processor-' + label; save()
        y, x = np.indices((height, width))
        rgb = np.stack(((x * 3 + y) % 256, (y * 5 + x) % 256,
                        (x + y * 7) % 256), axis=-1).astype(np.uint8)
        image = Image.fromarray(rgb)
        row = rows[label]
        if list(rgb.shape) != row['rgb_shape'] or hashlib.sha256(rgb.tobytes()).hexdigest() != row['rgb_sha256']:
            raise ValueError('Original synthetic RGB recipe differs: ' + label)
        inputs = processor(text=[rendered], images=[image], return_tensors='pt',
                           images_kwargs={'min_pixels': 448 * 448, 'max_pixels': 2880 * 2880})
        arrays = {key: value.detach().cpu().numpy() for key, value in inputs.items()}
        if set(arrays) != set(row['fields']):
            raise ValueError('Original processor field names differ: ' + label)
        for key, value in arrays.items():
            field = row['fields'][key]
            if (list(value.shape) != field['shape'] or str(value.dtype) != field['dtype']
                    or hashlib.sha256(value.tobytes()).hexdigest() != field['sha256']):
                raise ValueError('Original processor array differs: ' + label + ':' + key)
        if (arrays['input_ids'].tolist() != row['ids'] or arrays['image_grid_thw'].tolist() != row['grid']
                or arrays['attention_mask'].tolist() != row['attention_mask']):
            raise ValueError('Original processor IDs, grid or mask differ: ' + label)
        file = SCRATCH / 'original' / (label + '-processor.npz')
        with np.load(file, allow_pickle=False) as frozen:
            if set(frozen.files) != set(arrays) or any(not np.array_equal(frozen[k], arrays[k]) for k in arrays):
                raise ValueError('Original artifact arrays differ: ' + label)
        np.savez_compressed(OUT / (label + '-processor-replay.npz'), **arrays)
        REPORT['cases'].append({'case': label, 'rgb_sha256': row['rgb_sha256'],
                                'fields_sha256': {k: row['fields'][k]['sha256'] for k in arrays},
                                'ids_count': int(arrays['input_ids'].shape[1]), 'grid': row['grid'],
                                'exact_original_arrays': True})
        save()
        if label == 'square':
            first_image = image

    REPORT['phase'] = 'checkpoint'; save()
    checkpoint = Path(hf_hub_download(MODEL, 'model.safetensors', revision=REVISION))
    if checkpoint.stat().st_size != 1706030496 or sha(checkpoint) != os.environ['OVIS_CHECKPOINT_SHA']:
        raise ValueError('Pinned original checkpoint payload differs')
    REPORT['checkpoint_sha256'] = sha(checkpoint)
    REPORT['phase'] = 'original-vllm-init'; save()
    from vllm import LLM, SamplingParams
    model = LLM(model=str(source), tensor_parallel_size=1, gpu_memory_utilization=0.8,
                gdn_prefill_backend='triton')
    dtype = str(model.model_config.dtype)
    REPORT['resolved_vllm_dtype'] = dtype
    if dtype != 'torch.float16':
        raise ValueError('T4 README auto dtype did not resolve to FP16')
    vllm_tokenizer = model.get_tokenizer()
    if vllm_tokenizer.eos_token_id != 248046:
        raise ValueError('Original vLLM tokenizer EOS differs')
    REPORT['source_stop_contract'] = {'tokenizer_eos': 248046, 'generation_eos': 248044,
                                      'derived_stop_ids': [248044, 248046]}
    generated_prompt = vllm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if generated_prompt != rendered:
        raise ValueError('vLLM tokenizer rendered prompt differs from original processor')
    REPORT['phase'] = 'bounded-greedy'; save()
    cap = 256  # Explicit bounded departure from README's 16384; greedy settings unchanged.
    sampling = SamplingParams(max_tokens=cap, temperature=0.0)
    item = {'prompt': generated_prompt, 'multi_modal_data': {'image': first_image},
            'mm_processor_kwargs': {'images_kwargs': {'min_pixels': 448 * 448,
                                                      'max_pixels': 2880 * 2880}}}
    outputs = model.generate([item], sampling)
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise ValueError('Original vLLM output count differs')
    result = outputs[0].outputs[0]
    raw_text = result.text
    stripped = raw_text.strip()
    filtered = '\n\n'.join(block for block in stripped.split('\n\n')
                           if not block.strip().startswith('<img src="images/bbox_'))
    processed = clean_truncated_repeats(filtered)
    REPORT['generation'] = {'case': 'square', 'readme_max_tokens': 16384,
                            'bounded_max_tokens': cap, 'temperature': 0.0,
                            'dtype_policy': 'README auto -> T4 FP16',
                            'gdn_prefill_backend': 'triton',
                            'prompt_token_ids': (list(outputs[0].prompt_token_ids)
                                                 if outputs[0].prompt_token_ids is not None else None),
                            'token_ids': list(result.token_ids),
                            'finish_reason': result.finish_reason,
                            'stop_reason': result.stop_reason,
                            'raw_text': raw_text, 'stripped_text': stripped,
                            'processed_text': processed,
                            'raw_text_sha256': hashlib.sha256(raw_text.encode()).hexdigest(),
                            'processed_text_sha256': hashlib.sha256(processed.encode()).hexdigest(),
                            'limits': 'One synthetic image; 256-token cap; no decoded-quality or BF16 parity claim'}
    REPORT['status'] = 'completed bounded original vLLM reference'
    REPORT['phase'] = 'complete'; save()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        REPORT['status'] = 'failed original vLLM worker'
        REPORT['error_type'] = type(error).__name__
        save()
        raise
