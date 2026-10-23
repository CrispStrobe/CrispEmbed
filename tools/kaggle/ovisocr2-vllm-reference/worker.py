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


def original_cleanup(readme):
    methods = []
    for block in re.findall(r'```python\s*\n(.*?)```', readme, re.S):
        tree = ast.parse(block)
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == 'OvisOCR2Parser'):
            methods.extend(node for node in cls.body if isinstance(node, ast.FunctionDef)
                           and node.name == '_clean_truncated_repeats')
    if len(methods) != 1:
        raise ValueError('Original README repeat-cleaner ambiguous')
    namespace = {}
    code = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    exec(compile(code, 'pinned-original-README-cleaner', 'exec'), namespace)
    return lambda text: namespace['_clean_truncated_repeats'](None, text)


def main():
    save()
    import numpy as np
    import torch
    import transformers
    import tokenizers
    from PIL import Image, ImageDraw
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
    readme = (source / 'README.md').read_text()
    prompt = original_prompt(readme)
    clean_repeats = original_cleanup(readme)
    REPORT['original_readme_sha256'] = sha(source / 'README.md')
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
        replay = OUT / (label + '-processor-replay.npz')
        np.savez_compressed(replay, **arrays)
        actual_fields = {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                               'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
                         for key, value in arrays.items()}
        case_record = {'case': label, 'rgb_sha256': row['rgb_sha256'],
                       'replay_file': replay.name, 'replay_sha256': sha(replay),
                       'actual_fields': actual_fields, 'expected_fields': row['fields'],
                       'actual_ids': arrays['input_ids'].tolist() if 'input_ids' in arrays else None,
                       'actual_grid': arrays['image_grid_thw'].tolist() if 'image_grid_thw' in arrays else None,
                       'exact_original_arrays': False}
        REPORT['cases'].append(case_record); save()
        if set(arrays) != set(row['fields']):
            case_record['first_mismatch'] = {'kind': 'field_names',
                                             'actual': sorted(arrays), 'expected': sorted(row['fields'])}
            save()
            raise ValueError('Original processor field names differ: ' + label)
        for key, value in arrays.items():
            field = row['fields'][key]
            if actual_fields[key] != field:
                case_record['first_mismatch'] = {'kind': 'field', 'field': key,
                                                 'actual': actual_fields[key], 'expected': field}
                save()
                raise ValueError('Original processor array differs: ' + label + ':' + key)
        if (arrays['input_ids'].tolist() != row['ids'] or arrays['image_grid_thw'].tolist() != row['grid']
                or arrays['attention_mask'].tolist() != row['attention_mask']):
            case_record['first_mismatch'] = {'kind': 'ids_grid_or_mask',
                'actual': {key: arrays[key].tolist() for key in ('input_ids', 'image_grid_thw', 'attention_mask')},
                'expected': {key: row[key] for key in ('ids', 'grid', 'attention_mask')}}
            save()
            raise ValueError('Original processor IDs, grid or mask differ: ' + label)
        file = SCRATCH / 'original' / (label + '-processor.npz')
        with np.load(file, allow_pickle=False) as frozen:
            if set(frozen.files) != set(arrays) or any(not np.array_equal(frozen[k], arrays[k]) for k in arrays):
                case_record['first_mismatch'] = {'kind': 'artifact_array_equality',
                                                 'actual_fields': actual_fields,
                                                 'expected_fields': row['fields']}
                save()
                raise ValueError('Original artifact arrays differ: ' + label)
        case_record['exact_original_arrays'] = True
        save()
        if label == 'square':
            first_image = image

    REPORT['phase'] = 'owned-readable-fixture'; save()
    readable = Image.new('RGB', (448, 448), 'white')
    ImageDraw.Draw(readable).text((30, 80), 'Hello OCR\n12345', fill='black', font_size=24)
    readable_rgb_sha = hashlib.sha256(np.asarray(readable).tobytes()).hexdigest()
    readable_path = OUT / 'owned-readable.png'
    readable.save(readable_path)
    readable_inputs = processor(text=[rendered], images=[readable], return_tensors='pt',
                                images_kwargs={'min_pixels': 448 * 448, 'max_pixels': 2880 * 2880})
    readable_ids = readable_inputs['input_ids'].detach().cpu().numpy().tolist()[0]
    readable_grid = readable_inputs['image_grid_thw'].detach().cpu().numpy().tolist()
    REPORT['owned_readable'] = {'case': 'readable', 'gold_smoke_only': 'Hello OCR\n12345',
                                'rgb_sha256': readable_rgb_sha, 'png_sha256': sha(readable_path),
                                'width': 448, 'height': 448,
                                'hf_prompt_ids': readable_ids, 'hf_grid': readable_grid,
                                'recipe': "Image.new('RGB',(448,448),'white'); ImageDraw.Draw(image).text((30,80),'Hello OCR\\n12345',fill='black',font_size=24)"}
    save()
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
    runtime_generation = model.model_config.try_get_generation_config()
    probe = SamplingParams(max_tokens=256, temperature=0.0)
    probe.update_from_generation_config(runtime_generation, vllm_tokenizer.eos_token_id)
    REPORT['runtime_stop_probe'] = {'source': 'installed vLLM 0.22.1 SamplingParams.update_from_generation_config',
                                    'generation_config': runtime_generation,
                                    'primary_eos': probe._eos_token_id,
                                    'additional_stop_ids': sorted(probe.stop_token_ids),
                                    'all_stop_ids': sorted(probe._all_stop_token_ids)}
    if (probe._eos_token_id != 248046 or sorted(probe._all_stop_token_ids) != [248044, 248046]):
        raise ValueError('Original installed vLLM stop resolution differs')
    generated_prompt = vllm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if generated_prompt != rendered:
        raise ValueError('vLLM tokenizer rendered prompt differs from original processor')
    REPORT['phase'] = 'bounded-greedy'; save()
    cap = 256  # Explicit bounded departure from README's 16384; greedy settings unchanged.
    sampling = SamplingParams(max_tokens=cap, temperature=0.0)
    items = [{'prompt': generated_prompt, 'multi_modal_data': {'image': image},
              'mm_processor_kwargs': {'images_kwargs': {'min_pixels': 448 * 448,
                                                        'max_pixels': 2880 * 2880}}}
             for image in (first_image, readable)]
    outputs = model.generate(items, sampling)
    if len(outputs) != 2 or any(len(output.outputs) != 1 for output in outputs):
        raise ValueError('Original vLLM output count differs')
    REPORT['generations'] = []
    for label, output in zip(('square', 'readable'), outputs):
        expected_ids = rows['square']['ids'][0] if label == 'square' else readable_ids
        actual_ids = list(output.prompt_token_ids) if output.prompt_token_ids is not None else None
        result = output.outputs[0]
        raw_text = result.text
        stripped = raw_text.strip()
        filtered = '\n\n'.join(block for block in stripped.split('\n\n')
                               if not block.strip().startswith('<img src="images/bbox_'))
        processed = clean_repeats(filtered)
        generation = {'case': label, 'readme_max_tokens': 16384,
            'bounded_max_tokens': cap, 'temperature': 0.0, 'dtype_policy': 'README auto -> T4 FP16',
            'gdn_prefill_backend': 'triton',
            'prompt_token_ids': actual_ids, 'expected_hf_prompt_token_ids': expected_ids,
            'prompt_ids_exact': actual_ids == expected_ids,
            'token_ids': list(result.token_ids), 'finish_reason': result.finish_reason,
            'stop_reason': result.stop_reason, 'raw_text': raw_text, 'stripped_text': stripped,
            'processed_text': processed, 'raw_text_sha256': hashlib.sha256(raw_text.encode()).hexdigest(),
            'processed_text_sha256': hashlib.sha256(processed.encode()).hexdigest(),
            'limits': '256-token cap; synthetic image; no OCR quality or BF16 parity claim'}
        REPORT['generations'].append(generation)
        save()
        if not generation['prompt_ids_exact']:
            REPORT['first_mismatch'] = {'kind': 'expanded_prompt_ids', 'case': label,
                                        'actual': actual_ids, 'expected': expected_ids}
            save()
            raise ValueError('vLLM expanded prompt IDs differ from original HF processor: ' + label)
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
