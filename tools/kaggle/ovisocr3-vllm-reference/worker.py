#!/usr/bin/env python3
"""Original OvisOCR3 plain OCR GPU worker; executes only after launcher gates."""

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import urllib.request


OUT = Path(os.environ['OVIS_OUT'])
MODEL = 'ATH-MaaS/OvisOCR3'
REVISION = '5f6ce6932efee9ec10c30746db55bd780214b7fd'
PROMPT_SHA = 'e037cf0013a757e19eae6d4d701f3b90848449fbebe44d1f8fa7d8ca4a5af9a7'
SOURCE_SHA = '8ca73ca6a349b67a7533c36de6c14ac066e1152f2c8d517af849b2655087a331'
CASES = (('square', 448, 448), ('odd', 449, 417), ('small-control', 96, 64))
REPORT = {'phase': 'started', 'status': 'started', 'processor_cases': [], 'generations': [],
          'scope': 'Original plain OCR greedy OvisOCR3 on T4; Triton GDN and controlled Triton decoder attention; no BF16, auto-attention, native or benchmark claim'}


def save():
    (OUT / 'ovisocr3-vllm-worker.json').write_text(json.dumps(REPORT, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                ensure_ascii=True, allow_nan=False).encode())


def logprob_steps(result):
    """Retain each generated token's selected score and all requested top 20."""
    ids = list(result.token_ids)
    steps = result.logprobs
    if steps is None or len(steps) != len(ids):
        raise ValueError('vLLM omitted generated-token logprob steps')
    records = []
    for index, (token_id, candidates) in enumerate(zip(ids, steps)):
        if token_id not in candidates:
            raise ValueError('vLLM omitted selected generated-token score at step ' + str(index))
        row = []
        for candidate_id, item in sorted(candidates.items()):
            value = float(item.logprob)
            if not math.isfinite(value):
                raise ValueError('Nonfinite generated-token logprob at step ' + str(index))
            row.append({'token_id': int(candidate_id), 'logprob': value,
                        'rank': item.rank, 'decoded_token': item.decoded_token})
        records.append({'step': index, 'selected_token_id': int(token_id),
                        'selected_logprob': float(candidates[token_id].logprob),
                        'top_requested': 20, 'scores': row})
    return records


def main():
    save()
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoProcessor

    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'torchvision', 'transformers', 'tokenizers', 'numpy',
                 'Pillow', 'huggingface-hub', 'vllm')}
    REPORT['versions'] = versions
    REPORT['torch_cuda'] = torch.version.cuda
    if (versions['torch'].split('+')[0] != '2.11.0' or versions['torchvision'].split('+')[0] != '0.26.0'
            or versions['transformers'] != '5.16.1' or versions['tokenizers'] != '0.23.3'
            or versions['vllm'] not in ('0.22.1', '0.22.1+cu129')
            or not str(torch.version.cuda).startswith('12.9') or not torch.cuda.is_available()):
        raise ValueError('Pinned package or CUDA runtime differs')
    if torch.cuda.get_device_capability() != (7, 5) or 'T4' not in torch.cuda.get_device_name():
        raise ValueError('Torch selected another GPU; no fallback')
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda', dtype=torch.float16)
    b = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device='cuda', dtype=torch.float16)
    result = (a @ b).cpu().tolist()
    if result != [[2.0, 4.0], [6.0, 8.0]]:
        raise ValueError('Pinned T4 FP16 CUDA matmul control failed')
    REPORT['gpu_control'] = {'device': torch.cuda.get_device_name(), 'capability': [7, 5],
                             'dtype': 'float16', 'result': result, 'passed': True}
    REPORT['phase'] = 'reviewed-metadata'; save()
    url = ('https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/'
           + os.environ['OVIS_PROOF_COMMIT']
           + '/tests/results/qwen35-ocr/ovisocr3-metadata.json')
    with urllib.request.urlopen(url, timeout=30) as response:
        evidence = json.load(response)
    producer = evidence['producer']
    if (canonical_sha(producer) != os.environ['OVIS_PRODUCER_SHA']
            or evidence['independent_review']['metadata_positive'] is not True
            or producer['source_report_sha256'] != SOURCE_SHA):
        raise ValueError('Reviewed OvisOCR3 metadata proof differs in worker')
    names = producer['source_files_sha256']
    if (len(names) != 31 or 'model.safetensors' in names
            or any(name.startswith('/') or '..' in Path(name).parts for name in names)):
        raise ValueError('OvisOCR3 metadata file allowlist differs')
    source = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=sorted(names)))
    for name, expected in names.items():
        if sha(source / name) != expected:
            raise ValueError('Pinned OvisOCR3 metadata source differs: ' + name)
    REPORT['source_files_sha256'] = names
    processor = AutoProcessor.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    prompt = (source / 'prompts/plain_ocr.txt').read_text()
    if sha_bytes(prompt.encode()) != PROMPT_SHA or producer['task_prompt_sha256'] != PROMPT_SHA:
        raise ValueError('Original plain OCR task prompt differs')
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    rendered = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    rendered_sha = sha_bytes(rendered.encode())
    if rendered_sha != producer['rendered_prompt_sha256']:
        raise ValueError('Original rendered plain OCR prompt differs across CPU/GPU stacks')
    REPORT['prompt'] = {'task': 'plain_ocr', 'task_sha256': PROMPT_SHA,
                        'rendered_sha256': rendered_sha,
                        'rendered_utf8_bytes': len(rendered.encode())}
    rows = {row['case']: row for row in producer['inputs']}
    if set(rows) != {label for label, _, _ in CASES}:
        raise ValueError('Original processor case set differs')
    first_image = None
    for label, width, height in CASES:
        REPORT['phase'] = 'processor-' + label; save()
        y, x = np.indices((height, width))
        rgb = np.stack(((x * 3 + y) % 256, (y * 5 + x) % 256,
                        (x + y * 7) % 256), axis=-1).astype(np.uint8)
        image = Image.fromarray(rgb, 'RGB')
        row = rows[label]
        if list(rgb.shape) != row['rgb_shape'] or sha_bytes(rgb.tobytes()) != row['rgb_sha256']:
            raise ValueError('Source synthetic RGB differs: ' + label)
        prepared = processor(text=[rendered], images=[image], return_tensors='pt',
                             images_kwargs={'min_pixels': 448 * 448, 'max_pixels': 2880 * 2880})
        arrays = {key: value.detach().cpu().numpy() for key, value in prepared.items()}
        replay_path = OUT / (label + '-processor-replay.npz')
        np.savez_compressed(replay_path, **arrays)
        actual_fields = {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                               'sha256': sha_bytes(value.tobytes())} for key, value in arrays.items()}
        record = {'case': label, 'rgb_sha256': row['rgb_sha256'],
                  'replay_file': replay_path.name, 'replay_sha256': sha(replay_path),
                  'actual_fields': actual_fields, 'expected_fields': row['fields'],
                  'exact_original_fields': False}
        REPORT['processor_cases'].append(record); save()
        if set(actual_fields) != set(row['fields']):
            record['first_mismatch'] = {'kind': 'field_names', 'actual': sorted(actual_fields),
                                        'expected': sorted(row['fields'])}
            save(); raise ValueError('Original processor field set differs: ' + label)
        for key in sorted(actual_fields):
            if actual_fields[key] != row['fields'][key]:
                record['first_mismatch'] = {'kind': 'field', 'field': key,
                                            'actual': actual_fields[key], 'expected': row['fields'][key]}
                save(); raise ValueError('Original processor field differs: ' + label + ':' + key)
        for key, expected in (('input_ids', row['ids']), ('image_grid_thw', row['grid']),
                              ('attention_mask', row['attention_mask'])):
            if arrays[key].tolist() != expected:
                record['first_mismatch'] = {'kind': key, 'actual': arrays[key].tolist(), 'expected': expected}
                save(); raise ValueError('Original processor IDs/grid/mask differs: ' + label + ':' + key)
        record['exact_original_fields'] = True
        save()
        if label == 'square':
            first_image = image
    REPORT['phase'] = 'owned-readable'; save()
    readable = Image.new('RGB', (448, 448), 'white')
    ImageDraw.Draw(readable).text((30, 80), 'Hello OCR\n12345', fill='black', font_size=24)
    readable_path = OUT / 'owned-readable.png'
    readable.save(readable_path)
    readable_inputs = processor(text=[rendered], images=[readable], return_tensors='pt',
                                images_kwargs={'min_pixels': 448 * 448, 'max_pixels': 2880 * 2880})
    readable_ids = readable_inputs['input_ids'].detach().cpu().numpy().tolist()[0]
    REPORT['owned_readable'] = {'case': 'readable', 'gold_smoke_only': 'Hello OCR\n12345',
                                'rgb_sha256': sha_bytes(np.asarray(readable).tobytes()),
                                'png_sha256': sha(readable_path), 'width': 448, 'height': 448,
                                'hf_prompt_ids': readable_ids,
                                'hf_grid': readable_inputs['image_grid_thw'].detach().cpu().numpy().tolist()}
    save()
    REPORT['phase'] = 'checkpoint'; save()
    checkpoint = Path(hf_hub_download(MODEL, 'model.safetensors', revision=REVISION))
    checkpoint_sha = sha(checkpoint)
    if (checkpoint.stat().st_size != int(os.environ['OVIS_CHECKPOINT_BYTES'])
            or checkpoint_sha != os.environ['OVIS_CHECKPOINT_SHA']):
        raise ValueError('OvisOCR3 checkpoint bytes/SHA differ')
    REPORT['checkpoint_sha256'] = checkpoint_sha
    REPORT['phase'] = 'original-vllm-init'; save()
    from vllm import LLM, SamplingParams
    REPORT['attention_backend_override'] = {
        'readme_decoder_attention': 'auto', 'requested_decoder_attention': 'TRITON_ATTN',
        'vision_attention': 'auto unchanged',
        'reason': 'Controlled T4 route from retained OvisOCR2 FlashInfer warmup failure',
        'readme_auto_backend_equivalence_qualified': False}
    save()
    model = LLM(model=str(source), tensor_parallel_size=1, gpu_memory_utilization=0.8,
                gdn_prefill_backend='triton', attention_config={'backend': 'TRITON_ATTN'},
                limit_mm_per_prompt={'image': 2})
    backend = model.llm_engine.vllm_config.attention_config.backend
    REPORT['resolved_decoder_attention_backend'] = backend.name
    REPORT['resolved_vllm_dtype'] = str(model.model_config.dtype)
    if backend.name != 'TRITON_ATTN' or REPORT['resolved_vllm_dtype'] != 'torch.float16':
        raise ValueError('Controlled T4 backend/dtype resolution differs')
    tokenizer = model.get_tokenizer()
    runtime_generation = model.model_config.try_get_generation_config()
    probe = SamplingParams(max_tokens=256, temperature=0.0)
    probe.update_from_generation_config(runtime_generation, tokenizer.eos_token_id)
    REPORT['runtime_stop_probe'] = {
        'source': 'installed vLLM 0.22.1 SamplingParams.update_from_generation_config',
        'generation_config': runtime_generation, 'primary_eos': probe._eos_token_id,
        'additional_stop_ids': sorted(probe.stop_token_ids),
        'all_stop_ids': sorted(probe._all_stop_token_ids)}
    if (tokenizer.eos_token_id != 248046
            or REPORT['runtime_stop_probe']['all_stop_ids'] != [248044, 248046]
            or REPORT['runtime_stop_probe']['all_stop_ids'] != producer['effective_stops_source_probe']['all_stop_ids']):
        raise ValueError('Installed vLLM generation stop resolution differs')
    generated_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if generated_prompt != rendered:
        raise ValueError('vLLM tokenizer rendered prompt differs from original processor')
    REPORT['phase'] = 'bounded-greedy'; save()
    cap = 256
    sampling = SamplingParams(max_tokens=cap, temperature=0.0, logprobs=20)
    items = [{'prompt': generated_prompt, 'multi_modal_data': {'image': image},
              'mm_processor_kwargs': {'images_kwargs': {'min_pixels': 448 * 448,
                                                        'max_pixels': 2880 * 2880}}}
             for image in (first_image, readable)]
    outputs = model.generate(items, sampling)
    if len(outputs) != 2 or any(len(item.outputs) != 1 for item in outputs):
        raise ValueError('Original vLLM output count differs')
    for label, output in zip(('square', 'readable'), outputs):
        selected = output.outputs[0]
        expected_ids = rows['square']['ids'][0] if label == 'square' else readable_ids
        actual_ids = list(output.prompt_token_ids) if output.prompt_token_ids is not None else None
        scores = logprob_steps(selected)
        raw_text = selected.text
        tokens = list(selected.token_ids)
        record = {'case': label, 'readme_max_tokens': 16384, 'bounded_max_tokens': cap,
                  'temperature': 0.0, 'dtype_policy': 'README auto -> T4 FP16',
                  'gdn_prefill_backend': 'triton', 'decoder_attention_backend': 'TRITON_ATTN',
                  'prompt_token_ids': actual_ids, 'expected_hf_prompt_token_ids': expected_ids,
                  'prompt_ids_exact': actual_ids == expected_ids,
                  'token_ids': tokens, 'generated_scores': scores,
                  'finish_reason': selected.finish_reason, 'stop_reason': selected.stop_reason,
                  'raw_text': raw_text, 'raw_text_sha256': sha_bytes(raw_text.encode()),
                  'stripped_text': raw_text.strip(), 'reached_eos': bool(tokens and tokens[-1] in (248044, 248046)),
                  'cap_hit': len(tokens) >= cap,
                  'limits': '256-token cap; synthetic image; no OCR quality or BF16/native parity claim'}
        REPORT['generations'].append(record); save()
        if not record['prompt_ids_exact']:
            REPORT['first_mismatch'] = {'kind': 'expanded_prompt_ids', 'case': label,
                                        'actual': actual_ids, 'expected': expected_ids}
            save(); raise ValueError('Expanded vLLM prompt IDs differ: ' + label)
        if not record['reached_eos'] or record['cap_hit']:
            REPORT['first_mismatch'] = {'kind': 'generation_stop', 'case': label,
                                        'finish_reason': selected.finish_reason,
                                        'stop_reason': selected.stop_reason, 'token_count': len(tokens)}
            save(); raise ValueError('Bounded OvisOCR3 generation did not reach EOS: ' + label)
    REPORT['status'] = 'completed bounded original OvisOCR3 vLLM reference'
    REPORT['phase'] = 'complete'; save()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        REPORT['status'] = 'failed original OvisOCR3 vLLM worker'
        REPORT['error_type'] = type(error).__name__
        REPORT['error_message'] = str(error)[:300]
        save()
        raise
