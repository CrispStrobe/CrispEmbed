#!/usr/bin/env python3
"""Original OvisOCR2 GPU quality capture; runs only inside the pinned hosted venv."""

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time
import unicodedata

OUT = Path(os.environ['OVIS_OUT'])
SCRATCH = Path(os.environ['OVIS_SCRATCH'])
REPO = SCRATCH / 'source'
MODEL = 'ATH-MaaS/OvisOCR2'
REVISION = 'cbdede82e048659b83310476ff807da825f8fe78'
STOPS = {248044, 248046}
REPORT = {'phase': 'start', 'status': 'started', 'outputs': {},
          'scope': 'One T4 controlled original vLLM TRITON_ATTN quality reference; auto FP16. No native, BF16, auto-backend, or benchmark claim.'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save():
    (OUT / 'ovisocr2-vllm-quality-worker.json').write_text(json.dumps(REPORT, indent=2, ensure_ascii=True, allow_nan=False) + '\n')


def source_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_confidences(result):
    ids = list(result.token_ids)
    logprobs = result.logprobs
    if logprobs is None:
        return {'available': False, 'reason': 'vLLM did not return requested selected-token logprobs'}
    if len(logprobs) != len(ids):
        return {'available': False, 'reason': 'Returned logprob positions do not align with returned IDs',
                'positions': len(logprobs), 'ids': len(ids)}
    values = []
    for token, choices in zip(ids, logprobs):
        if token not in choices:
            return {'available': False, 'reason': 'Selected token absent from its vLLM logprob position'}
        logprob = choices[token].logprob
        if not math.isfinite(logprob) or logprob > 0:
            raise ValueError('Nonfinite or positive selected-token logprob')
        values.append(math.exp(logprob))
    return {'available': True, 'source': 'actual vLLM SamplingParams(logprobs=1) selected-token logprobs',
            'values': values, 'mean': sum(values) / len(values) if values else None}


def checked_stop(result, cap):
    ids = list(result.token_ids)
    reason, explicit = result.finish_reason, result.stop_reason
    if len(ids) > cap or any(type(i) is not int or i < 0 or i >= 248320 for i in ids):
        raise ValueError('Over-cap or invalid original vLLM generated IDs')
    terminal = ids[-1] if ids and ids[-1] in STOPS else None
    if any(i in STOPS for i in ids[:-1]) or (explicit is not None and explicit not in STOPS):
        raise ValueError('Unexpected internal EOS or vLLM stop reason')
    if explicit is not None and terminal is not None and explicit != terminal:
        raise ValueError('Contradictory generated EOS and vLLM stop reason')
    if reason == 'stop':
        if terminal is None and explicit is None:
            raise ValueError('vLLM stopped without a pinned EOS source')
        return {'complete': True, 'stop_id': terminal or explicit,
                'terminal_eos_in_ids': terminal is not None,
                'special_stop_omitted_from_ids': terminal is None}
    if reason == 'length' and terminal is None and explicit is None and len(ids) == cap:
        return {'complete': False, 'stop_id': None, 'terminal_eos_in_ids': False,
                'special_stop_omitted_from_ids': False}
    raise ValueError('Contradictory or unexplained original vLLM termination')


def score_case(contract, gold, row, cap):
    """Use the frozen scorer where IDs include EOS; preserve vLLM's omitted special stop."""
    if row['terminal_eos_in_ids']:
        normalized = {'raw_text': row['raw_text'], 'processed_text': row['processed_text'],
                      'token_ids': row['token_ids'], 'stop_id': row['stop_id'],
                      'finish_reason': row['finish_reason']}
        return contract.score_case(gold, normalized, cap)
    if row['special_stop_omitted_from_ids']:
        # vLLM documents omission of special stop tokens. Do not append a token
        # that was absent from its actual returned IDs.
        issues = []
        actual = unicodedata.normalize('NFC', row['processed_text'].replace('\r\n', '\n')).strip()
        if not actual:
            issues.append('empty')
        if '<img src="images/bbox_' in actual:
            issues.append('unexpected-image-tag')
        if any(ord(c) < 32 and c not in '\n\t' for c in actual):
            issues.append('control-character')
        if actual.count('```') % 2:
            issues.append('unclosed-code-fence')
        reference = unicodedata.normalize('NFC', gold.replace('\r\n', '\n')).strip()
        return {'gold': gold, 'raw_text': row['raw_text'], 'processed_text': row['processed_text'],
                'raw_exact': unicodedata.normalize('NFC', row['raw_text'].replace('\r\n', '\n')).strip() == reference,
                'processed_exact': actual == reference,
                'cer': contract._distance(reference, actual) / max(1, len(reference)),
                'wer': contract._distance(reference.split(), actual.split()) / max(1, len(reference.split())),
                'generated_tokens': len(row['token_ids']), 'cap_hit': False,
                'completed': not issues, 'format_issues': issues, 'stop_id': row['stop_id'],
                'finish_reason': row['finish_reason'],
                'scorer_note': 'Special EOS omitted by vLLM; token_ids preserved, explicit stop_reason validates completion'}
    normalized = {'raw_text': row['raw_text'], 'processed_text': row['processed_text'],
                  'token_ids': row['token_ids'], 'stop_id': None, 'finish_reason': 'length'}
    return contract.score_case(gold, normalized, cap)


def device_memory_used_mib():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    return int(raw.splitlines()[0].strip())


def main():
    save()
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'torchvision', 'transformers', 'tokenizers', 'numpy', 'Pillow', 'huggingface-hub', 'vllm')}
    REPORT['versions'] = versions
    REPORT['torch_cuda'] = torch.version.cuda
    if (versions['torch'].split('+')[0] != '2.11.0' or versions['transformers'] != '5.16.1'
            or versions['tokenizers'] != '0.23.3' or versions['Pillow'] != '12.1.0'
            or versions['vllm'] not in ('0.22.1', '0.22.1+cu129')
            or not str(torch.version.cuda).startswith('12.9') or not torch.cuda.is_available()
            or torch.cuda.get_device_capability() != (7, 5) or 'T4' not in torch.cuda.get_device_name()):
        raise ValueError('Pinned original GPU package or T4 differs')
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda', dtype=torch.float16)
    right = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device='cuda', dtype=torch.float16)
    if (left @ right).cpu().tolist() != [[2.0, 4.0], [6.0, 8.0]]:
        raise ValueError('T4 FP16 control failed')
    REPORT['gpu_control'] = {'dtype': 'float16', 'result': [[2.0, 4.0], [6.0, 8.0]]}
    REPORT['phase'] = 'source-and-images'; save()
    contract = source_module('ovis_quality_contract_gpu', 'tools/ci-heavy/ovisocr2_quality_contract.py')
    quality_run = source_module('ovis_quality_run_gpu', 'tools/ci-heavy/ovisocr2_quality_run.py')
    ocr_quality = source_module('ovis_document_scorer_gpu', 'tools/ocr_quality.py')
    panel = contract.load_panel(REPO / 'tests/results/qwen35-ocr/ovisocr2-quality-panel.json')
    synthetic = contract.render_panel(OUT / 'synthetic', panel)
    prepared = json.loads((REPO / 'tests/results/ocr-quality-prepared.json').read_text())
    reviewed = json.loads((REPO / 'tests/regression/ocr-quality/panel.json').read_text())
    if ([case['id'] for case in prepared['cases']] != [case['id'] for case in reviewed['cases']]
            or len(prepared['cases']) != 3 or prepared['frozen_derivatives_exact'] is not None
            or prepared['hosted_replay']['report']['frozen_derivatives_exact'] is not True):
        raise ValueError('Reviewed realistic document preparation differs')
    cases = []
    for spec, frozen in zip(panel['cases'], synthetic['cases']):
        cases.append({'id': spec['id'], 'split': spec['split'], 'kind': 'synthetic',
                      'license': panel['license'], 'gold': spec['gold'], 'cap': 256,
                      'file': str(OUT / 'synthetic' / frozen['file']),
                      'png_sha256': frozen['png_sha256'], 'rgb_sha256': frozen['rgb_sha256']})
    for spec, frozen in zip(reviewed['cases'], prepared['cases']):
        name = frozen['file']
        path = SCRATCH / 'realistic' / name
        if (spec['id'] != frozen['id'] or spec['max_new_tokens'] != 1024
                or sha(path) != frozen['png_sha256']):
            raise ValueError('Reviewed realistic PNG identity differs: ' + name)
        with Image.open(path) as picture:
            image = picture.convert('RGB')
            if (list(image.size) != frozen['size']
                    or hashlib.sha256(image.tobytes()).hexdigest() != frozen['rgb_sha256']):
                raise ValueError('Reviewed realistic RGB identity differs: ' + name)
        gold_path = REPO / spec['gold']
        if sha(gold_path) != frozen['gold_sha256']:
            raise ValueError('Reviewed realistic gold bytes differ: ' + spec['id'])
        gold = json.loads(gold_path.read_text())
        cases.append({'id': spec['id'], 'split': 'realistic-evaluation', 'kind': spec['kind'],
                      'license': spec['license'], 'gold': gold['text'], 'cap': 1024,
                      'file': str(path), 'png_sha256': frozen['png_sha256'],
                      'rgb_sha256': frozen['rgb_sha256'], 'gold_path': str(gold_path)})
    REPORT['inputs'] = [{k: value for k, value in case.items() if k not in ('file', 'gold_path')}
                        for case in cases]
    REPORT['rendered_panel_sha256'] = sha(OUT / 'synthetic' / 'rendered-panel.json')
    REPORT['prepared_manifest_sha256'] = sha(REPO / 'tests/results/ocr-quality-prepared.json')
    REPORT['inputs_frozen_before_model'] = True
    save()

    import urllib.request
    evidence_url = ('https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/'
                    + os.environ['OVIS_PROOF_COMMIT']
                    + '/tests/results/qwen35-ocr/ovisocr2-metadata.json')
    with urllib.request.urlopen(evidence_url, timeout=30) as response:
        evidence = json.load(response)
    producer = evidence['producer']
    source = Path(snapshot_download(MODEL, revision=REVISION,
                  allow_patterns=sorted(producer['checkpoint_files_sha256'])))
    for name, expected in producer['checkpoint_files_sha256'].items():
        if sha(source / name) != expected:
            raise ValueError('Original metadata source differs: ' + name)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    readme = (source / 'README.md').read_text()
    prompt, cleaner, filter_fn = quality_run.original_readme_functions(readme)
    REPORT['original_readme_sha256'] = sha(source / 'README.md')
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    rendered = processor.tokenizer.apply_chat_template(messages, tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=False)
    if rendered != evidence['rendered_prompt']:
        raise ValueError('Original rendered prompt differs')
    REPORT['rendered_prompt_sha256'] = hashlib.sha256(rendered.encode()).hexdigest()
    REPORT['phase'] = 'checkpoint'; save()
    checkpoint = Path(hf_hub_download(MODEL, 'model.safetensors', revision=REVISION))
    if checkpoint.stat().st_size != 1706030496 or sha(checkpoint) != os.environ['OVIS_CHECKPOINT_SHA']:
        raise ValueError('Original checkpoint payload differs')
    REPORT['checkpoint_sha256'] = sha(checkpoint)
    REPORT['phase'] = 'model-init'; save()
    model = LLM(model=str(source), tensor_parallel_size=1, gpu_memory_utilization=0.8,
                gdn_prefill_backend='triton', attention_config={'backend': 'TRITON_ATTN'})
    backend = model.llm_engine.vllm_config.attention_config.backend
    dtype = str(model.model_config.dtype)
    REPORT['resolved'] = {'dtype': dtype, 'decoder_attention': backend.name,
                          'gdn_prefill_backend': 'triton', 'vision_attention': 'auto unchanged'}
    if dtype != 'torch.float16' or backend.name != 'TRITON_ATTN':
        raise ValueError('Controlled original dtype/backend differs')
    tokenizer = model.get_tokenizer()
    runtime_generation = model.model_config.try_get_generation_config()
    probe = SamplingParams(max_tokens=256, temperature=0.0)
    probe.update_from_generation_config(runtime_generation, tokenizer.eos_token_id)
    REPORT['runtime_stops'] = {'primary_eos': probe._eos_token_id,
                               'additional': sorted(probe.stop_token_ids),
                               'all': sorted(probe._all_stop_token_ids)}
    if tokenizer.eos_token_id != 248046 or REPORT['runtime_stops']['all'] != [248044, 248046]:
        raise ValueError('Original runtime stop resolution differs')
    generated_prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
    if generated_prompt != rendered:
        raise ValueError('Original tokenizer prompt differs')
    REPORT['phase'] = 'original-gpu-quality'; save()
    for case in cases:
        REPORT['phase'] = 'original-gpu-quality-' + case['id']; save()
        image = Image.open(case['file']).convert('RGB')
        original_inputs = processor(text=[rendered], images=[image], return_tensors='pt',
                                    images_kwargs={'min_pixels': 448 * 448, 'max_pixels': 2880 * 2880})
        expected_ids = original_inputs['input_ids'].detach().cpu().tolist()[0]
        sampling = SamplingParams(max_tokens=case['cap'], temperature=0.0, logprobs=1)
        item = {'prompt': generated_prompt, 'multi_modal_data': {'image': image},
                'mm_processor_kwargs': {'images_kwargs': {'min_pixels': 448 * 448,
                                                          'max_pixels': 2880 * 2880}}}
        device_before = device_memory_used_mib()
        started = time.monotonic()
        generated = model.generate([item], sampling)
        wall = time.monotonic() - started
        device_after = device_memory_used_mib()
        if len(generated) != 1 or len(generated[0].outputs) != 1:
            raise ValueError('Original vLLM output count differs: ' + case['id'])
        actual_prompt_ids = list(generated[0].prompt_token_ids)
        result = generated[0].outputs[0]
        raw = result.text
        processed = quality_run.process_readme_output(raw, cleaner, filter_fn)
        termination = checked_stop(result, case['cap'])
        row = {'id': case['id'], 'split': case['split'], 'kind': case['kind'],
               'input_rgb_sha256': case['rgb_sha256'], 'input_png_sha256': case['png_sha256'],
               'cap': case['cap'], 'prompt_token_ids': actual_prompt_ids,
               'expected_hf_prompt_token_ids': expected_ids,
               'prompt_ids_exact': actual_prompt_ids == expected_ids,
               'token_ids': list(result.token_ids), 'finish_reason': result.finish_reason,
               'vllm_stop_reason': result.stop_reason, 'raw_text': raw,
               'processed_text': processed, 'wall_seconds': wall,
               'process_max_rss_kib_including_init': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'cuda_max_memory_allocated_bytes_including_init': torch.cuda.max_memory_allocated(),
               'gpu_memory_used_mib_snapshot_before': device_before,
               'gpu_memory_used_mib_snapshot_after': device_after,
               'selected_token_confidences': selected_confidences(result), **termination}
        REPORT['outputs'][case['id']] = row
        save()  # Preserve IDs/text/stops before scoring or any later case fails.
        if not row['prompt_ids_exact']:
            raise ValueError('Original vLLM expanded prompt IDs differ: ' + case['id'])
        scored = score_case(contract, case['gold'], row, case['cap'])
        row['score'] = scored
        if case['split'] == 'realistic-evaluation':
            gold = json.loads(Path(case['gold_path']).read_text())
            row['structure'] = ocr_quality.score(gold, processed, complete=scored['completed'])
        save()
    REPORT['summary'] = {}
    for split in ('development', 'heldout', 'realistic-evaluation'):
        group = [REPORT['outputs'][case['id']] for case in cases if case['split'] == split]
        REPORT['summary'][split] = {
            'cases': len(group), 'completed': sum(row['score']['completed'] for row in group),
            'cap_hits': sum(row['score']['cap_hit'] for row in group),
            'mean_cer': sum(row['score']['cer'] for row in group) / len(group),
            'mean_wer': sum(row['score']['wer'] for row in group) / len(group),
            'wall_seconds_total': sum(row['wall_seconds'] for row in group),
            'scope': 'One original T4 session; process RSS and GPU memory samples include initialization'}
    REPORT['status'] = 'completed bounded original vLLM quality reference'
    REPORT['phase'] = 'complete'; save()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        REPORT['status'] = 'failed original GPU quality reference'
        REPORT['error_type'] = type(error).__name__
        REPORT['error_phase'] = REPORT['phase']
        save()
        raise
