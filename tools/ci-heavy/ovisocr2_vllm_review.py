"""Independent, model-free review of retained original GPU reference artifacts."""
import ast
import hashlib
import json
from pathlib import Path
import struct
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npz_records(path):
    """Read bounded original NPY fields without importing a numerical framework."""
    records = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 16:
            raise ValueError('Duplicate/excess processor NPZ fields')
        for name in names:
            item = archive.getinfo(name)
            if Path(name).name != name or not name.endswith('.npy') or item.file_size > 16 * 1024 * 1024:
                raise ValueError('Unexpected processor NPZ field')
            data = archive.read(name)
            if data[:6] != b'\x93NUMPY' or data[6:8] not in (b'\x01\x00', b'\x02\x00'):
                raise ValueError('Unsupported processor NPY version')
            width = 2 if data[6] == 1 else 4
            size = int.from_bytes(data[8:8 + width], 'little')
            if size > 65536 or 8 + width + size > len(data):
                raise ValueError('Invalid processor NPY header size')
            header = ast.literal_eval(data[8 + width:8 + width + size].decode('ascii').strip())
            shape, dtype = header['shape'], header['descr']
            types = {'<i8': ('int64', 8), '<f4': ('float32', 4), '|u1': ('uint8', 1)}
            if header['fortran_order'] or dtype not in types or not isinstance(shape, tuple):
                raise ValueError('Unexpected processor NPY layout or dtype')
            count = 1
            for value in shape:
                if type(value) is not int or value < 1:
                    raise ValueError('Invalid processor NPY dimension')
                count *= value
            payload = data[8 + width + size:]
            if len(payload) != count * types[dtype][1]:
                raise ValueError('Processor NPY byte count differs')
            records[name[:-4]] = {'shape': list(shape), 'dtype': types[dtype][0],
                                  'sha256': hashlib.sha256(payload).hexdigest()}
    return records


def review(directory, metadata, recipe_sha, worker_sha):
    root = Path(directory)
    producer = json.loads((root / 'ovisocr2-vllm-reference.json').read_text())
    worker = json.loads((root / 'ovisocr2-vllm-worker.json').read_text())
    if (producer.get('status') != 'completed bounded original vLLM T4 reference'
            or worker.get('status') != 'completed bounded original vLLM reference'
            or producer.get('worker') != worker or producer.get('script_sha256') != recipe_sha
            or producer.get('worker_sha256') != worker_sha
            or worker.get('resolved_vllm_dtype') != 'torch.float16'
            or worker.get('checkpoint_sha256') != metadata['checkpoint_lfs_declaration']['file']['lfs']['sha256']
            or worker.get('gpu_control', {}).get('passed') is not True
            or worker.get('runtime_stop_probe', {}).get('all_stop_ids') != [248044, 248046]):
        raise ValueError('Original GPU execution/precision/source proof failed')
    upstream = metadata['producer']
    if (worker['metadata_files_sha256'] != upstream['checkpoint_files_sha256']
            or worker['rendered_prompt_sha256'] != upstream['rendered_prompt_sha256']):
        raise ValueError('Original GPU metadata or rendered prompt differs')
    expected = {row['case']: row for row in upstream['inputs']}
    if [row['case'] for row in worker['cases']] != list(expected):
        raise ValueError('Original GPU processor case coverage differs')
    for row in worker['cases']:
        original = expected[row['case']]
        file = root / row['replay_file']
        fields = npz_records(file)
        if (sha(file) != row['replay_sha256'] or fields != original['fields']
                or fields != row['actual_fields'] or fields != row['expected_fields']
                or row['rgb_sha256'] != original['rgb_sha256']
                or row['actual_ids'] != original['ids'] or row['actual_grid'] != original['grid']
                or row['exact_original_arrays'] is not True):
            raise ValueError('Retained GPU processor array bytes differ: ' + row['case'])
    readable = worker['owned_readable']
    if sha(root / 'owned-readable.png') != readable['png_sha256']:
        raise ValueError('Retained readable PNG differs')
    if [row['case'] for row in worker['generations']] != ['square', 'readable']:
        raise ValueError('Original decoded case coverage differs')
    for row in worker['generations']:
        ids = row['token_ids']
        expected_ids = expected['square']['ids'][0] if row['case'] == 'square' else readable['hf_prompt_ids']
        if (row['prompt_token_ids'] != expected_ids or row['expected_hf_prompt_token_ids'] != expected_ids
                or row['prompt_ids_exact'] is not True or row['temperature'] != 0
                or row['bounded_max_tokens'] != 256 or not ids or len(ids) > 256
                or any(type(i) is not int or i < 0 or i >= 248320 for i in ids)
                or row['finish_reason'] not in ('stop', 'length')
                or (row['finish_reason'] == 'length' and len(ids) != 256)
                or (row['finish_reason'] == 'stop' and row['stop_reason'] not in (None, 248044, 248046))):
            raise ValueError('Original generation IDs/prompt/completion differs: ' + row['case'])
        for field in ('raw_text', 'processed_text'):
            if hashlib.sha256(row[field].encode()).hexdigest() != row[field + '_sha256']:
                raise ValueError('Original decoded text hash differs')
    for row in producer['commands']:
        path = root / 'ovisocr2-vllm-logs' / row['log']
        if row['exit_code'] != 0 or row['timed_out'] or sha(path) != row['log_sha256']:
            raise ValueError('Original command exit/log differs')
    return {'genuine_gpu_forward_completed': True, 'all_three_processor_arrays_rehashed': True,
            'actual_dtype': worker['resolved_vllm_dtype'], 'stops': [248044, 248046],
            'raw_ids_prompt_text_completion_checked': True,
            'recipe_sha256': recipe_sha, 'worker_sha256': worker_sha,
            'files': {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()},
            'limits': ['Bounded synthetic reference only; not representative OCR quality.',
                       'T4 auto-FP16 is separate from CPU F32 and newer-GPU BF16.',
                       'vLLM token IDs/decoded output retained; full GPU vocabulary logits were not requested.']}
