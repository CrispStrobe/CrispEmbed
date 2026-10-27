#!/usr/bin/env python3
"""Hosted preservation and original-first head conversion; no backbone inference."""
import argparse
import ast
from collections import Counter
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
from josev2b_head_schema import reference, sha256
from convert_josev2b_head_f32 import convert

SOURCE = 'b2edf8c83c9aca1f8a9210198d47798ac13828b8'
RECEIPT = '51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090'
PUBLIC = ROOT/'tests/results/josev2b/original-reference.json'


def safe_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or len(p.parts) != 2 or p.parts[0] != 'arrays' or p.suffix != '.npy':
        raise ValueError('Unsafe original array member')
    return p


def frozen_origin(value):
    # The actual access reference stays in the operator's private configuration.
    parts = value.split('/')
    if len(parts) != 3 or not all(parts[:2]) or parts[2] != '5':
        raise ValueError('Exact configured version5 original required; no latest fallback')
    return value


def versioned_files(api, origin, request_factory):
    # Kaggle2.2.2 kernels_output parses /VERSION but does not send it. Set the
    # underlying SDK version_label explicitly on EVERY paginated request.
    owner, slug, version = frozen_origin(origin).split('/')
    files, token = {}, None
    with api.build_kaggle_client() as client:
        for _ in range(20):
            request = request_factory()
            request.user_name, request.kernel_slug = owner, slug
            request.version_label, request.page_size = "v"+version, 100
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                if item.file_name in files:
                    raise ValueError('Duplicate versioned output member')
                files[item.file_name] = item.url
            token = response.next_page_token
            if not token:
                return files
    raise ValueError('Versioned output listing exceeds metadata bound')


def array_header(path, row):
    """Bounded NPY header verification without importing an array framework."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes']:
        raise ValueError('Original array file extent differs')
    with path.open('rb') as f:
        if f.read(8) != b'\x93NUMPY\x01\x00':
            raise ValueError('Expected original NPY1 format')
        length = struct.unpack('<H', f.read(2))[0]
        if not 1 <= length <= 65535:
            raise ValueError('Invalid original NPY header bound')
        header = ast.literal_eval(f.read(length).decode('latin1'))
    types = {'float32': ('<f4', 4), 'int64': ('<i8', 8)}
    if row['dtype'] not in types:
        raise ValueError('Unsupported frozen array dtype')
    dtype, width = types[row['dtype']]
    shape = row['shape']
    if (not isinstance(shape, list) or not 1 <= len(shape) <= 4 or
            any(type(n) is not int or not 1 <= n <= 2048 for n in shape) or
            header != {'descr': dtype, 'fortran_order': False, 'shape': tuple(shape)} or
            path.stat().st_size != 10+length+math.prod(shape)*width):
        raise ValueError('Original array dtype/shape/layout differs')
    return math.prod(shape)*width


def verify_original(directory):
    report, cfg, _ = reference(directory/'reference.json', RECEIPT, SOURCE)
    freeze_path = directory/'input-freeze.json'
    if sha256(freeze_path) != report['input_freeze_sha256']:
        raise ValueError('Original input freeze differs')
    freeze = json.loads(freeze_path.read_text())
    if [r['id'] for r in freeze['records']] != [c['id'] for c in report['cases']]:
        raise ValueError('Original frozen input case order differs')
    for record, case in zip(freeze['records'], report['cases']):
        ids = record['encoded']['input_ids']
        if len(ids) != case['tokens'] or record['attention_mask'] != [1]*len(ids):
            raise ValueError('Original input token/mask contract differs')
        for field in record['encoded']['questions']:
            for a, b in [field['question_span'], *field['option_spans']]:
                if not 0 <= a < b <= len(ids):
                    raise ValueError('Original semantic span differs')
    if len(report['arrays']) != 251:
        raise ValueError('Complete original array coverage required')
    import numpy as np
    seen, payload_bytes, events = set(), 0, []
    counts = Counter()
    for row in report['arrays']:
        relative = safe_member(row['file'])
        if str(relative) in seen:
            raise ValueError('Duplicate original array member')
        seen.add(str(relative))
        path = directory/relative
        payload_bytes += array_header(path, row)
        if sha256(path) != row['sha256']:
            raise ValueError('Original array byte hash differs')
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        if np.isnan(data).any() or np.isposinf(data).any() or int(np.isneginf(data).sum()) != row['negative_infinity']:
            raise ValueError('Original finite/mask policy differs')
        key = row['case'], row['stage']
        occurrence = counts[key]; counts[key] += 1
        events.append({**row, 'occurrence': occurrence})
    if payload_bytes != report['array_payload_bytes']:
        raise ValueError('Original total payload extent differs')
    return report, cfg, freeze, events


def retrieve(directory):
    origin = frozen_origin(os.environ.get('JOSEV_REFERENCE_KERNEL', ''))
    # Bound transport before Kaggle's import-time authentication/update checks.
    import requests
    original = requests.Session.send
    def bounded(self, request, **kwargs):
        kwargs['timeout'] = 30
        return original(self, request, **kwargs)
    requests.Session.send = bounded
    try:
        # Do not print private access references from third-party diagnostics.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from kaggle import api
            from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
            files = versioned_files(api, origin, ApiListKernelSessionOutputRequest)
        report, _, _ = reference(PUBLIC, RECEIPT, SOURCE)
        public_freeze = ROOT/'tests/results/josev2b/original-input-freeze.json'
        declarations = [dict(file='reference.json',bytes=PUBLIC.stat().st_size,sha256=RECEIPT),
                        dict(file='input-freeze.json',bytes=public_freeze.stat().st_size,sha256=report['input_freeze_sha256']),
                        *report['arrays']]
        directory.mkdir()
        progress = dict(requested_version_label='v5',verified_files=[])
        for row in declarations:
            name = row['file']
            if name not in ('reference.json','input-freeze.json'):
                safe_member(name)
            url = files['josev2b-original/reference/'+name]
            if not url.startswith('https://'):
                raise ValueError('HTTPS output transport required')
            target = directory/name; target.parent.mkdir(parents=True, exist_ok=True)
            count, digest = 0, hashlib.sha256()
            with requests.get(url, stream=True, timeout=30) as response, target.open('xb') as out:
                response.raise_for_status()
                for chunk in response.iter_content(1024*1024):
                    count += len(chunk)
                    if count > row['bytes']:
                        raise ValueError('Original output transport exceeds exact extent')
                    digest.update(chunk); out.write(chunk)
            if count != row['bytes'] or digest.hexdigest() != row['sha256']:
                raise ValueError('Original output transport hash/extent differs')
            progress['verified_files'].append(name)
            (directory/'transfer.json').write_text(json.dumps(progress,indent=2)+'\n')
            if name == 'reference.json':
                reference(target, RECEIPT, SOURCE)
    finally:
        requests.Session.send = original


def download(cfg, name, path):
    declaration = cfg['files'][name]
    url = f'https://huggingface.co/{cfg["repository"]}/resolve/{cfg["revision"]}/{name}'
    h, count = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=120) as response, path.open('xb') as out:
        while chunk := response.read(1024*1024):
            count += len(chunk)
            if count > declaration['bytes']:
                raise ValueError('Published component source exceeds exact extent')
            h.update(chunk); out.write(chunk)
    if count != declaration['bytes'] or h.hexdigest() != declaration['sha256']:
        raise ValueError('Published component source identity differs')


def export_probe_inputs(directory, report, freeze, events, output):
    """Explicit batch1 storage adapter; every original observer shape is retained."""
    import numpy as np
    output.mkdir()
    records = []
    for record in freeze['records']:
        case = record['id']
        if PurePosixPath(case).name != case or case in ('.','..'):
            raise ValueError('Unsafe frozen case identity')
        target = output/case; target.mkdir()
        hidden = [r for r in events if r['case'] == case and r['stage'] == 'head/hidden_norm/input/0']
        lexical = [r for r in events if r['case'] == case and r['stage'] == 'head/option_lexical_projection/input/0']
        fields = record['encoded']['questions']; tokens = len(record['encoded']['input_ids'])
        if len(hidden) != 1 or len(lexical) != len(fields):
            raise ValueError('Complete ordered native-head source inputs required')
        manifest = dict(schema='crispembed-josev-head-input-v1', tokens=tokens,
                        hidden_f32='hidden.f32', fields=[])
        proof = dict(case=case, reference_sha256=RECEIPT, files={})
        def raw(row, filename, shape):
            if row['dtype'] != 'float32' or row['shape'] != shape:
                raise ValueError('Frozen head input shape/dtype differs')
            value = np.load(directory/safe_member(row['file']), allow_pickle=False, mmap_mode='r')
            payload = value.tobytes(order='C')
            if len(payload) != 4*math.prod(shape):
                raise ValueError('Head input storage extent differs')
            (target/filename).write_bytes(payload)
            proof['files'][filename] = dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
                original_array=row['file'],original_array_sha256=row['sha256'],original_shape=shape,
                stage=row['stage'],occurrence=row['occurrence'],storage_adapter='C-order bytes unchanged')
        raw(hidden[0], 'hidden.f32', [1,tokens,2048])
        for index, (field, source) in enumerate(zip(fields, lexical)):
            if source['occurrence'] != index:
                raise ValueError('Lexical field occurrence/order differs')
            name = f'lexical-{index}.f32'
            raw(source, name, [len(field['option_spans']),2048])
            manifest['fields'].append(dict(id=field['question_id'],type=field['question_type'],
                question_start=field['question_span'][0],question_end=field['question_span'][1],
                options=[dict(start=a,end=b) for a,b in field['option_spans']],lexical_f32=name))
        (target/'input.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
        records.append(proof)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        p.error('This preservation/conversion entry point is hosted CPU CI only')
    if args.output.exists():
        p.error('Fresh task-owned output required')
    # The already reviewed original precedes third-party imports or payload access.
    reference(PUBLIC, RECEIPT, SOURCE)
    frozen_origin(os.environ.get('JOSEV_REFERENCE_KERNEL', ''))
    args.output.mkdir(parents=True)
    small = args.output/'review'; small.mkdir()
    result = dict(schema='crispembed-josev-head-artifact-v1', completed=False,
                  reference_sha256=RECEIPT, original_source=SOURCE, native_parity=False,
                  model_quality=False, full_model=False, phase='retrieve',
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    try:
        import importlib.metadata
        for name, version in [('numpy','2.2.6'), ('gguf','0.17.1'), ('kaggle','2.2.2')]:
            if importlib.metadata.version(name) != version:
                raise ValueError('Pinned component tooling differs')
        original = args.output/'original'
        retrieve(original)
        result['phase'] = 'verify-original'
        report, cfg, freeze, events = verify_original(original)
        result.update(arrays=len(events), array_payload_bytes=report['array_payload_bytes'])
        (small/'ordered-events.json').write_text(json.dumps(events, indent=2)+'\n')
        result['phase'] = 'export-native-inputs'
        inputs = export_probe_inputs(original, report, freeze, events, args.output/'inputs')
        (small/'native-input-adapters.json').write_text(json.dumps(inputs,indent=2)+'\n')
        head = args.output/'head'; head.mkdir()
        result['phase'] = 'convert-head'
        with tempfile.TemporaryDirectory(prefix='josev-head-source-') as scratch:
            checkpoint = Path(scratch)/'joint_head.safetensors'
            download(cfg, 'joint_head.safetensors', checkpoint)
            for name in ('LICENSE', 'README.md', 'joint_schema_model.py', 'joint_head_config.json',
                         'config.json', 'unsloth_decision_config.json', 'tokenizer_config.json', 'processor_config.json'):
                download(cfg, name, head/name)
            converted = convert(checkpoint, original/'reference.json', RECEIPT, SOURCE, head/'joint-head-f32.gguf')
        shutil.copyfile(head/'joint-head-f32.conversion.json', small/'head-conversion.json')
        result.update(completed=True, phase='completed', gguf_sha256=converted['gguf_sha256'],
                      gguf_bytes=converted['gguf_bytes'], tensors=converted['tensor_count'])
    except Exception as error:
        # API exception strings can contain private origin URLs. Preserve phase
        # and error type while the immutable receipt and source enable diagnosis.
        result.update(error_type=type(error).__name__)
        if result['phase'] != 'retrieve':
            result['error_detail'] = str(error)
        raise RuntimeError('Hosted head artifact rejected at '+result['phase']+' ('+type(error).__name__+')') from None
    finally:
        result['files'] = {str(path.relative_to(args.output)): dict(bytes=path.stat().st_size,sha256=sha256(path))
                           for path in args.output.rglob('*') if path.is_file() and path != small/'artifact.json'}
        (small/'artifact.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
