#!/usr/bin/env python3
"""Hosted conversion and complete tensor readback. No inference/quality claim.

Dependencies are installed by the caller in an isolated environment using the
reviewed lighton4b-gdn-requirements.txt and a retained pip --report resolution.
The caller supplies an already-built portable crispembed-quantize executable.
No service credentials or provider identities are written to the report.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 2**20):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def command(argv, log):
    start = time.monotonic()
    with log.open('w') as stream:
        subprocess.run([str(x) for x in argv], cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    return {'seconds': time.monotonic()-start, 'log': log.name, 'log_sha256': digest(log)}


def load_converter():
    sys.path.insert(0, str(ROOT / 'models'))
    spec = importlib.util.spec_from_file_location('conversion', ROOT / 'models/convert-qwen35-ocr-to-gguf.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(path, plan, manifest, baseline=None, recipe=None):
    """Independent mmap reader, bounded finite/dequantization scans and byte hashes."""
    import numpy as np
    import gguf
    from gguf.quants import dequantize
    reader = gguf.GGUFReader(str(path), 'r')
    metadata = {}
    for name, field in reader.fields.items():
        h = hashlib.sha256()
        for part in field.parts:
            h.update(memoryview(part))
        metadata[name] = h.hexdigest()
    if baseline:
        for name, value in baseline['metadata_sha256'].items():
            if not name.startswith('GGUF.') and name not in ('general.file_type', 'general.quantization_version'):
                require(metadata.get(name) == value, 'source metadata changed: '+name)
    wanted = {r['name']: r for r in plan['records']}
    require(len(reader.tensors) == len(wanted) == 723, 'exact 723 tensor census required')
    require(len({t.name for t in reader.tensors}) == 723, 'duplicate tensor name')
    manifest = {r['name']: r for r in manifest['records']}
    result = {}
    ranges = []
    for tensor in reader.tensors:
        require(tensor.name in wanted, 'unexpected tensor')
        row = wanted[tensor.name]
        require(list(tensor.shape) == list(reversed(row['shape'] or [1])), 'tensor shape differs: '+tensor.name)
        kind = tensor.tensor_type
        require(kind in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
                         gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_K), 'undeclared storage type')
        raw = tensor.data.view(np.uint8).reshape(-1)
        h = hashlib.sha256()
        # Each scan chunk is a multiple of both supported quantized block byte sizes.
        for start in range(0, raw.size, 34*144*1024):
            chunk = raw[start:start+34*144*1024]
            h.update(memoryview(chunk))
            values = (chunk.view(np.float32) if kind == 0 else chunk.view(np.float16) if kind == 1
                      else dequantize(chunk, kind))
            require(np.isfinite(values).all(), 'nonfinite tensor: '+tensor.name)
        item = {'type': kind.name, 'shape': [int(n) for n in tensor.shape],
                'bytes': int(raw.size), 'sha256': h.hexdigest(), 'offset': int(tensor.data_offset)}
        require(item['offset'] % 32 == 0, 'unaligned tensor')
        ranges.append((item['offset'], item['offset']+item['bytes']))
        if baseline is None:
            require(kind.name == ('F16' if row['dtype'] == 'float16' else 'F32'), 'baseline type differs')
            require(item['sha256'] == manifest[tensor.name]['sha256'], 'baseline roundtrip payload differs')
        else:
            old = baseline['tensors'][tensor.name]
            protected = tensor.name.startswith('v.') or len(row['shape']) != 2 or 'norm' in tensor.name
            expected = (old['type'] if protected else 'Q8_0' if recipe == 'q8_0' or
                        'embed' in tensor.name or 'embd' in tensor.name else 'Q4_K')
            require(item['type'] == expected, 'planned quantizer type differs: '+tensor.name)
            if protected:
                require(item['type'] == old['type'] and item['sha256'] == old['sha256'],
                        'protected payload differs: '+tensor.name)
        result[tensor.name] = item
    ranges.sort()
    require(all(a[1] <= b[0] for a,b in zip(ranges, ranges[1:])), 'overlapping tensor ranges')
    require(ranges[-1][1] <= path.stat().st_size, 'truncated payload')
    if baseline:
        require(any(r['type'] in ('Q8_0', 'Q4_K') for r in result.values()), 'no quantized matrices')
    return {'filename': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path),
            'type_counts': dict(Counter(r['type'] for r in result.values())), 'tensors': result,
            'metadata_sha256': metadata,
            'reader_roundtrip_verified': True, 'native_parser_supported': False,
            'native_inference_observed': False, 'quality_qualified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--quantizer', type=Path, required=True)
    parser.add_argument('--quantizer-sha256', required=True,
                        help='Digest from the independently retained hosted-CI build receipt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('GITHUB_ACTIONS') == 'true' or
            (Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir()), 'hosted execution required')
    actual = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    require(actual == args.source_sha and len(actual) == 40, 'immutable source differs')
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], cwd=ROOT), 'tracked source dirty')
    require(digest(args.quantizer) == args.quantizer_sha256 and len(args.quantizer_sha256) == 64,
            'hosted-CI quantizer payload differs')
    args.output.mkdir(parents=True, exist_ok=True)
    require(not list(args.output.iterdir()), 'fresh output directory required')
    # Framework caches and writer temporary files share the measured filesystem.
    os.environ['HF_HOME'] = str(args.output.resolve()/'cache')
    os.environ['HF_HUB_CACHE'] = str(args.output.resolve()/'cache/hub')
    os.environ['TMPDIR'] = str(args.output.resolve()/'tmp')
    Path(os.environ['TMPDIR']).mkdir()
    report = {'schema': 'lighton4b-convert-quant/v1', 'status': 'failed', 'source_sha': actual,
              'runner_sha256': digest(__file__), 'quantizer_sha256': digest(args.quantizer),
              'native_supported': False, 'quality_qualified': False, 'artifacts': {}}
    report['submodules'] = subprocess.check_output(['git','submodule','status','--recursive'], cwd=ROOT, text=True).strip()
    require(not any(line.startswith(('-', '+', 'U')) for line in report['submodules'].splitlines()),
            'submodules must be initialized at their exact source pins')
    output = args.output / 'conversion-receipt.json'
    try:
        converter = load_converter()
        policy = converter._lighton4b_profile()
        config = json.loads((policy.FIXTURE/'config.json').read_text())
        plan = converter.lighton4b_tensor_plan(config, 'f16', recipe=converter.LIGHTON4B_VISION_RECIPE)
        payload = sum(math.prod(r['shape'])* (2 if r['dtype']=='float16' else 4) for r in plan['records'])
        # Two candidate upper bounds at baseline size, original checkpoint,
        # converter temporary duplicate, tokenizer/framework/build and headroom.
        needed = policy.CHECKPOINT_BYTES + 4*payload + 12*2**30
        report['storage_preflight'] = {'required_free_bytes': needed,
            'free_bytes': shutil.disk_usage(args.output).free, 'baseline_payload_bound': payload}
        require(shutil.disk_usage(args.output).free >= needed, 'insufficient disk before checkpoint acquisition')
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:')))*1024
        report['available_memory_bytes'] = available
        require(available >= payload+8*2**30, 'insufficient memory before checkpoint acquisition')
        url = f'https://huggingface.co/{policy.MODEL}/resolve/{policy.REVISION}/config.json'
        with urllib.request.urlopen(url, timeout=45) as response:
            acquired_config = response.read(2**20)
        require(acquired_config == (policy.FIXTURE/'config.json').read_bytes(), 'worker network/source delivery probe differs')
        report['network_config_sha256'] = hashlib.sha256(acquired_config).hexdigest()
        report['meta'] = command([sys.executable, ROOT/'tools/ci-heavy/lighton4b_export_meta.py',
                                  '--output',args.output/'meta.json'], args.output/'meta.log')
        baseline = args.output/'lightonocr3-4b-vision-f32.gguf'
        report['conversion'] = command([sys.executable, ROOT/'models/convert-qwen35-ocr-to-gguf.py',
            '--model',policy.MODEL,'--revision',policy.REVISION,'--profile','lightonocr3-4b',
            '--dtype','f16','--precision-recipe',converter.LIGHTON4B_VISION_RECIPE,
            '--output',baseline], args.output/'conversion.log')
        manifest = json.loads(baseline.with_suffix('.gguf.manifest.json').read_text())
        require(manifest['source_payload_verified'] and not manifest['gguf_roundtrip_verified'], 'export receipt contract differs')
        base = audit(baseline, plan, manifest)
        require(base['type_counts'] == {'F32':474,'F16':249}, 'baseline type census differs')
        report['artifacts']['baseline'] = base
        save(output, report)
        for recipe in ('q8_0','q4_k'):
            candidate = args.output/f'lightonocr3-4b-{recipe}-vision-f32.gguf'
            log = args.output/f'{recipe}.log'
            report[recipe] = command([args.quantizer.resolve(),baseline,candidate,recipe,
                                      '--tensor-type','v.*=f32'], log)
            require('(fallback ' not in log.read_text(), 'undeclared quantizer fallback')
            result = audit(candidate, plan, manifest, base, recipe)
            require(all(r['type'] in ('F32','F16','Q8_0') for r in result['tensors'].values())
                    if recipe == 'q8_0' else result['type_counts'].get('Q4_K',0)>0, 'candidate recipe differs')
            report['artifacts'][recipe] = result
            save(output, report)
        report['status'] = 'passed'
    except Exception as error:
        report['error_type'] = type(error).__name__
        report['error'] = str(error).replace(str(ROOT),'[repository]').replace(str(args.output),'[output]')
        raise
    finally:
        save(output, report)


if __name__ == '__main__':
    main()
