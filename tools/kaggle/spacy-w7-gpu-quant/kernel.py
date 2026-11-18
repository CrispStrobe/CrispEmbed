"""Real CUDA quantization/inference only; no CPU/native build or held-out replay."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import urllib.request

VERSION = 'spacy-w7-gpu-quant-v1'
SOURCE = '__SHA__'
HARNESS = '3b3dae4817b4ef0dded9dde8f4713178833de8d1'
ORACLE = 'dcf2ec9b4d6fba05fa5eec2634c6654266332cc9'
MODEL_SHA = 'fae2e6ab8d6c611d7a9dddd4b3259a871a1d2f95cacc8259f49c1e4ff31ff6cd'
HF = 'https://huggingface.co/cstr/crispembed-spacy-de-core-news-sm-3.8.0/resolve/'
CPU_CANDIDATES = {
    'f16': ('d17727c4e6afe6ee9e475e7a3a57e442ffbe07d2a8be731d3af68e3c60578351', 3699456),
    'q8_0': ('31cc15a311621e53357ffd21b89ec0adddfc90fbdbfffbbe1e32bd9bdf842280', 1980480),
    'q4_0': ('aab70fdb8d035bb418275d7c15d1d0ce6476107f504e5398d60f747fa1be37f8', 1676608),
}
OUT = Path('/kaggle/working')
SCRATCH = Path('/kaggle/temp/spacy-w7-gpu-quant-v1')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def download(url, path, sha, size=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Cache-Control': 'no-cache'}), timeout=180) as response:
        assert response.status == 200
        data = response.read(32 * 1024**2 + 1)
    assert len(data) <= 32 * 1024**2 and digest(data) == sha
    assert size is None or len(data) == size
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


class GGUF:
    def __init__(self, data):
        self.data, self.pos = data, 0
        assert self.read('<I') == 0x46554747 and self.read('<I') == 3
        count, keys = self.read('<Q'), self.read('<Q')
        assert count == 39 and keys in (13, 15)
        self.meta = {}
        for _ in range(keys):
            name, kind = self.string(), self.read('<I')
            assert name not in self.meta
            value = self.string() if kind == 8 else self.read('<I') if kind == 4 else self.read('<f') if kind == 6 else None
            assert value is not None
            self.meta[name] = (kind, value)
        assert self.meta['general.architecture'][1] == 'spacy_de_core_news_sm'
        self.info = {}
        for _ in range(count):
            name, rank = self.string(), self.read('<I')
            assert 1 <= rank <= 4 and name not in self.info
            ne = [self.read('<Q') for _ in range(rank)]
            kind, offset = self.read('<I'), self.read('<Q')
            assert kind in (0, 1, 2, 8)
            elements = 1
            for n in ne:
                assert 0 < n <= 5000
                elements *= n
            block, size = {0: (1, 4), 1: (1, 2), 2: (32, 18), 8: (32, 34)}[kind]
            assert ne[0] % block == 0
            self.info[name] = {'shape': tuple(reversed(ne)), 'kind': kind,
                               'offset': offset, 'bytes': elements // block * size}
        base = (self.pos + 31) // 32 * 32
        for name, item in self.info.items():
            start = base + item['offset']
            item['raw'] = data[start:start + item['bytes']]
            assert len(item['raw']) == item['bytes']

    def read(self, fmt):
        value = struct.unpack_from(fmt, self.data, self.pos)[0]
        self.pos += struct.calcsize(fmt)
        return value

    def string(self):
        size = self.read('<Q')
        assert size <= 4096
        result = self.data[self.pos:self.pos + size].decode()
        self.pos += size
        return result


def gguf_bytes(original, recipe, payloads):
    def string(text):
        value = text.encode()
        return struct.pack('<Q', len(value)) + value
    meta = dict(original.meta)
    meta['spacy.precision'] = (8, recipe)
    meta['general.quantization_version'] = (4, 2)
    meta['general.file_type'] = (4, {'f16': 1, 'q8_0': 7, 'q4_0': 2}[recipe])
    header = struct.pack('<IIQQ', 0x46554747, 3, 39, len(meta))
    for name, (kind, value) in meta.items():
        header += string(name) + struct.pack('<I', kind)
        header += string(value) if kind == 8 else struct.pack('<I', value) if kind == 4 else struct.pack('<f', value)
    body = bytearray()
    for name, item in original.info.items():
        shape = item['shape']
        while len(shape) > 1 and shape[0] == 1:
            shape = shape[1:]  # Same canonical unit-axis omission as gguf's writer.
        kind, raw = payloads[name]
        header += string(name) + struct.pack('<I', len(shape))
        header += b''.join(struct.pack('<Q', n) for n in reversed(shape))
        header += struct.pack('<IQ', kind, len(body))
        body += raw + bytes(-len(raw) % 32)
    return header + bytes(-len(header) % 32) + body


def main():
    OUT.mkdir(exist_ok=True)
    report = {'schema': 'crispembed.spacy.w7.kaggle-cuda-quant.v1', 'version': VERSION,
        'source': SOURCE, 'script_sha256': digest(Path(__file__).read_bytes()),
        'execution_complete': False, 'conclusive': False, 'gpu_model_started': False,
        'native_GPU_or_raw_text_qualified': False, 'qualified_native_profiles': [],
        'F32_CPU_baseline_replaced': False, 'CPU_build_or_forward': False,
        'consumed_native_W6_W7_heldout_replayed': False, 'speed_claim': False, 'profiles': []}

    def save():
        (OUT / 'quantization-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    try:
        # Gate actual hardware BEFORE downloads/importing a model. Never fish
        # for a different GPU by repushing a short hardware-rejected session.
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap',
                                      '--format=csv,noheader'], text=True)
        report['nvidia_smi'] = gpu
        cap = float(gpu.splitlines()[0].split(',')[-1].strip())
        if cap < 7.0:
            report['blocker'] = 'P100/sm60 unsupported by preinstalled Torch; GPU model/quantization UNRUN. No auto retry.'
            save()
            print('P100_UNSUPPORTED', flush=True)
            return
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import numpy as np
        import torch
        assert torch.cuda.is_available()
        assert torch.cuda.get_device_capability(0)[0] >= 7
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True)
        assert (torch.ones((4, 4), device='cuda') @ torch.eye(4, device='cuda')).is_cuda
        torch.cuda.synchronize()
        report['torch'] = {'version': torch.__version__, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0),
            'compiled_architectures': torch.cuda.get_arch_list(), 'TF32': False, 'device_used': 0}
        SCRATCH.mkdir(parents=True)
        # Required shared progress helper, exact source; no caches, secrets
        # datasets, W&B or CPU/native builds are attached or executed.
        harness = SCRATCH / 'CrispASR'
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', '--depth', '1',
                        'https://github.com/CrispStrobe/CrispASR.git', str(harness)], check=True)
        subprocess.run(['git', 'fetch', '--depth', '1', 'origin', HARNESS], cwd=harness, check=True)
        subprocess.run(['git', 'sparse-checkout', 'set', '--no-cone', '/tools/kaggle/kaggle_harness.py'], cwd=harness, check=True)
        subprocess.run(['git', 'checkout', '--detach', HARNESS], cwd=harness, check=True)
        sys.path.insert(0, str(harness / 'tools/kaggle'))
        import kaggle_harness as kh
        kh.init_progress(OUT / 'progress.jsonl')
        report['harness_source'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=harness, text=True).strip()
        comparator = SCRATCH / 'spacy_parity.py'
        download(f'https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/{SOURCE}/tests/spacy_parity.py', comparator,
                 'e83495dadcf199e068259e1cca30e65256bf76263a405bc6299765779fd77e32')
        spec = importlib.util.spec_from_file_location('unchanged_spacy_comparator', comparator)
        numeric = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(numeric)
        assert numeric.ATOL == 2e-4 and numeric.RTOL == 2e-5
        model_url = HF + '659dae33997d00296ae12b76510fa4dfe26540e0/bundle/model.gguf'
        model = GGUF(download(model_url, SCRATCH / 'model.gguf', MODEL_SHA, 7383104))
        report['model'] = {'url': model_url, 'bytes': 7383104, 'sha256': MODEL_SHA, 'fresh_empty_cache': True}
        for name, sha, size in (
            ('LICENSE', '3933c176979b68bc6d0bcc902c7d6c130f1d127f476f17ba5cdba8d99cfd0012', 1056),
            ('LICENSES_SOURCES', '0c3ab78745569d848b7c83a486183bce42b9643561084d6d6f67eb0229c79869', 13470),
        ):
            download(HF + '659dae33997d00296ae12b76510fa4dfe26540e0/bundle/' + name,
                     OUT / 'notices' / name, sha, size)
        reference_base = f'https://raw.githubusercontent.com/CrispStrobe/quds_office_suite/{ORACLE}/tools/proofing/spacy/fixtures/development/'
        manifest = json.loads(download(reference_base + 'manifest.json', SCRATCH / 'reference-manifest.json',
            '9a3bb6bae6114adcf579840834ddcb64dea4e0aaa8361718dc7a551d663c8f7c'))
        assert manifest['split'] == 'development' and len(manifest['cases']) == 16
        assert manifest['token_count'] == 113 and manifest['parser_step_count'] == 244 and manifest['trace_stage_count'] == 1201
        cases = []
        for item in manifest['cases']:
            files = {}
            for pin in item['files']:
                files[pin['path'].split('/')[-1]] = download(reference_base + pin['path'],
                    SCRATCH / 'reference' / pin['path'], pin['sha256'], pin['size_bytes'])
            cases.append((item, json.loads(files['case.json']), files['traces.f32']))
        original = {name: torch.from_numpy(np.frombuffer(item['raw'], dtype='<f4').copy().reshape(item['shape'])).to('cuda')
                    for name, item in model.info.items()}
        assert len(original) == 39 and all(value.is_cuda for value in original.values())

        def quantize(recipe):
            weights, payloads = {}, {}
            for name, value in original.items():
                item = model.info[name]
                kind = 0
                if value.ndim >= 2 and recipe == 'f16':
                    kind = 1
                elif value.ndim >= 2 and value.shape[-1] % 32 == 0 and recipe != 'f32':
                    kind = 8 if recipe == 'q8_0' or name.startswith('embed.') else 2
                if kind == 0:
                    expanded, raw = value, item['raw']
                elif kind == 1:
                    half = value.half()
                    expanded, raw = half.float(), half.cpu().numpy().astype('<f2').tobytes()
                else:
                    blocks = value.reshape(-1, 32)
                    maximum, at = blocks.abs().max(dim=1)
                    if kind == 8:
                        scale = maximum / 127.0
                        inverse = torch.where(scale != 0, 1.0 / scale, torch.zeros_like(scale))
                        y = blocks * inverse[:, None]
                        q = (torch.sign(y) * torch.floor(y.abs() + 0.5)).clamp(-127, 127).to(torch.int8)
                        half = scale.half()
                        expanded = (q.float() * half.float()[:, None]).reshape(value.shape)
                        record = np.empty(len(q), dtype=[('d', '<f2'), ('qs', 'i1', (32,))])
                        record['d'], record['qs'] = half.cpu().numpy(), q.cpu().numpy()
                    else:
                        signed = blocks.gather(1, at[:, None]).squeeze(1)
                        signed = torch.where(maximum == 0, torch.zeros_like(signed), signed)
                        scale = signed / -8.0
                        inverse = torch.where(scale != 0, 1.0 / scale, torch.zeros_like(scale))
                        q = torch.floor(blocks * inverse[:, None] + 8.5).clamp(0, 15).to(torch.uint8)
                        half = scale.half()
                        expanded = ((q.float() - 8.0) * half.float()[:, None]).reshape(value.shape)
                        packed = q[:, :16] | (q[:, 16:] << 4)
                        record = np.empty(len(q), dtype=[('d', '<f2'), ('qs', 'u1', (16,))])
                        record['d'], record['qs'] = half.cpu().numpy(), packed.cpu().numpy()
                    raw = record.tobytes()
                assert expanded.is_cuda and bool(torch.isfinite(expanded).all())
                weights[name], payloads[name] = expanded, (kind, raw)
            return weights, payloads

        def hash_keys(values, seed):
            def signed(value):
                return value if value < 2**63 else value - 2**64
            def right(value, bits):
                return (value >> bits) & (2**(64 - bits) - 1)
            def mix(value):
                value = (value ^ right(value, 33)) * signed(0xff51afd7ed558ccd)
                value = (value ^ right(value, 33)) * signed(0xc4ceb9fe1a85ec53)
                return value ^ right(value, 33)
            a = values * signed(0x87c37b91114253d5)
            a = ((a << 31) | right(a, 33)) * signed(0x4cf5ad432745937f)
            a = a ^ seed ^ 8
            b = torch.full_like(a, seed ^ 8)
            a = a + b
            b = b + a
            a, b = mix(a), mix(b)
            a = a + b
            b = b + a
            return torch.stack((a & 0xffffffff, right(a, 32), b & 0xffffffff, right(b, 32)), dim=1)
            a = values * signed(0x87c37b91114253d5)
            a = ((a << 31) | right(a, 33)) * signed(0x4cf5ad432745937f)
            a = a ^ seed ^ 8
            b = torch.full_like(a, seed ^ 8)
            a = a + b
            b = b + a
            a, b = mix(a), mix(b)
            a = a + b
            b = b + a
            return torch.stack((a & 0xffffffff, right(a, 32), b & 0xffffffff, right(b, 32)), dim=1)

        def forward(weights, case):
            trace = {}
            def emit(name, value):
                assert value.is_cuda and bool(torch.isfinite(value).all())
                trace[name] = value.detach().cpu().numpy().copy()
            def affine(x, prefix, bias=True):
                w = weights[prefix + '.W'].reshape(-1, x.shape[1])
                result = x @ w.T
                return result + weights[prefix + '.b'].reshape(1, -1) if bias else result
            def maxnorm(x, prefix, norm_prefix, label):
                projected = affine(x, prefix).reshape(len(x), 96, 3)
                emit(label + '.affine', projected)
                maximum = projected.max(dim=2).values
                emit(label + '.maxout', maximum)
                centered = maximum - maximum.mean(dim=1, keepdim=True)
                inverse = 1.0 / torch.sqrt((centered * centered).mean(dim=1, keepdim=True) + 1e-8)
                result = centered * inverse * weights[norm_prefix + '.G'] + weights[norm_prefix + '.b']
                emit(label + '.norm', result)
                return result
            features = case['features']['uint64_decimal']
            n = len(features)
            if not n:
                for name, width in [('tok2vec.output', 96), ('tagger.scores', 55), ('morphologizer.scores', 671), ('lemmatizer.scores', 1311)]:
                    emit(name, torch.empty((0, width), device='cuda'))
                return trace, [], True
            signed_features = [[int(x) if int(x) < 2**63 else int(x) - 2**64 for x in row] for row in features]
            ids = torch.tensor(signed_features, dtype=torch.int64, device='cuda')
            embeddings, buckets = [], []
            for feature, rows in enumerate((5000, 1000, 2500, 2500, 50, 50)):
                keys = hash_keys(ids[:, feature], 8 + feature) % rows
                buckets.append(keys)
                value = torch.zeros((n, 96), device='cuda')
                for slot in range(4):
                    value = value + weights[f'embed.{feature}.E'][keys[:, slot]]
                embeddings.append(value)
            own_buckets = torch.stack(buckets, dim=1).cpu().tolist()
            assert own_buckets == case['features']['buckets'], 'CUDA uint64 hash/bucket divergence'
            embedded = torch.cat(embeddings, dim=1)
            emit('tok2vec.embed.concat', embedded)
            mixed = maxnorm(embedded, 'embed.mix', 'embed.norm', 'tok2vec.embed')
            encoded = torch.cat((torch.zeros((4, 96), device='cuda'), mixed, torch.zeros((4, 96), device='cuda')))
            emit('tok2vec.encoder.input', encoded)
            for layer in range(4):
                prefix = f'tok2vec.encoder.layer.{layer}'
                window = torch.cat((torch.cat((torch.zeros((1, 96), device='cuda'), encoded[:-1])), encoded,
                                    torch.cat((encoded[1:], torch.zeros((1, 96), device='cuda')))), dim=1)
                emit(prefix + '.window', window)
                encoded = encoded + maxnorm(window, f'encoder.{layer}.maxout', f'encoder.{layer}.norm', prefix)
                emit(prefix + '.residual', encoded)
            output = encoded[4:-4]
            emit('tok2vec.output', output)
            for prefix in ('tagger', 'morphologizer', 'lemmatizer'):
                emit(prefix + '.scores', affine(output, prefix))
            reduced = affine(output, 'parser.reduce')
            emit('parser.reduce', reduced)
            lower = affine(reduced, 'parser.lower', bias=False).reshape(n, 8, 64, 2)
            cache = torch.cat((weights['parser.lower.pad'].reshape(1, 8, 64, 2), lower))
            emit('parser.lower_cache', cache)
            choices = []
            for step in case['parser_steps']:
                summed = torch.zeros((1, 64, 2), device='cuda')
                for feature, token in enumerate(step['feature_token_ids']):
                    summed = summed + cache[token + 1, feature].unsqueeze(0)
                summed = summed + weights['parser.lower.b']
                label = f"parser.step.{step['step']}"
                emit(label + '.sum', summed)
                hidden = summed.max(dim=2).values
                emit(label + '.hidden', hidden)
                scores = affine(hidden, 'parser.upper')
                emit(label + '.scores', scores)
                legal = torch.tensor(step['valid_actions'], device='cuda', dtype=torch.bool)
                choices.append(int(scores.masked_fill(~legal[None], -float('inf')).argmax(dim=1).item()))
            return trace, choices, True

        report['gpu_model_started'] = True
        for recipe in ('f32', 'f16', 'q8_0', 'q4_0'):
            print('CUDA_PROFILE', recipe, flush=True)
            started = time.time()
            weights, payloads = quantize(recipe)
            result = {'recipe': recipe, 'GPU_math': True, 'cases': [], 'native_profile_qualified': False}
            if recipe != 'f32':
                candidate = gguf_bytes(model, recipe, payloads)
                (OUT / (recipe + '.gguf')).write_bytes(candidate)
                sha, size = CPU_CANDIDATES[recipe]
                reference = GGUF(download(HF + '8ee4b35366003654a043c80f616d6ba77fd4e746/quantization-attempts/868dc0587435de053f930a1fcafb184f008078c5/' + recipe + '.gguf',
                    SCRATCH / ('CPU-' + recipe + '.gguf'), sha, size))
                comparisons = [{'tensor': name, 'type_equal': payloads[name][0] == reference.info[name]['kind'],
                    'raw_bytes_equal': payloads[name][1] == reference.info[name]['raw']} for name in payloads]
                result['GPU_produced_GGUF'] = {'sha256': digest(candidate), 'bytes': len(candidate),
                    'byte_identical_to_already_rejected_CPU_candidate': digest(candidate) == sha,
                    'CPU_candidate_sha256': sha, 'tensor_byte_comparisons': comparisons,
                    'all_tensor_types_and_bytes_equal': all(x['type_equal'] and x['raw_bytes_equal'] for x in comparisons)}
            for item, case, reference_payload in cases:
                actual, choices, hashed = forward(weights, case)
                assert set(actual) == set(case['traces'])
                stages, heads = [], True
                raw = bytearray()
                trace_index = {}
                for name, entry in sorted(case['traces'].items(), key=lambda pair: pair[1]['offset_bytes']):
                    expected = np.frombuffer(reference_payload, dtype='<f4', count=entry['size_bytes'] // 4,
                        offset=entry['offset_bytes']).reshape(entry['shape'])
                    stages.append(numeric.compare_stage(name, expected, actual[name]))
                    if name in ('tagger.scores', 'morphologizer.scores', 'lemmatizer.scores') and item['tokens']:
                        heads = heads and np.array_equal(actual[name].argmax(axis=1), expected.argmax(axis=1))
                    value = actual[name].astype('<f4').tobytes()
                    trace_index[name] = {'offset_bytes': len(raw), 'size_bytes': len(value), 'shape': list(actual[name].shape), 'sha256': digest(value)}
                    raw += value
                trace_dir = OUT / 'traces' / recipe
                trace_dir.mkdir(parents=True, exist_ok=True)
                (trace_dir / (item['id'] + '.f32')).write_bytes(raw)
                (trace_dir / (item['id'] + '.json')).write_text(json.dumps(trace_index, indent=2) + '\n')
                actions = choices == [step['chosen_action'] for step in case['parser_steps']]
                result['cases'].append({'id': item['id'], 'tokens': item['tokens'], 'parser_states': item['parser_steps'],
                    'CUDA_hash_buckets_exact': hashed, 'classifier_decisions_exact': bool(heads), 'legal_choices_on_reference_states_exact': actions,
                    'trace_payload_sha256': digest(raw), 'stages': stages,
                    'passed': bool(heads) and actions and all(x['passed'] for x in stages)})
            torch.cuda.synchronize()
            counts = {'cases': len(result['cases']), 'tokens': sum(x['tokens'] for x in result['cases']),
                'parser_states': sum(x['parser_states'] for x in result['cases']),
                'stages': sum(len(x['stages']) for x in result['cases']),
                'float_elements': sum(s['elements'] for x in result['cases'] for s in x['stages'])}
            assert counts == {'cases': 16, 'tokens': 113, 'parser_states': 244, 'stages': 1201, 'float_elements': 1406553}
            result.update(counts=counts, development_prepared_gate_passed=all(x['passed'] for x in result['cases']),
                elapsed_with_trace_copy_seconds=time.time() - started, CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            (OUT / (recipe + '-prepared.json')).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            report['profiles'].append({key: value for key, value in result.items() if key != 'cases'})
            save()
        report.update(execution_complete=True, conclusive=True,
            limits='Actual CUDA quantization/hash/complete prepared graph. No native GPU text backend, raw-text CUDA qualification, held-out replay, accepted quant profile, speed or app/grammar promotion.')
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1500]}
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
