"""Kaggle-only bounded captured-input attention diagnostics. Never model inference."""
import hashlib
import json
import os
import resource
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
from admission import load, digest, admit_capture_files
from runtime_guard import Deadline, command

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate_cache(path, runtime=False, llamafile=True):
    flags = {'GGML_LLAMAFILE': 'ON' if llamafile else 'OFF', 'GGML_AVX2': 'ON', 'GGML_NATIVE': 'OFF',
             'GGML_AVX512': 'OFF', 'GGML_BLAS': 'OFF'}
    if runtime:
        flags['CRISPEMBED_NATIVE'] = 'OFF'
    contents = Path(path).read_text().splitlines()
    for name, value in flags.items():
        require(f'{name}:BOOL={value}' in contents, 'actual CMake profile drift: ' + name)


def main():
    require(Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir(), 'Kaggle only')
    require(not list(Path('/dev').glob('nvidia[0-9]*')), 'CPU only')
    import numpy as np
    require(np.__version__ == '2.3.5', 'numpy pin')
    from scalar import run as scalar_run
    cfg = load(HERE / 'protocol-v1.json')
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'attention-primitives'
    scratch.mkdir()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {'schema': cfg['schema'], 'status': 'failed', 'actual_source_hidden_captured': False,
              'actual_native_hidden_closed': False, 'model_forward_executed': False,
              'quality_qualified': False, 'protocol_sha256': digest(HERE / 'protocol-v1.json')}

    report['commands'] = {}
    deadline = Deadline(cfg['total_timeout_seconds'])
    deadline.__enter__()

    def checkpoint():
        path = out / 'attention-primitives.json'
        temporary = out / 'attention-primitives.pending'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(path)

    def run(label, args):
        report['phase'] = label
        checkpoint()
        remaining = cfg['total_timeout_seconds'] - (time.monotonic() - started)
        require(remaining > 0, 'total deadline')
        receipt = report['commands'][label] = {}
        try:
            command(args, out / (label + '.log'), min(remaining, 900), receipt)
        finally:
            checkpoint()

    def diff(x, y):
        require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), 'finite same shape')
        delta = x.astype('f8') - y.astype('f8')
        return {'bit_exact': x.tobytes() == y.tobytes(), 'rms': float(np.sqrt(np.mean(delta * delta))),
                'max_abs': float(abs(delta).max()), 'reference_max_abs': float(abs(y).max())}

    try:
        run('source-pin', ['git', '-C', ROOT, 'rev-parse', 'HEAD'])
        source = (out / 'source-pin.log').read_text().strip()
        require(source == os.environ['LIGHTON4B_EXPECTED_SOURCE'], 'source pin')
        report['source_sha'] = source
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit != 'max':
            available = min(available, int(limit) - int(Path('/sys/fs/cgroup/memory.current').read_text()))
        report['preflight'] = {'available_bytes': available, 'disk_free_bytes': shutil.disk_usage(scratch).free,
                               'load': os.getloadavg(), 'cpuinfo': Path('/proc/cpuinfo').read_text()}
        require(available >= cfg['minimum_memory_bytes'] and shutil.disk_usage(scratch).free >= cfg['minimum_disk_bytes'], 'resource admission')
        binding = Path(os.environ['LIGHTON4B_PHYSICAL_INPUT'])
        case = binding / 'simple_form'
        require(digest(binding / 'physical-decode.json') == cfg['physical_report_sha256'], 'actual report binding')
        admit_capture_files(cfg, case)
        weights = scratch / 'weights'
        weights.mkdir()
        for row in cfg['tensors']:
            start, size = row['source_offset'], row['bytes']
            request = urllib.request.Request(cfg['source_url'], headers={'Range': f'bytes={start}-{start+size-1}'})
            path = weights / (row['physical_name'] + '.bin')
            with urllib.request.urlopen(request, timeout=120) as response, path.open('xb') as f:
                require(response.status == 206 and response.headers.get('Content-Range') == f"bytes {start}-{start+size-1}/{cfg['source_bytes']}", 'exact bounded range')
                count = 0
                while block := response.read(1 << 20):
                    count += len(block)
                    require(count <= size, 'range exceeded')
                    f.write(block)
            require(path.stat().st_size == size and digest(path) == row['source_packed_sha256'], 'whole source tensor authentication')
        up = scratch / 'llama'
        for label, args in [('up-init', ['git', 'init', up]), ('up-remote', ['git', '-C', up, 'remote', 'add', 'origin', 'https://github.com/ggml-org/llama.cpp']),
                            ('up-fetch', ['git', '-C', up, 'fetch', '--depth', '1', 'origin', cfg['upstream_commit']]),
                            ('up-checkout', ['git', '-C', up, 'checkout', '--detach', 'FETCH_HEAD']),
                            ('native-submodule', ['git', '-C', ROOT, 'submodule', 'update', '--init', 'ggml'])]:
            run(label, args)
        run('ggml-pin', ['git', '-C', ROOT / 'ggml', 'rev-parse', 'HEAD'])
        require((out / 'ggml-pin.log').read_text().strip() == cfg['native_ggml_commit'], 'native ggml pin')
        for name, expected in cfg['source_function_sha256'].items():
            require(digest(up / name) == expected, 'source function authentication')
        build = scratch / 'runtime-build'
        run('runtime-configure', ['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF', '-DCRISPEMBED_NATIVE=OFF', '-DGGML_AVX2=ON', '-DGGML_AVX512=OFF', '-DGGML_BLAS=OFF', '-DGGML_LLAMAFILE=ON', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF'])
        shutil.copy(build / 'CMakeCache.txt', out / 'runtime-CMakeCache.txt')
        validate_cache(out / 'runtime-CMakeCache.txt', runtime=True)
        run('runtime-build', ['cmake', '--build', build, '--target', 'lighton4b-attention-native-probe', '--parallel', '2'])
        exported = out / 'native'
        run('actual-native-closure', [build / 'lighton4b-attention-native-probe', weights, case, exported])
        report['actual_native_hidden_closed'] = True
        require(scalar_run(exported, out / 'scalar'), 'scalar actual closure')
        binaries = {}
        for tag, ggml in [('native', ROOT / 'ggml'), ('upstream', up / 'ggml')]:
            b = scratch / (tag + '-graph-build')
            run(tag + '-configure', ['cmake', '-S', HERE, '-B', b, '-DCMAKE_BUILD_TYPE=Release', '-DGGML_SOURCE=' + str(ggml)])
            run(tag + '-build', ['cmake', '--build', b, '--target', 'attention-primitive', '--parallel', '2'])
            binaries[tag] = b / 'attention-primitive'
            shutil.copy(b / 'CMakeCache.txt', out / (tag + '-CMakeCache.txt'))
            validate_cache(out / (tag + '-CMakeCache.txt'))
        report['comparisons'] = {}
        for n in (4, 16):
            for kind in ('f16', 'f32'):
                for tag, binary in binaries.items():
                    dirs = {}
                    for observation in ('composed', 'observed'):
                        dest = out / tag / str(n) / kind / observation
                        run(f'{tag}-{n}-{kind}-{observation}', [binary, exported, dest, n, kind, observation])
                        dirs[observation] = dest
                    require((dirs['observed'] / 'gated.f32').read_bytes() == (dirs['composed'] / 'gated.f32').read_bytes(), 'observer arithmetic neutrality')
                for stage in ('logits', 'probs', 'context', 'sigmoid', 'gated'):
                    x = np.fromfile(out / 'native' / str(n) / kind / 'observed' / (stage + '.f32'), dtype='<f4')
                    y = np.fromfile(out / 'upstream' / str(n) / kind / 'observed' / (stage + '.f32'), dtype='<f4')
                    report['comparisons'][f'{n}/{kind}/{stage}/backend'] = diff(x, y)
                    if stage != 'sigmoid':
                        arms = ('kv_f16', 'kvq_f16', 'kvqp_f16') if kind == 'f16' else ('f32',)
                        for arm in arms:
                            scalar = np.fromfile(out / 'scalar' / str(n) / arm / (stage + '.f32'), dtype='<f4')
                            report['comparisons'][f'{n}/{kind}/{stage}/scalar/{arm}'] = diff(scalar, y)
        report['status'] = 'measured_operator_only'
        report['tokenwise_cache_replay_complete'] = False
        report['limitation'] = 'Batch4/16 only; tokenwise cache replay remains a separate prerequisite for any cache-semantics attribution.'
    except BaseException as error:
        report['error_type'], report['error'] = type(error).__name__, str(error)
        raise
    finally:
        deadline.close()
        report['elapsed_seconds'] = time.monotonic() - started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report['worker_peak_rss_bytes'] = usage.ru_maxrss * 1024
        report['worker_user_seconds'], report['worker_system_seconds'] = usage.ru_utime, usage.ru_stime
        report['artifacts'] = {str(p.relative_to(out)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                               for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'attention-primitives.json'}
        (out / 'attention-primitives.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
