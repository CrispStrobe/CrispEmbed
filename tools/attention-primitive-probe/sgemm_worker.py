"""Private paired-F16 SGEMM operator verification, Kaggle CPU only."""
import json
import os
from pathlib import Path
import shutil
import time
from admission import digest
from runtime_guard import Deadline, command
from worker import require, validate_cache
HERE = Path(__file__).parent
ROOT = HERE.parents[1]


def main():
    require(Path('/kaggle/working').is_dir() and not list(Path('/dev').glob('nvidia[0-9]*')), 'Kaggle CPU only')
    import numpy as np
    require(np.__version__ == '2.3.5', 'numpy pin')
    cfg = json.loads((HERE / 'paired-v1.json').read_text())
    out = Path(os.environ['HEAVY_OUT'])
    work = Path(os.environ['HEAVY_SCRATCH']) / 'attention-sgemm'
    work.mkdir()
    started = time.monotonic()
    report = {'schema': 'lighton4b-attention-sgemm/v1', 'status': 'failed', 'model_forward_executed': False,
              'production_math_changed': False, 'quality_qualified': False, 'commands': {}, 'metrics': {}}
    def save():
        (out / 'attention-sgemm.json').write_text(json.dumps(report, indent=2) + '\n')
    def run(label, args):
        report['phase'] = label
        report['commands'][label] = {}
        save()
        try:
            command(args, out / (label + '.log'), min(600, 1200 - (time.monotonic() - started)), report['commands'][label])
        finally:
            save()
    deadline = Deadline(cfg['total_timeout_seconds'])
    deadline.__enter__()
    try:
        run('source-pin', ['git', '-C', ROOT, 'rev-parse', 'HEAD'])
        source = (out / 'source-pin.log').read_text().strip()
        require(source == os.environ['LIGHTON4B_EXPECTED_SOURCE'], 'exact source')
        report['source_sha'] = source
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit != 'max':
            available = min(available, int(limit) - int(Path('/sys/fs/cgroup/memory.current').read_text()))
        require(available >= 1 << 30 and shutil.disk_usage(work).free >= 2 << 30, 'resources')
        report['preflight'] = {'available_bytes': available, 'disk_free_bytes': shutil.disk_usage(work).free, 'load': os.getloadavg()}
        matches = [p for p in Path('/kaggle/input').rglob('attention-primitives.json') if p.is_file() and p.stat().st_size < 1 << 20 and digest(p) == cfg['frozen_report_sha256']]
        require(len(matches) == 1, 'one exact frozen attention V2 receipt')
        frozen = matches[0]
        old = json.loads(frozen.read_text())
        require(old['source_sha'] == cfg['frozen_source_sha'] and old['status'] == 'measured_operator_only' and old['actual_native_hidden_closed'], 'frozen actual trace admission')
        report['frozen_report_sha256'] = digest(frozen)
        for rel, row in cfg['files'].items():
            p = frozen.parent / rel
            require(p.stat().st_size == row['bytes'] and digest(p) == row['sha256'], 'frozen stage authentication: ' + rel)
        up = work / 'llama'
        for label, args in [('up-init', ['git', 'init', up]), ('up-remote', ['git', '-C', up, 'remote', 'add', 'origin', 'https://github.com/ggml-org/llama.cpp']),
                            ('up-fetch', ['git', '-C', up, 'fetch', '--depth', '1', 'origin', cfg['upstream_commit']]),
                            ('up-checkout', ['git', '-C', up, 'checkout', '--detach', 'FETCH_HEAD']),
                            ('native-submodule', ['git', '-C', ROOT, 'submodule', 'update', '--init', 'ggml']),
                            ('native-pin', ['git', '-C', ROOT / 'ggml', 'rev-parse', 'HEAD'])]:
            run(label, args)
        require((out / 'native-pin.log').read_text().strip() == cfg['native_ggml_commit'], 'native backend pin')
        for tag, ggml in [('upstream', up / 'ggml'), ('native', ROOT / 'ggml')]:
            build = work / (tag + '-build')
            run(tag + '-configure', ['cmake', '-S', HERE, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DGGML_SOURCE=' + str(ggml)])
            shutil.copy(build / 'CMakeCache.txt', out / (tag + '-CMakeCache.txt'))
            validate_cache(out / (tag + '-CMakeCache.txt'))
            run(tag + '-build', ['cmake', '--build', build, '--target', 'attention-paired-f16-sgemm', 'attention-primitive', '--parallel', '2'])
            for n in (4, 16):
                base = out / tag / str(n)
                gold = frozen.parent / f'upstream/{n}/f16/observed'
                for observation in ('composed', 'observed'):
                    run(f'{tag}-{n}-graph-{observation}', [build / 'attention-primitive', frozen.parent / 'native', base / 'graph' / observation, n, 'f16', observation])
                require((base / 'graph/composed/gated.f32').read_bytes() == (base / 'graph/observed/gated.f32').read_bytes(), 'source graph observer neutrality')
                if tag == 'upstream':
                    for stage in ('logits', 'probs', 'context', 'sigmoid', 'gated'):
                        require((base / 'graph/observed' / (stage + '.f32')).read_bytes() == (gold / (stage + '.f32')).read_bytes(), 'fresh actual-profile source stage drift: ' + stage)
                for arm in ('composed-pipeline', 'exact-sourceP'):
                    for observation in ('composed', 'observed'):
                        args = [build / 'attention-paired-f16-sgemm', frozen.parent / 'native', base / arm / observation, n, observation]
                        if arm == 'exact-sourceP':
                            args.append(gold / 'probs.f32')
                        run(f'{tag}-{n}-{arm}-{observation}', args)
                    require((base / arm / 'composed/gated.f32').read_bytes() == (base / arm / 'observed/gated.f32').read_bytes(), 'custom SGEMM observer neutrality')
                    for stage in ('logits', 'probs', 'context', 'sigmoid', 'gated'):
                        x = np.fromfile(base / arm / 'observed' / (stage + '.f32'), dtype='<f4')
                        y = np.fromfile(gold / (stage + '.f32'), dtype='<f4')
                        require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), 'finite same shape')
                        d = x.astype('f8') - y
                        report['metrics'][f'{tag}/{n}/{arm}/{stage}'] = {'bit_exact': x.tobytes() == y.tobytes(), 'rms': float(np.sqrt(np.mean(d*d))), 'max_abs': float(abs(d).max()), 'reference_max_abs': float(abs(y).max())}
        report['all_stages_bit_exact'] = all(row['bit_exact'] for row in report['metrics'].values())
        report['threadpool_contract'] = 'Same public four-thread CPU pool installed in backend and private custom SGEMM callbacks; no manual threads, vendor trait mutation or kernel fallback.'
        report['private_sgemm_supported'] = True
        report['status'] = 'measured_operator_only'
        report['limitations'] = 'Private SGEMM ABI/operator diagnostic, not a production path. Batch4/16 actual-native inputs only; no tokenwise/cache/full-model/decoded proof.'
    except BaseException as error:
        report['error_type'], report['error'] = type(error).__name__, str(error)
        raise
    finally:
        deadline.close()
        report['elapsed_seconds'] = time.monotonic() - started
        report['artifacts'] = {str(p.relative_to(out)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'attention-sgemm.json'}
        save()


if __name__ == '__main__':
    main()
