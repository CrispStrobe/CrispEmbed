"""Exact-source-P PV isolation and QK dispatch diagnostic, Kaggle CPU only."""
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
    work = Path(os.environ['HEAVY_SCRATCH']) / 'attention-dispatch'
    work.mkdir()
    started = time.monotonic()
    report = {'schema': 'lighton4b-attention-dispatch/v1', 'status': 'failed', 'model_forward_executed': False,
              'production_math_changed': False, 'quality_qualified': False, 'commands': {}, 'metrics': {}}
    def save():
        (out / 'attention-dispatch.json').write_text(json.dumps(report, indent=2) + '\n')
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
        # Close regenerated actual-profile source graph BEFORE interpreting
        # disabled-dispatch controls. The source context is anchored bytewise.
        for backend, ggml in [('upstream', up / 'ggml'), ('native', ROOT / 'ggml')]:
            for enabled in (True, False):
                tag = backend + ('-on' if enabled else '-off')
                build = work / (tag + '-build')
                run(tag + '-configure', ['cmake', '-S', HERE, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                                         '-DGGML_SOURCE=' + str(ggml), '-DATTENTION_DIAGNOSTIC_LLAMAFILE=' + ('ON' if enabled else 'OFF')])
                shutil.copy(build / 'CMakeCache.txt', out / (tag + '-CMakeCache.txt'))
                validate_cache(out / (tag + '-CMakeCache.txt'), llamafile=enabled)
                run(tag + '-build', ['cmake', '--build', build, '--target', 'attention-paired-f16', 'attention-primitive', '--parallel', '2'])
                run(tag + '-symbol-controls', [build / 'attention-paired-f16', '--selftest'])
                for n in (4, 16):
                    gold = frozen.parent / f'upstream/{n}/f16/observed'
                    base = out / tag / str(n)
                    for mode in ('composed', 'observed'):
                        run(f'{tag}-{n}-graph-{mode}', [build / 'attention-primitive', frozen.parent / 'native', base / mode, n, 'f16', mode])
                    require((base / 'composed/gated.f32').read_bytes() == (base / 'observed/gated.f32').read_bytes(), 'observer arithmetic neutrality')
                    if backend == 'upstream' and enabled:
                        for stage in ('logits', 'probs', 'context', 'sigmoid', 'gated'):
                            require((base / 'observed' / (stage + '.f32')).read_bytes() == (gold / (stage + '.f32')).read_bytes(), 'fresh actual-profile upstream stage drift: ' + stage)
                    for arm in ('paired', 'paired-sourceP'):
                        args = [build / 'attention-paired-f16', frozen.parent / 'native', base / arm, n]
                        if arm == 'paired-sourceP':
                            args.append(gold / 'probs.f32')
                        run(f'{tag}-{n}-{arm}', args)
                    for stage in ('logits', 'probs', 'context', 'sigmoid', 'gated'):
                        for arm in ('observed', 'paired', 'paired-sourceP'):
                            filename = stage + ('.sourceP' if arm == 'paired-sourceP' and stage in ('context', 'gated') else '') + '.f32'
                            x = np.fromfile(base / arm / filename, dtype='<f4')
                            y = np.fromfile(gold / (stage + '.f32'), dtype='<f4')
                            require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), 'finite same shape')
                            d = x.astype('f8') - y
                            report['metrics'][f'{tag}/{n}/{arm}/{stage}'] = {'bit_exact': x.tobytes() == y.tobytes(), 'rms': float(np.sqrt(np.mean(d*d))), 'max_abs': float(abs(d).max()), 'reference_max_abs': float(abs(y).max())}
                    # For KQ compare actual dispatch and direct dot on the same
                    # profile, avoiding any downstream softmax/PV inference.
                    x = np.fromfile(base / 'paired/logits.f32', dtype='<f4')
                    y = np.fromfile(base / 'observed/logits.f32', dtype='<f4')
                    d = x.astype('f8') - y
                    report['metrics'][f'{tag}/{n}/paired_vs_dispatch/rawQK'] = {'bit_exact': x.tobytes() == y.tobytes(), 'rms': float(np.sqrt(np.mean(d*d))), 'max_abs': float(abs(d).max()), 'reference_max_abs': float(abs(y).max())}
        report['exact_source_probability_pv_isolated'] = True
        report['fresh_upstream_on_all_source_stages_byte_exact'] = True
        report['dispatch_profiles'] = {'ON': 'actual source profile', 'OFF': 'diagnostic only; not actual source graph or quality setting'}
        report['status'] = 'measured_operator_only'
        report['limitations'] = 'Private backend ABI/operator diagnostic; OFF profile is not actual source. Batch4/16 actual-native inputs only, no tokenwise/cache/full-model/decoded proof.'
    except BaseException as error:
        report['error_type'], report['error'] = type(error).__name__, str(error)
        raise
    finally:
        deadline.close()
        report['elapsed_seconds'] = time.monotonic() - started
        report['artifacts'] = {str(p.relative_to(out)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'attention-dispatch.json'}
        save()


if __name__ == '__main__':
    main()
