#!/usr/bin/env python3
"""Genuine hosted model-free CPU operator runner; no original/model captures."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/moedern-router-topk-v1'))
sys.path.insert(0, str(ROOT / 'tools'))
from common import PROFILE, identity, read, require, sha, verify, write
from ci_process import run

CACHE = {'CMAKE_BUILD_TYPE': 'Release', 'BUILD_SHARED_LIBS': 'OFF', 'GGML_NATIVE': 'OFF',
         'GGML_LLAMAFILE': 'ON', 'GGML_BACKEND_DL': 'OFF', 'GGML_CUDA': 'OFF', 'GGML_BLAS': 'OFF',
         'GGML_METAL': 'OFF', 'GGML_VULKAN': 'OFF', 'GGML_OPENCL': 'OFF'}


def provider_admission(provider, source):
    require(platform.system() == 'Linux' and platform.machine() == 'x86_64' and hasattr(os, 'wait4'), 'Linux x64 CPU host required')
    if provider == 'github-actions':
        require(os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('RUNNER_ENVIRONMENT') == 'github-hosted'
                and os.environ.get('GITHUB_SHA') == source and os.environ.get('GH_TOKEN'), 'Genuine GHA environment required')
        req = urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/runs/' + os.environ['GITHUB_RUN_ID'],
                                     headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN'], 'Accept': 'application/vnd.github+json'})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read(2 * 2**20 + 1)
        except Exception:
            raise ValueError('GHA service admission failed') from None
        require(len(data) <= 2 * 2**20, 'Service metadata bound')
        record = json.loads(data)
        require(record['head_sha'] == source and record['status'] == 'in_progress'
                and record['id'] == int(os.environ['GITHUB_RUN_ID']) and record['event'] == 'workflow_dispatch'
                and record['path'] in ['.github/workflows/moedern-router-topk-operator-v1.yml', '.github/workflows/ovisocr2-metadata.yml'],
                'Actual run/source/workflow differs')
        return dict(provider=provider, source=source, run=record['id'], attempt=record['run_attempt'],
                    workflow=record['path'], service_verified=True)
    require(provider == 'kaggle-cpu' and not os.environ.get('GITHUB_ACTIONS')
            and Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir()
            and os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Batch', 'Genuine Kaggle batch environment required')
    return dict(provider=provider, source=source, run_type=os.environ['KAGGLE_KERNEL_RUN_TYPE'],
                service_verified=False, external_exact_session_get_status_list_script_required=True,
                admission='Observed Kaggle batch environment, not a remotely authenticated session identity')


def build_profile(cache, commands):
    for key, value in CACHE.items():
        require(re.search(r'^' + key + r':[^=]+=' + value + '$', cache, re.M), 'Actual CMake profile differs: ' + key)
    selected = [r for r in commands if Path(r['file']).name in ['probe.cpp', 'diff.cpp']]
    require(len(selected) == 2 and all('-ffp-contract=off' in r['command']
            and not any(x in r['command'] for x in ['-ffast-math', '-Ofast']) for r in selected), 'Operator/Ref arithmetic flags differ')
    return dict(cache=CACHE, commands=selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviewed-source', required=True)
    p.add_argument('--provider', choices=['github-actions', 'kaggle-cpu'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    a = p.parse_args()
    review, raw = a.out / 'review', a.out / 'raw'
    review.mkdir(parents=True, exist_ok=True)
    raw.mkdir(exist_ok=True)
    report = dict(schema='crispembed-moedern-router-topk-hosted-v1', source=a.reviewed_source, completed=False,
                  model_access=False, production_qualified=False, commands={}, phase='source')
    def save():
        write(review / 'hosted.json', report)
    save()
    deadline = time.monotonic() + 5400
    try:
        cfg = verify(a.reviewed_source)
        report['protocol_sha256'] = sha(PROFILE)
        report['provider'] = provider_admission(a.provider, a.reviewed_source)
        require(not a.scratch.exists(), 'Fresh task-owned scratch required')
        a.scratch.mkdir(parents=True)
        ram = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024
        disk = shutil.disk_usage(a.scratch).free
        require(ram >= 4 * 2**30 and disk >= 8 * 2**30 and os.cpu_count() >= 2, 'Model-free4GiB RAM/8GiB disk/2CPU guard')
        report['resources'] = dict(available_ram_bytes=ram, free_disk_bytes=disk, cpu_count=os.cpu_count(), gpu_used=False)
        env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_') and k not in ['PYTHONPATH', 'LD_PRELOAD']}
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PIP_DISABLE_PIP_VERSION_CHECK='1')
        def command(name, argv, timeout=180, expected=(0,), custom_env=None):
            report['phase'] = name
            save()
            log = review / (name + '.log')
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'Overall90minute task deadline')
            result = run(argv, log, cwd=ROOT, env=custom_env or env, timeout=min(timeout, remaining))
            report['commands'][name] = dict(result, argv=list(map(str, argv)), log=log.name, **identity(log))
            save()
            require(not result['timed_out'] and result['exit'] in expected, 'Hosted phase rejected: ' + name)
            return result
        # Retain authenticated primary operator source, never framework/model payloads.
        for path, pin in cfg['torch_source'].items():
            if path == 'revision':
                continue
            url = 'https://raw.githubusercontent.com/pytorch/pytorch/' + cfg['torch_source']['revision'] + '/aten/src/ATen/native/' + path
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read(64 * 1024 + 1)
            target = raw / 'primary-source' / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            require(len(content) == pin['bytes'] and sha(target) == pin['sha256'], 'Primary operator source differs: ' + path)
        command('compiler', ['c++', '--version'])
        build = a.scratch / 'build'
        command('configure', ['cmake', '-S', ROOT / 'tests/moedern-router-topk-v1', '-B', build, '-DCMAKE_BUILD_TYPE=Release'])
        command('build', ['cmake', '--build', build, '--target', 'router-topk-operator', 'router-topk-ref', '--parallel', '2'], 2400)
        cache = (build / 'CMakeCache.txt').read_text()
        commands = read(build / 'compile_commands.json')
        report['build_profile'] = build_profile(cache, commands)
        shutil.copyfile(build / 'CMakeCache.txt', review / 'CMakeCache.txt')
        write(review / 'compile-commands.json', commands)
        for name in ['router-topk-operator', 'router-topk-ref']:
            src = build / name
            target = raw / 'binaries' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            report.setdefault('binaries', {})[name] = identity(target)
            command(name + '-ldd', ['ldd', src])
            dependencies = {}
            for line in (review / (name + '-ldd.log')).read_text().splitlines():
                match = re.search(r'(?:=>\s+)?(/[^\s]+)\s+\(', line)
                if match and Path(match[1]).is_file():
                    dependencies[str(Path(match[1]).resolve())] = identity(match[1])
            require(any('libstdc++' in key for key in dependencies), 'Actual comparator standard library identity missing')
            report.setdefault('linked_libraries', {})[name] = dependencies
        command('compiler-headers', [build / 'router-topk-operator', '--compiler-info'])
        # Isolated Python; locked uv bootstrap and exact CPython profile, never system-site packages.
        pins = read(ROOT / 'tools/moedern-router-topk-v1/wheels.json')
        bootstrap = a.scratch / 'bootstrap'
        uv_lock = a.scratch / 'uv.txt'
        uv = pins['uv']
        uv_lock.write_text('uv @ ' + uv['url'] + ' --hash=sha256:' + uv['sha256'] + '\n')
        command('install-uv', [sys.executable, '-m', 'pip', 'install', '--no-deps', '--only-binary=:all:', '--no-cache-dir',
                               '--require-hashes', '--target', bootstrap, '--report', review / 'uv-install.json', '-r', uv_lock], 600)
        uv_env = dict(env, PYTHONPATH=str(bootstrap))
        venv = a.scratch / 'venv'
        command('isolated-python', [sys.executable, '-m', 'uv', 'venv', '--python', '3.12.11', '--seed', venv], 600, custom_env=uv_env)
        py = venv / 'bin/python'
        command('install-locked-wheels', [py, '-m', 'pip', 'install', '--force-reinstall', '--no-cache-dir', '--only-binary=:all:',
                                          '--require-hashes', '--report', review / 'install.json', '-r',
                                          ROOT / 'tools/moedern-router-topk-v1/requirements.txt'], 1200)
        installed = read(review / 'install.json')['install']
        observed = {r['metadata']['name'].lower().replace('-', '_'): r for r in installed}
        for name, pin in pins.items():
            if name == 'uv':
                continue
            row = observed.get(name.lower().replace('-', '_'))
            require(row and row['metadata']['version'] == pin['version']
                    and row['download_info']['archive_info']['hashes']['sha256'] == pin['sha256']
                    and row['download_info']['url'] == pin['url'], 'Actual installed archive identity differs: ' + name)
        report['all_locked_archives_authenticated'] = True
        command('pip-check', [py, '-m', 'pip', 'check'])
        command('freeze-environment', [py, '-m', 'pip', 'freeze', '--all'])
        # Actual numerical differences exit1, yet all records survive. Structural failure exits2.
        result = command('operator-panel', [py, '-I', ROOT / 'tools/moedern-router-topk-v1/worker.py',
                                            '--reviewed-source', a.reviewed_source, '--build', build, '--out', a.out],
                         1200, (0, 1, 2))
        operator = read(review / 'operator.json')
        require(operator['source'] == a.reviewed_source and operator['model_access'] is False,
                'Operator source/scope differs')
        report.update(completed=operator['completed'], operator_report=identity(review / 'operator.json'),
                      operator_exit=result['exit'], all_fixed_comparisons_passed=operator.get('all_fixed_comparisons_passed', False),
                      phase='completed' if operator['completed'] else 'operator-structural-rejection')
        save()
        return result['exit']
    except Exception as e:
        report.update(error=type(e).__name__ + ': ' + str(e))
        save()
        return 2
    finally:
        # Authored arrays and exact binaries are retained, never downloaded to the shared host.
        write(review / 'file-manifest.json', {str(p.relative_to(a.out)): identity(p)
                                             for p in sorted(a.out.rglob('*')) if p.is_file() and p.name != 'file-manifest.json'})


if __name__ == '__main__':
    raise SystemExit(main())
