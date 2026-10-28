"""Hosted-only CPU experiment; replace @COMMIT@ after source review.
No original rerun, publication or production hook. Keep all task output remote.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

COMMIT = '@COMMIT@'
OUT = Path('/kaggle/working/unlimited-q4k')
OUT.mkdir(parents=True, exist_ok=False)
START = time.monotonic()
report = dict(schema='crispembed-unlimited-ocr2-q4k-setup-v1', source=COMMIT, completed=False, commands=[])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def save():
    (OUT / 'setup.json').write_text(json.dumps(report, indent=1) + '\n')


def run(args, timeout=2400, accepted=(0,)):
    number = len(report['commands'])
    log = OUT / f'command-{number}.log'
    timed_out = False
    with log.open('wb') as stream:
        child = subprocess.Popen(list(map(str, args)), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    report['commands'].append(dict(argv=list(map(str, args)), exit=child.returncode,
                                    timed_out=timed_out, log=log.name, sha256=sha(log)))
    save()
    if timed_out or child.returncode not in accepted:
        raise RuntimeError('hosted command rejected: ' + str(number))


try:
    assert len(COMMIT) == 40 and all(c in '0123456789abcdef' for c in COMMIT), 'immutable source required'
    Path('/kaggle/temp').mkdir(parents=True, exist_ok=True)
    report['resources'] = dict(mem_available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                                   if line.startswith('MemAvailable:'))) * 1024,
                               disk_free=shutil.disk_usage('/kaggle/temp').free, cpu_count=os.cpu_count())
    save()
    assert report['resources']['mem_available'] >= 10 * 1024**3, '10GiB available RAM required'
    assert report['resources']['disk_free'] >= 32 * 1024**3, '32GiB scratch required'
    root = Path('/kaggle/temp/ce')
    run(['git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', root])
    os.chdir(root)
    run(['git', 'checkout', '-q', COMMIT])
    run(['git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml'])
    sys.path.insert(0, str(root / 'tools/unlimited-ocr2-q4k'))
    from contract import guard, references, F16_SHA, F16_BYTES, model_download_command
    from overlay import transform
    report['source_hashes'] = guard()
    references()
    run([sys.executable, 'tests/test_unlimited_ocr2_q4k.py'])
    run([sys.executable, 'tests/test_unlimited_real_decode.py'])
    run([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17'])
    run([sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv'])
    py = '/kaggle/temp/venv/bin/python'
    run([py, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'gguf==0.17.1', 'pillow==11.3.0'])
    run([py, '-m', 'pip', 'freeze'])
    build = '/kaggle/temp/build'
    run(['cmake', '-S', 'tests/unlimited-q4k', '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DGGML_METAL=OFF', '-DGGML_VULKAN=OFF'])
    run(['cmake', '--build', build, '--target', 'crispembed-cli', 'crispembed-quantize',
         'unlimited-q4-diff', 'unlimited-q4-projection', '--parallel', '2'], 5400)
    # A separate hosted worktree compiles only the declared observer delta.
    observed = Path('/kaggle/temp/observed-source')
    run(['git', 'worktree', 'add', '--detach', observed, COMMIT])
    run(['git', '-C', observed, 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml'])
    patched = transform((root / 'src/unlimited_ocr.cpp').read_bytes())
    (observed / 'src/unlimited_ocr.cpp').write_bytes(patched)
    changed = subprocess.check_output(['git', '-C', observed, 'diff', '--name-only']).decode().splitlines()
    assert changed == ['src/unlimited_ocr.cpp'], 'observer source closure differs'
    report['observed_source'] = dict(source=COMMIT, changed=changed, sha256=hashlib.sha256(patched).hexdigest())
    save()
    obuild = '/kaggle/temp/observed-build'
    run(['cmake', '-S', observed, '-B', obuild, '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
         '-DCRISPEMBED_NATIVE=OFF', '-DGGML_NATIVE=OFF', '-DGGML_LLAMAFILE=ON', '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', '-DGGML_VULKAN=OFF'])
    run(['cmake', '--build', obuild, '--target', 'crispembed-cli', '--parallel', '2'], 5400)
    # Root CMake has identical CPU/backend settings in both trees. Record actual
    # caches and compile commands; reject accidental source/flag mismatches.
    from build_profile import verify
    profile = verify(Path(build), Path(obuild), root, observed)
    (OUT / 'build-profile.json').write_text(json.dumps(profile, indent=1) + '\n')
    report['build_profile_sha256'] = sha(OUT / 'build-profile.json')
    symbols = subprocess.check_output(['nm', '-C', build + '/root/crispembed'], text=True)
    assert 'uocr_q4_audit::' not in symbols, 'ordinary observer symbol leaked'
    assert 'uocr_q4_audit::' in subprocess.check_output(['nm', '-C', obuild + '/crispembed'], text=True), 'positive observer archive control'
    report['production_observer_excluded'] = True
    save()
    # Real comparator positive + numerical/shape rejection controls before model access.
    run([py, 'tools/unlimited-ocr2-q4k/controls.py', '--diff', build + '/unlimited-q4-diff', '--output', OUT / 'ref-controls'])
    # Render from unchanged licensed fixture, then require the frozen PNG hashes.
    run([py, 'tools/kaggle/unlimited-ocr2-trial/trial_worker.py', 'render', OUT / 'pages'])
    _, images = references()
    assert all(sha(OUT / 'pages' / (case + '.png')) == expected for case, expected in images.items()), 'frozen PNG reproduction failed'
    model = Path('/kaggle/temp/unlimited-f16.gguf')
    run(model_download_command(model), 2400)
    assert model.stat().st_size == F16_BYTES and sha(model) == F16_SHA, 'published F16 SHA/extent differs'
    # Actual production text decoder on retained real-scan IDs. It reuses the
    # already authenticated F16 metadata; no model/reference forward is added.
    run([py, 'tools/unlimited-ocr2/replay_real_decode.py', '--model', model,
         '--out', OUT / 'real-decoder'], 300)
    report['real_decoder_sha256'] = sha(OUT / 'real-decoder/review.json')
    save()
    # Numeric failure is retained as a completed negative panel, not a setup error.
    remaining = int(11.2 * 3600 - (time.monotonic() - START))
    assert remaining >= 120, 'overall deadline reached before any forward'
    run([py, 'tools/unlimited-ocr2-q4k/panel.py', '--source', COMMIT, '--model', model,
         '--pages', OUT / 'pages', '--output', OUT / 'panel', '--cli', build + '/root/crispembed',
         '--observed', obuild + '/crispembed', '--diff', build + '/unlimited-q4-diff',
         '--projection', build + '/unlimited-q4-projection', '--quantizer', build + '/root/crispembed-quantize',
         '--threads', '4', '--budget-seconds', str(remaining - 60)], remaining, accepted=(0, 1))
    panel = json.loads((OUT / 'panel/panel.json').read_text())
    report['panel_completed'] = panel['completed']
    report['panel_sha256'] = sha(OUT / 'panel/panel.json')
    report['completed'] = bool(panel['completed'])
    save()
    if not report['completed']:
        raise RuntimeError('panel structural/baseline rejection; all partial evidence retained')
except Exception as error:
    report['error'] = str(error)
    save()
    raise
