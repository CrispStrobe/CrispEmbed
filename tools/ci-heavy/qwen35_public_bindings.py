"""Hosted public Python/Rust/Dart consumers, no native reference inputs injected."""
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

def verify(repo, build, model, image, out, scratch):
    import numpy as np
    from PIL import Image
    lib = build / ('libcrispembed.dylib' if sys.platform == 'darwin' else 'libcrispembed.so')
    loader = 'DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'
    env = dict(os.environ, CRISPEMBED_SYS_LIB_DIR=str(build), CARGO_BUILD_JOBS='2')
    env[loader] = str(build) + os.pathsep + str(build / 'ggml/src') + os.pathsep + env.get(loader, '')
    os.environ[loader] = env[loader]
    sys.path.insert(0, str(repo / 'python'))
    from crispembed import CrispOcrModel, CrispOcrOrchestrator
    from crispembed._binding import _CrispOcrStage, _ScanCleanupParams, _CrispOcrPipelineParams
    rgb = np.array(Image.open(image).convert('RGB'))
    raw = scratch / 'public-input.rgb'
    raw.write_bytes(rgb.tobytes())
    results = {'python': []}
    model_api = CrispOcrModel(str(model), n_threads=4, lib_path=str(lib))
    try:
        model_api.set_max_tokens(12)
        assert model_api.get_max_tokens() == 12
        assert model_api.recognize(rgb) == 'Hello OCR  \n12345'
        assert len(model_api.confidences()) == 9 and abs(model_api.mean_confidence - .995560692) < 1e-6
        assert model_api.set_prompt('x') and model_api.set_prompt(None)
        model_api.set_max_tokens(2)
        assert model_api.recognize(rgb) == 'Hello OCR'
        results['python'].append('PASS unified image/prompt/reset/cap/confidence')
    finally:
        model_api.__del__()
    pipeline = CrispOcrOrchestrator(stages=[{'engine':20, 'model_a':str(model), 'vlm_max_tokens':12}],
                                   router=False, lib_path=str(lib))
    try:
        assert pipeline.run(str(image))['text'] == 'Hello OCR  \n12345'
        results['python'].append('PASS orchestrator ID20')
    finally:
        pipeline.__del__()
    results['python_abi'] = {'stage_size':ctypes.sizeof(_CrispOcrStage), 'cleanup_size':ctypes.sizeof(_ScanCleanupParams),
                            'params_size':ctypes.sizeof(_CrispOcrPipelineParams),
                            'vlm_max_tokens_offset':_CrispOcrStage.vlm_max_tokens.offset,
                            'vlm_prompt_offset':_CrispOcrStage.vlm_prompt.offset}
    rust = scratch / 'rust-qwen35-public'
    (rust / 'src').mkdir(parents=True, exist_ok=True)
    (rust / 'Cargo.toml').write_text('[package]\nname="qwen35-probe"\nversion="0.0.0"\nedition="2021"\n[dependencies]\ncrispembed={path='+json.dumps(str(repo/'crispembed'))+'}\n')
    shutil.copy(repo / 'tests/qwen35_public_bindings.rs', rust / 'src/main.rs')
    subprocess.run(['cargo','build','--manifest-path',str(rust/'Cargo.toml')], env=env, check=True)
    run = subprocess.run([str(rust/'target/debug/qwen35-probe'),str(model),str(raw),str(rgb.shape[1]),str(rgb.shape[0]),str(image)],
                         env=env, capture_output=True, text=True)
    (out/'public-rust.log').write_text(run.stdout+run.stderr)
    run.check_returncode()
    results['rust'] = [line for line in run.stdout.splitlines() if line.startswith('PASS')]
    dart = scratch / 'dart-qwen35-public'
    (dart / 'lib').mkdir(parents=True, exist_ok=True)
    for name in ('crispembed.dart','crispembed_bindings.dart'):
        shutil.copy(repo/'flutter/crispembed/lib/src'/name,dart/'lib'/name)
    (dart/'pubspec.yaml').write_text("name: qwen35_probe\nenvironment:\n  sdk: '>=3.0.0 <4.0.0'\ndependencies:\n  ffi: ^2.0.0\n")
    shutil.copy(repo/'tests/qwen35_public_bindings.dart',dart/'main.dart')
    arch='arm64' if platform.machine().lower() in ('aarch64','arm64') else 'x64'
    archive=scratch/'qwen35-dart-sdk.zip'
    urllib.request.urlretrieve(f'https://storage.googleapis.com/dart-archive/channels/stable/release/3.13.5/sdk/dartsdk-linux-{arch}-release.zip',archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(scratch)
        for entry in z.infolist():
            mode=entry.external_attr>>16
            if mode & 0o111: (scratch/entry.filename).chmod(mode & 0o777)
    sdk=scratch/'dart-sdk/bin/dart'
    subprocess.run([str(sdk),'--disable-analytics'],check=True)
    subprocess.run([str(sdk),'pub','get'],cwd=dart,check=True)
    subprocess.run([str(sdk),'analyze','--no-fatal-warnings'],cwd=dart,check=True)
    subprocess.run([str(sdk),'format',str(dart)],check=True)
    for name in ('crispembed.dart','crispembed_bindings.dart'):
        shutil.copy(dart/'lib'/name,out/('formatted-'+name))
    run=subprocess.run([str(sdk),'run',str(dart/'main.dart'),str(model),str(raw),str(rgb.shape[1]),str(rgb.shape[0]),str(image),str(lib)],
                       env=env,cwd=dart,capture_output=True,text=True)
    (out/'public-dart.log').write_text(run.stdout+run.stderr)
    run.check_returncode()
    results['dart']=[line for line in run.stdout.splitlines() if line.startswith('PASS')]
    m=re.search(r'ABI stage_size=(\d+) cleanup_size=(\d+) params_size=(\d+)',run.stdout)
    assert m
    results['dart_abi']=dict(zip(('stage_size','cleanup_size','params_size'),map(int,m.groups())))
    cpp=(out/'native-public.log').read_text()
    m=re.search(r'ABI stage_size=(\d+) cleanup_size=(\d+) params_size=(\d+) vlm_max_tokens_offset=(\d+) vlm_prompt_offset=(\d+)',cpp)
    assert m
    expected=dict(zip(results['python_abi'],map(int,m.groups())))
    assert results['python_abi']==expected
    assert all(results['dart_abi'][k]==expected[k] for k in results['dart_abi'])
    (out/'public-bindings.json').write_text(json.dumps(results,indent=2)+'\n')


def entries(build, model, image, out):
    import socket
    import time
    import urllib.error
    results = {}
    cases = [('single', ['-m', str(model), '--ocr', str(image)], 'text'),
             ('stage', ['--ocr-pipeline', str(image), '--ocr-engine', 'qwen35-ocr', '--ocr-rec', str(model)], 'full_text')]
    for label, args, key in cases:
        run = subprocess.run([str(build/'crispembed'), *args, '--offline', '--ocr-max-tokens', '12', '--json', '-t', '4'],
                             capture_output=True, text=True)
        (out/('public-cli-'+label+'.log')).write_text(run.stdout+run.stderr)
        run.check_returncode()
        lines=[l for l in run.stdout.splitlines() if l.startswith('{')]
        answer=json.loads(lines[-1])
        assert answer[key]=='Hello OCR  \n12345',answer
        results['cli_'+label]={'text_exact':True,'json':answer}
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    log=open(out/'public-server.log','w')
    proc=subprocess.Popen([str(build/'crispembed-server'),'--offline','--ocr-pipeline','--ocr-engine','qwen35-ocr',
                           '--ocr-rec',str(model),'--ocr-max-tokens','12','--host','127.0.0.1','--port',str(port),
                           '--image-root',str(image.parent)],stdout=log,stderr=log)
    try:
        url=f'http://127.0.0.1:{port}/ocr/pipeline'
        deadline=time.monotonic()+60
        while True:
            if proc.poll() is not None: raise RuntimeError('server exited before ready')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2): break
            except urllib.error.HTTPError: break
            except (urllib.error.URLError,TimeoutError):
                if time.monotonic()>deadline: raise RuntimeError('server readiness timeout')
                time.sleep(.25)
        request=urllib.request.Request(url,json.dumps({'image':str(image)}).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=300) as response: answer=json.load(response)
        assert answer['text']=='Hello OCR  \n12345' and answer['n_regions']==1,answer
        results['server_stage']={'text_exact':True,'n_regions':answer['n_regions'],'mean_confidence':answer['mean_confidence']}
    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill();proc.wait()
        log.close()
    (out/'public-entries.json').write_text(json.dumps(results,indent=2)+'\n')
