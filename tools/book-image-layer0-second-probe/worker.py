#!/usr/bin/env python3
"""One bounded native quantized prefill; cache-closed test observer, no decoding."""
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import prepare
sys.path.insert(0, str(ROOT / 'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline, command
sys.path.insert(0, str(ROOT / 'tools/heldout'))
from transfer import download as authenticated_transfer


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    import numpy as np
    from PIL import Image
    audit = load('book_stage_audit', ROOT / 'tools/book-stage-compare/audit.py')
    profile = load('physical_profile', ROOT / 'tools/physical-f16-attention-probe/worker.py')
    base = load('physical_base', ROOT / 'tools/kaggle/lighton4b_upstream_physical_decode.py')
    cfg = json.loads((HERE / 'protocol.json').read_text())
    require, sha = audit.require, audit.sha
    require(Path('/kaggle/input').is_dir() and np.__version__ == '2.3.5' and Image.__version__ == '11.3.0' and platform.machine().lower() in ['x86_64', 'amd64'], 'pinned authorized worker profile')
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(exist_ok=True)
    work = scratch / 'book-image-layer0-second'
    work.mkdir()
    started = time.monotonic()
    deadline = Deadline(2400)
    deadline.__enter__()
    report = dict(schema='lighton4b-book-image-layer0-second-probe/v1', status='failed', commands={},
                  qualified=False, decoded_generation_executed=False, original_forward_executed=False,
                  full_f32_text_model_instantiated=False, native_prefill_executed=False)
    def save():
        (out / 'image-layer0-probe.json').write_text(json.dumps(report, indent=2) + '\n')
    def run(label, args, env=None, seconds=600):
        report['phase'] = label
        report['commands'][label] = {}
        save()
        try:
            command(args, out / (label + '.log'), min(seconds, 2400 - (time.monotonic() - started)), report['commands'][label], env=env, cwd=ROOT)
        finally:
            save()
    def locate(name, digest):
        matches = [p for p in Path('/kaggle/input').rglob(name) if p.is_file() and p.stat().st_size < 4 << 20 and sha(p) == digest]
        require(len(matches) == 1, 'one authenticated mounted receipt')
        return matches[0], json.loads(matches[0].read_text())
    try:
        run('controller-source', ['git', 'rev-parse', 'HEAD'])
        report['controller_source'] = (out / 'controller-source.log').read_text().strip()
        require(report['controller_source'] == os.environ['LIGHTON4B_EXPECTED_SOURCE'], 'exact controller source')
        report['resources'] = dict(memory=base.memory(), scratch_free=shutil.disk_usage(work).free, output_free=shutil.disk_usage(out).free, gpu_session=bool(list(Path('/dev').glob('nvidia[0-9]*'))))
        require(report['resources']['memory']['MemAvailable'] >= cfg['minimum_memory_available_bytes'] and report['resources']['scratch_free'] >= cfg['minimum_scratch_available_bytes'] and report['resources']['output_free'] >= 4 << 30, 'resource admission before source/model work')
        npth, historical = locate('planner-heldout.json', cfg['historical_native_receipt_sha256'])
        require(historical['status'] == 'failed' and historical['error_type'] == 'TimeoutExpired' and historical['source_sha'] == prepare.SOURCE and historical['old_baseline_all_rows_stages_exact'], 'preserved historical native evidence')
        spath, source = locate('book-text-trace.json', cfg['neutral_source_receipt_sha256'])
        require(source['status'] == 'measured_neutral_book_text_trace' and source['all654_scores_observer_neutral'], 'neutral source capture')
        opath, original = locate('heldout-result.json', '9b88195d5a8058dea4fab87472009d3826bc1fa3b5d0575aa2e33698316d5722')
        require(original['source_observer_full_neutral'], 'historical original observer neutrality')
        def old(name):
            return audit.owned(npth.parent, name, historical['artifacts'])
        seal = old('pre-forward-inputs.json')
        require(sha(seal) == cfg['input_seal_sha256'] == original['pre_forward_input_seal_sha256'], 'original input seal')
        png = audit.owned(opath.parent, 'inputs/input.png', original['artifacts'])
        rgb = audit.owned(opath.parent, 'inputs/input.rgb', original['artifacts'])
        image = Image.open(png).convert('RGB')
        require(image.tobytes() == rgb.read_bytes(), 'authenticated actual pixels')
        runtime = work / 'runtime-source'
        for label, args in [('init', ['git', 'init', runtime]), ('remote', ['git', '-C', runtime, 'remote', 'add', 'origin', 'https://github.com/CrispStrobe/CrispEmbed.git']), ('fetch', ['git', '-C', runtime, 'fetch', '--depth', '1', 'origin', prepare.SOURCE]), ('checkout', ['git', '-C', runtime, 'checkout', '--detach', 'FETCH_HEAD']), ('ggml', ['git', '-C', runtime, 'submodule', 'update', '--init', '--depth', '1', 'ggml'])]:
            run(label, args)
        require(subprocess.check_output(['git', '-C', runtime, 'rev-parse', 'HEAD'], text=True).strip() == prepare.SOURCE and subprocess.check_output(['git', '-C', runtime / 'ggml', 'rev-parse', 'HEAD'], text=True).strip() == '3a27b6c665b191cb230be367db5c52d81dd3e2ac', 'exact historical runtime/ggml')
        report['generated_driver'] = prepare.prepare(runtime)
        run('ordinary-sources-unchanged', ['git', '-C', runtime, 'diff', '--exit-code', '--', 'src', 'ggml', 'tests/lighton4b_native_document.cpp'])
        build = work / 'build'
        flags = dict(profile.BUILD_FLAGS, GGML_CPU_REPACK='ON', BUILD_SHARED_LIBS='ON')
        run('configure', ['cmake', '-S', runtime, '-B', build, '-DCMAKE_BUILD_TYPE=Release'] + ['-D%s=%s' % pair for pair in flags.items()])
        lines = (build / 'CMakeCache.txt').read_text().splitlines()
        profile.validate_profile(lines)
        require(all('%s:BOOL=%s' % pair in lines for pair in flags.items()), 'exact full build profile')
        shutil.copy2(build / 'CMakeCache.txt', out / 'CMakeCache.txt')
        run('baseline', ['python3', runtime / 'scripts/check-cpu-baseline.py', build])
        run('build', ['cmake', '--build', build, '--target', prepare.TARGET, '--parallel', '2'], seconds=900)
        binary = build / prepare.TARGET
        report['native_binary_sha256'] = sha(binary)
        sys.path.insert(0, str(ROOT / 'tools'))
        pack = load('physical_pack', ROOT / 'tools/lighton4b_upstream_physical_pack.py')
        model_cfg = json.loads((ROOT / 'docs/lighton4b-heldout-executor-v1.json').read_text())
        def download(url, path, row):
            report['phase'] = 'download-' + path.name
            entries = report.setdefault('downloads', {}).setdefault(path.name, [])
            save()
            try:
                authenticated_transfer(base.download, url, path, row['bytes'], row['sha256'], entries)
            finally:
                save()
        row = model_cfg['models']['Q4_K_M']
        upstream = work / row['file']
        download('https://huggingface.co/%s/resolve/%s/%s' % (model_cfg['model_repo'], model_cfg['model_revision'], row['file']), upstream, row)
        row = model_cfg['base_model']
        protected = work / row['file']
        download('https://huggingface.co/%s/resolve/%s/%s' % (row['repository'], row['revision'], row['file']), protected, row)
        candidate = work / 'candidate.gguf'
        packed = pack.convert(protected, upstream, candidate, out / 'physical-pack.json')
        require(packed['candidate_sha256'] == cfg['model_sha256'], 'unchanged candidate bytes')
        upstream.unlink()
        protected.unlink()
        require(base.memory()['MemAvailable'] >= cfg['minimum_memory_available_bytes'], 'memory admission before native prefill')
        dest = out / 'native'
        dest.mkdir()
        env = dict(os.environ)
        env.update({key: '0' for key in env if key.startswith('CRISPEMBED_')})
        env.update(historical['native_gates']['on'])
        report['native_gates'] = {key: value for key, value in env.items() if key.startswith('CRISPEMBED_')}
        require(report['native_gates'] == historical['native_gates']['on'], 'exact historical arithmetic policy vector')
        teacher = work / 'unused.ids'
        teacher.write_text('248046\n')
        args = [binary, candidate, rgb, image.height, image.width, teacher, dest, 1536, 4, 'greedy-full', '--provisional-heldout-cap1536']
        run('native-prefill', args, env=env, seconds=300)
        require('BOOK_IMAGE_LAYER0_CAPTURE PASS actual_prefill=1 decoded=0' in (out / 'native-prefill.log').read_text(), 'actual test executable closure marker')
        report['native_prefill_executed'] = True
        observed = json.loads((dest / 'image-layer0-capture.json').read_text())
        require(all(observed[key] is True for key in ['all_layer0_tiles_hidden_exact', 'final_layer0_cache_exact', 'previous_caches_immutable', 'core_replay_attn_state_conv_exact']), 'actual cache/tile/core closure')
        require((observed['offset'], observed['cache_before_tokens'], observed['cache_after_tokens']) == (20, 20, 36), 'exact second image capture cache scope')
        report['capture'] = observed
        report['test_overlay_sha256'] = {name: sha(HERE / name) for name in ['prepare.py', 'worker.py', 'protocol.json', 'test_controls.py']}
        stage_keys = [name for name in historical['artifacts'] if name.startswith(('corrected-composed/prefill.', 'corrected-composed/vision.', 'corrected-composed/processor.')) or name in ['corrected-composed/prompt.ids', 'corrected-composed/modalities.ids', 'corrected-composed/text.embedded.f32']]
        for name in stage_keys:
            prior = old(name)
            current = dest / name.split('/', 1)[1]
            require(current.stat().st_size == prior.stat().st_size and sha(current) == sha(prior), 'historical full prefill/vision/firstscore neutrality')
        report['all_historical_prefill_vision_stage_hashes_exact'] = len(stage_keys)
        graphs, unused_step = audit.read_graphs(spath.parent, source)
        image_graph = graphs[1]
        report['comparisons'] = []
        for source_name, suffix, width, role in [('model.input_embed', 'input', 2560, None)] + audit.STAGES[:5] + [('final_output-0', 'core_values', 4096, 'z')] + audit.STAGES[5:] + [('l_out-0', 'hidden', 2560, None)]:
            reference = audit.source_rows(image_graph, source_name, width, 252, role)[16:32]
            path = dest / ('image.layer0.' + suffix + '.f32')
            require(path.stat().st_size == 16*width*4, 'captured image dimensions')
            current = np.fromfile(path, dtype='<f4').reshape(16, width)
            report['comparisons'].append(dict(stage=source_name, **audit.delta(current, reference)))
        require(report['comparisons'][0]['bit_exact'], 'same actual image input')
        report['earliest_recorded_difference'] = next((x for x in report['comparisons'] if not x['bit_exact']), None)
        require(sha(candidate) == cfg['model_sha256'], 'weights unchanged')
        report['status'] = 'measured_cache_closed_image_layer0_trace'
        report['limits'] = ['One frozen book second image chunk (image rows16..31), no decoded generation or quality claim.', 'Actual main tile/final cache and core replay close bitwise; source persistent cache layout remains unverified and is not compared.', 'Source image graph252 versus native16; differing subsequent inputs preclude unique operator attribution without replay.']
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        deadline.close()
        report['elapsed_seconds'] = time.monotonic() - started
        report['artifacts'] = {str(path.relative_to(out)): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in out.rglob('*') if path.is_file() and path.name != 'image-layer0-probe.json'}
        save()


if __name__ == '__main__':
    main()
