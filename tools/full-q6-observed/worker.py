#!/usr/bin/env python3
"""Closed observed full-Q6 import/control/decode; no original or source forward."""
import hashlib
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
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline, command

GATE = 'CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL_Q6'
REPACK = 'CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK'
EOS = {248044, 248046}
VOCAB = 248320
PROFILE = dict(CRISPEMBED_NATIVE='OFF', GGML_NATIVE='OFF', GGML_AVX2='ON', GGML_FMA='ON',
               GGML_F16C='ON', GGML_LLAMAFILE='ON', GGML_CPU_REPACK='ON', GGML_AVX512='OFF',
               GGML_BLAS='OFF', GGML_BACKEND_DL='OFF', GGML_OPENMP='OFF', GGML_CUDA='OFF',
               GGML_VULKAN='OFF', GGML_METAL='OFF')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified(path, row):
    require(path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
            'authenticated owned artifact: ' + path.name)
    return path


def admit_protocol(path, expected=None):
    # Protocol is task configuration; content hashes are reported, never an ACK gate.
    cfg = json.loads(path.read_text())
    require(cfg['cases'] == ['simple_table', 'simple_form', 'arabic_printed_line'] and
            cfg['old_q4_cap'] == 512 and cfg['full_q6_cap'] == 256 and cfg['threads'] == 4 and
            not any(cfg[k] for k in ['heldout_allowed', 'source_forward_allowed', 'original_forward_allowed',
                                    'full_f32_model_allowed', 'quality_qualified']), 'closed observed scope')
    return cfg


def memory_available():
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    observed = 0
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        hierarchy, controllers, relative = line.split(':', 2)
        if hierarchy == '0' and controllers == '':
            base, limit_name, used_name = Path('/sys/fs/cgroup'), 'memory.max', 'memory.current'
        elif 'memory' in controllers.split(','):
            base, limit_name, used_name = Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        else:
            continue
        target = base / relative.lstrip('/')
        for node in (target, *target.parents):
            if node != base and base not in node.parents:
                break
            limit, used = node / limit_name, node / used_name
            if limit.is_file() and used.is_file():
                observed += 1
                raw = limit.read_text().strip()
                if raw != 'max' and int(raw) < 2**60:
                    available = min(available, max(0, int(raw) - int(used.read_text())))
    require(observed > 0, 'inspect actual ancestor cgroup memory')
    return available


def gates(frozen):
    old = dict(frozen)
    require(old.get(REPACK) == '1' and all(old.get(key) == '1' for key in [
        'CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL', 'CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED',
        'CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL', 'CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL_ALIGNED',
        'CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU', 'CRISPEMBED_LIGHTON4B_PHYSICAL_ROPE',
        'CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS']), 'frozen corrected Q4 gate profile')
    old[GATE] = '0'
    new = dict(old)
    new.update({GATE: '1', REPACK: '0'})
    require({key for key in old if old[key] != new[key]} == {GATE, REPACK}, 'only profile and absent Q4 repack change')
    return old, new


def edits(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (left != right)))
        previous = current
    return previous[-1]


def metric(ids, text, reference_ids, reference_text):
    prefix = 0
    for a, b in zip(ids, reference_ids):
        if a != b:
            break
        prefix += 1
    return dict(ids_exact=ids == reference_ids, text_exact=text == reference_text, matching_prefix_ids=prefix,
                character_edits=edits(text, reference_text), reference_codepoints=len(reference_text),
                word_edits=edits(text.split(), reference_text.split()), reference_words=len(reference_text.split()))


def stop_state(summary, ids, cap):
    require(ids and all(type(token) is int and 0 <= token < VOCAB for token in ids), 'actual vocabulary IDs')
    require(summary['generated_tokens'] == len(ids) <= cap and summary['cap'] == cap, 'actual decode cap/count')
    require(not any(token in EOS for token in ids[:-1]), 'no ignored earlier EOS')
    require(summary['eos'] == (ids[-1] in EOS) and (summary['eos'] or len(ids) == cap), 'EOS or explicit cap')
    require(summary['upstream_physical'] and not summary['source_fed_vision'] and summary['teacher_positions'] == 1,
            'fresh pixel independent greedy mode')


def admit_source_response(response, source, prompt_tokens):
    settings = response['generation_settings']
    require(response['tokens'] == source['ids'] and response['content'] == source['decoded'] and
            response['stop'] is True and response['stop_type'] == 'eos' and not response['truncated'] and
            response['tokens_predicted'] == len(source['ids']) == len(response['completion_probabilities']),
            'actual frozen Q6 response IDs/text/EOS/count')
    require(settings['max_tokens'] == settings['n_predict'] == 256 and settings['ignore_eos'] is False and
            settings['temperature'] == 0 and settings['top_k'] == 1 and
            source['input_tokens'] == response['tokens_evaluated'] == prompt_tokens and
            source['prompt_ids_exact_original'] is True, 'actual frozen source cap/greedy/prompt contract')


def numerical(a, b, np):
    require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(), 'finite paired geometry')
    x, y = a.astype('f8'), b.astype('f8')
    error = x - y
    return dict(bit_exact=a.tobytes() == b.tobytes(), rms=float(np.sqrt(np.mean(error*error))),
                max_abs=float(np.abs(error).max()), candidate_rms_magnitude=float(np.sqrt(np.mean(x*x))),
                reference_rms_magnitude=float(np.sqrt(np.mean(y*y))),
                candidate_abs_max=float(np.abs(x).max()), reference_abs_max=float(np.abs(y).max()))


def rows(folder, ids, np):
    hashes = []
    for i, token in enumerate(ids):
        path = folder / ('greedy.%d.logits.f32' % i)
        require(path.stat().st_size == VOCAB * 4, 'complete vocabulary bytes')
        values = np.fromfile(path, dtype='<f4')
        require(np.isfinite(values).all() and int(values.argmax()) == token, 'finite greedy row/actual argmax')
        hashes.append(sha(path))
    require(sha(folder/'prefill.logits.f32') == sha(folder/'teacher.0.logits.f32') == hashes[0], 'three firstscore anchors')
    return hashes


def main():
    import numpy as np
    from PIL import Image
    require(Path('/kaggle/input').is_dir(), 'Kaggle worker required for model execution')
    cfg = admit_protocol(HERE/'protocol.json')
    out = Path(os.environ['HEAVY_OUT'])
    work = Path(os.environ['HEAVY_SCRATCH'])/'full-q6-observed'
    work.mkdir()
    started = time.monotonic()
    report = dict(schema=cfg['schema'], status='failed', commands={}, cases={}, quality_qualified=False,
                  source_forward_executed=False, original_forward_executed=False, full_f32_model_instantiated=False,
                  heldout_data_accessed=False)
    def save():
        (out/'full-q6-observed.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    def run(label, args, env=None, limit=900):
        report['phase'] = label
        receipt = report['commands'].setdefault(label, {})
        save()
        print('PHASE '+label, flush=True)
        try:
            command(args, out/(label+'.log'), min(limit, cfg['worker_seconds']-(time.monotonic()-started)),
                    receipt, env=env, cwd=ROOT)
        finally:
            save()
    def download(repository, revision, row):
        path = work/row['file']
        url = 'https://huggingface.co/'+repository+'/resolve/'+revision+'/'+row['file']
        # Normal transfer retries; remove only this task's incomplete download.
        for attempt in range(1, 4):
            report['phase'] = 'download-'+path.name
            save()
            try:
                h, count = hashlib.sha256(), 0
                with urllib.request.urlopen(url, timeout=120) as src, path.open('wb') as target:
                    while block := src.read(4 << 20):
                        count += len(block)
                        require(count <= row['bytes'], 'bounded model transfer')
                        h.update(block)
                        target.write(block)
                require(count == row['bytes'] and h.hexdigest() == row['sha256'], 'downloaded model content differs')
                return path
            except (OSError, TimeoutError):
                if path.exists():
                    path.unlink()
                if attempt == 3:
                    raise
    deadline = Deadline(cfg['worker_seconds'])
    deadline.__enter__()
    previous = {sig: signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(SystemExit(128+signum)))
                for sig in [signal.SIGTERM, signal.SIGINT]}
    try:
        report['source_sha'] = subprocess.check_output(['git', '-C', ROOT, 'rev-parse', 'HEAD'], text=True).strip()
        report['protocol_sha256'] = sha(HERE/'protocol.json')
        report['capacity'] = dict(available_bytes=memory_available(), scratch_free=shutil.disk_usage(work).free,
                                  output_free=shutil.disk_usage(out).free, load=os.getloadavg())
        require(report['capacity']['available_bytes'] >= cfg['minimum_memory_gib'] << 30 and
                report['capacity']['scratch_free'] >= cfg['minimum_scratch_gib'] << 30 and
                report['capacity']['output_free'] >= cfg['minimum_output_gib'] << 30, 'sufficient model memory and disk')
        source_path = ROOT/cfg['source_report_path']
        independent = json.loads(source_path.read_text())
        source_cfg = json.loads((ROOT/'docs/lighton4b-independent-llama-baseline-v1.json').read_text())
        corrected = json.loads((ROOT/'tests/results/lighton4b/receipt-layer0-operators/aligned-observed.json').read_text())
        prepared = []
        for case in cfg['cases']:
            case_cfg = next(row for row in source_cfg['cases'] if row['id'] == case)
            image = Image.open(ROOT/case_cfg['image']).convert('RGB')
            rgb = work/(case+'.rgb')
            rgb.write_bytes(image.tobytes())
            original_path = source_path.parent/'simple_form-original.json' if case == 'simple_form' else ROOT/case_cfg['reference_path']
            original = json.loads(original_path.read_text())
            source = independent['arms']['Q6_K'][case]
            response = json.loads((source_path.parent/('Q6_K-'+case)/'response.json').read_text())
            admit_source_response(response, source, case_cfg['expected_prompt_tokens'])
            teacher = work/(case+'.teacher.ids')
            teacher.write_text(str(original['generated_ids'][0])+'\n')
            _, profile = gates(corrected['cases'][case]['arms']['on']['gates'])
            prepared.append((case, image, rgb, teacher, original, source, profile))
        run('python-controller-controls', [sys.executable, HERE/'test_controls.py'])
        run('python-packer-controls', [sys.executable, ROOT/'tools/test_lighton4b_upstream_q6_physical_pack.py'])
        run('submodule', ['git', 'submodule', 'update', '--init', 'ggml'])
        build = work/'build'
        run('configure', ['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', *['-D'+k+'='+v for k, v in PROFILE.items()]])
        shutil.copy2(build/'CMakeCache.txt', out/'CMakeCache.txt')
        run('build', ['cmake', '--build', build, '--target', 'lighton4b-native-document', *cfg['control_targets'], '--parallel', '2'], limit=1800)
        clean = dict(os.environ)
        clean.update({key: '0' for key in clean if key.startswith('CRISPEMBED_')})
        clean[GATE] = '0'
        for target, control in cfg['control_targets'].items():
            run(target, [build/target, *[ROOT/p for p in control['arguments']]], clean)
            log = (out/(target+'.log')).read_text()
            require(control['marker'] in log and 'UNSUPPORTED' not in log and 'SKIP' not in log, 'supported '+target)
        bundle = out/'runtime'
        (bundle/'bin').mkdir(parents=True)
        (bundle/'lib').mkdir()
        native = bundle/'bin/lighton4b-native-document'
        shutil.copy2(build/'lighton4b-native-document', native)
        for path in build.rglob('libggml*.so*'):
            if path.is_file():
                shutil.copy2(path, bundle/'lib'/path.name)
        report['runtime_bundle'] = {str(p.relative_to(bundle)): dict(bytes=p.stat().st_size, sha256=sha(p))
                                    for p in bundle.rglob('*') if p.is_file()}
        clean['LD_LIBRARY_PATH'] = str(bundle/'lib')
        sys.path.insert(0, str(ROOT/'tools'))
        import lighton4b_upstream_q6_physical_pack as pack6
        old_source = download(source_cfg['model_repo'], source_cfg['model_revision'], source_cfg['models']['Q4_K_M'])
        q6_source = download(source_cfg['model_repo'], source_cfg['model_revision'], source_cfg['models']['Q6_K'])
        base_cfg = json.loads((ROOT/'docs/lighton4b-composed-form-vision-v1.json').read_text())['base_model']
        protected = download(base_cfg['repository'], base_cfg['revision'], base_cfg)
        model = work/'q6-physical.gguf'
        packed = pack6.convert(protected, q6_source, model, out/'full-q6-pack.json', old_source)
        require(packed['census'] == {'0': 474, '14': 249}, 'full-Q6 storage profile')
        # Task-owned source copies can be released after successful import.
        for path in [old_source, q6_source, protected]:
            path.unlink()
        for case, image, rgb, teacher, original, source, profile in prepared:
            require(memory_available() >= 8 << 30 and shutil.disk_usage(out).free >= 2 << 30, 'per-arm memory/disk')
            dest = out/case
            dest.mkdir()
            env = dict(clean)
            env.update(profile)
            run(case, [native, model, rgb, image.height, image.width, teacher, dest, 256, 4, 'greedy-full'], env, cfg['arm_seconds'])
            summary = json.loads((dest/'native-summary.json').read_text())
            ids = list(map(int, (dest/'generated.ids').read_text().split()))
            text = (dest/'generated.txt').read_bytes().decode('utf-8')
            stop_state(summary, ids, 256)
            row_hashes = rows(dest, ids, np)
            require(summary['physical_full_q6'] and not summary['physical_q4_repack'] and
                    summary['physical_repack_entries'] == summary['physical_repack_bytes'] == 0 and
                    summary['converted_f32_weight_entries'] == summary['converted_f32_weight_bytes'] == 0,
                    'actual full-Q6 profile/no F32 cache')
            layers = []
            for index in range(32):
                values = np.fromfile(dest/('prefill.layer.%d.f32' % index), dtype='<f4')
                require(values.size and np.isfinite(values).all(), 'finite actual decoder hidden states')
                layers.append(dict(layer=index, rms_magnitude=float(np.sqrt(np.mean(values.astype('f8')**2))),
                                   absolute_max=float(np.abs(values).max())))
            candidate = metric(ids, text, original['generated_ids'], original['decoded'])
            source_floor = metric(source['ids'], source['decoded'], original['generated_ids'], original['decoded'])
            report['cases'][case] = dict(summary=summary, gates=profile, generated_ids=ids, decoded=text,
                row_sha256=row_hashes, native_hidden_magnitudes=layers, vs_same_q6_source=metric(ids, text, source['ids'], source['decoded']),
                vs_bf16=candidate, source_q6_vs_bf16=source_floor,
                meets_source_q6_floor=bool(summary['eos'] and candidate['character_edits'] <= source_floor['character_edits'] and
                                          candidate['word_edits'] <= source_floor['word_edits']))
            save()
        retained = out/'full-q6-physical.gguf'
        shutil.move(model, retained)
        report['retained_model'] = dict(file=retained.name, bytes=retained.stat().st_size, sha256=sha(retained))
        report.update(status='measured_full_q6_observed', source_q6_decoded_exact_cases=sum(
            row['vs_same_q6_source']['ids_exact'] and row['vs_same_q6_source']['text_exact'] for row in report['cases'].values()),
            source_q6_floors_passed=sum(row['meets_source_q6_floor'] for row in report['cases'].values()), limits=cfg['limits'])
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        deadline.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report['elapsed_seconds'] = time.monotonic()-started
        report['artifacts'] = {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha(p))
                               for p in out.rglob('*') if p.is_file() and p.name != 'full-q6-observed.json'}
        save()


if __name__ == '__main__':
    main()
