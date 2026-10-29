"""Hosted CPU paired F16/true-Q4 experiment. No original intermediate gold exists."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time

from contract import (ROOT, CASES, RECIPES, F16_SHA, F16_BYTES, require, digest, guard,
                      references, inventory, overrides, protected)
from adapter import write_pair, write_gguf, collect

sys.path.insert(0, str(ROOT / 'tools'))
from josev2b_composed_panel import metrics

loader = importlib.util.spec_from_file_location('old_precision_metrics', ROOT / 'tools/kaggle/unlimited-ocr2-precision/precision_worker.py')
old = importlib.util.module_from_spec(loader)
loader.loader.exec_module(old)
DEADLINE = None


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + '\n')


def command(args, directory, env=None, timeout=3600, accepted=(0,)):
    if DEADLINE is not None:
        remaining = DEADLINE - time.monotonic()
        require(remaining > 0, 'Declared panel CPU deadline reached before next command')
        timeout = min(timeout, remaining)
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    timed_out = False
    with (directory / 'stdout').open('wb') as out, (directory / 'stderr').open('wb') as err:
        process = subprocess.Popen(list(map(str, args)), env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    receipt = dict(argv=list(map(str, args)), exit=process.returncode, timed_out=timed_out,
                   wall_seconds=time.monotonic() - start,
                   stdout_sha256=digest(directory / 'stdout'), stderr_sha256=digest(directory / 'stderr'))
    save(directory / 'command.json', receipt)
    require(not timed_out and process.returncode in accepted, 'Hosted command rejected: ' + directory.name)
    return receipt


def parsed(directory):
    text = (directory / 'stdout').read_text(encoding='utf-8')
    require(text.endswith('\n'), 'CLI newline contract')
    text = text[:-1]  # Remove exactly the CLI's one appended newline.
    stderr = (directory / 'stderr').read_text(encoding='utf-8')
    lines = re.findall(r'gen_ids \((\d+)\):([^\n]*)', stderr)
    require(len(lines) == 1, 'Exactly one actual generated-ID trace required')
    count, values = lines[0]
    ids = [int(x) for x in values.split()]
    require(len(ids) == int(count) and 0 < len(ids) < 1024, 'Actual IDs/cap/completion contract')
    # The runtime trace omits EOS. Fewer than1024 outputs establishes no token cap;
    # page-count guarding/EOS is source-derived, not an invented observed EOS ID.
    return dict(ids=ids, text=text, output_count=len(ids), token_cap_reached=False,
                eos_observed=False, stop_contract='source-derived EOS/page guard; generated IDs exclude EOS')


def task_quality(actual, reference):
    delta = old.max_box_delta(actual['text'], reference['text'])
    body = old.strip_boxes(actual['text']) == old.strip_boxes(reference['text'])
    return dict(ids_exact=actual['ids'] == reference['ids'], raw_text_exact=actual['text'] == reference['text'],
                body_exact=body, aligned_box_max_absolute=delta, cer=old.cer(actual['text'], reference['text']),
                body_cer=old.cer(old.strip_boxes(actual['text']), old.strip_boxes(reference['text'])),
                task_completed=not actual['token_cap_reached'] and body and delta is not None and delta <= 3,
                table_body_exact=body)


def stage_axes(name, value):
    # Pinned embedding/head width from the Python inference blueprint, not the
    # smaller number of tokenizer entries or a rounded allocation estimate.
    require(isinstance(value, list) and len(value) == 3 and
            all(type(x) is int and 0 < x <= 129280 for x in value), 'Explicit bounded C-order axes')
    require(math.prod(value) * 4 <= 64 * 1024**2, 'Observed stage exceeds64MiB')
    if name == 'logits':
        require(value == [1, 1, 129280], 'Exact published vocabulary output axes required')
    return value


def stage_manifest(directory):
    rows = {}
    journal = directory / 'stages.jsonl'
    require(journal.is_file(), 'Observed first-prefill journal missing')
    for line in journal.read_text().splitlines():
        row = json.loads(line)
        name = row['name']
        require(name not in rows and re.fullmatch(r'[a-z0-9_]+', name), 'Unique safe stage key required')
        require(Path(row['file']).name == row['file'] and row['type'] in ('f32', 'i32'), 'Safe typed raw member required')
        shape = stage_axes(name, row['shape'])
        path = directory / row['file']
        require(not path.is_symlink() and path.stat().st_size == math.prod(shape) * 4, 'Stage extent differs')
        if row['type'] == 'f32':
            import numpy as np
            require(np.isfinite(np.memmap(path, dtype='<f4', mode='r')).all(), 'Nonfinite observed F32 stage')
        rows[name] = dict(row, path=str(path), sha256=digest(path), bytes=path.stat().st_size)
    # Declared fixed2048 chunk covers each of these four prompts in one chunk.
    expected = {'final_norm', 'logits'} | {f'layer_{layer}_chunk_0' for layer in range(12)}
    expected |= {f'layer_{layer}_chunk_0_{part}' for layer in range(1, 12) for part in ['hn', 'ids', 'gate', 'up']}
    require(set(rows) == expected, 'Exact58 first-prefill stages required (no silent subset)')
    tokens = rows['layer_0_chunk_0']['shape'][0]
    require(tokens <= 2048 and all(rows[f'layer_{layer}_chunk_0']['shape'] == [tokens, 1, 1280]
                                  for layer in range(12)), 'Full12-layer prefill coverage/axes differ')
    require(rows['final_norm']['shape'] == [1, 1, 1280] and
            rows['logits']['shape'] == [1, 1, 129280],
            'Final norm/vocabulary output axes differ')
    for layer in range(1, 12):
        key = f'layer_{layer}_chunk_0'
        n = rows[key + '_hn']['shape'][0]
        require(n == min(tokens, 16) and rows[key + '_hn']['shape'] == [n, 1, 1280] and
                rows[key + '_ids']['shape'] == [n, 1, 6] and
                rows[key + '_gate']['shape'] == [n, 6, 896] and rows[key + '_up']['shape'] == [n, 6, 896],
                'Selected projection axes differ')
    return rows


def compare(pairs, diff, directory):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=False)
    independent = {}
    for pair in pairs:
        a, b = pair['original'], pair['native']
        independent[pair['name']] = metrics(np.memmap(a['path'], dtype='<f4', mode='r', shape=tuple(a['shape'])),
                                             np.memmap(b['path'], dtype='<f4', mode='r', shape=tuple(b['shape'])))
    adapters = write_pair(directory / 'control.gguf', directory / 'candidate.gguf', pairs)
    receipt = command([diff, directory / 'control.gguf', directory / 'candidate.gguf', directory / 'ref.json'],
                      directory / 'command', accepted=(0, 1))
    cpp = collect(receipt['exit'], directory / 'ref.json', independent)
    save(directory / 'comparison.json', dict(cpp=cpp, independent=independent, adapters=adapters,
         reference_kind='same-host F16 control; not original four-page intermediate gold'))
    return dict(passed=all(x['passed'] for x in independent.values()), cpp=cpp, stages=independent,
                first_divergence=next((key for key in independent if not independent[key]['passed']), None),
                worst_absolute=max(x['max_absolute'] for x in independent.values()))


def float_spec(row):
    require(row['type'] == 'f32', 'F32 stage required')
    return {key: row[key] for key in ['path', 'shape', 'sha256']}


def projection_inputs(stages, layer, directory):
    directory.mkdir(parents=True, exist_ok=False)
    h, ids = stages[f'layer_{layer}_chunk_0_hn'], stages[f'layer_{layer}_chunk_0_ids']
    raw = Path(ids['path']).read_bytes()
    values = [x[0] for x in struct.iter_unpack('<i', raw)]
    require(all(0 <= x < 64 for x in values), 'F16 fixed-router IDs outside64 experts')
    f32 = directory / 'ids.f32'
    f32.write_bytes(struct.pack('<' + 'f' * len(values), *values))
    packet = directory / 'input.gguf'
    write_gguf(packet, [dict(name='hn', **float_spec(h)),
                        dict(name='ids', path=str(f32), shape=ids['shape'], sha256=digest(f32))])
    return packet, h['shape'][0]


def main(args):
    global DEADLINE
    DEADLINE = time.monotonic() + args.budget_seconds
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='crispembed-unlimited-ocr2-mixed-q4k-v1', completed=False,
                  production_qualified=False, publication_authorized=False, source=args.source,
                  original_intermediate_gold=False, baseline=[], candidates=[])
    save(out / 'panel.json', report)
    try:
        report['source_hashes'] = guard()
        from contract import git
        require(git('rev-parse', 'HEAD') == args.source, 'Exact declared producer source required')
        refs, images = references()
        report['original_ids_text_only'] = True
        for case in CASES:
            require(digest(Path(args.pages) / (case + '.png')) == images[case], 'Frozen original PNG differs')
        baseline = Path(args.model)
        require(baseline.stat().st_size == F16_BYTES and digest(baseline) == F16_SHA, 'Published F16 model identity differs')
        base_records = inventory(baseline)
        # Reject any latent four-axis F32 flattening before spending a forward.
        overrides(base_records, 'all11')
        report['baseline_inventory'] = base_records
        report['baseline_model'] = dict(bytes=F16_BYTES, sha256=F16_SHA,
            tokenizer_metadata_complete=False, added_piece_count_missing=831,
            added_piece_source='src/unlimited_ocr2_text.h::added_token, IDs128000..128830',
            added_piece_source_sha256=digest(ROOT / 'src/unlimited_ocr2_text.h'))
        report['binaries'] = {key: dict(sha256=digest(value), bytes=Path(value).stat().st_size)
                              for key, value in vars(args).items() if key in ['cli', 'observed', 'diff', 'projection', 'quantizer']}
        env = {k: v for k, v in os.environ.items() if not k.startswith(('CRISPEMBED_', 'UOCR'))}
        env.update(CRISPEMBED_UNLIMITED_OCR2='1', CRISPEMBED_UNLIMITED_OCR2_FAST='1',
                   UOCR_DBG='1', UOCR2_FAST_CHUNK='2048', OMP_NUM_THREADS=str(args.threads),
                   HF_HUB_OFFLINE='1')

        def forward(model, case, mode, parent):
            directory = parent / mode
            trace = parent / (mode + '-trace')
            selected_env = dict(env)
            if mode in ['observed', 'repeat']:
                trace.mkdir(parents=True, exist_ok=False)
                selected_env['UOCR_Q4_AUDIT_DIR'] = str(trace)
                selected_env['UOCR2_DUMP_VISION'] = str(trace / 'assembled-vision.f32')
            binary = args.cli if mode == 'plain' else args.observed
            receipt = command([binary, '-m', model, '--ocr', Path(args.pages) / (case + '.png'),
                               '--ocr-max-tokens', '1024', '-t', str(args.threads)], directory, selected_env, 3600)
            row = parsed(directory)
            row.update(command=receipt, quality=task_quality(row, refs[case]))
            if mode in ['observed', 'repeat']:
                row['stages'] = stage_manifest(trace)
                vision = trace / 'assembled-vision.f32'
                require(vision.is_file() and vision.stat().st_size > 0 and vision.stat().st_size % (1280 * 4) == 0,
                        'Assembled vision extent differs')
                row['vision'] = dict(bytes=vision.stat().st_size, sha256=digest(vision))
            return row

        original_controls = {}
        # All four strict original F16 and observer/repeat controls finish BEFORE any Q4 forward.
        for case in CASES:
            parent = out / 'baseline' / case
            modes = {mode: forward(baseline, case, mode, parent) for mode in ['plain', 'ordinary', 'observed', 'repeat']}
            require(all(r['ids'] == modes['plain']['ids'] and r['text'] == modes['plain']['text'] for r in modes.values()),
                    'F16 ordinary/observer/repeat decoded neutrality failed')
            require(modes['plain']['quality']['ids_exact'] and modes['plain']['quality']['raw_text_exact'] and
                    modes['plain']['quality']['task_completed'], 'First F16 strict original control failed')
            a, b = modes['observed']['stages'], modes['repeat']['stages']
            require(all(a[key]['shape'] == b[key]['shape'] and a[key]['sha256'] == b[key]['sha256'] for key in a),
                    'F16 observed/repeat raw byte neutrality failed')
            require(modes['observed']['vision'] == modes['repeat']['vision'], 'F16 vision repeat bytes differ')
            original_controls[case] = a
            report['baseline'].append(dict(id=case, modes=modes, raw_repeat_bit_exact=True))
            save(out / 'panel.json', report)
        report['f16_all_four_strict_control_passed'] = True

        # Keep every graduated candidate result, including unchanged numerical failures.
        for recipe, layers in RECIPES.items():
            model = out / ('unlimited-' + recipe + '.gguf')
            cmd = [args.quantizer, baseline, model, 'q4_k']
            for rule in overrides(base_records, recipe):
                cmd += ['--tensor-type', rule]
            command(cmd, out / ('quantize-' + recipe), timeout=2400)
            item = dict(recipe=recipe, cases=[], actual_inventory=inventory(model), model_sha256=digest(model))
            item['protected'] = protected(base_records, item['actual_inventory'], recipe)
            report['candidates'].append(item)
            save(out / 'panel.json', report)
            for case in CASES:
                row = dict(id=case, completed=False)
                item['cases'].append(row)
                try:
                    parent = out / recipe / case
                    plain, observed = forward(model, case, 'plain', parent), forward(model, case, 'observed', parent)
                    repeated = forward(model, case, 'repeat', parent)
                    require(plain['ids'] == observed['ids'] and plain['text'] == observed['text'], 'Candidate observer decoded neutrality failed')
                    require(repeated['ids'] == plain['ids'] and repeated['text'] == plain['text'] and
                            all(repeated['stages'][key]['sha256'] == value['sha256'] and
                                repeated['stages'][key]['shape'] == value['shape']
                                for key, value in observed['stages'].items()), 'Candidate repeat byte neutrality failed')
                    row.update(plain=plain, observed=observed, repeated=repeated,
                               ordinary_observer_exact=True, raw_repeat_bit_exact=True)
                    f16_case = next(x for x in report['baseline'] if x['id'] == case)
                    require(observed['vision'] == repeated['vision'] == f16_case['modes']['observed']['vision'],
                            'Protected vision actual output bytes changed')
                    row['protected_vision_bit_exact'] = True
                    expected, actual = original_controls[case], observed['stages']
                    require(set(expected) == set(actual), 'Paired trace coverage differs')
                    row['natural_router_ids_equal_f16'] = {key: expected[key]['sha256'] == actual[key]['sha256']
                                                           for key in expected if expected[key]['type'] == 'i32'}
                    pairs = [dict(name=key, original=float_spec(expected[key]), native=float_spec(actual[key]))
                             for key in expected if expected[key]['type'] == 'f32']
                    row['natural_f16_to_q4'] = compare(pairs, args.diff, parent / 'natural-comparison')
                    row['same_input_projection'] = []
                    for layer in layers:
                        packet, n = projection_inputs(expected, layer, parent / ('input-' + str(layer)))
                        for part in ['gate', 'up']:
                            weight = f'l.blk.{layer}.ffn_{part}_exps.weight'
                            prefix = parent / f'projection-{layer}-{part}'
                            for label, source in [('f16', baseline), ('q4', model)]:
                                command([args.projection, source, weight, packet, str(prefix) + '-' + label,
                                         str(args.threads)], parent / f'operator-{layer}-{part}-{label}', timeout=180)
                            def spec(label):
                                path = Path(str(prefix) + '-' + label + '.f32')
                                return dict(path=str(path), shape=[n, 6, 896], sha256=digest(path))
                            natural = float_spec(expected[f'layer_{layer}_chunk_0_{part}'])
                            measured = compare([
                                dict(name='f16_capture_vs_fixed_f16', original=natural, native=spec('f16.native')),
                                dict(name='fixed_f16_vs_q4', original=spec('f16.native'), native=spec('q4.native')),
                                dict(name='q4_stored_vs_dequant_f32', original=spec('q4.dequant'), native=spec('q4.native')),
                            ], args.diff, parent / f'projection-comparison-{layer}-{part}')
                            row['same_input_projection'].append(dict(weight=weight, fixed_ids_source='F16 observed first16 rows', **measured))
                    row['completed'] = True
                    row['numeric_passed'] = row['natural_f16_to_q4']['passed'] and all(x['passed'] for x in row['same_input_projection'])
                    row['task_quality_passed'] = plain['quality']['task_completed']
                except Exception as error:
                    row['error'] = str(error)
                save(out / 'panel.json', report)
            # Keep all raw outputs and logs. Remove only the task-produced candidate
            # weight after its full panel; baseline and original are never modified.
            item['model_retained'] = False
            model.unlink()
        report['completed'] = all(row['completed'] for item in report['candidates'] for row in item['cases'])
        report['all_numeric_passed'] = report['completed'] and all(row.get('numeric_passed') for item in report['candidates'] for row in item['cases'])
        report['all_task_quality_passed'] = report['completed'] and all(row.get('task_quality_passed') for item in report['candidates'] for row in item['cases'])
    except Exception as error:
        report['error'] = str(error)
    finally:
        # Manifest authenticates every retained report/raw/command file independently.
        report['files'] = {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=digest(p))
                           for p in sorted(out.rglob('*')) if p.is_file() and p != out / 'panel.json'}
        save(out / 'panel.json', report)
    return 0 if report.get('completed') and report.get('all_numeric_passed') and report.get('all_task_quality_passed') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['source', 'output', 'pages', 'model', 'cli', 'observed', 'diff', 'projection', 'quantizer']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--threads', type=int, default=4, choices=range(1, 17))
    parser.add_argument('--budget-seconds', type=int, required=True)
    sys.exit(main(parser.parse_args()))
