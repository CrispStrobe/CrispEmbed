#!/usr/bin/env python3
"""Hosted pure-Python-original-first SystemOne answer test. No packages/weights."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from josev2b_answer_serializer import (BASE, CPYTHON_SUM_SOURCE, REFERENCE_SHA, SOURCE_SHA, canonical, digest,
    f32_value, native_negatives, original_answers, panel, require, rounding_inputs, source_guard, write)
from josev2b_tokenizer_schema import blueprint  # Exact pinned AST declarations, no old source-guard waiver.


def hosted():
    require(os.environ.get('GITHUB_ACTIONS') == 'true', 'hosted answer-only task required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(head == os.environ.get('GITHUB_SHA'), 'answer service/source differs')
    require(platform.python_implementation() == 'CPython' and tuple(sys.version_info[:3]) == (3, 13, 15),
            'exact original CPython3.13.15 required')
    require(sys.float_info.mant_dig == 53 and sys.float_info.rounds == 1, 'binary64 nearest-rounding profile required')
    require(not {'torch', 'transformers', 'tokenizers'} & set(sys.modules), 'no framework/tokenizer imports allowed')
    return source_guard()


def command(args, path, *, env=None, timeout=90):
    with path.open('w') as stream:
        result = subprocess.run(list(map(str, args)), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    require(result.returncode == 0, 'probe command failed: ' + path.name)


def original(out):
    url = 'https://huggingface.co/mayflowergmbh/Josev-2B/resolve/6f0a31c8ed683aba79097eb3f48df2c9ab560afd/joint_schema_model.py'
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(23264)
    require(len(data) == 23263 and digest(data) == SOURCE_SHA, 'original publisher source differs')
    source = out / 'joint_schema_model.py'; source.write_bytes(data)
    module = blueprint(source)
    cases = panel(module)
    answers = original_answers(module, cases)
    write(out / 'review/original-answers.json', answers)
    (out / 'answer-inputs.jsonl').write_text(''.join(canonical(c) + '\n' for c in cases), encoding='utf-8')
    rounds = rounding_inputs()
    round_gold = []
    for row in rounds:
        value = struct.unpack('<d', struct.pack('<Q', int(row['binary64_hex'], 16)))[0]
        require(math.isfinite(value) and 0 <= value <= 8128, 'rounding gold outside declared bounded domain')
        round_gold.append({'id': row['id'], 'rounded_json': canonical(round(value, 4))})
    write(out / 'review/original-rounding.json', round_gold)
    write(out / 'original-gold.json', {'schema': 'crispembed-josev-answer-original-gold-v1',
          'source_sha256': SOURCE_SHA, 'reference_sha256': REFERENCE_SHA, 'python': platform.python_version(),
          'answers': answers, 'rounding': round_gold})
    require(all(a['official_response_equal'] is True for a in answers[:4]), 'original public response differs; actual outputs retained')
    counter = next(c for c in cases if c['id'] == 'score-compensated-counter')
    naive = 0.0
    for i, bits in enumerate(counter['probabilities_u32'][0]):
        naive += i * f32_value(bits)
    counter_response = next(a for a in answers if a['id'] == counter['id'])['answer']['answers']['q']
    require(counter_response['score'] != round(naive, 4), 'owned counterexample did not exercise Python3.13 compensated sum')
    (out / 'round-inputs.jsonl').write_text(''.join(canonical(c) + '\n' for c in rounds), encoding='utf-8')
    write(out / 'review/original.json', {'schema': 'crispembed-josev-answer-original-v1',
          'completed': True, 'source_head': os.environ['GITHUB_SHA'], 'publisher_source_sha256': SOURCE_SHA,
          'reference_sha256': REFERENCE_SHA, 'original_four_official_responses_exact': True, 'original_fields': 7,
          'answer_cases': len(cases), 'rounding_cases': len(rounds),
          'python': platform.python_version(), 'implementation': platform.python_implementation(),
          'binary64_mantissa': sys.float_info.mant_dig, 'rounding_mode': sys.float_info.rounds,
          'cpython_sum_source': CPYTHON_SUM_SOURCE,
          'compensation_counter': {'plain_fold': naive, 'plain_fold_rounded': round(naive, 4),
                                   'original_answer': counter_response},
          'files': {name: {'bytes': (out / name).stat().st_size, 'sha256': digest((out / name).read_bytes())}
                    for name in ('original-gold.json', 'answer-inputs.jsonl', 'round-inputs.jsonl', 'joint_schema_model.py')},
          'model_weights': False, 'tokenizer': False, 'framework': False, 'native_executed': False})


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def native(out, build):
    proof = json.loads((out / 'review/original.json').read_text())
    require(proof['completed'] is True and proof['source_head'] == os.environ['GITHUB_SHA'] and
            proof['publisher_source_sha256'] == SOURCE_SHA and proof['reference_sha256'] == REFERENCE_SHA and
            proof['original_four_official_responses_exact'] is True and proof['original_fields'] == 7 and
            proof['python'] == '3.13.15' and proof['native_executed'] is False and
            proof['model_weights'] is False and proof['framework'] is False and proof['tokenizer'] is False,
            'original-first answer barrier differs')
    require(set(proof['files']) == {'original-gold.json', 'answer-inputs.jsonl', 'round-inputs.jsonl', 'joint_schema_model.py'},
            'original proof file scope differs')
    for name, identity in proof['files'].items():
        require(identity == {'bytes': (out / name).stat().st_size, 'sha256': digest((out / name).read_bytes())},
                'original proof bytes changed: ' + name)
    gold = json.loads((out / 'original-gold.json').read_text())
    require(gold['source_sha256'] == SOURCE_SHA and gold['reference_sha256'] == REFERENCE_SHA and gold['python'] == '3.13.15',
            'gold provenance differs')
    cases = rows(out / 'answer-inputs.jsonl')
    commands = json.loads((build / 'compile_commands.json').read_text())
    arithmetic = [r for r in commands if Path(r['file']).name == 'josev2b_answer_serializer.cpp']
    require(len(arithmetic) == 2 and all('-ffp-contract=off' in r['command'] and
            not any(flag in r['command'] for flag in ('-ffast-math', '-Ofast', '-ffp-contract=fast')) for r in arithmetic),
            'actual ordinary/test arithmetic compiler flags differ')
    write(out / 'review/compile-commands.json', arithmetic)
    binary, plain = build / 'test-josev-answer', build / 'test-josev-answer-plain'
    env = dict(os.environ, CRISPEMBED_JOSEV_ANSWER_SERIALIZER='1')
    command([binary, '--gates', out / 'answer-inputs.jsonl', out / 'unused-gates.jsonl'], out / 'review/gates.log', env=env)
    command([binary, '--answers', out / 'answer-inputs.jsonl', out / 'native-answers.jsonl'], out / 'review/native-answers.log', env=env)
    command([plain, '--answers', out / 'answer-inputs.jsonl', out / 'ordinary-answers.jsonl'], out / 'review/ordinary-answers.log', env=env)
    require((out / 'native-answers.jsonl').read_bytes() == (out / 'ordinary-answers.jsonl').read_bytes(), 'test-export/ordinary answer neutrality failed')
    command([binary, '--round', out / 'round-inputs.jsonl', out / 'native-rounding.jsonl'], out / 'review/native-rounding.log', env=env)
    native_answers, native_rounds = rows(out / 'native-answers.jsonl'), rows(out / 'native-rounding.jsonl')
    failures = []
    require(len(native_answers) == len(gold['answers']) == len(cases), 'native answer count differs')
    require(len(native_rounds) == len(gold['rounding']), 'native round count differs')
    comparisons = []
    for actual, expected in zip(native_answers, gold['answers']):
        semantic = actual.get('accepted') is True and json.loads(actual.get('answer_json', 'null')) == expected['answer']
        exact = actual.get('answer_json') == expected['answer_json']
        comparisons.append({'id': expected['id'], 'semantic_equal': semantic, 'canonical_equal': exact,
                            'original_sha256': digest(expected['answer_json'].encode()),
                            'native_sha256': digest(actual.get('answer_json', '').encode())})
        if actual['id'] != expected['id'] or not semantic or not exact:
            failures.append({'kind': 'answer', 'expected': expected, 'actual': actual})
    for actual, expected in zip(native_rounds, gold['rounding']):
        if actual['id'] != expected['id'] or actual.get('accepted') is not True or actual.get('rounded_json') != expected['rounded_json']:
            failures.append({'kind': 'rounding', 'expected': expected, 'actual': actual})
    write(out / 'review/comparisons.json', comparisons)
    write(out / 'review/native-answers.json', native_answers)
    write(out / 'review/native-rounding.json', native_rounds)
    write(out / 'review/parity-failures.json', failures)
    # Keep every negative between identical valid cases in one immutable context.
    negatives = native_negatives(cases[0]); probes = []
    for negative in negatives:
        before = deepcopy(cases[0]); before['id'] = negative['id'] + '-before'
        after = deepcopy(cases[0]); after['id'] = negative['id'] + '-after'
        probes += [before, negative, after]
    (out / 'negative-inputs.jsonl').write_text(''.join(canonical(c) + '\n' for c in probes), encoding='utf-8')
    command([binary, '--answers', out / 'negative-inputs.jsonl', out / 'native-negatives.jsonl'], out / 'review/negative-recovery.log', env=env)
    negative_rows = rows(out / 'native-negatives.jsonl')
    write(out / 'review/negative-recovery.json', negative_rows)
    require(len(negative_rows) == len(probes), 'negative/recovery count differs')
    for i, negative in enumerate(negatives):
        before, bad, after = negative_rows[3*i:3*i+3]
        require(before['accepted'] is True and after['accepted'] is True and
                before['answer_json'] == after['answer_json'] == gold['answers'][0]['answer_json'],
                'answer context recovery failed: ' + negative['id'])
        require(bad['id'] == negative['id'] and bad['accepted'] is False and bad.get('output_empty') is True and
                bad.get('error') and 'probe_error' not in bad, 'negative did not exercise actual serializer: ' + negative['id'])
    symbols = subprocess.check_output(['nm', '-C', str(build / 'libjosev-answer-plain.a')], text=True)
    require('josev2b::answer_test::' not in symbols, 'test helper leaked into ordinary archive')
    ordinary_strings = subprocess.check_output(['strings', str(build / 'libjosev-answer-plain.a')], text=True)
    require('answer_test' not in ordinary_strings and 'CRISPEMBED_JOSEV_ANSWER_TESTING' not in ordinary_strings, 'diagnostic strings leaked into ordinary archive')
    tested_symbols = subprocess.check_output(['nm', '-C', str(build / 'libjosev-answer-tested.a')], text=True)
    require('josev2b::answer_test::rounded4' in tested_symbols, 'test symbol positive control absent')
    (out / 'review/ordinary-symbols.txt').write_text(symbols)
    write(out / 'review/native.json', {'schema': 'crispembed-josev-answer-native-v1', 'completed': True,
          'answer_serializer_parity': not failures, 'failures': failures, 'comparisons': comparisons,
          'answer_cases': len(cases), 'rounding_cases': len(native_rounds), 'negative_cases': len(negatives),
          'ordinary_test_neutrality': True, 'ordinary_excludes_test_symbols': True,
          'source_head': os.environ['GITHUB_SHA'], 'reference_sha256': REFERENCE_SHA,
          'files': {str(p.relative_to(out)): {'sha256': digest(p.read_bytes()), 'bytes': p.stat().st_size}
                    for p in (out / 'native-answers.jsonl', out / 'ordinary-answers.jsonl', out / 'native-rounding.jsonl', out / 'native-negatives.jsonl')},
          'compiled_files': {p.name: {'sha256': digest(p.read_bytes()), 'bytes': p.stat().st_size}
                             for p in (binary, plain, build / 'libjosev-answer-plain.a', build / 'libjosev-answer-tested.a')},
          'default_enabled': False, 'model_weights': False, 'tokenizer': False,
          'composed_model_parity': False, 'release_qualified': False})
    require(not failures, 'answer/rounding parity failed; all comparisons retained')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['preflight', 'original', 'native'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build', type=Path)
    args = parser.parse_args(); out = args.output.resolve(); (out / 'review').mkdir(parents=True, exist_ok=True)
    try:
        cfg = hosted()
        write(out / 'review/preflight.json', {'source_head': os.environ['GITHUB_SHA'], 'base': BASE, 'source_sha256': cfg['source_sha256'], 'model_weights': False})
        if args.phase == 'original': original(out)
        if args.phase == 'native':
            require(args.build is not None, 'native build path required'); native(out, args.build.resolve())
    except Exception as error:
        write(out / 'review' / ('rejection-' + args.phase + '.json'), {'phase': args.phase, 'error_type': type(error).__name__,
              'error': str(error), 'model_weights': False, 'tokenizer': False, 'release_qualified': False})
        raise


if __name__ == '__main__':
    main()
