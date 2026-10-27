#!/usr/bin/env python3
"""Internal answer-only fixture/source utilities; stdlib, no model/tokenizer."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'f424ec52880d5b289e4d9da8d80373e3dd1ff67a'
SOURCE_SHA = '0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3'
REFERENCE_SHA = '51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090'
FREEZE_SHA = '9c85fae56186517b5721be547bc74e8f92bd8e7cdb45d38afb4340e029161fe7'
CPYTHON_SUM_SOURCE = {'url': 'https://raw.githubusercontent.com/python/cpython/v3.13.15/Python/bltinmodule.c',
                      'bytes': 95949, 'sha256': '087395fe280641e8633a39a289fa50a0404cedd6d0d9a8074eb7c6baa91407b3'}
CONTRACT = ROOT / 'docs/josev2b-answer-serializer.json'
LANE_PATHS = {
    'src/josev2b_answer_serializer.h', 'src/josev2b_answer_serializer.cpp',
    'tests/test_josev2b_answer.cpp', 'tests/josev-answer/CMakeLists.txt',
    'tools/josev2b_answer_serializer.py', 'tools/ci-heavy/josev2b_answer_serializer.py',
    'tests/test_josev2b_answer_serializer.py', 'tests/fixtures/josev2b-answer-serializer-v1.json',
    'docs/josev2b-answer-serializer.md', '.github/workflows/josev2b-answer-serializer.yml',
    '.github/workflows/ovisocr2-metadata.yml'}
OLD_PATHS = {'CMakeLists.txt', '.gitmodules', 'cmake', 'python', 'crispembed', 'crispembed-sys', 'flutter', 'wasm',
             'Cargo.toml', 'Cargo.lock', 'examples', 'tests/regression',
             'tools/josev2b_contract.py', 'docs/josev2b-source-contract.json',
             'tools/josev2b_tokenizer_schema.py', 'tools/ci-heavy/josev2b_tokenizer_schema.py',
             'docs/josev2b-tokenizer-schema.json', 'docs/josev2b-tokenizer-schema.md',
             'docs/josev2b-tokenizer-schema-LICENSE.txt', 'tests/josev-text',
             'tests/test_josev2b_text.cpp', 'tests/test_josev2b_tokenizer_schema.py',
             'tests/fixtures/josev2b-tokenizer-schema-v1.json', '.github/workflows/josev2b-tokenizer-schema.yml',
             'tests/results/josev2b/original-reference.json', 'tests/results/josev2b/original-input-freeze.json'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def old_src():
    return {line.split('\t')[1] for line in git('ls-tree', '-r', BASE, '--', 'src').splitlines()}


def source_guard():
    cfg = json.loads(CONTRACT.read_text())
    require(cfg['schema'] == 'crispembed-josev-answer-serializer-v1' and cfg['base'] == BASE and
            cfg['default_enabled'] is False and cfg['reference_sha256'] == REFERENCE_SHA and
            cfg['input_freeze_sha256'] == FREEZE_SHA and cfg['python'] == '3.13.15' and
            cfg['cpython_sum_source'] == CPYTHON_SUM_SOURCE and cfg['model_weights'] is False,
            'answer source/contract policy differs')
    baseline = old_src()
    require(set(cfg['protected_objects']) == OLD_PATHS | baseline, 'old source closure omitted')
    current = {line.split('\t')[1] for line in git('ls-tree', '-r', 'HEAD', '--', 'src').splitlines()}
    require(current == baseline | {'src/josev2b_answer_serializer.h', 'src/josev2b_answer_serializer.cpp'}, 'unexpected native source delta')
    for name, expected in cfg['protected_objects'].items():
        require(git('rev-parse', 'HEAD:' + name) == expected == git('rev-parse', BASE + ':' + name), 'old source changed: ' + name)
    require(set(cfg['source_sha256']) == LANE_PATHS, 'answer source closure omitted')
    for name, expected in cfg['source_sha256'].items():
        require(digest((ROOT / name).read_bytes()) == expected ==
                digest(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)), 'answer source bytes differ: ' + name)
    require(not git('status', '--porcelain', '--', *sorted(OLD_PATHS | baseline | LANE_PATHS | {'docs/josev2b-answer-serializer.json'})),
            'dirty protected/answer source')
    return cfg


def f32_bits(value):
    if isinstance(value, str):
        value = float.fromhex(value) if value.lstrip('-').startswith('0x') else float(value)
    return struct.unpack('<I', struct.pack('<f', value))[0]


def f32_value(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def lexical(encoded):
    value = deepcopy(encoded)
    for q in value['questions']:
        q['lexical_ids'] = [value['input_ids'][a:b] for a, b in q['option_spans']]
    return value


def synthetic_encoded(request, module, tokens=None):
    """Authored logical metadata ONLY, never claimed as tokenizer output."""
    questions, position = [], 1
    for key, question in request['questions'].items():
        q = {'question_id': key, 'question_type': module.QUESTION_TYPES[question['type']],
             'question_span': [position, position + 1], 'option_ids': [], 'option_spans': []}
        position += 1
        for option_id, _ in module.question_options(question):
            q['option_ids'].append(option_id); q['option_spans'].append([position, position + 1]); position += 1
        questions.append(q)
    length = position if tokens is None else tokens
    require(position <= length <= 512, 'synthetic metadata bounds')
    return lexical({'input_ids': list(range(length)), 'attention_mask': [1] * length,
                    'questions': questions, 'record_id': str(request.get('id', 'unknown')), 'media': None})


def panel(module):
    require(digest((ROOT / 'tests/results/josev2b/original-reference.json').read_bytes()) == REFERENCE_SHA and
            digest((ROOT / 'tests/results/josev2b/original-input-freeze.json').read_bytes()) == FREEZE_SHA,
            'original frozen bytes differ')
    reference = json.loads((ROOT / 'tests/results/josev2b/original-reference.json').read_text())
    frozen = json.loads((ROOT / 'tests/results/josev2b/original-input-freeze.json').read_text())
    result = []
    field_count = 0
    for record, old in zip(frozen['records'], reference['cases']):
        require(record['id'] == old['id'] and len(record['encoded']['input_ids']) == old['tokens'], 'frozen request ordering/count differs')
        encoded = lexical(dict(record['encoded'], attention_mask=record['attention_mask']))
        bits = []
        for q, field in zip(encoded['questions'], old['fields']):
            require(q['question_id'] == field['id'] and q['option_ids'] == field['option_ids'], 'original field alignment differs')
            row = [f32_bits(p) for p in field['probabilities']]
            require([f32_value(p) for p in row] == field['probabilities'], 'published probability is not exact F32')
            bits.append(row); field_count += 1
        require(len(bits) == len(encoded['questions']) == len(old['fields']), 'original field count differs')
        result.append({'id': record['id'], 'request': canonical(record['request']), 'encoded': encoded,
                       'probabilities_u32': bits, 'official_response': old['official_response'], 'metadata_source': 'frozen-original51'})
    require(len(result) == 4 and field_count == 7, 'expected four originals/seven fields')
    fixture = json.loads((ROOT / 'tests/fixtures/josev2b-answer-serializer-v1.json').read_text())
    for case in fixture['cases']:
        request_text = case.get('request_json', canonical(case.get('request')))
        request = json.loads(request_text)
        encoded = synthetic_encoded(request, module, case.get('tokens'))
        bits = []
        for q in encoded['questions']:
            if case.get('probability_recipe') == 'compensated-positive-64':
                require(len(q['option_ids']) == 64, 'compensated recipe width')
                values = ['0.96875', '0.03125'] + ['0x1p-66'] * 62
            else:
                values = [case['probabilities'][q['question_id']][key] for key in q['option_ids']]
            bits.append([f32_bits(v) for v in values])
        result.append({'id': case['id'], 'request': request_text, 'encoded': encoded, 'probabilities_u32': bits,
                       'metadata_source': 'authored logical spans; not real tokenization'})
    require(len({r['id'] for r in result}) == len(result), 'duplicate answer fixture ID')
    return result


def original_answers(module, cases):
    result = []
    for case in cases:
        request = json.loads(case['request'])
        answers = {}
        for field, bits in zip(case['encoded']['questions'], case['probabilities_u32']):
            values = [f32_value(v) for v in bits]
            require(len(values) == len(field['option_ids']) and all(math.isfinite(p) and 0 <= p <= 1 for p in values), 'original manual F32 profile')
            answers[field['question_id']] = module.systemone_answer(request['questions'][field['question_id']],
                                                                   dict(zip(field['option_ids'], values)))
        response = {'model': request['model'], 'answers': answers,
                    'usage': {'input_tokens': len(case['encoded']['input_ids']), 'output_tokens': 0}}
        result.append({'id': case['id'], 'answer_json': canonical(response), 'answer': response,
                       'official_response_equal': canonical(response) == canonical(case['official_response']) if 'official_response' in case else None})
    return result


def rounding_inputs():
    fixture = json.loads((ROOT / 'tests/fixtures/josev2b-answer-serializer-v1.json').read_text())
    values = [float(x) for x in fixture['rounding']['decimal_values']]
    values += [float.fromhex(x) for x in fixture['rounding']['hex_values']]
    # Declared exact rational midpoints plus adjacent binary64 values. Python
    # conversion is part of the frozen original gold; it is never a CPP oracle.
    for q in fixture['rounding']['scaled_midpoint_integers']:
        value = (q + 0.5) / 10000
        values += [math.nextafter(value, 0), value, math.nextafter(value, math.inf)]
    return [{'id': 'round-' + str(i), 'binary64_hex': format(struct.unpack('<Q', struct.pack('<d', x))[0], '016x')}
            for i, x in enumerate(values)]


def native_negatives(valid):
    """Typed invalid metadata/inputs, preserving original fixtures separately."""
    result = []
    def add(name, mutate):
        case = deepcopy(valid); case.pop('official_response', None); case['id'] = name; mutate(case); result.append(case)
    def request(case, mutate):
        value = json.loads(case['request']); mutate(value); case['request'] = canonical(value)
    add('missing-model', lambda c: request(c, lambda v: v.pop('model')))
    add('missing-state', lambda c: request(c, lambda v: v.pop('state')))
    add('model-type', lambda c: request(c, lambda v: v.update(model=1)))
    add('unknown-request-key', lambda c: request(c, lambda v: v.update(images=[])))
    add('unknown-question-kind', lambda c: request(c, lambda v: v['questions'][next(iter(v['questions']))].update(type='unknown')))
    add('unknown-question-key', lambda c: request(c, lambda v: v['questions'][next(iter(v['questions']))].update(other=True)))
    add('field-count', lambda c: c['encoded']['questions'].pop())
    add('field-id', lambda c: c['encoded']['questions'][0].update(question_id='wrong'))
    add('field-type', lambda c: c['encoded']['questions'][0].update(question_type=2))
    add('option-id', lambda c: c['encoded']['questions'][0]['option_ids'].__setitem__(0, 'wrong'))
    add('probability-count', lambda c: c['probabilities_u32'][0].pop())
    for name, bits in [('nan-probability', 0x7fc00000), ('infinite-probability', 0x7f800000),
                       ('negative-probability', 0xbf000000), ('over-one-probability', 0x3f800001)]:
        add(name, lambda c, b=bits: c['probabilities_u32'][0].__setitem__(0, b))
    add('mask-value', lambda c: c['encoded']['attention_mask'].__setitem__(0, 0))
    add('mask-count', lambda c: c['encoded']['attention_mask'].pop())
    add('token-bounds', lambda c: c['encoded']['input_ids'].__setitem__(0, -1))
    add('token-overflow', lambda c: c['encoded'].update(input_ids=[1]*513, attention_mask=[1]*513))
    add('span-order', lambda c: c['encoded']['questions'][0]['question_span'].__setitem__(0, -1))
    add('lexical-content', lambda c: c['encoded']['questions'][0]['lexical_ids'][0].__setitem__(0, -1))
    add('record-id', lambda c: c['encoded'].update(record_id='wrong'))
    add('invalid-escape', lambda c: c.update(request='{"model":"\\q","state":null,"questions":{}}'))
    add('nonfinite-request', lambda c: c.update(request='{"model":"x","state":1e999,"questions":{}}'))
    return result
