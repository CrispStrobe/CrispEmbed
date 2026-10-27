#!/usr/bin/env python3
"""Pinned pure Josev serializer and metadata-only component; stdlib imports only."""
from __future__ import annotations
import ast
import dataclasses
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / 'docs/josev2b-tokenizer-schema.json'
BASE = 'c1e7d30049de22ef6eb04b6849c132601bb32865'
REVISION = '6f0a31c8ed683aba79097eb3f48df2c9ab560afd'
REPOSITORY = 'mayflowergmbh/Josev-2B'
SOURCE_SHA = '0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3'
TOKENIZER_SHA = '87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4'
CONFIG_SHA = '50470511708577d43f7c18c2d8e2ab02fb8abf3add1c40f60b74579d6d16e27e'
REFERENCE_SHA = '51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090'
FREEZE_SHA = '9c85fae56186517b5721be547bc74e8f92bd8e7cdb45d38afb4340e029161fe7'
PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
LANE_SOURCES = {
    'src/josev2b_text_frontend.h', 'src/josev2b_text_frontend.cpp',
    'tests/test_josev2b_text.cpp', 'tests/josev-text/CMakeLists.txt',
    'tools/josev2b_tokenizer_schema.py', 'tools/ci-heavy/josev2b_tokenizer_schema.py',
    'tests/test_josev2b_tokenizer_schema.py', 'tests/fixtures/josev2b-tokenizer-schema-v1.json',
    '.github/workflows/josev2b-tokenizer-schema.yml', '.github/workflows/ovisocr2-metadata.yml',
    'docs/josev2b-tokenizer-schema.md', 'docs/josev2b-tokenizer-schema-LICENSE.txt'}
PROTECTED_ROOTS = {'CMakeLists.txt', '.gitmodules', 'cmake', 'python', 'crispembed', 'crispembed-sys',
                   'flutter', 'wasm', 'Cargo.toml', 'Cargo.lock', 'examples',
                   'tools/josev2b_contract.py', 'docs/josev2b-source-contract.json',
                   'docs/josev2b-environment-lock.json', 'tests/regression',
                   'tests/results/josev2b/original-reference.json',
                   'tests/results/josev2b/original-input-freeze.json'}
PURE_DECLARATIONS = {'EncodedQuestion', 'EncodedRecord', 'render', 'question_options', '_tokens', '_encode_media', 'encode_record', 'systemone_answer'}
PURE_CONSTANTS = {'SYSTEM_PROMPT', 'IMAGE_PLACEHOLDER', 'VIDEO_PLACEHOLDER', 'MEDIA_BATCH_KEYS', 'MEDIA_TOKEN_KEYS', 'QUESTION_TYPES'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def source_guard(root=ROOT):
    cfg = json.loads((root / 'docs/josev2b-tokenizer-schema.json').read_text())
    require(cfg['schema'] == 'crispembed-josev-tokenizer-schema-v1' and cfg['default_enabled'] is False and
            cfg['model_weights'] is False and cfg['max_length'] == 512 and cfg['reference_sha256'] == REFERENCE_SHA,
            'lane policy changed')
    require(cfg['base'] == BASE, 'base source differs')
    baseline_sources = {line.split('\t', 1)[1] for line in git(root, 'ls-tree', '-r', BASE, '--', 'src').splitlines()}
    require(set(cfg['protected_objects']) == PROTECTED_ROOTS | baseline_sources, 'protected source closure omitted/changed')
    current_sources = {line.split('\t', 1)[1] for line in git(root, 'ls-tree', '-r', 'HEAD', '--', 'src').splitlines()}
    require(current_sources == baseline_sources | {'src/josev2b_text_frontend.h', 'src/josev2b_text_frontend.cpp'}, 'unexpected native source delta')
    require(set(cfg['source_sha256']) == LANE_SOURCES, 'lane source closure omitted/changed')
    for name, expected in cfg['protected_objects'].items():
        require(git(root, 'rev-parse', 'HEAD:' + name) == expected == git(root, 'rev-parse', cfg['base'] + ':' + name),
                'old production/source object differs: ' + name)
    require(git(root, 'ls-tree', 'HEAD', 'ggml').split()[2] == cfg['ggml'], 'GGML gitlink differs')
    for name, expected in cfg['source_sha256'].items():
        require(digest((root / name).read_bytes()) == expected and
                digest(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=root)) == expected,
                'lane source differs: ' + name)
    checked_paths = list(cfg['source_sha256']) + list(cfg['protected_objects']) + ['docs/josev2b-tokenizer-schema.json']
    require(not git(root, 'status', '--porcelain', '--', *checked_paths), 'dirty protected source')
    return cfg


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def blueprint(source):
    """Only declarations already audited by inspect_serialization; no torch import."""
    from josev2b_contract import inspect_serialization
    require(digest(source.read_bytes()) == SOURCE_SHA, 'publisher serializer source differs')
    inspect_serialization(source)  # Reuse unchanged upstream-semantics controls.
    tree = ast.parse(source.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if
                isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in PURE_DECLARATIONS or
                isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in PURE_CONSTANTS for t in node.targets)]
    require({n.name for n in selected if isinstance(n, (ast.FunctionDef, ast.ClassDef))} == PURE_DECLARATIONS,
            'pure declaration closure differs')
    module = types.ModuleType('_crisp_josev_tokenizer_blueprint')
    module.__dict__.update(json=json, dataclass=dataclasses.dataclass, dataclass_field=dataclasses.field)
    sys.modules[module.__name__] = module
    try:
        exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body + selected, type_ignores=[]),
                     'pinned-josev-serializer', 'exec'), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    return module


def encoded_dict(encoded):
    value = json.loads(dumps(dataclasses.asdict(encoded)))
    value['attention_mask'] = [1] * len(value['input_ids'])
    for q in value['questions']:
        q['lexical_ids'] = [value['input_ids'][a:b] for a, b in q['option_spans']]
    return value


def frozen_match(rows, frozen):
    require(len(rows) == len(frozen['records']) == 4 and frozen['max_length'] == 512, 'frozen case scope differs')
    for row, old in zip(rows, frozen['records']):
        expected = dict(old['encoded'], attention_mask=old['attention_mask'])
        for q in expected['questions']:
            q['lexical_ids'] = [expected['input_ids'][a:b] for a, b in q['option_spans']]
        require(row['id'] == old['id'] and row['accepted'] is True and row['encoded'] == expected,
                'original tokenizer-only reproduction differs from frozen original: ' + old['id'])


def component_metadata(tokenizer, config):
    require(tokenizer['normalizer'] == {'type': 'NFC'}, 'unsupported original normalizer')
    pre = tokenizer['pre_tokenizer']
    require(pre == {'type': 'Sequence', 'pretokenizers': [
        {'type': 'Split', 'pattern': {'Regex': PATTERN}, 'behavior': 'Isolated', 'invert': False},
        {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': True, 'use_regex': False}]},
        'unsupported original pretokenizer')
    require(tokenizer.get('padding') is None and tokenizer.get('truncation') is None, 'unexpected backend padding/truncation')
    require(config['pretokenize_regex'] == PATTERN and config['split_special_tokens'] is False and
            config['tokenizer_class'] == 'Qwen2Tokenizer', 'tokenizer config policy differs')
    model = tokenizer['model']
    require(model['type'] == 'BPE' and all(model.get(k) in (None, '') for k in
            ('dropout', 'unk_token', 'continuing_subword_prefix', 'end_of_word_suffix')) and
            not any(model.get(k, False) for k in ('fuse_unk', 'byte_fallback', 'ignore_merges')), 'unsupported BPE flags')
    pairs = dict(model['vocab'])
    require(all(isinstance(k, str) and type(v) is int and v >= 0 for k, v in pairs.items()), 'invalid vocab')
    added = tokenizer['added_tokens']
    require(len(added) == 33 and {str(a['id']): {k: v for k, v in a.items() if k != 'id'} for a in added} ==
            config['added_tokens_decoder'], 'added token config/source differs')
    added_ids, flags = [], []
    for item in added:
        require(set(item) == {'id', 'content', 'single_word', 'lstrip', 'rstrip', 'normalized', 'special'} and
                all(item[k] is False for k in ('single_word', 'lstrip', 'rstrip', 'normalized')) and
                type(item['special']) is bool and type(item['id']) is int and item['id'] >= 0 and item['content'],
                'unsupported added-token flag')
        require(item['content'] not in pairs or pairs[item['content']] == item['id'], 'added ID collision')
        pairs[item['content']] = item['id']; added_ids.append(item['id']); flags.append(16 if item['special'] else 0)
    require(len(set(pairs.values())) == len(pairs) and len(set(added_ids)) == len(added_ids), 'duplicate token IDs')
    maximum = max(pairs.values())
    require(200000 <= maximum + 1 <= 250000, 'vocab bounds')
    vocab = [''] * (maximum + 1)
    for piece, index in pairs.items():
        vocab[index] = piece
    merges = model['merges']
    if merges and isinstance(merges[0], list):
        require(all(len(m) == 2 and all(isinstance(s, str) and ' ' not in s for s in m) for m in merges), 'invalid merge pair')
        merges = [' '.join(m) for m in merges]
    require(merges and len(merges) < 1000000 and len(set(merges)) == len(merges) and
            all(isinstance(m, str) and m.count(' ') == 1 for m in merges), 'invalid merges')
    values = {
        'general.architecture': 'josev2b_tokenizer',
        'general.license': 'apache-2.0',
        'general.source.huggingface.repository': REPOSITORY,
        'general.source.huggingface.revision': REVISION,
        'josev2b.tokenizer.format': 'crispembed-josev-tokenizer-v1',
        'josev2b.tokenizer.source_sha256': TOKENIZER_SHA,
        'josev2b.tokenizer.config_sha256': CONFIG_SHA,
        'josev2b.serializer.source_sha256': SOURCE_SHA,
        'josev2b.tokenizer.normalizer': 'NFC',
        'josev2b.tokenizer.pattern': PATTERN,
        'tokenizer.ggml.tokens': vocab, 'tokenizer.ggml.merges': merges,
        'josev2b.tokenizer.added_ids': added_ids, 'josev2b.tokenizer.added_flags': flags,
        # Preserve full immutable descriptors, including unused decoder/postprocessor.
        'josev2b.tokenizer.descriptor_json': dumps({k: v for k, v in tokenizer.items() if k != 'model'}),
        'josev2b.tokenizer.config_json': dumps(config),
    }
    return values


def gguf_bytes(metadata):
    """GGUFv3 metadata only; no numpy, gguf package, tensors, or weight arrays."""
    def string(s):
        b = s.encode('utf-8'); return struct.pack('<Q', len(b)) + b
    out = bytearray(b'GGUF' + struct.pack('<IQQ', 3, 0, len(metadata)))
    for name, value in metadata.items():
        out += string(name)
        if isinstance(value, str):
            out += struct.pack('<I', 8) + string(value)
        elif isinstance(value, list) and value:
            kind = 8 if isinstance(value[0], str) else 5
            out += struct.pack('<IIQ', 9, kind, len(value))
            for element in value:
                out += string(element) if kind == 8 else struct.pack('<i', element)
        else:
            raise ValueError('unsupported/empty GGUF metadata')
    out += b'\0' * (-len(out) % 32)
    return bytes(out)


def panel_inputs():
    panel = json.loads((ROOT / 'tests/regression/josev2b/text-v1.json').read_text())
    owned = [{'id': c['id'], 'request': dumps(c['request']), 'max_length': 512, 'expect': 'accept', 'frozen': True}
             for c in panel['cases']]
    extra = json.loads((ROOT / 'tests/fixtures/josev2b-tokenizer-schema-v1.json').read_text())['cases']
    require(len({c['id'] for c in owned + extra}) == len(owned + extra), 'duplicate panel IDs')
    return owned + extra


def capture_original(module, tokenizer, cases):
    result = []
    for case in cases:
        try:
            request = json.loads(case['request'], parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
            value = encoded_dict(module.encode_record(tokenizer, request, max_length=case['max_length']))
            row = {'id': case['id'], 'accepted': True, 'encoded': value}
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            row = {'id': case['id'], 'accepted': False, 'error_type': type(error).__name__, 'error': str(error)}
        row['expectation_matched'] = row['accepted'] == (case['expect'] == 'accept')
        if not row['accepted'] and case['expect'] == 'reject':
            if case['id'].startswith('schema-overflow'):
                row['expectation_matched'] &= row.get('error_type') == 'ValueError' and row.get('error', '').startswith('schema requires ')
            elif case['id'].startswith('malformed-'):
                row['expectation_matched'] &= row.get('error_type') == 'JSONDecodeError'
        result.append(row)
    return result
