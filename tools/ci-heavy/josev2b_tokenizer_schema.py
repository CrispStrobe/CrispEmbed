#!/usr/bin/env python3
"""Hosted original-first tokenizer/schema proof. Never downloads inference weights."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from josev2b_tokenizer_schema import (CONFIG_SHA, FREEZE_SHA, REFERENCE_SHA, REPOSITORY, REVISION,
    SOURCE_SHA, TOKENIZER_SHA, blueprint, capture_original, component_metadata, digest, dumps,
    frozen_match, gguf_bytes, panel_inputs, require, source_guard, write)


def hosted():
    require(os.environ.get('GITHUB_ACTIONS') == 'true', 'GitHub hosted tokenizer-only task required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(head == os.environ.get('GITHUB_SHA'), 'service/checkout source differs')
    return source_guard()


def fetch(name, expected, target):
    require(name in {'joint_schema_model.py', 'tokenizer.json', 'tokenizer_config.json', 'LICENSE'}, 'payload not permitted')
    require(0 < expected['bytes'] <= 20 * 1024**2, 'source metadata budget')
    url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(expected['bytes'] + 1)
    require(len(data) == expected['bytes'] and digest(data) == expected['sha256'], 'source payload differs: ' + name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)
    return {'bytes': len(data), 'sha256': digest(data)}


def original(out, cfg):
    require('torch' not in sys.modules and 'transformers' not in sys.modules, 'framework imports are prohibited')
    expected_prefix = os.environ.get('CRISPEMBED_JOSEV_TEXT_VENV')
    require(expected_prefix and Path(sys.prefix).resolve() == Path(expected_prefix).resolve() and
            sys.prefix != sys.base_prefix, 'isolated tokenizer-only venv required')
    require(importlib.metadata.version('tokenizers') == '0.22.2', 'tokenizers version differs')
    import tokenizers
    require(Path(tokenizers.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), 'tokenizer import escaped venv')
    tokenizers_contract = json.loads((ROOT / 'docs/josev2b-source-contract.json').read_text())
    source_dir = out / 'source'
    proof = {name: fetch(name, tokenizers_contract['files'][name], source_dir / name)
             for name in ('joint_schema_model.py', 'tokenizer_config.json', 'tokenizer.json', 'LICENSE')}
    backend = tokenizers.Tokenizer.from_file(str(source_dir / 'tokenizer.json'))
    # Original uses add_special_tokens=False for EVERY fragment; no chat template.
    class Adapter:
        def __call__(self, text, add_special_tokens=False):
            require(add_special_tokens is False, 'original added-special policy changed')
            return types.SimpleNamespace(input_ids=backend.encode(text, add_special_tokens=False).ids)
    module = blueprint(source_dir / 'joint_schema_model.py')
    cases = panel_inputs()
    rows = capture_original(module, Adapter(), cases[:4])
    write(out / 'review/original-four.json', {'frozen_four_exact': False, 'cases': rows})
    frozen_path = ROOT / 'tests/results/josev2b/original-input-freeze.json'
    reference_path = ROOT / 'tests/results/josev2b/original-reference.json'
    require(digest(frozen_path.read_bytes()) == FREEZE_SHA and digest(reference_path.read_bytes()) == REFERENCE_SHA,
            'frozen GPU source/reference bytes differ')
    frozen_match(rows, json.loads(frozen_path.read_text()))
    write(out / 'review/original-four.json', {'frozen_four_exact': True, 'cases': rows})
    rows += capture_original(module, Adapter(), cases[4:])
    # Freeze real Rust gold BEFORE conversion/build/native execution; preserve failures.
    write(out / 'original-gold.json', {'schema': 'crispembed-josev-tokenizer-original-gold-v1',
          'source_sha256': SOURCE_SHA, 'tokenizer_sha256': TOKENIZER_SHA,
          'backend': 'tokenizers0.22.2 Rust; add_special_tokens=False', 'reference_sha256': REFERENCE_SHA,
          'frozen_four_exact': True, 'cases': rows})
    require(all(row['expectation_matched'] for row in rows), 'authored original expectation differs; actual gold retained')
    descriptor = json.loads((source_dir / 'tokenizer.json').read_text())
    config = json.loads((source_dir / 'tokenizer_config.json').read_text())
    # Preserve authenticated descriptor semantics even when strict metadata conversion rejects.
    write(out / 'review/tokenizer-policy.json', {
          'tokenizer_sha256': TOKENIZER_SHA, 'config_sha256': CONFIG_SHA,
          'normalizer': descriptor['normalizer'], 'pre_tokenizer': descriptor['pre_tokenizer'],
          'pretokenize_regex': config.get('pretokenize_regex'),
          'post_processor': descriptor['post_processor'], 'decoder': descriptor['decoder']})
    metadata = component_metadata(descriptor, config)
    component = out / 'josev2b-tokenizer.gguf'
    component.write_bytes(gguf_bytes(metadata))
    write(out / 'review/tokenizer-descriptor.json', {
          'normalizer': descriptor['normalizer'], 'pre_tokenizer': descriptor['pre_tokenizer'],
          'post_processor': descriptor['post_processor'], 'decoder': descriptor['decoder'],
          'model_flags': {k: v for k, v in descriptor['model'].items() if k not in ('vocab', 'merges')},
          'vocab_size': len(metadata['tokenizer.ggml.tokens']), 'merges': len(metadata['tokenizer.ggml.merges']),
          'added_tokens': descriptor['added_tokens'], 'config_sha256': CONFIG_SHA})
    files = {name: {'sha256': digest((out / name).read_bytes()), 'bytes': (out / name).stat().st_size}
             for name in ('original-gold.json', 'josev2b-tokenizer.gguf', 'review/tokenizer-descriptor.json', 'review/original-four.json')}
    write(out / 'review/original.json', {'schema': 'crispembed-josev-tokenizer-original-v1',
          'completed': True, 'frozen_four_exact': True, 'cases': len(rows), 'files': files,
          'source_payloads': proof, 'source_head': os.environ['GITHUB_SHA'], 'reference_sha256': REFERENCE_SHA,
          'framework_imports': False, 'model_weights': False, 'native_executed': False,
          'venv_isolated': True, 'tokenizers': tokenizers.__version__})


def command(args, log, env=None, expected=0, timeout=90):
    with log.open('w') as stream:
        result = subprocess.run(list(map(str, args)), cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout)
    require(result.returncode == expected, 'command returned ' + str(result.returncode) + ': ' + log.name)


def native(out, cfg, build):
    summary = json.loads((out / 'review/original.json').read_text())
    require(summary['completed'] is True and summary['frozen_four_exact'] is True and
            summary['source_head'] == os.environ['GITHUB_SHA'] and summary['reference_sha256'] == REFERENCE_SHA and
            summary['model_weights'] is False and summary['native_executed'] is False,
            'original-first barrier differs')
    require(set(summary['files']) == {'original-gold.json', 'josev2b-tokenizer.gguf', 'review/tokenizer-descriptor.json', 'review/original-four.json'},
            'original proof file scope differs')
    for name, identity in summary['files'].items():
        require({'bytes': (out / name).stat().st_size, 'sha256': digest((out / name).read_bytes())} == identity,
                'original-first proof bytes differ: ' + name)
    source_contract = json.loads((ROOT / 'docs/josev2b-source-contract.json').read_text())
    require(set(summary['source_payloads']) == {'joint_schema_model.py', 'tokenizer_config.json', 'tokenizer.json', 'LICENSE'},
            'original source payload scope differs')
    for name, identity in summary['source_payloads'].items():
        require(identity == {k: source_contract['files'][name][k] for k in ('bytes', 'sha256')} and
                identity == {'bytes': (out / 'source' / name).stat().st_size, 'sha256': digest((out / 'source' / name).read_bytes())},
                'original source bytes changed before native: ' + name)
    gold = json.loads((out / 'original-gold.json').read_text())
    require(all(row['expectation_matched'] for row in gold['cases']), 'original expected outcome barrier differs')
    require(gold['frozen_four_exact'] is True and gold['reference_sha256'] == REFERENCE_SHA and
            gold['source_sha256'] == SOURCE_SHA and gold['tokenizer_sha256'] == TOKENIZER_SHA,
            'original gold provenance differs')
    frozen_match(gold['cases'][:4], json.loads((ROOT / 'tests/results/josev2b/original-input-freeze.json').read_text()))
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'ggml', text=True).strip() == cfg['ggml'],
            'actual GGML checkout differs')
    binary = build / 'test-josev-text'
    component = out / 'josev2b-tokenizer.gguf'
    env = dict(os.environ, CRISPEMBED_JOSEV_TEXT_FRONTEND='1')
    command([binary, '--gates', component], out / 'review/gates.log', env=env)
    cases = panel_inputs()
    fixture = json.loads((ROOT / 'tests/fixtures/josev2b-tokenizer-schema-v1.json').read_text())
    negatives = fixture['native_subset_negatives']
    # Successful cases on both sides of every rejection prove context recovery.
    probes = []
    for case in negatives:
        probes.extend([dict(cases[0], id=case['id'] + '-before'), case, dict(cases[0], id=case['id'] + '-after')])
    requests = out / 'native-requests.jsonl'
    requests.write_text(''.join(dumps(c) + '\n' for c in cases + probes), encoding='utf-8')
    result_path = out / 'native.jsonl'
    command([binary, component, requests, result_path], out / 'review/native.log', env=env)
    result = [json.loads(line) for line in result_path.read_text().splitlines()]
    expected = gold['cases']
    require(len(result) == len(cases) + len(probes), 'native case count differs')
    failures = []
    for actual, old in zip(result[:len(cases)], expected):
        if (actual['id'] != old['id'] or actual['accepted'] != old['accepted'] or
                (old['accepted'] and actual['encoded'] != old['encoded'])):
            failures.append({'id': old['id'], 'expected': old, 'actual': actual})
    for i, negative in enumerate(negatives):
        before, rejected, after = result[len(cases) + 3*i:len(cases) + 3*i + 3]
        require(rejected['id'] == negative['id'] and rejected['accepted'] is False and rejected.get('error'),
                'native subset negative failed: ' + negative['id'])
        require(before['accepted'] and after['accepted'] and before['encoded'] == after['encoded'] == expected[0]['encoded'],
                'native context recovery differs: ' + negative['id'])
    # Valid metadata files, each with one deliberately wrong semantic field.
    source = json.loads((out / 'source/tokenizer.json').read_text())
    config = json.loads((out / 'source/tokenizer_config.json').read_text())
    metadata = component_metadata(source, config)
    changes = {'revision': ('general.source.huggingface.revision', 'wrong'),
               'normalizer': ('josev2b.tokenizer.normalizer', 'none'),
               'pattern': ('josev2b.tokenizer.pattern', 'wrong'),
               'source': ('josev2b.tokenizer.source_sha256', '0'*64),
               'flags': ('josev2b.tokenizer.added_flags', [1] + metadata['josev2b.tokenizer.added_flags'][1:])}
    for label, (key, value) in changes.items():
        bad = out / ('rejected-' + label + '.gguf')
        bad.write_bytes(gguf_bytes(dict(metadata, **{key: value})))
        log = out / 'review' / ('metadata-' + label + '.log')
        command([binary, bad, requests, out / ('rejected-' + label + '.jsonl')], log, env=env, expected=2)
        required = 'unsupported added token flags/ID' if label == 'flags' else 'tokenizer source/profile differs'
        require(required in log.read_text(), 'metadata negative rejected for an unrelated reason: ' + label)
    report = {'schema': 'crispembed-josev-tokenizer-native-v1', 'completed': True,
              'tokenizer_schema_parity': not failures, 'failures': failures,
              'cases': len(cases), 'original_four_exact': not any(f['id'] in {c['id'] for c in cases[:4]} for f in failures),
              'original_cases_sha256': digest((out / 'original-gold.json').read_bytes()),
              'native_rows_sha256': digest(result_path.read_bytes()), 'binary_sha256': digest(binary.read_bytes()),
              'gate_controls': 'unset/empty/0 reject; existing context immutable; reload clears',
              'metadata_negatives': list(changes), 'input_negatives_with_before_after': [n['id'] for n in negatives],
              'model_weights': False, 'embeddings': False, 'full_model_parity': False,
              'default_enabled': False, 'release_qualified': False, 'source_head': os.environ['GITHUB_SHA']}
    write(out / 'review/native.json', report)
    require(not failures, 'exact native tokenizer/schema parity failed; see native.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['preflight', 'original', 'native'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build', type=Path)
    args = parser.parse_args()
    out = args.output.resolve(); (out / 'review').mkdir(parents=True, exist_ok=True)
    try:
        cfg = hosted()
        write(out / 'review/preflight.json', {'schema': 'crispembed-josev-tokenizer-source-v1',
              'source_head': os.environ['GITHUB_SHA'], 'base': cfg['base'], 'ggml': cfg['ggml'],
              'source_sha256': cfg['source_sha256'], 'model_weights': False, 'default_enabled': False})
        if args.phase == 'original':
            original(out, cfg)
        elif args.phase == 'native':
            require(args.build is not None, 'native build path required')
            native(out, cfg, args.build.resolve())
    except Exception as error:
        write(out / 'review' / ('rejection-' + args.phase + '.json'), {
              'schema': 'crispembed-josev-tokenizer-rejection-v1', 'phase': args.phase,
              'error_type': type(error).__name__, 'error': str(error),
              'model_weights': False, 'release_qualified': False})
        raise


if __name__ == '__main__':
    main()
