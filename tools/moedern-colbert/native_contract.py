"""Dependency-free guards for the separately versioned 1.5M native experiment."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'docs/moedern-colbert-native-contract-v1.json'
LANE = ('src/moedern_colbert.cpp', 'src/moedern_colbert.h',
        'tests/moedern-native/CMakeLists.txt', 'tests/moedern-native/probe.cpp',
        'tests/moedern-native/controls.cpp', 'tools/moedern-colbert/native_contract.py',
        'tools/moedern-colbert/native.py', 'tests/test_moedern_native_contract.py',
        'docs/moedern-colbert-native-contract-v1.json', 'docs/moedern-colbert-native.md',
        '.github/workflows/moedern-colbert-native.yml')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def protocol():
    return json.loads(PATH.read_text(encoding='utf-8'))


def dependencies(value=None):
    cfg = protocol() if value is None else value
    pin = cfg['reviewed_original']
    if pin.get('reviewed') is not True:
        raise ValueError('Original review pending: model/build access forbidden')
    if type(pin.get('run')) is not int or pin['run'] <= 0 or pin['source'] != cfg['base_source']:
        raise ValueError('Exact reviewed original run/source required')
    for key in ('receipt_sha256', 'gguf_sha256', 'conversion_receipt_sha256'):
        if not isinstance(pin.get(key), str) or not re.fullmatch('[a-f0-9]{64}', pin[key]):
            raise ValueError('Reviewed original digest missing: ' + key)
    for key in ('review_artifact', 'raw_artifact'):
        identity = pin[key]
        if (type(identity.get('id')) is not int or identity['id'] <= 0 or
                type(identity.get('bytes')) is not int or not 0 < identity['bytes'] <= 256 * 1024**2 or
                not isinstance(identity.get('sha256'), str) or not re.fullmatch('[a-f0-9]{64}', identity['sha256'])):
            raise ValueError('Reviewed service artifact identity incomplete: ' + key)
    return pin


def source():
    cfg = protocol()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    if digest(ROOT / 'docs/moedern-colbert-source-contract.json') != cfg['original_contract_sha256']:
        raise ValueError('Historical original contract changed')
    if digest(ROOT / 'tests/fixtures/moedern-colbert/panel-v1.json') != cfg['panel_sha256']:
        raise ValueError('Frozen owned panel changed')
    changed = git('diff', '--name-only', cfg['base_source'], 'HEAD', '--', *cfg['shared_paths']).splitlines()
    if set(changed) != set(cfg['allowed_new_src']):
        raise ValueError('Shared production/build/binding source changed outside new model files')
    if git('status', '--porcelain', '--', *cfg['shared_paths'], *LANE):
        raise ValueError('Dirty/untracked measured source')
    if git('rev-parse', 'HEAD:ggml') != cfg['ggml']:
        raise ValueError('Declared ggml revision changed')
    return {'head': git('rev-parse', 'HEAD'), 'base': cfg['base_source'], 'ggml': cfg['ggml'],
            'files': {name: digest(ROOT / name) for name in LANE}}


def check_original(report):
    cfg = protocol()
    pin = dependencies(cfg)
    if (report.get('schema') != 'crispembed.moedern-colbert.original.v1' or
            report.get('completed') is not True or report.get('variant') != '1.5M' or
            report.get('source_contract_sha256') != cfg['original_contract_sha256'] or
            report.get('panel_sha256') != cfg['panel_sha256'] or report.get('github_sha') != pin['source']):
        raise ValueError('Reviewed original completion/source/panel differs')
    expected_drivers = {'tools/moedern-colbert/reference.py', 'tools/moedern-colbert/contract.py',
                        'tools/moedern-colbert/convert_f32.py'}
    if set(report.get('driver_sources', {})) != expected_drivers:
        raise ValueError('Original driver-source coverage incomplete')
    if report.get('revision') != '9d360b8b88ae11bea5784bf9012ff509fdd5c2da':
        raise ValueError('Original model revision differs')
    for name, expected in report['driver_sources'].items():
        raw = subprocess.check_output(['git', 'show', pin['source'] + ':' + name], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Original driver source SHA differs')
    panel = json.loads((ROOT / 'tests/fixtures/moedern-colbert/panel-v1.json').read_text())
    pairs = [(row.get('query'), row.get('document')) for row in report.get('scores', [])]
    if pairs != [tuple(pair) for pair in panel['score_pairs']]:
        raise ValueError('Original MaxSim pair coverage/order differs')
    expected_versions = {'torch':'2.8.0', 'transformers':'5.3.0', 'sentence-transformers':'6.1.0', 'numpy':'2.2.6'}
    if any(report.get('versions', {}).get(name, '').split('+')[0] != version for name, version in expected_versions.items()):
        raise ValueError('Original package profile differs')
    if len(report.get('cases', [])) != len(panel['cases']):
        raise ValueError('Original case coverage differs')
    for expected, actual in zip(panel['cases'], report['cases']):
        if any(actual.get(k) != expected[k] for k in ('id', 'role', 'text')):
            raise ValueError('Original request/role/order differs')
        ids = actual['token_ids']
        if (not 3 <= len(ids) <= 512 or any(type(i) is not int or not 0 <= i < 30002 for i in ids) or
                ids[:2] != [2, 30000 if expected['role'] == 'query' else 30001] or
                (expected['role'] == 'query' and len(ids) != 32) or
                any(len(actual[k]) != len(ids) for k in ('supplied_attention_mask', 'effective_attention_mask', 'scoring_mask')) or
                any(value not in (0,1,False,True) for key in ('supplied_attention_mask','effective_attention_mask','scoring_mask') for value in actual[key]) or
                actual.get('ordinary_observer_bit_exact') is not True or actual.get('public_slice_bit_exact') is not True):
            raise ValueError('Original ID/mask/neutrality contract differs')
    return panel


def skiplist_ids(tokenizer, config, tokenizer_config):
    """Independent exact-vocabulary equivalent of convert_tokens_to_ids.

    This is a token-ID experiment, not a text tokenizer implementation. Resolve
    the saved document punctuation policy without looking at any output masks.
    """
    vocab = tokenizer['model']['vocab']
    if isinstance(vocab, list):
        vocab = {item[0]: index for index, item in enumerate(vocab)}
    elif isinstance(vocab, dict):
        vocab = dict(vocab)
    else:
        raise ValueError('Unsupported actual vocabulary representation')
    for item in tokenizer.get('added_tokens', []):
        vocab[item['content']] = item['id']
    unknown = tokenizer_config['unk_token']
    if isinstance(unknown, dict):
        unknown = unknown['content']
    if unknown not in vocab:
        raise ValueError('Original unknown token not in actual vocabulary')
    resolved = []
    for word in config['skiplist_words']:
        token = vocab.get(word, vocab[unknown])
        if token != vocab[unknown] or word == unknown:
            resolved.append(token)
    return sorted(set(resolved))
