"""Separate SciFact native-v3 contract; historical producers remain immutable."""
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'tests/regression/ultrabert/scifact-native-protocol-v3.json'
PROTOCOL_SHA = '4a7a54f1ea9d3e40a51cb7bd86b0def08c520b859a5545537ba11c712a45606e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def protocol():
    if sha(ROOT / CONTRACT) != PROTOCOL_SHA:
        raise ValueError('Native-v3 protocol bytes differ')
    return json.loads((ROOT / CONTRACT).read_text(encoding='utf-8'))


def check_token_proof(report):
    if (report.get('source') != protocol()['token_proof']['source'] or not report.get('completed')
            or report.get('tokenizer_parity') is not True or report.get('all_original_ids_masks_fullhashes_exact') is not True
            or report.get('authored_cases_original_exact') != 14 or report.get('errors') != []
            or any(report.get(k) is not False for k in ('embeddings_authorized', 'new_embeddings_executed', 'model_weights_accessed'))
            or [r['mode'] for r in report.get('modes', [])] != ['unset', 'empty', '0', '1']
            or any(not r['passed'] or r['rows'] != 5483 for r in report['modes'])
            or report['modes'][-1]['clipped_mismatches'] or report['modes'][-1]['full_mismatches']):
        raise ValueError('Reviewed token-only evidence differs; no native authorization')


def safe_zip(archive):
    entries = archive.infolist()
    names = [i.filename for i in entries]
    if (len(names) != len(set(names)) or sum(i.file_size for i in entries) > 2 * 1024**3
            or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n
                   or not n or PurePosixPath(n).as_posix() != n.rstrip('/') or ((i.external_attr >> 16) & 0o170000) == 0o120000
                   for n, i in zip(names, entries))):
        raise ValueError('Unsafe/oversized artifact ZIP')


def check_rows(data, tokens):
    rows = data['queries'] + data['documents']
    if (len(data['queries']) != 300 or len(data['documents']) != 5183
            or len(tokens) != 5483 or len(set(r['id'] for r in rows)) != 5483
            or [r['id'] for r in rows] != [r['id'] for r in tokens]):
        raise ValueError('Frozen full ordered5483 row identity differs')
    for row in tokens:
        ids = row['ids']
        if (not 1 <= len(ids) <= 512 or ids[0] != 0 or any(type(n) is not int or n < 0 for n in ids)
                or row['attention_mask'] != [1] * len(ids)):
            raise ValueError('Frozen unpadded ID/mask semantics differ')
    return {r['id']: r for r in tokens}


def check_actual_ids(name, actual, frozen):
    if actual != frozen['ids']:
        raise ValueError('Actual public CAPI IDs differ: ' + name)
    # Padding attention semantics are source-derived, never a public-API observation.
    derived = [1] * len(actual)
    if derived != frozen['attention_mask']:
        raise ValueError('DERIVED implicit all-attended mask differs: ' + name)
    return dict(id=name, ids=actual, attention_mask=derived, mask_provenance='DERIVED')


def check_shard(receipt, mode, index, rows, frozen=None):
    p = protocol()
    schema = 'crispembed-scifact-' + mode + '-shard-' + ('v2' if mode == 'original' else 'v3')
    if (receipt.get('schema') != schema or receipt.get('status') != 'completed ' + mode + ' shard'
            or receipt.get('shard') != index or receipt.get('rows') != rows
            or receipt.get('data_freeze_sha256') != p['prepared']['data_freeze_sha256']):
        raise ValueError('Incomplete/different ' + mode + ' shard')
    if mode == 'original':
        if receipt != frozen or receipt.get('all_rows_hook_neutral') is not True:
            raise ValueError('Frozen original receipt/neutrality differs')
    elif (receipt.get('all_actual_ids_preflight_exact') is not True
          or receipt.get('gates') != p['gates'] or receipt.get('model_sha256') != p['native_model_sha256']
          or receipt.get('mask_contract') != p['mask_contract']):
        raise ValueError('Native model/gates/derived mask/preflight differs')
