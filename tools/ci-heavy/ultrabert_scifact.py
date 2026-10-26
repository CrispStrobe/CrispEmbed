"""Hosted original-first full SciFact preparation, shards and frozen scoring."""
import argparse
import gc
import gzip
import hashlib
import importlib.util
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from ultrabert_contract import MODEL, REVISION, QUERY, profile, sha256, validate_sources, validate_weight_file
from ultrabert_scifact import PROTOCOL_SHA, protocol, digest, equivalence, canonical_claims, migration_equivalence, qrels, shard_rows, rank_scores, quality
ST_SOURCE = 'b0acd1a4d22ba2bc10072868ddea7612561c9424'
ST_WRAPPER_SHA = '3d677757bf3e684c45d3634ae67918cd9a67cee6aac6760e5fa1c72dfe820650'
ST_POOL_SHA = 'b6e96e0eac50ebc33d6310ef3b8d35b7cd2b7e0e521809552bba34c05c13d43d'
MODULE_CONFIG_SHA = '3084164002c0bca01b0259c5327123803fce32e660a57feb93184ffead186fc8'
ST_PREPROCESS_LF = {
    'sentence_transformers.base.model': '46016f24f1bf6923ef32d704ba6bea7e1b06d1687b82372a3b34a6a94f040899',
    'sentence_transformers.base.modules.transformer': 'c382da40c06d37fc9ec38bf685a6f80e4b1d0b25bf9601af8c79b41005259f9e',
    'sentence_transformers.base.modality': '777461f474f000de6acc0befc8787be939ff142318df4bdf087502c851119073',
}

REPO = 'CrispStrobe/CrispEmbed'
WORKFLOW = '.github/workflows/ultrabert-scifact.yml'
CALLER_WORKFLOW = '.github/workflows/ovisocr2-metadata.yml'
PINS = {
    "tests/results/ultrabert/source-audit.json": "582c99998d5f20bd1f05b22493325e1f74e8e9f9d8b3dbba0aab83bfa09ab50b",
    "tests/results/ultrabert/embed-weight-schema.json": "91ee35692669f99998b1933fceff997cd64e72e4248845374ed9627d578f5367",
    "tests/results/ultrabert/reference-audit.json": "a3c16863b1a78ebaebcc2d50e24555b696e691d3309ee17b75075c0051032a36",
    "tests/results/ultrabert/public-f32-api.json": "97a54afa7cde1a46386805652ef6cd578c6971b1df6152076fad68e59081bc68",
    "tests/results/ultrabert/retrieval-smoke.json": "add393025718dc140ad370b750b168d13d66ca6ad240779e2a1a1a87506ef39b",
    "tests/results/ultrabert/initial-native-parity.json": "3db56f2af1abc5cb34e662b18e50e8edb752ae0c3cffa85bb38ac7c4c150a477",
    "tests/results/ultrabert/local-operator-audit.json": "a1ccd4ad359ca94f74ea9e710b1ff0575cca826e955b219d3ab57a2288da6a4c",
    "tests/results/ultrabert/initial-linear-capture-failure.json": "e910d51a230c63435670a3ac630fe68a7f9303fc10df1a22d03e75a37a21ead7",
    "tests/results/ultrabert/full-forward-capture-failure.json": "74be8372319979bcd9fc5b7a820334592f8f5c3ca21e8b44bfb1f3516508e8e8",
    "tests/results/ultrabert/m1-validation.json": "94cf1bcd7e5ddfcd0aad21de6b251466fb31b559b3fea8dd3a512c28e59a7370",
    "src/ultrabert_embed.cpp": "1bc4b04421234373b2568b12cb0de89055231596888795f34e7ef199d70bbd18",
    "src/ultrabert_profile.h": "fc5af20f91a6b2cb6fa9c07efa875a9d42d3e7e076038b833a908019a0affc31",
    "tools/ultrabert_contract.py": "16b7f9c93017f4be97b367fa5aec33877a8ff053dfdc12771ec71109a311c9d8",
    "models/convert-ultrabert-to-gguf.py": "90d30b6207e7b7dbfa0e7d0e9aa9d23f6dee02fbf006786fbb2b3bff1bdfe532"
}
FILES = ('tools/ci-heavy/ultrabert_scifact.py', 'tools/ultrabert_scifact.py',
         'tests/test_ultrabert_scifact.py', 'tests/test_ultrabert_scifact_wrapper.py',
         'tests/regression/ultrabert/scifact-protocol-v1.json',
         '.github/workflows/ultrabert-scifact.yml', 'src/ultrabert_profile.h', 'src/ultrabert_embed.cpp',
         'python/crispembed/_binding.py', 'models/convert-ultrabert-to-gguf.py', 'CMakeLists.txt',
         'tests/test_ultrabert_truncation.cpp', 'docs/ultrabert-scifact-notices.md',
         '.github/actions/scifact-setup/action.yml', '.github/workflows/ovisocr2-metadata.yml',
         'tools/ultrabert_contract.py')
GGML = '890278a8342c620197c90e702e1188bcab94f510'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load(path):
    return json.loads(Path(path).read_text())


def source():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    if any(sha256(ROOT / p) != h for p, h in PINS.items()):
        raise ValueError('Frozen historical receipts/scoped UltraBERT sources differ')
    if (git('rev-parse', 'HEAD:ggml') != GGML
            or git('-C', str(ROOT / 'ggml'), 'rev-parse', 'HEAD') != GGML
            or git('-C', str(ROOT / 'ggml'), 'status', '--porcelain')
            or git('diff', '--name-only', 'HEAD', '--', 'src', 'ggml')
            or git('ls-files', '--others', '--exclude-standard', '--', 'src')):
        raise ValueError('Native source checkout differs')
    runtime = git('ls-files', '--', 'src').splitlines()
    files = {p: sha256(ROOT / p) for p in FILES}
    files.update({p: sha256(ROOT / p) for p in runtime if (ROOT / p).is_file()})
    return dict(head=git('rev-parse', 'HEAD'), ggml=GGML,
                files=files, protected=PINS, run=os.environ.get('GITHUB_RUN_ID'),
                run_attempt=os.environ.get('GITHUB_RUN_ATTEMPT'))


def fetch(url, path, *, size=None, checksum=None, max_bytes=20 * 1024 ** 2, etag=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as out:
        if etag and response.headers.get('ETag', '').strip('"') != etag:
            raise ValueError('Author immutable object ETag changed')
        if 'versionId=' in url and response.headers.get('x-amz-version-id') != url.split('versionId=', 1)[1]:
            raise ValueError('Author immutable object version differs')
        count = 0
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > max_bytes:
                raise ValueError('Download exceeds declared bound')
            out.write(block)
    if (size is not None and path.stat().st_size != size) or (checksum and sha256(path) != checksum):
        raise ValueError('Downloaded immutable bytes differ: ' + path.name)
    return dict(url=url, bytes=path.stat().st_size, sha256=sha256(path))


def inventory(directory):
    return {p.relative_to(directory).as_posix(): dict(bytes=p.stat().st_size, sha256=sha256(p))
            for p in sorted(Path(directory).rglob('*')) if p.is_file()}


def verify_inventory(directory, expected):
    if inventory(directory) != expected:
        raise ValueError('Retained artifact member bytes/hashes differ')


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', 'repos/' + REPO + '/' + path], timeout=120))


def reuse(run_id, name, destination, *, report_sha=None):
    if not str(run_id).isdigit():
        raise ValueError('Immutable numeric producer run required')
    meta = api('actions/runs/' + str(run_id))
    if meta.get('path') not in (WORKFLOW, CALLER_WORKFLOW):
        raise ValueError('Producer workflow differs')
    if str(run_id) == os.environ.get('GITHUB_RUN_ID') and name != 'ultrabert-scifact-prepared-' + str(run_id):
        # Within this panel, dependency job completion is the producer barrier.
        match = __import__('re').fullmatch(r'ultrabert-scifact-(original|native)-(\d)', name)
        job_name = match[1] + '-shard-' + match[2] if match else 'freeze-original'
        jobs = api('actions/runs/' + str(run_id) + '/jobs?per_page=100')['jobs']
        jobs = [j for j in jobs if j['name'] == job_name or j['name'].endswith(' / ' + job_name)]
        if len(jobs) != 1 or jobs[0].get('status') != 'completed' or jobs[0].get('conclusion') != 'success':
            raise ValueError('Required producer job did not complete successfully')
    elif meta.get('status') != 'completed' or meta.get('conclusion') != 'success':
        raise ValueError('Prior preparation workflow incomplete/failed')
    artifacts = api('actions/runs/' + str(run_id) + '/artifacts?per_page=100')['artifacts']
    matches = [a for a in artifacts if a['name'] == name and a.get('expired') is False]
    if len(matches) != 1:
        raise ValueError('Retained immutable artifact absent/ambiguous')
    artifact = matches[0]
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    archive = target.parent / (target.name + '.zip')
    with archive.open('wb') as stream:
        subprocess.run(['gh', 'api', 'repos/' + REPO + '/actions/artifacts/' + str(artifact['id']) + '/zip'],
                       stdout=stream, check=True, timeout=1200)
    if (archive.stat().st_size != artifact['size_in_bytes']
            or 'sha256:' + sha256(archive) != artifact.get('digest')):
        raise ValueError('Service artifact ZIP bytes/hash differ')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if (len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts
                or '\\' in n for n in names) or sum(i.file_size for i in z.infolist()) > 2 * 1024 ** 3):
            raise ValueError('Unsafe retained artifact')
        z.extractall(target)
    receipt = load(target / 'receipt.json')
    if report_sha and sha256(target / 'receipt.json') != report_sha:
        raise ValueError('Reviewed preparation receipt SHA256 differs')
    if receipt['source']['head'] != meta['head_sha'] or receipt['source']['files'] != source()['files']:
        raise ValueError('Producer/current measured source closure differs')
    expected = receipt['files']
    actual = inventory(target)
    actual.pop('receipt.json', None)
    if actual != expected:
        raise ValueError('Producer artifact file coverage/hash differs')
    return receipt, dict(run=int(run_id), head=meta['head_sha'], artifact_id=artifact['id'],
        name=name, bytes=artifact['size_in_bytes'], digest=artifact['digest'], expires_at=artifact['expires_at'],
        receipt_sha256=sha256(target / 'receipt.json'))


def original_snapshot(weights):
    from huggingface_hub import snapshot_download
    patterns = list(profile()['inspected_file_sha256']) + ['sentence_bert_config.json'] + (['model.safetensors'] if weights else [])
    directory = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=patterns))
    validate_sources(directory)
    module_config(directory)
    if weights:
        validate_weight_file(directory / 'model.safetensors')
    return directory


def module_config(directory):
    path = Path(directory) / 'sentence_bert_config.json'
    if not path.is_file() or path.stat().st_size != 241 or sha256(path) != MODULE_CONFIG_SHA:
        raise ValueError('Actual published SentenceTransformer module config differs')
    return load(path)


def tokenizer(directory):
    import transformers
    if transformers.__version__ != '5.16.1':
        raise ValueError('Pinned Transformers required')
    result = transformers.AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=True)
    if result.truncation_side != 'right' or result.bos_token_id != 0 or result.pad_token_id != 0:
        raise ValueError('Original tokenizer side/BOS/pad differs')
    return result


def input_text(row):
    return (QUERY if row['id'].startswith('q:') else '') + row['text']


def lengths(data, tok):
    rows = []
    for row in data['queries'] + data['documents']:
        text = input_text(row)
        ids = tok(text, truncation=False, add_special_tokens=True)['input_ids']
        clipped = tok(text, truncation=True, max_length=512, add_special_tokens=True)
        if (not ids or ids[0] != 0 or clipped['input_ids'] != ids[:512]
                or clipped['attention_mask'] != [1] * min(len(ids), 512)
                or len(text.encode()) > 65536 or '<|' in text or '\x00' in text):
            raise ValueError('Unqualified complete original tokenizer row: ' + row['id'])
        rows.append(dict(id=row['id'], input_utf8_sha256=hashlib.sha256(text.encode()).hexdigest(),
            untruncated_tokens=len(ids), untruncated_ids_sha256=digest(ids), ids=ids[:512],
            attention_mask=[1] * min(len(ids), 512), truncated=len(ids) > 512))
    return rows


def original_model(directory):
    import torch
    import transformers
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    published_module_config = module_config(directory)
    preprocessing_sources = {}
    for name, expected in ST_PREPROCESS_LF.items():
        module = importlib.import_module(name)
        raw = Path(inspect.getfile(module)).read_bytes()
        canonical = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
        if canonical != expected:
            raise ValueError('Actual original preprocessing module source differs: ' + name)
        preprocessing_sources[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), lf_sha256=canonical)
    if any(importlib.metadata.version(name) != version for name, version in
           {'numpy':'2.2.6', 'gguf':'0.19.0', 'huggingface-hub':'1.33.0', 'pyarrow':'21.0.0'}.items()):
        raise ValueError('Pinned original environment dependency differs')
    if (torch.__version__.split('+')[0] != '2.14.0' or transformers.__version__ != '5.16.1'
            or sentence_transformers.__version__ != '6.0.1' or sha256(inspect.getfile(SentenceTransformer)) != ST_WRAPPER_SHA):
        raise ValueError('Pinned actual original wrapper differs')
    torch.set_num_threads(2)
    model = SentenceTransformer(str(directory), device='cpu', trust_remote_code=True, local_files_only=True,
                               model_kwargs={'attn_implementation': 'eager', 'dtype': torch.float32})
    if (len(model) != 2 or sha256(inspect.getfile(type(model[1]))) != ST_POOL_SHA
            or model[1].pooling_mode != 'cls' or not model[1].include_prompt
            or model.default_prompt_name is not None or model.prompts.get('query') != QUERY
            or model[0].auto_model.config._attn_implementation != 'eager'
            or sha256(inspect.getfile(type(model[0].auto_model))) != profile()['inspected_file_sha256']['modeling_ultrabert.py']
            or any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in model.parameters())):
        raise ValueError('Original source/module/CLS/CPU/F32 contract differs')
    model.max_seq_length = 512
    if model.max_seq_length != 512 or model[0].tokenizer.model_max_length != 512:
        raise ValueError('Original max_seq_length setter did not set actual tokenizer512')
    return model, dict(torch=torch.__version__, transformers=transformers.__version__, wrapper=sentence_transformers.__version__,
        wrapper_source=ST_SOURCE, wrapper_sha256=ST_WRAPPER_SHA, pooling_sha256=ST_POOL_SHA,
        preprocessing_sources=preprocessing_sources, published_module_config=published_module_config,
        published_module_config_sha256=MODULE_CONFIG_SHA,
        custom_source_sha256=profile()['inspected_file_sha256']['modeling_ultrabert.py'],
        weight_sha256=sha256(directory / 'model.safetensors'), cpu_capability=torch.backends.cpu.get_cpu_capability(),
        torch_config=torch.__config__.show(), threads=2, max_seq_length=512, batch_size=1)


class WrapperTokenMismatch(ValueError):
    def __init__(self, row, method, expected, ids, mask):
        super().__init__('Actual original wrapper token preflight differs: ' + row + '/' + method)
        self.details = dict(id=row, method=method, original_ids=expected['ids'], actual_ids=ids,
                            original_mask=expected['attention_mask'], actual_mask=mask)


def wrapper_token_preflight(model, data, token_rows):
    """No forward calls: check both actual wrapper prompt paths for every row."""
    rows = data['queries'] + data['documents']
    if len(rows) != 5483 or [row['id'] for row in rows] != [row['id'] for row in token_rows]:
        raise ValueError('Complete ordered5483 wrapper input inventory differs')
    for row, expected in zip(rows, token_rows):
        prompt = QUERY if row['id'].startswith('q:') else ''
        paths = [('resolved_prompt', model.preprocess([row['text']], prompt=prompt)),
                 ('literal_prefix', model.tokenize([input_text(row)]))]
        for method, features in paths:
            ids = features['input_ids'].tolist()
            mask = features['attention_mask'].tolist()
            if ids != [expected['ids']] or mask != [expected['attention_mask']]:
                raise WrapperTokenMismatch(row['id'], method, expected, ids, mask)
    return dict(rows=5483, all_ids_and_masks_exact=True, both_prompt_paths_checked=True,
                before_first_forward=True, ordered_ids_sha256=digest([row['id'] for row in rows]))


def save_vectors(directory, raw, normalized, ids, timing):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / 'raw.npy', np.asarray(raw, dtype='<f4'), allow_pickle=False)
    np.save(directory / 'normalized.npy', np.asarray(normalized, dtype='<f4'), allow_pickle=False)
    write(directory / 'ids.json', ids); write(directory / 'timing.json', timing)


def encode_original(model, rows, token_rows, raw_dir):
    import numpy as np
    import torch
    raw, normalized, timing, ids_seen = [], [], [], []
    deadline = time.monotonic() + 220 * 60
    for row in rows:
        if time.monotonic() > deadline:
            save_vectors(raw_dir, raw, normalized, ids_seen, timing)
            raise TimeoutError('Original shard220min execution bound reached; partial rows retained')
        expected = token_rows[row['id']]
        text = input_text(row)
        actual = model.tokenize([text])
        if (actual['input_ids'][0].tolist() != expected['ids']
                or actual['attention_mask'][0].tolist() != expected['attention_mask']):
            raise ValueError('Actual wrapper token IDs/mask differ: ' + row['id'])
        captured = []
        def capture(module, args):
            features = args[0]
            if (features['input_ids'][0].tolist() != expected['ids']
                    or features['attention_mask'][0].tolist() != expected['attention_mask']):
                raise ValueError('Actual forward IDs/mask differ')
            captured.append(features['token_embeddings'][0, 0].detach().cpu().numpy().copy())
        handle = model[1].register_forward_pre_hook(capture)
        start = time.perf_counter()
        try:
            a = model.encode([row['text']], prompt_name='query' if row['id'].startswith('q:') else None,
                             prompt=None if row['id'].startswith('q:') else '', batch_size=1,
                             normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False)[0]
        finally:
            handle.remove()
        elapsed = time.perf_counter() - start
        if len(captured) != 1 or captured[0].dtype != a.dtype or captured[0].tobytes() != a.tobytes():
            raise ValueError('Actual original wrapper is not observed raw CLS')
        # Actual unobserved ordinary raw output must match the hooked one for
        # every row, including duplicate text/token rows with distinct IDs.
        ordinary_start = time.perf_counter()
        plain = model.encode([row['text']], prompt_name='query' if row['id'].startswith('q:') else None,
                             prompt=None if row['id'].startswith('q:') else '', batch_size=1,
                             normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False)[0]
        ordinary_seconds = time.perf_counter() - ordinary_start
        if a.dtype != plain.dtype or a.tobytes() != plain.tobytes():
            raise ValueError('Original observer changed ordinary raw output: ' + row['id'])
        # Actual wrapper normalization is measured, not a replacement output.
        normal_start = time.perf_counter()
        b = model.encode([row['text']], prompt_name='query' if row['id'].startswith('q:') else None,
                         prompt=None if row['id'].startswith('q:') else '', batch_size=1,
                         normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        normal_seconds = time.perf_counter() - normal_start
        if (a.dtype != np.dtype('float32') or b.dtype != a.dtype
                or a.shape != (768,) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
                or np.max(abs(a.astype('float64') / np.linalg.norm(a.astype('float64')) - b)) > 1e-6):
            raise ValueError('Original raw/normalized vector invalid')
        raw.append(a); normalized.append(b); ids_seen.append(dict(id=row['id'], ids=expected['ids']))
        timing.append(dict(id=row['id'], tokens=len(expected['ids']), untruncated_tokens=expected['untruncated_tokens'],
                           raw_seconds=elapsed, ordinary_raw_seconds=ordinary_seconds,
                           normalized_seconds=normal_seconds, total_seconds=elapsed + ordinary_seconds + normal_seconds))
        if len(rows) <= 16 or len(raw) % 16 == 0:
            save_vectors(raw_dir, raw, normalized, ids_seen, timing)
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.save(raw_dir / 'raw.npy', np.asarray(raw, dtype='<f4'), allow_pickle=False)
    np.save(raw_dir / 'normalized.npy', np.asarray(normalized, dtype='<f4'), allow_pickle=False)
    write(raw_dir / 'ids.json', ids_seen)
    write(raw_dir / 'timing.json', timing)
    return timing


def prepare(args, out, scratch, closure):
    p = protocol(); spec = p['dataset']
    payload = out / 'payload'; downloads = payload / 'sources'; notices = payload / 'notices'
    downloads.mkdir(parents=True); notices.mkdir()
    sources = {}
    author_url = 'https://raw.githubusercontent.com/allenai/scifact/' + spec['author_revision'] + '/'
    for name, url, checksum in [
        ('author-LICENSE.md', author_url + 'LICENSE.md', '56ee69bc639cbab8f0b2e4b67091ec5e9390aa99a1d65f0a35e4274686c9a806'),
        ('author-README.md', author_url + 'README.md', '3f0eb7f6d009b3785fede6592918b8b583e7193d26f14173fdf0c79aba1dcc79'),
        ('BEIR-scifact-README.md', 'https://huggingface.co/datasets/BeIR/scifact/raw/' + spec['beir_revision'] + '/README.md', 'b5f57be1839fdb75867073986a019a78afe0a149950f79d1165509efc5e5cf2f'),
        ('BEIR-qrels-README.md', 'https://huggingface.co/datasets/BeIR/scifact-qrels/raw/' + spec['qrels_revision'] + '/README.md', 'fd34a2830e557f1e32d46d0b34a218d280c9800ee4071579610d4ff587b4fd45')]:
        sources[name] = fetch(url, notices / name, checksum=checksum)
    shutil.copy2(ROOT / 'docs/ultrabert-scifact-notices.md', notices / 'ATTRIBUTION.md')
    author = downloads / 'author-data.tar.gz'
    sources['author_archive'] = fetch(spec['author_archive_url'], author, size=spec['author_archive_bytes'],
                                      etag=spec['author_archive_etag'],
                                      checksum='11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be')
    extracted = {}
    with tarfile.open(author, 'r:gz') as archive:
        members = archive.getmembers()
        if sum(m.size for m in members) > 50 * 1024 ** 2:
            raise ValueError('Author archive expansion exceeds bound')
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
                raise ValueError('Unsafe author archive member')
            if member.isfile() and (path.name == 'corpus.jsonl' or path.name.startswith('claims_') and path.suffix == '.jsonl'):
                if path.name in extracted:
                    raise ValueError('Ambiguous author data member')
                content = archive.extractfile(member).read()
                extracted[path.name] = [json.loads(line) for line in content.splitlines() if line]
                sources['author/' + path.name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    if 'corpus.jsonl' not in extracted or 'claims_dev.jsonl' not in extracted:
        raise ValueError('Author complete corpus/dev claims absent')
    # No author-derived evidence/citation mapping is silently treated as gold.
    claims = canonical_claims(extracted)
    parquet = {}
    for kind, size, checksum in [('corpus', 4469916, '243324b35f03d82bd6d98a5f575966876e86cad7ce16e5333a35b1b793dc4f45'),
                                 ('queries', 64982, '1c37956c5dc8b810b60302323c24d1a9e79e26411ba8f5ad9d0888642e2a9034')]:
        target = downloads / (kind + '.parquet')
        url = 'https://huggingface.co/datasets/BeIR/scifact/resolve/' + spec['beir_revision'] + '/' + kind + '/' + kind + '-00000-of-00001.parquet'
        sources['BEIR/' + kind] = fetch(url, target, size=size, checksum=checksum)
        import pyarrow.parquet as pq
        parquet[kind] = pq.read_table(target).to_pylist()
    target = downloads / 'test.tsv'
    url = 'https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/' + spec['qrels_revision'] + '/test.tsv'
    sources['BEIR/qrels'] = fetch(url, target, size=spec['qrels_bytes'])
    content = target.read_bytes()
    if hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() != spec['qrels_git_blob']:
        raise ValueError('Canonical BEIR qrels immutable Git blob differs')
    gold = qrels(content)
    try:
        legacy = {}
        for kind, size, checksum in [('corpus', 2758600, '2f4a2a264485e5b5b95ae9cd5534817447757423991a4767d6da811539c0853e'),
                                     ('queries', 49271, 'c29574b9cb8fdf043df8494f689de15f4604ac4b585eb34bda9b6a389aa6c4d3')]:
            path = downloads / ('legacy-' + kind + '.jsonl.gz')
            url = 'https://huggingface.co/datasets/BeIR/scifact/resolve/984eed826375f18d27936c4a32bf0f8491e3f414/' + kind + '.jsonl.gz'
            sources['BEIR/legacy-' + kind] = fetch(url, path, size=size, checksum=checksum)
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                legacy[kind] = [json.loads(line) for line in stream if line.strip()]
        migration = migration_equivalence(legacy['corpus'], legacy['queries'], parquet['corpus'], parquet['queries'])
        write(out / 'legacy-parquet-equivalence.json', migration)
        data, mapping = equivalence(extracted['corpus.jsonl'], claims, parquet['corpus'], parquet['queries'], gold)
    except Exception as error:
        write(out / 'initial-data-equivalence-failure.json', dict(status='rejected before model outputs',
            source=closure, protocol_sha256=PROTOCOL_SHA, sources=sources, error=str(error),
            mismatch_diagnostics=getattr(error, 'details', None)))
        raise
    mapping['author_claim_files'] = ['claims_train.jsonl', 'claims_dev.jsonl', 'claims_test.jsonl']
    mapping['unused_cross_validation_files'] = sorted(name for name in extracted if name.startswith('claims_') and name not in mapping['author_claim_files'])
    mapping['legacy_parquet_equivalence'] = migration
    write(out / 'initial-author-gold-mapping.json', mapping)
    write(payload / 'dataset.json', data)
    write(payload / 'protocol.json', p)
    directory = original_snapshot(False)
    try:
        rows = lengths(data, tokenizer(directory))
    except Exception as error:
        write(out / 'initial-input-policy-failure.json', dict(status='rejected before weights/model outputs',
            source=closure, sources=sources, mapping=mapping, protocol_sha256=PROTOCOL_SHA, error=str(error)))
        raise
    write(payload / 'lengths.json', rows)
    freeze = dict(schema='crispembed-scifact-data-freeze-v1', source=closure, protocol_sha256=PROTOCOL_SHA,
        sources=sources, mapping=mapping, dataset_sha256=sha256(payload / 'dataset.json'),
        original_model_files_sha256={**profile()['inspected_file_sha256'], 'sentence_bert_config.json':MODULE_CONFIG_SHA},
        lengths_sha256=sha256(payload / 'lengths.json'), rows=len(rows), queries=300, corpus=5183,
        truncated_rows=sum(r['truncated'] for r in rows), max_untruncated_tokens=max(r['untruncated_tokens'] for r in rows),
        created_before_first_model_output=True, applicable_terms=p['licenses'], acceptance=p['acceptance'])
    write(payload / 'data-freeze.json', freeze)
    write(out / 'data-freeze.json', freeze)
    write(out / 'input-length-summary.json', dict(rows=len(rows), queries=300, corpus=5183,
        truncated_queries=sum(r['truncated'] for r in rows if r['id'].startswith('q:')),
        truncated_documents=sum(r['truncated'] for r in rows if r['id'].startswith('d:')),
        max_untruncated_tokens=max(r['untruncated_tokens'] for r in rows),
        ordered_ids_sha256=digest([r['id'] for r in rows]), token_rows_sha256=sha256(payload / 'lengths.json')))
    for notice in notices.iterdir():
        shutil.copy2(notice, out / (notice.name + '.txt'))
    # Every dataset/input/gold/protocol bit is frozen before loading weights or any vector output.
    directory = original_snapshot(True)
    model, env = original_model(directory)
    try:
        actual_wrapper_inputs = wrapper_token_preflight(model, data, rows)
    except Exception as error:
        write(out / 'initial-wrapper-token-failure.json', dict(status='rejected before first model forward',
            source=closure, protocol_sha256=PROTOCOL_SHA, data_freeze_sha256=sha256(payload / 'data-freeze.json'),
            error=str(error), details=getattr(error, 'details', None)))
        raise
    write(payload / 'wrapper-token-preflight.json', actual_wrapper_inputs)
    write(out / 'wrapper-token-preflight.json', actual_wrapper_inputs)
    lookup = {r['id']: r for r in data['queries'] + data['documents']}
    query = sorted(data['queries'], key=lambda r: (next(x for x in rows if x['id'] == r['id'])['untruncated_tokens'], r['id']))
    documents = sorted([r for r in rows if r['id'].startswith('d:')], key=lambda r: (len(r['ids']), r['id']))
    selected = [data['queries'][0]['id'], query[-1]['id']] + [documents[round(i * (len(documents) - 1) / 10)]['id'] for i in range(11)]
    selected = list(dict.fromkeys(selected))
    timings = encode_original(model, [lookup[name] for name in selected], {r['id']: r for r in rows}, payload / 'preflight')
    worst = max(t['total_seconds'] for t in timings)
    projection = max(len(shard_rows(data, i)) for i in range(8)) * worst * p['execution']['projection_safety_factor']
    receipt = dict(schema='crispembed-scifact-preparation-v1', source=closure, status='prepared original-only preflight',
        protocol_sha256=PROTOCOL_SHA, data_freeze_sha256=sha256(payload / 'data-freeze.json'), mapping=mapping,
        original_environment=env, actual_wrapper_input_preflight=actual_wrapper_inputs,
        original_preflight=timings, original_hook_neutral=True,
        projected_worst_original_shard_seconds=projection, projected_native_runtime='unmeasured; bounded by240min job timeout',
        projected_original_within_bound=projection <= p['execution']['max_projected_shard_seconds'],
        full_panel_authorized=False, review_required='Root independently reviews dataset/freeze/lengths/timing and actual source before full panel',
        files=inventory(payload))
    write(payload / 'receipt.json', receipt)
    write(out / 'preparation.json', receipt)
    return 0


def prepared(args, scratch):
    target = scratch / 'prepared'
    receipt, proof = reuse(args.prepare_run, 'ultrabert-scifact-prepared-' + str(args.prepare_run), target,
                           report_sha=args.prepare_sha)
    if (receipt.get('schema') != 'crispembed-scifact-preparation-v1'
            or receipt.get('status') != 'prepared original-only preflight'
            or receipt.get('protocol_sha256') != PROTOCOL_SHA or receipt.get('original_hook_neutral') is not True
            or receipt.get('projected_original_within_bound') is not True):
        raise ValueError('Reviewed preparation did not complete bounded original preflight')
    freeze = load(target / 'data-freeze.json')
    if (sha256(target / 'data-freeze.json') != receipt['data_freeze_sha256']
            or freeze['dataset_sha256'] != sha256(target / 'dataset.json')
            or freeze['lengths_sha256'] != sha256(target / 'lengths.json')
            or freeze['rows'] != 5483 or freeze['queries'] != 300 or freeze['corpus'] != 5183
            or freeze['acceptance'] != protocol()['acceptance']
            or load(target / 'protocol.json') != protocol()):
        raise ValueError('Prepared dataset/input/gold/protocol closure differs')
    data = load(target / 'dataset.json'); token_rows = load(target / 'lengths.json')
    if ([r['id'] for r in token_rows] != [r['id'] for r in data['queries'] + data['documents']]
            or len(set(r['id'] for r in token_rows)) != 5483):
        raise ValueError('Prepared exact complete ordered row identity differs')
    return target, data, {r['id']: r for r in token_rows}, receipt, proof


def original_shard(args, out, scratch, closure):
    prepared_root, data, tokens, preparation, proof = prepared(args, scratch)
    directory = original_snapshot(True)
    model, env = original_model(directory)
    rows = shard_rows(data, args.shard)
    payload = out / 'payload'
    timing = encode_original(model, rows, tokens, payload)
    shutil.copytree(prepared_root / 'notices', payload / 'notices')
    receipt = dict(schema='crispembed-scifact-original-shard-v1', source=closure, shard=args.shard,
        status='completed original shard', preparation=proof, protocol_sha256=PROTOCOL_SHA,
        data_freeze_sha256=preparation['data_freeze_sha256'], original_environment=env,
        rows=[r['id'] for r in rows], raw_seconds=sum(r['raw_seconds'] for r in timing),
        all_rows_hook_neutral=True, total_seconds=sum(r['total_seconds'] for r in timing), files=inventory(payload))
    write(payload / 'receipt.json', receipt)
    write(out / ('original-' + str(args.shard) + '.json'), receipt)
    return 0


def freeze_original(args, out, scratch, closure):
    _, data, _, preparation, proof = prepared(args, scratch)
    shards, proofs = [], []
    for i in range(8):
        receipt, lineage = reuse(os.environ['GITHUB_RUN_ID'], 'ultrabert-scifact-original-' + str(i), scratch / ('original-' + str(i)))
        if (receipt.get('schema') != 'crispembed-scifact-original-shard-v1' or receipt.get('shard') != i
                or receipt.get('status') != 'completed original shard' or receipt.get('all_rows_hook_neutral') is not True
                or receipt.get('data_freeze_sha256') != preparation['data_freeze_sha256']
                or receipt.get('rows') != [r['id'] for r in shard_rows(data, i)]):
            raise ValueError('Incomplete/different original shard')
        shards.append(receipt); proofs.append(lineage)
    # This inventory freezes all original vector bytes before any native job may start.
    payload = out / 'payload'; payload.mkdir()
    receipt = dict(schema='crispembed-scifact-original-freeze-v1', source=closure,
        status='all8 original shards frozen before native', preparation=proof, protocol_sha256=PROTOCOL_SHA,
        data_freeze_sha256=preparation['data_freeze_sha256'], shards=shards, service=proofs,
        original_run=int(os.environ['GITHUB_RUN_ID']), original_rows=5483,
        no_native_outputs_yet=True, files={})
    write(payload / 'receipt.json', receipt)
    write(out / 'original-freeze.json', receipt)
    return 0


def original_freeze(args, scratch):
    # The current panel is still running. Verify the completed producer job and
    # artifact rather than incorrectly requiring this parent run to be complete.
    return reuse(args.panel_run, 'ultrabert-scifact-original-freeze', scratch / 'original-freeze')


def native_shard(args, out, scratch, closure):
    import numpy as np
    prepared_root, data, tokens, preparation, proof = prepared(args, scratch)
    frozen, frozen_proof = original_freeze(args, scratch)
    if (frozen.get('status') != 'all8 original shards frozen before native' or frozen.get('original_rows') != 5483
            or frozen.get('data_freeze_sha256') != preparation['data_freeze_sha256']
            or frozen.get('no_native_outputs_yet') is not True):
        raise ValueError('All original shards must be frozen before native')
    import importlib.metadata
    if any(importlib.metadata.version(name) != version for name, version in
           {'numpy':'2.2.6', 'gguf':'0.19.0', 'huggingface-hub':'1.33.0'}.items()):
        raise ValueError('Pinned native conversion/array environment differs')
    directory = original_snapshot(True)
    converter_spec = importlib.util.spec_from_file_location('scifact_converter', ROOT / 'models/convert-ultrabert-to-gguf.py')
    converter = importlib.util.module_from_spec(converter_spec); converter_spec.loader.exec_module(converter)
    model_path = scratch / 'native-f32.gguf'
    conversion = converter.convert(directory, model_path, 'f32')
    if conversion['sha256'] != '89478b9cbbad2fd1121be4031d760f1529c989fedf23b6b19251ba2c87422566':
        raise ValueError('Qualified F32 conversion bytes differ')
    binary = Path(os.environ['AUDIT_BIN_DIR']) / 'test-ultrabert-truncation'
    env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')}
    env['CRISPEMBED_ULTRABERT'] = '1'
    with (out / 'truncation-controls.log').open('w') as stream:
        subprocess.run([str(binary), str(model_path)], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=120)
    library = next(Path(os.environ['AUDIT_BIN_DIR']).rglob('libcrispembed.so'))
    sys.path.insert(0, str(ROOT / 'python'))
    from crispembed import CrispEmbed
    os.environ['CRISPEMBED_ULTRABERT'] = '1'; os.environ['CRISPEMBED_FORCE_CPU'] = '1'
    os.environ['CRISPEMBED_ULTRABERT_TRUNCATE'] = '1'
    model = CrispEmbed(str(model_path), n_threads=2, lib_path=str(library), offline=True)
    if (Path(model._lib._name).resolve() != library.resolve()
            or Path(inspect.getfile(CrispEmbed)).resolve() != (ROOT / 'python/crispembed/_binding.py').resolve()):
        raise ValueError('Native binding/library fallback differs')
    raw, normalized, ids_seen, timing = [], [], [], []
    rows = shard_rows(data, args.shard)
    payload = out / 'payload'
    deadline = time.monotonic() + 220 * 60
    for row in rows:
        if time.monotonic() > deadline:
            save_vectors(payload, raw, normalized, ids_seen, timing)
            raise TimeoutError('Native shard220min execution bound reached; partial rows retained')
        model.set_prefix(QUERY if row['id'].startswith('q:') else '')
        start = time.perf_counter()
        actual_ids, hidden = model.encode_tokens(row['text'], normalize=False)
        elapsed = time.perf_counter() - start
        ids = actual_ids.tolist()
        if ids != tokens[row['id']]['ids'] or hidden.shape != (len(ids), 768):
            write(out / 'initial-id-mismatch.json', dict(id=row['id'], actual=ids, original=tokens[row['id']]['ids']))
            raise ValueError('Actual native text IDs differ: ' + row['id'])
        a = hidden[0].copy(); b = model.encode(row['text'])
        if (a.dtype != np.dtype('float32') or b.dtype != a.dtype
                or a.shape != (768,) or b.shape != (768,) or not np.isfinite(a).all() or not np.isfinite(b).all()
                or np.max(abs(a.astype('float64') / np.linalg.norm(a.astype('float64')) - b)) > 1e-6):
            raise ValueError('Native raw/normalized vector invalid')
        raw.append(a); normalized.append(b); ids_seen.append(dict(id=row['id'], ids=ids))
        timing.append(dict(id=row['id'], tokens=len(ids), raw_seconds=elapsed))
        if len(raw) % 16 == 0:
            save_vectors(payload, raw, normalized, ids_seen, timing)
    payload.mkdir(exist_ok=True)
    np.save(payload / 'raw.npy', np.asarray(raw, dtype='<f4'), allow_pickle=False)
    np.save(payload / 'normalized.npy', np.asarray(normalized, dtype='<f4'), allow_pickle=False)
    write(payload / 'ids.json', ids_seen); write(payload / 'timing.json', timing)
    shutil.copytree(prepared_root / 'notices', payload / 'notices')
    receipt = dict(schema='crispembed-scifact-native-shard-v1', status='completed native shard', source=closure,
        shard=args.shard, preparation=proof, original_freeze=frozen_proof, protocol_sha256=PROTOCOL_SHA,
        data_freeze_sha256=preparation['data_freeze_sha256'], rows=[r['id'] for r in rows],
        library_sha256=sha256(library), model_sha256=sha256(model_path),
        truncation_control_log_sha256=sha256(out / 'truncation-controls.log'),
        gates={'CRISPEMBED_ULTRABERT':'1', 'CRISPEMBED_FORCE_CPU':'1', 'CRISPEMBED_ULTRABERT_TRUNCATE':'1'},
        raw_seconds=sum(r['raw_seconds'] for r in timing), files=inventory(payload))
    write(payload / 'receipt.json', receipt); write(out / ('native-' + str(args.shard) + '.json'), receipt)
    return 0


def aggregate(args, out, scratch, closure):
    import numpy as np
    prepared_root, data, tokens, preparation, proof = prepared(args, scratch)
    frozen, frozen_proof = original_freeze(args, scratch)
    all_rows = data['queries'] + data['documents']; ordered_ids = [r['id'] for r in all_rows]
    records, service_proofs = {}, []
    for mode in ('original', 'native'):
        values = {}
        for i in range(8):
            target = scratch / (mode + '-' + str(i))
            receipt, lineage = reuse(args.panel_run, 'ultrabert-scifact-' + mode + '-' + str(i), target)
            if (receipt.get('schema') != 'crispembed-scifact-' + mode + '-shard-v1'
                    or receipt.get('shard') != i or receipt.get('status') != 'completed ' + mode + ' shard'
                    or receipt.get('data_freeze_sha256') != preparation['data_freeze_sha256']
                    or receipt.get('rows') != [r['id'] for r in shard_rows(data, i)]):
                raise ValueError('Incomplete/different actual shard')
            if mode == 'original' and (receipt != frozen['shards'][i] or lineage != frozen['service'][i]):
                raise ValueError('Frozen original shard/service bytes changed')
            raw = np.load(target / 'raw.npy', allow_pickle=False); normalized = np.load(target / 'normalized.npy', allow_pickle=False)
            ids = load(target / 'ids.json')
            if (raw.dtype != np.dtype('<f4') or normalized.dtype != raw.dtype
                    or raw.shape != (len(receipt['rows']), 768) or normalized.shape != raw.shape
                    or not np.isfinite(raw).all() or not np.isfinite(normalized).all()
                    or [r['id'] for r in ids] != receipt['rows']):
                raise ValueError('Actual shard vector/ID coverage differs')
            for n, name in enumerate(receipt['rows']):
                if name in values or ids[n]['ids'] != tokens[name]['ids']:
                    raise ValueError('Duplicate/mismatched exact native/original IDs')
                norm = np.linalg.norm(raw[n].astype('float64'))
                if norm <= 1e-12 or np.max(abs(raw[n] / norm - normalized[n])) > 1e-6:
                    raise ValueError('Raw/normalized shard vector differs')
                values[name] = dict(raw=raw[n], normalized=normalized[n])
            service_proofs.append(dict(mode=mode, shard=i, proof=lineage))
        if set(values) != set(ordered_ids):
            raise ValueError('Full5483 row coverage incomplete')
        records[mode] = values
    comparisons = []
    for name in ordered_ids:
        for mode in ('raw', 'normalized'):
            a = records['original'][name][mode].astype('float64'); b = records['native'][name][mode].astype('float64')
            error = abs(a - b); an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            cosine = float(np.dot(a, b) / (an * bn)) if an * bn else 0.
            maximum = float(error.max())
            comparisons.append(dict(id=name, mode=mode, max_abs=maximum, mean_abs=float(error.mean()),
                rms=float(np.sqrt(np.mean(error ** 2))), original_norm=an, native_norm=bn,
                cosine=cosine, passed=maximum <= .002 and cosine >= .99999))
    payload = out / 'payload'; payload.mkdir()
    rankings = {}
    document_ids = [r['id'] for r in data['documents']]
    for mode in ('original', 'native'):
        matrix = np.asarray([records[mode][name]['normalized'] for name in ordered_ids], dtype='float64')
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        scores = matrix[:300] @ matrix[300:].T
        np.save(payload / (mode + '-scores.npy'), scores, allow_pickle=False)
        rankings[mode] = [dict(query_id=row['id'], **rank_scores(document_ids, scores[i], row['relevant_document_ids']))
                          for i, row in enumerate(data['queries'])]
        write(payload / (mode + '-rankings.json'), rankings[mode])
    result = quality(rankings['original'], rankings['native'])
    changed = [dict(query_id=a['query_id'], original_top1=a['top1'], native_top1=b['top1'],
                    original_margin=a['top_margin'], native_margin=b['top_margin'],
                    original_relevant_ranks=a['relevant_ranks'], native_relevant_ranks=b['relevant_ranks'])
               for a, b in zip(rankings['original'], rankings['native']) if a['top1'] != b['top1']]
    write(payload / 'vector-comparisons.json', comparisons)
    shutil.copytree(prepared_root / 'notices', payload / 'notices')
    passed = result['passed'] and all(r['passed'] for r in comparisons)
    receipt = dict(schema='crispembed-scifact-quality-v1', source=closure,
        status='passed bounded full SciFact task gates' if passed else 'rejected full SciFact task gates',
        protocol_sha256=PROTOCOL_SHA, preparation=proof, original_freeze=frozen_proof,
        data_freeze_sha256=preparation['data_freeze_sha256'], service=service_proofs,
        queries=300, corpus=5183, vector_comparisons=10966, all5483_ids_exact=True, quality=result,
        top1_changes=changed, vector_failures=[r for r in comparisons if not r['passed']],
        all_misses_and_margins='Original/native full300-query rankings and all relevant ranks in retained payload',
        limits=protocol()['limits'], applicable_terms=protocol()['licenses'], files=inventory(payload))
    write(payload / 'receipt.json', receipt); write(out / 'scifact-quality.json', receipt)
    return int(not passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'original', 'freeze-original', 'native', 'aggregate', 'preflight'), default='prepare')
    parser.add_argument('--prepare-run'); parser.add_argument('--prepare-sha')
    parser.add_argument('--panel-run'); parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    closure = source()
    if args.phase == 'preflight':
        print(json.dumps(dict(source=closure, protocol=protocol()), indent=2)); return 0
    if args.phase != 'prepare' and (not args.prepare_run or not args.prepare_sha):
        raise ValueError('Full panel requires explicitly reviewed immutable preparation run and receipt SHA256')
    out = Path(os.environ['HEAVY_OUT']); scratch = Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True); scratch.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scratch).free < 10 * 1024 ** 3:
        raise ValueError('At least10GiB hosted scratch required')
    report = dict(phase=args.phase, source=closure, status='started', protocol_sha256=PROTOCOL_SHA)
    write(out / 'phase.json', report)
    try:
        result = {'prepare':prepare, 'original':original_shard, 'freeze-original':freeze_original,
                  'native':native_shard, 'aggregate':aggregate}[args.phase](args, out, scratch, closure)
        report.update(status='completed' if not result else 'completed with retained numerical/quality failures', exit_code=result)
        write(out / 'phase.json', report)
        return result
    except Exception as error:
        message = str(error)
        for path, label in [(str(ROOT), '<repo>'), (str(scratch), '<scratch>'), (str(out), '<out>'),
                            (os.environ.get('HF_HOME', ''), '<hf-cache>')]:
            if path:
                message = message.replace(path, label)
        report.update(status='failed structural/source contract', error=message)
        write(out / 'phase.json', report)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
