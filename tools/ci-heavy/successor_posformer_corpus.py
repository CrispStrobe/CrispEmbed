#!/usr/bin/env python3
"""Hosted reproduction/preservation of frozen PosFormer PNG bytes; no inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from probe_posformer_release import MANIFEST_SHA
sys.path.insert(0, str(Path(__file__).parent))
from successor_assets import api, artifact

APP_SOURCE = '23f1470df0a0853234880ac7e0c6135fd4c02f59'
RENDERERS = {
    'provision_handwriting_quality.py': '54106994efe6c688d6376e99431f88e97e689b29d641d7579fe691e4aea9c416',
    'provision_ocr_quality.py': 'a3689613e27c2741a9a7353830f42afbdde50a86038b74f977e8d0af7e6b52b5',
}
FROZEN = ROOT / 'tests/results/posformer/main-backport-reference.json'
WORKFLOW = '.github/workflows/successor-posformer-corpus.yml'
ARTIFACT = 'posformer-frozen-corpus'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_cases():
    rows = json.loads(FROZEN.read_text(encoding='utf-8'))['reports']['original-baseline.json']['report']['cases']
    if len(rows) != 50 or len({r['id'] for r in rows}) != 50:
        raise ValueError('Frozen case coverage changed')
    return rows


def verify(corpus, cases=None, manifest_sha=MANIFEST_SHA):
    """Require original encoded bytes, not merely dimensions or similar pixels."""
    cases = frozen_cases() if cases is None else cases
    expected = {'manifest.json'} | {c['id'] + '.png' for c in cases}
    actual = {p.name for p in corpus.iterdir()}
    if actual != expected or any(not p.is_file() or p.is_symlink() for p in corpus.iterdir()):
        raise ValueError('Corpus member coverage differs')
    if sha(corpus / 'manifest.json') != manifest_sha:
        raise ValueError('Frozen manifest bytes changed')
    manifest = json.loads((corpus / 'manifest.json').read_text(encoding='utf-8'))
    if [r['id'] for r in manifest['cases']] != [r['id'] for r in cases]:
        raise ValueError('Frozen case order differs')
    rows = []
    for case in cases:
        path = corpus / (case['id'] + '.png')
        actual_sha = sha(path)
        if actual_sha != case['image_sha256']:
            raise ValueError('Original image bytes changed: ' + case['id'] +
                             ' expected=' + case['image_sha256'] + ' actual=' + actual_sha)
        rows.append({'id': case['id'], 'png_sha256': actual_sha, 'bytes': path.stat().st_size})
    return rows


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def hosted():
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise ValueError('Hosted only; no local corpus rendering/download')


def prepare(app, output, report):
    hosted()
    if platform.system() != 'Linux':
        raise ValueError('Frozen renderer proof must run on Linux')
    if subprocess.check_output(['git', '-C', str(app), 'rev-parse', 'HEAD'], text=True).strip() != APP_SOURCE:
        raise ValueError('App renderer source changed')
    for name, digest in RENDERERS.items():
        if sha(app / 'tool' / name) != digest:
            raise ValueError('Renderer bytes changed: ' + name)
    import PIL
    from PIL import Image, features
    if PIL.__version__ != '12.3.0':
        raise ValueError('Renderer Pillow version changed')
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, 'tool/provision_handwriting_quality.py'], cwd=app, check=True, timeout=900)
    source = app / '.dart_tool/inference/handwriting-corpus'
    rows = verify(source)  # No pixel equivalence is claimed until ALL original PNG bytes match.
    for row, case in zip(rows, frozen_cases()):
        with Image.open(source / (row['id'] + '.png')) as image:
            rgb = image.convert('RGB')
            if rgb.size != (case['image_width'], case['image_height']):
                raise ValueError('Frozen dimensions differ: ' + row['id'])
            row.update(width=rgb.width, height=rgb.height,
                       rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    shutil.copytree(source, output / 'corpus')
    value = {'schema': 'crispembed-posformer-frozen-corpus-v1', 'completed': True,
             'source_commit': os.environ['GITHUB_SHA'], 'run_id': os.environ['GITHUB_RUN_ID'],
             'app_source': APP_SOURCE, 'renderer_sha256': RENDERERS,
             'helper_sha256': sha(__file__), 'frozen_report_sha256': sha(FROZEN),
             'manifest_sha256': MANIFEST_SHA, 'cases': rows,
             'environment': {'platform': platform.platform(), 'python': sys.version,
                             'pillow': PIL.__version__, 'pillow_zlib': features.version('zlib')},
             'limits': ['All50 original encoded PNG hashes reproduced before RGB hashes recorded.',
                        'No model, library, recognition, score or gold recapture.']}
    write(output / 'receipt.json', value)
    write(report, value)


def consume(run, corpus, report):
    hosted()
    if not str(run).isdigit():
        raise ValueError('Invalid corpus run')
    service = api('actions/runs/' + str(run))
    if (service['status'] != 'completed' or service['conclusion'] != 'success'
            or service['path'] not in (WORKFLOW, '.github/workflows/ovisocr2-metadata.yml')
            or service['event'] != 'workflow_dispatch'):
        raise ValueError('Corpus producer did not complete successfully in a registered lane')
    jobs = api('actions/runs/' + str(run) + '/jobs?per_page=100')
    proofs = [j for j in jobs['jobs'] if j['name'] in ('reproduce', 'successor-corpus / reproduce')]
    if jobs['total_count'] > 100 or len(proofs) != 1 or proofs[0]['conclusion'] != 'success':
        raise ValueError('Missing successful corpus reproduction job')
    source = service['head_sha']
    subprocess.run(['git', 'fetch', '--depth=1', 'origin', source], cwd=ROOT, check=True, timeout=180)
    guarded = ('tools/ci-heavy/successor_posformer_corpus.py', WORKFLOW,
               'tests/results/posformer/main-backport-reference.json',
               'tools/probe_posformer_release.py', 'tools/ci-heavy/successor_assets.py',
               'tools/successor_asset_manifest.py')
    for path in guarded:
        before = subprocess.check_output(['git', 'rev-parse', source + ':' + path], cwd=ROOT, text=True).strip()
        after = subprocess.check_output(['git', 'rev-parse', 'HEAD:' + path], cwd=ROOT, text=True).strip()
        if before != after:
            raise ValueError('Corpus proof source changed: ' + path)
    report.parent.mkdir(parents=True, exist_ok=True)
    archive = report.parent / 'frozen-corpus.zip'
    identity = artifact(run, ARTIFACT, archive, 16 * 1024**2)
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        expected = {'receipt.json', 'corpus/manifest.json'} | {'corpus/' + c['id'] + '.png' for c in frozen_cases()}
        files = [n for n in names if not n.endswith('/')]
        if len(files) != len(set(files)) or set(files) != expected or sum(i.file_size for i in package.infolist()) > 32 * 1024**2:
            raise ValueError('Corpus artifact member coverage differs')
        receipt = json.loads(package.read('receipt.json'))
        if (receipt.get('completed') is not True or receipt['source_commit'] != source
                or receipt['run_id'] != str(run) or receipt['app_source'] != APP_SOURCE
                or receipt['renderer_sha256'] != RENDERERS or receipt['helper_sha256'] != sha(__file__)
                or receipt['frozen_report_sha256'] != sha(FROZEN) or receipt['manifest_sha256'] != MANIFEST_SHA):
            raise ValueError('Corpus producer provenance changed')
        corpus.mkdir(parents=True, exist_ok=False)
        for name in sorted(expected - {'receipt.json'}):
            (corpus / Path(name).name).write_bytes(package.read(name))
    rows = verify(corpus)
    if [{k: r[k] for k in ('id', 'png_sha256', 'bytes')} for r in receipt['cases']] != rows:
        raise ValueError('Corpus producer member hashes changed')
    from PIL import Image
    for row in receipt['cases']:
        with Image.open(corpus / (row['id'] + '.png')) as image:
            rgb = image.convert('RGB')
            if (rgb.size != (row['width'], row['height'])
                    or hashlib.sha256(rgb.tobytes()).hexdigest() != row['rgb_sha256']):
                raise ValueError('Consumer decoded RGB differs from byte-proven original: ' + row['id'])
    write(report, {'schema': 'crispembed-posformer-frozen-corpus-consumer-v1', 'completed': True,
                   'producer_run': str(run), 'producer_source': source,
                   'consumer_source': os.environ['GITHUB_SHA'], 'artifact': identity,
                   'cases': rows, 'producer_receipt': receipt})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--app', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p = sub.add_parser('consume'); p.add_argument('--run', required=True); p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == 'prepare': prepare(args.app, args.output, args.report)
        else: consume(args.run, args.corpus, args.report)
    except Exception as error:
        write(args.report, {'schema': 'crispembed-posformer-frozen-corpus-rejection-v1',
                            'completed': False, 'source_commit': os.environ.get('GITHUB_SHA'),
                            'mode': args.mode, 'error': str(error),
                            'limits': ['Original PNG identity remains required; no gold or image replacement accepted.']})
        raise
