#!/usr/bin/env python3
"""Durably retain exact existing W7 reports; no build, native load or inference."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import spacy_w7_publish as common

PINS = {
    '44262982569': {
        'new-panel-result.json': 'a6546bb627c997a9e05771030d303cb627dded996b340e256e909ea55829cdaf',
        'new-panel-seal.json': '99a4707fb070d67a6fe9407001088a2a6c17e4da05d1e05692a185bd864b8eb7',
        'new-panel-freeze.json': '12dca1cd6b980ca6c8528d9145b9319bfc5d5eac3dd7ede9e6b5df36b656b20f',
        'coverage-execution.json': '7b1e24ee0f114909b884ff6ff49e9eb7fda7f150d28a8556b3e75da56d76f2e8'},
    '44264083664': {
        'quantization-assessment.json': '4dbe4cea1bf8e5e13f620363d61ae33e94901face22a68d5e57d9094f13e253f',
        'f16-proof.json': '608030643e0976f016031674c0b9a23a4725ef5e82084c6d8907b176e37ae339',
        'f16-parity.log': '1d2b3ed6fa14d05825581eae093bee86a2e95eca552b6af78be0ba963551ec52'},
    '44265244322': {
        'quantization-assessment.json': 'd6789fb6db1648406d8df571224489644b4bdca9c8124e1eb3205624da596ff7',
        'q4_0-proof.json': 'ca3f9ab3923e1d06dc920e80c3cf23beb597793dec280d42a99daf252ccc5901',
        'q4_0-parity.log': '498432fa94a5205cfaf0e0d6edac077b5b6d3779ae16d4436cbe842a056865f0'}
}


def main():
    assert os.environ.get('GITHUB_ACTIONS') == 'true'
    assert os.environ.get('GITHUB_REPOSITORY') == 'CrispStrobe/CrispEmbed'
    assert os.environ.get('SPACY_W7_RETAIN_ONLY') == '1'
    assert os.environ.get('SPACY_W7_RETAIN_ARGS', '') == ''
    assert os.environ.get('SPACY_W7_RETAIN_PIP', '') == ''
    assert os.environ.get('SPACY_W7_RETAIN_RUNNER') == 'ubuntu-latest'
    assert os.environ.get('SPACY_W7_RETAIN_MODE') == 'portable'
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'w7-report-retention'
    scratch.mkdir()
    payload = scratch / 'payload'
    payload.mkdir()
    for run, pins in PINS.items():
        downloaded = scratch / run
        subprocess.run(['gh', 'run', 'download', run, '--repo', 'CrispStrobe/CrispEmbed',
                        '--name', 'heavy-' + run, '--dir', str(downloaded)], check=True)
        target = payload / run
        target.mkdir()
        for name, sha in pins.items():
            file = downloaded / name
            assert file.stat().st_size <= 8 * 1024 * 1024 and common.digest(file) == sha
            shutil.copyfile(file, target / name)
    (payload / 'README.md').write_text('# Preserved W7 reports\n\n'
        'Exact original source-bound new grammar result/seal and initial quantization failures. '
        'Retention only; no inference, build or acceptance/grammar panel replay. '
        'Original W6 5/30 baseline,22/30 combined and eight misses remain unchanged.\n')
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub==0.36.0'], check=True)
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get('HF_PUBLISH_TOKEN') or os.environ.get('HF_TOKEN_2'))
    prefix = 'retained-w7-evidence/' + os.environ['GITHUB_SHA']
    commit = api.upload_folder(repo_id=common.REPOSITORY, folder_path=str(payload), path_in_repo=prefix,
                              commit_message='Preserve original W7 grammar report and initial quantization failures').oid
    fresh = scratch / 'fresh-readback'
    fresh.mkdir()
    receipt = {'schema': 'crispembed.spacy.w7.report-retention.v1', 'passed': True,
               'source': os.environ['GITHUB_SHA'], 'revision': commit, 'files': {},
               'build_or_inference': False, 'consumed_panel_replay': False}
    for file in payload.rglob('*'):
        if not file.is_file():
            continue
        name = file.relative_to(payload).as_posix()
        destination = fresh / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        receipt['files'][name] = common.download(
            f'https://huggingface.co/{common.REPOSITORY}/resolve/{commit}/{prefix}/{name}',
            destination, common.digest(file), file.stat().st_size)
    out = Path(os.environ['HEAVY_OUT'])
    (out / 'report-retention.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (out / 'summary.md').write_text('# W7 durable report retention\n\nExact existing reports '
        'published and freshly downloaded by immutable revision. No build/model inference or panel replay.\n')


if __name__ == '__main__':
    main()
