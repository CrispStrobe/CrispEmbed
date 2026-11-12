#!/usr/bin/env python3
"""Repeat the plain-environment full and shallow pytest audit on pinned main."""
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'plain-pytest-current'


def run(args, **kwargs):
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    # Bind the subject to the actual ancestor before any test; no working tree files
    # from this harness branch are substituted into the tested checkout.
    subject = subprocess.check_output(['git', 'rev-parse', 'HEAD^'], cwd=ROOT, text=True).strip()
    expected = subprocess.check_output(['git', 'rev-parse', 'c364feb97'], cwd=ROOT, text=True).strip()
    if subject != expected:
        raise RuntimeError('harness parent is not the declared main commit')
    venv = SCRATCH / 'venv'
    run([sys.executable, '-m', 'venv', venv])
    py = venv / 'bin/python'
    run([py, '-m', 'pip', 'install', 'pytest', 'numpy', 'pillow', 'gguf', 'sentencepiece', 'tokenizers'])
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('HF_', 'HUGGING', 'KAGGLE_', 'CRISPEMBED_'))
           and k not in ('PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONUSERBASE')}
    env['PATH'] = str(venv / 'bin') + os.pathsep + env.get('PATH', '')
    report = {'schema': 'plain-pytest-current-v1', 'subject_commit': subject,
              'harness_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'environment': 'clean venv; no model, token, native build, torch or transformers', 'modes': {}}
    for mode in ('full', 'shallow'):
        d = SCRATCH / ('repo-' + mode)
        if mode == 'full':
            run(['git', 'clone', '-q', 'https://github.com/CrispStrobe/CrispEmbed.git', d], env=env)
            run(['git', '-C', d, 'checkout', '-q', subject], env=env)
        else:
            run(['git', 'init', '-q', d], env=env)
            run(['git', '-C', d, 'fetch', '-q', '--depth', '1', 'https://github.com/CrispStrobe/CrispEmbed.git', subject], env=env)
            run(['git', '-C', d, 'checkout', '-q', 'FETCH_HEAD'], env=env)
        assert subprocess.check_output(['git', '-C', d, 'rev-parse', 'HEAD'], text=True).strip() == subject
        xml = SCRATCH / (mode + '.xml')
        proc = subprocess.run([py, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider', '-rfEs',
                               '--junitxml=' + str(xml)], cwd=d, env=env, capture_output=True, text=True)
        (OUT / (mode + '-tail.txt')).write_text('\n'.join(proc.stdout.splitlines()[-400:]) + '\n')
        counts, failures, errors, skips = collections.Counter(), [], [], collections.Counter()
        if not xml.exists():
            raise RuntimeError(mode + ' produced no JUnit report')
        for tc in ET.parse(xml).getroot().iter('testcase'):
            test_id = f"{tc.get('classname')}::{tc.get('name')}"
            result = next((c for c in tc if c.tag in ('failure', 'error', 'skipped')), None)
            kind = result.tag if result is not None else 'passed'
            counts[kind] += 1
            if kind == 'failure':
                failures.append(test_id)
            elif kind == 'error':
                errors.append(test_id)
            elif kind == 'skipped':
                skips[(result.get('message') or '')[:200]] += 1
        report['modes'][mode] = {'exit': proc.returncode, 'counts': dict(counts), 'failures': failures,
                                 'errors': errors, 'skip_reasons': dict(skips)}
        (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    # The audit reports red suites faithfully, while keeping the evidence artifact.
    if any(row['exit'] for row in report['modes'].values()):
        raise SystemExit('pytest reported failures; inspect report.json')


if __name__ == '__main__':
    main()
