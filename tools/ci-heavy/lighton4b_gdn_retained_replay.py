#!/usr/bin/env python3
"""Replay the exact failed synthetic capture without importing its framework."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
REPORT_SHA = '8709072c0f080a47a5d1a8b52b34c3eb6eba9bee9152de2c87872aa2c4e88fa0'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retained', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        parser.error('Native build/replay belongs on hosted CI')
    if args.output.exists() or args.build_dir.exists():
        parser.error('Fresh output/build directories required')
    source = args.retained / 'gdn'
    raw_report = (source / 'component-report.json').read_bytes()
    if hashlib.sha256(raw_report).hexdigest() != REPORT_SHA:
        raise ValueError('Retained report hash differs')
    old = json.loads(raw_report)
    if old['github_run_id'] != '41923599391' or old['source_commit'] != '52275dd1fbe07a1920aa409ca2ad5f3f7ee51d01':
        raise ValueError('Retained source identity differs')
    if old['component_passed'] or not old['original_references_frozen_before_native']:
        raise ValueError('Expected original-first negative capture')
    spec = importlib.util.spec_from_file_location('frozen_gdn', ROOT / 'tools/ci-heavy/lighton4b_gdn_reference.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    import numpy as np
    if np.__version__ != '2.3.5':
        raise ValueError('Numpy pin differs')
    args.output.mkdir(parents=True)
    report = {'schema': 'lighton4b-retained-gdn-replay/v1', 'source_run': '41923599391',
              'source_artifact_id': 13367509290, 'source_report_sha256': REPORT_SHA,
              'original_forward_executed': False, 'component_passed': False,
              **{name: False for name in driver.FALSE_QUALIFICATIONS}, 'cases': []}
    code = 1
    try:
        # Verify all four complete references and every captured raw tensor
        # before native build/execution. Original output bytes are never regenerated.
        if [c['name'] for c in old['cases']] != [c['name'] for c in driver.CASES]:
            raise ValueError('Retained case coverage differs')
        for case in old['cases']:
            folder = source / case['name']
            for filename, field in (('weights.gguf', 'weights'), ('reference.gguf', 'reference')):
                if driver.file_record(folder / filename) != case[field]:
                    raise ValueError('Retained GGUF hash differs')
            for name, record in case['reference_tensors'].items():
                if driver.file_record(folder / 'reference' / (name + '.f32')) != {k: record[k] for k in ('bytes', 'sha256')}:
                    raise ValueError('Retained raw reference differs')
            target = args.output / case['name']
            target.mkdir()
            shutil.copytree(folder / 'reference', target / 'reference')
            for filename in ('weights.gguf', 'reference.gguf'):
                shutil.copyfile(folder / filename, target / filename)
        report['all_frozen_references_verified_before_native'] = True
        report['source_commit'] = driver.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        report['native_sources'] = {name: driver.file_record(ROOT / name) for name in driver.NATIVE_SOURCES}
        binary = driver.portable_build(args.build_dir, args.output)
        for old_case in old['cases']:
            case = {k: v for k, v in old_case.items() if k != 'native'}
            case['native'] = driver.compare_native(case, args.output / case['name'], binary, np)
            report['cases'].append(case)
        for name, before in report['native_sources'].items():
            if driver.file_record(ROOT / name) != before:
                raise ValueError('Replay source changed during execution')
        report['component_passed'] = all(c['native']['passed'] for c in report['cases'])
        code = 0 if report['component_passed'] else 1
    except Exception as error:
        report['error'] = str(error)
    finally:
        (args.output / 'retained-replay-report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return code

if __name__ == '__main__':
    raise SystemExit(main())
