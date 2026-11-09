#!/usr/bin/env python3
"""Small retained replay review; no framework imports or native execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
p = Path(sys.argv[1])
x = json.loads(p.read_bytes())
spec = importlib.util.spec_from_file_location('gdn', Path(__file__).with_name('lighton4b_gdn_reference.py'))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
assert x['source_run'] == '41923599391' and x['source_artifact_id'] == 13367509290
assert x['original_forward_executed'] is False and x['all_frozen_references_verified_before_native'] is True
assert x['component_passed'] is True
assert [c['name'] for c in x['cases']] == [c['name'] for c in d.CASES]
expected = [(lane, stage) for lane, stage, _ in d.expected_stages()]
for c in x['cases']:
    n = c['native']
    assert n['passed'] and n['exit_code'] == 0 and n['frozen_original_artifacts_unchanged']
    assert n['stage_count'] == len(expected) == 52
    assert [(r['lane'], r['stage']) for r in n['stages']] == expected
    assert all(r['passed'] and r['finite'] for r in n['stages'])
    assert n['controls_passed'] and any(r['name'] == 'empty_cache_is_null' and r['passed'] for r in n['controls'])
    print(c['name'], 'all52 passed', 'max_abs', max(r['max_abs'] for r in n['stages']), 'min_cos', min(r['cos_min'] for r in n['stages']))
for field in d.FALSE_QUALIFICATIONS:
    assert x[field] is False
print('report_sha256', hashlib.sha256(p.read_bytes()).hexdigest())
