"""Versioned authored Josev task protocol and public-response scorer (stdlib only).

This module never imports a model framework or changes historical guards/golds.
The new60 field labels were reviewed independently before any model output.
"""
from __future__ import annotations
import copy
from collections import Counter
from decimal import Decimal
import json
import math
from pathlib import Path
import subprocess

from josev2b_contract import (environment_versions, require_cuda_capability, sha256,
                            validate_request, verify_snapshot)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/josev2b-task-quality-v1.json'
PANEL = ROOT / 'tests/regression/josev2b/task-quality-v1.json'
REVIEW = ROOT / 'tests/regression/josev2b/task-quality-label-review-v1.json'
PUBLISHER = ROOT / 'docs/josev2b-source-contract.json'
PUBLISHER_SHA = '6fbe3efe8958b2d87f5fe42c52fa71857c2d644efe3ad363bb60d8dfa52885da'
PANEL_SHA = '2b1a3c8d0c20be4a64a7f21695b80da8f5f727dd8b50a4d8fd00523aa20efb18'
F32 = dict(max_absolute=.002, minimum_row_cosine=.99999, logits_absolute=.002,
           raw_probabilities_absolute=.0001, ordinal_expectation_absolute=.001)
Q4 = dict(discrete_decision_agreement=1., no_accuracy_drop=True,
          maximum_public_probability_absolute=.01, maximum_ordinal_index_absolute=.01)
FLOORS = dict(overall=.90, per_type=.85, per_language=.85)


def require(ok, why):
    if not ok:
        raise ValueError(why)


def declaration():
    d = json.loads(CONTRACT.read_text())
    require(d['schema'] == 'crispembed-josev-task-quality-protocol-v1' and
            d['publisher_contract_sha256'] == PUBLISHER_SHA and d['panel_sha256'] == PANEL_SHA and
            d['f32_required'] == F32 and d['q4_new_task_required'] == Q4 and d['accuracy_floors'] == FLOORS,
            'Declared original/F32/new-task budgets differ')
    require(sha256(PUBLISHER) == PUBLISHER_SHA, 'Unchanged publisher profile differs')
    require(sha256(REVIEW) == d['label_review_sha256'] and d['label_review_completed_before_outputs'] is True,
            'Frozen independent coordinator label review differs')
    require(d['old_four_case_thresholds_changed'] is False and d['q4_strict_f32_flags'] == 'retain-report-separately',
            'Historical/F32 gates may not be relabeled')
    return d


def protocol():
    """A NEW profile using the exact old publisher math/env with the new input panel.

    No historical module global, reference, source guard or request is patched.
    The immutable publisher contract remains independently authenticated.
    """
    d = declaration()
    cfg = copy.deepcopy(json.loads(PUBLISHER.read_text()))
    cfg['schema'] = 'crispembed-josev-task-quality-original-v1'
    cfg['panel'] = dict(path=str(PANEL.relative_to(ROOT)), sha256=PANEL_SHA, cases=20)
    cfg['task_quality'] = d
    return cfg


def panel():
    declaration()
    require(sha256(PANEL) == PANEL_SHA, 'Frozen authored inputs/labels differ')
    data = json.loads(PANEL.read_text())
    cases = data['cases']
    require(data['license'] == 'CC0-1.0' and len(cases) == 20 and len({c['id'] for c in cases}) == 20,
            'Exact20 authored requests required')
    require({lang: sum(c['language'] == lang for c in cases) for lang in ['en', 'de']} == {'en': 10, 'de': 10},
            'Balanced language coverage differs')
    require(len({c['scenario_id'] for c in cases}) == 10, 'Ten translated scenario pairs required')
    require(all(sum(c['scenario_id'] == key for c in cases) == 2 for key in {c['scenario_id'] for c in cases}),
            'Every scenario must retain both languages')
    for c in cases:
        validate_request(c['request'])
        questions = c['request']['questions']
        require(len(questions) == 3 and set(questions) == set(c['gold']) and
                {q['type'] for q in questions.values()} == {'noul', 'choice', 'score'},
                'Three typed fields and exact gold coverage required')
        for key, q in questions.items():
            gold = c['gold'][key]
            require(q['type'] == gold['type'] and isinstance(gold['rationale'], str) and gold['rationale'], 'Typed gold rationale required')
            options = ['true', 'false'] if q['type'] == 'noul' else list(q['criteria']) if q['type'] == 'choice' else [str(i) for i in range(len(q['criteria']))]
            require(gold['label'] in options, 'Gold label outside actual option IDs')
            if q['type'] != 'noul':
                require(len(q['criteria']) == 3, 'Exact three-level/choice scope required')
    review = json.loads(REVIEW.read_text())
    require(review['panel_sha256'] == PANEL_SHA and review['reviewed_requests'] == 20 and review['reviewed_fields'] == 60 and
            review['all_labels_logically_follow_states'] is True and review['model_outputs_seen'] is False and
            review['human_adjudication'] is False, 'Independent pre-output source-label review required')
    return data


def expected_stages(case):
    template = declaration()['original_stage_template']
    expected = Counter(template['counts'])
    for field in case['request']['questions']:
        for form in template['per_field_additions']:
            expected[form.format(field=field)] += 1
    require(sum(expected.values()) == 74, 'Full original stage plan differs')
    return expected


def source_guard():
    d = declaration()
    mandatory={'tools/josev2b_task_quality_v1.py','tools/dump_josev2b_task_quality_v1.py',
        'tools/kaggle/josev2b_task_quality_reference_v1.py','tools/ci-heavy/josev2b_task_quality_v1.py',
        'tools/josev2b_task_quality_panel_v1.py','tools/josev2b_contract.py','tools/josev2b_environment.py',
        'tools/ci-heavy/josev2b_gpu_install.py','tools/ci-heavy/josev2b_wheel_metadata.py',
        'tools/josev2b_composed_panel.py','tools/josev2b_composed_diff.py','tools/josev2b_head_schema.py',
        'tests/josev-task-quality/CMakeLists.txt','tests/josev-task-quality/probe.cpp','tests/josev-task-quality/diff.cpp',
        'docs/josev2b-environment-lock.json','tools/kaggle/josev2b-original/requirements.lock'}
    require(mandatory<=set(d['source_sha256']) and all(len(value)==64 for value in d['source_sha256'].values()),
            'Complete executed source/installer/scorer/build closure required')
    names = [str(CONTRACT.relative_to(ROOT)), *d['source_sha256']]
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', *names], text=True).strip()
    require(not dirty, 'Dirty quality source/protocol')
    for name, expected in d['source_sha256'].items():
        require(sha256(ROOT / name) == expected, 'Quality dependency SHA differs: ' + name)
    panel()
    return {name: sha256(ROOT / name) for name in names}


def native_prerequisites():
    d = declaration()['native_prerequisites']
    require(d['reviewed_original20'] is True and d['reviewed_current_q4_four_case'] is True and
            d['runtime_source'] and len(d['runtime_source']) == 40 and
            d['original_source'] and len(d['original_source']) == 40 and d['runtime_protected_git_objects'] and
            d['f32_profile'] and d['q4_profile'] and
            all(d.get(k) and len(d[k]) == 64 for k in ['original_reference_sha256', 'original_freeze_sha256',
                'original_raw_manifest_sha256', 'original_protocol_sha256', 'candidate_report_sha256', 'f32_model_sha256', 'q4_model_sha256']),
            'Explicit reviewed original20/candidate/runtime/model pins pending; no native model access')
    return d


def number(value, low=0., high=1.):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
            'Finite typed public numeric value outside range')
    return float(value)


def decoded_fields(case, response):
    require(set(response) == {'model', 'answers', 'usage'} and response['model'] == case['request']['model'],
            'Complete official public response required')
    require(response['usage']['output_tokens'] == 0 and type(response['usage']['input_tokens']) is int and
            0 < response['usage']['input_tokens'] <= 512, 'Actual public usage contract differs')
    questions = case['request']['questions']
    require(set(response['answers']) == set(questions), 'Missing/extra typed public answer')
    fields = {}
    for key, q in questions.items():
        answer = response['answers'][key]
        require(answer['type'] == q['type'], 'Public answer type differs')
        kind = q['type']
        if kind == 'noul':
            require(set(answer) == {'type', 'noul'}, 'NOUL public fields differ')
            p = number(answer['noul'])
            probabilities = {'true': p, 'false': float(Decimal('1') - Decimal(str(p)))}
            label = 'true' if p >= .5 else 'false'  # Declared benchmark tie: official true-first order.
        else:
            ids = list(q['criteria']) if kind == 'choice' else [str(i) for i in range(len(q['criteria']))]
            require(set(answer['probabilities']) == set(ids), 'Public option coverage differs')
            probabilities = {option: number(answer['probabilities'][option]) for option in ids}
            require(abs(sum(probabilities.values()) - 1.) <= len(ids) * .00005 + 1e-12,
                    'Rounded official probabilities do not sum within four-place rounding bound')
            confidence = number(answer['confidence'])
            require(abs(confidence - max(probabilities.values())) <= .0001, 'Public confidence disagrees with probabilities')
            if kind == 'choice':
                require(set(answer) == {'type', 'choice', 'confidence', 'probabilities'} and answer['choice'] in ids,
                        'Choice public fields/label differ')
                label = answer['choice']  # Preserve original raw-argmax/tie semantics; do not recompute a rounded tie.
                require(max(probabilities.values()) - probabilities[label] <= .0001, 'Choice not consistent with public distribution')
            else:
                require(set(answer) == {'type', 'score', 'confidence', 'legend', 'probabilities'} and
                        answer['legend'] == dict(zip(ids, q['criteria'])), 'Ordered score legend differs')
                label = max(ids, key=lambda option: probabilities[option])  # Lowest criterion index wins rounded ties.
        row = dict(type=kind, label=label, probabilities=probabilities, gold=case['gold'][key]['label'])
        ordered = sorted(probabilities.values(), reverse=True)
        row['top_two_margin'] = ordered[0] - ordered[1]
        row['correct'] = label == row['gold']
        if kind == 'score':
            row['public_score'] = number(answer['score'], high=len(q['criteria']) - 1)
            row['ordinal_expected_index'] = float(sum(Decimal(i) * Decimal(str(p)) for i, p in probabilities.items()))
            rounding = .00005 * (1 + sum(range(len(q['criteria'])))) + 1e-12
            require(abs(row['public_score'] - row['ordinal_expected_index']) <= rounding,
                    'Public score must use true0..N-1 criterion-index expectation, never normalized scale')
            row['gold_index_absolute_error'] = abs(row['ordinal_expected_index'] - int(row['gold']))
        fields[key] = row
    return fields


def accuracy(rows, predicate=lambda row: True):
    selected = [r for r in rows if predicate(r)]
    require(selected, 'Empty declared scoring group')
    return dict(correct=sum(r['correct'] for r in selected), total=len(selected),
                accuracy=sum(r['correct'] for r in selected) / len(selected))


def evaluate(responses, data=None):
    data = panel() if data is None else data
    require(set(responses) == {c['id'] for c in data['cases']}, 'Exact20 response IDs required; no query dropping')
    rows, requests = [], []
    for c in data['cases']:
        fields = decoded_fields(c, responses[c['id']])
        requests.append(dict(id=c['id'], all_fields_correct=all(r['correct'] for r in fields.values())))
        rows += [dict(case=c['id'], scenario=c['scenario_id'], language=c['language'], stratum=c['stratum'], field=key, **row)
                 for key, row in fields.items()]
    overall = accuracy(rows)
    languages = {lang: accuracy(rows, lambda r: r['language'] == lang) for lang in ['en', 'de']}
    types = {kind: accuracy(rows, lambda r: r['type'] == kind) for kind in ['noul', 'choice', 'score']}
    floors = overall['accuracy'] >= .90 and all(r['accuracy'] >= .85 for r in [*languages.values(), *types.values()])
    confusion = {}
    for row in rows:
        key = row['type'] + '/' + row['gold'] + '->' + row['label']
        confusion[key] = confusion.get(key, 0) + 1
    return dict(overall=overall, per_language=languages, per_type=types, floors_passed=floors,
                request_accuracy=sum(r['all_fields_correct'] for r in requests) / len(requests),
                misses=[r for r in rows if not r['correct']], rows=rows, confusion=confusion,
                statistical_scope='20 correlated requests,10 translated scenario pairs;60 field labels are not60 independent samples')


def compare_public(original, candidate, *, precision, data=None):
    require(precision in ['f32', 'q4'], 'Explicit native precision required')
    data = panel() if data is None else data
    a, b = evaluate(original, data), evaluate(candidate, data)
    require([(r['case'], r['field']) for r in a['rows']] == [(r['case'], r['field']) for r in b['rows']], 'Ordered field coverage differs')
    rows = []
    for old, new in zip(a['rows'], b['rows']):
        require(set(old['probabilities']) == set(new['probabilities']), 'Option IDs differ')
        errors = [float(abs(Decimal(str(old['probabilities'][k])) - Decimal(str(new['probabilities'][k])))) for k in old['probabilities']]
        row = dict(case=old['case'], field=old['field'], type=old['type'], original_label=old['label'],
                   native_label=new['label'], discrete_agreement=old['label'] == new['label'],
                   maximum_public_probability_absolute=max(errors), total_variation=.5 * sum(errors),
                   original_margin=old['top_two_margin'], native_margin=new['top_two_margin'])
        if old['type'] == 'score':
            row['ordinal_index_absolute'] = float(abs(Decimal(str(old['ordinal_expected_index'])) - Decimal(str(new['ordinal_expected_index']))))
            row['public_score_absolute'] = float(abs(Decimal(str(old['public_score'])) - Decimal(str(new['public_score']))))
        probability_budget = .0001 if precision == 'f32' else .01
        ordinal_budget = .001 if precision == 'f32' else .01
        row['task_probability_passed'] = max(errors) <= probability_budget
        row['task_ordinal_passed'] = row.get('ordinal_index_absolute', 0.) <= ordinal_budget and row.get('public_score_absolute', 0.) <= ordinal_budget
        row['strict_f32_public_probability_flag'] = max(errors) <= .0001
        row['strict_f32_ordinal_flag'] = row.get('ordinal_index_absolute', 0.) <= .001 and row.get('public_score_absolute', 0.) <= .001
        rows.append(row)
    groups_a = {'overall': a['overall'], **{'language/' + k: v for k, v in a['per_language'].items()}, **{'type/' + k: v for k, v in a['per_type'].items()}}
    groups_b = {'overall': b['overall'], **{'language/' + k: v for k, v in b['per_language'].items()}, **{'type/' + k: v for k, v in b['per_type'].items()}}
    no_drop = all(groups_b[key]['correct'] >= row['correct'] for key, row in groups_a.items())
    passed = a['floors_passed'] and b['floors_passed'] and no_drop and all(r['discrete_agreement'] and r['task_probability_passed'] and r['task_ordinal_passed'] for r in rows)
    return dict(precision=precision, independent_task_quality_passed=passed, no_accuracy_drop=no_drop,
                original=a, candidate=b, fields=rows, publication_authorized=False, full_model_qualified=False,
                strict_f32_flags_all_passed=all(r['strict_f32_public_probability_flag'] and r['strict_f32_ordinal_flag'] for r in rows))
