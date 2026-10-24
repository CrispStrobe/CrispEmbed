"""Source/candidate gates only; no model, ML package, binary, or reference access."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('ovis_quant_raw_test',
    ROOT / 'tools/ci-heavy/ovisocr2_quant_raw_replay.py')
SUBJECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBJECT)
SHA = 'a' * 64
HEAD = 'b' * 40
RUN = 123


def refresh(receipt):
    receipt['independent_review']['producer_canonical_sha256'] = hashlib.sha256(
        json.dumps(receipt['producer'], sort_keys=True, separators=(',', ':'),
                   ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def fixture():
    names = [f'case-{i}' for i in range(11)]
    producer = {'source_commit': HEAD, 'requested_phase': 'q8',
        'models': {'q8': {'sha256': SHA, 'bytes': 1_200_000_000, 'recipe': 'q8-protected'}},
        'inputs': [{'id': name} for name in names],
        'native_cases': {'q8': {name: {'token_ids': [248046], 'confidences': [1.0], 'cap': 256}
                               for name in names}},
        'scores': {'q8': {}}, 'quant_numerical': {'passed': False, 'log_sha256': SHA}}
    receipt = {'run': RUN, 'source_commit': HEAD, 'producer': producer,
        'artifact_id': 100, 'artifact_name': f'ovisocr2-quality-reports-{RUN}',
        'artifact_zip_sha256': SHA, 'report_sha256': SHA,
        'candidate_artifact': {'id': 101, 'name': f'ovisocr2-quality-candidate-{RUN}',
                               'zip_sha256': SHA},
        'independent_review': {'quant_diagnostic_complete': True, 'artifact_zip_sha256': SHA,
                               'candidate_artifact_zip_sha256': SHA}}
    refresh(receipt)
    return receipt


class QuantRawReplayControls(unittest.TestCase):
    def test_reviewed_failed_numeric_result_can_be_replayed_without_qualifying(self):
        value = fixture()
        pins = SUBJECT.checked_receipt(value, 'q8', RUN)
        self.assertEqual(pins['model_sha256'], SHA)
        self.assertFalse(value['producer']['quant_numerical']['passed'])
        self.assertNotIn('quality_positive', value['independent_review'])

    def test_missing_review_wrong_phase_source_run_or_artifact_rejected(self):
        value = fixture()
        for section, field, replacement in (
            ('independent_review', 'quant_diagnostic_complete', False),
            ('independent_review', 'candidate_artifact_zip_sha256', 'c' * 64),
            ('candidate_artifact', 'name', 'another-run'),
            ('candidate_artifact', 'id', 0), ('producer', 'source_commit', 'main')):
            bad = copy.deepcopy(value)
            bad[section][field] = replacement
            refresh(bad)
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                SUBJECT.checked_receipt(bad, 'q8', RUN)
        for phase, run in (('f16', RUN), ('q8', 0), ('q8', RUN + 1)):
            with self.subTest(phase=phase, run=run), self.assertRaises(ValueError):
                SUBJECT.checked_receipt(value, phase, run)

    def test_phase_recipe_mismatch_rejected_before_weights(self):
        value = fixture()
        value['producer']['models']['q8']['recipe'] = 'q4-attention-q8'
        refresh(value)
        with self.assertRaises(ValueError):
            SUBJECT.checked_receipt(value, 'q8', RUN)

    def test_partial_panel_stale_canonical_nan_and_bad_id_are_rejected(self):
        for change in ('partial', 'canonical', 'nan', 'ids', 'score-count', 'caps'):
            bad = fixture()
            row = bad['producer']['native_cases']['q8']['case-0']
            if change == 'partial':
                del bad['producer']['native_cases']['q8']['case-10']
            elif change == 'canonical':
                bad['producer']['models']['q8']['bytes'] += 1
            elif change == 'nan':
                row['confidences'] = [float('nan')]
            elif change == 'ids':
                row['token_ids'] = [-1]
            elif change == 'score-count':
                row['confidences'] = []
            else:
                row['cap'] = 0
            if change not in ('canonical', 'nan'):
                refresh(bad)
            with self.subTest(change=change), self.assertRaises(ValueError):
                SUBJECT.checked_receipt(bad, 'q8', RUN)

    def test_service_must_bind_completed_exact_run_and_both_artifacts(self):
        pins = SUBJECT.checked_receipt(fixture(), 'q8', RUN)
        run = {'id': RUN, 'head_sha': HEAD, 'status': 'completed', 'conclusion': 'failure',
               'path': '.github/workflows/ovisocr2-metadata.yml'}
        def api(path):
            if path.startswith('actions/runs/'):
                return run
            ident = int(path.rsplit('/', 1)[1])
            return {'id': ident, 'name': f'ovisocr2-quality-{"reports" if ident == 100 else "candidate"}-{RUN}',
                    'expired': False, 'workflow_run': {'id': RUN}, 'digest': 'sha256:' + SHA}
        self.assertEqual(SUBJECT.service_proof(pins, api)['run']['conclusion'], 'failure')
        for key, value in (('status', 'in_progress'), ('conclusion', 'cancelled'),
                           ('head_sha', 'd' * 40), ('path', '.github/workflows/unrelated.yml')):
            bad = dict(run, **{key: value})
            def altered(path):
                return bad if path.startswith('actions/runs/') else api(path)
            with self.subTest(key=key), self.assertRaises(ValueError):
                SUBJECT.service_proof(pins, altered)
        def changed_artifact(path):
            result = dict(api(path))
            if path == 'actions/artifacts/101':
                result['digest'] = 'sha256:' + 'e' * 64
            return result
        with self.assertRaises(ValueError):
            SUBJECT.service_proof(pins, changed_artifact)

    def test_capacity_guard_rejects_low_memory_and_disk(self):
        with mock.patch.object(SUBJECT.shutil, 'disk_usage',
                               return_value=SimpleNamespace(free=20_000_000_000)):
            with mock.patch.object(Path, 'read_text', return_value='MemAvailable: 8388608 kB\nSwapFree: 0 kB\n'):
                self.assertGreater(SUBJECT.capacity(Path('/tmp/task'), 1_200_000_000)['MemAvailable_kib'], 0)
            with mock.patch.object(Path, 'read_text', return_value='MemAvailable: 1048576 kB\nSwapFree: 0 kB\n'):
                with self.assertRaises(ValueError):
                    SUBJECT.capacity(Path('/tmp/task'), 1_200_000_000)
        with mock.patch.object(SUBJECT.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)), \
             mock.patch.object(Path, 'read_text', return_value='MemAvailable: 8388608 kB\nSwapFree: 0 kB\n'):
            with self.assertRaises(ValueError):
                SUBJECT.capacity(Path('/tmp/task'), 1_200_000_000)


if __name__ == '__main__':
    unittest.main()
