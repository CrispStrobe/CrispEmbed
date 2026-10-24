"""Pure publication consumer gates; no build, credentials, weights, or ML imports."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('ovis_published_consumer',
    ROOT / 'tools/ci-heavy/ovisocr2_published_consumer.py')
CONSUMER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONSUMER)
SHA = 'a' * 64
COMMIT = 'b' * 40
PRESET = {'name': 'both', 'enabled_gates': ['CRISPEMBED_QWEN35_VISION_PRECISE_NORM',
                                             'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN'],
          'values': {'CRISPEMBED_QWEN35_VISION_PRECISE_NORM': '1',
                     'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN': '1',
                     'CRISPEMBED_QWEN35_VISION_CASCADE_NORM': '0'}}


def sample():
    pins = {'precision': 'f16', 'destination': 'cstr/OvisOCR2-GGUF',
            'candidate_sha256': SHA, 'candidate_bytes': 1717222848,
            'recipe': 'original-f16', 'selected_vision_preset': PRESET,
            'receipt_sha256': {'model': SHA, 'quality': SHA}}
    filename = 'ovisocr2-f16.gguf'
    url = f'https://huggingface.co/cstr/OvisOCR2-GGUF/resolve/{COMMIT}/{filename}'
    receipt = {'producer': {'status': 'published and immutable remote file reverified',
                            'source_commit': COMMIT,
                            'qualification': pins, 'destination': pins['destination'],
                            'published_commit': COMMIT,
                            'remote': {'repo': pins['destination'], 'commit': COMMIT,
                                       'filename': filename, 'sha256': SHA,
                                       'bytes': pins['candidate_bytes'],
                                       'small_files_sha256': {'manifest.json': SHA}}},
               'independent_review': {'publication_positive': True}}
    registry = '{ "ovisocr2-f16", "ovisocr2-f16.gguf", "' + url + '", "description", "1.7 GB", "apache-2.0" },'
    hashes = '{ "' + url + '", "' + SHA + '" },'
    return pins, receipt, registry, hashes


class PublishedConsumerControls(unittest.TestCase):
    def test_real_frozen_native_receipt_and_public_preservation_agree(self):
        receipt = json.loads((ROOT / 'tests/results/qwen35-ocr/ovisocr2-model-audit.json').read_text())
        preservation = json.loads((ROOT / 'tests/results/qwen35-ocr/ovisocr2-vllm-reference-preservation.json').read_text())
        pins = CONSUMER.require_frozen_image(preservation, receipt)
        self.assertEqual(len(pins['native_ids']), 9)
        self.assertEqual(pins['native_ids'][-1], 248046)
        self.assertEqual(pins['expected_text'], 'Hello OCR\n\n12345')
        for changed in ('sha256', 'bytes'):
            bad = copy.deepcopy(preservation)
            bad['source']['files']['owned-readable.png'][changed] = 0
            with self.assertRaises(ValueError):
                CONSUMER.require_frozen_image(bad, receipt)

    def test_publication_exact_source_alias_and_generated_pin(self):
        pins, receipt, registry, hashes = sample()
        valid = CONSUMER.require_publication(receipt, pins, registry, hashes)
        self.assertEqual(valid['registry']['alias'], 'ovisocr2-f16')
        wrapped_registry = registry.replace(f'/{COMMIT}/ovisocr2-f16.gguf"',
                                            f'/{COMMIT}/"\n        "ovisocr2-f16.gguf"')
        self.assertEqual(CONSUMER.require_publication(receipt, pins, wrapped_registry,
                                                      hashes)['registry']['url'], valid['registry']['url'])
        for section, field, value in (
                ('independent_review', 'publication_positive', False),
                ('producer', 'published_commit', 'main'),
                ('producer', 'qualification', {}),
                ('producer', 'status', 'draft')):
            bad = copy.deepcopy(receipt)
            bad[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                CONSUMER.require_publication(bad, pins, registry, hashes)
        for field, value in [('sha256', 'c'*64), ('bytes', 1), ('filename', 'wrong.gguf')]:
            bad = copy.deepcopy(receipt)
            bad['producer']['remote'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                CONSUMER.require_publication(bad, pins, registry, hashes)
        for altered_registry, altered_hashes in (
                ('', hashes), (registry.replace(COMMIT, 'main'), hashes),
                (registry + '\n' + registry, hashes), (registry, ''),
                (registry, hashes.replace(SHA, 'c'*64)),
                ('// ' + registry, hashes),
                (registry, '// ' + hashes)):
            with self.assertRaises(ValueError):
                CONSUMER.require_publication(receipt, pins, altered_registry, altered_hashes)
        wrong_selection = copy.deepcopy(pins)
        wrong_selection['selected_vision_preset']['name'] = 'baseline'
        with self.assertRaises(ValueError):
            CONSUMER.require_publication(receipt, wrong_selection, registry, hashes)

    def test_preflight_missing_publication_fails_before_registry_or_remote(self):
        pins, _, _, _ = sample()
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(CONSUMER, 'PUBLICATION', Path(temp) / 'absent.json'), \
                mock.patch.object(CONSUMER, 'module') as importer, \
                mock.patch.object(CONSUMER, 'bounded_get', side_effect=AssertionError('remote accessed')):
            importer.return_value.require_receipts.return_value = pins
            with self.assertRaisesRegex(ValueError, 'publication receipt absent'):
                CONSUMER.preflight('f16')
            importer.return_value.require_receipts.assert_called_once_with('f16')

    def test_small_manifest_and_png_provenance(self):
        pins, receipt, registry, hashes = sample()
        publication = CONSUMER.require_publication(receipt, pins, registry, hashes)
        image = b'fake-small-png'
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, 'w') as archive:
            archive.writestr('owned-readable.png', image)
        payload = mem.getvalue()
        fixture = {'url': 'https://example.invalid/reference.zip',
                   'archive_sha256': hashlib.sha256(payload).hexdigest(),
                   'archive_bytes': len(payload), 'png_sha256': hashlib.sha256(image).hexdigest(),
                   'png_bytes': len(image)}
        manifest = {'candidate': {'filename': 'ovisocr2-f16.gguf', 'sha256': SHA,
                                  'bytes': pins['candidate_bytes'], 'precision': 'f16',
                                  'recipe': pins['recipe']},
                    'selected_vision_preset': PRESET, 'receipts': pins['receipt_sha256']}
        data = json.dumps(manifest).encode()
        publication['remote']['small_files_sha256']['manifest.json'] = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(CONSUMER, 'bounded_get', side_effect=[data, payload]):
                path, proof = CONSUMER.fetch_small_provenance(publication, fixture, Path(temp))
            self.assertEqual(path.read_bytes(), image)
            self.assertEqual(proof['png_sha256'], fixture['png_sha256'])
            self.assertFalse((Path(temp) / 'frozen-fixture.zip').exists())
            wrong = copy.deepcopy(manifest)
            wrong['candidate']['bytes'] += 1
            changed = json.dumps(wrong).encode()
            publication['remote']['small_files_sha256']['manifest.json'] = hashlib.sha256(changed).hexdigest()
            with mock.patch.object(CONSUMER, 'bounded_get', side_effect=[changed, payload]), self.assertRaises(ValueError):
                CONSUMER.fetch_small_provenance(publication, fixture, Path(temp))

    def test_cli_text_confidence_and_cache_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log = root / 'cli.log'
            log.write_text('{"full_text":"Hello OCR\\n\\n12345","n_regions":1,"mean_confidence":0.995}\n')
            self.assertEqual(CONSUMER.parse_cli(log, 'Hello OCR\n\n12345')['n_regions'], 1)
            for value in ('null', 'NaN', '1.1'):
                log.write_text('{"full_text":"Hello OCR\\n\\n12345","n_regions":1,"mean_confidence":'+value+'}\n')
                with self.assertRaises(ValueError):
                    CONSUMER.parse_cli(log, 'Hello OCR\n\n12345')
            file = root / 'model.gguf'
            file.write_bytes(b'test')
            remote = {'bytes': 4, 'sha256': hashlib.sha256(b'test').hexdigest()}
            self.assertEqual(CONSUMER.check_cached(file, remote)['bytes'], 4)
            file.write_bytes(b'bad!')
            with self.assertRaises(ValueError):
                CONSUMER.check_cached(file, remote)

    def test_runtime_gates_and_precision_are_explicit(self):
        with mock.patch.dict('os.environ', {'CRISPEMBED_QWEN35_IMATRIX': '1',
                                            'CRISPEMBED_OFFLINE': '1'}):
            f16 = CONSUMER.env_for(Path('/tmp/consumer-cache'), PRESET, 'f16')
            quant = CONSUMER.env_for(Path('/tmp/consumer-cache-quant'), PRESET, 'q8')
        self.assertTrue(all(f16[name] == '1' for name in CONSUMER.GATES))
        self.assertTrue(all(f16[name] == '0' for name in CONSUMER.BASELINE_ZERO))
        self.assertEqual(f16['CRISPEMBED_QWEN35_QUANT'], '0')
        self.assertEqual(quant['CRISPEMBED_QWEN35_QUANT'], '1')
        self.assertEqual(f16['CRISPEMBED_OFFLINE'], '0')
        self.assertEqual(f16['CRISPEMBED_QWEN35_VISION_PRECISE_NORM'], '1')
        self.assertEqual(f16['CRISPEMBED_QWEN35_VISION_PRECISE_ATTN'], '1')

    def test_fresh_offline_http_confidence_comparison(self):
        cli = {'text': 'Hello OCR\n\n12345', 'n_regions': 1, 'mean_confidence': .995}
        http = {'text': cli['text'], 'n_regions': 1, 'mean_confidence': .9954}
        reference = {'mean_confidence': .9953883820109897}
        self.assertTrue(CONSUMER.compare_public_answers(cli, cli, http,
                                                        reference)['fresh_offline_confidence_exact'])
        for altered in ({**http, 'mean_confidence': .997},
                        {**http, 'text': 'wrong'}):
            with self.assertRaises(ValueError):
                CONSUMER.compare_public_answers(cli, cli, altered, reference)
        with self.assertRaises(ValueError):
            CONSUMER.compare_public_answers(cli, {**cli, 'mean_confidence': .994}, http, reference)
        # The previous arbitrary .005 allowance accepted this; the reviewed
        # CLI printing budget must reject it.
        with self.assertRaises(ValueError):
            CONSUMER.compare_public_answers({**cli, 'mean_confidence': .998},
                                            {**cli, 'mean_confidence': .998}, http, reference)
        with self.assertRaises(ValueError):
            CONSUMER.compare_public_answers(cli, cli, {**http, 'mean_confidence': float('nan')},
                                            reference)

    def test_quality_reference_requires_same_png_rgb_and_candidate(self):
        model = json.loads((ROOT / 'tests/results/qwen35-ocr/ovisocr2-model-audit.json').read_text())
        preserved = json.loads((ROOT / 'tests/results/qwen35-ocr/ovisocr2-vllm-reference-preservation.json').read_text())
        fixture = CONSUMER.require_frozen_image(preserved, model)
        original = json.loads((ROOT / 'tests/results/qwen35-ocr/ovisocr2-original-quality.json').read_text())
        case = original['producer']['original_cases']['dev-hello']
        self.assertEqual(case['input_png_sha256'], fixture['png_sha256'])
        self.assertEqual(case['input_rgb_sha256'], fixture['rgb_sha256'])
        pins, _, _, _ = sample()
        quality = {'independent_review': {'quality_positive': True},
                   'producer': {'models': {'f16': {'sha256': SHA}},
                                'native_cases': {'f16': {'dev-hello': case}},
                                'cabi_smoke': {'f16': {'case': 'dev-hello',
                                                       'raw_text': fixture['expected_text'],
                                                       'mean_confidence': case['mean_confidence']}}}}
        self.assertEqual(CONSUMER.require_quality_confidence(quality, pins, fixture)['token_ids'],
                         CONSUMER.FROZEN_IDS)
        for section, field, value in (('native_cases', 'input_png_sha256', '0'*64),
                                      ('native_cases', 'mean_confidence', .4),
                                      ('cabi_smoke', 'mean_confidence', .4)):
            bad = copy.deepcopy(quality)
            if section == 'native_cases':
                bad['producer']['native_cases']['f16']['dev-hello'][field] = value
            else:
                bad['producer']['cabi_smoke']['f16'][field] = value
            with self.assertRaises(ValueError):
                CONSUMER.require_quality_confidence(bad, pins, fixture)


if __name__ == '__main__':
    unittest.main()
